import torch
import torch.nn as nn
import lightning as pl
import wandb
import os
import copy
from diffsynth import WanVideoReCamMasterPipeline, ModelManager
import json
import numpy as np
from PIL import Image
import imageio
import random
from torchvision.transforms import v2
from einops import rearrange
from pose_classifier import PoseClassifier

def compute_relative_pose(pose_a, pose_b, use_torch=False):
    """
    计算相机B相对于相机A的相对位姿矩阵
    """
    assert pose_a.shape == (4, 4), f"相机A外参矩阵形状应为(4,4)，实际为{pose_a.shape}"
    assert pose_b.shape == (4, 4), f"相机B外参矩阵形状应为(4,4)，实际为{pose_b.shape}"
    
    if use_torch:
        if not isinstance(pose_a, torch.Tensor):
            pose_a = torch.from_numpy(pose_a).float()
        if not isinstance(pose_b, torch.Tensor):
            pose_b = torch.from_numpy(pose_b).float()
        
        pose_a_inv = torch.inverse(pose_a)
        relative_pose = torch.matmul(pose_b, pose_a_inv)
    else:
        if not isinstance(pose_a, np.ndarray):
            pose_a = np.array(pose_a, dtype=np.float32)
        if not isinstance(pose_b, np.ndarray):
            pose_b = np.array(pose_b, dtype=np.float32)
        
        pose_a_inv = np.linalg.inv(pose_a)
        relative_pose = np.matmul(pose_b, pose_a_inv)
    
    return relative_pose

class OpenXFramePackDataset(torch.utils.data.Dataset):
    """OpenX数据集的FramePack训练数据集"""
    
    def __init__(self, base_path, steps_per_epoch, 
                 min_condition_frames=10, max_condition_frames=40,
                 target_frames=10, height=480, width=832):
        
        self.base_path = base_path
        self.min_condition_frames = min_condition_frames
        self.max_condition_frames = max_condition_frames
        self.target_frames = target_frames
        self.height = height
        self.width = width
        self.steps_per_epoch = steps_per_epoch
        self.pose_classifier = PoseClassifier()
        
        # VAE时间压缩比例
        self.time_compression_ratio = 4  # VAE将时间维度压缩4倍
        
        # 查找所有处理好的episode
        self.episode_dirs = []
        print(f"🔧 扫描OpenX数据集: {base_path}")
        
        if os.path.exists(base_path):
            for item in os.listdir(base_path):
                episode_dir = os.path.join(base_path, item)
                if os.path.isdir(episode_dir):
                    encoded_path = os.path.join(episode_dir, "encoded_video.pth")
                    if os.path.exists(encoded_path):
                        self.episode_dirs.append(episode_dir)
            
            print(f"  ✅ 找到 {len(self.episode_dirs)} 个episodes")
        else:
            print(f"  ⚠️ 路径不存在: {base_path}")
        
        assert len(self.episode_dirs) > 0, "No encoded episodes found!"

    def select_dynamic_segment_framepack(self, full_latents):
        """🔧 FramePack风格的动态选择条件帧和目标帧 - 适配OpenX数据"""
        total_lens = full_latents.shape[1]
        
        min_condition_compressed = self.min_condition_frames // self.time_compression_ratio
        max_condition_compressed = self.max_condition_frames // self.time_compression_ratio
        target_frames_compressed = self.target_frames // self.time_compression_ratio
        max_condition_compressed = min(total_lens-target_frames_compressed-1, max_condition_compressed)

        ratio = random.random()
        if ratio < 0.15:
            condition_frames_compressed = 1
        elif 0.15 <= ratio < 0.9 or total_lens <= 2*target_frames_compressed + 1:
            condition_frames_compressed = random.randint(min_condition_compressed, max_condition_compressed)
        else:
            condition_frames_compressed = target_frames_compressed
        
        # 确保有足够的帧数
        min_required_frames = condition_frames_compressed + target_frames_compressed
        if total_lens < min_required_frames:
            return None
        
        start_frame_compressed = random.randint(0, total_lens - min_required_frames - 1)
        condition_end_compressed = start_frame_compressed + condition_frames_compressed
        target_end_compressed = condition_end_compressed + target_frames_compressed

        # 🔧 FramePack风格的索引处理
        latent_indices = torch.arange(condition_end_compressed, target_end_compressed)  # 只预测未来帧
        
        # 🔧 根据实际的condition_frames_compressed生成索引
        # 1x帧：起始帧 + 最后1帧
        clean_latent_indices_start = torch.tensor([start_frame_compressed])
        clean_latent_1x_indices = torch.tensor([condition_end_compressed - 1])
        clean_latent_indices = torch.cat([clean_latent_indices_start, clean_latent_1x_indices])
        
        # 🔧 2x帧：根据实际condition长度确定
        if condition_frames_compressed >= 2:
            # 取最后2帧（如果有的话）
            clean_latent_2x_start = max(start_frame_compressed, condition_end_compressed - 2-1)
            clean_latent_2x_indices = torch.arange(clean_latent_2x_start, condition_end_compressed-1)
        else:
            # 如果condition帧数不足2帧，创建空索引
            clean_latent_2x_indices = torch.tensor([], dtype=torch.long)
        
        # 🔧 4x帧：根据实际condition长度确定，最多16帧
        if condition_frames_compressed > 3:
            # 取最多16帧的历史（如果有的话）
            clean_4x_start = max(start_frame_compressed, condition_end_compressed - 16-3)
            clean_latent_4x_indices = torch.arange(clean_4x_start, condition_end_compressed-3)
        else:
            clean_latent_4x_indices = torch.tensor([], dtype=torch.long)
        
        # 对应的原始关键帧索引
        keyframe_original_idx = []
        for compressed_idx in range(start_frame_compressed, target_end_compressed):
            keyframe_original_idx.append(compressed_idx * 4)
        
        return {
            'start_frame': start_frame_compressed,
            'condition_frames': condition_frames_compressed,
            'target_frames': target_frames_compressed,
            'condition_range': (start_frame_compressed, condition_end_compressed),
            'target_range': (condition_end_compressed, target_end_compressed),
            
            # FramePack风格的索引
            'latent_indices': latent_indices,
            'clean_latent_indices': clean_latent_indices,
            'clean_latent_2x_indices': clean_latent_2x_indices,
            'clean_latent_4x_indices': clean_latent_4x_indices,
            
            'keyframe_original_idx': keyframe_original_idx,
            'original_condition_frames': condition_frames_compressed * self.time_compression_ratio,
            'original_target_frames': target_frames_compressed * self.time_compression_ratio,
        }

    def create_pose_embeddings(self, cam_data, segment_info):
        """🔧 创建pose embeddings - 为所有帧（condition + target）提取camera信息"""
        cam_data_seq = cam_data['extrinsic']   # N * 4 * 4
        
        # 🔧 为所有帧（condition + target）计算camera embedding
        start_frame = segment_info['start_frame'] * self.time_compression_ratio
        end_frame = segment_info['target_range'][1] * self.time_compression_ratio
        
        # 为所有帧计算相对pose
        all_keyframe_indices = []
        for compressed_idx in range(segment_info['start_frame'], segment_info['target_range'][1]):
            keyframe_idx = compressed_idx * 4
            if keyframe_idx + 4 < len(cam_data_seq):
                all_keyframe_indices.append(keyframe_idx)
        
        relative_cams = []
        for idx in all_keyframe_indices:
            if idx + 4 < len(cam_data_seq):
                cam_prev = cam_data_seq[idx]
                cam_next = cam_data_seq[idx + 4]
                relative_cam = compute_relative_pose(cam_prev, cam_next)
                relative_cams.append(torch.as_tensor(relative_cam[:3, :]))
            else:
                # 如果没有下一帧，使用单位矩阵
                identity_cam = torch.eye(3, 4)
                relative_cams.append(identity_cam)
        
        if len(relative_cams) == 0:
            return None
            
        pose_embedding = torch.stack(relative_cams, dim=0)
        pose_embedding = rearrange(pose_embedding, 'b c d -> b (c d)')
        pose_embedding = pose_embedding.to(torch.bfloat16)

        return pose_embedding

    def prepare_framepack_inputs(self, full_latents, segment_info):
        """🔧 准备FramePack风格的多尺度输入 - 适配OpenX数据"""
        # 🔧 处理4维输入 [C, T, H, W]，添加batch维度
        if len(full_latents.shape) == 4:
            full_latents = full_latents.unsqueeze(0)  # [C, T, H, W] -> [1, C, T, H, W]
            B, C, T, H, W = full_latents.shape
        else:
            B, C, T, H, W = full_latents.shape
        
        # 主要latents（用于去噪预测）
        latent_indices = segment_info['latent_indices']
        main_latents = full_latents[:, :, latent_indices, :, :]
        
        # 🔧 1x条件帧（起始帧 + 最后1帧）
        clean_latent_indices = segment_info['clean_latent_indices']
        clean_latents = full_latents[:, :, clean_latent_indices, :, :]
        
        # 🔧 4x条件帧 - 总是16帧，直接用真实索引 + 0填充
        clean_latent_4x_indices = segment_info['clean_latent_4x_indices']
        
        # 创建固定长度16的latents，初始化为0
        clean_latents_4x = torch.zeros(B, C, 16, H, W, dtype=full_latents.dtype)
        clean_latent_4x_indices_final = torch.full((16,), -1, dtype=torch.long)  # -1表示padding
        
        # 🔧 检查是否有有效的4x索引
        if len(clean_latent_4x_indices) > 0:
            actual_4x_frames = len(clean_latent_4x_indices)
            # 从后往前填充，确保最新的帧在最后
            start_pos = max(0, 16 - actual_4x_frames)
            end_pos = 16
            actual_start = max(0, actual_4x_frames - 16)  # 如果超过16帧，只取最后16帧
            
            clean_latents_4x[:, :, start_pos:end_pos, :, :] = full_latents[:, :, clean_latent_4x_indices[actual_start:], :, :]
            clean_latent_4x_indices_final[start_pos:end_pos] = clean_latent_4x_indices[actual_start:]
        
        # 🔧 2x条件帧 - 总是2帧，直接用真实索引 + 0填充
        clean_latent_2x_indices = segment_info['clean_latent_2x_indices']
        
        # 创建固定长度2的latents，初始化为0
        clean_latents_2x = torch.zeros(B, C, 2, H, W, dtype=full_latents.dtype)
        clean_latent_2x_indices_final = torch.full((2,), -1, dtype=torch.long)  # -1表示padding
        
        # 🔧 检查是否有有效的2x索引
        if len(clean_latent_2x_indices) > 0:
            actual_2x_frames = len(clean_latent_2x_indices)
            # 从后往前填充，确保最新的帧在最后
            start_pos = max(0, 2 - actual_2x_frames)
            end_pos = 2
            actual_start = max(0, actual_2x_frames - 2)  # 如果超过2帧，只取最后2帧
            
            clean_latents_2x[:, :, start_pos:end_pos, :, :] = full_latents[:, :, clean_latent_2x_indices[actual_start:], :, :]
            clean_latent_2x_indices_final[start_pos:end_pos] = clean_latent_2x_indices[actual_start:]
        
        # 🔧 移除添加的batch维度，返回原始格式
        if B == 1:
            main_latents = main_latents.squeeze(0)  # [1, C, T, H, W] -> [C, T, H, W]
            clean_latents = clean_latents.squeeze(0)
            clean_latents_2x = clean_latents_2x.squeeze(0)
            clean_latents_4x = clean_latents_4x.squeeze(0)
        
        return {
            'latents': main_latents,
            'clean_latents': clean_latents,
            'clean_latents_2x': clean_latents_2x,
            'clean_latents_4x': clean_latents_4x,
            'latent_indices': segment_info['latent_indices'],
            'clean_latent_indices': segment_info['clean_latent_indices'],
            'clean_latent_2x_indices': clean_latent_2x_indices_final,
            'clean_latent_4x_indices': clean_latent_4x_indices_final,
        }

    def __getitem__(self, index):
        while True:
            try:
                # 随机选择一个episode
                episode_dir = random.choice(self.episode_dirs)
                episode_name = os.path.basename(episode_dir)
                
                # 加载编码的视频数据
                encoded_data = torch.load(
                    os.path.join(episode_dir, "encoded_video.pth"),
                    weights_only=False,
                    map_location="cpu"
                )
                
                full_latents = encoded_data['latents']  # [C, T, H, W]
                if full_latents.shape[1] <= 10:
                    continue
                cam_data = encoded_data['cam_emb']
                
                # 🔧 使用FramePack风格的段落选择
                segment_info = self.select_dynamic_segment_framepack(full_latents)
                if segment_info is None:
                    continue
                
                # 🔧 为所有帧创建pose embeddings
                all_camera_embeddings = self.create_pose_embeddings(cam_data, segment_info)
                if all_camera_embeddings is None:
                    continue
                
                # 🔧 准备FramePack风格的多尺度输入
                framepack_inputs = self.prepare_framepack_inputs(full_latents, segment_info)
                
                n = segment_info["condition_frames"]
                m = segment_info['target_frames']
                
                # 🔧 处理camera embedding with mask
                mask = torch.zeros(n+m, dtype=torch.float32)
                mask[:n] = 1.0  # condition帧标记为1
                mask = mask.view(-1, 1)
                
                # 添加mask到camera embeddings
                camera_with_mask = torch.cat([all_camera_embeddings, mask], dim=1)
                
                result = {
                    # 🔧 FramePack风格的多尺度输入
                    "latents": framepack_inputs['latents'],  # 主要预测目标
                    "clean_latents": framepack_inputs['clean_latents'],  # 条件帧(2帧)
                    "clean_latents_2x": framepack_inputs['clean_latents_2x'],  # 2x条件帧(2帧，不足用0填充)
                    "clean_latents_4x": framepack_inputs['clean_latents_4x'],  # 4x条件帧(16帧，不足用0填充)
                    "latent_indices": framepack_inputs['latent_indices'],
                    "clean_latent_indices": framepack_inputs['clean_latent_indices'],
                    "clean_latent_2x_indices": framepack_inputs['clean_latent_2x_indices'],
                    "clean_latent_4x_indices": framepack_inputs['clean_latent_4x_indices'],
                    
                    # 🔧 直接传递带mask的camera embeddings
                    "camera": camera_with_mask,  # 所有帧的camera embeddings（带mask）
                    
                    "prompt_emb": encoded_data["prompt_emb"],
                    "image_emb": encoded_data.get("image_emb", {}),
                    
                    "condition_frames": n,
                    "target_frames": m,
                    "episode_name": episode_name,
                    "dataset_name": "openx-fractal",
                    "original_condition_frames": segment_info['original_condition_frames'],
                    "original_target_frames": segment_info['original_target_frames'],
                }
                
                return result
                
            except Exception as e:
                print(f"Error loading sample from {episode_dir}: {e}")
                import traceback
                traceback.print_exc()
                continue

    def __len__(self):
        return self.steps_per_epoch

def replace_dit_model_in_manager():
    """在模型加载前替换DiT模型类"""
    from diffsynth.models.wan_video_dit_recam_future import WanModelFuture
    from diffsynth.configs.model_config import model_loader_configs
    
    # 修改model_loader_configs中的配置
    for i, config in enumerate(model_loader_configs):
        keys_hash, keys_hash_with_shape, model_names, model_classes, model_resource = config
        
        # 检查是否包含wan_video_dit模型
        if 'wan_video_dit' in model_names:
            # 找到wan_video_dit的索引并替换为WanModelFuture
            new_model_names = []
            new_model_classes = []
            
            for name, cls in zip(model_names, model_classes):
                if name == 'wan_video_dit':
                    new_model_names.append(name)  # 保持名称不变
                    new_model_classes.append(WanModelFuture)  # 替换为新的类
                    print(f"✅ 替换了模型类: {name} -> WanModelFuture")
                else:
                    new_model_names.append(name)
                    new_model_classes.append(cls)
            
            # 更新配置
            model_loader_configs[i] = (keys_hash, keys_hash_with_shape, new_model_names, new_model_classes, model_resource)

class OpenXLightningModelForTrain(pl.LightningModule):
    def __init__(
        self,
        dit_path,
        learning_rate=1e-5,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        resume_ckpt_path=None
    ):
        super().__init__()
        replace_dit_model_in_manager()  # 在这里调用
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        if os.path.isfile(dit_path):
            model_manager.load_models([dit_path])
        else:
            dit_path = dit_path.split(",")
            model_manager.load_models([dit_path])
        model_manager.load_models(["models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"])
        
        self.pipe = WanVideoReCamMasterPipeline.from_model_manager(model_manager)
        self.pipe.scheduler.set_timesteps(1000, training=True)

        # 🔧 添加FramePack的clean_x_embedder
        self.add_framepack_components()

        # 添加相机编码器
        dim = self.pipe.dit.blocks[0].self_attn.q.weight.shape[0]
        for block in self.pipe.dit.blocks:
            block.cam_encoder = nn.Linear(13, dim)
            block.projector = nn.Linear(dim, dim)
            block.cam_encoder.weight.data.zero_()
            block.cam_encoder.bias.data.zero_()
            block.projector.weight = nn.Parameter(torch.eye(dim))
            block.projector.bias = nn.Parameter(torch.zeros(dim))
        
        if resume_ckpt_path is not None:
            state_dict = torch.load(resume_ckpt_path, map_location="cpu")
            self.pipe.dit.load_state_dict(state_dict, strict=True)
            print('load checkpoint:', resume_ckpt_path)

        self.freeze_parameters()
        
        # 只训练相机相关和注意力模块以及FramePack相关组件
        for name, module in self.pipe.denoising_model().named_modules():
            if any(keyword in name for keyword in ["cam_encoder", "projector", "self_attn", "clean_x_embedder"]):
                for param in module.parameters():
                    param.requires_grad = True
        
        self.learning_rate = learning_rate
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        
        # 创建可视化目录
        self.vis_dir = "openx_training/visualizations"
        os.makedirs(self.vis_dir, exist_ok=True)

    def add_framepack_components(self):
        """🔧 添加FramePack相关组件"""
        if not hasattr(self.pipe.dit, 'clean_x_embedder'):
            inner_dim = self.pipe.dit.blocks[0].self_attn.q.weight.shape[0]
            
            class CleanXEmbedder(nn.Module):
                def __init__(self, inner_dim):
                    super().__init__()
                    # 参考hunyuan_video_packed.py的设计，但适配OpenX数据的分辨率
                    self.proj = nn.Conv3d(16, inner_dim, kernel_size=(1, 2, 2), stride=(1, 2, 2))
                    self.proj_2x = nn.Conv3d(16, inner_dim, kernel_size=(2, 4, 4), stride=(2, 4, 4))
                    self.proj_4x = nn.Conv3d(16, inner_dim, kernel_size=(4, 8, 8), stride=(4, 8, 8))
                
                def forward(self, x, scale="1x"):
                    if scale == "1x":
                        return self.proj(x)
                    elif scale == "2x":
                        return self.proj_2x(x)
                    elif scale == "4x":
                        return self.proj_4x(x)
                    else:
                        raise ValueError(f"Unsupported scale: {scale}")
            
            self.pipe.dit.clean_x_embedder = CleanXEmbedder(inner_dim)
            print("✅ 添加了FramePack的clean_x_embedder组件")
        
    def freeze_parameters(self):
        self.pipe.requires_grad_(False)
        self.pipe.eval()
        self.pipe.denoising_model().train()

    def training_step(self, batch, batch_idx):
        """🔧 使用FramePack风格的训练步骤 - 适配OpenX数据"""
        condition_frames = batch["condition_frames"][0].item()
        target_frames = batch["target_frames"][0].item()
        
        original_condition_frames = batch.get("original_condition_frames", [condition_frames * 4])[0]
        original_target_frames = batch.get("original_target_frames", [target_frames * 4])[0]

        dataset_name = batch.get("dataset_name", ["unknown"])[0]
        episode_name = batch.get("episode_name", ["unknown"])[0]
        
        # 🔧 准备FramePack风格的输入 - 确保有batch维度
        latents = batch["latents"].to(self.device)
        if len(latents.shape) == 4:  # [C, T, H, W]
            latents = latents.unsqueeze(0)  # -> [1, C, T, H, W]
        
        # 🔧 条件输入（处理空张量和维度）
        clean_latents = batch["clean_latents"].to(self.device) if batch["clean_latents"].numel() > 0 else None
        if clean_latents is not None and len(clean_latents.shape) == 4:
            clean_latents = clean_latents.unsqueeze(0)
        
        clean_latents_2x = batch["clean_latents_2x"].to(self.device) if batch["clean_latents_2x"].numel() > 0 else None
        if clean_latents_2x is not None and len(clean_latents_2x.shape) == 4:
            clean_latents_2x = clean_latents_2x.unsqueeze(0)
        
        clean_latents_4x = batch["clean_latents_4x"].to(self.device) if batch["clean_latents_4x"].numel() > 0 else None
        if clean_latents_4x is not None and len(clean_latents_4x.shape) == 4:
            clean_latents_4x = clean_latents_4x.unsqueeze(0)
        
        # 🔧 索引（处理空张量）
        latent_indices = batch["latent_indices"].to(self.device)
        clean_latent_indices = batch["clean_latent_indices"].to(self.device) if batch["clean_latent_indices"].numel() > 0 else None
        clean_latent_2x_indices = batch["clean_latent_2x_indices"].to(self.device) if batch["clean_latent_2x_indices"].numel() > 0 else None
        clean_latent_4x_indices = batch["clean_latent_4x_indices"].to(self.device) if batch["clean_latent_4x_indices"].numel() > 0 else None
        
        # 🔧 直接使用带mask的camera embeddings
        cam_emb = batch["camera"].to(self.device)
        camera_dropout_prob = 0.1  # 10%概率丢弃camera条件
        if random.random() < camera_dropout_prob:
            # 创建零camera embedding
            cam_emb = torch.zeros_like(cam_emb)
            print("应用camera dropout for CFG training")
        
        prompt_emb = batch["prompt_emb"]
        prompt_emb["context"] = prompt_emb["context"][0].to(self.device)
        image_emb = batch["image_emb"]

        if "clip_feature" in image_emb:
            image_emb["clip_feature"] = image_emb["clip_feature"][0].to(self.device)
        if "y" in image_emb:
            image_emb["y"] = image_emb["y"][0].to(self.device)

        # Loss计算
        self.pipe.device = self.device
        noise = torch.randn_like(latents)
        timestep_id = torch.randint(0, self.pipe.scheduler.num_train_timesteps, (1,))
        timestep = self.pipe.scheduler.timesteps[timestep_id].to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
        
        # 🔧 FramePack风格的噪声处理
        noisy_condition_latents = None
        if clean_latents is not None:
            noisy_condition_latents = copy.deepcopy(clean_latents)
            is_add_noise = random.random()
            if is_add_noise > 0.2:  # 80%概率添加噪声
                noise_cond = torch.randn_like(clean_latents)
                timestep_id_cond = torch.randint(0, self.pipe.scheduler.num_train_timesteps//4*3, (1,))
                timestep_cond = self.pipe.scheduler.timesteps[timestep_id_cond].to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
                noisy_condition_latents = self.pipe.scheduler.add_noise(clean_latents, noise_cond, timestep_cond)

        extra_input = self.pipe.prepare_extra_input(latents)
        origin_latents = copy.deepcopy(latents)
        noisy_latents = self.pipe.scheduler.add_noise(latents, noise, timestep)
        
        training_target = self.pipe.scheduler.training_target(latents, noise, timestep)
        
        # 🔧 使用FramePack风格的forward调用
        noise_pred = self.pipe.denoising_model()(
            noisy_latents, 
            timestep=timestep, 
            cam_emb=cam_emb,  # 🔧 直接传递带mask的camera embeddings
            # 🔧 FramePack风格的条件输入
            latent_indices=latent_indices,
            clean_latents=noisy_condition_latents if noisy_condition_latents is not None else clean_latents,
            clean_latent_indices=clean_latent_indices,
            clean_latents_2x=clean_latents_2x,
            clean_latent_2x_indices=clean_latent_2x_indices,
            clean_latents_4x=clean_latents_4x,
            clean_latent_4x_indices=clean_latent_4x_indices,
            **prompt_emb, 
            **extra_input, 
            **image_emb,
            use_gradient_checkpointing=self.use_gradient_checkpointing,
            use_gradient_checkpointing_offload=self.use_gradient_checkpointing_offload
        )
        
        # 计算loss
        loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
        loss = loss * self.pipe.scheduler.training_weight(timestep)
        print(f"----------loss{loss}--------------")

        return loss

    def configure_optimizers(self):
        trainable_modules = filter(lambda p: p.requires_grad, self.pipe.denoising_model().parameters())
        optimizer = torch.optim.AdamW(trainable_modules, lr=self.learning_rate)
        return optimizer
    
    def on_save_checkpoint(self, checkpoint):
        checkpoint_dir = "/share_zhuyixuan05/zhuyixuan05/ICLR2026/openx/openx_framepack"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        current_step = self.global_step
        checkpoint.clear()
        
        state_dict = self.pipe.denoising_model().state_dict()
        torch.save(state_dict, os.path.join(checkpoint_dir, f"step{current_step}.ckpt"))
        print(f"Saved OpenX FramePack model checkpoint: step{current_step}.ckpt")

def train_openx(args):
    """训练OpenX数据集的FramePack模型"""
    
    dataset = OpenXFramePackDataset(
        args.dataset_path,
        steps_per_epoch=args.steps_per_epoch,
        min_condition_frames=args.min_condition_frames,
        max_condition_frames=args.max_condition_frames,
        target_frames=args.target_frames,
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        batch_size=1,
        num_workers=args.dataloader_num_workers
    )
    
    model = OpenXLightningModelForTrain(
        dit_path=args.dit_path,
        learning_rate=args.learning_rate,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        resume_ckpt_path=args.resume_ckpt_path,
    )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices="auto",
        precision="bf16",
        strategy=args.training_strategy,
        default_root_dir=args.output_path,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=[],
        logger=False
    )
    trainer.fit(model, dataloader)

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Train OpenX Fractal Dataset with FramePack")
    parser.add_argument("--dataset_path", type=str, 
                       default="/share_zhuyixuan05/zhuyixuan05/openx-fractal-encoded",
                       help="OpenX编码数据集路径")
    parser.add_argument("--dit_path", type=str, default="models/Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors")
    parser.add_argument("--output_path", type=str, default="./")
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--steps_per_epoch", type=int, default=500)
    parser.add_argument("--max_epochs", type=int, default=30)
    parser.add_argument("--min_condition_frames", type=int, default=4, help="最小条件帧数")
    parser.add_argument("--max_condition_frames", type=int, default=120, help="最大条件帧数")
    parser.add_argument("--target_frames", type=int, default=32, help="目标帧数")
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--training_strategy", type=str, default="deepspeed_stage_1")
    parser.add_argument("--use_gradient_checkpointing", action="store_true")
    parser.add_argument("--use_gradient_checkpointing_offload", action="store_true")
    parser.add_argument("--resume_ckpt_path", type=str, default="/share_zhuyixuan05/zhuyixuan05/ICLR2026/openx/openx_framepack/step750.ckpt")
    
    args = parser.parse_args()
    
    print("🚀 开始训练OpenX Fractal数据集:")
    print(f"📁 数据集路径: {args.dataset_path}")
    print(f"🎯 条件帧范围: {args.min_condition_frames}-{args.max_condition_frames}")
    print(f"🎯 目标帧数: {args.target_frames}")
    
    train_openx(args)