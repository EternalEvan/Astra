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
from scipy.spatial.transform import Rotation as R
import traceback
import argparse

def compute_relative_pose_matrix(pose1, pose2):
    """
    计算相邻两帧的相对位姿，返回3×4的相机矩阵 [R_rel | t_rel]
    
    参数:
    pose1: 第i帧的相机位姿，形状为(7,)的数组 [tx1, ty1, tz1, qx1, qy1, qz1, qw1]
    pose2: 第i+1帧的相机位姿，形状为(7,)的数组 [tx2, ty2, tz2, qx2, qy2, qz2, qw2]
    
    返回:
    relative_matrix: 3×4的相对位姿矩阵，前3列是旋转矩阵R_rel，第4列是平移向量t_rel
    """
    # 分离平移向量和四元数
    t1 = pose1[:3]  # 第i帧平移 [tx1, ty1, tz1]
    q1 = pose1[3:]  # 第i帧四元数 [qx1, qy1, qz1, qw1]
    t2 = pose2[:3]  # 第i+1帧平移
    q2 = pose2[3:]  # 第i+1帧四元数
    
    # 1. 计算相对旋转矩阵 R_rel
    rot1 = R.from_quat(q1)  # 第i帧旋转
    rot2 = R.from_quat(q2)  # 第i+1帧旋转
    rot_rel = rot2 * rot1.inv()  # 相对旋转 = 后一帧旋转 × 前一帧旋转的逆
    R_rel = rot_rel.as_matrix()  # 转换为3×3矩阵
    
    # 2. 计算相对平移向量 t_rel
    R1_T = rot1.as_matrix().T  # 前一帧旋转矩阵的转置（等价于逆）
    t_rel = R1_T @ (t2 - t1)   # 相对平移 = R1^T × (t2 - t1)
    
    # 3. 组合为3×4矩阵 [R_rel | t_rel]
    relative_matrix = np.hstack([R_rel, t_rel.reshape(3, 1)])
    
    return relative_matrix


class SpatialVidFramePackDataset(torch.utils.data.Dataset):
    """支持FramePack机制的SpatialVid数据集"""
    
    def __init__(self, base_path, steps_per_epoch, 
                 min_condition_frames=10, max_condition_frames=40,
                 target_frames=10, height=900, width=1600):
        self.base_path = base_path
        self.scenes_path = base_path
        self.min_condition_frames = min_condition_frames
        self.max_condition_frames = max_condition_frames
        self.target_frames = target_frames
        self.height = height
        self.width = width
        self.steps_per_epoch = steps_per_epoch
        self.pose_classifier = PoseClassifier()
        
        # VAE时间压缩比例
        self.time_compression_ratio = 4  # VAE将时间维度压缩4倍
        
        # 查找所有处理好的场景
        self.scene_dirs = []
        if os.path.exists(self.scenes_path):
            for item in os.listdir(self.scenes_path):
                scene_dir = os.path.join(self.scenes_path, item)
                if os.path.isdir(scene_dir):
                    encoded_path = os.path.join(scene_dir, "encoded_video.pth")
                    if os.path.exists(encoded_path):
                        self.scene_dirs.append(scene_dir)
        
        print(f"🔧 找到 {len(self.scene_dirs)} 个SpatialVid场景")
        assert len(self.scene_dirs) > 0, "No encoded scenes found!"

    def select_dynamic_segment_framepack(self, full_latents):
        """🔧 FramePack风格的动态选择条件帧和目标帧 - SpatialVid版本"""
        total_lens = full_latents.shape[1]
        
        min_condition_compressed = self.min_condition_frames // self.time_compression_ratio
        max_condition_compressed = self.max_condition_frames // self.time_compression_ratio
        target_frames_compressed = self.target_frames // self.time_compression_ratio
        max_condition_compressed = min(max_condition_compressed, total_lens - target_frames_compressed)
        
        ratio = random.random()
        #print('ratio:', ratio)
        if ratio < 0.15:
            condition_frames_compressed = 1
        elif 0.15 <= ratio < 0.9:
            condition_frames_compressed = random.randint(min_condition_compressed, max_condition_compressed)
        else:
            condition_frames_compressed = target_frames_compressed
        
        # 确保有足够的帧数
        min_required_frames = condition_frames_compressed + target_frames_compressed
        if total_lens < min_required_frames:
            print(f"压缩后帧数不足: {total_lens} < {min_required_frames}")
            return None
        
        # 随机选择起始位置（基于压缩后的帧数）
        max_start = total_lens - min_required_frames - 1
        start_frame_compressed = random.randint(0, max_start)
        
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
            clean_latent_2x_start = max(start_frame_compressed, condition_end_compressed - 2)
            clean_latent_2x_indices = torch.arange(clean_latent_2x_start-1, condition_end_compressed-1)
        else:
            # 如果condition帧数不足2帧，创建空索引
            clean_latent_2x_indices = torch.tensor([], dtype=torch.long)
        
        # 🔧 4x帧：根据实际condition长度确定，最多16帧
        if condition_frames_compressed >= 1:
            # 取最多16帧的历史（如果有的话）
            clean_4x_start = max(start_frame_compressed, condition_end_compressed - 16)
            clean_latent_4x_indices = torch.arange(clean_4x_start-3, condition_end_compressed-3)
        else:
            clean_latent_4x_indices = torch.tensor([], dtype=torch.long)
        
        # 对应的原始关键帧索引 - SpatialVid特有：每隔1帧而不是4帧
        keyframe_original_idx = []
        for compressed_idx in range(start_frame_compressed, target_end_compressed):
            keyframe_original_idx.append(compressed_idx)  # SpatialVid使用1倍间隔
        
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
        """🔧 创建SpatialVid风格的pose embeddings - camera间隔为1帧而非4帧"""
        cam_data_seq = cam_data['extrinsic']   # N * 4 * 4
        
        # 🔧 为所有帧（condition + target）计算camera embedding
        # SpatialVid特有：每隔1帧而不是4帧
        keyframe_original_idx = segment_info['keyframe_original_idx']
        
        relative_cams = []
        for idx in keyframe_original_idx:
            if idx + 1 < len(cam_data_seq):
                cam_prev = cam_data_seq[idx]
                cam_next = cam_data_seq[idx + 1]  # SpatialVid: 每隔1帧
                relative_cam = compute_relative_pose_matrix(cam_prev, cam_next)
                relative_cams.append(torch.as_tensor(relative_cam[:3, :]))
            else:
                # 如果没有下一帧，使用零运动
                identity_cam = torch.zeros(3, 4)
                relative_cams.append(identity_cam)
        
        if len(relative_cams) == 0:
            return None
            
        pose_embedding = torch.stack(relative_cams, dim=0)
        pose_embedding = rearrange(pose_embedding, 'b c d -> b (c d)')
        pose_embedding = pose_embedding.to(torch.bfloat16)

        return pose_embedding

    def prepare_framepack_inputs(self, full_latents, segment_info):
        """🔧 准备FramePack风格的多尺度输入 - SpatialVid版本"""
        # 🔧 修正：处理4维输入 [C, T, H, W]，添加batch维度
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
        
        # 🔧 修正：检查是否有有效的4x索引
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
        
        # 🔧 修正：检查是否有有效的2x索引
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
                # 随机选择一个场景
                scene_dir = random.choice(self.scene_dirs)
                
                # 加载编码的视频数据
                encoded_data = torch.load(
                    os.path.join(scene_dir, "encoded_video.pth"),
                    weights_only=False,
                    map_location="cpu"
                )
                
                # 🔧 验证latent帧数是否符合预期
                full_latents = encoded_data['latents']  # [C, T, H, W]
                cam_data = encoded_data['cam_emb']
                actual_latent_frames = full_latents.shape[1]
                
                # 动态选择段落
                segment_info = self.select_dynamic_segment_framepack(full_latents)
                if segment_info is None:
                    continue
                
                # 创建pose embeddings - SpatialVid版本
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
                    "clean_latents": framepack_inputs['clean_latents'],  # 条件帧
                    "clean_latents_2x": framepack_inputs['clean_latents_2x'],
                    "clean_latents_4x": framepack_inputs['clean_latents_4x'],
                    "latent_indices": framepack_inputs['latent_indices'],
                    "clean_latent_indices": framepack_inputs['clean_latent_indices'],
                    "clean_latent_2x_indices": framepack_inputs['clean_latent_2x_indices'],
                    "clean_latent_4x_indices": framepack_inputs['clean_latent_4x_indices'],
                    
                    # 🔧 直接传递带mask的camera embeddings
                    "camera": camera_with_mask,  # 所有帧的camera embeddings（带mask）
                    
                    "prompt_emb": encoded_data["prompt_emb"],
                    "image_emb": encoded_data.get("image_emb", {}),
                    
                    "condition_frames": n,  # 压缩后的帧数
                    "target_frames": m,  # 压缩后的帧数
                    "scene_name": os.path.basename(scene_dir),
                    "dataset_name": "spatialvid",
                    # 🔧 新增：记录原始帧数用于调试
                    "original_condition_frames": segment_info['original_condition_frames'],
                    "original_target_frames": segment_info['original_target_frames'],
                }
                
                return result
                
            except Exception as e:
                print(f"Error loading sample: {e}")
                traceback.print_exc()
                continue
    
    def __len__(self):
        return self.steps_per_epoch


def replace_dit_model_in_manager():
    """在模型加载前替换DiT模型类"""
    from diffsynth.models.wan_video_dit_moe import WanModelMoe
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
                    new_model_classes.append(WanModelMoe)  # 替换为新的类
                    print(f"✅ 替换了模型类: {name} -> WanModelMoe")
                else:
                    new_model_names.append(name)
                    new_model_classes.append(cls)
            
            # 更新配置
            model_loader_configs[i] = (keys_hash, keys_hash_with_shape, new_model_names, new_model_classes, model_resource)


class SpatialVidFramePackLightningModel(pl.LightningModule):
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
        self.add_moe_components()

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
            self.pipe.dit.load_state_dict(state_dict, strict=False)
            print('load checkpoint:', resume_ckpt_path)

        self.freeze_parameters()
        
        # 只训练相机相关和注意力模块以及FramePack相关组件
        for name, module in self.pipe.denoising_model().named_modules():
            if any(keyword in name for keyword in ["moe","sekai_processor"]):
                for param in module.parameters():
                    param.requires_grad = True
        
        self.learning_rate = learning_rate
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        
        # 创建可视化目录
        self.vis_dir = "spatialvid_framepack/visualizations"
        os.makedirs(self.vis_dir, exist_ok=True)

    def add_framepack_components(self):
        """🔧 添加FramePack相关组件"""
        if not hasattr(self.pipe.dit, 'clean_x_embedder'):
            inner_dim = self.pipe.dit.blocks[0].self_attn.q.weight.shape[0]
            
            class CleanXEmbedder(nn.Module):
                def __init__(self, inner_dim):
                    super().__init__()
                    # 参考hunyuan_video_packed.py的设计
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
            
    def add_moe_components(self):
        """🔧 添加MoE相关组件 - 类似add_framepack_components的方式"""
        if not hasattr(self.pipe.dit, 'moe_config'):
            self.pipe.dit.moe_config = self.moe_config
            print("✅ 添加了MoE配置到模型")
        
        # 为每个block动态添加MoE组件
        dim = self.pipe.dit.blocks[0].self_attn.q.weight.shape[0]
        unified_dim = 25
        
        for i, block in enumerate(self.pipe.dit.blocks):
            from diffsynth.models.wan_video_dit_moe import ModalityProcessor, MultiModalMoE
            
            # Sekai模态处理器 - 输出unified_dim
            block.sekai_processor = ModalityProcessor("sekai", 13, unified_dim)
            
            # NuScenes模态处理器 - 输出unified_dim
            # block.nuscenes_processor = ModalityProcessor("nuscenes", 8, unified_dim)

            # block.openx_processor = ModalityProcessor("openx", 13, unified_dim)  # OpenX使用13维输入，类似sekai但独立处理

            
            # MoE网络 - 输入unified_dim，输出dim
            block.moe = MultiModalMoE(
                unified_dim=unified_dim,
                output_dim=dim,  # 输出维度匹配transformer block的dim
                num_experts=1,
                top_k=1
            )
            
        
    def freeze_parameters(self):
        self.pipe.requires_grad_(False)
        self.pipe.eval()
        self.pipe.denoising_model().train()

    def training_step(self, batch, batch_idx):
        """🔧 使用FramePack风格的训练步骤 - SpatialVid版本"""
        condition_frames = batch["condition_frames"][0].item()
        target_frames = batch["target_frames"][0].item()
        
        original_condition_frames = batch.get("original_condition_frames", [condition_frames * 4])[0]
        original_target_frames = batch.get("original_target_frames", [target_frames * 4])[0]

        dataset_name = batch.get("dataset_name", ["unknown"])[0]
        scene_name = batch.get("scene_name", ["unknown"])[0]
        
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
        noise_pred, moe_loss = self.pipe.denoising_model()(
            noisy_latents, 
            timestep=timestep, 
            cam_emb=cam_emb,  # 🔧 直接传递带mask的camera embeddings
            # 🔧 FramePack风格的条件输入
            modality_inputs={"sekai": cam_emb},
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
        print(f'--------loss ({dataset_name})------------:', loss)

        return loss

    def configure_optimizers(self):
        trainable_modules = filter(lambda p: p.requires_grad, self.pipe.denoising_model().parameters())
        optimizer = torch.optim.AdamW(trainable_modules, lr=self.learning_rate)
        return optimizer
    
    def on_save_checkpoint(self, checkpoint):
        checkpoint_dir = "/share_zhuyixuan05/zhuyixuan05/ICLR2026/spatialvid/spatialvid_moe_test"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        current_step = self.global_step
        checkpoint.clear()
        
        state_dict = self.pipe.denoising_model().state_dict()
        torch.save(state_dict, os.path.join(checkpoint_dir, f"step{current_step}.ckpt"))
        print(f"Saved SpatialVid FramePack model checkpoint: step{current_step}.ckpt")


def train_spatialvid_framepack(args):
    """训练支持FramePack机制的SpatialVid模型"""
    dataset = SpatialVidFramePackDataset(
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
    
    model = SpatialVidFramePackLightningModel(
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
        callbacks=[pl.pytorch.callbacks.ModelCheckpoint(save_top_k=-1)],
        logger=False
    )
    trainer.fit(model, dataloader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train SpatialVid FramePack Dynamic ReCamMaster")
    parser.add_argument("--dataset_path", type=str, default="/share_zhuyixuan05/zhuyixuan05/spatialvid")
    parser.add_argument("--dit_path", type=str, default="models/Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors")
    parser.add_argument("--output_path", type=str, default="./")
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--steps_per_epoch", type=int, default=400)
    parser.add_argument("--max_epochs", type=int, default=30)
    parser.add_argument("--min_condition_frames", type=int, default=10, help="最小条件帧数")
    parser.add_argument("--max_condition_frames", type=int, default=40, help="最大条件帧数")
    parser.add_argument("--target_frames", type=int, default=32, help="目标帧数")
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--training_strategy", type=str, default="deepspeed_stage_1")
    parser.add_argument("--use_gradient_checkpointing", action="store_true")
    parser.add_argument("--use_gradient_checkpointing_offload", action="store_true")
    parser.add_argument("--resume_ckpt_path", type=str, default="/share_zhuyixuan05/zhuyixuan05/ICLR2026/sekai/sekai_walking_framepack/step1000_framepack.ckpt")
    
    args = parser.parse_args()
    
    print("🔧 开始训练SpatialVid FramePack模型:")
    print(f"📁 数据集路径: {args.dataset_path}")
    print(f"🎯 条件帧范围: {args.min_condition_frames}-{args.max_condition_frames}")
    print(f"🎯 目标帧数: {args.target_frames}")
    print("🔧 特殊优化:")
    print("  - 使用WanModelFuture模型架构")
    print("  - 添加FramePack多尺度输入支持")
    print("  - SpatialVid特有：camera间隔为1帧")
    print("  - CFG训练支持（10%概率camera dropout）")
    
    train_spatialvid_framepack(args)