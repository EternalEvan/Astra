import torch
import torch.nn as nn
import lightning as pl
import wandb
import os
import copy
from diffsynth import WanVideoReCamMasterPipeline, ModelManager
import os
import json
import torch
import numpy as np
from PIL import Image
import imageio
import random
from torchvision.transforms import v2
from einops import rearrange
from pose_classifier import PoseClassifier
from scipy.spatial.transform import Rotation as R

import pdb
# cam_c2w, [N * 4 * 4]
# stride, frame stride
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

def compute_relative_pose(pose_a, pose_b, use_torch=False):
    """
    计算相机B相对于相机A的相对位姿矩阵
    
    参数:
        pose_a: 相机A的外参矩阵 (4x4)，可以是numpy数组或PyTorch张量
                表示从世界坐标系到相机A坐标系的变换 (world → camera A)
        pose_b: 相机B的外参矩阵 (4x4)，可以是numpy数组或PyTorch张量
                表示从世界坐标系到相机B坐标系的变换 (world → camera B)
        use_torch: 是否使用PyTorch进行计算，默认使用NumPy
        
    返回:
        relative_pose: 相对位姿矩阵 (4x4)，表示从相机A坐标系到相机B坐标系的变换
                       (camera A → camera B)
    """
    # 检查输入形状
    assert pose_a.shape == (4, 4), f"相机A外参矩阵形状应为(4,4)，实际为{pose_a.shape}"
    assert pose_b.shape == (4, 4), f"相机B外参矩阵形状应为(4,4)，实际为{pose_b.shape}"
    
    if use_torch:
        # 确保输入是PyTorch张量
        if not isinstance(pose_a, torch.Tensor):
            pose_a = torch.from_numpy(pose_a).float()
        if not isinstance(pose_b, torch.Tensor):
            pose_b = torch.from_numpy(pose_b).float()
        
        # 计算相对位姿: relative_pose = pose_b × inverse(pose_a)
        pose_a_inv = torch.inverse(pose_a)
        relative_pose = torch.matmul(pose_b, pose_a_inv)
    else:
        # 确保输入是NumPy数组
        if not isinstance(pose_a, np.ndarray):
            pose_a = np.array(pose_a, dtype=np.float32)
        if not isinstance(pose_b, np.ndarray):
            pose_b = np.array(pose_b, dtype=np.float32)
        
        # 计算相对位姿: relative_pose = pose_b × inverse(pose_a)
        pose_a_inv = np.linalg.inv(pose_a)
        relative_pose = np.matmul(pose_b, pose_a_inv)
    
    return relative_pose


class DynamicSekaiDataset(torch.utils.data.Dataset):
    """支持动态历史长度的NuScenes数据集"""
    
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
        
        # 🔧 新增：VAE时间压缩比例
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
                # print(f"Found {len(self.scene_dirs)} scenes with encoded data")
        assert len(self.scene_dirs) > 0, "No encoded scenes found!"
        
        # 预处理设置
        # self.frame_process = v2.Compose([
        #     v2.CenterCrop(size=(height, width)),
        #     v2.Resize(size=(height, width), antialias=True),
        #     v2.ToTensor(),
        #     v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        # ])

    def calculate_relative_rotation(self, current_rotation, reference_rotation):
        """计算相对旋转四元数"""
        q_current = torch.tensor(current_rotation, dtype=torch.float32)
        q_ref = torch.tensor(reference_rotation, dtype=torch.float32)

        q_ref_inv = torch.tensor([q_ref[0], -q_ref[1], -q_ref[2], -q_ref[3]])

        w1, x1, y1, z1 = q_ref_inv
        w2, x2, y2, z2 = q_current

        relative_rotation = torch.tensor([
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        ])

        return relative_rotation
    
    def select_dynamic_segment(self, full_latents):
        """动态选择条件帧和目标帧 - 修正版本处理VAE时间压缩"""
        total_lens = full_latents.shape[1]
        # print(f"原始总帧数: {total_frames}, 压缩后: {compressed_total_frames}")
        # print(f"原始关键帧: {keyframe_indices[:5]}..., 压缩后: {compressed_keyframe_indices[:5]}...")
        
        # 随机选择条件帧长度（基于压缩后的帧数）

        min_condition_compressed = self.min_condition_frames // self.time_compression_ratio
        max_condition_compressed = self.max_condition_frames // self.time_compression_ratio
        
        target_frames_compressed = self.target_frames // self.time_compression_ratio
        max_condition_compressed = min(max_condition_compressed,total_lens - target_frames_compressed)
        # min_condition_compressed = min()

        ratio = random.random()
        print('ratio:',ratio)
        if ratio<0.15:
            condition_frames_compressed = 1
        elif 0.15<=ratio<0.3:
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
  

        
        # 使用条件段的最后一个关键帧作为reference
        reference_keyframe_compressed = start_frame_compressed
        
        # 🔧 找到对应的原始关键帧索引用于pose查找
        keyframe_original_idx = []
        for compressed_idx in range(start_frame_compressed,target_end_compressed):
            keyframe_original_idx.append(compressed_idx)
        
     
        
        return {
            'start_frame': start_frame_compressed,  # 压缩后的起始帧
            'condition_frames': condition_frames_compressed,  # 压缩后的条件帧数
            'target_frames': target_frames_compressed,  # 压缩后的目标帧数
            'condition_range': (start_frame_compressed, condition_end_compressed),
            'target_range': (condition_end_compressed, target_end_compressed),
            'keyframe_original_idx': keyframe_original_idx,  # 原始关键帧索引
     
            'original_condition_frames': condition_frames_compressed * self.time_compression_ratio,  # 用于记录
            'original_target_frames': target_frames_compressed * self.time_compression_ratio,
        }
    

    def create_pose_embeddings(self, cam_data, segment_info):
        """创建pose embeddings - 修正版本，确保与latent帧数对齐"""
        cam_data_seq = cam_data['extrinsic']   # 300 * 4 * 4
        # print(cam_data_seq.shape)
        keyframe_original_idx = segment_info['keyframe_original_idx']
        # target_keyframe_indices = segment_info['target_keyframe_indices']
        
        start_frame = segment_info['start_frame'] * self.time_compression_ratio
        end_frame = segment_info['target_range'][1] * self.time_compression_ratio
        # frame_range = cam_data_seq[start_frame:end_frame]

        relative_cams = []
        for idx in keyframe_original_idx:
            cam_prev = cam_data_seq[idx]
            cam_next = cam_data_seq[idx+1]
            # print('cam_prev:',cam_prev)
            # print('idx:',idx)
            # assert False
            relative_cam = compute_relative_pose_matrix(cam_prev,cam_next)
            # print(relative_cam)
            # print('relative_cam:',relative_cam)
            # assert False
            relative_cams.append(torch.as_tensor(relative_cam[:3,:]))
        
        pose_embedding = torch.stack(relative_cams, dim=0)
        pose_embedding = rearrange(pose_embedding, 'b c d -> b (c d)')
        # print(pose_embedding)
        pose_embedding = pose_embedding.to(torch.bfloat16)
        
        # print(pose_embedding.shape)
        # assert False
        # print()
        # traj_pos_coord_full, tarj_pos_angle_full = get_traj_position_change(cam_data_seq, self.time_compression_ratio)
        # traj_rot_angle_full = get_traj_rotation_change(cam_data_seq, self.time_compression_ratio)

        # motion_emb = 

        return {
            'camera': pose_embedding
        }
        
    def __getitem__(self, index):
        while True:
            try:
                # 随机选择一个场景
                scene_dir = random.choice(self.scene_dirs)
                
                # 加载场景信息
                # with open(os.path.join(scene_dir, "scene_info.json"), 'r') as f:
                #     scene_info = json.load(f)
                
                # 加载编码的视频数据
                encoded_data = torch.load(
                    os.path.join(scene_dir, "encoded_video.pth"),
                    weights_only=False,
                    map_location="cpu"
                )
                
                # 🔧 验证latent帧数是否符合预期
                full_latents = encoded_data['latents']  # [C, T, H, W]
                cam_data = encoded_data['cam_emb']
                # expected_latent_frames = scene_info['total_frames'] // self.time_compression_ratio
                actual_latent_frames = full_latents.shape[1]
                
                # print(f"场景 {os.path.basename(scene_dir)}: 原始帧数={scene_info['total_frames']}, "
                #       f"预期latent帧数={expected_latent_frames}, 实际latent帧数={actual_latent_frames}")
                
                # if abs(actual_latent_frames - expected_latent_frames) > 2:  # 允许小的舍入误差
                #     print(f"⚠️  Latent帧数不匹配，跳过此样本")
                #     continue
                
                # 动态选择段落
                segment_info = self.select_dynamic_segment(full_latents)
                # print(segment_info)
                if segment_info is None:
                    continue
                # print("segment_info:",segment_info)
                # 创建pose embeddings
                pose_data = self.create_pose_embeddings(cam_data, segment_info)
                if pose_data is None:
                    continue
                
                n = segment_info["condition_frames"]
                m = segment_info['target_frames']


                mask = torch.zeros(n+m, dtype=torch.float32)
                mask[:n] = 1.0
                mask = mask.view(-1, 1)


                pose_data["camera"] = torch.cat([pose_data["camera"], mask], dim=1)
                # print(pose_data['camera'].shape)
                # assert False
                # 🔧 使用压缩后的索引提取latent段落
                start_frame = segment_info['start_frame']  # 已经是压缩后的索引
                condition_frames = segment_info['condition_frames']  # 已经是压缩后的帧数
                target_frames = segment_info['target_frames']  # 已经是压缩后的帧数
                
                # print(f"提取latent段落: start={start_frame}, condition={condition_frames}, target={target_frames}")
                # print(f"Full latents shape: {full_latents.shape}")
                
                # # 确保索引不越界
                # if start_frame + condition_frames + target_frames > full_latents.shape[1]:
                #     print(f"索引越界，跳过: {start_frame + condition_frames + target_frames} > {full_latents.shape[1]}")
                #     continue
                
                condition_latents = full_latents[:, start_frame:start_frame+condition_frames, :, :]

                

                target_latents = full_latents[:, start_frame+condition_frames:start_frame+condition_frames+target_frames, :, :]
                
                # print(f"Condition latents shape: {condition_latents.shape}")
                # print(f"Target latents shape: {target_latents.shape}")
                
                # 拼接latents [condition, target]
                combined_latents = torch.cat([condition_latents, target_latents], dim=1)
                # print('latent:',combined_latents.requires_grad)
                # print('prompt:',encoded_data["prompt_emb"]["context"].requires_grad)
                # print('camera:',pose_data['camera'].requires_grad)
                result = {
                    "latents": combined_latents,
                    "prompt_emb": encoded_data["prompt_emb"],
                    "image_emb": encoded_data.get("image_emb", {}),
                    "camera": pose_data['camera'],
                    
                    "condition_frames": condition_frames,  # 压缩后的帧数
                    "target_frames": target_frames,  # 压缩后的帧数
                    "scene_name": os.path.basename(scene_dir),
                    # 🔧 新增：记录原始帧数用于调试
                    "original_condition_frames": segment_info['original_condition_frames'],
                    "original_target_frames": segment_info['original_target_frames'],
                }
                
                return result
                
            except Exception as e:
                print(f"Error loading sample: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    def __len__(self):
        return self.steps_per_epoch
    
class DynamicLightningModelForTrain(pl.LightningModule):
    def __init__(
        self,
        dit_path,
        learning_rate=1e-5,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        resume_ckpt_path=None
    ):
        super().__init__()
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        if os.path.isfile(dit_path):
            model_manager.load_models([dit_path])
        else:
            dit_path = dit_path.split(",")
            model_manager.load_models([dit_path])
        model_manager.load_models(["models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"])
        
        self.pipe = WanVideoReCamMasterPipeline.from_model_manager(model_manager)
        self.pipe.scheduler.set_timesteps(1000, training=True)

        # 添加相机编码器
        dim = self.pipe.dit.blocks[0].self_attn.q.weight.shape[0]
        for block in self.pipe.dit.blocks:
            block.cam_encoder = nn.Linear(13 , dim)
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
        
        # 只训练相机相关和注意力模块
        for name, module in self.pipe.denoising_model().named_modules():
            if any(keyword in name for keyword in ["cam_encoder", "projector", "self_attn"]):
                for param in module.parameters():
                    param.requires_grad = True
        
        self.learning_rate = learning_rate
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        
        # 创建可视化目录
        self.vis_dir = "sekai_dynamic/visualizations_dynamic"
        os.makedirs(self.vis_dir, exist_ok=True)
        
    def freeze_parameters(self):
        self.pipe.requires_grad_(False)
        self.pipe.eval()
        self.pipe.denoising_model().train()

    def training_step(self, batch, batch_idx):
        # 获取动态长度信息（这些已经是压缩后的帧数）
        condition_frames = batch["condition_frames"][0].item()  # 压缩后的condition长度
        target_frames = batch["target_frames"][0].item()       # 压缩后的target长度
        
        # 🔧 获取原始帧数用于日志记录
        original_condition_frames = batch.get("original_condition_frames", [condition_frames * 4])[0]
        print("condition_frames:",batch["condition_frames"])
        original_target_frames = batch.get("original_target_frames", [target_frames * 4])[0]
        
        # Data
        latents = batch["latents"].to(self.device)
        # print(f"压缩后condition帧数: {condition_frames}, target帧数: {target_frames}")
        # print(f"原始condition帧数: {original_condition_frames}, target帧数: {original_target_frames}")
        # print(f"Latents shape: {latents.shape}")
        
        # 裁剪空间尺寸以节省内存
        # target_height, target_width = 50, 70
        # current_height, current_width = latents.shape[3], latents.shape[4]
        
        # if current_height > target_height or current_width > target_width:
        #     h_start = (current_height - target_height) // 2
        #     w_start = (current_width - target_width) // 2
        #     latents = latents[:, :, :, 
        #                     h_start:h_start+target_height, 
        #                     w_start:w_start+target_width]
        
        prompt_emb = batch["prompt_emb"]
        prompt_emb["context"] = prompt_emb["context"][0].to(self.device)
        image_emb = batch["image_emb"]
        # print(f"裁剪后latents shape: {latents.shape}")

        if "clip_feature" in image_emb:
            image_emb["clip_feature"] = image_emb["clip_feature"][0].to(self.device)
        if "y" in image_emb:
            image_emb["y"] = image_emb["y"][0].to(self.device)
        
        cam_emb = batch["camera"].to(self.device)

        # Loss计算
        self.pipe.device = self.device
        noise = torch.randn_like(latents)
        timestep_id = torch.randint(0, self.pipe.scheduler.num_train_timesteps, (1,))
        timestep = self.pipe.scheduler.timesteps[timestep_id].to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
        
        noisy_condition_latents = copy.deepcopy(latents[:, :, :condition_frames, ...])
        is_add_noise = random.random()
        if is_add_noise > 0.2:
            # add noise to condition
            noise_cond = torch.randn_like(latents[:, :, :condition_frames, ...])
            timestep_id_cond = torch.randint(0, self.pipe.scheduler.num_train_timesteps//4*3, (1,))
            timestep_cond = self.pipe.scheduler.timesteps[timestep_id_cond].to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
            noisy_condition_latents = self.pipe.scheduler.add_noise(latents[:, :, :condition_frames, ...], noise_cond, timestep_cond)

        extra_input = self.pipe.prepare_extra_input(latents)
        origin_latents = copy.deepcopy(latents)
        noisy_latents = self.pipe.scheduler.add_noise(latents, noise, timestep)
        
        # 🔧 关键：使用压缩后的condition长度
        # condition部分保持clean，只对target部分加噪
        noisy_latents[:, :, :condition_frames, ...] = noisy_condition_latents #origin_latents[:, :, :condition_frames, ...]
        training_target = self.pipe.scheduler.training_target(latents, noise, timestep)
        # print(f"targe尺寸: {training_target.shape}")
        # 预测噪声
        noise_pred = self.pipe.denoising_model()(
            noisy_latents, timestep=timestep, cam_emb=cam_emb, **prompt_emb, **extra_input, **image_emb,
            use_gradient_checkpointing=self.use_gradient_checkpointing,
            use_gradient_checkpointing_offload=self.use_gradient_checkpointing_offload
        )
        # print(f"pred尺寸: {training_target.shape}")
        # 🔧 只对target部分计算loss（使用压缩后的索引）
        target_noise_pred = noise_pred[:, :, condition_frames:condition_frames+target_frames, ...]
        target_training_target = training_target[:, :, condition_frames:condition_frames+target_frames, ...]
        
        loss = torch.nn.functional.mse_loss(target_noise_pred.float(), target_training_target.float())
        loss = loss * self.pipe.scheduler.training_weight(timestep)
        print('--------loss------------:',loss)

        # 记录额外信息
        wandb.log({
            "train_loss": loss.item(),
            "timestep": timestep.item(),
            "condition_frames_compressed": condition_frames,  # 压缩后的帧数000
            "target_frames_compressed": target_frames,
            "condition_frames_original": original_condition_frames,  # 原始帧数
            "target_frames_original": original_target_frames,
            "total_frames_compressed": condition_frames + target_frames,
            "total_frames_original": original_condition_frames + original_target_frames,
            "global_step": self.global_step
        })

        return loss

    def configure_optimizers(self):
        trainable_modules = filter(lambda p: p.requires_grad, self.pipe.denoising_model().parameters())
        optimizer = torch.optim.AdamW(trainable_modules, lr=self.learning_rate)
        return optimizer
    
    def on_save_checkpoint(self, checkpoint):
        checkpoint_dir = "/share_zhuyixuan05/zhuyixuan05/ICLR2026/spatialvid/train_0"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        current_step = self.global_step
        checkpoint.clear()
        
        state_dict = self.pipe.denoising_model().state_dict()
        torch.save(state_dict, os.path.join(checkpoint_dir, f"step{current_step}_dynamic.ckpt"))
        print(f"Saved dynamic model checkpoint: step{current_step}_dynamic.ckpt")

def train_dynamic(args):
    """训练支持动态历史长度的模型"""
    dataset = DynamicSekaiDataset(
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
    
    model = DynamicLightningModelForTrain(
        dit_path=args.dit_path,
        learning_rate=args.learning_rate,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        resume_ckpt_path=args.resume_ckpt_path,
    )

    wandb.init(
        project="nuscenes-dynamic-recam",
        name=f"dynamic-{args.min_condition_frames}-{args.max_condition_frames}",
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
    )
    trainer.fit(model, dataloader)

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Dynamic ReCamMaster")
    parser.add_argument("--dataset_path", type=str, default="/share_zhuyixuan05/zhuyixuan05/spatialvid")
    parser.add_argument("--dit_path", type=str, default="models/Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors")
    parser.add_argument("--output_path", type=str, default="./")
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--steps_per_epoch", type=int, default=8000)
    parser.add_argument("--max_epochs", type=int, default=30)
    parser.add_argument("--min_condition_frames", type=int, default=10, help="最小条件帧数")
    parser.add_argument("--max_condition_frames", type=int, default=40, help="最大条件帧数")
    parser.add_argument("--target_frames", type=int, default=32, help="目标帧数")
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--training_strategy", type=str, default="deepspeed_stage_1")
    parser.add_argument("--use_gradient_checkpointing", action="store_true")
    parser.add_argument("--use_gradient_checkpointing_offload", action="store_true")
    parser.add_argument("--resume_ckpt_path", type=str, default=None)
    
    args = parser.parse_args()
    
    train_dynamic(args)