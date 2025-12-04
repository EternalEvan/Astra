import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import imageio
import json
from diffsynth import WanVideoReCamMasterPipeline, ModelManager
import argparse
from torchvision.transforms import v2
from einops import rearrange
import random
import copy
from datetime import datetime

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

def load_encoded_video_from_pth(pth_path, start_frame=0, num_frames=10):
    """从pth文件加载预编码的视频数据"""
    print(f"Loading encoded video from {pth_path}")
    
    encoded_data = torch.load(pth_path, weights_only=False, map_location="cpu")
    full_latents = encoded_data['latents']  # [C, T, H, W]
    
    print(f"Full latents shape: {full_latents.shape}")
    print(f"Extracting frames {start_frame} to {start_frame + num_frames}")
    
    if start_frame + num_frames > full_latents.shape[1]:
        raise ValueError(f"Not enough frames: requested {start_frame + num_frames}, available {full_latents.shape[1]}")
    
    condition_latents = full_latents[:, start_frame:start_frame + num_frames, :, :]
    print(f"Extracted condition latents shape: {condition_latents.shape}")
    
    return condition_latents, encoded_data


def compute_relative_pose(pose_a, pose_b, use_torch=False):
    """计算相机B相对于相机A的相对位姿矩阵"""
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


def replace_dit_model_in_manager():
    """替换DiT模型类为MoE版本"""
    from diffsynth.models.wan_video_dit_moe import WanModelMoe
    from diffsynth.configs.model_config import model_loader_configs
    
    for i, config in enumerate(model_loader_configs):
        keys_hash, keys_hash_with_shape, model_names, model_classes, model_resource = config
        
        if 'wan_video_dit' in model_names:
            new_model_names = []
            new_model_classes = []
            
            for name, cls in zip(model_names, model_classes):
                if name == 'wan_video_dit':
                    new_model_names.append(name)
                    new_model_classes.append(WanModelMoe)
                    print(f"✅ 替换了模型类: {name} -> WanModelMoe")
                else:
                    new_model_names.append(name)
                    new_model_classes.append(cls)
            
            model_loader_configs[i] = (keys_hash, keys_hash_with_shape, new_model_names, new_model_classes, model_resource)


def add_framepack_components(dit_model):
    """添加FramePack相关组件"""
    if not hasattr(dit_model, 'clean_x_embedder'):
        inner_dim = dit_model.blocks[0].self_attn.q.weight.shape[0]
        
        class CleanXEmbedder(nn.Module):
            def __init__(self, inner_dim):
                super().__init__()
                self.proj = nn.Conv3d(16, inner_dim, kernel_size=(1, 2, 2), stride=(1, 2, 2))
                self.proj_2x = nn.Conv3d(16, inner_dim, kernel_size=(2, 4, 4), stride=(2, 4, 4))
                self.proj_4x = nn.Conv3d(16, inner_dim, kernel_size=(4, 8, 8), stride=(4, 8, 8))
            
            def forward(self, x, scale="1x"):
                if scale == "1x":
                    x = x.to(self.proj.weight.dtype)
                    return self.proj(x)
                elif scale == "2x":
                    x = x.to(self.proj_2x.weight.dtype)
                    return self.proj_2x(x)
                elif scale == "4x":
                    x = x.to(self.proj_4x.weight.dtype)
                    return self.proj_4x(x)
                else:
                    raise ValueError(f"Unsupported scale: {scale}")
        
        dit_model.clean_x_embedder = CleanXEmbedder(inner_dim)
        model_dtype = next(dit_model.parameters()).dtype
        dit_model.clean_x_embedder = dit_model.clean_x_embedder.to(dtype=model_dtype)
        print("✅ 添加了FramePack的clean_x_embedder组件")


def add_moe_components(dit_model, moe_config):
    """🔧 添加MoE相关组件 - 修正版本"""
    if not hasattr(dit_model, 'moe_config'):
        dit_model.moe_config = moe_config
        print("✅ 添加了MoE配置到模型")
    dit_model.top_k = moe_config.get("top_k", 1)

    # 为每个block动态添加MoE组件
    dim = dit_model.blocks[0].self_attn.q.weight.shape[0]
    unified_dim = moe_config.get("unified_dim", 25)
    num_experts = moe_config.get("num_experts", 4)
    from diffsynth.models.wan_video_dit_moe import ModalityProcessor, MultiModalMoE
    dit_model.sekai_processor = ModalityProcessor("sekai", 13, unified_dim)
    dit_model.nuscenes_processor = ModalityProcessor("nuscenes", 8, unified_dim)
    dit_model.openx_processor = ModalityProcessor("openx", 13, unified_dim)  # OpenX使用13维输入，类似sekai但独立处理
    dit_model.global_router = nn.Linear(unified_dim, num_experts)


    for i, block in enumerate(dit_model.blocks):
        # MoE网络 - 输入unified_dim，输出dim
        block.moe = MultiModalMoE(
            unified_dim=unified_dim,
            output_dim=dim,  # 输出维度匹配transformer block的dim
            num_experts=moe_config.get("num_experts", 4),
            top_k=moe_config.get("top_k", 2)
        )
        
        print(f"✅ Block {i} 添加了MoE组件 (unified_dim: {unified_dim}, experts: {moe_config.get('num_experts', 4)})")


def generate_sekai_camera_embeddings_sliding(
    cam_data, 
    start_frame, 
    initial_condition_frames, 
    new_frames, 
    total_generated, 
    use_real_poses=True,
    direction="left"):
    """
    为Sekai数据集生成camera embeddings - 滑动窗口版本
    
    Args:
        cam_data: 包含Sekai相机外参的字典, 键'extrinsic'对应一个N*4*4的numpy数组
        start_frame: 当前生成起始帧索引
        initial_condition_frames: 初始条件帧数
        new_frames: 本次生成的新帧数
        total_generated: 已生成的总帧数
        use_real_poses: 是否使用真实的Sekai相机位姿
        direction: 相机运动方向，默认为"left"
        
    Returns:
        camera_embedding: 形状为(M, 3*4 + 1)的torch张量, M为生成的总帧数
    """
    time_compression_ratio = 4
    
    # 计算FramePack实际需要的camera帧数
    # 1帧初始 + 16帧4x + 2帧2x + 1帧1x + new_frames
    framepack_needed_frames = 1 + 16 + 2 + 1 + new_frames
    
    if use_real_poses and cam_data is not None and 'extrinsic' in cam_data:
        print("🔧 使用真实Sekai camera数据")
        cam_extrinsic = cam_data['extrinsic']
        
        # 确保生成足够长的camera序列
        max_needed_frames = max(
            start_frame + initial_condition_frames + new_frames,
            framepack_needed_frames,
            30
        )
        
        print(f"🔧 计算Sekai camera序列长度:")
        print(f"  - 基础需求: {start_frame + initial_condition_frames + new_frames}")
        print(f"  - FramePack需求: {framepack_needed_frames}")
        print(f"  - 最终生成: {max_needed_frames}")
        
        relative_poses = []
        for i in range(max_needed_frames):
            # 计算当前帧在原始序列中的位置
            frame_idx = i * time_compression_ratio
            next_frame_idx = frame_idx + time_compression_ratio
            
            if next_frame_idx < len(cam_extrinsic):
                cam_prev = cam_extrinsic[frame_idx]
                cam_next = cam_extrinsic[next_frame_idx]
                relative_pose = compute_relative_pose(cam_prev, cam_next)
                relative_poses.append(torch.as_tensor(relative_pose[:3, :]))
            else:
                # 超出范围，使用零运动
                print(f"⚠️ 帧{frame_idx}超出camera数据范围，使用零运动")
                relative_poses.append(torch.zeros(3, 4))
        
        pose_embedding = torch.stack(relative_poses, dim=0)
        pose_embedding = rearrange(pose_embedding, 'b c d -> b (c d)')
        
        # 创建对应长度的mask序列
        mask = torch.zeros(max_needed_frames, 1, dtype=torch.float32)
        # 从start_frame到start_frame+initial_condition_frames标记为condition
        condition_end = min(start_frame + initial_condition_frames, max_needed_frames)
        mask[start_frame:condition_end] = 1.0
        
        camera_embedding = torch.cat([pose_embedding, mask], dim=1)
        print(f"🔧 Sekai真实camera embedding shape: {camera_embedding.shape}")
        return camera_embedding.to(torch.bfloat16)
        
    else:
        # 确保生成足够长的camera序列
        max_needed_frames = max(
            start_frame + initial_condition_frames + new_frames, 
            framepack_needed_frames, 
            30)
            
        print(f"🔧 生成Sekai合成camera帧数: {max_needed_frames}")
        
        CONDITION_FRAMES = initial_condition_frames
        STAGE_1 = new_frames//2
        STAGE_2 = new_frames - STAGE_1
        
        if direction=="left":
            print("--------------- LEFT TURNING MODE ---------------")
            relative_poses = []
            for i in range(max_needed_frames):
                if i < CONDITION_FRAMES:
                    # 输入的条件帧默认的相机位姿为零运动
                    pose = np.eye(4, dtype=np.float32)
                elif i < CONDITION_FRAMES+STAGE_1+STAGE_2:
                    # 左转
                    yaw_per_frame = 0.03

                    # 旋转矩阵
                    cos_yaw = np.cos(yaw_per_frame)
                    sin_yaw = np.sin(yaw_per_frame)
                    
                    # 前进
                    forward_speed = 0.00

                    pose = np.eye(4, dtype=np.float32)
                    
                    pose[0, 0] = cos_yaw
                    pose[0, 2] = sin_yaw
                    pose[2, 0] = -sin_yaw
                    pose[2, 2] = cos_yaw
                    pose[2, 3] = -forward_speed
                else:
                    # 超出条件帧与目标帧的部分，保持静止
                    pose = np.eye(4, dtype=np.float32)
                
                relative_pose = pose[:3, :]
                relative_poses.append(torch.as_tensor(relative_pose))
        
        elif direction=="right":
            print("--------------- RIGHT TURNING MODE ---------------")
            relative_poses = []
            for i in range(max_needed_frames):
                if i < CONDITION_FRAMES:
                    # 输入的条件帧默认的相机位姿为零运动
                    pose = np.eye(4, dtype=np.float32)
                elif i < CONDITION_FRAMES+STAGE_1+STAGE_2:
                    # 右转
                    yaw_per_frame = -0.03

                    # 旋转矩阵
                    cos_yaw = np.cos(yaw_per_frame)
                    sin_yaw = np.sin(yaw_per_frame)
                    
                    # 前进
                    forward_speed = 0.00

                    pose = np.eye(4, dtype=np.float32)
                    
                    pose[0, 0] = cos_yaw
                    pose[0, 2] = sin_yaw
                    pose[2, 0] = -sin_yaw
                    pose[2, 2] = cos_yaw
                    pose[2, 3] = -forward_speed
                else:
                    # 超出条件帧与目标帧的部分，保持静止
                    pose = np.eye(4, dtype=np.float32)
                
                relative_pose = pose[:3, :]
                relative_poses.append(torch.as_tensor(relative_pose))
        
        elif direction=="forward_left":
            print("--------------- FORWARD LEFT MODE ---------------")
            relative_poses = []
            for i in range(max_needed_frames):
                if i < CONDITION_FRAMES:
                    # 输入的条件帧默认的相机位姿为零运动
                    pose = np.eye(4, dtype=np.float32)
                elif i < CONDITION_FRAMES+STAGE_1+STAGE_2:
                    # 左转
                    yaw_per_frame = 0.03

                    # 旋转矩阵
                    cos_yaw = np.cos(yaw_per_frame)
                    sin_yaw = np.sin(yaw_per_frame)
                    
                    # 前进
                    forward_speed = 0.03

                    pose = np.eye(4, dtype=np.float32)
                    
                    pose[0, 0] = cos_yaw
                    pose[0, 2] = sin_yaw
                    pose[2, 0] = -sin_yaw
                    pose[2, 2] = cos_yaw
                    pose[2, 3] = -forward_speed
                
                else:
                    # 超出条件帧与目标帧的部分，保持静止
                    pose = np.eye(4, dtype=np.float32)
                    
                relative_pose = pose[:3, :]
                relative_poses.append(torch.as_tensor(relative_pose))
        
        elif direction=="forward_right":
            print("--------------- FORWARD RIGHT MODE ---------------")
            relative_poses = []
            for i in range(max_needed_frames):
                if i < CONDITION_FRAMES:
                    # 输入的条件帧默认的相机位姿为零运动
                    pose = np.eye(4, dtype=np.float32)
                elif i < CONDITION_FRAMES+STAGE_1+STAGE_2:
                    # 右转
                    yaw_per_frame = -0.03

                    # 旋转矩阵
                    cos_yaw = np.cos(yaw_per_frame)
                    sin_yaw = np.sin(yaw_per_frame)
                    
                    # 前进
                    forward_speed = 0.03

                    pose = np.eye(4, dtype=np.float32)
                    
                    pose[0, 0] = cos_yaw
                    pose[0, 2] = sin_yaw
                    pose[2, 0] = -sin_yaw
                    pose[2, 2] = cos_yaw
                    pose[2, 3] = -forward_speed
                
                else:
                    # 超出条件帧与目标帧的部分，保持静止
                    pose = np.eye(4, dtype=np.float32)
                    
                relative_pose = pose[:3, :]
                relative_poses.append(torch.as_tensor(relative_pose))
        
        elif direction=="s_curve":
            print("--------------- S CURVE MODE ---------------")
            relative_poses = []
            for i in range(max_needed_frames):
                if i < CONDITION_FRAMES:
                    # 输入的条件帧默认的相机位姿为零运动
                    pose = np.eye(4, dtype=np.float32)
                elif i < CONDITION_FRAMES+STAGE_1:
                    # 左转
                    yaw_per_frame = 0.03

                    # 旋转矩阵
                    cos_yaw = np.cos(yaw_per_frame)
                    sin_yaw = np.sin(yaw_per_frame)
                    
                    # 前进
                    forward_speed = 0.03

                    pose = np.eye(4, dtype=np.float32)
                    
                    pose[0, 0] = cos_yaw
                    pose[0, 2] = sin_yaw
                    pose[2, 0] = -sin_yaw
                    pose[2, 2] = cos_yaw
                    pose[2, 3] = -forward_speed
                    
                elif i < CONDITION_FRAMES+STAGE_1+STAGE_2:
                    # 右转
                    yaw_per_frame = -0.03

                    # 旋转矩阵
                    cos_yaw = np.cos(yaw_per_frame)
                    sin_yaw = np.sin(yaw_per_frame)
                    
                    # 前进
                    forward_speed = 0.03
                    # 轻微向左漂移，保持惯性
                    if i < CONDITION_FRAMES+STAGE_1+STAGE_2//3:
                        radius_shift = -0.01
                    else:
                        radius_shift = 0.00

                    pose = np.eye(4, dtype=np.float32)
                    
                    pose[0, 0] = cos_yaw
                    pose[0, 2] = sin_yaw
                    pose[2, 0] = -sin_yaw
                    pose[2, 2] = cos_yaw
                    pose[2, 3] = -forward_speed
                    pose[0, 3] = radius_shift
                    
                else:
                    # 超出条件帧与目标帧的部分，保持静止
                    pose = np.eye(4, dtype=np.float32)
                    
                relative_pose = pose[:3, :]
                relative_poses.append(torch.as_tensor(relative_pose))
                    
        elif direction=="left_right":
            print("--------------- LEFT RIGHT MODE ---------------")
            relative_poses = []
            for i in range(max_needed_frames):
                if i < CONDITION_FRAMES:
                    # 输入的条件帧默认的相机位姿为零运动
                    pose = np.eye(4, dtype=np.float32)
                elif i < CONDITION_FRAMES+STAGE_1:
                    # 左转
                    yaw_per_frame = 0.03

                    # 旋转矩阵
                    cos_yaw = np.cos(yaw_per_frame)
                    sin_yaw = np.sin(yaw_per_frame)
                    
                    # 前进
                    forward_speed = 0.00

                    pose = np.eye(4, dtype=np.float32)
                    
                    pose[0, 0] = cos_yaw
                    pose[0, 2] = sin_yaw
                    pose[2, 0] = -sin_yaw
                    pose[2, 2] = cos_yaw
                    pose[2, 3] = -forward_speed
                    
                elif i < CONDITION_FRAMES+STAGE_1+STAGE_2:
                    # 右转
                    yaw_per_frame = -0.03

                    # 旋转矩阵
                    cos_yaw = np.cos(yaw_per_frame)
                    sin_yaw = np.sin(yaw_per_frame)
                    
                    # 前进
                    forward_speed = 0.00

                    pose = np.eye(4, dtype=np.float32)
                    
                    pose[0, 0] = cos_yaw
                    pose[0, 2] = sin_yaw
                    pose[2, 0] = -sin_yaw
                    pose[2, 2] = cos_yaw
                    pose[2, 3] = -forward_speed
                
                else:
                    # 超出条件帧与目标帧的部分，保持静止
                    pose = np.eye(4, dtype=np.float32)
                    
                relative_pose = pose[:3, :]
                relative_poses.append(torch.as_tensor(relative_pose))
                
            else:
                raise ValueError(f"未定义的相机运动方向: {direction}")
            
        pose_embedding = torch.stack(relative_poses, dim=0)
        pose_embedding = rearrange(pose_embedding, 'b c d -> b (c d)')
        
        # 创建对应长度的mask序列
        mask = torch.zeros(max_needed_frames, 1, dtype=torch.float32)
        condition_end = min(start_frame + initial_condition_frames + 1, max_needed_frames)
        mask[start_frame:condition_end] = 1.0
        
        camera_embedding = torch.cat([pose_embedding, mask], dim=1)
        print(f"🔧 Sekai合成camera embedding shape: {camera_embedding.shape}")
        return camera_embedding.to(torch.bfloat16)


def generate_openx_camera_embeddings_sliding(
    encoded_data, start_frame, initial_condition_frames, new_frames, use_real_poses):
    """为OpenX数据集生成camera embeddings - 滑动窗口版本"""
    time_compression_ratio = 4
    
    # 计算FramePack实际需要的camera帧数
    framepack_needed_frames = 1 + 16 + 2 + 1 + new_frames
    
    if use_real_poses and encoded_data is not None and 'cam_emb' in encoded_data and 'extrinsic' in encoded_data['cam_emb']:
        print("🔧 使用OpenX真实camera数据")
        cam_extrinsic = encoded_data['cam_emb']['extrinsic']
        
        # 确保生成足够长的camera序列
        max_needed_frames = max(
            start_frame + initial_condition_frames + new_frames,
            framepack_needed_frames,
            30
        )
        
        print(f"🔧 计算OpenX camera序列长度:")
        print(f"  - 基础需求: {start_frame + initial_condition_frames + new_frames}")
        print(f"  - FramePack需求: {framepack_needed_frames}")
        print(f"  - 最终生成: {max_needed_frames}")
        
        relative_poses = []
        for i in range(max_needed_frames):
            # OpenX使用4倍间隔，类似sekai但处理更短的序列
            frame_idx = i * time_compression_ratio
            next_frame_idx = frame_idx + time_compression_ratio
            
            if next_frame_idx < len(cam_extrinsic):
                cam_prev = cam_extrinsic[frame_idx]
                cam_next = cam_extrinsic[next_frame_idx]
                relative_pose = compute_relative_pose(cam_prev, cam_next)
                relative_poses.append(torch.as_tensor(relative_pose[:3, :]))
            else:
                # 超出范围，使用零运动
                print(f"⚠️ 帧{frame_idx}超出OpenX camera数据范围，使用零运动")
                relative_poses.append(torch.zeros(3, 4))
        
        pose_embedding = torch.stack(relative_poses, dim=0)
        pose_embedding = rearrange(pose_embedding, 'b c d -> b (c d)')
        
        # 创建对应长度的mask序列
        mask = torch.zeros(max_needed_frames, 1, dtype=torch.float32)
        # 从start_frame到start_frame + initial_condition_frames标记为condition
        condition_end = min(start_frame + initial_condition_frames, max_needed_frames)
        mask[start_frame:condition_end] = 1.0
        
        camera_embedding = torch.cat([pose_embedding, mask], dim=1)
        print(f"🔧 OpenX真实camera embedding shape: {camera_embedding.shape}")
        return camera_embedding.to(torch.bfloat16)
        
    else:
        print("🔧 使用OpenX合成camera数据")
        
        max_needed_frames = max(
            start_frame + initial_condition_frames + new_frames,
            framepack_needed_frames,
            30
        )
        
        print(f"🔧 生成OpenX合成camera帧数: {max_needed_frames}")
        relative_poses = []
        for i in range(max_needed_frames):
            # OpenX机器人操作运动模式 - 较小的运动幅度
            # 模拟机器人手臂的精细操作运动
            roll_per_frame = 0.02   # 轻微翻滚
            pitch_per_frame = 0.01  # 轻微俯仰
            yaw_per_frame = 0.015   # 轻微偏航
            forward_speed = 0.003   # 较慢的前进速度
            
            pose = np.eye(4, dtype=np.float32)
            
            # 复合旋转 - 模拟机器人手臂的复杂运动
            # 绕X轴旋转（roll）
            cos_roll = np.cos(roll_per_frame)
            sin_roll = np.sin(roll_per_frame)
            # 绕Y轴旋转（pitch）
            cos_pitch = np.cos(pitch_per_frame)
            sin_pitch = np.sin(pitch_per_frame)
            # 绕Z轴旋转（yaw）
            cos_yaw = np.cos(yaw_per_frame)
            sin_yaw = np.sin(yaw_per_frame)
            
            # 简化的复合旋转矩阵（ZYX顺序）
            pose[0, 0] = cos_yaw * cos_pitch
            pose[0, 1] = cos_yaw * sin_pitch * sin_roll - sin_yaw * cos_roll
            pose[0, 2] = cos_yaw * sin_pitch * cos_roll + sin_yaw * sin_roll
            pose[1, 0] = sin_yaw * cos_pitch
            pose[1, 1] = sin_yaw * sin_pitch * sin_roll + cos_yaw * cos_roll
            pose[1, 2] = sin_yaw * sin_pitch * cos_roll - cos_yaw * sin_roll
            pose[2, 0] = -sin_pitch
            pose[2, 1] = cos_pitch * sin_roll
            pose[2, 2] = cos_pitch * cos_roll
            
            # 平移 - 模拟机器人操作的精细移动
            pose[0, 3] = forward_speed * 0.5   # X方向轻微移动
            pose[1, 3] = forward_speed * 0.3   # Y方向轻微移动
            pose[2, 3] = -forward_speed        # Z方向（深度）主要移动
            
            relative_pose = pose[:3, :]
            relative_poses.append(torch.as_tensor(relative_pose))
        
        pose_embedding = torch.stack(relative_poses, dim=0)
        pose_embedding = rearrange(pose_embedding, 'b c d -> b (c d)')
        
        # 创建对应长度的mask序列
        mask = torch.zeros(max_needed_frames, 1, dtype=torch.float32)
        condition_end = min(start_frame + initial_condition_frames, max_needed_frames)
        mask[start_frame:condition_end] = 1.0
        
        camera_embedding = torch.cat([pose_embedding, mask], dim=1)
        print(f"🔧 OpenX合成camera embedding shape: {camera_embedding.shape}")
        return camera_embedding.to(torch.bfloat16)


def generate_nuscenes_camera_embeddings_sliding(
    scene_info, start_frame, initial_condition_frames, new_frames):
    """为NuScenes数据集生成camera embeddings - 滑动窗口版本 - 修正版，与train_moe.py保持一致"""
    time_compression_ratio = 4
    
    # 计算FramePack实际需要的camera帧数
    framepack_needed_frames = 1 + 16 + 2 + 1 + new_frames
    
    if scene_info is not None and 'keyframe_poses' in scene_info:
        print("🔧 使用NuScenes真实pose数据")
        keyframe_poses = scene_info['keyframe_poses']
        
        if len(keyframe_poses) == 0:
            print("⚠️ NuScenes keyframe_poses为空，使用零pose")
            max_needed_frames = max(framepack_needed_frames, 30)
            
            pose_sequence = torch.zeros(max_needed_frames, 7, dtype=torch.float32)
            
            mask = torch.zeros(max_needed_frames, 1, dtype=torch.float32)
            condition_end = min(start_frame + initial_condition_frames, max_needed_frames)
            mask[start_frame:condition_end] = 1.0
            
            camera_embedding = torch.cat([pose_sequence, mask], dim=1)  # [max_needed_frames, 8]
            print(f"🔧 NuScenes零pose embedding shape: {camera_embedding.shape}")
            return camera_embedding.to(torch.bfloat16)
        
        # 使用第一个pose作为参考
        reference_pose = keyframe_poses[0]
        
        max_needed_frames = max(framepack_needed_frames, 30)
        
        pose_vecs = []
        for i in range(max_needed_frames):
            if i < len(keyframe_poses):
                current_pose = keyframe_poses[i]
                
                # 计算相对位移
                translation = torch.tensor(
                    np.array(current_pose['translation']) - np.array(reference_pose['translation']),
                    dtype=torch.float32
                )
                
                # 计算相对旋转（简化版本）
                rotation = torch.tensor(current_pose['rotation'], dtype=torch.float32)
                
                pose_vec = torch.cat([translation, rotation], dim=0)  # [7D]
            else:
                # 超出范围，使用零pose
                pose_vec = torch.cat([
                    torch.zeros(3, dtype=torch.float32),
                    torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)
                ], dim=0)  # [7D]
            
            pose_vecs.append(pose_vec)
        
        pose_sequence = torch.stack(pose_vecs, dim=0)  # [max_needed_frames, 7]
        
        # 创建mask
        mask = torch.zeros(max_needed_frames, 1, dtype=torch.float32)
        condition_end = min(start_frame + initial_condition_frames, max_needed_frames)
        mask[start_frame:condition_end] = 1.0
        
        camera_embedding = torch.cat([pose_sequence, mask], dim=1)  # [max_needed_frames, 8]
        print(f"🔧 NuScenes真实pose embedding shape: {camera_embedding.shape}")
        return camera_embedding.to(torch.bfloat16)
    
    else:
        print("🔧 使用NuScenes合成pose数据")
        max_needed_frames = max(framepack_needed_frames, 30)
        
        # 创建合成运动序列
        pose_vecs = []
        for i in range(max_needed_frames):
            # 左转运动模式 - 类似城市驾驶中的左转弯
            angle = i * 0.04  # 每帧转动0.08弧度（稍微慢一点的转弯）
            radius = 15.0     # 较大的转弯半径，更符合汽车转弯
            
            # 计算圆弧轨迹上的位置
            x = radius * np.sin(angle)
            y = 0.0  # 保持水平面运动
            z = radius * (1 - np.cos(angle))
            
            translation = torch.tensor([x, y, z], dtype=torch.float32)
            
            # 车辆朝向 - 始终沿着轨迹切线方向
            yaw = angle + np.pi/2  # 相对于初始前进方向的偏航角
            # 四元数表示绕Y轴的旋转
            rotation = torch.tensor([
                np.cos(yaw/2),  # w (实部)
                0.0,            # x
                0.0,            # y  
                np.sin(yaw/2)   # z (虚部，绕Y轴)
            ], dtype=torch.float32)
            
            pose_vec = torch.cat([translation, rotation], dim=0)  # [7D: tx,ty,tz,qw,qx,qy,qz]
            pose_vecs.append(pose_vec)
        
        pose_sequence = torch.stack(pose_vecs, dim=0)
        
        # 创建mask
        mask = torch.zeros(max_needed_frames, 1, dtype=torch.float32)
        condition_end = min(start_frame + initial_condition_frames, max_needed_frames)
        mask[start_frame:condition_end] = 1.0
        
        camera_embedding = torch.cat([pose_sequence, mask], dim=1)  # [max_needed_frames, 8]
        print(f"🔧 NuScenes合成左转pose embedding shape: {camera_embedding.shape}")
        return camera_embedding.to(torch.bfloat16)
    
def prepare_framepack_sliding_window_with_camera_moe(
    history_latents, 
    target_frames_to_generate, 
    camera_embedding_full, 
    start_frame, 
    modality_type, 
    max_history_frames=49):
    """FramePack滑动窗口机制 - MoE版本"""
    # history_latents: [C, T, H, W] 当前的历史latents
    C, T, H, W = history_latents.shape
    
    # 固定索引结构（这决定了需要的camera帧数）
    # 1帧起始 + 16帧4x + 2帧2x + 1帧1x + target_frames_to_generate
    total_indices_length = 1 + 16 + 2 + 1 + target_frames_to_generate
    indices = torch.arange(0, total_indices_length)
    split_sizes = [1, 16, 2, 1, target_frames_to_generate]
    clean_latent_indices_start, clean_latent_4x_indices, clean_latent_2x_indices, clean_latent_1x_indices, latent_indices = \
        indices.split(split_sizes, dim=0)
    clean_latent_indices = torch.cat([clean_latent_indices_start, clean_latent_1x_indices], dim=0)
    
    # 检查camera长度是否足够
    if camera_embedding_full.shape[0] < total_indices_length:
        print(f"⚠️ camera_embedding长度不足，进行零补齐: 当前长度 {camera_embedding_full.shape[0]}, 需要长度 {total_indices_length}")
        shortage = total_indices_length - camera_embedding_full.shape[0]
        padding = torch.zeros(shortage, camera_embedding_full.shape[1], 
                            dtype=camera_embedding_full.dtype, device=camera_embedding_full.device)
        camera_embedding_full = torch.cat([camera_embedding_full, padding], dim=0)
    
    # 从完整camera序列中选取对应部分
    combined_camera = torch.zeros(
        total_indices_length, 
        camera_embedding_full.shape[1],
        dtype=camera_embedding_full.dtype,
        device=camera_embedding_full.device)
    
    # 历史条件帧的相机位姿
    history_slice = camera_embedding_full[max(T - 19, 0):T, :].clone()
    combined_camera[19 - history_slice.shape[0]:19, :] = history_slice
    
    # 目标帧的相机位姿
    target_slice = camera_embedding_full[T:T + target_frames_to_generate, :].clone()
    combined_camera[19:19 + target_slice.shape[0], :] = target_slice
    
    # 根据当前history length重新设置mask
    combined_camera[:, -1] = 0.0  # 先全部设为target (0)
    
    # 设置condition mask：前19帧根据实际历史长度决定
    if T > 0:
        available_frames = min(T, 19)
        start_pos = 19 - available_frames
        combined_camera[start_pos:19, -1] = 1.0  # 将有效的clean latents对应的camera标记为condition
    
    print(f"🔧 MoE Camera mask更新:")
    print(f"  - 历史帧数: {T}")
    print(f"  - 有效condition帧数: {available_frames if T > 0 else 0}")
    print(f"  - 模态类型: {modality_type}")
    
    # 处理latents
    clean_latents_combined = torch.zeros(C, 19, H, W, dtype=history_latents.dtype, device=history_latents.device)
    
    if T > 0:
        available_frames = min(T, 19)
        start_pos = 19 - available_frames
        clean_latents_combined[:, start_pos:, :, :] = history_latents[:, -available_frames:, :, :]
    
    clean_latents_4x = clean_latents_combined[:, 0:16, :, :]
    clean_latents_2x = clean_latents_combined[:, 16:18, :, :]
    clean_latents_1x = clean_latents_combined[:, 18:19, :, :]
    
    if T > 0:
        start_latent = history_latents[:, 0:1, :, :]
    else:
        start_latent = torch.zeros(C, 1, H, W, dtype=history_latents.dtype, device=history_latents.device)
    
    clean_latents = torch.cat([start_latent, clean_latents_1x], dim=1)
    
    return {
        'latent_indices': latent_indices,
        'clean_latents': clean_latents,
        'clean_latents_2x': clean_latents_2x,
        'clean_latents_4x': clean_latents_4x,
        'clean_latent_indices': clean_latent_indices,
        'clean_latent_2x_indices': clean_latent_2x_indices,
        'clean_latent_4x_indices': clean_latent_4x_indices,
        'camera_embedding': combined_camera,
        'modality_type': modality_type,  # 新增模态类型信息
        'current_length': T,
        'next_length': T + target_frames_to_generate
    }

def overlay_controls(frame_img, pose_vec, icons):
    """
    根据相机位姿在帧上叠加控制图标(WASD 和箭头)
    pose_vec: 12 个元素（展平的 3x4 矩阵）+ mask
    """
    if pose_vec is None or np.all(pose_vec[:12] == 0):
        return frame_img
        
    # 提取平移向量（基于展平的 3x4 矩阵的索引）
    # [r00, r01, r02, tx, r10, r11, r12, ty, r20, r21, r22, tz]
    tx = pose_vec[3]
    # ty = pose_vec[7]
    tz = pose_vec[11]
    
    # 提取旋转（偏航和俯仰）
    # 偏航：绕 Y 轴。sin(偏航) = r02, cos(偏航) = r00
    r00 = pose_vec[0]
    r02 = pose_vec[2]
    yaw = np.arctan2(r02, r00)
    
    # 俯仰：绕 X 轴。sin(俯仰) = -r12, cos(俯仰) = r22
    r12 = pose_vec[6]
    r22 = pose_vec[10]
    pitch = np.arctan2(-r12, r22)
    
    # 按键激活的阈值
    TRANS_THRESH = 0.01
    ROT_THRESH = 0.005
    
    # 确定按键状态
    # 平移（WASD）
    # 假设 -Z 为前进，+X 为右
    is_forward = tz < -TRANS_THRESH
    is_backward = tz > TRANS_THRESH
    is_left = tx < -TRANS_THRESH
    is_right = tx > TRANS_THRESH
    
    # 旋转（箭头）
    # 偏航：+ 为左，- 为右
    is_turn_left = yaw > ROT_THRESH
    is_turn_right = yaw < -ROT_THRESH
    
    # 俯仰：+ 为下，- 为上
    is_turn_up = pitch < -ROT_THRESH
    is_turn_down = pitch > ROT_THRESH
    
    W, H = frame_img.size
    spacing = 60
    
    def paste_icon(name_active, name_inactive, is_active, x, y):
        name = name_active if is_active else name_inactive
        if name in icons:
            icon = icons[name]
        # 使用 alpha 通道粘贴
            frame_img.paste(icon, (int(x), int(y)), icon)
    
    # 叠加 WASD（左下角）
    base_x_right = 100
    base_y = H - 100
    
    # W
    paste_icon('move_forward.png', 'not_move_forward.png', is_forward, base_x_right, base_y - spacing)
    # A
    paste_icon('move_left.png', 'not_move_left.png', is_left, base_x_right - spacing, base_y)
    # S
    paste_icon('move_backward.png', 'not_move_backward.png', is_backward, base_x_right, base_y)
    # D
    paste_icon('move_right.png', 'not_move_right.png', is_right, base_x_right + spacing, base_y)
    
    # 叠加 ↑↓←→（右下角）
    base_x_left = W - 150
    
    # ↑
    paste_icon('turn_up.png', 'not_turn_up.png', is_turn_up, base_x_left, base_y - spacing)
    # ←
    paste_icon('turn_left.png', 'not_turn_left.png', is_turn_left, base_x_left - spacing, base_y)
    # ↓
    paste_icon('turn_down.png', 'not_turn_down.png', is_turn_down, base_x_left, base_y)
    # →
    paste_icon('turn_right.png', 'not_turn_right.png', is_turn_right, base_x_left + spacing, base_y)
    
    return frame_img


def inference_moe_framepack_sliding_window(
    condition_pth_path,
    dit_path,
    output_path="../examples/output_videos/output_moe_framepack_sliding.mp4",
    start_frame=0,
    initial_condition_frames=8,
    frames_per_generation=4,
    total_frames_to_generate=32,
    max_history_frames=49,
    device="cuda",
    prompt="A video of a scene shot using a pedestrian's front camera while walking",
    modality_type="sekai",  # "sekai" 或 "nuscenes"
    use_real_poses=True,
    scene_info_path=None,  # 对于NuScenes数据集
    # CFG参数
    use_camera_cfg=True,
    camera_guidance_scale=2.0,
    text_guidance_scale=1.0,
    # MoE参数
    moe_num_experts=4,
    moe_top_k=2,
    moe_hidden_dim=None,
    direction="left",
    use_gt_prompt=True,
    add_icons=False
):
    """
    MoE FramePack滑动窗口视频生成 - 支持多模态
    """
    # 创建输出目录
    dir_path = os.path.dirname(output_path)
    os.makedirs(dir_path, exist_ok=True)
    
    print(f"🔧 MoE FramePack滑动窗口生成开始...")
    print(f"模态类型: {modality_type}")
    print(f"Camera CFG: {use_camera_cfg}, Camera guidance scale: {camera_guidance_scale}")
    print(f"Text guidance scale: {text_guidance_scale}")
    print(f"MoE配置: experts={moe_num_experts}, top_k={moe_top_k}")
    
    # 1. 模型初始化
    replace_dit_model_in_manager()
    
    model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
    model_manager.load_models([
        "/mnt/data/louis_crq/models/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors",
        "/mnt/data/louis_crq/models/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
        "/mnt/data/louis_crq/models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
    ])
    pipe = WanVideoReCamMasterPipeline.from_model_manager(model_manager, device="cuda")

    # 2. 添加传统camera编码器（兼容性）
    dim = pipe.dit.blocks[0].self_attn.q.weight.shape[0]
    for block in pipe.dit.blocks:
        block.cam_encoder = nn.Linear(13, dim)
        block.projector = nn.Linear(dim, dim)
        block.cam_encoder.weight.data.zero_()
        block.cam_encoder.bias.data.zero_()
        block.projector.weight = nn.Parameter(torch.eye(dim))
        block.projector.bias = nn.Parameter(torch.zeros(dim))
    
    # 3. 添加FramePack组件
    add_framepack_components(pipe.dit)
    
    # 4. 添加MoE组件
    moe_config = {
        "num_experts": moe_num_experts,
        "top_k": moe_top_k,
        "hidden_dim": moe_hidden_dim or dim * 2,
        "sekai_input_dim": 13,    # Sekai: 12维pose + 1维mask
        "nuscenes_input_dim": 8,   # NuScenes: 7维pose + 1维mask
        "openx_input_dim": 13       # OpenX: 12维pose + 1维mask (类似sekai)
    }
    add_moe_components(pipe.dit, moe_config)
    
    # 5. 加载训练好的权重
    dit_state_dict = torch.load(dit_path, map_location="cpu")
    pipe.dit.load_state_dict(dit_state_dict, strict=False)  # 使用strict=False以兼容新增的MoE组件
    pipe = pipe.to(device)
    model_dtype = next(pipe.dit.parameters()).dtype
    
    if hasattr(pipe.dit, 'clean_x_embedder'):
        pipe.dit.clean_x_embedder = pipe.dit.clean_x_embedder.to(dtype=model_dtype)
    
    # 设置去噪步数
    pipe.scheduler.set_timesteps(50)
    
    # 6. 加载初始条件
    print("Loading initial condition frames...")
    initial_latents, encoded_data = load_encoded_video_from_pth(
        condition_pth_path, 
        start_frame=start_frame,
        num_frames=initial_condition_frames
    )
    
    # 空间裁剪
    target_height, target_width = 60, 104
    C, T, H, W = initial_latents.shape
    
    if H > target_height or W > target_width:
        h_start = (H - target_height) // 2
        w_start = (W - target_width) // 2
        initial_latents = initial_latents[:, :, h_start:h_start+target_height, w_start:w_start+target_width]
        H, W = target_height, target_width
    
    history_latents = initial_latents.to(device, dtype=model_dtype)

    print(f"初始history_latents shape: {history_latents.shape}")
    
    # 7. 编码prompt - 支持CFG
    if use_gt_prompt and 'prompt_emb' in encoded_data:
        print("✅ 使用预编码的GT prompt embedding")
        prompt_emb_pos = encoded_data['prompt_emb']
        # 将prompt_emb移到正确的设备和数据类型
        if 'context' in prompt_emb_pos:
            prompt_emb_pos['context'] = prompt_emb_pos['context'].to(device, dtype=model_dtype)
        if 'context_mask' in prompt_emb_pos:
            prompt_emb_pos['context_mask'] = prompt_emb_pos['context_mask'].to(device, dtype=model_dtype)
        
        # 如果使用Text CFG，生成负向prompt
        if text_guidance_scale > 1.0:
            prompt_emb_neg = pipe.encode_prompt("")
            print(f"使用Text CFG with GT prompt，guidance scale: {text_guidance_scale}")
        else:
            prompt_emb_neg = None
            print("不使用Text CFG")
        
        # 🔧 打印GT prompt文本（如果有）
        if 'prompt' in encoded_data['prompt_emb']:
            gt_prompt_text = encoded_data['prompt_emb']['prompt']
            print(f"📝 GT Prompt文本: {gt_prompt_text}")
    else:
        # 使用传入的prompt参数重新编码
        print(f"🔄 重新编码prompt: {prompt}")
        if text_guidance_scale > 1.0:
            prompt_emb_pos = pipe.encode_prompt(prompt)
            prompt_emb_neg = pipe.encode_prompt("")
            print(f"使用Text CFG，guidance scale: {text_guidance_scale}")
        else:
            prompt_emb_pos = pipe.encode_prompt(prompt)
            prompt_emb_neg = None
            print("不使用Text CFG")
    
    # 8. 加载场景信息（对于NuScenes）
    scene_info = None
    if modality_type == "nuscenes" and scene_info_path and os.path.exists(scene_info_path):
        with open(scene_info_path, 'r') as f:
            scene_info = json.load(f)
        print(f"加载NuScenes场景信息: {scene_info_path}")
    
    # 9. 预生成完整的camera embedding序列
    if modality_type == "sekai":
        camera_embedding_full = generate_sekai_camera_embeddings_sliding(
            encoded_data.get('cam_emb', None),
            start_frame,
            initial_condition_frames,
            total_frames_to_generate,
            0,
            use_real_poses=use_real_poses,
            direction=direction
        ).to(device, dtype=model_dtype)
    elif modality_type == "nuscenes":
        camera_embedding_full = generate_nuscenes_camera_embeddings_sliding(
            scene_info,
            start_frame,
            initial_condition_frames,
            total_frames_to_generate
        ).to(device, dtype=model_dtype)
    elif modality_type == "openx":
        camera_embedding_full = generate_openx_camera_embeddings_sliding(
            encoded_data,
            start_frame,
            initial_condition_frames,
            total_frames_to_generate,
            use_real_poses=use_real_poses
        ).to(device, dtype=model_dtype)        
    else:
        raise ValueError(f"不支持的模态类型: {modality_type}")
    
    print(f"完整camera序列shape: {camera_embedding_full.shape}")
    
    # 10. 为Camera CFG创建无条件的camera embedding
    if use_camera_cfg:
        camera_embedding_uncond = torch.zeros_like(camera_embedding_full)
        print(f"创建无条件camera embedding用于CFG")
    
    # 11. 滑动窗口生成循环
    total_generated = 0
    all_generated_frames = []
    
    while total_generated < total_frames_to_generate:
        current_generation = min(frames_per_generation, total_frames_to_generate - total_generated)
        print(f"\n🔧 生成步骤 {total_generated // frames_per_generation + 1}")
        print(f"当前历史长度: {history_latents.shape[1]}, 本次生成: {current_generation}")
        
        # FramePack数据准备 - MoE版本
        framepack_data = prepare_framepack_sliding_window_with_camera_moe(
            history_latents,
            current_generation,
            camera_embedding_full,
            start_frame,
            modality_type,
            max_history_frames
        )
        
        # 准备输入
        clean_latents = framepack_data['clean_latents'].unsqueeze(0)
        clean_latents_2x = framepack_data['clean_latents_2x'].unsqueeze(0)
        clean_latents_4x = framepack_data['clean_latents_4x'].unsqueeze(0)
        camera_embedding = framepack_data['camera_embedding'].unsqueeze(0)
        
        # 准备modality_inputs
        modality_inputs = {modality_type: camera_embedding}
        
        # 为CFG准备无条件camera embedding
        if use_camera_cfg:
            camera_embedding_uncond_batch = camera_embedding_uncond[:camera_embedding.shape[1], :].unsqueeze(0)
            modality_inputs_uncond = {modality_type: camera_embedding_uncond_batch}
        
        # 索引处理
        latent_indices = framepack_data['latent_indices'].unsqueeze(0).cpu()
        clean_latent_indices = framepack_data['clean_latent_indices'].unsqueeze(0).cpu()
        clean_latent_2x_indices = framepack_data['clean_latent_2x_indices'].unsqueeze(0).cpu()
        clean_latent_4x_indices = framepack_data['clean_latent_4x_indices'].unsqueeze(0).cpu()
        
        # 初始化要生成的latents
        new_latents = torch.randn(
            1, C, current_generation, H, W,
            device=device, dtype=model_dtype
        )
        
        extra_input = pipe.prepare_extra_input(new_latents)
        
        print(f"Camera embedding shape: {camera_embedding.shape}")
        print(f"Camera mask分布 - condition: {torch.sum(camera_embedding[0, :, -1] == 1.0).item()}, target: {torch.sum(camera_embedding[0, :, -1] == 0.0).item()}")
        
        # 去噪循环 - 支持CFG
        timesteps = pipe.scheduler.timesteps
        
        for i, timestep in enumerate(timesteps):
            if i % 10 == 0:
                print(f"  去噪步骤 {i+1}/{len(timesteps)}")
            
            timestep_tensor = timestep.unsqueeze(0).to(device, dtype=model_dtype)
            
            with torch.no_grad():
                # CFG推理
                if use_camera_cfg and camera_guidance_scale > 1.0:
                    # 条件预测（有camera）
                    noise_pred_cond, moe_loess = pipe.dit(
                        new_latents,
                        timestep=timestep_tensor,
                        cam_emb=camera_embedding,
                        modality_inputs=modality_inputs,  # MoE模态输入
                        latent_indices=latent_indices,
                        clean_latents=clean_latents,
                        clean_latent_indices=clean_latent_indices,
                        clean_latents_2x=clean_latents_2x,
                        clean_latent_2x_indices=clean_latent_2x_indices,
                        clean_latents_4x=clean_latents_4x,
                        clean_latent_4x_indices=clean_latent_4x_indices,
                        **prompt_emb_pos,
                        **extra_input
                    )
                    
                    # 无条件预测（无camera）
                    noise_pred_uncond, moe_loess = pipe.dit(
                        new_latents,
                        timestep=timestep_tensor,
                        cam_emb=camera_embedding_uncond_batch,
                        modality_inputs=modality_inputs_uncond,  # MoE无条件模态输入
                        latent_indices=latent_indices,
                        clean_latents=clean_latents,
                        clean_latent_indices=clean_latent_indices,
                        clean_latents_2x=clean_latents_2x,
                        clean_latent_2x_indices=clean_latent_2x_indices,
                        clean_latents_4x=clean_latents_4x,
                        clean_latent_4x_indices=clean_latent_4x_indices,
                        **(prompt_emb_neg if prompt_emb_neg else prompt_emb_pos),
                        **extra_input
                    )
                    
                    # Camera CFG
                    noise_pred = noise_pred_uncond + camera_guidance_scale * (noise_pred_cond - noise_pred_uncond)
                    
                    # 如果同时使用Text CFG
                    if text_guidance_scale > 1.0 and prompt_emb_neg:
                        noise_pred_text_uncond, moe_loess = pipe.dit(
                            new_latents,
                            timestep=timestep_tensor,
                            cam_emb=camera_embedding,
                            modality_inputs=modality_inputs,
                            latent_indices=latent_indices,
                            clean_latents=clean_latents,
                            clean_latent_indices=clean_latent_indices,
                            clean_latents_2x=clean_latents_2x,
                            clean_latent_2x_indices=clean_latent_2x_indices,
                            clean_latents_4x=clean_latents_4x,
                            clean_latent_4x_indices=clean_latent_4x_indices,
                            **prompt_emb_neg,
                            **extra_input
                        )
                        
                        # 应用Text CFG到已经应用Camera CFG的结果
                        noise_pred = noise_pred_text_uncond + text_guidance_scale * (noise_pred - noise_pred_text_uncond)
                
                elif text_guidance_scale > 1.0 and prompt_emb_neg:
                    # 只使用Text CFG
                    noise_pred_cond, moe_loess = pipe.dit(
                        new_latents,
                        timestep=timestep_tensor,
                        cam_emb=camera_embedding,
                        modality_inputs=modality_inputs,
                        latent_indices=latent_indices,
                        clean_latents=clean_latents,
                        clean_latent_indices=clean_latent_indices,
                        clean_latents_2x=clean_latents_2x,
                        clean_latent_2x_indices=clean_latent_2x_indices,
                        clean_latents_4x=clean_latents_4x,
                        clean_latent_4x_indices=clean_latent_4x_indices,
                        **prompt_emb_pos,
                        **extra_input
                    )
                    
                    noise_pred_uncond, moe_loess= pipe.dit(
                        new_latents,
                        timestep=timestep_tensor,
                        cam_emb=camera_embedding,
                        modality_inputs=modality_inputs,
                        latent_indices=latent_indices,
                        clean_latents=clean_latents,
                        clean_latent_indices=clean_latent_indices,
                        clean_latents_2x=clean_latents_2x,
                        clean_latent_2x_indices=clean_latent_2x_indices,
                        clean_latents_4x=clean_latents_4x,
                        clean_latent_4x_indices=clean_latent_4x_indices,
                        **prompt_emb_neg,
                        **extra_input
                    )
                    
                    noise_pred = noise_pred_uncond + text_guidance_scale * (noise_pred_cond - noise_pred_uncond)
                
                else:
                    # 标准推理（无CFG）
                    noise_pred, moe_loess = pipe.dit(
                        new_latents,
                        timestep=timestep_tensor,
                        cam_emb=camera_embedding,
                        modality_inputs=modality_inputs,  # MoE模态输入
                        latent_indices=latent_indices,
                        clean_latents=clean_latents,
                        clean_latent_indices=clean_latent_indices,
                        clean_latents_2x=clean_latents_2x,
                        clean_latent_2x_indices=clean_latent_2x_indices,
                        clean_latents_4x=clean_latents_4x,
                        clean_latent_4x_indices=clean_latent_4x_indices,
                        **prompt_emb_pos,
                        **extra_input
                    )
            
            new_latents = pipe.scheduler.step(noise_pred, timestep, new_latents)
        
        # 更新历史
        new_latents_squeezed = new_latents.squeeze(0)
        history_latents = torch.cat([history_latents, new_latents_squeezed], dim=1)
        
        # 维护滑动窗口
        if history_latents.shape[1] > max_history_frames:
            first_frame = history_latents[:, 0:1, :, :]
            recent_frames = history_latents[:, -(max_history_frames-1):, :, :]
            history_latents = torch.cat([first_frame, recent_frames], dim=1)
            print(f"历史窗口已满，保留第一帧+最新{max_history_frames-1}帧")
        
        print(f"更新后history_latents shape: {history_latents.shape}")
        
        all_generated_frames.append(new_latents_squeezed)
        total_generated += current_generation
        
        print(f"✅ 已生成 {total_generated}/{total_frames_to_generate} 帧")
    
    # 12. 解码和保存
    print("\n🔧 解码生成的视频...")
    
    all_generated = torch.cat(all_generated_frames, dim=1)
    final_video = torch.cat([initial_latents.to(all_generated.device), all_generated], dim=1).unsqueeze(0)
    
    print(f"最终视频shape: {final_video.shape}")
    
    decoded_video = pipe.decode_video(final_video, tiled=True, tile_size=(34, 34), tile_stride=(18, 16))
    
    print(f"Saving video to {output_path} ...")
    
    video_np = decoded_video[0].to(torch.float32).permute(1, 2, 3, 0).cpu().numpy()
    video_np = (video_np * 0.5 + 0.5).clip(0, 1)
    video_np = (video_np * 255).astype(np.uint8)

    icons = {}
    video_camera_poses = None
    if add_icons:
        # 加载用于叠加的图标资源
        icons_dir = os.path.join(ROOT_DIR, 'icons')
        icon_names = ['move_forward.png', 'not_move_forward.png', 
                      'move_backward.png', 'not_move_backward.png',
                      'move_left.png', 'not_move_left.png',
                      'move_right.png', 'not_move_right.png',
                      'turn_up.png', 'not_turn_up.png',
                      'turn_down.png', 'not_turn_down.png',
                      'turn_left.png', 'not_turn_left.png',
                      'turn_right.png', 'not_turn_right.png']
        for name in icon_names:
            path = os.path.join(icons_dir, name)
            if os.path.exists(path):
                try:
                    icon = Image.open(path).convert("RGBA")
                    # 调整图标尺寸
                    icon = icon.resize((50, 50), Image.Resampling.LANCZOS)
                    icons[name] = icon
                except Exception as e:
                    print(f"Error loading icon {name}: {e}")
            else:
                print(f"Warning: Icon {name} not found at {path}")

        # 获取与视频帧对应的相机姿态
        time_compression_ratio = 4
        camera_poses = camera_embedding_full.detach().float().cpu().numpy()
        video_camera_poses = [x for x in camera_poses for _ in range(time_compression_ratio)]

    with imageio.get_writer(output_path, fps=20) as writer:
        for i, frame in enumerate(video_np):
            # Convert to PIL for overlay
            img = Image.fromarray(frame)
            
            if add_icons and video_camera_poses is not None and icons:
                # Video frame i corresponds to camera_embedding_full[start_frame + i]
                pose_idx = start_frame + i
                if pose_idx < len(video_camera_poses):
                    pose_vec = video_camera_poses[pose_idx]
                    img = overlay_controls(img, pose_vec, icons)
            
            writer.append_data(np.array(img))

    print(f"🔧 MoE FramePack滑动窗口生成完成! 保存到: {output_path}")
    print(f"总共生成了 {total_generated} 帧 (压缩后), 对应原始 {total_generated * 4} 帧")
    print(f"使用模态: {modality_type}")
    

def main():
    parser = argparse.ArgumentParser(description="MoE FramePack滑动窗口视频生成 - 支持多模态")
    
    # 基础参数
    parser.add_argument("--condition_pth", type=str, 
                        default="../examples/condition_pth/garden_1.pth")
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--initial_condition_frames", type=int, default=1)
    parser.add_argument("--frames_per_generation", type=int, default=8)
    parser.add_argument("--total_frames_to_generate", type=int, default=24)
    parser.add_argument("--max_history_frames", type=int, default=100)
    parser.add_argument("--use_real_poses", default=False)
    parser.add_argument("--dit_path", type=str, default=None, required=True, 
                        help="path to the pretrained DiT MoE model checkpoint")
    parser.add_argument("--output_path", type=str, 
                       default='./examples/output_videos/output_moe_framepack_sliding.mp4')
    parser.add_argument("--prompt", type=str, default=None, 
                        help="text prompt for video generation")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--add_icons", action="store_true", default=False,
                        help="在生成的视频上叠加控制图标")
    
    # 模态类型参数
    parser.add_argument("--modality_type", type=str, choices=["sekai", "nuscenes", "openx"], 
                       default="sekai", help="模态类型：sekai 或 nuscenes 或 openx")
    parser.add_argument("--scene_info_path", type=str, default=None,
                       help="NuScenes场景信息文件路径（仅用于nuscenes模态）")
    
    # CFG参数
    parser.add_argument("--use_camera_cfg", default=False,
                       help="使用Camera CFG")
    parser.add_argument("--camera_guidance_scale", type=float, default=2.0,
                       help="Camera guidance scale for CFG")
    parser.add_argument("--text_guidance_scale", type=float, default=1.0,
                       help="Text guidance scale for CFG")
    
    # MoE参数
    parser.add_argument("--moe_num_experts", type=int, default=3, help="专家数量")
    parser.add_argument("--moe_top_k", type=int, default=1, help="Top-K专家")
    parser.add_argument("--moe_hidden_dim", type=int, default=None, help="MoE隐藏层维度")
    parser.add_argument("--direction", type=str, default="left", help="生成视频的行进轨迹方向")
    parser.add_argument("--use_gt_prompt", action="store_true", default=False,
                       help="使用数据集中的ground truth prompt embedding")
    
    args = parser.parse_args()

    print(f"🔧 MoE FramePack CFG生成设置:")
    print(f"模态类型: {args.modality_type}")
    print(f"Camera CFG: {args.use_camera_cfg}")
    if args.use_camera_cfg:
        print(f"Camera guidance scale: {args.camera_guidance_scale}")
    print(f"使用GT Prompt: {args.use_gt_prompt}")
    print(f"Text guidance scale: {args.text_guidance_scale}")
    print(f"MoE配置: experts={args.moe_num_experts}, top_k={args.moe_top_k}")
    print(f"DiT{args.dit_path}")
    
    # 验证NuScenes参数
    if args.modality_type == "nuscenes" and not args.scene_info_path:
        print("⚠️ 使用NuScenes模态但未提供scene_info_path，将使用合成pose数据")
    
    inference_moe_framepack_sliding_window(
        condition_pth_path=args.condition_pth,
        dit_path=args.dit_path,
        output_path=args.output_path,
        start_frame=args.start_frame,
        initial_condition_frames=args.initial_condition_frames,
        frames_per_generation=args.frames_per_generation,
        total_frames_to_generate=args.total_frames_to_generate,
        max_history_frames=args.max_history_frames,
        device=args.device,
        prompt=args.prompt,
        modality_type=args.modality_type,
        use_real_poses=args.use_real_poses,
        scene_info_path=args.scene_info_path,
        # CFG参数
        use_camera_cfg=args.use_camera_cfg,
        camera_guidance_scale=args.camera_guidance_scale,
        text_guidance_scale=args.text_guidance_scale,
        # MoE参数
        moe_num_experts=args.moe_num_experts,
        moe_top_k=args.moe_top_k,
        moe_hidden_dim=args.moe_hidden_dim,
        direction=args.direction,
        use_gt_prompt=args.use_gt_prompt,
        add_icons=args.add_icons
    )


if __name__ == "__main__":
    main()