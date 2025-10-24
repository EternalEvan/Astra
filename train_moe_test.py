#融合nuscenes和sekai数据集的MoE训练
import torch
import torch.nn as nn
import lightning as pl
import wandb
import os
import copy
import json
import numpy as np
import random
import traceback
from diffsynth import WanVideoReCamMasterPipeline, ModelManager
from torchvision.transforms import v2
from einops import rearrange
from pose_classifier import PoseClassifier
import argparse
from scipy.spatial.transform import Rotation as R

def get_traj_position_change(cam_c2w, stride=1):
    positions = cam_c2w[:, :3, 3]
    
    traj_coord = []
    tarj_angle = []
    for i in range(0, len(positions) - 2 * stride):
        v1 = positions[i + stride] - positions[i]
        v2 = positions[i + 2 * stride] - positions[i + stride]

        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 < 1e-6 or norm2 < 1e-6:
            continue

        cos_angle = np.dot(v1, v2) / (norm1 * norm2)
        angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

        traj_coord.append(v1)
        tarj_angle.append(angle)
    
    return traj_coord, tarj_angle

def get_traj_rotation_change(cam_c2w, stride=1):
    rotations = cam_c2w[:, :3, :3]
    
    traj_rot_angle = []
    for i in range(0, len(rotations) - stride):
        z1 = rotations[i][:, 2]
        z2 = rotations[i + stride][:, 2]

        norm1 = np.linalg.norm(z1)
        norm2 = np.linalg.norm(z2)
        if norm1 < 1e-6 or norm2 < 1e-6:
            continue

        cos_angle = np.dot(z1, z2) / (norm1 * norm2)
        angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
        traj_rot_angle.append(angle)

    return traj_rot_angle

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

class MultiDatasetDynamicDataset(torch.utils.data.Dataset):
    """支持FramePack机制的多数据集动态历史长度数据集 - 融合nuscenes和sekai"""
    
    def __init__(self, dataset_configs, steps_per_epoch, 
                 min_condition_frames=10, max_condition_frames=40,
                 target_frames=10, height=900, width=1600):
        """
        Args:
            dataset_configs: 数据集配置列表，每个配置包含 {
                'name': 数据集名称,
                'paths': 数据集路径列表,
                'type': 数据集类型 ('sekai' 或 'nuscenes'),
                'weight': 采样权重
            }
        """
        self.dataset_configs = dataset_configs
        self.min_condition_frames = min_condition_frames
        self.max_condition_frames = max_condition_frames
        self.target_frames = target_frames
        self.height = height
        self.width = width
        self.steps_per_epoch = steps_per_epoch
        self.pose_classifier = PoseClassifier()
        
        # VAE时间压缩比例
        self.time_compression_ratio = 4
        
        # 🔧 扫描所有数据集，建立统一的场景索引
        self.scene_dirs = []
        self.dataset_info = {}  # 记录每个场景的数据集信息
        self.dataset_weights = []  # 每个场景的采样权重
        
        total_scenes = 0
        
        for config in self.dataset_configs:
            dataset_name = config['name']
            dataset_paths = config['paths'] if isinstance(config['paths'], list) else [config['paths']]
            dataset_type = config['type']
            dataset_weight = config.get('weight', 1.0)
            
            print(f"🔧 扫描数据集: {dataset_name} (类型: {dataset_type})")
            
            dataset_scenes = []
            for dataset_path in dataset_paths:
                print(f"  📁 检查路径: {dataset_path}")
                if os.path.exists(dataset_path):                    
                    if dataset_type == 'nuscenes':
                        # NuScenes使用 base_path/scenes 结构
                        scenes_path = os.path.join(dataset_path, "scenes")
                        print(f"  📂 扫描NuScenes scenes目录: {scenes_path}")
                        for item in os.listdir(scenes_path):
                            scene_dir = os.path.join(scenes_path, item)
                            if os.path.isdir(scene_dir):
                                self.scene_dirs.append(scene_dir)
                                dataset_scenes.append(scene_dir)
                                self.dataset_info[scene_dir] = {
                                    'name': dataset_name,
                                    'type': dataset_type,
                                    'weight': dataset_weight
                                }
                                self.dataset_weights.append(dataset_weight)

                    elif dataset_type == 'sekai':
                        # Sekai等其他数据集直接扫描根目录
                        for item in os.listdir(dataset_path):
                            scene_dir = os.path.join(dataset_path, item)
                            if os.path.isdir(scene_dir):
                                encoded_path = os.path.join(scene_dir, "encoded_video.pth")
                                if os.path.exists(encoded_path):
                                    self.scene_dirs.append(scene_dir)
                                    dataset_scenes.append(scene_dir)
                                    self.dataset_info[scene_dir] = {
                                        'name': dataset_name,
                                        'type': dataset_type,
                                        'weight': dataset_weight
                                    }
                                    self.dataset_weights.append(dataset_weight)

                    elif dataset_type in ['sekai', 'spatialvid', 'openx']:  # 🔧 添加openx类型
                        # Sekai、spatialvid、OpenX等数据集直接扫描根目录
                        for item in os.listdir(dataset_path):
                            scene_dir = os.path.join(dataset_path, item)
                            if os.path.isdir(scene_dir):
                                encoded_path = os.path.join(scene_dir, "encoded_video.pth")
                                if os.path.exists(encoded_path):
                                    self.scene_dirs.append(scene_dir)
                                    dataset_scenes.append(scene_dir)
                                    self.dataset_info[scene_dir] = {
                                        'name': dataset_name,
                                        'type': dataset_type,
                                        'weight': dataset_weight
                                    }
                                    self.dataset_weights.append(dataset_weight)
                else:
                    print(f"  ❌ 路径不存在: {dataset_path}")
                
                print(f"  ✅ 找到 {len(dataset_scenes)} 个场景")
                total_scenes += len(dataset_scenes)
                    
        # 统计各数据集场景数
        dataset_counts = {}
        for scene_dir in self.scene_dirs:
            dataset_name = self.dataset_info[scene_dir]['name']
            dataset_type = self.dataset_info[scene_dir]['type']
            key = f"{dataset_name} ({dataset_type})"
            dataset_counts[key] = dataset_counts.get(key, 0) + 1
        
        for dataset_key, count in dataset_counts.items():
            print(f"  - {dataset_key}: {count} 个场景")
        
        assert len(self.scene_dirs) > 0, "No encoded scenes found!"
        
        # 🔧 计算采样概率
        total_weight = sum(self.dataset_weights)
        self.sampling_probs = [w / total_weight for w in self.dataset_weights]

    def select_dynamic_segment_nuscenes(self, scene_info):
        """🔧 NuScenes专用的FramePack风格段落选择"""
        keyframe_indices = scene_info['keyframe_indices']  # 原始帧索引
        total_frames = scene_info['total_frames']  # 原始总帧数
        
        if len(keyframe_indices) < 2:
            return None
        
        # 计算压缩后的帧数
        compressed_total_frames = total_frames // self.time_compression_ratio
        compressed_keyframe_indices = [idx // self.time_compression_ratio for idx in keyframe_indices]
        
        min_condition_compressed = self.min_condition_frames // self.time_compression_ratio
        max_condition_compressed = self.max_condition_frames // self.time_compression_ratio
        target_frames_compressed = self.target_frames // self.time_compression_ratio
        
        # FramePack风格的采样策略
        ratio = random.random()
        if ratio < 0.15:
            condition_frames_compressed = 1
        elif 0.15 <= ratio < 0.9:
            condition_frames_compressed = random.randint(min_condition_compressed, max_condition_compressed)
        else:
            condition_frames_compressed = target_frames_compressed
        
        # 确保有足够的帧数
        min_required_frames = condition_frames_compressed + target_frames_compressed
        if compressed_total_frames < min_required_frames:
            return None
        
        start_frame_compressed = random.randint(0, compressed_total_frames - min_required_frames - 1)
        condition_end_compressed = start_frame_compressed + condition_frames_compressed
        target_end_compressed = condition_end_compressed + target_frames_compressed

        # FramePack风格的索引处理
        latent_indices = torch.arange(condition_end_compressed, target_end_compressed)
        
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
                    
        # 🔧 NuScenes特有：查找关键帧索引
        condition_keyframes_compressed = [idx for idx in compressed_keyframe_indices 
                                        if start_frame_compressed <= idx < condition_end_compressed]
        
        target_keyframes_compressed = [idx for idx in compressed_keyframe_indices 
                                    if condition_end_compressed <= idx < target_end_compressed]
        
        if not condition_keyframes_compressed:
            return None
        
        # 使用条件段的最后一个关键帧作为reference
        reference_keyframe_compressed = max(condition_keyframes_compressed)
        
        # 找到对应的原始关键帧索引用于pose查找
        reference_keyframe_original_idx = None
        for i, compressed_idx in enumerate(compressed_keyframe_indices):
            if compressed_idx == reference_keyframe_compressed:
                reference_keyframe_original_idx = i
                break
        
        if reference_keyframe_original_idx is None:
            return None
        
        # 找到目标段对应的原始关键帧索引
        target_keyframes_original_indices = []
        for compressed_idx in target_keyframes_compressed:
            for i, comp_idx in enumerate(compressed_keyframe_indices):
                if comp_idx == compressed_idx:
                    target_keyframes_original_indices.append(i)
                    break
        
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
            
            # 🔧 NuScenes特有数据
            'reference_keyframe_idx': reference_keyframe_original_idx,
            'target_keyframe_indices': target_keyframes_original_indices,
        }

    def calculate_relative_rotation(self, current_rotation, reference_rotation):
        """计算相对旋转四元数 - NuScenes专用"""
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


    def prepare_framepack_inputs(self, full_latents, segment_info):
        """🔧 准备FramePack风格的多尺度输入 - 修正版，正确处理空索引"""
        # 🔧 修正：处理4维输入 [C, T, H, W]，添加batch维度
        if len(full_latents.shape) == 4:
            full_latents = full_latents.unsqueeze(0)  # [C, T, H, W] -> [1, C, T, H, W]
            B, C, T, H, W = full_latents.shape
        else:
            B, C, T, H, W = full_latents.shape
        
        # 主要latents（用于去噪预测）
        latent_indices = segment_info['latent_indices']
        main_latents = full_latents[:, :, latent_indices, :, :]  # 注意维度顺序
        
        # 🔧 1x条件帧（起始帧 + 最后1帧）
        clean_latent_indices = segment_info['clean_latent_indices']
        clean_latents = full_latents[:, :, clean_latent_indices, :, :]  # 注意维度顺序
        
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
            'clean_latent_2x_indices': clean_latent_2x_indices_final,  # 🔧 使用真实索引（含-1填充）
            'clean_latent_4x_indices': clean_latent_4x_indices_final,  # 🔧 使用真实索引（含-1填充）
        }

    def create_sekai_pose_embeddings(self, cam_data, segment_info):
        """创建Sekai风格的pose embeddings"""
        cam_data_seq = cam_data['extrinsic']
        
        # 为所有帧计算相对pose
        all_keyframe_indices = []
        for compressed_idx in range(segment_info['start_frame'], segment_info['target_range'][1]):
            all_keyframe_indices.append(compressed_idx * 4)
        
        relative_cams = []
        for idx in all_keyframe_indices:
            cam_prev = cam_data_seq[idx]
            cam_next = cam_data_seq[idx + 4]
            relative_cam = compute_relative_pose(cam_prev, cam_next)
            relative_cams.append(torch.as_tensor(relative_cam[:3, :]))
        
        pose_embedding = torch.stack(relative_cams, dim=0)
        pose_embedding = rearrange(pose_embedding, 'b c d -> b (c d)')
        pose_embedding = pose_embedding.to(torch.bfloat16)

        return pose_embedding

    def create_openx_pose_embeddings(self, cam_data, segment_info):
        """🔧 创建OpenX风格的pose embeddings - 类似sekai但处理更短的序列"""
        cam_data_seq = cam_data['extrinsic']
        
        # 为所有帧计算相对pose - OpenX使用4倍间隔
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
    
    def create_spatialvid_pose_embeddings(self, cam_data, segment_info):
        """🔧 创建Spatialvid风格的pose embeddings - camera间隔为1帧而非4帧"""
        cam_data_seq = cam_data['extrinsic']
        
        # 为所有帧计算相对pose - spatialvid特有：每隔1帧而不是4帧
        all_keyframe_indices = []
        for compressed_idx in range(segment_info['start_frame'], segment_info['target_range'][1]):
            # 🔧 spatialvid关键差异：camera每隔4帧有一个，但索引递增1
            all_keyframe_indices.append(compressed_idx)
        
        relative_cams = []
        for idx in all_keyframe_indices:
            # 🔧 spatialvid关键差异：current和next是+1而不是+4
            cam_prev = cam_data_seq[idx]
            cam_next = cam_data_seq[idx + 1]  # 这里是+1，不是+4
            relative_cam = compute_relative_pose_matrix(cam_prev, cam_next)
            relative_cams.append(torch.as_tensor(relative_cam[:3, :]))
        
        pose_embedding = torch.stack(relative_cams, dim=0)
        pose_embedding = rearrange(pose_embedding, 'b c d -> b (c d)')
        pose_embedding = pose_embedding.to(torch.bfloat16)

        return pose_embedding
               
    def create_nuscenes_pose_embeddings_framepack(self, scene_info, segment_info):
        """创建NuScenes风格的pose embeddings - FramePack版本（简化版本，直接7维）"""
        keyframe_poses = scene_info['keyframe_poses']
        reference_keyframe_idx = segment_info['reference_keyframe_idx']
        target_keyframe_indices = segment_info['target_keyframe_indices']
        
        if reference_keyframe_idx >= len(keyframe_poses):
            return None
        
        reference_pose = keyframe_poses[reference_keyframe_idx]
        
        # 为所有帧（condition + target）创建pose embeddings
        start_frame = segment_info['start_frame']
        condition_end_compressed = start_frame + segment_info['condition_frames']
        target_end_compressed = condition_end_compressed + segment_info['target_frames']
        
        # 压缩后的关键帧索引
        compressed_keyframe_indices = [idx // self.time_compression_ratio for idx in scene_info['keyframe_indices']]
        
        # 找到condition段的关键帧
        condition_keyframes_compressed = [idx for idx in compressed_keyframe_indices 
                                        if start_frame <= idx < condition_end_compressed]
        
        # 找到对应的原始关键帧索引
        condition_keyframes_original_indices = []
        for compressed_idx in condition_keyframes_compressed:
            for i, comp_idx in enumerate(compressed_keyframe_indices):
                if comp_idx == compressed_idx:
                    condition_keyframes_original_indices.append(i)
                    break
        
        pose_vecs = []
        
        # 为condition帧计算pose
        for i in range(segment_info['condition_frames']):
            if not condition_keyframes_original_indices:
                translation = torch.zeros(3, dtype=torch.float32)
                rotation = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)
            else:
                # 为condition帧分配pose
                if len(condition_keyframes_original_indices) == 1:
                    keyframe_idx = condition_keyframes_original_indices[0]
                else:
                    if segment_info['condition_frames'] == 1:
                        keyframe_idx = condition_keyframes_original_indices[0]
                    else:
                        interp_ratio = i / (segment_info['condition_frames'] - 1)
                        interp_idx = int(interp_ratio * (len(condition_keyframes_original_indices) - 1))
                        keyframe_idx = condition_keyframes_original_indices[interp_idx]
                
                if keyframe_idx >= len(keyframe_poses):
                    translation = torch.zeros(3, dtype=torch.float32)
                    rotation = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)
                else:
                    condition_pose = keyframe_poses[keyframe_idx]
                    
                    translation = torch.tensor(
                        np.array(condition_pose['translation']) - np.array(reference_pose['translation']),
                        dtype=torch.float32
                    )
                    
                    relative_rotation = self.calculate_relative_rotation(
                        condition_pose['rotation'],
                        reference_pose['rotation']
                    )
                    
                    rotation = relative_rotation
            
            # 🔧 简化：直接7维 [translation(3) + rotation(4)]
            pose_vec = torch.cat([translation, rotation], dim=0)  # [7D]
            pose_vecs.append(pose_vec)
        
        # 为target帧计算pose
        if not target_keyframe_indices:
            for i in range(segment_info['target_frames']):
                pose_vec = torch.cat([
                    torch.zeros(3, dtype=torch.float32),
                    torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
                ], dim=0)  # [7D]
                pose_vecs.append(pose_vec)
        else:
            for i in range(segment_info['target_frames']):
                if len(target_keyframe_indices) == 1:
                    target_keyframe_idx = target_keyframe_indices[0]
                else:
                    if segment_info['target_frames'] == 1:
                        target_keyframe_idx = target_keyframe_indices[0]
                    else:
                        interp_ratio = i / (segment_info['target_frames'] - 1)
                        interp_idx = int(interp_ratio * (len(target_keyframe_indices) - 1))
                        target_keyframe_idx = target_keyframe_indices[interp_idx]
                
                if target_keyframe_idx >= len(keyframe_poses):
                    pose_vec = torch.cat([
                        torch.zeros(3, dtype=torch.float32),
                        torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
                    ], dim=0)  # [7D]
                else:
                    target_pose = keyframe_poses[target_keyframe_idx]
                    
                    relative_translation = torch.tensor(
                        np.array(target_pose['translation']) - np.array(reference_pose['translation']),
                        dtype=torch.float32
                    )
                    
                    relative_rotation = self.calculate_relative_rotation(
                        target_pose['rotation'],
                        reference_pose['rotation']
                    )
                    
                    # 🔧 简化：直接7维 [translation(3) + rotation(4)]
                    pose_vec = torch.cat([relative_translation, relative_rotation], dim=0)  # [7D]
                
                pose_vecs.append(pose_vec)
        
        if not pose_vecs:
            return None
        
        pose_sequence = torch.stack(pose_vecs, dim=0)  # [total_frames, 7]
        
        return pose_sequence

    # 修改select_dynamic_segment方法
    def select_dynamic_segment(self, full_latents, dataset_type, scene_info=None):
        """🔧 根据数据集类型选择不同的段落选择策略"""
        if dataset_type == 'nuscenes' and scene_info is not None:
            return self.select_dynamic_segment_nuscenes(scene_info)
        else:
            # 原有的sekai方式
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

            # FramePack风格的索引处理
            latent_indices = torch.arange(condition_end_compressed, target_end_compressed)
            
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
                if dataset_type == 'spatialvid':
                    keyframe_original_idx.append(compressed_idx)  # spatialvid直接使用compressed_idx
                elif dataset_type == 'openx' or 'sekai':  # 🔧 新增openx处理
                    keyframe_original_idx.append(compressed_idx * 4)  # openx使用4倍间隔

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

    # 修改create_pose_embeddings方法
    def create_pose_embeddings(self, cam_data, segment_info, dataset_type, scene_info=None):
        """🔧 根据数据集类型创建pose embeddings"""
        if dataset_type == 'nuscenes' and scene_info is not None:
            return self.create_nuscenes_pose_embeddings_framepack(scene_info, segment_info)
        elif dataset_type == 'spatialvid':  # 🔧 新增spatialvid处理
            return self.create_spatialvid_pose_embeddings(cam_data, segment_info)
        elif dataset_type == 'sekai':
            return self.create_sekai_pose_embeddings(cam_data, segment_info)
        elif dataset_type == 'openx':  # 🔧 新增openx处理
            return self.create_openx_pose_embeddings(cam_data, segment_info)
        
    def __getitem__(self, index):
        while True:
            try:
                # 根据权重随机选择场景
                scene_idx = np.random.choice(len(self.scene_dirs), p=self.sampling_probs)
                scene_dir = self.scene_dirs[scene_idx]
                dataset_info = self.dataset_info[scene_dir]
                
                dataset_name = dataset_info['name']
                dataset_type = dataset_info['type']
                
                # 🔧 根据数据集类型加载数据
                scene_info = None
                if dataset_type == 'nuscenes':
                    # NuScenes需要加载scene_info.json
                    scene_info_path = os.path.join(scene_dir, "scene_info.json")
                    if os.path.exists(scene_info_path):
                        with open(scene_info_path, 'r') as f:
                            scene_info = json.load(f)
                    
                    # NuScenes使用不同的编码文件名
                    encoded_path = os.path.join(scene_dir, "encoded_video-480p.pth")
                    if not os.path.exists(encoded_path):
                        encoded_path = os.path.join(scene_dir, "encoded_video.pth")  # fallback
                    
                    encoded_data = torch.load(encoded_path, weights_only=True, map_location="cpu")
                else:
                    # Sekai数据集
                    encoded_path = os.path.join(scene_dir, "encoded_video.pth")
                    encoded_data = torch.load(encoded_path, weights_only=False, map_location="cpu")
                
                full_latents = encoded_data['latents']
                if full_latents.shape[1] <= 10:
                    continue
                cam_data = encoded_data.get('cam_emb', encoded_data)
                
                # 🔧 验证NuScenes的latent帧数
                if dataset_type == 'nuscenes' and scene_info is not None:
                    expected_latent_frames = scene_info['total_frames'] // self.time_compression_ratio
                    actual_latent_frames = full_latents.shape[1]
                    
                    if abs(actual_latent_frames - expected_latent_frames) > 2:
                        print(f"⚠️  NuScenes Latent帧数不匹配，跳过此样本")
                        continue
                
                # 使用数据集特定的段落选择策略
                segment_info = self.select_dynamic_segment(full_latents, dataset_type, scene_info)
                if segment_info is None:
                    continue
                
                # 创建数据集特定的pose embeddings
                all_camera_embeddings = self.create_pose_embeddings(cam_data, segment_info, dataset_type, scene_info)
                if all_camera_embeddings is None:
                    continue
                
                # 准备FramePack风格的多尺度输入
                framepack_inputs = self.prepare_framepack_inputs(full_latents, segment_info)
                
                n = segment_info["condition_frames"]
                m = segment_info['target_frames']
                
                # 处理camera embedding with mask
                mask = torch.zeros(n+m, dtype=torch.float32)
                mask[:n] = 1.0
                mask = mask.view(-1, 1)
                
                # 🔧 NuScenes返回的是直接的embedding，Sekai返回的是tensor
                if isinstance(all_camera_embeddings, torch.Tensor):
                    camera_with_mask = torch.cat([all_camera_embeddings, mask], dim=1)
                else:
                    # NuScenes风格，直接就是最终的embedding
                    camera_with_mask = torch.cat([all_camera_embeddings, mask], dim=1)
                
                result = {
                    # FramePack风格的多尺度输入
                    "latents": framepack_inputs['latents'],
                    "clean_latents": framepack_inputs['clean_latents'],
                    "clean_latents_2x": framepack_inputs['clean_latents_2x'],
                    "clean_latents_4x": framepack_inputs['clean_latents_4x'],
                    "latent_indices": framepack_inputs['latent_indices'],
                    "clean_latent_indices": framepack_inputs['clean_latent_indices'],
                    "clean_latent_2x_indices": framepack_inputs['clean_latent_2x_indices'],
                    "clean_latent_4x_indices": framepack_inputs['clean_latent_4x_indices'],
                    
                    # Camera数据
                    "camera": camera_with_mask,
                    
                    # 其他数据
                    "prompt_emb": encoded_data["prompt_emb"],
                    "image_emb": encoded_data.get("image_emb", {}),
                    
                    # 元信息
                    "condition_frames": n,
                    "target_frames": m,
                    "scene_name": os.path.basename(scene_dir),
                    "dataset_name": dataset_name,
                    "dataset_type": dataset_type,
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
    """在模型加载前替换DiT模型类为MoE版本"""
    from diffsynth.models.wan_video_dit_moe import WanModelMoe
    from diffsynth.configs.model_config import model_loader_configs
    
    # 修改model_loader_configs中的配置
    for i, config in enumerate(model_loader_configs):
        keys_hash, keys_hash_with_shape, model_names, model_classes, model_resource = config
        
        # 检查是否包含wan_video_dit模型
        if 'wan_video_dit' in model_names:
            new_model_names = []
            new_model_classes = []
            
            for name, cls in zip(model_names, model_classes):
                if name == 'wan_video_dit':
                    new_model_names.append(name)
                    new_model_classes.append(WanModelMoe)  # 🔧 使用MoE版本
                    print(f"✅ 替换了模型类: {name} -> WanModelMoe")
                else:
                    new_model_names.append(name)
                    new_model_classes.append(cls)
            
            # 更新配置
            model_loader_configs[i] = (keys_hash, keys_hash_with_shape, new_model_names, new_model_classes, model_resource)

class MultiDatasetLightningModelForTrain(pl.LightningModule):
    def __init__(
        self,
        dit_path,
        learning_rate=1e-5,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        resume_ckpt_path=None,
        # 🔧 MoE参数
        use_moe=False,
        moe_config=None
    ):
        super().__init__()
        self.use_moe = use_moe
        self.moe_config = moe_config or {}
        
        replace_dit_model_in_manager()
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        if os.path.isfile(dit_path):
            model_manager.load_models([dit_path])
        else:
            dit_path = dit_path.split(",")
            model_manager.load_models([dit_path])
        model_manager.load_models(["models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"])
        
        self.pipe = WanVideoReCamMasterPipeline.from_model_manager(model_manager)
        self.pipe.scheduler.set_timesteps(1000, training=True)

        # 添加FramePack的clean_x_embedder
        self.add_framepack_components()
        if self.use_moe:
            self.add_moe_components()

        # 🔧 添加camera编码器（wan_video_dit_moe.py已经包含MoE逻辑）
        dim = self.pipe.dit.blocks[0].self_attn.q.weight.shape[0]
        for block in self.pipe.dit.blocks:
            # 🔧 简化：只添加传统camera编码器，MoE逻辑在wan_video_dit_moe.py中
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
        
        # 🔧 训练参数设置
        for name, module in self.pipe.denoising_model().named_modules():
            if any(keyword in name for keyword in [
                                                "moe", "sekai_processor"]):
                for param in module.parameters():
                    param.requires_grad = True
        
        self.learning_rate = learning_rate
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        
        # 创建可视化目录
        self.vis_dir = "multi_dataset_dynamic/visualizations"
        os.makedirs(self.vis_dir, exist_ok=True)

    def add_moe_components(self):
        """🔧 添加MoE相关组件 - 类似add_framepack_components的方式"""
        if not hasattr(self.pipe.dit, 'moe_config'):
            self.pipe.dit.moe_config = self.moe_config
            print("✅ 添加了MoE配置到模型")
        
        # 为每个block动态添加MoE组件
        dim = self.pipe.dit.blocks[0].self_attn.q.weight.shape[0]
        unified_dim = self.moe_config.get("unified_dim", 30)
        
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
                num_experts=self.moe_config.get("num_experts", 4),
                top_k=self.moe_config.get("top_k", 2)
            )
            
            print(f"✅ Block {i} 添加了MoE组件 (unified_dim: {unified_dim}, experts: {self.moe_config.get('num_experts', 4)})")


    def add_framepack_components(self):
        """🔧 添加FramePack相关组件"""
        if not hasattr(self.pipe.dit, 'clean_x_embedder'):
            inner_dim = self.pipe.dit.blocks[0].self_attn.q.weight.shape[0]
            
            class CleanXEmbedder(nn.Module):
                def __init__(self, inner_dim):
                    super().__init__()
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
        """🔧 多数据集训练步骤"""
        condition_frames = batch["condition_frames"][0].item()
        target_frames = batch["target_frames"][0].item()
        
        original_condition_frames = batch.get("original_condition_frames", [condition_frames * 4])[0]
        original_target_frames = batch.get("original_target_frames", [target_frames * 4])[0]

        dataset_name = batch.get("dataset_name", ["unknown"])[0]
        dataset_type = batch.get("dataset_type", ["sekai"])[0]
        scene_name = batch.get("scene_name", ["unknown"])[0]
        
        # 准备输入数据
        latents = batch["latents"].to(self.device)
        if len(latents.shape) == 4:
            latents = latents.unsqueeze(0)
        
        clean_latents = batch["clean_latents"].to(self.device) if batch["clean_latents"].numel() > 0 else None
        if clean_latents is not None and len(clean_latents.shape) == 4:
            clean_latents = clean_latents.unsqueeze(0)
        
        clean_latents_2x = batch["clean_latents_2x"].to(self.device) if batch["clean_latents_2x"].numel() > 0 else None
        if clean_latents_2x is not None and len(clean_latents_2x.shape) == 4:
            clean_latents_2x = clean_latents_2x.unsqueeze(0)
        
        clean_latents_4x = batch["clean_latents_4x"].to(self.device) if batch["clean_latents_4x"].numel() > 0 else None
        if clean_latents_4x is not None and len(clean_latents_4x.shape) == 4:
            clean_latents_4x = clean_latents_4x.unsqueeze(0)
        
        # 索引处理
        latent_indices = batch["latent_indices"].to(self.device)
        clean_latent_indices = batch["clean_latent_indices"].to(self.device) if batch["clean_latent_indices"].numel() > 0 else None
        clean_latent_2x_indices = batch["clean_latent_2x_indices"].to(self.device) if batch["clean_latent_2x_indices"].numel() > 0 else None
        clean_latent_4x_indices = batch["clean_latent_4x_indices"].to(self.device) if batch["clean_latent_4x_indices"].numel() > 0 else None
        
        # Camera embedding处理
        cam_emb = batch["camera"].to(self.device)
        
        # 🔧 根据数据集类型设置modality_inputs
        if dataset_type == "sekai":
            modality_inputs = {"sekai": cam_emb}
        elif dataset_type == "spatialvid":  # 🔧 spatialvid使用sekai processor
            modality_inputs = {"sekai": cam_emb}  # 注意：这里使用"sekai"键
        elif dataset_type == "nuscenes":
            modality_inputs = {"nuscenes": cam_emb}
        elif dataset_type == "openx":  # 🔧 新增：openx使用独立的processor
            modality_inputs = {"openx": cam_emb}
        else:
            modality_inputs = {"sekai": cam_emb}  # 默认
        
        camera_dropout_prob = 0.05
        if random.random() < camera_dropout_prob:
            cam_emb = torch.zeros_like(cam_emb)
            # 同时清空modality_inputs
            for key in modality_inputs:
                modality_inputs[key] = torch.zeros_like(modality_inputs[key])
            print(f"应用camera dropout for CFG training (dataset: {dataset_name}, type: {dataset_type})")
        
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
        
        # FramePack风格的噪声处理
        noisy_condition_latents = None
        if clean_latents is not None:
            noisy_condition_latents = copy.deepcopy(clean_latents)
            is_add_noise = random.random()
            if is_add_noise > 0.2:
                noise_cond = torch.randn_like(clean_latents)
                timestep_id_cond = torch.randint(0, self.pipe.scheduler.num_train_timesteps//4*3, (1,))
                timestep_cond = self.pipe.scheduler.timesteps[timestep_id_cond].to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
                noisy_condition_latents = self.pipe.scheduler.add_noise(clean_latents, noise_cond, timestep_cond)

        extra_input = self.pipe.prepare_extra_input(latents)
        origin_latents = copy.deepcopy(latents)
        noisy_latents = self.pipe.scheduler.add_noise(latents, noise, timestep)
        
        training_target = self.pipe.scheduler.training_target(latents, noise, timestep)
        
        # 🔧 Forward调用 - 传递modality_inputs
        noise_pred, moe_loss = self.pipe.denoising_model()(
            noisy_latents, 
            timestep=timestep, 
            cam_emb=cam_emb,
            modality_inputs=modality_inputs,  # 🔧 传递多模态输入
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
        
        print(f'--------loss ({dataset_name}-{dataset_type})------------:', loss)

        return loss

    def configure_optimizers(self):
        trainable_modules = filter(lambda p: p.requires_grad, self.pipe.denoising_model().parameters())
        optimizer = torch.optim.AdamW(trainable_modules, lr=self.learning_rate)
        return optimizer
    
    def on_save_checkpoint(self, checkpoint):
        checkpoint_dir = "/share_zhuyixuan05/zhuyixuan05/ICLR2026/framepack_moe_test"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        current_step = self.global_step
        checkpoint.clear()
        
        state_dict = self.pipe.denoising_model().state_dict()
        torch.save(state_dict, os.path.join(checkpoint_dir, f"step{current_step}_moe.ckpt"))
        print(f"Saved MoE model checkpoint: step{current_step}_moe.ckpt")

def train_multi_dataset(args):
    """训练支持多数据集MoE的模型"""
    
    # 🔧 数据集配置
    dataset_configs = [
        {
            'name': 'sekai-drone',
            'paths': ['/share_zhuyixuan05/zhuyixuan05/sekai-game-drone'],
            'type': 'sekai',
            'weight': 1.0
        },
        {
            'name': 'sekai-walking',
            'paths': ['/share_zhuyixuan05/zhuyixuan05/sekai-game-walking'],
            'type': 'sekai',
            'weight': 1.0
        },
        # {
        #     'name': 'spatialvid',
        #     'paths': ['/share_zhuyixuan05/zhuyixuan05/spatialvid'],
        #     'type': 'spatialvid',
        #     'weight': 1.0
        # },
        # {
        #     'name': 'nuscenes',
        #     'paths': ['/share_zhuyixuan05/zhuyixuan05/nuscenes_video_generation_dynamic'],
        #     'type': 'nuscenes',
        #     'weight': 4.0
        # },
        # {
        #     'name': 'openx-fractal',
        #     'paths': ['/share_zhuyixuan05/zhuyixuan05/openx-fractal-encoded'],
        #     'type': 'openx',
        #     'weight': 1.0
        # }
    ]
    
    dataset = MultiDatasetDynamicDataset(
        dataset_configs,
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
    
    # 🔧 MoE配置
    moe_config = {
        "unified_dim": args.unified_dim,  # 新增
        "num_experts": args.moe_num_experts,
        "top_k": args.moe_top_k,
        "moe_loss_weight": args.moe_loss_weight,
        "sekai_input_dim": 13,
        "nuscenes_input_dim": 8,
        "openx_input_dim": 13  
    }
    
    model = MultiDatasetLightningModelForTrain(
        dit_path=args.dit_path,
        learning_rate=args.learning_rate,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        resume_ckpt_path=args.resume_ckpt_path,
        use_moe=True,  # 总是使用MoE
        moe_config=moe_config
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
    parser = argparse.ArgumentParser(description="Train Multi-Dataset FramePack with MoE")
    parser.add_argument("--dit_path", type=str, default="models/Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors")
    parser.add_argument("--output_path", type=str, default="./")
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--steps_per_epoch", type=int, default=2000)
    parser.add_argument("--max_epochs", type=int, default=100000)
    parser.add_argument("--min_condition_frames", type=int, default=8, help="最小条件帧数")
    parser.add_argument("--max_condition_frames", type=int, default=120, help="最大条件帧数")
    parser.add_argument("--target_frames", type=int, default=32, help="目标帧数")
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--training_strategy", type=str, default="deepspeed_stage_1")
    parser.add_argument("--use_gradient_checkpointing", default=False)
    parser.add_argument("--use_gradient_checkpointing_offload", action="store_true")
    parser.add_argument("--resume_ckpt_path", type=str, default="/share_zhuyixuan05/zhuyixuan05/ICLR2026/framepack_moe_test/step1500_moe.ckpt")
    
    # 🔧 MoE参数
    parser.add_argument("--unified_dim", type=int, default=25, help="统一的中间维度")
    parser.add_argument("--moe_num_experts", type=int, default=1, help="专家数量")
    parser.add_argument("--moe_top_k", type=int, default=1, help="Top-K专家")
    parser.add_argument("--moe_loss_weight", type=float, default=0.00, help="MoE损失权重")
    
    args = parser.parse_args()
    
    print("🔧 多数据集MoE训练配置:")
    print(f"  - 使用wan_video_dit_moe.py作为模型")
    print(f"  - 统一维度: {args.unified_dim}")
    print(f"  - 专家数量: {args.moe_num_experts}")
    print(f"  - Top-K: {args.moe_top_k}")
    print(f"  - MoE损失权重: {args.moe_loss_weight}")
    print("  - 数据集:")
    print("    - sekai-game-drone (sekai模态)")
    print("    - sekai-game-walking (sekai模态)")
    print("    - spatialvid (使用sekai模态处理器)") 
    print("    - openx-fractal (使用sekai模态处理器)")
    print(f"   - nuscenes (nuscenes模态)")
    
    train_multi_dataset(args)