# Mixture of Experts (MoE) training for fusing NuScenes and Sekai datasets
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
from diffsynth import WanVideoAstraPipeline, ModelManager
from torchvision.transforms import v2
from einops import rearrange
from pose_classifier import PoseClassifier
import argparse
from scipy.spatial.transform import Rotation as R

# def get_traj_position_change(cam_c2w, stride=1):
#     positions = cam_c2w[:, :3, 3]
    
#     traj_coord = []
#     tarj_angle = []
#     for i in range(0, len(positions) - 2 * stride):
#         v1 = positions[i + stride] - positions[i]
#         v2 = positions[i + 2 * stride] - positions[i + stride]

#         norm1 = np.linalg.norm(v1)
#         norm2 = np.linalg.norm(v2)
#         if norm1 < 1e-6 or norm2 < 1e-6:
#             continue

#         cos_angle = np.dot(v1, v2) / (norm1 * norm2)
#         angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

#         traj_coord.append(v1)
#         tarj_angle.append(angle)
    
#     return traj_coord, tarj_angle

# def get_traj_rotation_change(cam_c2w, stride=1):
#     rotations = cam_c2w[:, :3, :3]
    
#     traj_rot_angle = []
#     for i in range(0, len(rotations) - stride):
#         z1 = rotations[i][:, 2]
#         z2 = rotations[i + stride][:, 2]

#         norm1 = np.linalg.norm(z1)
#         norm2 = np.linalg.norm(z2)
#         if norm1 < 1e-6 or norm2 < 1e-6:
#             continue

#         cos_angle = np.dot(z1, z2) / (norm1 * norm2)
#         angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
#         traj_rot_angle.append(angle)

#     return traj_rot_angle

def compute_relative_pose(pose_a, pose_b, use_torch=False):
    """Compute the relative pose matrix of camera B with respect to camera A"""
    assert pose_a.shape == (4, 4), f"Camera A extrinsic matrix shape must be (4,4), got {pose_a.shape}"
    assert pose_b.shape == (4, 4), f"Camera B extrinsic matrix shape must be (4,4), got {pose_b.shape}"
    
    if use_torch:
        if not isinstance(pose_a, torch.Tensor):
            pose_a = torch.from_numpy(pose_a).float()
        if not isinstance(pose_b, torch.Tensor):
            pose_b = torch.from_numpy(pose_b).float()
        
        pose_a_inv = torch.inverse(pose_a.float())
        relative_pose = torch.matmul(pose_b.float(), pose_a_inv)
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
    Compute the relative pose between two adjacent frames, returning a 3x4 camera matrix [R_rel | t_rel]
    
    Args:
    pose1: Camera pose of frame i, an array of shape (7,) [tx1, ty1, tz1, qx1, qy1, qz1, qw1]
    pose2: Camera pose of frame i+1, an array of shape (7,) [tx2, ty2, tz2, qx2, qy2, qz2, qw2]
    
    Returns:
    relative_matrix: 3x4 relative pose matrix, first 3 columns are rotation matrix R_rel, 4th column is translation vector t_rel
    """
    
    pose1 = pose1.detach().to(torch.float64).cpu().numpy()
    pose2 = pose2.detach().to(torch.float64).cpu().numpy()
    
    
    # Separate translation vector and quaternion
    t1 = pose1[:3]  # Translation of frame i [tx1, ty1, tz1]
    q1 = pose1[3:]  # Quaternion of frame i [qx1, qy1, qz1, qw1]
    t2 = pose2[:3]  # Translation of frame i+1
    q2 = pose2[3:]  # Quaternion of frame i+1
    
    # 1. Compute relative rotation matrix R_rel
    rot1 = R.from_quat(q1)  # Rotation of frame i
    rot2 = R.from_quat(q2)  # Rotation of frame i+1
    rot_rel = rot2 * rot1.inv()  # Relative rotation = Rotation of next frame x Inverse of previous frame rotation
    R_rel = rot_rel.as_matrix()  # Convert to 3x3 matrix
    
    # 2. Compute relative translation vector t_rel
    R1_T = rot1.as_matrix().T  # Transpose of previous frame rotation matrix (equivalent to inverse)
    t_rel = R1_T @ (t2 - t1)   # Relative translation = R1^T x (t2 - t1)
    
    # 3. Combine into 3x4 matrix [R_rel | t_rel]
    relative_matrix = np.hstack([R_rel, t_rel.reshape(3, 1)])
    
    return relative_matrix

class MultiDatasetDynamicDataset(torch.utils.data.Dataset):
    """Multi-Dataset Dynamic History Length Dataset supporting FramePack mechanism - Fusing NuScenes and Sekai"""
    
    def __init__(self, dataset_configs, steps_per_epoch, 
                 min_condition_frames=10, max_condition_frames=40,
                 target_frames=10, height=900, width=1600):
        """
        Args:
            dataset_configs: List of dataset configurations, each containing {
                'name': Dataset name,
                'paths': List of dataset paths,
                'type': Dataset type ('sekai' or 'nuscenes'),
                'weight': Sampling weight
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
        
        # VAE time compression ratio
        self.time_compression_ratio = 4
        
        # 🔧 Scan all datasets, build a unified scene index
        self.scene_dirs = []
        self.dataset_info = {}  # Record dataset information for each scene
        self.dataset_weights = []  # Sampling weight for each scene
        
        total_scenes = 0
        
        for config in self.dataset_configs:
            dataset_name = config['name']
            # dataset_paths = config['paths'] if isinstance(config['paths'], list) else [config['paths']]
            dataset_manifests = config['manifest'] if isinstance(config['manifest'], list) else [config['manifest']]
            dataset_type = config['type']
            dataset_weight = config.get('weight', 1.0)
            
            print(f"🔧 Scanning dataset: {dataset_name} (Type: {dataset_type})")
            
            dataset_scenes = []
            for dataset_manifest in dataset_manifests:
                print(f"  📁 Checking path: {dataset_manifest}")
                if os.path.exists(dataset_manifest):
                    with open(dataset_manifest, "r") as f:
                        data = json.load(f)
                        pth_list = [d["pth"] for d in data["entries"]]
                        print(f"  📁 Found {len(pth_list)} paths in manifest")
                        for pth in pth_list:
                            scene_dir = os.path.join("/mnt/data/louis_crq/preprocessed_data/SpatialVID_Wan2", pth)
                            if not os.path.exists(scene_dir):
                                print(f"  ❌ Path does not exist: {scene_dir}")
                                continue
                            else:
                                self.scene_dirs.append(scene_dir)
                                dataset_scenes.append(scene_dir)
                                self.dataset_info[scene_dir] = {
                                    'name': dataset_name,
                                    'type': dataset_type,
                                    'weight': dataset_weight
                                }
                                self.dataset_weights.append(dataset_weight)      
                else:
                    print(f"  ❌ Path does not exist: {dataset_manifest}")
                
                print(f"  ✅ Found {len(dataset_scenes)} scenes")
                total_scenes += len(dataset_scenes)
                    
        # Count scenes per dataset
        dataset_counts = {}
        for scene_dir in self.scene_dirs:
            dataset_name = self.dataset_info[scene_dir]['name']
            dataset_type = self.dataset_info[scene_dir]['type']
            key = f"{dataset_name} ({dataset_type})"
            dataset_counts[key] = dataset_counts.get(key, 0) + 1
        
        for dataset_key, count in dataset_counts.items():
            print(f"  - {dataset_key}: {count} scenes")
        
        assert len(self.scene_dirs) > 0, "No encoded scenes found!"
        
        # 🔧 Calculate sampling probabilities
        total_weight = sum(self.dataset_weights)
        self.sampling_probs = [w / total_weight for w in self.dataset_weights]

    def select_dynamic_segment_nuscenes(self, scene_info):
        """🔧 NuScenes specific FramePack style segment selection"""
        keyframe_indices = scene_info['keyframe_indices']  # Original frame indices
        total_frames = scene_info['total_frames']  # Original total frames
        
        if len(keyframe_indices) < 2:
            return None
        
        # Calculate compressed frame count
        compressed_total_frames = total_frames // self.time_compression_ratio
        compressed_keyframe_indices = [idx // self.time_compression_ratio for idx in keyframe_indices]
        
        min_condition_compressed = self.min_condition_frames // self.time_compression_ratio
        max_condition_compressed = self.max_condition_frames // self.time_compression_ratio
        target_frames_compressed = self.target_frames // self.time_compression_ratio
        
        # FramePack style sampling strategy
        ratio = random.random()
        if ratio < 0.15:
            condition_frames_compressed = 1
        elif 0.15 <= ratio < 0.9:
            condition_frames_compressed = random.randint(min_condition_compressed, max_condition_compressed)
        else:
            condition_frames_compressed = target_frames_compressed
        
        # Ensure enough frames
        min_required_frames = condition_frames_compressed + target_frames_compressed
        if compressed_total_frames < min_required_frames:
            return None
        
        start_frame_compressed = random.randint(0, compressed_total_frames - min_required_frames - 1)
        condition_end_compressed = start_frame_compressed + condition_frames_compressed
        target_end_compressed = condition_end_compressed + target_frames_compressed

        # FramePack style index handling
        latent_indices = torch.arange(condition_end_compressed, target_end_compressed)
        
        # 1x frames: Start frame + Last 1 frame
        clean_latent_indices_start = torch.tensor([start_frame_compressed])
        clean_latent_1x_indices = torch.tensor([condition_end_compressed - 1])
        clean_latent_indices = torch.cat([clean_latent_indices_start, clean_latent_1x_indices])
        
        # 🔧 2x frames: Determined by actual condition length
        if condition_frames_compressed >= 2:
            # Take last 2 frames (if available)
            clean_latent_2x_start = max(start_frame_compressed, condition_end_compressed - 2)
            clean_latent_2x_indices = torch.arange(clean_latent_2x_start-1, condition_end_compressed-1)
        else:
            # If not enough condition frames (< 2), create empty indices
            clean_latent_2x_indices = torch.tensor([], dtype=torch.long)
        
        # 🔧 4x frames: Determined by actual condition length, max 16 frames
        if condition_frames_compressed >= 1:
            # Take max 16 frames of history (if available)
            clean_4x_start = max(start_frame_compressed, condition_end_compressed - 16)
            clean_latent_4x_indices = torch.arange(clean_4x_start-3, condition_end_compressed-3)
        else:
            clean_latent_4x_indices = torch.tensor([], dtype=torch.long)
                    
        # 🔧 NuScenes specific: Find keyframe indices
        condition_keyframes_compressed = [idx for idx in compressed_keyframe_indices 
                                        if start_frame_compressed <= idx < condition_end_compressed]
        
        target_keyframes_compressed = [idx for idx in compressed_keyframe_indices 
                                    if condition_end_compressed <= idx < target_end_compressed]
        
        if not condition_keyframes_compressed:
            return None
        
        # Use the last keyframe of the condition segment as reference
        reference_keyframe_compressed = max(condition_keyframes_compressed)
        
        # Find the corresponding original keyframe index for pose lookup
        reference_keyframe_original_idx = None
        for i, compressed_idx in enumerate(compressed_keyframe_indices):
            if compressed_idx == reference_keyframe_compressed:
                reference_keyframe_original_idx = i
                break
        
        if reference_keyframe_original_idx is None:
            return None
        
        # Find the corresponding original keyframe index for the target segment
        target_keyframes_original_indices = []
        for compressed_idx in target_keyframes_compressed:
            for i, comp_idx in enumerate(compressed_keyframe_indices):
                if comp_idx == compressed_idx:
                    target_keyframes_original_indices.append(i)
                    break
        
        # Corresponding original keyframe indices
        keyframe_original_idx = []
        for compressed_idx in range(start_frame_compressed, target_end_compressed):
            keyframe_original_idx.append(compressed_idx * 4)
        
        return {
            'start_frame': start_frame_compressed,
            'condition_frames': condition_frames_compressed,
            'target_frames': target_frames_compressed,
            'condition_range': (start_frame_compressed, condition_end_compressed),
            'target_range': (condition_end_compressed, target_end_compressed),
            
            # FramePack style indices
            'latent_indices': latent_indices,
            'clean_latent_indices': clean_latent_indices,
            'clean_latent_2x_indices': clean_latent_2x_indices,
            'clean_latent_4x_indices': clean_latent_4x_indices,
            
            'keyframe_original_idx': keyframe_original_idx,
            'original_condition_frames': condition_frames_compressed * self.time_compression_ratio,
            'original_target_frames': target_frames_compressed * self.time_compression_ratio,
            
            # 🔧 NuScenes specific data
            'reference_keyframe_idx': reference_keyframe_original_idx,
            'target_keyframe_indices': target_keyframes_original_indices,
        }

    def calculate_relative_rotation(self, current_rotation, reference_rotation):
        """Compute relative rotation quaternion - NuScenes specific"""
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
        """🔧 Prepare FramePack style multi-scale inputs - Revised version, correctly handling empty indices"""
        # 🔧 Correction: Handle 4D input [C, T, H, W], add batch dimension
        if len(full_latents.shape) == 4:
            full_latents = full_latents.unsqueeze(0)  # [C, T, H, W] -> [1, C, T, H, W]
            B, C, T, H, W = full_latents.shape
        else:
            B, C, T, H, W = full_latents.shape
        
        # Main latents (for denoising prediction)
        latent_indices = segment_info['latent_indices']
        main_latents = full_latents[:, :, latent_indices, :, :]  # Note dimension order
        
        # 🔧 1x condition frames (Start frame + Last 1 frame)
        clean_latent_indices = segment_info['clean_latent_indices']
        clean_latents = full_latents[:, :, clean_latent_indices, :, :]  # Note dimension order
        
        # 🔧 4x condition frames - Always 16 frames, use real indices + 0 padding
        clean_latent_4x_indices = segment_info['clean_latent_4x_indices']
        
        # Create fixed length 16 latents, initialized to 0
        clean_latents_4x = torch.zeros(B, C, 16, H, W, dtype=full_latents.dtype)
        clean_latent_4x_indices_final = torch.full((16,), -1, dtype=torch.long)  # -1 means padding
        
        # 🔧 Correction: Check if there are valid 4x indices
        if len(clean_latent_4x_indices) > 0:
            actual_4x_frames = len(clean_latent_4x_indices)
            # Fill from back to front, ensuring the latest frames are at the end
            start_pos = max(0, 16 - actual_4x_frames)
            end_pos = 16
            actual_start = max(0, actual_4x_frames - 16)  # If more than 16 frames, only take the last 16
            
            clean_latents_4x[:, :, start_pos:end_pos, :, :] = full_latents[:, :, clean_latent_4x_indices[actual_start:], :, :]
            clean_latent_4x_indices_final[start_pos:end_pos] = clean_latent_4x_indices[actual_start:]
        
        # 🔧 2x condition frames - Always 2 frames, use real indices + 0 padding
        clean_latent_2x_indices = segment_info['clean_latent_2x_indices']
        
        # Create fixed length 2 latents, initialized to 0
        clean_latents_2x = torch.zeros(B, C, 2, H, W, dtype=full_latents.dtype)
        clean_latent_2x_indices_final = torch.full((2,), -1, dtype=torch.long)  # -1 means padding
        
        # 🔧 Correction: Check if there are valid 2x indices
        if len(clean_latent_2x_indices) > 0:
            actual_2x_frames = len(clean_latent_2x_indices)
            # Fill from back to front, ensuring the latest frames are at the end
            start_pos = max(0, 2 - actual_2x_frames)
            end_pos = 2
            actual_start = max(0, actual_2x_frames - 2)  # If more than 2 frames, only take the last 2
            
            clean_latents_2x[:, :, start_pos:end_pos, :, :] = full_latents[:, :, clean_latent_2x_indices[actual_start:], :, :]
            clean_latent_2x_indices_final[start_pos:end_pos] = clean_latent_2x_indices[actual_start:]
        
        # 🔧 Remove added batch dimension, return original format
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
            'clean_latent_2x_indices': clean_latent_2x_indices_final,  # 🔧 Use actual indices (with -1 padding)
            'clean_latent_4x_indices': clean_latent_4x_indices_final,  # 🔧 Use actual indices (with -1 padding)
        }

    def create_sekai_pose_embeddings(self, cam_data, segment_info):
        """Create Sekai style pose embeddings"""
        cam_data_seq = cam_data['extrinsic']
        
        # Compute relative pose for all frames
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
        """🔧 Create OpenX style pose embeddings - similar to sekai but handles shorter sequences"""
        cam_data_seq = cam_data['extrinsic']
        
        # Compute relative pose for all frames - OpenX uses 4x interval
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
                # If no next frame, use identity matrix
                identity_cam = torch.eye(3, 4)
                relative_cams.append(identity_cam)
        
        if len(relative_cams) == 0:
            return None
            
        pose_embedding = torch.stack(relative_cams, dim=0)
        pose_embedding = rearrange(pose_embedding, 'b c d -> b (c d)')
        pose_embedding = pose_embedding.to(torch.bfloat16)

        return pose_embedding
    
    def create_spatialvid_pose_embeddings(self, cam_data, segment_info):
        """🔧 Create SpatialVid style pose embeddings - camera interval is 1 frame instead of 4 frames"""
        cam_data_seq = cam_data['extrinsic']   # N * 4 * 4
        
        # 🔧 Compute camera embedding for all frames (condition + target)
        # SpatialVid specific: Every 1 frame instead of 4 frames
        keyframe_original_idx = segment_info['keyframe_original_idx']
        
        relative_cams = []
        for idx in keyframe_original_idx:
            if idx + 1 < len(cam_data_seq):
                cam_prev = cam_data_seq[idx]
                cam_next = cam_data_seq[idx + 1]  # SpatialVid: Every 1 frame
                relative_cam = compute_relative_pose_matrix(cam_prev, cam_next)
                relative_cams.append(torch.as_tensor(relative_cam[:3, :]))
            else:
                # If no next frame, use zero motion
                identity_cam = torch.zeros(3, 4)
                relative_cams.append(identity_cam)
        
        if len(relative_cams) == 0:
            return None
            
        pose_embedding = torch.stack(relative_cams, dim=0)
        pose_embedding = rearrange(pose_embedding, 'b c d -> b (c d)')
        pose_embedding = pose_embedding.to(torch.bfloat16)

        return pose_embedding
               
    def create_nuscenes_pose_embeddings_framepack(self, scene_info, segment_info):
        """Create NuScenes style pose embeddings - FramePack version (simplified to 7D)"""
        keyframe_poses = scene_info['keyframe_poses']
        reference_keyframe_idx = segment_info['reference_keyframe_idx']
        target_keyframe_indices = segment_info['target_keyframe_indices']
        
        if reference_keyframe_idx >= len(keyframe_poses):
            return None
        
        reference_pose = keyframe_poses[reference_keyframe_idx]
        
        # Create pose embeddings for all frames (condition + target)
        start_frame = segment_info['start_frame']
        condition_end_compressed = start_frame + segment_info['condition_frames']
        target_end_compressed = condition_end_compressed + segment_info['target_frames']
        
        # Compressed keyframe indices
        compressed_keyframe_indices = [idx // self.time_compression_ratio for idx in scene_info['keyframe_indices']]
        
        # Find keyframes in the condition segment
        condition_keyframes_compressed = [idx for idx in compressed_keyframe_indices 
                                        if start_frame <= idx < condition_end_compressed]
        
        # Find corresponding original keyframe indices
        condition_keyframes_original_indices = []
        for compressed_idx in condition_keyframes_compressed:
            for i, comp_idx in enumerate(compressed_keyframe_indices):
                if comp_idx == compressed_idx:
                    condition_keyframes_original_indices.append(i)
                    break
        
        pose_vecs = []
        
        # Compute pose for condition frames
        for i in range(segment_info['condition_frames']):
            if not condition_keyframes_original_indices:
                translation = torch.zeros(3, dtype=torch.float32)
                rotation = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)
            else:
                # Assign pose for condition frames
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
            
            # 🔧 Simplified: Direct 7D [translation(3) + rotation(4)]
            pose_vec = torch.cat([translation, rotation], dim=0)  # [7D]
            pose_vecs.append(pose_vec)
        
        # Compute pose for target frames
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
                    
                    # 🔧 Simplified: Direct 7D [translation(3) + rotation(4)]
                    pose_vec = torch.cat([relative_translation, relative_rotation], dim=0)  # [7D]
                
                pose_vecs.append(pose_vec)
        
        if not pose_vecs:
            return None
        
        pose_sequence = torch.stack(pose_vecs, dim=0)  # [total_frames, 7]
        
        return pose_sequence

    # Modify create_pose_embeddings method
    def create_pose_embeddings(self, cam_data, segment_info, dataset_type, scene_info=None):
        """🔧 Create pose embeddings based on dataset type"""
        if dataset_type == 'nuscenes' and scene_info is not None:
            return self.create_nuscenes_pose_embeddings_framepack(scene_info, segment_info)
        elif dataset_type == 'spatialvid':  # 🔧 Added spatialvid handling
            return self.create_spatialvid_pose_embeddings(cam_data, segment_info)
        elif dataset_type == 'sekai':
            return self.create_sekai_pose_embeddings(cam_data, segment_info)
        elif dataset_type == 'openx':  # 🔧 Added openx handling
            return self.create_openx_pose_embeddings(cam_data, segment_info)

    def select_dynamic_segment(self, full_latents, dataset_type, scene_info=None):
        """🔧 Select different segment selection strategy based on dataset type"""
        if dataset_type == 'nuscenes' and scene_info is not None:
            return self.select_dynamic_segment_nuscenes(scene_info)
        else:
            # Original sekai method
            total_lens = full_latents.shape[1]
            
            min_condition_compressed = self.min_condition_frames // self.time_compression_ratio
            max_condition_compressed = self.max_condition_frames // self.time_compression_ratio
            target_frames_compressed = self.target_frames // self.time_compression_ratio
            max_condition_compressed = min(total_lens-target_frames_compressed-1, max_condition_compressed)

            # 🔧 New: spatialvid dataset 80% probability uses only the first frame as condition
            if dataset_type == 'spatialvid':
                ratio = random.random()
                if ratio < 0.4:  # 40% probability uses the first frame (actually first_latent.pth)
                    condition_frames_compressed = 1
                elif ratio < 0.9:  # 50% probability uses random history length
                    condition_frames_compressed = random.randint(min_condition_compressed, max_condition_compressed)
                else:  # 10% probability uses target_frames length
                    condition_frames_compressed = target_frames_compressed
            else:
                # Other datasets maintain original logic
                ratio = random.random()
                if ratio < 0.15:
                    condition_frames_compressed = 1
                elif 0.15 <= ratio < 0.9 or total_lens <= 2*target_frames_compressed + 1:
                    condition_frames_compressed = random.randint(min_condition_compressed, max_condition_compressed)
                else:
                    condition_frames_compressed = target_frames_compressed
            
            # Ensure enough frames
            min_required_frames = condition_frames_compressed + target_frames_compressed
            if total_lens < min_required_frames:
                return None
            
            start_frame_compressed = random.randint(0, total_lens - min_required_frames - 1)
            condition_end_compressed = start_frame_compressed + condition_frames_compressed
            target_end_compressed = condition_end_compressed + target_frames_compressed

            # FramePack style index handling
            latent_indices = torch.arange(condition_end_compressed, target_end_compressed)
            
            # 1x frames: Start frame + Last 1 frame
            clean_latent_indices_start = torch.tensor([start_frame_compressed])
            clean_latent_1x_indices = torch.tensor([condition_end_compressed - 1])
            clean_latent_indices = torch.cat([clean_latent_indices_start, clean_latent_1x_indices])
            
            # 🔧 2x frames: Determined by actual condition length
            if condition_frames_compressed >= 2:
                clean_latent_2x_start = max(start_frame_compressed, condition_end_compressed - 2-1)
                clean_latent_2x_indices = torch.arange(clean_latent_2x_start, condition_end_compressed-1)
            else:
                clean_latent_2x_indices = torch.tensor([], dtype=torch.long)
            
            # 🔧 4x frames: Determined by actual condition length, max 16 frames
            if condition_frames_compressed > 3:
                clean_4x_start = max(start_frame_compressed, condition_end_compressed - 16-3)
                clean_latent_4x_indices = torch.arange(clean_4x_start, condition_end_compressed-3)
            else:
                clean_latent_4x_indices = torch.tensor([], dtype=torch.long)
            
            # Corresponding original keyframe indices
            keyframe_original_idx = []
            for compressed_idx in range(start_frame_compressed, target_end_compressed):
                if dataset_type == 'spatialvid':
                    keyframe_original_idx.append(compressed_idx)
                elif dataset_type == 'openx' or 'sekai':
                    keyframe_original_idx.append(compressed_idx * 4)

            return {
                'start_frame': start_frame_compressed,
                'condition_frames': condition_frames_compressed,
                'target_frames': target_frames_compressed,
                'condition_range': (start_frame_compressed, condition_end_compressed),
                'target_range': (condition_end_compressed, target_end_compressed),
                
                # FramePack style indices
                'latent_indices': latent_indices,
                'clean_latent_indices': clean_latent_indices,
                'clean_latent_2x_indices': clean_latent_2x_indices,
                'clean_latent_4x_indices': clean_latent_4x_indices,
                
                'keyframe_original_idx': keyframe_original_idx,
                'original_condition_frames': condition_frames_compressed * self.time_compression_ratio,
                'original_target_frames': target_frames_compressed * self.time_compression_ratio,
                
                # 🔧 New: Flag whether to use first_latent
                'use_first_latent': dataset_type == 'spatialvid' and condition_frames_compressed == 1,
            }

    def __getitem__(self, index):
        while True:
            try:
                # Randomly select scene based on weight
                scene_idx = np.random.choice(len(self.scene_dirs), p=self.sampling_probs)
                scene_dir = self.scene_dirs[scene_idx]
                dataset_info = self.dataset_info[scene_dir]
                
                dataset_name = dataset_info['name']
                dataset_type = dataset_info['type']
                
                # 🔧 Load data based on dataset type
                scene_info = None
                if dataset_type == 'nuscenes':
                    scene_info_path = os.path.join(scene_dir, "scene_info.json")
                    if os.path.exists(scene_info_path):
                        with open(scene_info_path, 'r') as f:
                            scene_info = json.load(f)
                    
                    encoded_path = os.path.join(scene_dir, "encoded_video-480p.pth")
                    if not os.path.exists(encoded_path):
                        encoded_path = os.path.join(scene_dir, "encoded_video.pth")
                    
                    encoded_data = torch.load(encoded_path, weights_only=True, map_location="cpu")
                else:
                    # encoded_path = os.path.join(scene_dir, "encoded_video.pth")
                    encoded_path = scene_dir
                    encoded_data = torch.load(encoded_path, weights_only=False, map_location="cpu")
                
                full_latents = encoded_data['latents']
                if full_latents.shape[1] <= 10:
                    continue
                cam_data = encoded_data.get('cam_emb', encoded_data)
                
                # 🔧 Validate NuScenes latent frame count
                if dataset_type == 'nuscenes' and scene_info is not None:
                    expected_latent_frames = scene_info['total_frames'] // self.time_compression_ratio
                    actual_latent_frames = full_latents.shape[1]
                    
                    if abs(actual_latent_frames - expected_latent_frames) > 2:
                        print(f"⚠️ NuScenes Latent frame count mismatch, skipping sample")
                        continue
                
                # Use dataset-specific segment selection strategy
                segment_info = self.select_dynamic_segment(full_latents, dataset_type, scene_info)
                if segment_info is None:
                    continue
                
                # 🔧 New: For spatialvid, if using first_latent, load it
                if segment_info.get('use_first_latent', False):
                    # first_latent_path = os.path.join(scene_dir, "first_latent.pth")
                    first_latent_path = scene_dir.replace(
                        "SpatialVID_Wan2/","SpatialVID_Wan2_first4/"
                    ).replace(".pth", "_first4.pth")
                    if os.path.exists(first_latent_path):
                        first_latent_data = torch.load(first_latent_path, weights_only=False, map_location="cpu")
                        # first_latent.pth contains the encoded result of the first frame repeated 4 times
                        # Shape should be [C, 1, H, W] (because 4 frames are compressed to 1 frame by VAE)
                        first_latent = first_latent_data['latents_first4']  # [C, 1, H, W]
                        
                        # Replace the first frame of full_latents with first_latent
                        # Note: We keep other frames of full_latents unchanged, only replace the frame used as condition
                        full_latents[:, 0:1, :, :] = first_latent
                        
                        print(f"✅ SpatialVid: Using first_latent.pth as condition (40% probability)")
                    else:
                        print(f"⚠️ first_latent.pth does not exist: {first_latent_path}, using original latent")
                
                # Create dataset-specific pose embeddings
                all_camera_embeddings = self.create_pose_embeddings(cam_data, segment_info, dataset_type, scene_info)
                if all_camera_embeddings is None:
                    continue
                
                # Prepare FramePack style multi-scale inputs
                framepack_inputs = self.prepare_framepack_inputs(full_latents, segment_info)
                
                n = segment_info["condition_frames"]
                m = segment_info['target_frames']
                
                # Handle camera embedding with mask
                mask = torch.zeros(n+m, dtype=torch.float32)
                mask[:n] = 1.0
                mask = mask.view(-1, 1)
                
                if isinstance(all_camera_embeddings, torch.Tensor):
                    camera_with_mask = torch.cat([all_camera_embeddings, mask], dim=1)
                else:
                    camera_with_mask = torch.cat([all_camera_embeddings, mask], dim=1)
                
                result = {
                    # FramePack style multi-scale inputs
                    "latents": framepack_inputs['latents'],
                    "clean_latents": framepack_inputs['clean_latents'],
                    "clean_latents_2x": framepack_inputs['clean_latents_2x'],
                    "clean_latents_4x": framepack_inputs['clean_latents_4x'],
                    "latent_indices": framepack_inputs['latent_indices'],
                    "clean_latent_indices": framepack_inputs['clean_latent_indices'],
                    "clean_latent_2x_indices": framepack_inputs['clean_latent_2x_indices'],
                    "clean_latent_4x_indices": framepack_inputs['clean_latent_4x_indices'],
                    
                    # Camera data
                    "camera": camera_with_mask,
                    
                    # Other data
                    "prompt_emb": encoded_data["prompt_emb"],
                    "image_emb": encoded_data.get("image_emb", {}),
                    
                    # Metadata
                    "condition_frames": n,
                    "target_frames": m,
                    "scene_name": os.path.basename(scene_dir),
                    "dataset_name": dataset_name,
                    "dataset_type": dataset_type,
                    "original_condition_frames": segment_info['original_condition_frames'],
                    "original_target_frames": segment_info['original_target_frames'],
                    "use_first_latent": segment_info.get('use_first_latent', False),  # 🔧 Add flag
                }
                
                return result
                
            except Exception as e:
                print(f"Error loading sample: {e}")
                traceback.print_exc()
                continue

    def __len__(self):
        return self.steps_per_epoch

def replace_dit_model_in_manager():
    """Replace the DiT model class with the MoE version before model loading"""
    from diffsynth.models.wan_video_dit_moe import WanModelMoe
    from diffsynth.configs.model_config import model_loader_configs
    
    # Modify config in model_loader_configs
    for i, config in enumerate(model_loader_configs):
        keys_hash, keys_hash_with_shape, model_names, model_classes, model_resource = config
        
        # Check if wan_video_dit model is included
        if 'wan_video_dit' in model_names:
            new_model_names = []
            new_model_classes = []
            
            for name, cls in zip(model_names, model_classes):
                if name == 'wan_video_dit':
                    new_model_names.append(name)
                    new_model_classes.append(WanModelMoe)  # 🔧 Use MoE version
                    print(f"✅ Replaced model class: {name} -> WanModelMoe")
                else:
                    new_model_names.append(name)
                    new_model_classes.append(cls)
            
            # Update config
            model_loader_configs[i] = (keys_hash, keys_hash_with_shape, new_model_names, new_model_classes, model_resource)

class MultiDatasetLightningModelForTrain(pl.LightningModule):
    def __init__(
        self,
        dit_path,
        learning_rate=1e-5,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        resume_ckpt_path=None,
        # 🔧 MoE parameters
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
        model_manager.load_models(["/mnt/data/louis_crq/models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"])
        
        self.pipe = WanVideoAstraPipeline.from_model_manager(model_manager)
        self.pipe.scheduler.set_timesteps(1000, training=True)

        # Add FramePack's clean_x_embedder
        self.add_framepack_components()
        if self.use_moe:
            self.add_moe_components()

        # 🔧 Add camera encoder (MoE logic is already in wan_video_dit_moe.py)
        dim = self.pipe.dit.blocks[0].self_attn.q.weight.shape[0]
        for block in self.pipe.dit.blocks:
            # 🔧 Simplified: Only add traditional camera encoder, MoE logic in wan_video_dit_moe.py
            block.cam_encoder = nn.Linear(13, dim)
            block.projector = nn.Linear(dim, dim)
            block.cam_encoder.weight.data.zero_()
            block.cam_encoder.bias.data.zero_()
            block.projector.weight = nn.Parameter(torch.eye(dim))
            block.projector.bias = nn.Parameter(torch.zeros(dim))
        
        if resume_ckpt_path is not None:
            state_dict = torch.load(resume_ckpt_path, map_location="cpu")
            state_dict.pop("global_router.weight", None)
            state_dict.pop("global_router.bias", None)
            self.pipe.dit.load_state_dict(state_dict, strict=False)
            print('load checkpoint:', resume_ckpt_path)

        self.freeze_parameters()
        
        # 🔧 Training parameter setup
        for name, module in self.pipe.denoising_model().named_modules():
            if any(keyword in name for keyword in ["cam_encoder", "projector", "self_attn", "clean_x_embedder", 
                                                "moe", "sekai_processor", "nuscenes_processor","openx_processor"]):
                for param in module.parameters():
                    param.requires_grad = True
        
        self.learning_rate = learning_rate
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        
        # Create visualization directory
        self.vis_dir = "multi_dataset_dynamic/visualizations"
        os.makedirs(self.vis_dir, exist_ok=True)

    def add_moe_components(self):
        """🔧 Add MoE related components - Simplified, only add MoE to each block, global processor in WanModelMoe"""
        if not hasattr(self.pipe.dit, 'moe_config'):
            self.pipe.dit.moe_config = self.moe_config
            print("✅ Added MoE configuration to the model")
        self.pipe.dit.top_k = self.moe_config.get("top_k", 1)
        
        # Add MoE components to each block (modality processors are created globally in WanModelMoe)
        dim = self.pipe.dit.blocks[0].self_attn.q.weight.shape[0]
        unified_dim = self.moe_config.get("unified_dim", 30)
        num_experts = self.moe_config.get("num_experts", 4)
        from diffsynth.models.wan_video_dit_moe import MultiModalMoE, ModalityProcessor

        self.pipe.dit.sekai_processor = ModalityProcessor("sekai", 13, unified_dim)
        self.pipe.dit.nuscenes_processor = ModalityProcessor("nuscenes", 8, unified_dim)
        self.pipe.dit.openx_processor = ModalityProcessor("openx", 13, unified_dim)  # OpenX uses 13D input, similar to sekai but processed independently
        self.pipe.dit.global_router = nn.Linear(unified_dim, num_experts)

        for i, block in enumerate(self.pipe.dit.blocks):            
            # Only add MoE network to each block
            block.moe = MultiModalMoE(
                unified_dim=unified_dim,
                output_dim=dim,
                num_experts=self.moe_config.get("num_experts", 4),
                top_k=self.moe_config.get("top_k", 2)
            )
            
            print(f"✅ Block {i} added MoE component (unified_dim: {unified_dim}, experts: {self.moe_config.get('num_experts', 4)})")


    def add_framepack_components(self):
        """🔧 Add FramePack related components"""
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
            print("✅ Added FramePack's clean_x_embedder component")
        
    def freeze_parameters(self):
        self.pipe.requires_grad_(False)
        self.pipe.eval()
        self.pipe.denoising_model().train()

    def training_step(self, batch, batch_idx):
        """🔧 Multi-Dataset Training Step"""
        condition_frames = batch["condition_frames"][0].item()
        target_frames = batch["target_frames"][0].item()
        
        original_condition_frames = batch.get("original_condition_frames", [condition_frames * 4])[0]
        original_target_frames = batch.get("original_target_frames", [target_frames * 4])[0]

        dataset_name = batch.get("dataset_name", ["unknown"])[0]
        dataset_type = batch.get("dataset_type", ["sekai"])[0]
        scene_name = batch.get("scene_name", ["unknown"])[0]
        
        # Prepare input data
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
        
        # Index handling
        latent_indices = batch["latent_indices"].to(self.device)
        clean_latent_indices = batch["clean_latent_indices"].to(self.device) if batch["clean_latent_indices"].numel() > 0 else None
        clean_latent_2x_indices = batch["clean_latent_2x_indices"].to(self.device) if batch["clean_latent_2x_indices"].numel() > 0 else None
        clean_latent_4x_indices = batch["clean_latent_4x_indices"].to(self.device) if batch["clean_latent_4x_indices"].numel() > 0 else None
        
        # Camera embedding handling
        cam_emb = batch["camera"].to(self.device)
        
        # 🔧 Set modality_inputs based on dataset type
        if dataset_type == "sekai":
            modality_inputs = {"sekai": cam_emb}
        elif dataset_type == "spatialvid":  # 🔧 spatialvid uses sekai processor
            modality_inputs = {"sekai": cam_emb}  # Note: uses "sekai" key here
        elif dataset_type == "nuscenes":
            modality_inputs = {"nuscenes": cam_emb}
        elif dataset_type == "openx":  # 🔧 New: openx uses a dedicated processor
            modality_inputs = {"openx": cam_emb}
        else:
            modality_inputs = {"sekai": cam_emb}  # Default
        
        camera_dropout_prob = 0.05
        if random.random() < camera_dropout_prob:
            cam_emb = torch.zeros_like(cam_emb)
            # Also clear modality_inputs
            for key in modality_inputs:
                modality_inputs[key] = torch.zeros_like(modality_inputs[key])
            print(f"Applying camera dropout for CFG training (dataset: {dataset_name}, type: {dataset_type})")
        
        prompt_emb = batch["prompt_emb"]
        prompt_emb["context"] = prompt_emb["context"][0].to(self.device)
        image_emb = batch["image_emb"]

        if "clip_feature" in image_emb:
            image_emb["clip_feature"] = image_emb["clip_feature"][0].to(self.device)
        if "y" in image_emb:
            image_emb["y"] = image_emb["y"][0].to(self.device)

        # Loss calculation
        self.pipe.device = self.device
        noise = torch.randn_like(latents)
        timestep_id = torch.randint(0, self.pipe.scheduler.num_train_timesteps, (1,))
        timestep = self.pipe.scheduler.timesteps[timestep_id].to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
        
        # FramePack style noise handling
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
        
        noise_pred, specialization_loss = self.pipe.denoising_model()(
            noisy_latents, 
            timestep=timestep, 
            cam_emb=cam_emb,
            modality_inputs=modality_inputs,  # 🔧 Pass multi-modal inputs
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
        
        # Calculate loss
        # 🔧 Calculate total loss = Reconstruction loss + MoE specialization loss
        reconstruction_loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
        reconstruction_loss = reconstruction_loss * self.pipe.scheduler.training_weight(timestep)
        
        # 🔧 Add MoE specialization loss (Cross-Entropy loss)
        specialization_loss_weight = self.moe_config.get("moe_loss_weight", 0.1)
        total_loss = reconstruction_loss + specialization_loss_weight * specialization_loss
        
        print(f'\n loss info (step {self.global_step}):')
        print(f'  - diff loss: {reconstruction_loss.item():.6f}')
        print(f'  - MoE specification loss: {specialization_loss.item():.6f}')
        print(f'  - Expert loss weight: {specialization_loss_weight}')
        print(f'  - Total Loss: {total_loss.item():.6f}')
        
        # 🔧 Display expected expert mapping
        modality_to_expert = {
            "sekai": 0,
            "nuscenes": 1, 
            "openx": 2
        }
        expected_expert = modality_to_expert.get(dataset_type, 0)
        print(f'  - current modality: {dataset_type} -> expected expert: {expected_expert}')

        return total_loss

    def configure_optimizers(self):
        trainable_modules = filter(lambda p: p.requires_grad, self.pipe.denoising_model().parameters())
        optimizer = torch.optim.AdamW(trainable_modules, lr=self.learning_rate)
        return optimizer
    
    def on_save_checkpoint(self, checkpoint):
        checkpoint_dir = "/mnt/data/louis_crq/astra2/playground/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        current_step = self.global_step
        checkpoint.clear()
        
        state_dict = self.pipe.denoising_model().state_dict()
        torch.save(state_dict, os.path.join(checkpoint_dir, f"step{current_step}_origin_other_continue3.ckpt"))
        print(f"Saved MoE model checkpoint: step{current_step}_origin.ckpt")

def train_multi_dataset(args):
    """Train Multi-Dataset MoE Model"""
    
    # 🔧 Dataset configuration
    dataset_configs = [
        # {
        #     'name': 'sekai-drone',
        #     'paths': ['/share_zhuyixuan05/zhuyixuan05/sekai-game-drone'],
        #     'type': 'sekai',
        #     'weight': 0.7
        # },
        # {
        #     'name': 'sekai-walking',
        #     'paths': ['/share_zhuyixuan05/zhuyixuan05/sekai-game-walking'],
        #     'type': 'sekai',
        #     'weight': 0.7
        # },
        {
            'name': 'spatialvid',
            'manifest': ["/mnt/data/louis_crq/preprocessed_data/SpatialVID_Wan2/manifest.json"],
            'type': 'spatialvid',
            'weight': 1.0
        },
        # {
        #     'name': 'nuscenes',
        #     'paths': ['/share_zhuyixuan05/zhuyixuan05/nuscenes_video_generation_dynamic'],
        #     'type': 'nuscenes',
        #     'weight': 7.0
        # },
        # {
        #     'name': 'openx-fractal',
        #     'paths': ['/share_zhuyixuan05/zhuyixuan05/openx-fractal-encoded'],
        #     'type': 'openx',
        #     'weight': 1.1
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
    
    # 🔧 MoE configuration
    moe_config = {
        "unified_dim": args.unified_dim,  # New
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
        use_moe=True,  # Always use MoE
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
    parser.add_argument("--dit_path", type=str, default="/mnt/data/louis_crq/models/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors")
    parser.add_argument("--output_path", type=str, default="./")
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--steps_per_epoch", type=int, default=20000)
    parser.add_argument("--max_epochs", type=int, default=100000)
    parser.add_argument("--min_condition_frames", type=int, default=8, help="Minimum number of condition frames")
    parser.add_argument("--max_condition_frames", type=int, default=120, help="Maximum number of condition frames")
    parser.add_argument("--target_frames", type=int, default=32, help="Target number of frames")
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--training_strategy", type=str, default="ddp_find_unused_parameters_true")
    parser.add_argument("--use_gradient_checkpointing", default=False)
    parser.add_argument("--use_gradient_checkpointing_offload", action="store_true")
    parser.add_argument("--resume_ckpt_path", type=str, default="/share_zhuyixuan05/zhuyixuan05/ICLR2026/framepack_moe/step23000_origin_other_continue_con.ckpt")
    
    # 🔧 MoE parameters
    parser.add_argument("--unified_dim", type=int, default=25, help="Unified intermediate dimension")
    parser.add_argument("--moe_num_experts", type=int, default=3, help="Number of experts")
    parser.add_argument("--moe_top_k", type=int, default=1, help="Top-K experts")
    parser.add_argument("--moe_loss_weight", type=float, default=0.1, help="MoE loss weight")
    
    args = parser.parse_args()
    
    print("🔧 Multi-Dataset MoE Training Configuration:")
    print(f"  - Using wan_video_dit_moe.py as model")
    print(f"  - Unified Dimension: {args.unified_dim}")
    print(f"  - Number of Experts: {args.moe_num_experts}")
    print(f"  - Top-K: {args.moe_top_k}")
    print(f"  - MoE Loss Weight: {args.moe_loss_weight}")
    print("  - Datasets:")
    print("    - sekai-game-drone (sekai modality)")
    print("    - sekai-game-walking (sekai modality)")
    print("    - spatialvid (uses sekai modality processor)") 
    print("    - openx-fractal (uses sekai modality processor)")
    print(f"   - nuscenes (nuscenes modality)")
    
    train_multi_dataset(args)