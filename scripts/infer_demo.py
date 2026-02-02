import os
import sys
from pathlib import Path
from typing import Optional

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import imageio
import json
from diffsynth import WanVideoAstraPipeline, ModelManager
import argparse
from torchvision.transforms import v2
from einops import rearrange
from scipy.spatial.transform import Rotation as R
import random
import copy
from datetime import datetime

VALID_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
class InlineVideoEncoder:

    def __init__(self, pipe: WanVideoAstraPipeline, device="cuda"):
        self.device = getattr(pipe, "device", device)
        self.tiler_kwargs = {"tiled": True, "tile_size": (34, 34), "tile_stride": (18, 16)}
        self.frame_process = v2.Compose([
            v2.ToTensor(),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        self.pipe = pipe

    @staticmethod
    def _crop_and_resize(image: Image.Image) -> Image.Image:
        target_w, target_h = 832, 480
        return v2.functional.resize(
            image,
            (round(target_h), round(target_w)),
            interpolation=v2.InterpolationMode.BILINEAR,
        )

    def preprocess_frame(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB")
        image = self._crop_and_resize(image)
        return self.frame_process(image)

    def load_video_frames(self, video_path: Path) -> Optional[torch.Tensor]:
        reader = imageio.get_reader(str(video_path))
        frames = []
        for frame_data in reader:
            frame = Image.fromarray(frame_data)
            frames.append(self.preprocess_frame(frame))
        reader.close()

        if not frames:
            return None

        frames = torch.stack(frames, dim=0)
        return rearrange(frames, "T C H W -> C T H W")

    def encode_frames_to_latents(self, frames: torch.Tensor) -> torch.Tensor:
        frames = frames.unsqueeze(0).to(self.device, dtype=torch.bfloat16)
        with torch.no_grad():
            latents = self.pipe.encode_video(frames, **self.tiler_kwargs)[0]

        if latents.dim() == 5 and latents.shape[0] == 1:
            latents = latents.squeeze(0)
        return latents.cpu()
    
def image_to_frame_stack(
    image_path: Path, 
    encoder: InlineVideoEncoder, 
    repeat_count: int = 10
) -> torch.Tensor:
    """Repeat a single image into a tensor with specified number of frames, shape [C, T, H, W]"""
    if image_path.suffix.lower() not in VALID_IMAGE_EXTENSIONS:
        raise ValueError(f"Unsupported image format: {image_path.suffix}")

    image = Image.open(str(image_path))
    frame = encoder.preprocess_frame(image)
    frames = torch.stack([frame for _ in range(repeat_count)], dim=0)
    return rearrange(frames, "T C H W -> C T H W")


def load_or_encode_condition(
    condition_pth_path: Optional[str],
    condition_video: Optional[str],
    condition_image: Optional[str],
    start_frame: int,
    num_frames: int,
    device: str,
    pipe: WanVideoAstraPipeline,
) -> tuple[torch.Tensor, dict]:
    if condition_pth_path:
        return load_encoded_video_from_pth(condition_pth_path, start_frame, num_frames)

    encoder = InlineVideoEncoder(pipe=pipe, device=device)

    if condition_video:
        video_path = Path(condition_video).expanduser().resolve()
        if not video_path.exists():
            raise FileNotFoundError(f"File not Found: {video_path}")
        frames = encoder.load_video_frames(video_path)
        if frames is None:
            raise ValueError(f"no valid frames in {video_path}")
    elif condition_image:
        image_path = Path(condition_image).expanduser().resolve()
        if not image_path.exists():
            raise FileNotFoundError(f"File not Found: {image_path}")
        frames = image_to_frame_stack(image_path, encoder, repeat_count=10)
    else:
        raise ValueError("condition video or image is needed for video generation.")

    latents = encoder.encode_frames_to_latents(frames)
    encoded_data = {"latents": latents}

    if start_frame + num_frames > latents.shape[1]:
        raise ValueError(
            f"Not enough frames after encoding: requested {start_frame + num_frames}, available {latents.shape[1]}"
        )

    condition_latents = latents[:, start_frame:start_frame + num_frames, :, :]
    return condition_latents, encoded_data



def compute_relative_pose_matrix(pose1, pose2):
    """
    Compute relative pose between two consecutive frames, return 3x4 camera matrix [R_rel | t_rel]
    
    Args:
    pose1: Camera pose of frame i, shape (7,) array [tx1, ty1, tz1, qx1, qy1, qz1, qw1]
    pose2: Camera pose of frame i+1, shape (7,) array [tx2, ty2, tz2, qx2, qy2, qz2, qw2]
    
    Returns:
    relative_matrix: 3x4 relative pose matrix, 
    first 3 columns are rotation matrix R_rel, 
    last column is translation vector t_rel
    """
    # Separate translation vector and quaternion
    t1 = pose1[:3]  # Translation of frame i [tx1, ty1, tz1]
    q1 = pose1[3:]  # Quaternion of frame i [qx1, qy1, qz1, qw1]
    t2 = pose2[:3]  # Translation of frame i+1
    q2 = pose2[3:]  # Quaternion of frame i+1
    
    # 1. Compute relative rotation matrix R_rel
    rot1 = R.from_quat(q1)  # Rotation of frame i
    rot2 = R.from_quat(q2)  # Rotation of frame i+1
    rot_rel = rot2 * rot1.inv()  # Relative rotation = next frame rotation × inverse of current frame rotation
    R_rel = rot_rel.as_matrix()  # Convert to 3x3 matrix
    
    # 2. Compute relative translation vector t_rel
    R1_T = rot1.as_matrix().T  # Transpose of current frame rotation matrix (equivalent to inverse)
    t_rel = R1_T @ (t2 - t1)   # Relative translation = R1^T × (t2 - t1)
    
    # 3. Combine into 3x4 matrix [R_rel | t_rel]
    relative_matrix = np.hstack([R_rel, t_rel.reshape(3, 1)])
    
    return relative_matrix

def load_encoded_video_from_pth(pth_path, start_frame=0, num_frames=10):
    """Load pre-encoded video data from pth file"""
    print(f"Loading encoded video from {pth_path}")
    
    encoded_data = torch.load(pth_path, weights_only=False, map_location="cpu")
    full_latents = encoded_data['latents']  # [C, T, H, W]
    
    print(f"Full latents shape: {full_latents.shape}")
    print(f"Extracting frames {start_frame} to {start_frame + num_frames}")
    
    if start_frame + num_frames > full_latents.shape[1]:
        raise ValueError(f"Not enough frames: requested {start_frame + num_frames}, available {full_latents.shape[1]}")
    
    condition_latents = full_latents[:, start_frame:start_frame + num_frames, :, :]
    
    print(f"✅ Extracted condition latents shape: {condition_latents.shape}")
    
    return condition_latents, encoded_data

def compute_relative_pose(pose_a, pose_b, use_torch=False):
    """Compute relative pose matrix of camera B with respect to camera A"""
    assert pose_a.shape == (4, 4), f"Camera A extrinsic matrix should be (4,4), got {pose_a.shape}"
    assert pose_b.shape == (4, 4), f"Camera B extrinsic matrix should be (4,4), got {pose_b.shape}"
    
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
    """Replace DiT model class with MoE version"""
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
                    # print(f"Replaced model class: {name} -> WanModelMoe")
                else:
                    new_model_names.append(name)
                    new_model_classes.append(cls)
            
            model_loader_configs[i] = (keys_hash, keys_hash_with_shape, new_model_names, new_model_classes, model_resource)


def add_framepack_components(dit_model):
    """Add FramePack related components"""
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
        
        # print("✅ Added FramePack clean_x_embedder component")


def add_moe_components(dit_model, moe_config):
    """Add MoE related components - corrected version"""
    if not hasattr(dit_model, 'moe_config'):
        dit_model.moe_config = moe_config
        print("Added MoE config to model")
    dit_model.top_k = moe_config.get("top_k", 1)

    # Dynamically add MoE components for each block
    dim = dit_model.blocks[0].self_attn.q.weight.shape[0]
    unified_dim = moe_config.get("unified_dim", 25)
    num_experts = moe_config.get("num_experts", 4)
    from diffsynth.models.wan_video_dit_moe import ModalityProcessor, MultiModalMoE
    dit_model.sekai_processor = ModalityProcessor("sekai", 13, unified_dim)
    dit_model.nuscenes_processor = ModalityProcessor("nuscenes", 8, unified_dim)
    dit_model.openx_processor = ModalityProcessor("openx", 13, unified_dim)  # OpenX uses 13-dim input, similar to sekai but handled independently
    dit_model.global_router = nn.Linear(unified_dim, num_experts)


    for i, block in enumerate(dit_model.blocks):
        # MoE network - input unified_dim, output dim
        block.moe = MultiModalMoE(
            unified_dim=unified_dim,
            output_dim=dim,  # Output dimension matches transformer block dim
            num_experts=moe_config.get("num_experts", 4),
            top_k=moe_config.get("top_k", 2)
        )
        
        # print(f"✅ Block {i} added MoE component (unified_dim: {unified_dim}, experts: {moe_config.get('num_experts', 4)})")


def generate_sekai_camera_embeddings_sliding(
    cam_data, 
    start_frame, 
    initial_condition_frames, 
    new_frames, 
    total_generated, 
    use_real_poses=True,
    cam_type=1):
    """
    Generate camera embeddings for Sekai dataset - sliding window version
    
    Args:
        cam_data: Dictionary containing Sekai camera extrinsic parameters, key 'extrinsic' corresponds to an N*4*4 numpy array
        start_frame: Current generation start frame index
        initial_condition_frames: Initial condition frame count
        new_frames: Number of new frames to generate this time
        total_generated: Total frames already generated
        use_real_poses: Whether to use real Sekai camera poses
        cam_type: Camera type for synthetic trajectory generation, default 1
        
    Returns:
        camera_embedding: Torch tensor of shape (M, 3*4 + 1), where M is the total number of generated frames
    """
    time_compression_ratio = 4
    
    # Calculate the actual number of camera frames needed for FramePack
    # 1 initial frame + 16 frames 4x + 2 frames 2x + 1 frame 1x + new_frames
    framepack_needed_frames = 1 + 16 + 2 + 1 + new_frames
    
    if use_real_poses and cam_data is not None and 'extrinsic' in cam_data:
        print("🔧 Using real Sekai camera data")
        cam_extrinsic = cam_data['extrinsic']
        
        # Ensure generating a sufficiently long camera sequence
        max_needed_frames = max(
            start_frame + initial_condition_frames + new_frames,
            framepack_needed_frames,
            30
        )
        
        print(f"🔧 Calculating Sekai camera sequence length:")
        print(f"  - Basic requirement: {start_frame + initial_condition_frames + new_frames}")
        print(f"  - FramePack requirement: {framepack_needed_frames}")
        print(f"  - Final generation: {max_needed_frames}")
        
        relative_poses = []
        for i in range(max_needed_frames):
            # Calculate the position of the current frame in the original sequence
            frame_idx = i * time_compression_ratio
            next_frame_idx = frame_idx + time_compression_ratio
            
            if next_frame_idx < len(cam_extrinsic):
                cam_prev = cam_extrinsic[frame_idx]
                cam_next = cam_extrinsic[next_frame_idx]
                relative_pose = compute_relative_pose(cam_prev, cam_next)
                relative_poses.append(torch.as_tensor(relative_pose[:3, :]))
            else:
                # Out of range, use zero motion
                print(f"⚠️ Frame {frame_idx} exceeds camera data range, using zero motion")
                relative_poses.append(torch.zeros(3, 4))
        
        pose_embedding = torch.stack(relative_poses, dim=0)
        pose_embedding = rearrange(pose_embedding, 'b c d -> b (c d)')
        
        # Create mask sequence of corresponding length
        mask = torch.zeros(max_needed_frames, 1, dtype=torch.float32)
        # Mark from start_frame to start_frame+initial_condition_frames as condition
        condition_end = min(start_frame + initial_condition_frames, max_needed_frames)
        mask[start_frame:condition_end] = 1.0
        
        camera_embedding = torch.cat([pose_embedding, mask], dim=1)
        print(f"🔧 Sekai real camera embedding shape: {camera_embedding.shape}")
        return camera_embedding.to(torch.bfloat16)
        
    else:
        # Ensure generating a sufficiently long camera sequence
        max_needed_frames = max(
            start_frame + initial_condition_frames + new_frames, 
            framepack_needed_frames, 
            30)
            
        print(f"🔧 Generating Sekai synthetic camera frames: {max_needed_frames}")
        
        CONDITION_FRAMES = initial_condition_frames
        STAGE_1 = new_frames//2
        STAGE_2 = new_frames - STAGE_1
        
        if cam_type==1:
            print("--------------- FORWARD MODE ---------------")
            relative_poses = []
            for i in range(max_needed_frames):
                if i < CONDITION_FRAMES:
                    # Input condition frames default to zero motion camera pose
                    pose = np.eye(4, dtype=np.float32)
                elif i < CONDITION_FRAMES+STAGE_1+STAGE_2:
                    # Forward
                    forward_speed = 0.03

                    pose = np.eye(4, dtype=np.float32)
                    pose[2, 3] = -forward_speed
                else:
                    # The part beyond condition frames and target frames remains stationary
                    pose = np.eye(4, dtype=np.float32)
                
                relative_pose = pose[:3, :]
                relative_poses.append(torch.as_tensor(relative_pose))
        
        elif cam_type==2:
            print("--------------- LEFT TURNING MODE ---------------")
            relative_poses = []
            for i in range(max_needed_frames):
                if i < CONDITION_FRAMES:
                    # Input condition frames default to zero motion camera pose
                    pose = np.eye(4, dtype=np.float32)
                elif i < CONDITION_FRAMES+STAGE_1+STAGE_2:
                    # Left turn
                    yaw_per_frame = 0.03

                    # Rotation matrix
                    cos_yaw = np.cos(yaw_per_frame)
                    sin_yaw = np.sin(yaw_per_frame)
                    
                    # Forward
                    forward_speed = 0.00

                    pose = np.eye(4, dtype=np.float32)
                    
                    pose[0, 0] = cos_yaw
                    pose[0, 2] = sin_yaw
                    pose[2, 0] = -sin_yaw
                    pose[2, 2] = cos_yaw
                    pose[2, 3] = -forward_speed
                else:
                    # The part beyond condition frames and target frames remains stationary
                    pose = np.eye(4, dtype=np.float32)
                
                relative_pose = pose[:3, :]
                relative_poses.append(torch.as_tensor(relative_pose))
        
        elif cam_type==3:
            print("--------------- RIGHT TURNING MODE ---------------")
            relative_poses = []
            for i in range(max_needed_frames):
                if i < CONDITION_FRAMES:
                    # Input condition frames default to zero motion camera pose
                    pose = np.eye(4, dtype=np.float32)
                elif i < CONDITION_FRAMES+STAGE_1+STAGE_2:
                    # Right turn
                    yaw_per_frame = -0.03

                    # Rotation matrix
                    cos_yaw = np.cos(yaw_per_frame)
                    sin_yaw = np.sin(yaw_per_frame)
                    
                    # Forward
                    forward_speed = 0.00

                    pose = np.eye(4, dtype=np.float32)
                    
                    pose[0, 0] = cos_yaw
                    pose[0, 2] = sin_yaw
                    pose[2, 0] = -sin_yaw
                    pose[2, 2] = cos_yaw
                    pose[2, 3] = -forward_speed
                else:
                    # The part beyond condition frames and target frames remains stationary
                    pose = np.eye(4, dtype=np.float32)
                
                relative_pose = pose[:3, :]
                relative_poses.append(torch.as_tensor(relative_pose))
        
        elif cam_type==4:
            print("--------------- FORWARD LEFT MODE ---------------")
            relative_poses = []
            for i in range(max_needed_frames):
                if i < CONDITION_FRAMES:
                    # Input condition frames default to zero motion camera pose
                    pose = np.eye(4, dtype=np.float32)
                elif i < CONDITION_FRAMES+STAGE_1+STAGE_2:
                    # Left turn
                    yaw_per_frame = 0.03

                    # Rotation matrix
                    cos_yaw = np.cos(yaw_per_frame)
                    sin_yaw = np.sin(yaw_per_frame)
                    
                    # Forward
                    forward_speed = 0.03

                    pose = np.eye(4, dtype=np.float32)
                    
                    pose[0, 0] = cos_yaw
                    pose[0, 2] = sin_yaw
                    pose[2, 0] = -sin_yaw
                    pose[2, 2] = cos_yaw
                    pose[2, 3] = -forward_speed
                
                else:
                    # The part beyond condition frames and target frames remains stationary
                    pose = np.eye(4, dtype=np.float32)
                    
                relative_pose = pose[:3, :]
                relative_poses.append(torch.as_tensor(relative_pose))
        
        elif cam_type==5:
            print("--------------- FORWARD RIGHT MODE ---------------")
            relative_poses = []
            for i in range(max_needed_frames):
                if i < CONDITION_FRAMES:
                    # Input condition frames default to zero motion camera pose
                    pose = np.eye(4, dtype=np.float32)
                elif i < CONDITION_FRAMES+STAGE_1+STAGE_2:
                    # Right turn
                    yaw_per_frame = -0.03

                    # Rotation matrix
                    cos_yaw = np.cos(yaw_per_frame)
                    sin_yaw = np.sin(yaw_per_frame)
                    
                    # Forward
                    forward_speed = 0.03

                    pose = np.eye(4, dtype=np.float32)
                    
                    pose[0, 0] = cos_yaw
                    pose[0, 2] = sin_yaw
                    pose[2, 0] = -sin_yaw
                    pose[2, 2] = cos_yaw
                    pose[2, 3] = -forward_speed
                
                else:
                    # The part beyond condition frames and target frames remains stationary
                    pose = np.eye(4, dtype=np.float32)
                    
                relative_pose = pose[:3, :]
                relative_poses.append(torch.as_tensor(relative_pose))
        
        elif cam_type==6:
            print("--------------- S CURVE MODE ---------------")
            relative_poses = []
            for i in range(max_needed_frames):
                if i < CONDITION_FRAMES:
                    # Input condition frames default to zero motion camera pose
                    pose = np.eye(4, dtype=np.float32)
                elif i < CONDITION_FRAMES+STAGE_1:
                    # Left turn
                    yaw_per_frame = 0.03

                    # Rotation matrix
                    cos_yaw = np.cos(yaw_per_frame)
                    sin_yaw = np.sin(yaw_per_frame)
                    
                    # Forward
                    forward_speed = 0.03

                    pose = np.eye(4, dtype=np.float32)
                    
                    pose[0, 0] = cos_yaw
                    pose[0, 2] = sin_yaw
                    pose[2, 0] = -sin_yaw
                    pose[2, 2] = cos_yaw
                    pose[2, 3] = -forward_speed
                    
                elif i < CONDITION_FRAMES+STAGE_1+STAGE_2:
                    # Right turn
                    yaw_per_frame = -0.03

                    # Rotation matrix
                    cos_yaw = np.cos(yaw_per_frame)
                    sin_yaw = np.sin(yaw_per_frame)
                    
                    # Forward
                    forward_speed = 0.03
                    # Slight left drift to maintain inertia
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
                    # The part beyond condition frames and target frames remains stationary
                    pose = np.eye(4, dtype=np.float32)
                    
                relative_pose = pose[:3, :]
                relative_poses.append(torch.as_tensor(relative_pose))
                    
        elif cam_type==7:
            print("--------------- LEFT RIGHT MODE ---------------")
            relative_poses = []
            for i in range(max_needed_frames):
                if i < CONDITION_FRAMES:
                    # Input condition frames default to zero motion camera pose
                    pose = np.eye(4, dtype=np.float32)
                elif i < CONDITION_FRAMES+STAGE_1:
                    # Left turn
                    yaw_per_frame = 0.03

                    # Rotation matrix
                    cos_yaw = np.cos(yaw_per_frame)
                    sin_yaw = np.sin(yaw_per_frame)
                    
                    # Forward
                    forward_speed = 0.00

                    pose = np.eye(4, dtype=np.float32)
                    
                    pose[0, 0] = cos_yaw
                    pose[0, 2] = sin_yaw
                    pose[2, 0] = -sin_yaw
                    pose[2, 2] = cos_yaw
                    pose[2, 3] = -forward_speed
                    
                elif i < CONDITION_FRAMES+STAGE_1+STAGE_2:
                    # Right turn
                    yaw_per_frame = -0.03

                    # Rotation matrix
                    cos_yaw = np.cos(yaw_per_frame)
                    sin_yaw = np.sin(yaw_per_frame)
                    
                    # Forward
                    forward_speed = 0.00

                    pose = np.eye(4, dtype=np.float32)
                    
                    pose[0, 0] = cos_yaw
                    pose[0, 2] = sin_yaw
                    pose[2, 0] = -sin_yaw
                    pose[2, 2] = cos_yaw
                    pose[2, 3] = -forward_speed
                
                else:
                    # The part beyond condition frames and target frames remains stationary
                    pose = np.eye(4, dtype=np.float32)
                    
                relative_pose = pose[:3, :]
                relative_poses.append(torch.as_tensor(relative_pose))
                
        else:
            raise ValueError(f"Not Defined Camera Type: {cam_type}")
            
        pose_embedding = torch.stack(relative_poses, dim=0)
        pose_embedding = rearrange(pose_embedding, 'b c d -> b (c d)')
        
        # Create mask sequence of corresponding length
        mask = torch.zeros(max_needed_frames, 1, dtype=torch.float32)
        condition_end = min(start_frame + initial_condition_frames + 1, max_needed_frames)
        mask[start_frame:condition_end] = 1.0
        
        camera_embedding = torch.cat([pose_embedding, mask], dim=1)
        print(f"🔧 Sekai synthetic camera embedding shape: {camera_embedding.shape}")
        return camera_embedding.to(torch.bfloat16)


def generate_openx_camera_embeddings_sliding(
    encoded_data, start_frame, initial_condition_frames, new_frames, use_real_poses):
    """Generate camera embeddings for OpenX dataset - sliding window version"""
    time_compression_ratio = 4
    
    # Calculate the actual number of camera frames needed for FramePack
    framepack_needed_frames = 1 + 16 + 2 + 1 + new_frames
    
    if use_real_poses and encoded_data is not None and 'cam_emb' in encoded_data and 'extrinsic' in encoded_data['cam_emb']:
        print("🔧 Using OpenX real camera data")
        cam_extrinsic = encoded_data['cam_emb']['extrinsic']
        
        # Ensure generating a sufficiently long camera sequence
        max_needed_frames = max(
            start_frame + initial_condition_frames + new_frames,
            framepack_needed_frames,
            30
        )
        
        print(f"🔧 Calculating OpenX camera sequence length:")
        print(f"  - Basic requirement: {start_frame + initial_condition_frames + new_frames}")
        print(f"  - FramePack requirement: {framepack_needed_frames}")
        print(f"  - Final generation: {max_needed_frames}")
        
        relative_poses = []
        for i in range(max_needed_frames):
            # OpenX uses 4x interval, similar to sekai but handles shorter sequences
            frame_idx = i * time_compression_ratio
            next_frame_idx = frame_idx + time_compression_ratio
            
            if next_frame_idx < len(cam_extrinsic):
                cam_prev = cam_extrinsic[frame_idx]
                cam_next = cam_extrinsic[next_frame_idx]
                relative_pose = compute_relative_pose(cam_prev, cam_next)
                relative_poses.append(torch.as_tensor(relative_pose[:3, :]))
            else:
                # Out of range, use zero motion
                print(f"⚠️ Frame {frame_idx} exceeds OpenX camera data range, using zero motion")
                relative_poses.append(torch.zeros(3, 4))
        
        pose_embedding = torch.stack(relative_poses, dim=0)
        pose_embedding = rearrange(pose_embedding, 'b c d -> b (c d)')
        
        # Create mask sequence of corresponding length
        mask = torch.zeros(max_needed_frames, 1, dtype=torch.float32)
        # Mark from start_frame to start_frame + initial_condition_frames as condition
        condition_end = min(start_frame + initial_condition_frames, max_needed_frames)
        mask[start_frame:condition_end] = 1.0
        
        camera_embedding = torch.cat([pose_embedding, mask], dim=1)
        print(f"🔧 OpenX real camera embedding shape: {camera_embedding.shape}")
        return camera_embedding.to(torch.bfloat16)
        
    else:
        print("🔧 Using OpenX synthetic camera data")
        
        max_needed_frames = max(
            start_frame + initial_condition_frames + new_frames,
            framepack_needed_frames,
            30
        )
        
        print(f"🔧 Generating OpenX synthetic camera frames: {max_needed_frames}")
        relative_poses = []
        for i in range(max_needed_frames):
            # OpenX robot operation motion mode - smaller motion amplitude
            # Simulate fine operation motion of robot arm
            roll_per_frame = 0.02   # Slight roll
            pitch_per_frame = 0.01  # Slight pitch
            yaw_per_frame = 0.015   # Slight yaw
            forward_speed = 0.003   # Slower forward speed
            
            pose = np.eye(4, dtype=np.float32)
            
            # Compound rotation - simulate complex motion of robot arm
            # Rotate around X-axis (roll)
            cos_roll = np.cos(roll_per_frame)
            sin_roll = np.sin(roll_per_frame)
            # Rotate around Y-axis (pitch)
            cos_pitch = np.cos(pitch_per_frame)
            sin_pitch = np.sin(pitch_per_frame)
            # Rotate around Z-axis (yaw)
            cos_yaw = np.cos(yaw_per_frame)
            sin_yaw = np.sin(yaw_per_frame)
            
            # Simplified compound rotation matrix (ZYX order)
            pose[0, 0] = cos_yaw * cos_pitch
            pose[0, 1] = cos_yaw * sin_pitch * sin_roll - sin_yaw * cos_roll
            pose[0, 2] = cos_yaw * sin_pitch * cos_roll + sin_yaw * sin_roll
            pose[1, 0] = sin_yaw * cos_pitch
            pose[1, 1] = sin_yaw * sin_pitch * sin_roll + cos_yaw * cos_roll
            pose[1, 2] = sin_yaw * sin_pitch * cos_roll - cos_yaw * sin_roll
            pose[2, 0] = -sin_pitch
            pose[2, 1] = cos_pitch * sin_roll
            pose[2, 2] = cos_pitch * cos_roll
            
            # Translation - simulate fine movement of robot operation
            pose[0, 3] = forward_speed * 0.5   # Slight movement in X direction
            pose[1, 3] = forward_speed * 0.3   # Slight movement in Y direction
            pose[2, 3] = -forward_speed        # Main movement in Z direction (depth)
            
            relative_pose = pose[:3, :]
            relative_poses.append(torch.as_tensor(relative_pose))
        
        pose_embedding = torch.stack(relative_poses, dim=0)
        pose_embedding = rearrange(pose_embedding, 'b c d -> b (c d)')
        
        # Create mask sequence of corresponding length
        mask = torch.zeros(max_needed_frames, 1, dtype=torch.float32)
        condition_end = min(start_frame + initial_condition_frames, max_needed_frames)
        mask[start_frame:condition_end] = 1.0
        
        camera_embedding = torch.cat([pose_embedding, mask], dim=1)
        print(f"🔧 OpenX synthetic camera embedding shape: {camera_embedding.shape}")
        return camera_embedding.to(torch.bfloat16)


def generate_nuscenes_camera_embeddings_sliding(
    scene_info, start_frame, initial_condition_frames, new_frames):
    """
    Generate camera embeddings for NuScenes dataset - sliding window version
    
    corrected version, consistent with train_moe.py
    """
    time_compression_ratio = 4
    
    # Calculate the actual number of camera frames needed for FramePack
    framepack_needed_frames = 1 + 16 + 2 + 1 + new_frames
    
    if scene_info is not None and 'keyframe_poses' in scene_info:
        print("🔧 Using NuScenes real pose data")
        keyframe_poses = scene_info['keyframe_poses']
        
        if len(keyframe_poses) == 0:
            print("⚠️ NuScenes keyframe_poses is empty, using zero pose")
            max_needed_frames = max(framepack_needed_frames, 30)
            
            pose_sequence = torch.zeros(max_needed_frames, 7, dtype=torch.float32)
            
            mask = torch.zeros(max_needed_frames, 1, dtype=torch.float32)
            condition_end = min(start_frame + initial_condition_frames, max_needed_frames)
            mask[start_frame:condition_end] = 1.0
            
            camera_embedding = torch.cat([pose_sequence, mask], dim=1)  # [max_needed_frames, 8]
            print(f"🔧 NuScenes zero pose embedding shape: {camera_embedding.shape}")
            return camera_embedding.to(torch.bfloat16)
        
        # Use first pose as reference
        reference_pose = keyframe_poses[0]
        
        max_needed_frames = max(framepack_needed_frames, 30)
        
        pose_vecs = []
        for i in range(max_needed_frames):
            if i < len(keyframe_poses):
                current_pose = keyframe_poses[i]
                
                # Calculate relative displacement
                translation = torch.tensor(
                    np.array(current_pose['translation']) - np.array(reference_pose['translation']),
                    dtype=torch.float32
                )
                
                # Calculate relative rotation (simplified version)
                rotation = torch.tensor(current_pose['rotation'], dtype=torch.float32)
                
                pose_vec = torch.cat([translation, rotation], dim=0)  # [7D]
            else:
                # Out of range, use zero pose
                pose_vec = torch.cat([
                    torch.zeros(3, dtype=torch.float32),
                    torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)
                ], dim=0)  # [7D]
            
            pose_vecs.append(pose_vec)
        
        pose_sequence = torch.stack(pose_vecs, dim=0)  # [max_needed_frames, 7]
        
        # Create mask
        mask = torch.zeros(max_needed_frames, 1, dtype=torch.float32)
        condition_end = min(start_frame + initial_condition_frames, max_needed_frames)
        mask[start_frame:condition_end] = 1.0
        
        camera_embedding = torch.cat([pose_sequence, mask], dim=1)  # [max_needed_frames, 8]
        print(f"🔧 NuScenes real pose embedding shape: {camera_embedding.shape}")
        return camera_embedding.to(torch.bfloat16)
    
    else:
        print("🔧 Using NuScenes synthetic pose data")
        max_needed_frames = max(framepack_needed_frames, 30)
        
        # Create synthetic motion sequence
        pose_vecs = []
        for i in range(max_needed_frames):
            # Left turn motion mode - similar to left turns in city driving
            angle = i * 0.04  # Rotate 0.08 radians per frame (slightly slower turn)
            radius = 15.0     # Larger turning radius, more suitable for car turns
            
            # Calculate position on circular arc trajectory
            x = radius * np.sin(angle)
            y = 0.0  # Keep horizontal plane motion
            z = radius * (1 - np.cos(angle))
            
            translation = torch.tensor([x, y, z], dtype=torch.float32)
            
            # Vehicle orientation - always along trajectory tangent direction
            yaw = angle + np.pi/2  # Yaw angle relative to initial forward direction
            # Quaternion representation of rotation around Y-axis
            rotation = torch.tensor([
                np.cos(yaw/2),  # w (real part)
                0.0,            # x
                0.0,            # y  
                np.sin(yaw/2)   # z (imaginary part, around Y-axis)
            ], dtype=torch.float32)
            
            pose_vec = torch.cat([translation, rotation], dim=0)  # [7D: tx,ty,tz,qw,qx,qy,qz]
            pose_vecs.append(pose_vec)
        
        pose_sequence = torch.stack(pose_vecs, dim=0)
        
        # Create mask
        mask = torch.zeros(max_needed_frames, 1, dtype=torch.float32)
        condition_end = min(start_frame + initial_condition_frames, max_needed_frames)
        mask[start_frame:condition_end] = 1.0
        
        camera_embedding = torch.cat([pose_sequence, mask], dim=1)  # [max_needed_frames, 8]
        print(f"🔧 NuScenes synthetic left turn pose embedding shape: {camera_embedding.shape}")
        return camera_embedding.to(torch.bfloat16)

def prepare_framepack_sliding_window_with_camera_moe(
    history_latents, 
    target_frames_to_generate, 
    camera_embedding_full,
    modality_type
):
    """FramePack sliding window mechanism - MoE version"""
    # history_latents: [C, T, H, W] current history latents
    C, T, H, W = history_latents.shape
    
    # Fixed index structure (this determines the number of camera frames needed)
    # 1 start frame + 16 frames 4x + 2 frames 2x + 1 frame 1x + target_frames_to_generate
    total_indices_length = 1 + 16 + 2 + 1 + target_frames_to_generate
    
    start_frame = max(T - 20, 0)
    indices = torch.arange(start_frame, start_frame + total_indices_length)
    available_frames = min(T, 20)
    start_pos = 20 - available_frames # populate the nearest positions to target frames first
    indices[0] = indices[start_pos] # NOTE: always start from the first valid latent
    if start_pos > 1:
        indices[1:start_pos] = -1 # mark invalid latents (zero-latent)
    
    split_sizes = [1, 16, 2, 1, target_frames_to_generate]
    clean_latent_indices_start, clean_latent_4x_indices, clean_latent_2x_indices, clean_latent_1x_indices, latent_indices = \
        indices.split(split_sizes, dim=0)
    clean_latent_indices = torch.cat([clean_latent_indices_start, clean_latent_1x_indices], dim=0)
    
    # Process latents
    clean_latents_combined = torch.zeros(C, 20, H, W, dtype=history_latents.dtype, device=history_latents.device)
    clean_latents_combined[:, start_pos:, :, :] = history_latents[:, -available_frames:, :, :]
    start_latent     = clean_latents_combined[:, start_pos:start_pos+1, :, :]
    clean_latents_4x = clean_latents_combined[:, 1:17, :, :]
    clean_latents_2x = clean_latents_combined[:, 17:19, :, :]
    clean_latents_1x = clean_latents_combined[:, 19:20, :, :]
    
    clean_latents = torch.cat([start_latent, clean_latents_1x], dim=1)
    
    # Process cam_emb
    actual_needed_frames = T + target_frames_to_generate
    if camera_embedding_full.shape[0] < actual_needed_frames:
        print(f"⚠️ camera_embedding length insufficient, performing zero padding...")
        print(f"- Current length {camera_embedding_full.shape[0]}")
        print(f"- Required length {actual_needed_frames}")
        
        shortage = actual_needed_frames - camera_embedding_full.shape[0]
        zero_motions = torch.eye(3, 4).unsqueeze(0).repeat(shortage, 1, 1) # use identity rather than zero poses
        padding = rearrange(zero_motions, 'b c d -> b (c d)')
        camera_embedding_full = torch.cat([camera_embedding_full, padding], dim=0)
    
    # Select corresponding part from complete camera sequence
    combined_camera = torch.zeros(
        actual_needed_frames,
        camera_embedding_full.shape[1],
        dtype=camera_embedding_full.dtype,
        device=camera_embedding_full.device
    )
    combined_camera = camera_embedding_full[0:actual_needed_frames, :].clone()
    
    # Reset mask according to current history length
    combined_camera[:, -1] = 0.0  # First set all to target (0)
    combined_camera[0:T, -1] = 1.0  # Mark valid clean latents as condition

    
    print(f"🔧 MoE Camera mask update:")
    print(f"  - History frames: {T}")
    print(f"  - Valid condition frames: {available_frames if T > 0 else 0}")
    print(f"  - Modality type: {modality_type}")
    
    return {
        'latent_indices': latent_indices,
        'clean_latents': clean_latents,
        'clean_latents_2x': clean_latents_2x,
        'clean_latents_4x': clean_latents_4x,
        'clean_latent_indices': clean_latent_indices,
        'clean_latent_2x_indices': clean_latent_2x_indices,
        'clean_latent_4x_indices': clean_latent_4x_indices,
        'camera_embedding': combined_camera,
        'modality_type': modality_type,  # Added modality type information
        'current_length': T,
        'next_length': T + target_frames_to_generate
    }

def overlay_controls(frame_img, pose_vec, icons):
    """
    Overlay control icons (WASD and arrows) on frame based on camera pose
    pose_vec: 12 elements (flattened 3x4 matrix) + mask
    """
    if pose_vec is None or np.all(pose_vec[:12] == 0):
        return frame_img
        
    # Extract translation vector (based on flattened 3x4 matrix indices)
    # [r00, r01, r02, tx, r10, r11, r12, ty, r20, r21, r22, tz]
    tx = pose_vec[3]
    # ty = pose_vec[7]
    tz = pose_vec[11]
    
    # Extract rotation (yaw and pitch)
    # Yaw: around Y axis. sin(yaw) = r02, cos(yaw) = r00
    r00 = pose_vec[0]
    r02 = pose_vec[2]
    yaw = np.arctan2(r02, r00)
    
    # Pitch: around X axis. sin(pitch) = -r12, cos(pitch) = r22
    r12 = pose_vec[6]
    r22 = pose_vec[10]
    pitch = np.arctan2(-r12, r22)
    
    # Threshold for key activation
    TRANS_THRESH = 0.01
    ROT_THRESH = 0.005
    
    # Determine key states
    # Translation (WASD)
    # Assume -Z is forward, +X is right
    is_forward = tz < -TRANS_THRESH
    is_backward = tz > TRANS_THRESH
    is_left = tx < -TRANS_THRESH
    is_right = tx > TRANS_THRESH
    
    # Rotation (arrows)
    # Yaw: + is left, - is right
    is_turn_left = yaw > ROT_THRESH
    is_turn_right = yaw < -ROT_THRESH
    
    # Pitch: + is down, - is up
    is_turn_up = pitch < -ROT_THRESH
    is_turn_down = pitch > ROT_THRESH
    
    W, H = frame_img.size
    spacing = 60
    
    def paste_icon(name_active, name_inactive, is_active, x, y):
        name = name_active if is_active else name_inactive
        if name in icons:
            icon = icons[name]
        # Paste using alpha channel
            frame_img.paste(icon, (int(x), int(y)), icon)
    
    # Overlay WASD (bottom left)
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
    
    # Overlay arrows (bottom right)
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
    condition_pth_path=None,
    condition_video=None,
    condition_image=None,
    dit_path=None,
    wan_model_path=None,
    output_path="../examples/output_videos/output_moe_framepack_sliding.mp4",
    start_frame=0,
    initial_condition_frames=8,
    frames_per_generation=4,
    total_frames_to_generate=32,
    max_history_frames=49,
    device="cuda",
    prompt="",
    modality_type="sekai",  # "sekai" or "nuscenes"
    use_real_poses=True,
    scene_info_path=None,  # For NuScenes dataset
    # CFG parameters
    use_camera_cfg=True,
    camera_guidance_scale=2.0,
    text_guidance_scale=1.0,
    # MoE parameters
    moe_num_experts=4,
    moe_top_k=2,
    moe_hidden_dim=None,
    cam_type=1,
    use_gt_prompt=True,
    add_icons=False
):
    """
    MoE FramePack sliding window video generation - multi-modal support
    """
    # Create output directory
    dir_path = os.path.dirname(output_path)
    os.makedirs(dir_path, exist_ok=True)
    
    print(f"🔧 Starting MoE FramePack sliding window generation...")
    print(f"- Modality type: {modality_type}")
    print(f"- Camera CFG: {use_camera_cfg}, Camera guidance scale: {camera_guidance_scale}")
    print(f"- Text guidance scale: {text_guidance_scale}")
    print(f"- MoE config: experts={moe_num_experts}, top_k={moe_top_k}")
    
    # 1. Model initialization
    replace_dit_model_in_manager()
    
    model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
    model_manager.load_models([
        os.path.join(wan_model_path, "diffusion_pytorch_model.safetensors"),
        os.path.join(wan_model_path, "models_t5_umt5-xxl-enc-bf16.pth"),
        os.path.join(wan_model_path, "Wan2.1_VAE.pth"),
    ])
    pipe = WanVideoAstraPipeline.from_model_manager(model_manager, device="cuda")

    # 2. Add traditional camera encoder (compatibility)
    dim = pipe.dit.blocks[0].self_attn.q.weight.shape[0]
    for block in pipe.dit.blocks:
        block.cam_encoder = nn.Linear(13, dim)
        block.projector = nn.Linear(dim, dim)
        block.cam_encoder.weight.data.zero_()
        block.cam_encoder.bias.data.zero_()
        block.projector.weight = nn.Parameter(torch.eye(dim))
        block.projector.bias = nn.Parameter(torch.zeros(dim))
    
    # 3. Add FramePack components
    add_framepack_components(pipe.dit)
    
    # 4. Add MoE components
    moe_config = {
        "num_experts": moe_num_experts,
        "top_k": moe_top_k,
        "hidden_dim": moe_hidden_dim or dim * 2,
        "sekai_input_dim": 13,    # Sekai: 12-dim pose + 1-dim mask
        "nuscenes_input_dim": 8,   # NuScenes: 7-dim pose + 1-dim mask
        "openx_input_dim": 13       # OpenX: 12-dim pose + 1-dim mask (similar to sekai)
    }
    add_moe_components(pipe.dit, moe_config)
    
    # 5. Load trained weights
    dit_state_dict = torch.load(dit_path, map_location="cpu")
    pipe.dit.load_state_dict(dit_state_dict, strict=False)  # Use strict=False to be compatible with newly added MoE components
    pipe = pipe.to(device)
    model_dtype = next(pipe.dit.parameters()).dtype
    
    if hasattr(pipe.dit, 'clean_x_embedder'):
        pipe.dit.clean_x_embedder = pipe.dit.clean_x_embedder.to(dtype=model_dtype)
    
    # Set denoising steps
    pipe.scheduler.set_timesteps(50)
    
    # 6. Load initial conditions
    print("\n🔄 Loading initial condition frames...")
    initial_latents, encoded_data = load_or_encode_condition(
        condition_pth_path,
        condition_video,
        condition_image,
        start_frame,
        initial_condition_frames,
        device,
        pipe,
    )
    
    # Spatial cropping
    target_height, target_width = 60, 104
    C, T, H, W = initial_latents.shape
    
    if H > target_height or W > target_width:
        h_start = (H - target_height) // 2
        w_start = (W - target_width) // 2
        initial_latents = initial_latents[:, :, h_start:h_start+target_height, w_start:w_start+target_width]
        H, W = target_height, target_width
    
    history_latents = initial_latents.to(device, dtype=model_dtype)

    print(f"✅ Initial history_latents shape: {history_latents.shape}\n")
    
    # 7. Encode prompt - support CFG
    if use_gt_prompt and 'prompt_emb' in encoded_data:
        print("✅ Using pre-encoded GT prompt embedding")
        prompt_emb_pos = encoded_data['prompt_emb']
        # Move prompt_emb to correct device and dtype
        if 'context' in prompt_emb_pos:
            prompt_emb_pos['context'] = prompt_emb_pos['context'].to(device, dtype=model_dtype)
        if 'context_mask' in prompt_emb_pos:
            prompt_emb_pos['context_mask'] = prompt_emb_pos['context_mask'].to(device, dtype=model_dtype)
        
        # Generate negative prompt if using Text CFG
        if text_guidance_scale > 1.0:
            prompt_emb_neg = pipe.encode_prompt("")
            print(f"Using Text CFG with GT prompt, guidance scale: {text_guidance_scale}")
        else:
            prompt_emb_neg = None
            print("Not using Text CFG")
        
        # Print GT prompt text if available
        if 'prompt' in encoded_data['prompt_emb']:
            gt_prompt_text = encoded_data['prompt_emb']['prompt']
            print(f"📝 GT Prompt text: {gt_prompt_text}")
    else:
        # Re-encode using provided prompt parameter
        print(f"🔄 Re-encoding prompt: {prompt}")
        if text_guidance_scale > 1.0:
            prompt_emb_pos = pipe.encode_prompt(prompt)
            prompt_emb_neg = pipe.encode_prompt("")
            print(f"Using Text CFG, guidance scale: {text_guidance_scale}\n")
        else:
            prompt_emb_pos = pipe.encode_prompt(prompt)
            prompt_emb_neg = None
            print("Not using Text CFG\n")
    
    # 8. Load scene information (for NuScenes)
    scene_info = None
    if modality_type == "nuscenes" and scene_info_path and os.path.exists(scene_info_path):
        with open(scene_info_path, 'r') as f:
            scene_info = json.load(f)
        print(f"✅ Loaded NuScenes scene information: {scene_info_path}")
    
    # 9. Pre-generate complete camera embedding sequence
    if modality_type == "sekai":
        camera_embedding_full = generate_sekai_camera_embeddings_sliding(
            encoded_data.get('cam_emb', None),
            start_frame,
            initial_condition_frames,
            total_frames_to_generate,
            0,
            use_real_poses=use_real_poses,
            cam_type=cam_type
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
        raise ValueError(f"Unsupported modality type: {modality_type}")
    
    print(f"✅ Complete camera sequence shape: {camera_embedding_full.shape}")
    
    # 10. Create unconditional camera embedding for Camera CFG
    if use_camera_cfg:
        camera_embedding_uncond = torch.zeros_like(camera_embedding_full)
        print(f"🔄 Creating unconditional camera embedding for CFG")
    
    # 11. Sliding window generation loop
    total_generated = 0
    all_generated_frames = []
    
    while total_generated < total_frames_to_generate:
        current_generation = min(frames_per_generation, total_frames_to_generate - total_generated)
        print(f"\nGeneration step {total_generated // frames_per_generation + 1}")
        print(f"Current history length: {history_latents.shape[1]}, generating: {current_generation}")
        
        # FramePack data preparation - MoE version
        framepack_data = prepare_framepack_sliding_window_with_camera_moe(
            history_latents,
            current_generation,
            camera_embedding_full,
            modality_type
        )
        
        # Prepare input
        clean_latents = framepack_data['clean_latents'].unsqueeze(0)
        clean_latents_2x = framepack_data['clean_latents_2x'].unsqueeze(0)
        clean_latents_4x = framepack_data['clean_latents_4x'].unsqueeze(0)
        camera_embedding = framepack_data['camera_embedding'].unsqueeze(0)
        
        # Prepare modality_inputs
        modality_inputs = {modality_type: camera_embedding}
        
        # Prepare unconditional camera embedding for CFG
        if use_camera_cfg:
            camera_embedding_uncond_batch = camera_embedding_uncond[:camera_embedding.shape[1], :].unsqueeze(0)
            modality_inputs_uncond = {modality_type: camera_embedding_uncond_batch}
        
        # Index processing
        latent_indices = framepack_data['latent_indices'].unsqueeze(0).cpu()
        clean_latent_indices = framepack_data['clean_latent_indices'].unsqueeze(0).cpu()
        clean_latent_2x_indices = framepack_data['clean_latent_2x_indices'].unsqueeze(0).cpu()
        clean_latent_4x_indices = framepack_data['clean_latent_4x_indices'].unsqueeze(0).cpu()
        
        # Initialize latents to generate
        new_latents = torch.randn(
            1, C, current_generation, H, W,
            device=device, dtype=model_dtype
        )
        
        extra_input = pipe.prepare_extra_input(new_latents)
        
        print(f"Camera embedding shape: {camera_embedding.shape}")
        print(f"Camera mask distribution - condition: {torch.sum(camera_embedding[0, :, -1] == 1.0).item()}, target: {torch.sum(camera_embedding[0, :, -1] == 0.0).item()}")
        
        # Denoising loop - supports CFG
        timesteps = pipe.scheduler.timesteps
        
        for i, timestep in enumerate(timesteps):
            if i % 10 == 0:
                print(f"  Denoising step {i+1}/{len(timesteps)}")
            
            timestep_tensor = timestep.unsqueeze(0).to(device, dtype=model_dtype)
            
            with torch.no_grad():
                # CFG inference
                if use_camera_cfg and camera_guidance_scale > 1.0:
                    # Conditional prediction (with camera)
                    noise_pred_cond, moe_loess = pipe.dit(
                        new_latents,
                        timestep=timestep_tensor,
                        cam_emb=camera_embedding,
                        modality_inputs=modality_inputs,  # MoE modality input
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
                    
                    # Unconditional prediction (no camera)
                    noise_pred_uncond, moe_loess = pipe.dit(
                        new_latents,
                        timestep=timestep_tensor,
                        cam_emb=camera_embedding_uncond_batch,
                        modality_inputs=modality_inputs_uncond,  # MoE unconditional modality input
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
                    
                    # If using Text CFG at the same time
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
                        
                        # Apply Text CFG to results that have already applied Camera CFG
                        noise_pred = noise_pred_text_uncond + text_guidance_scale * (noise_pred - noise_pred_text_uncond)
                
                elif text_guidance_scale > 1.0 and prompt_emb_neg:
                    # Use Text CFG only
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
                    # Standard inference (no CFG)
                    noise_pred, moe_loess = pipe.dit(
                        new_latents,
                        timestep=timestep_tensor,
                        cam_emb=camera_embedding,
                        modality_inputs=modality_inputs,  # MoE modality input
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
        
        # Update history
        new_latents_squeezed = new_latents.squeeze(0)
        history_latents = torch.cat([history_latents, new_latents_squeezed], dim=1)
        
        # Maintain sliding window
        if history_latents.shape[1] > max_history_frames:
            first_frame = history_latents[:, 0:1, :, :]
            recent_frames = history_latents[:, -(max_history_frames-1):, :, :]
            history_latents = torch.cat([first_frame, recent_frames], dim=1)
            print(f"⚠️ History window full, keeping first frame + latest {max_history_frames-1} frames")
        
        print(f"History_latents shape after update: {history_latents.shape}")
        
        all_generated_frames.append(new_latents_squeezed)
        total_generated += current_generation
        
        print(f"✅ Generated {total_generated}/{total_frames_to_generate} frames")
    
    # 12. Decode and save
    print("\nDecoding generated video...")
    
    all_generated = torch.cat(all_generated_frames, dim=1)
    final_video = torch.cat([initial_latents.to(all_generated.device), all_generated], dim=1).unsqueeze(0)
    
    print(f"Final video shape: {final_video.shape}")
    
    decoded_video = pipe.decode_video(final_video, tiled=True, tile_size=(34, 34), tile_stride=(18, 16))
    
    print(f"Saving video to {output_path} ...")
    
    video_np = decoded_video[0].to(torch.float32).permute(1, 2, 3, 0).cpu().numpy()
    video_np = (video_np * 0.5 + 0.5).clip(0, 1)
    video_np = (video_np * 255).astype(np.uint8)

    icons = {}
    video_camera_poses = None
    if add_icons:
        # Load icon resources for overlay
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
                    # Adjust icon size
                    icon = icon.resize((50, 50), Image.Resampling.LANCZOS)
                    icons[name] = icon
                except Exception as e:
                    print(f"Error loading icon {name}: {e}")
            else:
                print(f"⚠️ Warning: Icon {name} not found at {path}")

        # Get camera poses corresponding to video frames
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

    print(f"✅ MoE FramePack sliding window generation completed! Saved to: {output_path}")
    print(f"-  Total generated {total_generated} frames (compressed), corresponding to original {total_generated * 4} frames")
    print(f"-  Using modality: {modality_type}")
    

def main():
    parser = argparse.ArgumentParser(description="MoE FramePack sliding window video generation - supports multi-modal")
    
    # Basic parameters
    parser.add_argument("--condition_pth",
                        type=str,
                        default=None,
                        help="Path to pre-encoded condition pth file")
    parser.add_argument("--condition_video", 
                        type=str, 
                        default=None,
                        help="Input video for novel view synthesis.")
    parser.add_argument("--condition_image",
                        type=str,
                        default=None,
                        required=True,
                        help="Input image for novel view synthesis.")
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--initial_condition_frames", type=int, default=1)
    parser.add_argument("--frames_per_generation", type=int, default=8)
    parser.add_argument("--total_frames_to_generate", type=int, default=24)
    parser.add_argument("--max_history_frames", type=int, default=100)
    parser.add_argument("--use_real_poses", default=False)
    parser.add_argument("--dit_path", type=str, 
                        default="../models/Astra/checkpoints/diffusion_pytorch_model.ckpt",
                        help="path to the pretrained DiT MoE model checkpoint")
    parser.add_argument("--wan_model_path",
                        type=str,
                        default="../models/Wan-AI/Wan2.1-T2V-1.3B",
                        help="path to Wan2.1-T2V-1.3B")
    parser.add_argument("--output_path", type=str, 
                       default='../examples/output_videos/output_moe_framepack_sliding.mp4')
    parser.add_argument("--prompt", 
                        type=str, 
                        default="",
                        help="text prompt for video generation")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--add_icons", action="store_true", default=False,
                        help="Overlay control icons on generated video")
    
    # Modality type parameters
    parser.add_argument("--modality_type", type=str, choices=["sekai", "nuscenes", "openx"], 
                       default="sekai", help="Modality type: sekai, nuscenes, or openx")
    parser.add_argument("--scene_info_path", type=str, default=None,
                       help="NuScenes scene info file path (for nuscenes modality only)")
    
    # CFG parameters
    parser.add_argument("--use_camera_cfg", default=False,
                       help="Use Camera CFG")
    parser.add_argument("--camera_guidance_scale", type=float, default=2.0,
                       help="Camera guidance scale for CFG")
    parser.add_argument("--text_guidance_scale", type=float, default=1.0,
                       help="Text guidance scale for CFG")
    
    # MoE parameters
    parser.add_argument("--moe_num_experts", type=int, default=3, help="Number of experts")
    parser.add_argument("--moe_top_k", type=int, default=1, help="Top-K experts")
    parser.add_argument("--moe_hidden_dim", type=int, default=None, help="MoE hidden dimension")
    parser.add_argument("--cam_type", type=str, default="left", help="Camera type for video trajectory")
    parser.add_argument("--use_gt_prompt", action="store_true", default=False,
                       help="Use ground truth prompt embedding from dataset")
    
    args = parser.parse_args()

    print(f"🔧 MoE FramePack CFG generation settings:")
    print(f"- Modality type: {args.modality_type}")
    print(f"- Camera CFG: {args.use_camera_cfg}")
    if args.use_camera_cfg:
        print(f"- Camera guidance scale: {args.camera_guidance_scale}")
    print(f"- Using GT Prompt: {args.use_gt_prompt}")
    print(f"- Text guidance scale: {args.text_guidance_scale}")
    print(f"- MoE config: experts={args.moe_num_experts}, top_k={args.moe_top_k}")
    print(f"- DiT: {args.dit_path}\n")
    
    # Validate NuScenes parameters
    if args.modality_type == "nuscenes" and not args.scene_info_path:
        print("⚠️ Warning: Using NuScenes modality but scene_info_path not provided, will use synthetic pose data")
        
    if not args.use_gt_prompt and (args.prompt is None or args.prompt.strip() == ""):
        print("⚠️ Warning: No prompt provided, will use empty string as prompt")
        
    if not any([args.condition_pth, args.condition_video, args.condition_image]):
        raise ValueError("Need to provide condition_pth, condition_video, or condition_image as condition input")
    
    if args.condition_pth:
        print(f"Using pre-encoded pth: {args.condition_pth}\n")
    elif args.condition_video:
        print(f"Using condition video for online encoding: {args.condition_video}\n")
    elif args.condition_image:
        print(f"Using condition image for online encoding: {args.condition_image}\n")
    
    inference_moe_framepack_sliding_window(
        condition_pth_path=args.condition_pth,
        condition_video=args.condition_video,
        condition_image=args.condition_image,
        dit_path=args.dit_path,
        wan_model_path=args.wan_model_path,
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
        # CFG parameters
        use_camera_cfg=args.use_camera_cfg,
        camera_guidance_scale=args.camera_guidance_scale,
        text_guidance_scale=args.text_guidance_scale,
        # MoE parameters
        moe_num_experts=args.moe_num_experts,
        moe_top_k=args.moe_top_k,
        moe_hidden_dim=args.moe_hidden_dim,
        cam_type=int(args.cam_type),
        use_gt_prompt=args.use_gt_prompt,
        add_icons=args.add_icons
    )


if __name__ == "__main__":
    main()