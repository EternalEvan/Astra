
import os
import torch
import lightning as pl
from PIL import Image
from diffsynth import WanVideoReCamMasterPipeline, ModelManager
import json
import imageio
from torchvision.transforms import v2
from einops import rearrange
import argparse
import numpy as np
import pdb
from tqdm import tqdm
import pandas as pd

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from scipy.spatial.transform import Slerp
from scipy.spatial.transform import Rotation as R

def interpolate_camera_poses(original_frames, original_poses, target_frames):
    """
    对相机姿态进行插值，生成目标帧对应的姿态参数
    
    参数:
    original_frames: 原始帧索引列表，如[0,6,12,...]
    original_poses: 原始姿态数组，形状为(n,7)，每行[tx, ty, tz, qx, qy, qz, qw]
    target_frames: 目标帧索引列表，如[0,4,8,12,...]
    
    返回:
    target_poses: 插值后的姿态数组，形状为(m,7),m为目标帧数量
    """
    # 确保输入有效
    print('original_frames:',len(original_frames))
    print('original_poses:',len(original_poses))
    if len(original_frames) != len(original_poses):
        raise ValueError("原始帧数量与姿态数量不匹配")
    
    if original_poses.shape[1] != 7:
        raise ValueError(f"原始姿态应为(n,7)格式，实际为{original_poses.shape}")
    
    target_poses = []
    
    # 提取旋转部分并转换为Rotation对象
    rotations = R.from_quat(original_poses[:, 3:7])  # 提取四元数部分
    
    for t in target_frames:
        # 找到t前后的原始帧索引
        idx = np.searchsorted(original_frames, t, side='left')
        
        # 处理边界情况
        if idx == 0:
            # 使用第一个姿态
            target_poses.append(original_poses[0])
            continue
        if idx >= len(original_frames):
            # 使用最后一个姿态
            target_poses.append(original_poses[-1])
            continue
            
        # 获取前后帧的信息
        t_prev, t_next = original_frames[idx-1], original_frames[idx]
        pose_prev, pose_next = original_poses[idx-1], original_poses[idx]
        
        # 计算插值权重
        alpha = (t - t_prev) / (t_next - t_prev)
        
        # 1. 平移向量的线性插值
        translation_prev = pose_prev[:3]
        translation_next = pose_next[:3]
        interpolated_translation = translation_prev + alpha * (translation_next - translation_prev)
        
        # 2. 旋转四元数的球面线性插值(SLERP)
        # 创建Slerp对象
        slerp = Slerp([t_prev, t_next], rotations[idx-1:idx+1])
        interpolated_rotation = slerp(t)
        
        # 组合平移和旋转
        interpolated_pose = np.concatenate([
            interpolated_translation,
            interpolated_rotation.as_quat()  # 转换回四元数
        ])
        
        target_poses.append(interpolated_pose)
    
    return np.array(target_poses)

class VideoEncoder(pl.LightningModule):
    def __init__(self, text_encoder_path, vae_path, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)):
        super().__init__()
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        model_manager.load_models([text_encoder_path, vae_path])
        self.pipe = WanVideoReCamMasterPipeline.from_model_manager(model_manager)
        self.tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}
        
        self.frame_process = v2.Compose([
            v2.ToTensor(),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    
    def crop_and_resize(self, image):
        width, height = image.size
        width_ori, height_ori_ = 832 , 480
        image = v2.functional.resize(
            image,
            (round(height_ori_), round(width_ori)),
            interpolation=v2.InterpolationMode.BILINEAR
        )
        return image

    def load_single_frame(self, video_path, frame_idx):
        """只加载指定的单帧"""
        reader = imageio.get_reader(video_path)
        
        try:
            # 直接跳转到指定帧
            frame_data = reader.get_data(frame_idx)
            frame = Image.fromarray(frame_data)
            frame = self.crop_and_resize(frame)
            frame = self.frame_process(frame)
            
            # 添加batch和time维度: [C, H, W] -> [1, C, 1, H, W]
            frame = frame.unsqueeze(0).unsqueeze(2)
            
        except Exception as e:
            print(f"Error loading frame {frame_idx} from {video_path}: {e}")
            return None
        finally:
            reader.close()
        
        return frame

    def load_video_frames(self, video_path):
        """加载完整视频（保留用于兼容性）"""
        reader = imageio.get_reader(video_path)
        frames = []
        
        for frame_data in reader:
            frame = Image.fromarray(frame_data)
            frame = self.crop_and_resize(frame)
            frame = self.frame_process(frame)
            frames.append(frame)
        
        reader.close()
        
        if len(frames) == 0:
            return None
            
        frames = torch.stack(frames, dim=0)
        frames = rearrange(frames, "T C H W -> C T H W")
        return frames

def encode_scenes(scenes_path, text_encoder_path, vae_path,output_dir):
    """编码所有场景的视频"""

    encoder = VideoEncoder(text_encoder_path, vae_path)
    encoder = encoder.cuda()
    encoder.pipe.device = "cuda"
    
    processed_count = 0
    processed_chunk_count = 0

    metadata = pd.read_csv('/share_zhuyixuan05/public_datasets/SpatialVID-HQ/data/train/SpatialVID_HQ_metadata.csv')

    os.makedirs(output_dir,exist_ok=True)
    chunk_size = 300

    for i, scene_name in enumerate(os.listdir(scenes_path)):
        if i < 2:
            continue
        print('group:',i)
        scene_dir = os.path.join(scenes_path, scene_name)
        
        print('in:',scene_dir)
        for j, video_name in tqdm(enumerate(os.listdir(scene_dir)),total=len(os.listdir(scene_dir))):
            print(video_name)
            video_path = os.path.join(scene_dir, video_name)
            if not video_path.endswith(".mp4"):
                continue
            
            video_info = metadata[metadata['id'] == video_name[:-4]]
            num_frames = video_info['num frames'].iloc[0]

            scene_cam_dir = video_path.replace("videos","annotations")[:-4]
            scene_cam_path = os.path.join(scene_cam_dir,'poses.npy')
            scene_caption_path = os.path.join(scene_cam_dir,'caption.json')

            with open(scene_caption_path, 'r', encoding='utf-8') as f:
                caption_data = json.load(f)
                caption = caption_data["SceneSummary"]
            
            if not os.path.exists(scene_cam_path):
                print(f"Pose not found: {scene_cam_path}")
                continue

            camera_poses = np.load(scene_cam_path)
            cam_data_len = camera_poses.shape[0]

            if not os.path.exists(video_path):
                print(f"Video not found: {video_path}")
                continue

            video_name = video_name[:-4].split('_')[0]
            start_frame = 0
            end_frame = num_frames

            cam_interval = end_frame // (cam_data_len - 1)
            
            cam_frames = np.linspace(start_frame, end_frame, cam_data_len, endpoint=True)
            cam_frames = np.round(cam_frames).astype(int)
            cam_frames = cam_frames.tolist()

            sampled_range = range(start_frame, end_frame, chunk_size)
            sampled_frames = list(sampled_range)

            print(f"Encoding scene {video_name}...")
            chunk_count_in_one_video = 0
            
            for sampled_chunk_start in sampled_frames:
                if num_frames - sampled_chunk_start < 100:
                    continue
                    
                sampled_chunk_end = sampled_chunk_start + chunk_size
                start_str = f"{sampled_chunk_start:07d}"
                end_str = f"{sampled_chunk_end:07d}"

                chunk_name = f"{video_name}_{start_str}_{end_str}"
                save_chunk_dir = os.path.join(output_dir, chunk_name)
                os.makedirs(save_chunk_dir, exist_ok=True)
                
                print(f"Encoding chunk {chunk_name}...")

                first_latent_path = os.path.join(save_chunk_dir, "first_latent.pth")
                
                if os.path.exists(first_latent_path):
                    print(f"First latent for chunk {chunk_name} already exists, skipping...")
                    continue

                # 只加载需要的那一帧
                first_frame_idx = sampled_chunk_start
                print(f"first_frame:{first_frame_idx}")
                first_frame = encoder.load_single_frame(video_path, first_frame_idx)
                
                if first_frame is None:
                    print(f"Failed to load frame {first_frame_idx} from: {video_path}")
                    continue
                
                first_frame = first_frame.to("cuda", dtype=torch.bfloat16)
                
                # 重复4次
                repeated_first_frame = first_frame.repeat(1, 1, 4, 1, 1)
                print(f"Repeated first frame shape: {repeated_first_frame.shape}")

                with torch.no_grad():
                    first_latents = encoder.pipe.encode_video(repeated_first_frame, **encoder.tiler_kwargs)[0]

                first_latent_data = {
                    "latents": first_latents.cpu(),
                }
                torch.save(first_latent_data, first_latent_path)
                print(f"Saved first latent: {first_latent_path}")

                processed_chunk_count += 1
                chunk_count_in_one_video += 1

        processed_count += 1
        print("Encoded scene number:", processed_count)
        print("Encoded chunk number:", processed_chunk_count)
    
    print(f"Encoding completed! Processed {processed_count} scenes.")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes_path", type=str, default="/share_zhuyixuan05/public_datasets/SpatialVID-HQ/SpatialVid/HQ/videos/")
    parser.add_argument("--text_encoder_path", type=str, 
                       default="models/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth")
    parser.add_argument("--vae_path", type=str,
                       default="models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth")
    
    parser.add_argument("--output_dir",type=str,
                       default="/share_zhuyixuan05/zhuyixuan05/spatialvid")

    args = parser.parse_args()
    encode_scenes(args.scenes_path, args.text_encoder_path, args.vae_path,args.output_dir)
