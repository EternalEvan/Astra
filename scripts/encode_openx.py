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
from tqdm import tqdm

# 🔧 关键修复：设置环境变量避免GCS连接
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TFDS_DISABLE_GCS"] = "1"

import tensorflow_datasets as tfds
import tensorflow as tf

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
    
    def crop_and_resize(self, image, target_width=832, target_height=480):
        """调整图像尺寸"""
        image = v2.functional.resize(
            image,
            (target_height, target_width),
            interpolation=v2.InterpolationMode.BILINEAR
        )
        return image

    def load_episode_frames(self, episode_data, max_frames=300):
        """🔧 从fractal数据集加载视频帧 - 基于实际observation字段优化"""
        frames = []
        
        steps = episode_data['steps']
        frame_count = 0
        
        print(f"开始提取帧，最多 {max_frames} 帧...")
        
        for step_idx, step in enumerate(steps):
            if frame_count >= max_frames:
                break
            
            try:
                obs = step['observation']
                
                # 🔧 基于实际的observation字段，优先使用'image'
                img_data = None
                image_keys_to_try = [
                    'image',                 # ✅ 确认存在的主要图像字段
                    'rgb',                   # 备用RGB图像
                    'camera_image',          # 备用相机图像
                    'exterior_image_1_left', # 可能的外部摄像头
                    'wrist_image',           # 可能的手腕摄像头
                ]
                
                for img_key in image_keys_to_try:
                    if img_key in obs:
                        try:
                            img_tensor = obs[img_key]
                            img_data = img_tensor.numpy()
                            if step_idx < 3:  # 只为前几个步骤打印
                                print(f"✅ 找到图像字段: {img_key}, 形状: {img_data.shape}")
                            break
                        except Exception as e:
                            if step_idx < 3:
                                print(f"尝试字段 {img_key} 失败: {e}")
                            continue
                
                if img_data is not None:
                    # 确保图像数据格式正确
                    if len(img_data.shape) == 3:  # [H, W, C]
                        if img_data.dtype == np.uint8:
                            frame = Image.fromarray(img_data)
                        else:
                            # 如果是归一化的浮点数，转换为uint8
                            if img_data.max() <= 1.0:
                                img_data = (img_data * 255).astype(np.uint8)
                            else:
                                img_data = img_data.astype(np.uint8)
                            frame = Image.fromarray(img_data)
                        
                        # 转换为RGB如果需要
                        if frame.mode != 'RGB':
                            frame = frame.convert('RGB')
                        
                        frame = self.crop_and_resize(frame)
                        frame = self.frame_process(frame)
                        frames.append(frame)
                        frame_count += 1
                        
                        if frame_count % 50 == 0:
                            print(f"已处理 {frame_count} 帧")
                    else:
                        if step_idx < 5:
                            print(f"步骤 {step_idx}: 图像形状不正确 {img_data.shape}")
                else:
                    # 如果找不到图像，打印可用的观测键
                    if step_idx < 5:  # 只为前几个步骤打印
                        available_keys = list(obs.keys())
                        print(f"步骤 {step_idx}: 未找到图像，可用键: {available_keys}")
                        
            except Exception as e:
                print(f"处理步骤 {step_idx} 时出错: {e}")
                continue
        
        print(f"成功提取 {len(frames)} 帧")
        
        if len(frames) == 0:
            return None
            
        frames = torch.stack(frames, dim=0)
        frames = rearrange(frames, "T C H W -> C T H W")
        return frames

    def extract_camera_poses(self, episode_data, num_frames):
        """🔧 从fractal数据集提取相机位姿信息 - 基于实际observation和action字段优化"""
        camera_poses = []
        
        steps = episode_data['steps']
        frame_count = 0
        
        print("提取相机位姿信息...")
        
        # 🔧 累积位姿信息
        cumulative_translation = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        cumulative_rotation = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # 欧拉角
        
        for step_idx, step in enumerate(steps):
            if frame_count >= num_frames:
                break
                
            try:
                obs = step['observation']
                action = step.get('action', {})
                
                # 🔧 基于实际的字段提取位姿变化
                pose_data = {}
                found_pose = False
                
                # 1. 优先使用action中的world_vector（世界坐标系中的位移）
                if 'world_vector' in action:
                    try:
                        world_vector = action['world_vector'].numpy()
                        if len(world_vector) == 3:
                            # 累积世界坐标位移
                            cumulative_translation += world_vector
                            pose_data['translation'] = cumulative_translation.copy()
                            found_pose = True
                            
                            if step_idx < 3:
                                print(f"使用action.world_vector: {world_vector}, 累积位移: {cumulative_translation}")
                    except Exception as e:
                        if step_idx < 3:
                            print(f"action.world_vector提取失败: {e}")
                
                # 2. 使用action中的rotation_delta（旋转变化）
                if 'rotation_delta' in action:
                    try:
                        rotation_delta = action['rotation_delta'].numpy()
                        if len(rotation_delta) == 3:
                            # 累积旋转变化
                            cumulative_rotation += rotation_delta
                            
                            # 转换为四元数（简化版本）
                            euler_angles = cumulative_rotation
                            # 欧拉角转四元数（ZYX顺序）
                            roll, pitch, yaw = euler_angles[0], euler_angles[1], euler_angles[2]
                            
                            # 简化的欧拉角到四元数转换
                            cy = np.cos(yaw * 0.5)
                            sy = np.sin(yaw * 0.5)
                            cp = np.cos(pitch * 0.5)
                            sp = np.sin(pitch * 0.5)
                            cr = np.cos(roll * 0.5)
                            sr = np.sin(roll * 0.5)
                            
                            qw = cr * cp * cy + sr * sp * sy
                            qx = sr * cp * cy - cr * sp * sy
                            qy = cr * sp * cy + sr * cp * sy
                            qz = cr * cp * sy - sr * sp * cy
                            
                            pose_data['rotation'] = np.array([qw, qx, qy, qz], dtype=np.float32)
                            found_pose = True
                            
                            if step_idx < 3:
                                print(f"使用action.rotation_delta: {rotation_delta}, 累积旋转: {cumulative_rotation}")
                    except Exception as e:
                        if step_idx < 3:
                            print(f"action.rotation_delta提取失败: {e}")
                
                # 确保rotation字段存在
                if 'rotation' not in pose_data:
                    # 使用当前累积的旋转计算四元数
                    roll, pitch, yaw = cumulative_rotation[0], cumulative_rotation[1], cumulative_rotation[2]
                    
                    cy = np.cos(yaw * 0.5)
                    sy = np.sin(yaw * 0.5)
                    cp = np.cos(pitch * 0.5)
                    sp = np.sin(pitch * 0.5)
                    cr = np.cos(roll * 0.5)
                    sr = np.sin(roll * 0.5)
                    
                    qw = cr * cp * cy + sr * sp * sy
                    qx = sr * cp * cy - cr * sp * sy
                    qy = cr * sp * cy + sr * cp * sy
                    qz = cr * cp * sy - sr * sp * cy
                    
                    pose_data['rotation'] = np.array([qw, qx, qy, qz], dtype=np.float32)
                
                camera_poses.append(pose_data)
                frame_count += 1
                
            except Exception as e:
                print(f"提取位姿步骤 {step_idx} 时出错: {e}")
                # 添加默认位姿
                pose_data = {
                    'translation': cumulative_translation.copy(),
                    'rotation': np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
                }
                camera_poses.append(pose_data)
                frame_count += 1
        
        print(f"提取了 {len(camera_poses)} 个位姿")
        print(f"最终累积位移: {cumulative_translation}")
        print(f"最终累积旋转: {cumulative_rotation}")
        
        return camera_poses

    def create_camera_matrices(self, camera_poses):
        """将位姿转换为4x4变换矩阵"""
        matrices = []
        
        for pose in camera_poses:
            matrix = np.eye(4, dtype=np.float32)
            
            # 设置平移
            matrix[:3, 3] = pose['translation']
            
            # 设置旋转 - 假设是四元数 [w, x, y, z]
            if len(pose['rotation']) == 4:
                # 四元数转旋转矩阵
                q = pose['rotation']
                w, x, y, z = q[0], q[1], q[2], q[3]
                
                # 四元数到旋转矩阵的转换
                matrix[0, 0] = 1 - 2*(y*y + z*z)
                matrix[0, 1] = 2*(x*y - w*z)
                matrix[0, 2] = 2*(x*z + w*y)
                matrix[1, 0] = 2*(x*y + w*z)
                matrix[1, 1] = 1 - 2*(x*x + z*z)
                matrix[1, 2] = 2*(y*z - w*x)
                matrix[2, 0] = 2*(x*z - w*y)
                matrix[2, 1] = 2*(y*z + w*x)
                matrix[2, 2] = 1 - 2*(x*x + y*y)
            elif len(pose['rotation']) == 3:
                # 欧拉角转换（如果需要）
                pass
            
            matrices.append(matrix)
        
        return np.array(matrices)

def encode_fractal_dataset(dataset_path, text_encoder_path, vae_path, output_dir, max_episodes=None):
    """🔧 编码fractal20220817_data数据集 - 基于实际字段结构优化"""
    
    encoder = VideoEncoder(text_encoder_path, vae_path)
    encoder = encoder.cuda()
    encoder.pipe.device = "cuda"
    
    os.makedirs(output_dir, exist_ok=True)
    
    processed_count = 0
    prompt_emb = None
        
    try:
        # 🔧 使用你提供的成功方法加载数据集
        ds = tfds.load(
            "fractal20220817_data",
            split="train",
            data_dir=dataset_path,
        )
        
        print(f"✅ 成功加载fractal20220817_data数据集")
        
        # 限制处理的episode数量
        if max_episodes:
            ds = ds.take(max_episodes)
            print(f"限制处理episodes数量: {max_episodes}")
        
    except Exception as e:
        print(f"❌ 加载数据集失败: {e}")
        return
    
    for episode_idx, episode in enumerate(tqdm(ds, desc="处理episodes")):
        try:
            episode_name = f"episode_{episode_idx:06d}"
            save_episode_dir = os.path.join(output_dir, episode_name)
            
            # 检查是否已经处理过
            encoded_path = os.path.join(save_episode_dir, "encoded_video.pth")
            if os.path.exists(encoded_path):
                print(f"Episode {episode_name} 已处理，跳过...")
                processed_count += 1
                continue
            
            os.makedirs(save_episode_dir, exist_ok=True)
            
            print(f"\n🔧 处理episode {episode_name}...")
            
            # 🔧 分析episode结构（仅对前几个episode）
            if episode_idx < 2:
                print("Episode结构分析:")
                for key in episode.keys():
                    print(f"  - {key}: {type(episode[key])}")
                
                # 分析第一个step的结构
                steps = episode['steps']
                for step in steps.take(1):
                    print("第一个step结构:")
                    for key in step.keys():
                        print(f"    - {key}: {type(step[key])}")
                    
                    if 'observation' in step:
                        obs = step['observation']
                        print("  observation键:")
                        print(f"    🔍 可用字段: {list(obs.keys())}")
                        
                        # 重点检查图像和位姿相关字段
                        key_fields = ['image', 'vector_to_go', 'rotation_delta_to_go', 'base_pose_tool_reached']
                        for key in key_fields:
                            if key in obs:
                                try:
                                    value = obs[key]
                                    if hasattr(value, 'shape'):
                                        print(f"      ✅ {key}: {type(value)}, shape: {value.shape}")
                                    else:
                                        print(f"      ✅ {key}: {type(value)}")
                                except Exception as e:
                                    print(f"      ❌ {key}: 无法访问 ({e})")
                    
                    if 'action' in step:
                        action = step['action']
                        print("  action键:")
                        print(f"    🔍 可用字段: {list(action.keys())}")
                        
                        # 重点检查位姿相关字段
                        key_fields = ['world_vector', 'rotation_delta', 'base_displacement_vector']
                        for key in key_fields:
                            if key in action:
                                try:
                                    value = action[key]
                                    if hasattr(value, 'shape'):
                                        print(f"      ✅ {key}: {type(value)}, shape: {value.shape}")
                                    else:
                                        print(f"      ✅ {key}: {type(value)}")
                                except Exception as e:
                                    print(f"      ❌ {key}: 无法访问 ({e})")
            
            # 加载视频帧
            video_frames = encoder.load_episode_frames(episode)
            if video_frames is None:
                print(f"❌ 无法加载episode {episode_name}的视频帧")
                continue
            
            print(f"✅ Episode {episode_name} 视频形状: {video_frames.shape}")
            
            # 提取相机位姿
            num_frames = video_frames.shape[1]
            camera_poses = encoder.extract_camera_poses(episode, num_frames)
            camera_matrices = encoder.create_camera_matrices(camera_poses)
            
            print(f"🔧 编码episode {episode_name}...")
            
            # 准备相机数据
            cam_emb = {
                'extrinsic': camera_matrices,
                'intrinsic': np.eye(3, dtype=np.float32)
            }
            
            # 编码视频
            frames_batch = video_frames.unsqueeze(0).to("cuda", dtype=torch.bfloat16)
            
            with torch.no_grad():
                latents = encoder.pipe.encode_video(frames_batch, **encoder.tiler_kwargs)[0]
                
                # 编码文本prompt（第一次）
                if prompt_emb is None:
                    print('🔧 编码prompt...')
                    prompt_emb = encoder.pipe.encode_prompt(
                        "A video of robotic manipulation task with camera movement"
                    )
                    # 释放prompter以节省内存
                    del encoder.pipe.prompter
                
                # 保存编码结果
                encoded_data = {
                    "latents": latents.cpu(),
                    "prompt_emb": {k: v.cpu() if isinstance(v, torch.Tensor) else v 
                                 for k, v in prompt_emb.items()},
                    "cam_emb": cam_emb,
                    "episode_info": {
                        "episode_idx": episode_idx,
                        "total_frames": video_frames.shape[1],
                        "pose_extraction_method": "observation_action_based"
                    }
                }
                
                torch.save(encoded_data, encoded_path)
                print(f"✅ 保存编码数据: {encoded_path}")
            
            processed_count += 1
            print(f"✅ 已处理 {processed_count} 个episodes")
            
        except Exception as e:
            print(f"❌ 处理episode {episode_idx}时出错: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"🎉 编码完成! 总共处理了 {processed_count} 个episodes")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encode Open-X Fractal20220817 Dataset - Based on Real Structure")
    parser.add_argument("--dataset_path", type=str, 
                       default="/share_zhuyixuan05/public_datasets/open-x/0.1.0",
                       help="Path to tensorflow_datasets directory")
    parser.add_argument("--text_encoder_path", type=str, 
                       default="models/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth")
    parser.add_argument("--vae_path", type=str,
                       default="models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth")
    parser.add_argument("--output_dir", type=str,
                       default="/share_zhuyixuan05/zhuyixuan05/openx-fractal-encoded")
    parser.add_argument("--max_episodes", type=int, default=10000,
                       help="Maximum number of episodes to process (default: 10 for testing)")
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("🚀 开始编码Open-X Fractal数据集 (基于实际字段结构)...")
    print(f"📁 数据集路径: {args.dataset_path}")
    print(f"💾 输出目录: {args.output_dir}")
    print(f"🔢 最大处理episodes: {args.max_episodes}")
    print("🔧 基于实际observation和action字段的位姿提取方法")
    print("✅ 优先使用 'image' 字段获取图像数据")

    encode_fractal_dataset(
        args.dataset_path,
        args.text_encoder_path, 
        args.vae_path,
        args.output_dir,
        args.max_episodes
    )