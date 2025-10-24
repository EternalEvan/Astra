
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
            # v2.CenterCrop(size=(900, 1600)),
            # v2.Resize(size=(900, 1600), antialias=True),
            v2.ToTensor(),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    
    def crop_and_resize(self, image):
        width, height = image.size
        # print(width,height)
        width_ori, height_ori_ = 832 , 480
        image = v2.functional.resize(
            image,
            (round(height_ori_), round(width_ori)),
            interpolation=v2.InterpolationMode.BILINEAR
        )
        return image

    def load_video_frames(self, video_path):
        """加载完整视频"""
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

    prompt_emb = 0

    metadata = pd.read_csv('/share_zhuyixuan05/public_datasets/SpatialVID-HQ/data/train/SpatialVID_HQ_metadata.csv')


    os.makedirs(output_dir,exist_ok=True)
    chunk_size = 300
    required_keys = ["latents", "cam_emb", "prompt_emb"]

    for i, scene_name in enumerate(os.listdir(scenes_path)):
        # print('index-----:',type(i))
        if i < 3 :#or i >=2000:
        #     # print('index-----:',i)
            continue
            # print('index:',i)
        print('group:',i)
        scene_dir = os.path.join(scenes_path, scene_name)
        
        # save_dir = os.path.join(output_dir,scene_name.split('.')[0])
        print('in:',scene_dir)
        # print('out:',save_dir)
        for j, video_name in tqdm(enumerate(os.listdir(scene_dir)),total=len(os.listdir(scene_dir))):
            
            # if j < 1000 :#or i >=2000:
                # print('index:',j)
                # continue
            print(video_name)
            video_path = os.path.join(scene_dir, video_name)
            if not video_path.endswith(".mp4"):# or os.path.isdir(output_dir):
                continue
            
            video_info = metadata[metadata['id'] == video_name[:-4]]
            num_frames = video_info['num frames'].iloc[0]

            scene_cam_dir = video_path.replace( "videos","annotations")[:-4]
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
                
                # cam_emb = {k: data[k].cpu() if isinstance(data[k], torch.Tensor) else data[k] for k in cam_data}
            # with open(scene_cam_path, 'rb') as f:
            #     cam_data = np.load(f)  # 此时cam_data仅包含数据，无文件句柄引用

            # 加载视频
            # video_path = scene_dir
            if not os.path.exists(video_path):
                print(f"Video not found: {video_path}")
                continue
            
            start_str = f"{0:07d}"
            end_str = f"{chunk_size:07d}"
            chunk_name = f"{video_name[:-4]}_{start_str}_{end_str}"
            first_save_chunk_dir = os.path.join(output_dir,chunk_name)
            
            first_chunk_encoded_path = os.path.join(first_save_chunk_dir, "encoded_video.pth")
            # print(first_chunk_encoded_path)
            if os.path.exists(first_chunk_encoded_path):
                data = torch.load(first_chunk_encoded_path,weights_only=False)
                if 'latents' in data:
                    video_frames = 1
            else:
                video_frames = encoder.load_video_frames(video_path)
                if video_frames is None:
                    print(f"Failed to load video: {video_path}")
                    continue
                print('video shape:',video_frames.shape)

                
                    
                video_frames = video_frames.unsqueeze(0).to("cuda", dtype=torch.bfloat16)
                print('video shape:',video_frames.shape)

            video_name = video_name[:-4].split('_')[0]
            start_frame = 0
            end_frame = num_frames
            # print("num_frames:",num_frames)

            cam_interval = end_frame // (cam_data_len - 1)
            
            cam_frames = np.linspace(start_frame, end_frame, cam_data_len, endpoint=True)
            cam_frames = np.round(cam_frames).astype(int)
            cam_frames = cam_frames.tolist()
            # list(range(0, end_frame + 1 , cam_interval))
            

            sampled_range = range(start_frame, end_frame , chunk_size)
            sampled_frames = list(sampled_range)

            sampled_chunk_end = sampled_frames[0] + chunk_size
            start_str = f"{sampled_frames[0]:07d}"
            end_str = f"{sampled_chunk_end:07d}"

            chunk_name = f"{video_name}_{start_str}_{end_str}"
            # save_chunk_path = os.path.join(output_dir,chunk_name,"encoded_video.pth")
            
            # if os.path.exists(save_chunk_path):
            #     print(f"Video {video_name} already encoded, skipping...")
            #     continue
            
            
            
            
            
            # print(sampled_frames)

            print(f"Encoding scene {video_name}...")
            chunk_count_in_one_video = 0
            for sampled_chunk_start in sampled_frames:
                if num_frames - sampled_chunk_start < 100:
                    continue
                sampled_chunk_end = sampled_chunk_start + chunk_size
                start_str = f"{sampled_chunk_start:07d}"
                end_str = f"{sampled_chunk_end:07d}"

                resample_cam_frame = list(range(sampled_chunk_start, sampled_chunk_end , 4))

                # 生成保存目录名（假设video_name已定义）
                chunk_name = f"{video_name}_{start_str}_{end_str}"
                save_chunk_dir = os.path.join(output_dir,chunk_name)

                os.makedirs(save_chunk_dir,exist_ok=True)  
                print(f"Encoding chunk {chunk_name}...")

                encoded_path = os.path.join(save_chunk_dir, "encoded_video.pth")

                missing_keys = required_keys
                if os.path.exists(encoded_path):
                    print('error:',encoded_path)
                    data = torch.load(encoded_path,weights_only=False)
                    missing_keys = [key for key in required_keys if key not in data]
                    # print(missing_keys)
                    # print(f"Chunk {chunk_name} already encoded, skipping...")
                    if missing_keys:
                        print(f"警告: 文件中缺少以下必要元素: {missing_keys}")
                    if len(missing_keys) == 0 :
                        continue
                else:
                    print(f"警告: 缺少pth文件: {encoded_path}")
                    if not isinstance(video_frames, torch.Tensor):
                        
                        video_frames = encoder.load_video_frames(video_path)
                        if video_frames is None:
                            print(f"Failed to load video: {video_path}")
                            continue
                            
                        video_frames = video_frames.unsqueeze(0).to("cuda", dtype=torch.bfloat16)

                    print('video shape:',video_frames.shape)
                if "latents" in missing_keys:
                    chunk_frames = video_frames[:,:, sampled_chunk_start - start_frame : sampled_chunk_end - start_frame,...]
                    
                    # print('extrinsic:',cam_emb['extrinsic'].shape)
                    
                    # chunk_cam_emb ={'extrinsic':cam_emb['extrinsic'][sampled_chunk_start - start_frame : sampled_chunk_end - start_frame],
                    #                 'intrinsic':cam_emb['intrinsic']}

                    # print('chunk shape:',chunk_frames.shape)

                    with torch.no_grad():
                        latents = encoder.pipe.encode_video(chunk_frames, **encoder.tiler_kwargs)[0]
                else:
                    latents = data['latents']
                if "cam_emb" in missing_keys:  
                    cam_emb = interpolate_camera_poses(cam_frames, camera_poses,resample_cam_frame)
                    chunk_cam_emb ={'extrinsic':cam_emb}
                    print(f"视频长度:{chunk_size},重采样相机长度:{cam_emb.shape[0]}")
                else:
                    chunk_cam_emb = data['cam_emb']

                if "prompt_emb" in missing_keys:
                    # 编码文本
                    if chunk_count_in_one_video == 0:
                        print(caption)
                        with torch.no_grad():
                            prompt_emb = encoder.pipe.encode_prompt(caption)
                else:
                    prompt_emb = data['prompt_emb']
                    
                    #     del encoder.pipe.prompter
                    # pdb.set_trace()
                    # 保存编码结果
                encoded_data = {
                        "latents": latents.cpu(),
                        "prompt_emb": {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in prompt_emb.items()},
                        "cam_emb": chunk_cam_emb
                    }
                    # pdb.set_trace()
                torch.save(encoded_data, encoded_path)
                print(f"Saved encoded data: {encoded_path}")
                processed_chunk_count += 1
                chunk_count_in_one_video += 1

        processed_count += 1

        print("Encoded scene numebr:",processed_count)
        print("Encoded chunk numebr:",processed_chunk_count)

        # os.makedirs(save_dir,exist_ok=True)  
        # # 检查是否已编码
        # encoded_path = os.path.join(save_dir, "encoded_video.pth")
        # if os.path.exists(encoded_path):
        #     print(f"Scene {scene_name} already encoded, skipping...")
        #     continue
        
        # 加载场景信息

        
        
        # try:
        # print(f"Encoding scene {scene_name}...")
        
        # 加载和编码视频
        
        # 编码视频
        # with torch.no_grad():
        #     latents = encoder.pipe.encode_video(video_frames, **encoder.tiler_kwargs)[0]
            
        #     # 编码文本
        #     if processed_count == 0:
        #         print('encode prompt!!!')
        #         prompt_emb = encoder.pipe.encode_prompt("A video of a scene shot using a pedestrian's front camera while walking")
        #         del encoder.pipe.prompter
        #     # pdb.set_trace()
        #     # 保存编码结果
        #     encoded_data = {
        #         "latents": latents.cpu(),
        #         #"prompt_emb": {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in prompt_emb.items()},
        #         "cam_emb": cam_emb
        #     }
        #     # pdb.set_trace()
        #     torch.save(encoded_data, encoded_path)
        #     print(f"Saved encoded data: {encoded_path}")
        #     processed_count += 1
                
        # except Exception as e:
        #     print(f"Error encoding scene {scene_name}: {e}")
        #     continue
    
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
