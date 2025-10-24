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

os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
    prompt_emb = 0

    os.makedirs(output_dir,exist_ok=True)

    for i, scene_name in tqdm(enumerate(os.listdir(scenes_path)),total=len(os.listdir(scenes_path))):
        # if i < 1700:
        #     continue
        scene_dir = os.path.join(scenes_path, scene_name)
        save_dir = os.path.join(output_dir,scene_name.split('.')[0])
        # print('in:',scene_dir)
        # print('out:',save_dir)
        
        if not scene_dir.endswith(".mp4"):# or os.path.isdir(output_dir):
            continue


        os.makedirs(save_dir,exist_ok=True)  
        # 检查是否已编码
        encoded_path = os.path.join(save_dir, "encoded_video.pth")
        if os.path.exists(encoded_path):
            print(f"Scene {scene_name} already encoded, skipping...")
            continue
        
        # 加载场景信息

        scene_cam_path = scene_dir.replace(".mp4", ".npz")
        if not os.path.exists(scene_cam_path):
            continue

        with np.load(scene_cam_path) as data:  
            cam_data = data.files
            cam_emb = {k: data[k].cpu() if isinstance(data[k], torch.Tensor) else data[k] for k in cam_data}
        # with open(scene_cam_path, 'rb') as f:
        #     cam_data = np.load(f)  # 此时cam_data仅包含数据，无文件句柄引用
      
        # 加载视频
        video_path = scene_dir
        if not os.path.exists(video_path):
            print(f"Video not found: {video_path}")
            continue
        
        # try:
        print(f"Encoding scene {scene_name}...")
        
        # 加载和编码视频
        video_frames = encoder.load_video_frames(video_path)
        if video_frames is None:
            print(f"Failed to load video: {video_path}")
            continue
            
        video_frames = video_frames.unsqueeze(0).to("cuda", dtype=torch.bfloat16)
        print('video shape:',video_frames.shape)
        # 编码视频
        with torch.no_grad():
            latents = encoder.pipe.encode_video(video_frames, **encoder.tiler_kwargs)[0]
            
            # 编码文本
            if processed_count == 0:
                print('encode prompt!!!')
                prompt_emb = encoder.pipe.encode_prompt("A video of a scene shot using a pedestrian's front camera while walking")
                del encoder.pipe.prompter
            # pdb.set_trace()
            # 保存编码结果
            encoded_data = {
                "latents": latents.cpu(),
                #"prompt_emb": {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in prompt_emb.items()},
                "cam_emb": cam_emb
            }
            # pdb.set_trace()
            torch.save(encoded_data, encoded_path)
            print(f"Saved encoded data: {encoded_path}")
            processed_count += 1
                
        # except Exception as e:
        #     print(f"Error encoding scene {scene_name}: {e}")
        #     continue
    
    print(f"Encoding completed! Processed {processed_count} scenes.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes_path", type=str, default="/share_zhuyixuan05/public_datasets/sekai/Sekai-Project/sekai-game-walking")
    parser.add_argument("--text_encoder_path", type=str, 
                       default="models/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth")
    parser.add_argument("--vae_path", type=str,
                       default="models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth")
    
    parser.add_argument("--output_dir",type=str,
                       default="/share_zhuyixuan05/zhuyixuan05/sekai-game-walking")

    args = parser.parse_args()
    encode_scenes(args.scenes_path, args.text_encoder_path, args.vae_path,args.output_dir)
