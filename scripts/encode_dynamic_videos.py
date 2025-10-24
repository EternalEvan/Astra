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
from tqdm import tqdm
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

def encode_scenes(scenes_path, text_encoder_path, vae_path):
    """编码所有场景的视频"""
    encoder = VideoEncoder(text_encoder_path, vae_path)
    encoder = encoder.cuda()
    encoder.pipe.device = "cuda"
    
    processed_count = 0
    
    for idx, scene_name in enumerate(tqdm(os.listdir(scenes_path))):
        if idx < 450:
            continue
        scene_dir = os.path.join(scenes_path, scene_name)
        if not os.path.isdir(scene_dir):
            continue
            
        # 检查是否已编码
        encoded_path = os.path.join(scene_dir, "encoded_video-480p-1.pth")
        if os.path.exists(encoded_path):
            print(f"Scene {scene_name} already encoded, skipping...")
            continue
        
        # 加载场景信息
        scene_info_path = os.path.join(scene_dir, "scene_info.json")
        if not os.path.exists(scene_info_path):
            continue
            
        with open(scene_info_path, 'r') as f:
            scene_info = json.load(f)
        
        # 加载视频
        video_path = os.path.join(scene_dir, scene_info['video_path'])
        if not os.path.exists(video_path):
            print(f"Video not found: {video_path}")
            continue
        
        try:
            print(f"Encoding scene {scene_name}...")
            
            # 加载和编码视频
            video_frames = encoder.load_video_frames(video_path)
            if video_frames is None:
                print(f"Failed to load video: {video_path}")
                continue
                
            video_frames = video_frames.unsqueeze(0).to("cuda", dtype=torch.bfloat16)
            
            # 编码视频
            with torch.no_grad():
                latents = encoder.pipe.encode_video(video_frames, **encoder.tiler_kwargs)[0]
                # print(latents.shape)
                # assert False
                # 编码文本
                # prompt_emb = encoder.pipe.encode_prompt("A car driving scene captured by front camera")
                if processed_count == 0:
                    print('encode prompt!!!')
                    prompt_emb = encoder.pipe.encode_prompt("A car driving scene captured by front camera")
                    del encoder.pipe.prompter

                # 保存编码结果
                encoded_data = {
                    "latents": latents.cpu(),
                    "prompt_emb": {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in prompt_emb.items()},
                    "image_emb": {}
                }
                
                torch.save(encoded_data, encoded_path)
                print(f"Saved encoded data: {encoded_path}")
                processed_count += 1
                
        except Exception as e:
            print(f"Error encoding scene {scene_name}: {e}")
            continue
    
    print(f"Encoding completed! Processed {processed_count} scenes.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes_path", type=str, default="/share_zhuyixuan05/zhuyixuan05/nuscenes_video_generation_dynamic/scenes")
    parser.add_argument("--text_encoder_path", type=str, 
                       default="models/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth")
    parser.add_argument("--vae_path", type=str,
                       default="models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth")
    
    args = parser.parse_args()
    encode_scenes(args.scenes_path, args.text_encoder_path, args.vae_path)
