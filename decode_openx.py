import os
import torch
import numpy as np
from PIL import Image
import imageio
import argparse
from diffsynth import WanVideoReCamMasterPipeline, ModelManager
from tqdm import tqdm
import json

class VideoDecoder:
    def __init__(self, vae_path, device="cuda"):
        """初始化视频解码器"""
        self.device = device
        
        # 初始化模型管理器
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        model_manager.load_models([vae_path])
        
        # 创建pipeline并只保留VAE
        self.pipe = WanVideoReCamMasterPipeline.from_model_manager(model_manager)
        self.pipe = self.pipe.to(device)
        
        # 🔧 关键修复：确保VAE及其所有组件都在正确设备上
        self.pipe.vae = self.pipe.vae.to(device)
        if hasattr(self.pipe.vae, 'model'):
            self.pipe.vae.model = self.pipe.vae.model.to(device)
        
        print(f"✅ VAE解码器初始化完成，设备: {device}")

    def decode_latents_to_video(self, latents, output_path, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)):
        """
        将latents解码为视频 - 修正版本，修复维度处理问题
        """
        print(f"🔧 开始解码latents...")
        print(f"输入latents形状: {latents.shape}")
        print(f"输入latents设备: {latents.device}")
        print(f"输入latents数据类型: {latents.dtype}")
        
        # 确保latents有batch维度
        if len(latents.shape) == 4:  # [C, T, H, W]
            latents = latents.unsqueeze(0)  # -> [1, C, T, H, W]
        
        # 🔧 关键修正：确保latents在正确的设备上且数据类型匹配
        model_dtype = next(self.pipe.vae.parameters()).dtype
        model_device = next(self.pipe.vae.parameters()).device
        
        print(f"模型设备: {model_device}")
        print(f"模型数据类型: {model_dtype}")
        
        # 将latents移动到正确的设备和数据类型
        latents = latents.to(device=model_device, dtype=model_dtype)
        
        print(f"解码latents形状: {latents.shape}")
        print(f"解码latents设备: {latents.device}")
        print(f"解码latents数据类型: {latents.dtype}")
        
        # 🔧 强制设置pipeline设备，确保所有操作在同一设备上
        self.pipe.device = model_device
        
        # 使用VAE解码
        with torch.no_grad():
            try:
                if tiled:
                    print("🔧 尝试tiled解码...")
                    decoded_video = self.pipe.decode_video(
                        latents, 
                        tiled=True, 
                        tile_size=tile_size, 
                        tile_stride=tile_stride
                    )
                else:
                    print("🔧 使用非tiled解码...")
                    decoded_video = self.pipe.decode_video(latents, tiled=False)
                    
            except Exception as e:
                print(f"decode_video失败，错误: {e}")
                import traceback
                traceback.print_exc()
                
                # 🔧 fallback: 尝试直接调用VAE
                try:
                    print("🔧 尝试直接调用VAE解码...")
                    decoded_video = self.pipe.vae.decode(
                        latents.squeeze(0),  # 移除batch维度 [C, T, H, W]
                        device=model_device, 
                        tiled=False
                    )
                    # 手动调整维度: VAE输出 [T, H, W, C] -> [1, T, H, W, C]
                    if len(decoded_video.shape) == 4:  # [T, H, W, C]
                        decoded_video = decoded_video.unsqueeze(0)  # -> [1, T, H, W, C]
                except Exception as e2:
                    print(f"直接VAE解码也失败: {e2}")
                    raise e2
        
        print(f"解码后视频形状: {decoded_video.shape}")
        
        # 🔧 关键修正：正确处理维度顺序
        video_np = None
        
        if len(decoded_video.shape) == 5:
            # 检查不同的可能维度顺序
            if decoded_video.shape == torch.Size([1, 3, 113, 480, 832]):
                # 格式: [B, C, T, H, W] -> 需要转换为 [T, H, W, C]
                print("🔧 检测到格式: [B, C, T, H, W]")
                video_np = decoded_video[0].permute(1, 2, 3, 0).to(torch.float32).cpu().numpy()  # [T, H, W, C]
            elif decoded_video.shape[1] == 3:
                # 如果第二个维度是3，可能是 [B, C, T, H, W]
                print("🔧 检测到可能的格式: [B, C, T, H, W]")
                video_np = decoded_video[0].permute(1, 2, 3, 0).to(torch.float32).cpu().numpy()  # [T, H, W, C]
            elif decoded_video.shape[-1] == 3:
                # 如果最后一个维度是3，可能是 [B, T, H, W, C]
                print("🔧 检测到格式: [B, T, H, W, C]")
                video_np = decoded_video[0].to(torch.float32).cpu().numpy()  # [T, H, W, C]
            else:
                # 尝试找到维度为3的位置
                shape = list(decoded_video.shape)
                if 3 in shape:
                    channel_dim = shape.index(3)
                    print(f"🔧 检测到通道维度在位置: {channel_dim}")
                    
                    if channel_dim == 1:  # [B, C, T, H, W]
                        video_np = decoded_video[0].permute(1, 2, 3, 0).to(torch.float32).cpu().numpy()
                    elif channel_dim == 4:  # [B, T, H, W, C]
                        video_np = decoded_video[0].to(torch.float32).cpu().numpy()
                    else:
                        print(f"⚠️ 未知的通道维度位置: {channel_dim}")
                        raise ValueError(f"Cannot handle channel dimension at position {channel_dim}")
                else:
                    print(f"⚠️ 未找到通道维度为3的位置，形状: {decoded_video.shape}")
                    raise ValueError(f"Cannot find channel dimension of size 3 in shape {decoded_video.shape}")
                    
        elif len(decoded_video.shape) == 4:
            # 4维张量，检查可能的格式
            if decoded_video.shape[-1] == 3:  # [T, H, W, C]
                video_np = decoded_video.to(torch.float32).cpu().numpy()
            elif decoded_video.shape[0] == 3:  # [C, T, H, W]
                video_np = decoded_video.permute(1, 2, 3, 0).to(torch.float32).cpu().numpy()
            else:
                print(f"⚠️ 无法处理的4D视频形状: {decoded_video.shape}")
                raise ValueError(f"Cannot handle 4D video tensor shape: {decoded_video.shape}")
        else:
            print(f"⚠️ 意外的视频维度数: {len(decoded_video.shape)}")
            raise ValueError(f"Unexpected video tensor dimensions: {decoded_video.shape}")
        
        if video_np is None:
            raise ValueError("Failed to convert video tensor to numpy array")
            
        print(f"转换后视频数组形状: {video_np.shape}")
        
        # 🔧 验证最终形状
        if len(video_np.shape) != 4:
            raise ValueError(f"Expected 4D array [T, H, W, C], got {video_np.shape}")
        
        if video_np.shape[-1] != 3:
            print(f"⚠️ 通道数异常: 期望3，实际{video_np.shape[-1]}")
            print(f"完整形状: {video_np.shape}")
            # 尝试其他维度排列
            if video_np.shape[0] == 3:  # [C, T, H, W]
                print("🔧 尝试重新排列: [C, T, H, W] -> [T, H, W, C]")
                video_np = np.transpose(video_np, (1, 2, 3, 0))
            elif video_np.shape[1] == 3:  # [T, C, H, W]
                print("🔧 尝试重新排列: [T, C, H, W] -> [T, H, W, C]")
                video_np = np.transpose(video_np, (0, 2, 3, 1))
            else:
                raise ValueError(f"Expected 3 channels (RGB), got {video_np.shape[-1]} channels")
        
        # 反归一化
        video_np = (video_np * 0.5 + 0.5).clip(0, 1)  # 反归一化
        video_np = (video_np * 255).astype(np.uint8)
        
        print(f"最终视频数组形状: {video_np.shape}")
        print(f"视频数组值范围: {video_np.min()} - {video_np.max()}")
        
        # 保存视频
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        try:
            with imageio.get_writer(output_path, fps=10, quality=8) as writer:
                for frame_idx, frame in enumerate(video_np):
                    # 🔧 验证每一帧的形状
                    if len(frame.shape) != 3 or frame.shape[-1] != 3:
                        print(f"⚠️ 帧 {frame_idx} 形状异常: {frame.shape}")
                        continue
                    
                    writer.append_data(frame)
                    if frame_idx % 10 == 0:
                        print(f"  写入帧 {frame_idx}/{len(video_np)}")
        except Exception as e:
            print(f"保存视频失败: {e}")
            # 🔧 尝试保存前几帧为图片进行调试
            debug_dir = os.path.join(os.path.dirname(output_path), "debug_frames")
            os.makedirs(debug_dir, exist_ok=True)
            
            for i in range(min(5, len(video_np))):
                frame = video_np[i]
                debug_path = os.path.join(debug_dir, f"debug_frame_{i}.png")
                try:
                    if len(frame.shape) == 3 and frame.shape[-1] == 3:
                        Image.fromarray(frame).save(debug_path)
                        print(f"调试: 保存帧 {i} 到 {debug_path}")
                    else:
                        print(f"调试: 帧 {i} 形状异常: {frame.shape}")
                except Exception as e2:
                    print(f"调试: 保存帧 {i} 失败: {e2}")
            raise e
        
        print(f"✅ 视频保存到: {output_path}")
        return video_np

    def save_frames_as_images(self, video_np, output_dir, prefix="frame"):
        """将视频帧保存为单独的图像文件"""
        os.makedirs(output_dir, exist_ok=True)
        
        for i, frame in enumerate(video_np):
            frame_path = os.path.join(output_dir, f"{prefix}_{i:04d}.png")
            # 🔧 验证帧形状
            if len(frame.shape) == 3 and frame.shape[-1] == 3:
                Image.fromarray(frame).save(frame_path)
            else:
                print(f"⚠️ 跳过形状异常的帧 {i}: {frame.shape}")
        
        print(f"✅ 保存了 {len(video_np)} 帧到: {output_dir}")

def decode_single_episode(encoded_pth_path, vae_path, output_base_dir, device="cuda"):
    """解码单个episode的编码数据 - 修正版本"""
    print(f"\n🔧 解码episode: {encoded_pth_path}")
    
    # 加载编码数据
    try:
        encoded_data = torch.load(encoded_pth_path, weights_only=False, map_location="cpu")
        print(f"✅ 成功加载编码数据")
    except Exception as e:
        print(f"❌ 加载编码数据失败: {e}")
        return False
    
    # 检查数据结构
    print("🔍 编码数据结构:")
    for key, value in encoded_data.items():
        if isinstance(value, torch.Tensor):
            print(f"  - {key}: {value.shape}, dtype: {value.dtype}, device: {value.device}")
        elif isinstance(value, dict):
            print(f"  - {key}: dict with keys {list(value.keys())}")
        else:
            print(f"  - {key}: {type(value)}")
    
    # 获取latents
    latents = encoded_data.get('latents')
    if latents is None:
        print("❌ 未找到latents数据")
        return False
    
    # 🔧 确保latents在CPU上（加载时的默认状态）
    if latents.device != torch.device('cpu'):
        latents = latents.cpu()
        print(f"🔧 将latents移动到CPU: {latents.device}")
    
    episode_info = encoded_data.get('episode_info', {})
    episode_idx = episode_info.get('episode_idx', 'unknown')
    total_frames = episode_info.get('total_frames', latents.shape[1] * 4)  # 估算原始帧数
    
    print(f"Episode信息:")
    print(f"  - Episode索引: {episode_idx}")
    print(f"  - Latents形状: {latents.shape}")
    print(f"  - Latents设备: {latents.device}")
    print(f"  - Latents数据类型: {latents.dtype}")
    print(f"  - 原始总帧数: {total_frames}")
    print(f"  - 压缩后帧数: {latents.shape[1]}")
    
    # 创建输出目录
    episode_name = f"episode_{episode_idx:06d}" if isinstance(episode_idx, int) else f"episode_{episode_idx}"
    output_dir = os.path.join(output_base_dir, episode_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化解码器
    try:
        decoder = VideoDecoder(vae_path, device)
    except Exception as e:
        print(f"❌ 初始化解码器失败: {e}")
        return False
    
    # 解码为视频
    video_output_path = os.path.join(output_dir, "decoded_video.mp4")
    try:
        video_np = decoder.decode_latents_to_video(
            latents, 
            video_output_path,
            tiled=False,  # 🔧 首先尝试非tiled解码，避免tiled的复杂性
            tile_size=(34, 34),
            tile_stride=(18, 16)
        )
        
        # 保存前几帧为图像（用于快速检查）
        frames_dir = os.path.join(output_dir, "frames")
        sample_frames = video_np[:min(10, len(video_np))]  # 只保存前10帧
        decoder.save_frames_as_images(sample_frames, frames_dir, f"frame_{episode_idx}")
        
        # 保存解码信息
        decode_info = {
            "source_pth": encoded_pth_path,
            "decoded_video_path": video_output_path,
            "latents_shape": list(latents.shape),
            "decoded_video_shape": list(video_np.shape),
            "original_total_frames": total_frames,
            "decoded_frames": len(video_np),
            "compression_ratio": total_frames / len(video_np) if len(video_np) > 0 else 0,
            "latents_dtype": str(latents.dtype),
            "latents_device": str(latents.device),
            "vae_compression_ratio": total_frames / latents.shape[1] if latents.shape[1] > 0 else 0
        }
        
        info_path = os.path.join(output_dir, "decode_info.json")
        with open(info_path, 'w') as f:
            json.dump(decode_info, f, indent=2)
        
        print(f"✅ Episode {episode_idx} 解码完成")
        print(f"  - 原始帧数: {total_frames}")
        print(f"  - 解码帧数: {len(video_np)}")
        print(f"  - 压缩比: {decode_info['compression_ratio']:.2f}")
        print(f"  - VAE时间压缩比: {decode_info['vae_compression_ratio']:.2f}")
        return True
        
    except Exception as e:
        print(f"❌ 解码失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def batch_decode_episodes(encoded_base_dir, vae_path, output_base_dir, max_episodes=None, device="cuda"):
    """批量解码episodes"""
    print(f"🔧 批量解码Open-X episodes")
    print(f"源目录: {encoded_base_dir}")
    print(f"输出目录: {output_base_dir}")
    
    # 查找所有编码的episodes
    episode_dirs = []
    if os.path.exists(encoded_base_dir):
        for item in sorted(os.listdir(encoded_base_dir)):  # 排序确保一致性
            episode_dir = os.path.join(encoded_base_dir, item)
            if os.path.isdir(episode_dir):
                encoded_path = os.path.join(episode_dir, "encoded_video.pth")
                if os.path.exists(encoded_path):
                    episode_dirs.append(encoded_path)
    
    print(f"找到 {len(episode_dirs)} 个编码的episodes")
    
    if max_episodes and len(episode_dirs) > max_episodes:
        episode_dirs = episode_dirs[:max_episodes]
        print(f"限制处理前 {max_episodes} 个episodes")
    
    # 批量解码
    success_count = 0
    for i, encoded_pth_path in enumerate(tqdm(episode_dirs, desc="解码episodes")):
        print(f"\n{'='*60}")
        print(f"处理 {i+1}/{len(episode_dirs)}: {os.path.basename(os.path.dirname(encoded_pth_path))}")
        
        success = decode_single_episode(encoded_pth_path, vae_path, output_base_dir, device)
        if success:
            success_count += 1
        
        print(f"当前成功率: {success_count}/{i+1} ({success_count/(i+1)*100:.1f}%)")
    
    print(f"\n🎉 批量解码完成!")
    print(f"总处理: {len(episode_dirs)} 个episodes")
    print(f"成功解码: {success_count} 个episodes")
    print(f"成功率: {success_count/len(episode_dirs)*100:.1f}%")

def main():
    parser = argparse.ArgumentParser(description="解码Open-X编码的latents以验证正确性 - 修正版本")
    parser.add_argument("--mode", type=str, choices=["single", "batch"], default="batch",
                       help="解码模式：single (单个episode) 或 batch (批量)")
    parser.add_argument("--encoded_pth", type=str, 
                       default="/share_zhuyixuan05/zhuyixuan05/openx-fractal-encoded/episode_000000/encoded_video.pth",
                       help="单个编码文件路径（single模式）")
    parser.add_argument("--encoded_base_dir", type=str,
                       default="/share_zhuyixuan05/zhuyixuan05/openx-fractal-encoded",
                       help="编码数据基础目录（batch模式）")
    parser.add_argument("--vae_path", type=str,
                       default="models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
                       help="VAE模型路径")
    parser.add_argument("--output_dir", type=str,
                       default="./decoded_results_fixed",
                       help="解码输出目录")
    parser.add_argument("--max_episodes", type=int, default=5,
                       help="最大解码episodes数量（batch模式，用于测试）")
    parser.add_argument("--device", type=str, default="cuda",
                       help="计算设备")
    
    args = parser.parse_args()
    
    print("🔧 Open-X Latents 解码验证工具 (修正版本 - Fixed)")
    print(f"模式: {args.mode}")
    print(f"VAE路径: {args.vae_path}")
    print(f"输出目录: {args.output_dir}")
    print(f"设备: {args.device}")
    
    # 🔧 检查CUDA可用性
    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠️ CUDA不可用，切换到CPU")
        args.device = "cpu"
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.mode == "single":
        print(f"输入文件: {args.encoded_pth}")
        if not os.path.exists(args.encoded_pth):
            print(f"❌ 输入文件不存在: {args.encoded_pth}")
            return
        
        success = decode_single_episode(args.encoded_pth, args.vae_path, args.output_dir, args.device)
        if success:
            print("✅ 单个episode解码成功")
        else:
            print("❌ 单个episode解码失败")
    
    elif args.mode == "batch":
        print(f"输入目录: {args.encoded_base_dir}")
        print(f"最大episodes: {args.max_episodes}")
        
        if not os.path.exists(args.encoded_base_dir):
            print(f"❌ 输入目录不存在: {args.encoded_base_dir}")
            return
        
        batch_decode_episodes(args.encoded_base_dir, args.vae_path, args.output_dir, args.max_episodes, args.device)

if __name__ == "__main__":
    main()