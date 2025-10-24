import os
import torch
import numpy as np
from PIL import Image
import imageio
import json
from diffsynth import WanVideoReCamMasterPipeline, ModelManager
import argparse
from torchvision.transforms import v2
from einops import rearrange
import torch.nn as nn


def load_encoded_video_from_pth(pth_path, start_frame=0, num_frames=10):
    """
    从pth文件加载预编码的视频数据
    Args:
        pth_path: pth文件路径
        start_frame: 起始帧索引（基于压缩后的latent帧数）
        num_frames: 需要的帧数（基于压缩后的latent帧数）
    Returns:
        condition_latents: [C, T, H, W] 格式的latent tensor
    """
    print(f"Loading encoded video from {pth_path}")
    
    # 加载编码数据
    encoded_data = torch.load(pth_path, weights_only=False, map_location="cpu")
    
    # 获取latent数据
    full_latents = encoded_data['latents']  # [C, T, H, W]
    
    print(f"Full latents shape: {full_latents.shape}")
    print(f"Extracting frames {start_frame} to {start_frame + num_frames}")
    
    # 检查帧数是否足够
    if start_frame + num_frames > full_latents.shape[1]:
        raise ValueError(f"Not enough frames: requested {start_frame + num_frames}, available {full_latents.shape[1]}")
    
    # 提取指定帧数
    condition_latents = full_latents[:, start_frame:start_frame + num_frames, :, :]
    
    print(f"Extracted condition latents shape: {condition_latents.shape}")
    
    return condition_latents, encoded_data


def compute_relative_pose(pose_a, pose_b, use_torch=False):
    """
    计算相机B相对于相机A的相对位姿矩阵
    """
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


def generate_camera_poses_from_data(cam_data, start_frame, condition_frames, target_frames):
    """
    从实际相机数据生成pose embeddings
    Args:
        cam_data: 相机外参数据
        start_frame: 起始帧（原始帧索引）
        condition_frames: 条件帧数（压缩后）
        target_frames: 目标帧数（压缩后）
    """
    time_compression_ratio = 4
    total_frames = condition_frames + target_frames
    
    # 获取相机外参序列
    cam_extrinsic = cam_data  # [N, 4, 4]
    
    # 计算原始帧索引
    start_frame_original = start_frame * time_compression_ratio
    end_frame_original = (start_frame + total_frames) * time_compression_ratio
    
    print(f"Using camera data from frame {start_frame_original} to {end_frame_original}")
    
    # 计算相对pose
    relative_poses = []
    for i in range(total_frames):
        frame_idx = start_frame_original + i * time_compression_ratio
        next_frame_idx = frame_idx + time_compression_ratio
        
      
        cam_prev = cam_extrinsic[frame_idx]
      
            

        relative_poses.append(torch.as_tensor(cam_prev))  # 取前3行

        print(cam_prev)
    # 组装pose embedding
    pose_embedding = torch.stack(relative_poses, dim=0)
    # print('pose_embedding init:',pose_embedding[0])
    print('pose_embedding:',pose_embedding)
    # assert False

    # pose_embedding = rearrange(pose_embedding, 'b c d -> b (c d)')  # [frames, 12]
    
    # 添加mask信息
    mask = torch.zeros(total_frames, dtype=torch.float32)
    mask[:condition_frames] = 1.0  # condition frames
    mask = mask.view(-1, 1)
    
    # 组合pose和mask
    camera_embedding = torch.cat([pose_embedding, mask], dim=1)  # [frames, 13]
    
    print(f"Generated camera embedding shape: {camera_embedding.shape}")
    
    return camera_embedding.to(torch.bfloat16)


def generate_camera_poses(direction="forward", target_frames=10, condition_frames=20):
    """
    根据指定方向生成相机pose序列（合成数据）
    """
    time_compression_ratio = 4
    total_frames = condition_frames + target_frames
    
    poses = []
    
    for i in range(total_frames):
        t = i / max(1, total_frames - 1)  # 0 to 1
        
        # 创建变换矩阵
        pose = np.eye(4, dtype=np.float32)
        
        if direction == "forward":
            # 前进：沿z轴负方向移动
            pose[2, 3] = -t * 0.04
            print('forward!')
            
        elif direction == "backward":
            # 后退：沿z轴正方向移动
            pose[2, 3] = t * 2.0
            
        elif direction == "left_turn":
            # 左转：前进 + 绕y轴旋转
            pose[2, 3] = -t * 0.03  # 前进
            pose[0, 3] = t * 0.02   # 左移
            # 添加旋转
            yaw = t * 1
            pose[0, 0] = np.cos(yaw)
            pose[0, 2] = np.sin(yaw)
            pose[2, 0] = -np.sin(yaw)
            pose[2, 2] = np.cos(yaw)
            
        elif direction == "right_turn":
            # 右转：前进 + 绕y轴反向旋转
            pose[2, 3] = -t * 0.03 # 前进
            pose[0, 3] = -t * 0.02 # 右移
            # 添加旋转
            yaw = - t * 1
            pose[0, 0] = np.cos(yaw)
            pose[0, 2] = np.sin(yaw)
            pose[2, 0] = -np.sin(yaw)
            pose[2, 2] = np.cos(yaw)
            
        poses.append(pose)
    
    # 计算相对pose
    relative_poses = []
    for i in range(len(poses) - 1):
        relative_pose = compute_relative_pose(poses[i], poses[i + 1])
        relative_poses.append(torch.as_tensor(relative_pose[:3, :]))  # 取前3行
    
    # 为了匹配模型输入，需要确保帧数正确
    if len(relative_poses) < total_frames:
        # 补充最后一帧
        relative_poses.append(relative_poses[-1])
    
    pose_embedding = torch.stack(relative_poses[:total_frames], dim=0)
    
    print('pose_embedding init:',pose_embedding[0])

    print('pose_embedding:',pose_embedding[-5:])

    pose_embedding = rearrange(pose_embedding, 'b c d -> b (c d)')  # [frames, 12]
    
    # 添加mask信息
    mask = torch.zeros(total_frames, dtype=torch.float32)
    mask[:condition_frames] = 1.0  # condition frames
    mask = mask.view(-1, 1)
    
    # 组合pose和mask
    camera_embedding = torch.cat([pose_embedding, mask], dim=1)  # [frames, 13]
    
    print(f"Generated {direction} movement poses:")
    print(f"  Total frames: {total_frames}")
    print(f"  Camera embedding shape: {camera_embedding.shape}")
    
    return camera_embedding.to(torch.bfloat16)


def inference_sekai_video_from_pth(
    condition_pth_path,
    dit_path,
    output_path="sekai/infer_results/output_sekai.mp4",
    start_frame=0,
    condition_frames=10,  # 压缩后的帧数
    target_frames=2,      # 压缩后的帧数
    device="cuda",
    prompt="a robotic arm executing precise manipulation tasks on a clean, organized desk",
    direction="forward",
    use_real_poses=True
):
    """
    从pth文件进行Sekai视频推理
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Setting up models for {direction} movement...")
    
    # 1. Load models
    model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
    model_manager.load_models([
        "models/Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors",
        "models/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
        "models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
    ])
    pipe = WanVideoReCamMasterPipeline.from_model_manager(model_manager, device="cuda")

    # Add camera components to DiT
    dim = pipe.dit.blocks[0].self_attn.q.weight.shape[0]
    for block in pipe.dit.blocks:
        block.cam_encoder = nn.Linear(30, dim)  # 13维embedding (12D pose + 1D mask)
        block.projector = nn.Linear(dim, dim)
        block.cam_encoder.weight.data.zero_()
        block.cam_encoder.bias.data.zero_()
        block.projector.weight = nn.Parameter(torch.eye(dim))
        block.projector.bias = nn.Parameter(torch.zeros(dim))
    
    # Load trained DiT weights
    dit_state_dict = torch.load(dit_path, map_location="cpu")
    pipe.dit.load_state_dict(dit_state_dict, strict=True)
    pipe = pipe.to(device)
    pipe.scheduler.set_timesteps(50)
    
    print("Loading condition video from pth...")
    
    # Load condition video from pth
    condition_latents, encoded_data = load_encoded_video_from_pth(
        condition_pth_path, 
        start_frame=start_frame,
        num_frames=condition_frames
    )
    
    condition_latents = condition_latents.unsqueeze(0).to(device, dtype=pipe.torch_dtype)
    
    print("Processing poses...")
    
    # 生成相机pose embedding
    if use_real_poses and 'cam_emb' in encoded_data:
        print("Using real camera poses from data")
        camera_embedding = generate_camera_poses_from_data(
            encoded_data['cam_emb'],
            start_frame=start_frame,
            condition_frames=condition_frames,
            target_frames=target_frames
        )
    else:
        print(f"Using synthetic {direction} poses")
        camera_embedding = generate_camera_poses(
            direction=direction,
            target_frames=target_frames,
            condition_frames=condition_frames
        )


    
    camera_embedding = camera_embedding.unsqueeze(0).to(device, dtype=torch.bfloat16)
    
    print(f"Camera embedding shape: {camera_embedding.shape}")
    
    print("Encoding prompt...")
    
    # Encode text prompt
    prompt_emb = pipe.encode_prompt(prompt)
    
    print("Generating video...")
    
    # Generate target latents
    batch_size = 1
    channels = condition_latents.shape[1]
    latent_height = condition_latents.shape[3]
    latent_width = condition_latents.shape[4]
    
    # 空间裁剪以节省内存（如果需要）
    target_height, target_width = 64, 64
    
    if latent_height > target_height or latent_width > target_width:
        # 中心裁剪
        h_start = (latent_height - target_height) // 2
        w_start = (latent_width - target_width) // 2
        condition_latents = condition_latents[:, :, :, 
                        h_start:h_start+target_height, 
                        w_start:w_start+target_width]
        latent_height = target_height
        latent_width = target_width
                
    # Initialize target latents with noise
    target_latents = torch.randn(
        batch_size, channels, target_frames, latent_height, latent_width,
        device=device, dtype=pipe.torch_dtype
    )
    
    print(f"Condition latents shape: {condition_latents.shape}")
    print(f"Target latents shape: {target_latents.shape}")
    print(f"Camera embedding shape: {camera_embedding.shape}")
    
    # Combine condition and target latents
    combined_latents = torch.cat([condition_latents, target_latents], dim=2)
    print(f"Combined latents shape: {combined_latents.shape}")

    # Prepare extra inputs
    extra_input = pipe.prepare_extra_input(combined_latents)
    
    # Denoising loop
    timesteps = pipe.scheduler.timesteps
        
    for i, timestep in enumerate(timesteps):
        print(f"Denoising step {i+1}/{len(timesteps)}")
        
        # Prepare timestep
        timestep_tensor = timestep.unsqueeze(0).to(device, dtype=pipe.torch_dtype)
        
        # Predict noise
        with torch.no_grad():
            noise_pred = pipe.dit(
                combined_latents,
                timestep=timestep_tensor,
                cam_emb=camera_embedding,
                **prompt_emb,
                **extra_input
            )
        
        # Update only target part
        target_noise_pred = noise_pred[:, :, condition_frames:, :, :]
        target_latents = pipe.scheduler.step(target_noise_pred, timestep, target_latents)
        
        # Update combined latents
        combined_latents[:, :, condition_frames:, :, :] = target_latents
    
    print("Decoding video...")
    
    # Decode final video
    final_video = torch.cat([condition_latents, target_latents], dim=2)
    decoded_video = pipe.decode_video(final_video, tiled=True, tile_size=(34, 34), tile_stride=(18, 16))

    # Save video
    print(f"Saving video to {output_path}")

    # Convert to numpy and save
    video_np = decoded_video[0].to(torch.float32).permute(1, 2, 3, 0).cpu().numpy()
    video_np = (video_np * 0.5 + 0.5).clip(0, 1)  # Denormalize
    video_np = (video_np * 255).astype(np.uint8)

    with imageio.get_writer(output_path, fps=20) as writer:
        for frame in video_np:
            writer.append_data(frame)

    print(f"Video generation completed! Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Sekai Video Generation Inference from PTH")
    parser.add_argument("--condition_pth", type=str,
                       default="/share_zhuyixuan05/zhuyixuan05/rlbench/OpenBox_demo_49/encoded_video.pth")
    parser.add_argument("--start_frame", type=int, default=0,
                       help="Starting frame index (compressed latent frames)")
    parser.add_argument("--condition_frames", type=int, default=8,
                       help="Number of condition frames (compressed latent frames)")
    parser.add_argument("--target_frames", type=int, default=8,
                       help="Number of target frames to generate (compressed latent frames)")
    parser.add_argument("--direction", type=str, default="left_turn",
                       choices=["forward", "backward", "left_turn", "right_turn"],
                       help="Direction of camera movement (if not using real poses)")
    parser.add_argument("--use_real_poses",  default=False,
                       help="Use real camera poses from data")
    parser.add_argument("--dit_path", type=str, default="/home/zhuyixuan05/ReCamMaster/RLBench-train/step2000_dynamic.ckpt",
                       help="Path to trained DiT checkpoint")
    parser.add_argument("--output_path", type=str, default='/home/zhuyixuan05/ReCamMaster/rlbench/infer_results/output_rl_2.mp4',
                       help="Output video path")
    parser.add_argument("--prompt", type=str, 
                       default="a robotic arm executing precise manipulation tasks on a clean, organized desk",
                       help="Text prompt for generation")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to run inference on")
    
    args = parser.parse_args()
    
    # 生成输出路径
    if args.output_path is None:
        pth_filename = os.path.basename(args.condition_pth)
        name_parts = os.path.splitext(pth_filename)
        output_dir = "rlbench/infer_results"
        os.makedirs(output_dir, exist_ok=True)
        
        if args.use_real_poses:
            output_filename = f"{name_parts[0]}_real_poses_{args.start_frame}_{args.condition_frames}_{args.target_frames}.mp4"
        else:
            output_filename = f"{name_parts[0]}_{args.direction}_{args.start_frame}_{args.condition_frames}_{args.target_frames}.mp4"
        
        output_path = os.path.join(output_dir, output_filename)
    else:
        output_path = args.output_path

    print(f"Input pth: {args.condition_pth}")
    print(f"Start frame: {args.start_frame} (compressed)")
    print(f"Condition frames: {args.condition_frames} (compressed, original: {args.condition_frames * 4})")
    print(f"Target frames: {args.target_frames} (compressed, original: {args.target_frames * 4})")
    print(f"Use real poses: {args.use_real_poses}")
    print(f"Output video will be saved to: {output_path}")
    
    inference_sekai_video_from_pth(
        condition_pth_path=args.condition_pth,
        dit_path=args.dit_path,
        output_path=output_path,
        start_frame=args.start_frame,
        condition_frames=args.condition_frames,
        target_frames=args.target_frames,
        device=args.device,
        prompt=args.prompt,
        direction=args.direction,
        use_real_poses=args.use_real_poses
    )


if __name__ == "__main__":
    main()