import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import imageio
import json
from diffsynth import WanVideoReCamMasterPipeline, ModelManager
import argparse
from torchvision.transforms import v2
from einops import rearrange
import copy


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
    
    encoded_data = torch.load(pth_path, weights_only=False, map_location="cpu")
    full_latents = encoded_data['latents']  # [C, T, H, W]
    
    print(f"Full latents shape: {full_latents.shape}")
    print(f"Extracting frames {start_frame} to {start_frame + num_frames}")
    
    if start_frame + num_frames > full_latents.shape[1]:
        raise ValueError(f"Not enough frames: requested {start_frame + num_frames}, available {full_latents.shape[1]}")
    
    condition_latents = full_latents[:, start_frame:start_frame + num_frames, :, :]
    print(f"Extracted condition latents shape: {condition_latents.shape}")
    
    return condition_latents, encoded_data


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


def prepare_framepack_inputs(full_latents, condition_frames, target_frames, start_frame=0):
    """🔧 准备FramePack风格的多尺度输入"""
    # 确保有batch维度
    if len(full_latents.shape) == 4:  # [C, T, H, W]
        full_latents = full_latents.unsqueeze(0)  # -> [1, C, T, H, W]
        squeeze_batch = True
    else:
        squeeze_batch = False
    
    B, C, T, H, W = full_latents.shape
    
    # 主要latents（用于去噪预测）
    target_start = start_frame + condition_frames
    target_end = target_start + target_frames
    latent_indices = torch.arange(target_start, target_end)
    main_latents = full_latents[:, :, latent_indices, :, :]
    
    # 🔧 1x条件帧（起始帧 + 最后1帧）
    clean_latent_indices = torch.tensor([start_frame, start_frame + condition_frames - 1])
    clean_latents = full_latents[:, :, clean_latent_indices, :, :]
    
    # 🔧 2x条件帧（最后2帧）
    clean_latents_2x = torch.zeros(B, C, 2, H, W, dtype=full_latents.dtype)
    clean_latent_2x_indices = torch.full((2,), -1, dtype=torch.long)
    
    if condition_frames >= 2:
        actual_indices = torch.arange(max(start_frame, start_frame + condition_frames - 2), 
                                    start_frame + condition_frames)
        start_pos = 2 - len(actual_indices)
        clean_latents_2x[:, :, start_pos:, :, :] = full_latents[:, :, actual_indices, :, :]
        clean_latent_2x_indices[start_pos:] = actual_indices
    
    # 🔧 4x条件帧（最多16帧）
    clean_latents_4x = torch.zeros(B, C, 16, H, W, dtype=full_latents.dtype)
    clean_latent_4x_indices = torch.full((16,), -1, dtype=torch.long)
    
    if condition_frames >= 1:
        actual_indices = torch.arange(max(start_frame, start_frame + condition_frames - 16), 
                                    start_frame + condition_frames)
        start_pos = 16 - len(actual_indices)
        clean_latents_4x[:, :, start_pos:, :, :] = full_latents[:, :, actual_indices, :, :]
        clean_latent_4x_indices[start_pos:] = actual_indices
    
    # 移除batch维度（如果原来没有）
    if squeeze_batch:
        main_latents = main_latents.squeeze(0)
        clean_latents = clean_latents.squeeze(0)
        clean_latents_2x = clean_latents_2x.squeeze(0)
        clean_latents_4x = clean_latents_4x.squeeze(0)
    
    return {
        'latents': main_latents,
        'clean_latents': clean_latents,
        'clean_latents_2x': clean_latents_2x,
        'clean_latents_4x': clean_latents_4x,
        'latent_indices': latent_indices,
        'clean_latent_indices': clean_latent_indices,
        'clean_latent_2x_indices': clean_latent_2x_indices,
        'clean_latent_4x_indices': clean_latent_4x_indices,
    }


def generate_camera_poses_from_data(cam_data, start_frame, condition_frames, target_frames):
    """从实际相机数据生成pose embeddings"""
    time_compression_ratio = 4
    total_frames = condition_frames + target_frames
    
    cam_extrinsic = cam_data['extrinsic']  # [N, 4, 4]
    start_frame_original = start_frame * time_compression_ratio
    
    print(f"Using camera data from frame {start_frame_original}")
    
    # 计算相对pose
    relative_poses = []
    for i in range(total_frames):
        frame_idx = start_frame_original + i * time_compression_ratio
        next_frame_idx = frame_idx + time_compression_ratio
        
        if next_frame_idx >= len(cam_extrinsic):
            print('Out of temporal range, using last available pose')
            relative_poses.append(relative_poses[-1] if relative_poses else torch.zeros(3, 4))
        else:
            cam_prev = cam_extrinsic[frame_idx]
            cam_next = cam_extrinsic[next_frame_idx]
            
            relative_pose = compute_relative_pose(cam_prev, cam_next)
            relative_poses.append(torch.as_tensor(relative_pose[:3, :]))
    
    pose_embedding = torch.stack(relative_poses, dim=0)
    pose_embedding = rearrange(pose_embedding, 'b c d -> b (c d)')  # [frames, 12]
    
    # 添加mask信息
    mask = torch.zeros(total_frames, dtype=torch.float32)
    mask[:condition_frames] = 1.0  # condition frames
    mask = mask.view(-1, 1)
    
    camera_embedding = torch.cat([pose_embedding, mask], dim=1)  # [frames, 13]
    print(f"Generated camera embedding shape: {camera_embedding.shape}")
    
    return camera_embedding.to(torch.bfloat16)


def generate_synthetic_camera_poses(direction="forward", target_frames=10, condition_frames=20):
    """根据指定方向生成相机pose序列（合成数据）"""
    total_frames = condition_frames + target_frames
    poses = []
    
    for i in range(total_frames):
        t = i / max(1, total_frames - 1)
        pose = np.eye(4, dtype=np.float32)
        
        if direction == "forward":
            pose[2, 3] = -t * 0.04
        elif direction == "backward":
            pose[2, 3] = t * 2.0
        elif direction == "left_turn":
            pose[2, 3] = -t * 0.03
            pose[0, 3] = t * 0.02
            yaw = t * 1
            pose[0, 0] = np.cos(yaw)
            pose[0, 2] = np.sin(yaw)
            pose[2, 0] = -np.sin(yaw)
            pose[2, 2] = np.cos(yaw)
        elif direction == "right_turn":
            pose[2, 3] = -t * 0.03
            pose[0, 3] = -t * 0.02
            yaw = -t * 1
            pose[0, 0] = np.cos(yaw)
            pose[0, 2] = np.sin(yaw)
            pose[2, 0] = -np.sin(yaw)
            pose[2, 2] = np.cos(yaw)
            
        poses.append(pose)
    
    # 计算相对pose
    relative_poses = []
    for i in range(len(poses) - 1):
        relative_pose = compute_relative_pose(poses[i], poses[i + 1])
        relative_poses.append(torch.as_tensor(relative_pose[:3, :]))
    
    if len(relative_poses) < total_frames:
        relative_poses.append(relative_poses[-1])
    
    pose_embedding = torch.stack(relative_poses[:total_frames], dim=0)
    pose_embedding = rearrange(pose_embedding, 'b c d -> b (c d)')  # [frames, 12]
    
    # 添加mask信息
    mask = torch.zeros(total_frames, dtype=torch.float32)
    mask[:condition_frames] = 1.0
    mask = mask.view(-1, 1)
    
    camera_embedding = torch.cat([pose_embedding, mask], dim=1)  # [frames, 13]
    print(f"Generated {direction} movement poses: {camera_embedding.shape}")
    
    return camera_embedding.to(torch.bfloat16)


def replace_dit_model_in_manager():
    """替换DiT模型类为FramePack版本"""
    from diffsynth.models.wan_video_dit_recam_future import WanModelFuture
    from diffsynth.configs.model_config import model_loader_configs
    
    for i, config in enumerate(model_loader_configs):
        keys_hash, keys_hash_with_shape, model_names, model_classes, model_resource = config
        
        if 'wan_video_dit' in model_names:
            new_model_names = []
            new_model_classes = []
            
            for name, cls in zip(model_names, model_classes):
                if name == 'wan_video_dit':
                    new_model_names.append(name)
                    new_model_classes.append(WanModelFuture)
                    print(f"✅ 替换了模型类: {name} -> WanModelFuture")
                else:
                    new_model_names.append(name)
                    new_model_classes.append(cls)
            
            model_loader_configs[i] = (keys_hash, keys_hash_with_shape, new_model_names, new_model_classes, model_resource)

def add_framepack_components(dit_model):
    """添加FramePack相关组件"""
    if not hasattr(dit_model, 'clean_x_embedder'):
        inner_dim = dit_model.blocks[0].self_attn.q.weight.shape[0]
        
        class CleanXEmbedder(nn.Module):
            def __init__(self, inner_dim):
                super().__init__()
                self.proj = nn.Conv3d(16, inner_dim, kernel_size=(1, 2, 2), stride=(1, 2, 2))
                self.proj_2x = nn.Conv3d(16, inner_dim, kernel_size=(2, 4, 4), stride=(2, 4, 4))
                self.proj_4x = nn.Conv3d(16, inner_dim, kernel_size=(4, 8, 8), stride=(4, 8, 8))
            
            def forward(self, x, scale="1x"):
                # 🔧 确保输入和权重的数据类型匹配
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
        # 🔧 修复：使用模型参数的dtype而不是模型的dtype属性
        model_dtype = next(dit_model.parameters()).dtype
        dit_model.clean_x_embedder = dit_model.clean_x_embedder.to(dtype=model_dtype)
        print("✅ 添加了FramePack的clean_x_embedder组件")

def inference_sekai_framepack_from_pth(
    condition_pth_path,
    dit_path,
    output_path="sekai/infer_results/output_sekai_framepack.mp4",
    start_frame=0,
    condition_frames=10,
    target_frames=2,
    device="cuda",
    prompt="A video of a scene shot using a pedestrian's front camera while walking",
    direction="forward",
    use_real_poses=True
):
    """
    FramePack风格的Sekai视频推理
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Setting up FramePack models for {direction} movement...")
    
    # 1. 替换模型类并加载模型
    replace_dit_model_in_manager()
    
    model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
    model_manager.load_models([
        "models/Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors",
        "models/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
        "models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
    ])
    pipe = WanVideoReCamMasterPipeline.from_model_manager(model_manager, device="cuda")

    # 2. 添加camera components和FramePack components
    dim = pipe.dit.blocks[0].self_attn.q.weight.shape[0]
    for block in pipe.dit.blocks:
        block.cam_encoder = nn.Linear(13, dim)
        block.projector = nn.Linear(dim, dim)
        block.cam_encoder.weight.data.zero_()
        block.cam_encoder.bias.data.zero_()
        block.projector.weight = nn.Parameter(torch.eye(dim))
        block.projector.bias = nn.Parameter(torch.zeros(dim))
    
    # 添加FramePack组件
    add_framepack_components(pipe.dit)
    
    # 3. 加载训练的权重
    dit_state_dict = torch.load(dit_path, map_location="cpu")
    pipe.dit.load_state_dict(dit_state_dict, strict=True)
    
    pipe = pipe.to(device)
    model_dtype = next(pipe.dit.parameters()).dtype
    pipe.dit = pipe.dit.to(dtype=model_dtype)
    if hasattr(pipe.dit, 'clean_x_embedder'):
        pipe.dit.clean_x_embedder = pipe.dit.clean_x_embedder.to(dtype=model_dtype)
    
    pipe.scheduler.set_timesteps(50)
    print("Loading condition video from pth...")
    
    # 4. 加载条件视频数据
    condition_latents, encoded_data = load_encoded_video_from_pth(
        condition_pth_path, 
        start_frame=start_frame,
        num_frames=condition_frames
    )
    
    print("Preparing FramePack inputs...")
    
    # 5. 🔧 准备FramePack风格的多尺度输入
    full_latents = encoded_data['latents']
    framepack_inputs = prepare_framepack_inputs(
        full_latents, condition_frames, target_frames, start_frame
    )
    
    # 🔧 转换为正确的设备和数据类型，确保与DiT模型一致
    for key in framepack_inputs:
        if torch.is_tensor(framepack_inputs[key]):
            framepack_inputs[key] = framepack_inputs[key].to(device, dtype=model_dtype)
    
    print("Processing poses...")
    
    # 6. 生成相机pose embedding
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
        camera_embedding = generate_synthetic_camera_poses(
            direction=direction,
            target_frames=target_frames,
            condition_frames=condition_frames
        )
    
    camera_embedding = camera_embedding.unsqueeze(0).to(device, dtype=model_dtype)
    print("Encoding prompt...")
    
    # 7. 编码文本提示
    prompt_emb = pipe.encode_prompt(prompt)
    print("Generating video...")
    
    # 8. 生成目标latents
    batch_size = 1
    channels = framepack_inputs['latents'].shape[0]  # 现在latents没有batch维度
    latent_height = framepack_inputs['latents'].shape[2]
    latent_width = framepack_inputs['latents'].shape[3]
    
    # 空间裁剪以节省内存
    target_height, target_width = 60, 104
    
    if latent_height > target_height or latent_width > target_width:
        h_start = (latent_height - target_height) // 2
        w_start = (latent_width - target_width) // 2
        
        # 裁剪所有inputs
        for key in ['latents', 'clean_latents', 'clean_latents_2x', 'clean_latents_4x']:
            if key in framepack_inputs and torch.is_tensor(framepack_inputs[key]):
                framepack_inputs[key] = framepack_inputs[key][:, :, 
                    h_start:h_start+target_height, 
                    w_start:w_start+target_width]
        
        latent_height = target_height
        latent_width = target_width
    
    # 为推理添加batch维度
    for key in ['latents', 'clean_latents', 'clean_latents_2x', 'clean_latents_4x']:
        if key in framepack_inputs and torch.is_tensor(framepack_inputs[key]):
            framepack_inputs[key] = framepack_inputs[key].unsqueeze(0)
    
    # 🔧 修复：为索引张量添加batch维度并确保正确的数据类型
    for key in ['latent_indices', 'clean_latent_indices', 'clean_latent_2x_indices', 'clean_latent_4x_indices']:
        if key in framepack_inputs and torch.is_tensor(framepack_inputs[key]):
            # 确保索引是long类型，并且在CPU上
            framepack_inputs[key] = framepack_inputs[key].long().cpu().unsqueeze(0)
    
    # 初始化target latents with noise
    target_latents = torch.randn(
        batch_size, channels, target_frames, latent_height, latent_width,
        device=device, dtype=model_dtype  # 🔧 使用模型的dtype
    )
    
    print(f"FramePack inputs:")
    for key, value in framepack_inputs.items():
        if torch.is_tensor(value):
            print(f"  {key}: {value.shape} {value.dtype}")
        else:
            print(f"  {key}: {value}")
    print(f"Camera embedding shape: {camera_embedding.shape}")
    print(f"Target latents shape: {target_latents.shape}")
    
    # 9. 准备额外输入
    extra_input = pipe.prepare_extra_input(target_latents)
    
    # 10. 🔧 FramePack风格的去噪循环
    timesteps = pipe.scheduler.timesteps
    
    for i, timestep in enumerate(timesteps):
        print(f"Denoising step {i+1}/{len(timesteps)}")
        
        timestep_tensor = timestep.unsqueeze(0).to(device, dtype=model_dtype)
        
        # 🔧 使用FramePack风格的forward调用
        with torch.no_grad():
            noise_pred = pipe.dit(
                target_latents,
                timestep=timestep_tensor,
                cam_emb=camera_embedding,
                # FramePack参数
                latent_indices=framepack_inputs['latent_indices'],
                clean_latents=framepack_inputs['clean_latents'],
                clean_latent_indices=framepack_inputs['clean_latent_indices'],
                clean_latents_2x=framepack_inputs['clean_latents_2x'],
                clean_latent_2x_indices=framepack_inputs['clean_latent_2x_indices'],
                clean_latents_4x=framepack_inputs['clean_latents_4x'],
                clean_latent_4x_indices=framepack_inputs['clean_latent_4x_indices'],
                **prompt_emb,
                **extra_input
            )
        
        # 更新target latents
        target_latents = pipe.scheduler.step(noise_pred, timestep, target_latents)
    
    print("Decoding video...")
    
    # 11. 解码最终视频
    # 拼接condition和target用于解码
    condition_for_decode = framepack_inputs['clean_latents'][:, :, -1:, :, :]  # 取最后一帧作为条件
    final_video = torch.cat([condition_for_decode, target_latents], dim=2)
    decoded_video = pipe.decode_video(final_video, tiled=True, tile_size=(34, 34), tile_stride=(18, 16))

    # 12. 保存视频
    print(f"Saving video to {output_path}")
    
    video_np = decoded_video[0].to(torch.float32).permute(1, 2, 3, 0).cpu().numpy()
    video_np = (video_np * 0.5 + 0.5).clip(0, 1)
    video_np = (video_np * 255).astype(np.uint8)

    with imageio.get_writer(output_path, fps=20) as writer:
        for frame in video_np:
            writer.append_data(frame)

    print(f"FramePack video generation completed! Saved to {output_path}")
        
def main():
    parser = argparse.ArgumentParser(description="Sekai FramePack Video Generation Inference from PTH")
    parser.add_argument("--condition_pth", type=str,
                       default="/share_zhuyixuan05/zhuyixuan05/sekai-game-walking/00100100001_0004650_0004950/encoded_video.pth")
    parser.add_argument("--start_frame", type=int, default=0,
                       help="Starting frame index (compressed latent frames)")
    parser.add_argument("--condition_frames", type=int, default=8,
                       help="Number of condition frames (compressed latent frames)")
    parser.add_argument("--target_frames", type=int, default=8,
                       help="Number of target frames to generate (compressed latent frames)")
    parser.add_argument("--direction", type=str, default="left_turn",
                       choices=["forward", "backward", "left_turn", "right_turn"],
                       help="Direction of camera movement (if not using real poses)")
    parser.add_argument("--use_real_poses", action="store_true", default=False,
                       help="Use real camera poses from data")
    parser.add_argument("--dit_path", type=str, 
                       default="/share_zhuyixuan05/zhuyixuan05/ICLR2026/sekai/sekai_walking_framepack/step24000_framepack.ckpt",
                       help="Path to trained FramePack DiT checkpoint")
    parser.add_argument("--output_path", type=str, 
                       default='/home/zhuyixuan05/ReCamMaster/sekai/infer_framepack_results/output_sekai_framepack.mp4',
                       help="Output video path")
    parser.add_argument("--prompt", type=str, 
                       default="A drone flying scene in a game world",
                       help="Text prompt for generation")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to run inference on")
    
    args = parser.parse_args()
    
    # 生成输出路径
    if args.output_path is None:
        pth_filename = os.path.basename(args.condition_pth)
        name_parts = os.path.splitext(pth_filename)
        output_dir = "sekai/infer_framepack_results"
        os.makedirs(output_dir, exist_ok=True)
        
        if args.use_real_poses:
            output_filename = f"{name_parts[0]}_framepack_real_{args.start_frame}_{args.condition_frames}_{args.target_frames}.mp4"
        else:
            output_filename = f"{name_parts[0]}_framepack_{args.direction}_{args.start_frame}_{args.condition_frames}_{args.target_frames}.mp4"
        
        output_path = os.path.join(output_dir, output_filename)
    else:
        output_path = args.output_path

    print(f"🔧 FramePack Inference Settings:")
    print(f"Input pth: {args.condition_pth}")
    print(f"Start frame: {args.start_frame} (compressed)")
    print(f"Condition frames: {args.condition_frames} (compressed, original: {args.condition_frames * 4})")
    print(f"Target frames: {args.target_frames} (compressed, original: {args.target_frames * 4})")
    print(f"Use real poses: {args.use_real_poses}")
    print(f"Direction: {args.direction}")
    print(f"Output video will be saved to: {output_path}")
    
    inference_sekai_framepack_from_pth(
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