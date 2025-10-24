from PIL import Image
from diffsynth import WanVideoReCamMasterPipeline, ModelManager
from torchvision.transforms import v2
from einops import rearrange
import os
import torch
import torch.nn as nn
import argparse
import numpy as np
import imageio
import copy
import random

def load_encoded_video_from_pth(pth_path, start_frame=0, num_frames=10):
    """从pth文件加载预编码的视频数据"""
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

def replace_dit_model_in_manager():
    """在模型加载前替换DiT模型类"""
    from diffsynth.models.wan_video_dit_recam_future import WanModelFuture
    from diffsynth.configs.model_config import model_loader_configs
    
    # 修改model_loader_configs中的配置
    for i, config in enumerate(model_loader_configs):
        keys_hash, keys_hash_with_shape, model_names, model_classes, model_resource = config
        
        # 检查是否包含wan_video_dit模型
        if 'wan_video_dit' in model_names:
            # 找到wan_video_dit的索引并替换为WanModelFuture
            new_model_names = []
            new_model_classes = []
            
            for name, cls in zip(model_names, model_classes):
                if name == 'wan_video_dit':
                    new_model_names.append(name)  # 保持名称不变
                    new_model_classes.append(WanModelFuture)  # 替换为新的类
                    print(f"✅ 替换了模型类: {name} -> WanModelFuture")
                else:
                    new_model_names.append(name)
                    new_model_classes.append(cls)
            
            # 更新配置
            model_loader_configs[i] = (keys_hash, keys_hash_with_shape, new_model_names, new_model_classes, model_resource)

def add_framepack_components(dit_model):
    """添加FramePack相关组件"""
    if not hasattr(dit_model, 'clean_x_embedder'):
        inner_dim = dit_model.blocks[0].self_attn.q.weight.shape[0]
        
        class CleanXEmbedder(nn.Module):
            def __init__(self, inner_dim):
                super().__init__()
                # 参考hunyuan_video_packed.py的设计
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
        
        dit_model.clean_x_embedder = CleanXEmbedder(inner_dim)
        model_dtype = next(dit_model.parameters()).dtype
        dit_model.clean_x_embedder = dit_model.clean_x_embedder.to(dtype=model_dtype)
        print("✅ 添加了FramePack的clean_x_embedder组件")

def generate_openx_camera_embeddings_sliding(cam_data, start_frame, current_history_length, new_frames, total_generated, use_real_poses=True):
    """为OpenX数据集生成camera embeddings - 滑动窗口版本"""
    time_compression_ratio = 4
    
    # 计算FramePack实际需要的camera帧数
    framepack_needed_frames = 1 + 16 + 2 + 1 + new_frames
    
    if use_real_poses and cam_data is not None and 'extrinsic' in cam_data:
        print("🔧 使用真实OpenX camera数据")
        cam_extrinsic = cam_data['extrinsic']
        
        # 确保生成足够长的camera序列
        max_needed_frames = max(
            start_frame + current_history_length + new_frames,
            framepack_needed_frames,
            30
        )
        
        print(f"🔧 计算OpenX camera序列长度:")
        print(f"  - 基础需求: {start_frame + current_history_length + new_frames}")
        print(f"  - FramePack需求: {framepack_needed_frames}")
        print(f"  - 最终生成: {max_needed_frames}")
        
        relative_poses = []
        for i in range(max_needed_frames):
            # OpenX特有：每隔4帧
            frame_idx = i * time_compression_ratio
            next_frame_idx = frame_idx + time_compression_ratio
            
            if next_frame_idx < len(cam_extrinsic):
                cam_prev = cam_extrinsic[frame_idx]
                cam_next = cam_extrinsic[next_frame_idx]
                relative_cam = compute_relative_pose(cam_prev, cam_next)
                relative_poses.append(torch.as_tensor(relative_cam[:3, :]))
            else:
                # 超出范围，使用零运动
                print(f"⚠️ 帧{frame_idx}超出camera数据范围，使用零运动")
                relative_poses.append(torch.zeros(3, 4))
        
        pose_embedding = torch.stack(relative_poses, dim=0)
        pose_embedding = rearrange(pose_embedding, 'b c d -> b (c d)')
        
        # 创建对应长度的mask序列
        mask = torch.zeros(max_needed_frames, 1, dtype=torch.float32)
        # 从start_frame到current_history_length标记为condition
        condition_end = min(start_frame + current_history_length, max_needed_frames)
        mask[start_frame:condition_end] = 1.0
        
        camera_embedding = torch.cat([pose_embedding, mask], dim=1)
        print(f"🔧 OpenX真实camera embedding shape: {camera_embedding.shape}")
        return camera_embedding.to(torch.bfloat16)
        
    else:
        print("🔧 使用OpenX合成camera数据")
        
        max_needed_frames = max(
            start_frame + current_history_length + new_frames,
            framepack_needed_frames,
            30
        )
        
        print(f"🔧 生成OpenX合成camera帧数: {max_needed_frames}")
        relative_poses = []
        for i in range(max_needed_frames):
            # OpenX机器人操作模式 - 稳定的小幅度运动
            # 模拟机器人手臂的精细操作
            forward_speed = 0.001  # 每帧前进距离（很小，因为是精细操作）
            lateral_motion = 0.0005 * np.sin(i * 0.05)  # 轻微的左右移动
            vertical_motion = 0.0003 * np.cos(i * 0.1)  # 轻微的上下移动
            
            # 旋转变化（模拟视角微调）
            yaw_change = 0.01 * np.sin(i * 0.03)  # 轻微的偏航
            pitch_change = 0.008 * np.cos(i * 0.04)  # 轻微的俯仰
            
            pose = np.eye(4, dtype=np.float32)
            
            # 旋转矩阵（绕Y轴和X轴的小角度旋转）
            cos_yaw = np.cos(yaw_change)
            sin_yaw = np.sin(yaw_change)
            cos_pitch = np.cos(pitch_change)
            sin_pitch = np.sin(pitch_change)
            
            # 组合旋转（先pitch后yaw）
            pose[0, 0] = cos_yaw
            pose[0, 2] = sin_yaw
            pose[1, 1] = cos_pitch
            pose[1, 2] = -sin_pitch
            pose[2, 0] = -sin_yaw
            pose[2, 1] = sin_pitch
            pose[2, 2] = cos_yaw * cos_pitch
            
            # 平移（精细操作的小幅度移动）
            pose[0, 3] = lateral_motion  # X轴（左右）
            pose[1, 3] = vertical_motion  # Y轴（上下）
            pose[2, 3] = -forward_speed  # Z轴（前后，负值表示前进）
            
            relative_pose = pose[:3, :]
            relative_poses.append(torch.as_tensor(relative_pose))
        
        pose_embedding = torch.stack(relative_poses, dim=0)
        pose_embedding = rearrange(pose_embedding, 'b c d -> b (c d)')
        
        # 创建对应长度的mask序列
        mask = torch.zeros(max_needed_frames, 1, dtype=torch.float32)
        condition_end = min(start_frame + current_history_length, max_needed_frames)
        mask[start_frame:condition_end] = 1.0
        
        camera_embedding = torch.cat([pose_embedding, mask], dim=1)
        print(f"🔧 OpenX合成camera embedding shape: {camera_embedding.shape}")
        return camera_embedding.to(torch.bfloat16)

def prepare_framepack_sliding_window_with_camera(history_latents, target_frames_to_generate, camera_embedding_full, start_frame, max_history_frames=49):
    """FramePack滑动窗口机制 - OpenX版本"""
    # history_latents: [C, T, H, W] 当前的历史latents
    C, T, H, W = history_latents.shape
    
    # 固定索引结构（这决定了需要的camera帧数）
    total_indices_length = 1 + 16 + 2 + 1 + target_frames_to_generate
    indices = torch.arange(0, total_indices_length)
    split_sizes = [1, 16, 2, 1, target_frames_to_generate]
    clean_latent_indices_start, clean_latent_4x_indices, clean_latent_2x_indices, clean_latent_1x_indices, latent_indices = \
        indices.split(split_sizes, dim=0)
    clean_latent_indices = torch.cat([clean_latent_indices_start, clean_latent_1x_indices], dim=0)
    
    # 检查camera长度是否足够
    if camera_embedding_full.shape[0] < total_indices_length:
        shortage = total_indices_length - camera_embedding_full.shape[0]
        padding = torch.zeros(shortage, camera_embedding_full.shape[1], 
                            dtype=camera_embedding_full.dtype, device=camera_embedding_full.device)
        camera_embedding_full = torch.cat([camera_embedding_full, padding], dim=0)
    
    # 从完整camera序列中选取对应部分
    combined_camera = camera_embedding_full[:total_indices_length, :].clone()
    
    # 根据当前history length重新设置mask
    combined_camera[:, -1] = 0.0  # 先全部设为target (0)
    
    # 设置condition mask：前19帧根据实际历史长度决定
    if T > 0:
        available_frames = min(T, 19)
        start_pos = 19 - available_frames
        combined_camera[start_pos:19, -1] = 1.0  # 将有效的clean latents对应的camera标记为condition
    
    print(f"🔧 OpenX Camera mask更新:")
    print(f"  - 历史帧数: {T}")
    print(f"  - 有效condition帧数: {available_frames if T > 0 else 0}")
    
    # 处理latents
    clean_latents_combined = torch.zeros(C, 19, H, W, dtype=history_latents.dtype, device=history_latents.device)
    
    if T > 0:
        available_frames = min(T, 19)
        start_pos = 19 - available_frames
        clean_latents_combined[:, start_pos:, :, :] = history_latents[:, -available_frames:, :, :]
    
    clean_latents_4x = clean_latents_combined[:, 0:16, :, :]
    clean_latents_2x = clean_latents_combined[:, 16:18, :, :]
    clean_latents_1x = clean_latents_combined[:, 18:19, :, :]
    
    if T > 0:
        start_latent = history_latents[:, 0:1, :, :]
    else:
        start_latent = torch.zeros(C, 1, H, W, dtype=history_latents.dtype, device=history_latents.device)
    
    clean_latents = torch.cat([start_latent, clean_latents_1x], dim=1)
    
    return {
        'latent_indices': latent_indices,
        'clean_latents': clean_latents,
        'clean_latents_2x': clean_latents_2x,
        'clean_latents_4x': clean_latents_4x,
        'clean_latent_indices': clean_latent_indices,
        'clean_latent_2x_indices': clean_latent_2x_indices,
        'clean_latent_4x_indices': clean_latent_4x_indices,
        'camera_embedding': combined_camera,
        'current_length': T,
        'next_length': T + target_frames_to_generate
    }

def inference_openx_framepack_sliding_window(
    condition_pth_path,
    dit_path,
    output_path="openx_results/output_openx_framepack_sliding.mp4",
    start_frame=0,
    initial_condition_frames=8,
    frames_per_generation=4,
    total_frames_to_generate=32,
    max_history_frames=49,
    device="cuda",
    prompt="A video of robotic manipulation task with camera movement",
    use_real_poses=True,
    # CFG参数
    use_camera_cfg=True,
    camera_guidance_scale=2.0,
    text_guidance_scale=1.0
):
    """
    OpenX FramePack滑动窗口视频生成
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"🔧 OpenX FramePack滑动窗口生成开始...")
    print(f"Camera CFG: {use_camera_cfg}, Camera guidance scale: {camera_guidance_scale}")
    print(f"Text guidance scale: {text_guidance_scale}")
    
    # 1. 模型初始化
    replace_dit_model_in_manager()
    
    model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
    model_manager.load_models([
        "models/Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors",
        "models/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
        "models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
    ])
    pipe = WanVideoReCamMasterPipeline.from_model_manager(model_manager, device="cuda")

    # 2. 添加camera编码器
    dim = pipe.dit.blocks[0].self_attn.q.weight.shape[0]
    for block in pipe.dit.blocks:
        block.cam_encoder = nn.Linear(13, dim)
        block.projector = nn.Linear(dim, dim)
        block.cam_encoder.weight.data.zero_()
        block.cam_encoder.bias.data.zero_()
        block.projector.weight = nn.Parameter(torch.eye(dim))
        block.projector.bias = nn.Parameter(torch.zeros(dim))
    
    # 3. 添加FramePack组件
    add_framepack_components(pipe.dit)
    
    # 4. 加载训练好的权重
    dit_state_dict = torch.load(dit_path, map_location="cpu")
    pipe.dit.load_state_dict(dit_state_dict, strict=True)
    pipe = pipe.to(device)
    model_dtype = next(pipe.dit.parameters()).dtype
    
    if hasattr(pipe.dit, 'clean_x_embedder'):
        pipe.dit.clean_x_embedder = pipe.dit.clean_x_embedder.to(dtype=model_dtype)
    
    pipe.scheduler.set_timesteps(50)
    
    # 5. 加载初始条件
    print("Loading initial condition frames...")
    initial_latents, encoded_data = load_encoded_video_from_pth(
        condition_pth_path, 
        start_frame=start_frame,
        num_frames=initial_condition_frames
    )
    
    # 空间裁剪（适配OpenX数据尺寸）
    target_height, target_width = 60, 104
    C, T, H, W = initial_latents.shape
    
    if H > target_height or W > target_width:
        h_start = (H - target_height) // 2
        w_start = (W - target_width) // 2
        initial_latents = initial_latents[:, :, h_start:h_start+target_height, w_start:w_start+target_width]
        H, W = target_height, target_width
    
    history_latents = initial_latents.to(device, dtype=model_dtype)
    
    print(f"初始history_latents shape: {history_latents.shape}")
    
    # 6. 编码prompt - 支持CFG
    if text_guidance_scale > 1.0:
        prompt_emb_pos = pipe.encode_prompt(prompt)
        prompt_emb_neg = pipe.encode_prompt("")
        print(f"使用Text CFG，guidance scale: {text_guidance_scale}")
    else:
        prompt_emb_pos = pipe.encode_prompt(prompt)
        prompt_emb_neg = None
        print("不使用Text CFG")
    
    # 7. 预生成完整的camera embedding序列
    camera_embedding_full = generate_openx_camera_embeddings_sliding(
        encoded_data.get('cam_emb', None),
        0,
        max_history_frames,
        0,
        0,
        use_real_poses=use_real_poses
    ).to(device, dtype=model_dtype)
    
    print(f"完整camera序列shape: {camera_embedding_full.shape}")
    
    # 8. 为Camera CFG创建无条件的camera embedding
    if use_camera_cfg:
        camera_embedding_uncond = torch.zeros_like(camera_embedding_full)
        print(f"创建无条件camera embedding用于CFG")
    
    # 9. 滑动窗口生成循环
    total_generated = 0
    all_generated_frames = []
    
    while total_generated < total_frames_to_generate:
        current_generation = min(frames_per_generation, total_frames_to_generate - total_generated)
        print(f"\n🔧 生成步骤 {total_generated // frames_per_generation + 1}")
        print(f"当前历史长度: {history_latents.shape[1]}, 本次生成: {current_generation}")
        
        # FramePack数据准备 - OpenX版本
        framepack_data = prepare_framepack_sliding_window_with_camera(
            history_latents,
            current_generation,
            camera_embedding_full,
            start_frame,
            max_history_frames
        )
        
        # 准备输入
        clean_latents = framepack_data['clean_latents'].unsqueeze(0)
        clean_latents_2x = framepack_data['clean_latents_2x'].unsqueeze(0)
        clean_latents_4x = framepack_data['clean_latents_4x'].unsqueeze(0)
        camera_embedding = framepack_data['camera_embedding'].unsqueeze(0)
        
        # 为CFG准备无条件camera embedding
        if use_camera_cfg:
            camera_embedding_uncond_batch = camera_embedding_uncond[:camera_embedding.shape[1], :].unsqueeze(0)
        
        # 索引处理
        latent_indices = framepack_data['latent_indices'].unsqueeze(0).cpu()
        clean_latent_indices = framepack_data['clean_latent_indices'].unsqueeze(0).cpu()
        clean_latent_2x_indices = framepack_data['clean_latent_2x_indices'].unsqueeze(0).cpu()
        clean_latent_4x_indices = framepack_data['clean_latent_4x_indices'].unsqueeze(0).cpu()
        
        # 初始化要生成的latents
        new_latents = torch.randn(
            1, C, current_generation, H, W,
            device=device, dtype=model_dtype
        )
        
        extra_input = pipe.prepare_extra_input(new_latents)
        
        print(f"Camera embedding shape: {camera_embedding.shape}")
        print(f"Camera mask分布 - condition: {torch.sum(camera_embedding[0, :, -1] == 1.0).item()}, target: {torch.sum(camera_embedding[0, :, -1] == 0.0).item()}")
        
        # 去噪循环 - 支持CFG
        timesteps = pipe.scheduler.timesteps
        
        for i, timestep in enumerate(timesteps):
            if i % 10 == 0:
                print(f"  去噪步骤 {i}/{len(timesteps)}")
            
            timestep_tensor = timestep.unsqueeze(0).to(device, dtype=model_dtype)
            
            with torch.no_grad():
                # 正向预测（带条件）
                noise_pred_pos = pipe.dit(
                    new_latents,
                    timestep=timestep_tensor,
                    cam_emb=camera_embedding,
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
                
                # CFG处理
                if use_camera_cfg and camera_guidance_scale > 1.0:
                    # 无条件预测（无camera条件）
                    noise_pred_uncond = pipe.dit(
                        new_latents,
                        timestep=timestep_tensor,
                        cam_emb=camera_embedding_uncond_batch,
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
                    
                    # Camera CFG
                    noise_pred = noise_pred_uncond + camera_guidance_scale * (noise_pred_pos - noise_pred_uncond)
                else:
                    noise_pred = noise_pred_pos
                
                # Text CFG
                if prompt_emb_neg is not None and text_guidance_scale > 1.0:
                    noise_pred_text_uncond = pipe.dit(
                        new_latents,
                        timestep=timestep_tensor,
                        cam_emb=camera_embedding,
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
                    
                    # Text CFG
                    noise_pred = noise_pred_text_uncond + text_guidance_scale * (noise_pred - noise_pred_text_uncond)
            
            new_latents = pipe.scheduler.step(noise_pred, timestep, new_latents)
        
        # 更新历史
        new_latents_squeezed = new_latents.squeeze(0)
        history_latents = torch.cat([history_latents, new_latents_squeezed], dim=1)
        
        # 维护滑动窗口
        if history_latents.shape[1] > max_history_frames:
            first_frame = history_latents[:, 0:1, :, :]
            recent_frames = history_latents[:, -(max_history_frames-1):, :, :]
            history_latents = torch.cat([first_frame, recent_frames], dim=1)
            print(f"历史窗口已满，保留第一帧+最新{max_history_frames-1}帧")
        
        print(f"更新后history_latents shape: {history_latents.shape}")
        
        all_generated_frames.append(new_latents_squeezed)
        total_generated += current_generation
        
        print(f"✅ 已生成 {total_generated}/{total_frames_to_generate} 帧")
    
    # 10. 解码和保存
    print("\n🔧 解码生成的视频...")
    
    all_generated = torch.cat(all_generated_frames, dim=1)
    final_video = torch.cat([initial_latents.to(all_generated.device), all_generated], dim=1).unsqueeze(0)
    
    print(f"最终视频shape: {final_video.shape}")
    
    decoded_video = pipe.decode_video(final_video, tiled=True, tile_size=(34, 34), tile_stride=(18, 16))
    
    print(f"Saving video to {output_path}")
    
    video_np = decoded_video[0].to(torch.float32).permute(1, 2, 3, 0).cpu().numpy()
    video_np = (video_np * 0.5 + 0.5).clip(0, 1)
    video_np = (video_np * 255).astype(np.uint8)

    with imageio.get_writer(output_path, fps=20) as writer:
        for frame in video_np:
            writer.append_data(frame)

    print(f"🔧 OpenX FramePack滑动窗口生成完成! 保存到: {output_path}")
    print(f"总共生成了 {total_generated} 帧 (压缩后), 对应原始 {total_generated * 4} 帧")

def main():
    parser = argparse.ArgumentParser(description="OpenX FramePack滑动窗口视频生成")
    
    # 基础参数
    parser.add_argument("--condition_pth", type=str,
                       default="/share_zhuyixuan05/zhuyixuan05/openx-fractal-encoded/episode_000001/encoded_video.pth",
                       help="输入编码视频路径")
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--initial_condition_frames", type=int, default=16)
    parser.add_argument("--frames_per_generation", type=int, default=8)
    parser.add_argument("--total_frames_to_generate", type=int, default=24)
    parser.add_argument("--max_history_frames", type=int, default=100)
    parser.add_argument("--use_real_poses", action="store_true", default=False)
    parser.add_argument("--dit_path", type=str, 
                       default="/share_zhuyixuan05/zhuyixuan05/ICLR2026/openx/openx_framepack/step2000.ckpt",
                       help="训练好的模型权重路径")
    parser.add_argument("--output_path", type=str, 
                       default='openx_results/output_openx_framepack_sliding.mp4')
    parser.add_argument("--prompt", type=str, 
                       default="A video of robotic manipulation task with camera movement")
    parser.add_argument("--device", type=str, default="cuda")
    
    # CFG参数
    parser.add_argument("--use_camera_cfg", action="store_true", default=True,
                       help="使用Camera CFG")
    parser.add_argument("--camera_guidance_scale", type=float, default=2.0,
                       help="Camera guidance scale for CFG")
    parser.add_argument("--text_guidance_scale", type=float, default=1.0,
                       help="Text guidance scale for CFG")
    
    args = parser.parse_args()

    print(f"🔧 OpenX FramePack CFG生成设置:")
    print(f"Camera CFG: {args.use_camera_cfg}")
    if args.use_camera_cfg:
        print(f"Camera guidance scale: {args.camera_guidance_scale}")
    print(f"Text guidance scale: {args.text_guidance_scale}")
    print(f"OpenX特有特性: camera间隔为4帧，适用于机器人操作任务")
    
    inference_openx_framepack_sliding_window(
        condition_pth_path=args.condition_pth,
        dit_path=args.dit_path,
        output_path=args.output_path,
        start_frame=args.start_frame,
        initial_condition_frames=args.initial_condition_frames,
        frames_per_generation=args.frames_per_generation,
        total_frames_to_generate=args.total_frames_to_generate,
        max_history_frames=args.max_history_frames,
        device=args.device,
        prompt=args.prompt,
        use_real_poses=args.use_real_poses,
        # CFG参数
        use_camera_cfg=args.use_camera_cfg,
        camera_guidance_scale=args.camera_guidance_scale,
        text_guidance_scale=args.text_guidance_scale
    )

if __name__ == "__main__":
    main()