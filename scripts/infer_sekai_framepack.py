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
        print("✅ 添加了FramePack的clean_x_embedder组件")
        
def generate_camera_embeddings_sliding(cam_data, start_frame, current_history_length, new_frames, total_generated, use_real_poses=True):
    """🔧 为滑动窗口生成camera embeddings - 修正长度计算，确保包含start_latent帧"""
    time_compression_ratio = 4
    
    # 🔧 计算FramePack实际需要的camera帧数
    # FramePack结构: 1(start) + 16(4x) + 2(2x) + 1(1x) + target_frames
    framepack_needed_frames = 1 + 16 + 2 + 1 + new_frames
    
    if use_real_poses and cam_data is not None and 'extrinsic' in cam_data:
        print("🔧 使用真实camera数据")
        cam_extrinsic = cam_data['extrinsic']
        
        # 🔧 确保生成足够长的camera序列
        # 需要考虑：当前历史位置 + FramePack所需的完整结构
        max_needed_frames = max(
            start_frame + current_history_length + new_frames,  # 基础需求
            framepack_needed_frames,  # FramePack结构需求
            30  # 最小保证长度
        )
        
        print(f"🔧 计算camera序列长度:")
        print(f"  - 基础需求: {start_frame + current_history_length + new_frames}")
        print(f"  - FramePack需求: {framepack_needed_frames}")
        print(f"  - 最终生成: {max_needed_frames}")
        
        relative_poses = []
        for i in range(max_needed_frames):
            # 计算当前帧在原始序列中的位置
            frame_idx = i * time_compression_ratio
            next_frame_idx = frame_idx + time_compression_ratio
            
            if next_frame_idx < len(cam_extrinsic):
                cam_prev = cam_extrinsic[frame_idx]
                cam_next = cam_extrinsic[next_frame_idx]
                relative_pose = compute_relative_pose(cam_prev, cam_next)
                relative_poses.append(torch.as_tensor(relative_pose[:3, :]))
            else:
                # 超出范围，使用零运动
                print(f"⚠️ 帧{frame_idx}超出camera数据范围，使用零运动")
                relative_poses.append(torch.zeros(3, 4))
        
        pose_embedding = torch.stack(relative_poses, dim=0)
        pose_embedding = rearrange(pose_embedding, 'b c d -> b (c d)')
        
        # 🔧 创建对应长度的mask序列
        mask = torch.zeros(max_needed_frames, 1, dtype=torch.float32)
        # 从start_frame到current_history_length标记为condition
        condition_end = min(start_frame + current_history_length, max_needed_frames)
        mask[start_frame:condition_end] = 1.0
        
        camera_embedding = torch.cat([pose_embedding, mask], dim=1)
        print(f"🔧 真实camera embedding shape: {camera_embedding.shape} (总长度:{max_needed_frames})")
        return camera_embedding.to(torch.bfloat16)
        
    else:
        print("🔧 使用合成camera数据")
        
        # 🔧 确保合成数据也有足够长度
        max_needed_frames = max(
            start_frame + current_history_length + new_frames,
            framepack_needed_frames,
            30
        )
        
        print(f"🔧 生成合成camera帧数: {max_needed_frames}")
        print(f"  - FramePack需求: {framepack_needed_frames}")
        
        relative_poses = []
        for i in range(max_needed_frames):
            # 🔧 持续左转运动模式
            # 每帧旋转一个固定角度，同时前进
            yaw_per_frame = -0.05  # 每帧左转（正角度表示左转）
            forward_speed = 0.005  # 每帧前进距离
            
            # 计算当前累积角度
            current_yaw = i * yaw_per_frame
            
            # 创建相对变换矩阵（从第i帧到第i+1帧的变换）
            pose = np.eye(4, dtype=np.float32)
            
            # 旋转矩阵（绕Y轴左转）
            cos_yaw = np.cos(yaw_per_frame)
            sin_yaw = np.sin(yaw_per_frame)
            
            pose[0, 0] = cos_yaw
            pose[0, 2] = sin_yaw
            pose[2, 0] = -sin_yaw
            pose[2, 2] = cos_yaw
            
            # 平移（在旋转后的局部坐标系中前进）
            pose[2, 3] = -forward_speed  # 局部Z轴负方向（前进）
            
            # 可选：添加轻微的向心运动，模拟圆形轨迹
            radius_drift = 0.002  # 向圆心的轻微漂移
            pose[0, 3] = radius_drift  # 局部X轴负方向（向左）
            
            relative_pose = pose[:3, :]
            relative_poses.append(torch.as_tensor(relative_pose))
        
        pose_embedding = torch.stack(relative_poses, dim=0)
        pose_embedding = rearrange(pose_embedding, 'b c d -> b (c d)')
        
        # 创建对应长度的mask序列
        mask = torch.zeros(max_needed_frames, 1, dtype=torch.float32)
        condition_end = min(start_frame + current_history_length, max_needed_frames)
        mask[start_frame:condition_end] = 1.0
        
        camera_embedding = torch.cat([pose_embedding, mask], dim=1)
        print(f"🔧 合成camera embedding shape: {camera_embedding.shape} (总长度:{max_needed_frames})")
        return camera_embedding.to(torch.bfloat16)

def prepare_framepack_sliding_window_with_camera(history_latents, target_frames_to_generate, camera_embedding_full, start_frame, max_history_frames=49):
    """🔧 FramePack滑动窗口机制 - 修正camera mask更新逻辑"""
    # history_latents: [C, T, H, W] 当前的历史latents
    C, T, H, W = history_latents.shape
    
    # 🔧 固定索引结构（这决定了需要的camera帧数）
    total_indices_length = 1 + 16 + 2 + 1 + target_frames_to_generate
    indices = torch.arange(0, total_indices_length)
    split_sizes = [1, 16, 2, 1, target_frames_to_generate]
    clean_latent_indices_start, clean_latent_4x_indices, clean_latent_2x_indices, clean_latent_1x_indices, latent_indices = \
        indices.split(split_sizes, dim=0)
    clean_latent_indices = torch.cat([clean_latent_indices_start, clean_latent_1x_indices], dim=0)
    
    # 🔧 检查camera长度是否足够
    if camera_embedding_full.shape[0] < total_indices_length:
        shortage = total_indices_length - camera_embedding_full.shape[0]
        padding = torch.zeros(shortage, camera_embedding_full.shape[1], 
                            dtype=camera_embedding_full.dtype, device=camera_embedding_full.device)
        camera_embedding_full = torch.cat([camera_embedding_full, padding], dim=0)
    
    # 🔧 从完整camera序列中选取对应部分
    combined_camera = camera_embedding_full[:total_indices_length, :].clone()  # clone to avoid modifying original
    
    # 🔧 关键修正：根据当前history length重新设置mask
    # combined_camera的结构对应: [1(start) + 16(4x) + 2(2x) + 1(1x) + target_frames]
    # 前19帧对应clean latents，后面对应target
    
    # 清空所有mask，重新设置
    combined_camera[:, -1] = 0.0  # 先全部设为target (0)
    
    # 设置condition mask：前19帧根据实际历史长度决定
    if T > 0:
        # 根据clean_latents的填充逻辑，确定哪些位置应该是condition
        available_frames = min(T, 19)
        start_pos = 19 - available_frames
        
        # 对应的camera位置也应该标记为condition
        combined_camera[start_pos:19, -1] = 1.0  # 将有效的clean latents对应的camera标记为condition
    
    # target部分保持为0（已经在上面设置）
    
    print(f"🔧 Camera mask更新:")
    print(f"  - 历史帧数: {T}")
    print(f"  - 有效condition帧数: {available_frames if T > 0 else 0}")
    print(f"  - Condition mask (前19帧): {combined_camera[:19, -1].cpu().tolist()}")
    print(f"  - Target mask (后{target_frames_to_generate}帧): {combined_camera[19:, -1].cpu().tolist()}")    
    # 其余处理逻辑保持不变...
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
        'camera_embedding': combined_camera,  # 🔧 现在包含正确更新的mask
        'current_length': T,
        'next_length': T + target_frames_to_generate
    }

def inference_sekai_framepack_sliding_window(
    condition_pth_path,
    dit_path,
    output_path="sekai/infer_results/output_sekai_framepack_sliding.mp4",
    start_frame=0,
    initial_condition_frames=8,
    frames_per_generation=4,
    total_frames_to_generate=32,
    max_history_frames=49,
    device="cuda",
    prompt="A video of a scene shot using a pedestrian's front camera while walking",
    use_real_poses=True,
    synthetic_direction="forward",
    # 🔧 新增CFG参数
    use_camera_cfg=True,
    camera_guidance_scale=2.0,
    text_guidance_scale=7.5
):
    """
    🔧 FramePack滑动窗口视频生成 - 支持Camera CFG
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"🔧 FramePack滑动窗口生成开始...")
    print(f"Camera CFG: {use_camera_cfg}, Camera guidance scale: {camera_guidance_scale}")
    print(f"Text guidance scale: {text_guidance_scale}")
    print(f"初始条件帧: {initial_condition_frames}, 每次生成: {frames_per_generation}, 总生成: {total_frames_to_generate}")
    print(f"使用真实姿态: {use_real_poses}")
    if not use_real_poses:
        print(f"合成camera方向: {synthetic_direction}")
    
    # 1-3. 模型初始化和组件添加（保持不变）
    replace_dit_model_in_manager()
    
    model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
    model_manager.load_models([
        "models/Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors",
        "models/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
        "models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
    ])
    pipe = WanVideoReCamMasterPipeline.from_model_manager(model_manager, device="cuda")

    dim = pipe.dit.blocks[0].self_attn.q.weight.shape[0]
    for block in pipe.dit.blocks:
        block.cam_encoder = nn.Linear(13, dim)
        block.projector = nn.Linear(dim, dim)
        block.cam_encoder.weight.data.zero_()
        block.cam_encoder.bias.data.zero_()
        block.projector.weight = nn.Parameter(torch.eye(dim))
        block.projector.bias = nn.Parameter(torch.zeros(dim))
    
    add_framepack_components(pipe.dit)
    
    dit_state_dict = torch.load(dit_path, map_location="cpu")
    pipe.dit.load_state_dict(dit_state_dict, strict=True)
    pipe = pipe.to(device)
    model_dtype = next(pipe.dit.parameters()).dtype
    
    if hasattr(pipe.dit, 'clean_x_embedder'):
        pipe.dit.clean_x_embedder = pipe.dit.clean_x_embedder.to(dtype=model_dtype)
    
    pipe.scheduler.set_timesteps(50)
    
    # 4. 加载初始条件
    print("Loading initial condition frames...")
    initial_latents, encoded_data = load_encoded_video_from_pth(
        condition_pth_path, 
        start_frame=start_frame,
        num_frames=initial_condition_frames
    )
    
    # 空间裁剪
    target_height, target_width = 60, 104
    C, T, H, W = initial_latents.shape
    
    if H > target_height or W > target_width:
        h_start = (H - target_height) // 2
        w_start = (W - target_width) // 2
        initial_latents = initial_latents[:, :, h_start:h_start+target_height, w_start:w_start+target_width]
        H, W = target_height, target_width
    
    history_latents = initial_latents.to(device, dtype=model_dtype)
    
    print(f"初始history_latents shape: {history_latents.shape}")
    
    # 编码prompt - 支持CFG
    if text_guidance_scale > 1.0:
        # 编码positive prompt
        prompt_emb_pos = pipe.encode_prompt(prompt)
        # 编码negative prompt (空字符串)
        prompt_emb_neg = pipe.encode_prompt("")
        print(f"使用Text CFG，guidance scale: {text_guidance_scale}")
    else:
        prompt_emb_pos = pipe.encode_prompt(prompt)
        prompt_emb_neg = None
        print("不使用Text CFG")
    
    # 预生成完整的camera embedding序列
    camera_embedding_full = generate_camera_embeddings_sliding(
        encoded_data.get('cam_emb', None),
        0,
        max_history_frames,
        0,
        0,
        use_real_poses=use_real_poses
    ).to(device, dtype=model_dtype)
    
    print(f"完整camera序列shape: {camera_embedding_full.shape}")
    
    # 🔧 为Camera CFG创建无条件的camera embedding
    if use_camera_cfg:
        # 创建零camera embedding（无条件）
        camera_embedding_uncond = torch.zeros_like(camera_embedding_full)
        print(f"创建无条件camera embedding用于CFG")
    
    # 滑动窗口生成循环
    total_generated = 0
    all_generated_frames = []
    
    while total_generated < total_frames_to_generate:
        current_generation = min(frames_per_generation, total_frames_to_generate - total_generated)
        print(f"\n🔧 生成步骤 {total_generated // frames_per_generation + 1}")
        print(f"当前历史长度: {history_latents.shape[1]}, 本次生成: {current_generation}")
        
        # FramePack数据准备
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
        
        # 🔧 为CFG准备无条件camera embedding
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
                print(f"  去噪步骤 {i+1}/{len(timesteps)}")
            
            timestep_tensor = timestep.unsqueeze(0).to(device, dtype=model_dtype)
            
            with torch.no_grad():
                # 🔧 CFG推理
                if use_camera_cfg and camera_guidance_scale > 1.0:
                    # 条件预测（有camera）
                    noise_pred_cond = pipe.dit(
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
                    
                    # 无条件预测（无camera）
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
                        **(prompt_emb_neg if prompt_emb_neg else prompt_emb_pos),
                        **extra_input
                    )
                    
                    # Camera CFG
                    noise_pred = noise_pred_uncond + camera_guidance_scale * (noise_pred_cond - noise_pred_uncond)
                    
                    # 如果同时使用Text CFG
                    if text_guidance_scale > 1.0 and prompt_emb_neg:
                        # 还需要计算text无条件预测
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
                        
                        # 应用Text CFG到已经应用Camera CFG的结果
                        noise_pred = noise_pred_text_uncond + text_guidance_scale * (noise_pred - noise_pred_text_uncond)
                
                elif text_guidance_scale > 1.0 and prompt_emb_neg:
                    # 只使用Text CFG
                    noise_pred_cond = pipe.dit(
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
                    
                    noise_pred_uncond = pipe.dit(
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
                    
                    noise_pred = noise_pred_uncond + text_guidance_scale * (noise_pred_cond - noise_pred_uncond)
                
                else:
                    # 标准推理（无CFG）
                    noise_pred = pipe.dit(
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
    
    # 7. 解码和保存
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

    print(f"🔧 FramePack滑动窗口生成完成! 保存到: {output_path}")
    print(f"总共生成了 {total_generated} 帧 (压缩后), 对应原始 {total_generated * 4} 帧")
    
def main():
    parser = argparse.ArgumentParser(description="Sekai FramePack滑动窗口视频生成 - 支持CFG")
    parser.add_argument("--condition_pth", type=str,
                       default="/share_zhuyixuan05/zhuyixuan05/sekai-game-walking/00100100001_0004650_0004950/encoded_video.pth")
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--initial_condition_frames", type=int, default=16)
    parser.add_argument("--frames_per_generation", type=int, default=8)
    parser.add_argument("--total_frames_to_generate", type=int, default=40)
    parser.add_argument("--max_history_frames", type=int, default=100)
    parser.add_argument("--use_real_poses", action="store_true", default=False)
    parser.add_argument("--dit_path", type=str, 
                       default="/share_zhuyixuan05/zhuyixuan05/ICLR2026/sekai/sekai_walking_framepack/step1000_framepack.ckpt")
    parser.add_argument("--output_path", type=str, 
                       default='/home/zhuyixuan05/ReCamMaster/sekai/infer_framepack_results/output_sekai_framepack_sliding.mp4')
    parser.add_argument("--prompt", type=str, 
                       default="A drone flying scene in a game world")
    parser.add_argument("--device", type=str, default="cuda")
    
    # 🔧 新增CFG参数
    parser.add_argument("--use_camera_cfg", default=True,
                       help="使用Camera CFG")
    parser.add_argument("--camera_guidance_scale", type=float, default=2.0,
                       help="Camera guidance scale for CFG")
    parser.add_argument("--text_guidance_scale", type=float, default=1.0,
                       help="Text guidance scale for CFG")
    
    args = parser.parse_args()

    print(f"🔧 FramePack CFG生成设置:")
    print(f"Camera CFG: {args.use_camera_cfg}")
    if args.use_camera_cfg:
        print(f"Camera guidance scale: {args.camera_guidance_scale}")
    print(f"Text guidance scale: {args.text_guidance_scale}")
    
    inference_sekai_framepack_sliding_window(
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
        # 🔧 CFG参数
        use_camera_cfg=args.use_camera_cfg,
        camera_guidance_scale=args.camera_guidance_scale,
        text_guidance_scale=args.text_guidance_scale
    )

if __name__ == "__main__":
    main()