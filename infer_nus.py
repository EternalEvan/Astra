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
from pose_classifier import PoseClassifier


def load_video_frames(video_path, num_frames=20, height=900, width=1600):
    """Load video frames and preprocess them"""
    frame_process = v2.Compose([
        # v2.CenterCrop(size=(height, width)),
        # v2.Resize(size=(height, width), antialias=True),
        v2.ToTensor(),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    
    def crop_and_resize(image):
        w, h = image.size
        # scale = max(width / w, height / h)
        image = v2.functional.resize(
            image,
            (round(480), round(832)),
            interpolation=v2.InterpolationMode.BILINEAR
        )
        return image
    
    reader = imageio.get_reader(video_path)
    frames = []
    
    for i, frame_data in enumerate(reader):
        if i >= num_frames:
            break
        frame = Image.fromarray(frame_data)
        frame = crop_and_resize(frame)
        frame = frame_process(frame)
        frames.append(frame)
    
    reader.close()
    
    if len(frames) == 0:
        return None
        
    frames = torch.stack(frames, dim=0)
    frames = rearrange(frames, "T C H W -> C T H W")
    return frames

def calculate_relative_rotation(current_rotation, reference_rotation):
    """计算相对旋转四元数"""
    q_current = torch.tensor(current_rotation, dtype=torch.float32)
    q_ref = torch.tensor(reference_rotation, dtype=torch.float32)

    # 计算参考旋转的逆 (q_ref^-1)
    q_ref_inv = torch.tensor([q_ref[0], -q_ref[1], -q_ref[2], -q_ref[3]])

    # 四元数乘法计算相对旋转: q_relative = q_ref^-1 * q_current
    w1, x1, y1, z1 = q_ref_inv
    w2, x2, y2, z2 = q_current

    relative_rotation = torch.tensor([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    ])

    return relative_rotation

def generate_direction_poses(direction="left", target_frames=10, condition_frames=20):
    """
    根据指定方向生成pose类别embedding，包含condition和target帧
    Args:
        direction: 'forward', 'backward', 'left_turn', 'right_turn'
        target_frames: 目标帧数
        condition_frames: 条件帧数
    """
    classifier = PoseClassifier()
    
    total_frames = condition_frames + target_frames
    print(f"conditon{condition_frames}")
    print(f"target{target_frames}")
    poses = []
    
    # 🔧 生成condition帧的pose（相对稳定的前向运动）
    for i in range(condition_frames):
        t = i / max(1, condition_frames - 1)  # 0 to 1
        
        # condition帧保持相对稳定的前向运动
        translation = [-t * 0.5, 0.0, 0.0]  # 缓慢前进
        rotation = [1.0, 0.0, 0.0, 0.0]     # 无旋转
        frame_type = 0.0  # condition
        
        pose_vec = translation + rotation + [frame_type]  # 8D vector
        poses.append(pose_vec)
    
    # 🔧 生成target帧的pose（根据指定方向）
    for i in range(target_frames):
        t = i / max(1, target_frames - 1)  # 0 to 1
        
        if direction == "forward":
            # 前进：x负方向移动，无旋转
            translation = [-(condition_frames * 0.5 + t * 2.0), 0.0, 0.0]
            rotation = [1.0, 0.0, 0.0, 0.0]  # 单位四元数
            
        elif direction == "backward":
            # 后退：x正方向移动，无旋转
            translation = [-(condition_frames * 0.5) + t * 2.0, 0.0, 0.0]
            rotation = [1.0, 0.0, 0.0, 0.0]
            
        elif direction == "left_turn":
            # 左转：前进 + 绕z轴正向旋转
            translation = [-(condition_frames * 0.5 + t * 1.5), t * 0.5, 0.0]  # 前进并稍微左移
            yaw = t * 0.3  # 左转角度（弧度）
            rotation = [
                np.cos(yaw/2),  # w
                0.0,            # x
                0.0,            # y  
                np.sin(yaw/2)   # z (左转为正)
            ]
            
        elif direction == "right_turn":
            # 右转：前进 + 绕z轴负向旋转
            translation = [-(condition_frames * 0.5 + t * 1.5), -t * 0.5, 0.0]  # 前进并稍微右移
            yaw = -t * 0.3  # 右转角度（弧度）
            rotation = [
                np.cos(abs(yaw)/2),  # w
                0.0,                 # x
                0.0,                 # y
                np.sin(yaw/2)        # z (右转为负)
            ]
        else:
            raise ValueError(f"Unknown direction: {direction}")
        
        frame_type = 1.0  # target
        pose_vec = translation + rotation + [frame_type]  # 8D vector
        poses.append(pose_vec)
    
    pose_sequence = torch.tensor(poses, dtype=torch.float32)
    
    # 🔧 只对target部分进行分类（前7维，去掉frame type）
    target_pose_sequence = pose_sequence[condition_frames:, :7]
    
    # 🔧 使用增强的embedding生成方法
    condition_classes = torch.full((condition_frames,), 0, dtype=torch.long)  # condition都是forward
    target_classes = classifier.classify_pose_sequence(target_pose_sequence)
    full_classes = torch.cat([condition_classes, target_classes], dim=0)
    
    # 创建增强的embedding
    class_embeddings = create_enhanced_class_embedding_for_inference(
        full_classes, pose_sequence, embed_dim=512
    )
    
    print(f"Generated {direction} poses:")
    print(f"  Total frames: {total_frames} (condition: {condition_frames}, target: {target_frames})")
    analysis = classifier.analyze_pose_sequence(target_pose_sequence)
    print(f"  Target class distribution: {analysis['class_distribution']}")
    print(f"  Target motion segments: {len(analysis['motion_segments'])}")
    
    return class_embeddings

def create_enhanced_class_embedding_for_inference(class_labels: torch.Tensor, pose_sequence: torch.Tensor, embed_dim: int = 512) -> torch.Tensor:
    """推理时创建增强的类别embedding"""
    num_classes = 4
    num_frames = len(class_labels)
    
    # 基础的方向embedding
    direction_vectors = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],  # forward
        [-1.0, 0.0, 0.0, 0.0], # backward
        [0.0, 1.0, 0.0, 0.0],  # left_turn
        [0.0, -1.0, 0.0, 0.0], # right_turn
    ], dtype=torch.float32)
    
    # One-hot编码
    one_hot = torch.zeros(num_frames, num_classes)
    one_hot.scatter_(1, class_labels.unsqueeze(1), 1)
    
    # 基于方向向量的基础embedding
    base_embeddings = one_hot @ direction_vectors  # [num_frames, 4]
    
    # 添加frame type信息
    frame_types = pose_sequence[:, -1]  # 最后一维是frame type
    frame_type_embeddings = torch.zeros(num_frames, 2)
    frame_type_embeddings[:, 0] = (frame_types == 0).float()  # condition
    frame_type_embeddings[:, 1] = (frame_types == 1).float()  # target
    
    # 添加pose的几何信息
    translations = pose_sequence[:, :3]  # [num_frames, 3]
    rotations = pose_sequence[:, 3:7]    # [num_frames, 4]
    
    # 组合所有特征
    combined_features = torch.cat([
        base_embeddings,         # [num_frames, 4]
        frame_type_embeddings,   # [num_frames, 2]
        translations,            # [num_frames, 3]
        rotations,               # [num_frames, 4]
    ], dim=1)  # [num_frames, 13]
    
    # 扩展到目标维度
    if embed_dim > 13:
        expand_matrix = torch.randn(13, embed_dim) * 0.1
        expand_matrix[:13, :13] = torch.eye(13)
        embeddings = combined_features @ expand_matrix
    else:
        embeddings = combined_features[:, :embed_dim]
    
    return embeddings

def generate_poses_from_file(poses_path, target_frames=10):
    """从poses.json文件生成类别embedding"""
    classifier = PoseClassifier()
    
    with open(poses_path, 'r') as f:
        poses_data = json.load(f)
    
    target_relative_poses = poses_data['target_relative_poses']
    
    if not target_relative_poses:
        print("No poses found in file, using forward direction")
        return generate_direction_poses("forward", target_frames)
    
    # 创建pose序列
    pose_vecs = []
    for i in range(target_frames):
        if len(target_relative_poses) == 1:
            pose_data = target_relative_poses[0]
        else:
            pose_idx = min(i * len(target_relative_poses) // target_frames, 
                         len(target_relative_poses) - 1)
            pose_data = target_relative_poses[pose_idx]
        
        # 提取相对位移和旋转
        translation = torch.tensor(pose_data['relative_translation'], dtype=torch.float32)
        current_rotation = torch.tensor(pose_data['current_rotation'], dtype=torch.float32)
        reference_rotation = torch.tensor(pose_data['reference_rotation'], dtype=torch.float32)
        
        # 计算相对旋转
        relative_rotation = calculate_relative_rotation(current_rotation, reference_rotation)
        
        # 组合为7D向量
        pose_vec = torch.cat([translation, relative_rotation], dim=0)
        pose_vecs.append(pose_vec)
    
    pose_sequence = torch.stack(pose_vecs, dim=0)
    
    # 使用分类器生成class embedding
    class_embeddings = classifier.create_class_embedding(
        classifier.classify_pose_sequence(pose_sequence), 
        embed_dim=512
    )
    
    print(f"Generated poses from file:")
    analysis = classifier.analyze_pose_sequence(pose_sequence)
    print(f"  Class distribution: {analysis['class_distribution']}")
    print(f"  Motion segments: {len(analysis['motion_segments'])}")
    
    return class_embeddings

def inference_nuscenes_video(
    condition_video_path,
    dit_path,
    text_encoder_path,
    vae_path,
    output_path="nus/infer_results/output_nuscenes.mp4",
    condition_frames=20,
    target_frames=3,
    height=900,
    width=1600,
    device="cuda",
    prompt="A car driving scene captured by front camera",
    poses_path=None,
    direction="forward"
):
    """
    使用方向类别控制的推理函数 - 支持condition和target pose区分
    """
    os.makedirs(os.path.dirname(output_path),exist_ok=True)

    print(f"Setting up models for {direction} movement...")
    
    # 1. Load models (same as before)
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
        block.cam_encoder = nn.Linear(512, dim)  # 保持512维embedding
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
    
    print("Loading condition video...")
    
    # Load condition video
    condition_video = load_video_frames(
        condition_video_path, 
        num_frames=condition_frames,
        height=height,
        width=width
    )
    
    if condition_video is None:
        raise ValueError(f"Failed to load condition video from {condition_video_path}")
    
    condition_video = condition_video.unsqueeze(0).to(device, dtype=pipe.torch_dtype)
    
    print("Processing poses...")
    
    # 🔧 修改：生成包含condition和target的pose embedding
    print(f"Generating {direction} movement poses...")
    camera_embedding = generate_direction_poses(
        direction=direction, 
        target_frames=target_frames,
        condition_frames=int(condition_frames/4)  # 压缩后的condition帧数
    )
    
    camera_embedding = camera_embedding.unsqueeze(0).to(device, dtype=torch.bfloat16)
    
    print(f"Camera embedding shape: {camera_embedding.shape}")
    print(f"Generated poses for direction: {direction}")
    
    print("Encoding inputs...")
    
    # Encode text prompt
    prompt_emb = pipe.encode_prompt(prompt)
    
    # Encode condition video
    condition_latents = pipe.encode_video(condition_video, tiled=True, tile_size=(34, 34), tile_stride=(18, 16))[0]
    
    print("Generating video...")
    
    # Generate target latents
    batch_size = 1
    channels = condition_latents.shape[0]
    latent_height = condition_latents.shape[2]
    latent_width = condition_latents.shape[3]
    target_height, target_width = 60, 104  # 根据你的需求调整
    
    if latent_height > target_height or latent_width > target_width:
        # 中心裁剪
        h_start = (latent_height - target_height) // 2
        w_start = (latent_width - target_width) // 2
        condition_latents = condition_latents[:, :, 
                        h_start:h_start+target_height, 
                        w_start:w_start+target_width]
    latent_height =   target_height
    latent_width = target_width
    condition_latents = condition_latents.to(device, dtype=pipe.torch_dtype)
    condition_latents = condition_latents.unsqueeze(0)
    condition_latents = condition_latents + 0.05 * torch.randn_like(condition_latents)  # 添加少量噪声以增加多样性
                
    # Initialize target latents with noise
    target_latents = torch.randn(
        batch_size, channels, target_frames, latent_height, latent_width,
        device=device, dtype=pipe.torch_dtype
    )
    print(target_latents.shape)
    print(camera_embedding.shape)
    # Combine condition and target latents
    combined_latents = torch.cat([condition_latents, target_latents], dim=2)
    print(combined_latents.shape)

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
        target_noise_pred = noise_pred[:, :, int(condition_frames/4):, :, :]
        target_latents = pipe.scheduler.step(target_noise_pred, timestep, target_latents)
        
        # Update combined latents
        combined_latents[:, :, int(condition_frames/4):, :, :] = target_latents
    
    print("Decoding video...")
    
    # Decode final video
    final_video = torch.cat([condition_latents, target_latents], dim=2)
    decoded_video = pipe.decode_video(final_video, tiled=True, tile_size=(34, 34), tile_stride=(18, 16))

    # Save video
    print(f"Saving video to {output_path}")

    # Convert to numpy and save
    video_np = decoded_video[0].to(torch.float32).permute(1, 2, 3, 0).cpu().numpy()  # 转换为 Float32
    video_np = (video_np * 0.5 + 0.5).clip(0, 1)  # Denormalize
    video_np = (video_np * 255).astype(np.uint8)

    with imageio.get_writer(output_path, fps=20) as writer:
        for frame in video_np:
            writer.append_data(frame)

    print(f"Video generation completed! Saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="NuScenes Video Generation Inference with Direction Control")
    parser.add_argument("--condition_video", type=str, default="/home/zhuyixuan05/ReCamMaster/nus/videos/4032/right.mp4",
                       help="Path to condition video")
    parser.add_argument("--direction", type=str, default="left_turn",
                       choices=["forward", "backward", "left_turn", "right_turn"],
                       help="Direction of camera movement")
    parser.add_argument("--dit_path", type=str, default="/home/zhuyixuan05/ReCamMaster/nus_dynamic/step15000_dynamic.ckpt",
                       help="Path to trained DiT checkpoint")
    parser.add_argument("--text_encoder_path", type=str, 
                       default="models/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
                       help="Path to text encoder")
    parser.add_argument("--vae_path", type=str,
                       default="models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth", 
                       help="Path to VAE")
    parser.add_argument("--output_path", type=str, default="nus/infer_results-15000/right_left.mp4",
                       help="Output video path")
    parser.add_argument("--poses_path", type=str, default=None,
                       help="Path to poses.json file (optional, will use direction if not provided)")
    parser.add_argument("--prompt", type=str, 
                       default="A car driving scene captured by front camera",
                       help="Text prompt for generation")
    parser.add_argument("--condition_frames", type=int, default=40,
                       help="Number of condition frames")
    # 这个是原始帧数
    parser.add_argument("--target_frames", type=int, default=8,
                       help="Number of target frames to generate")
    # 这个要除以4
    parser.add_argument("--height", type=int, default=900,
                       help="Video height")
    parser.add_argument("--width", type=int, default=1600,
                       help="Video width")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to run inference on")
    
    args = parser.parse_args()
    
    condition_video_path = args.condition_video
    input_filename = os.path.basename(condition_video_path)
    output_dir = "nus/infer_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # 🔧 修改：在输出文件名中包含方向信息
    if args.output_path is None:
        name_parts = os.path.splitext(input_filename)
        output_filename = f"{name_parts[0]}_{args.direction}{name_parts[1]}"
        output_path = os.path.join(output_dir, output_filename)
    else:
        output_path = args.output_path

    print(f"Output video will be saved to: {output_path}") 
    inference_nuscenes_video(
        condition_video_path=args.condition_video,
        dit_path=args.dit_path,
        text_encoder_path=args.text_encoder_path,
        vae_path=args.vae_path,
        output_path=output_path,
        condition_frames=args.condition_frames,
        target_frames=args.target_frames,
        height=args.height,
        width=args.width,
        device=args.device,
        prompt=args.prompt,
        poses_path=args.poses_path,
        direction=args.direction  # 🔧 新增
    )

if __name__ == "__main__":
    main()