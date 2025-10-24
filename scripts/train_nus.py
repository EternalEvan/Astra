import copy
import os
import re
import torch, os, imageio, argparse
from torchvision.transforms import v2
from einops import rearrange
import lightning as pl
import pandas as pd
from diffsynth import WanVideoReCamMasterPipeline, ModelManager, load_state_dict
import torchvision
from PIL import Image
import numpy as np
import random
import json
import torch.nn as nn
import torch.nn.functional as F
import shutil
import wandb
import pdb
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from pose_classifier import PoseClassifier

class NuScenesVideoDataset(torch.utils.data.Dataset):
    def __init__(self, base_path, height=480, width=832, condition_frames=20, target_frames=10, default_text_prompt="A car driving scene captured by front camera", is_i2v=False):
        self.base_path = base_path
        self.samples_path = os.path.join(base_path, "samples")
        
        # Get all sample directories
        self.sample_dirs = []
        for item in os.listdir(self.samples_path):
            sample_path = os.path.join(self.samples_path, item)
            if os.path.isdir(sample_path):
                # Check if required files exist
                condition_path = os.path.join(sample_path, "condition.mp4")
                target_path = os.path.join(sample_path, "target.mp4")
                poses_path = os.path.join(sample_path, "poses.json")
                
                if all(os.path.exists(p) for p in [condition_path, target_path, poses_path]):
                    self.sample_dirs.append(sample_path)
        
        print(f"Found {len(self.sample_dirs)} valid samples in NuScenes dataset.")
        
        self.height = height
        self.width = width
        self.condition_frames = condition_frames
        self.target_frames = target_frames
        self.default_text_prompt = default_text_prompt
        self.is_i2v = is_i2v
        
        self.frame_process = v2.Compose([
            v2.CenterCrop(size=(height, width)),
            v2.Resize(size=(height, width), antialias=True),
            v2.ToTensor(),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    
    def crop_and_resize(self, image):
        width, height = image.size
        scale = max(self.width / width, self.height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        return image

    def load_video_frames(self, video_path):
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

    def __getitem__(self, data_id):
        sample_dir = self.sample_dirs[data_id]
        
        # Load condition video (first 20 frames)
        condition_path = os.path.join(sample_dir, "condition.mp4")
        condition_video = self.load_video_frames(condition_path)
        
        # Load target video (next 10 frames)
        target_path = os.path.join(sample_dir, "target.mp4")
        target_video = self.load_video_frames(target_path)
        
        # Use default text prompt
        text_prompt = self.default_text_prompt
        
        # Concatenate condition and target videos
        if condition_video is not None and target_video is not None:
            full_video = torch.cat([condition_video, target_video], dim=1)  # Concatenate along time dimension
        else:
            return self.__getitem__((data_id + 1) % len(self.sample_dirs))
        
        data = {
            "text": text_prompt,
            "video": full_video,
            "condition_video": condition_video,
            "target_video": target_video,
            "path": sample_dir
        }
        
        if self.is_i2v:
            # Use first frame of condition video as reference image
            first_frame = condition_video[:, 0, :, :]  # C H W
            first_frame_pil = v2.ToPILImage()(first_frame * 0.5 + 0.5)  # Denormalize
            data["first_frame"] = np.array(first_frame_pil)
        
        return data

    def __len__(self):
        return len(self.sample_dirs)

class NuScenesTensorDataset(torch.utils.data.Dataset):
    def __init__(self, base_path, steps_per_epoch, condition_frames=20, target_frames=10):
        self.base_path = base_path
        self.samples_path = os.path.join(base_path, "samples")
        self.condition_frames = condition_frames
        self.target_frames = target_frames
        self.pose_classifier = PoseClassifier()
        
        # Find all samples with encoded data
        self.encoded_paths = []
        if os.path.exists(self.samples_path):
            for item in os.listdir(self.samples_path):
                if item.endswith(".recam.pth"):
                    encoded_path = os.path.join(self.samples_path, item)
                    self.encoded_paths.append(encoded_path)
        
        print(f"Found {len(self.encoded_paths)} encoded samples in NuScenes dataset.")
        assert len(self.encoded_paths) > 0, "No encoded data found!"
        
        self.steps_per_epoch = steps_per_epoch
        self.skip = 0

    def calculate_relative_rotation(self, current_rotation, reference_rotation):
        """
        计算相对于参考帧的相对旋转。
        Args:
            current_rotation: 当前帧的四元数 (q_current) [4]
            reference_rotation: 参考帧的四元数 (q_ref) [4]
        Returns:
            relative_rotation: 相对旋转的四元数 [4]
        """
        # 将四元数转换为 PyTorch 张量
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
    
    def process_poses(self, poses_path):
        """Process poses to create camera embeddings"""
        with open(poses_path, 'r') as f:
            poses_data = json.load(f)
        
        target_relative_poses = poses_data['target_relative_poses']
        
        # Generate pose embeddings for target frames
        pose_embeddings = []
        
        if len(target_relative_poses) == 0:
            # If no target poses, use zero vectors
            for i in range(self.target_frames):
                pose_vec = torch.zeros(7, dtype=torch.float32)  # 3 translation + 4 rotation
                pose_embeddings.append(pose_vec)
        else:
            # Create pose vectors for target frames
            for i in range(self.target_frames):
                if len(target_relative_poses) == 1:
                    # Use the single pose for all frames
                    pose_data = target_relative_poses[0]
                else:
                    # Simple selection - use closest pose or interpolate indices
                    pose_idx = min(i * len(target_relative_poses) // self.target_frames, 
                                 len(target_relative_poses) - 1)
                    pose_data = target_relative_poses[pose_idx]
                
                # Extract translation (3D) and rotation (4D quaternion)
                translation = torch.tensor(pose_data['relative_translation'], dtype=torch.float32)
                rotation = self.calculate_relative_rotation(
                            current_rotation=pose_data['current_rotation'],
                            reference_rotation=pose_data['reference_rotation']
                        )
                
                # Concatenate to form 7D pose vector
                pose_vec = torch.cat([translation, rotation], dim=0)  # [7]
                pose_embeddings.append(pose_vec)
        
        # Stack pose embeddings
        pose_embedding = torch.stack(pose_embeddings, dim=0)  # [target_frames, 7]
        
        pose_analysis = self.pose_classifier.analyze_pose_sequence(pose_embedding)
        pose_classes = pose_analysis['classifications']
        
        # 创建类别embedding
        class_embeddings = self.pose_classifier.create_class_embedding(
            pose_classes, embed_dim=512
        )
        
        return {
            'raw_poses': pose_embedding,
            'pose_classes': pose_classes,
            'class_embeddings': class_embeddings,
            'pose_analysis': pose_analysis 
        }
    
    def __getitem__(self, index):
        while True:
            try:
                data_id = torch.randint(0, len(self.encoded_paths), (1,))[0]
                data_id = (data_id + index) % len(self.encoded_paths)
                
                encoded_path = self.encoded_paths[data_id]
                data = torch.load(encoded_path, weights_only=True, map_location="cpu")
                
                # Get poses path
                sample_name = os.path.basename(encoded_path).replace(".recam.pth", "")
                poses_path = os.path.join(self.samples_path, sample_name, "poses.json")
                
                if not os.path.exists(poses_path):
                    raise FileNotFoundError(f"poses.json not found for sample {sample_name}")
                    
                pose_data = self.process_poses(poses_path)
                
                # pose_analysis = pose_data['pose_analysis']
                # class_distribution = pose_analysis['class_distribution']
                # if class_distribution["backward"] > 0 or class_distribution["forward"] > 0:
                #     index = (index + 1) % len(self.encoded_paths)
                #     self.skip += 1
                #     print(f"skip {self.skip}")
                #     continue

                result = {
                    "latents": data["latents"],
                    "prompt_emb": data["prompt_emb"],
                    "image_emb": data.get("image_emb", {}),
                    "camera": pose_data['class_embeddings'].to(torch.bfloat16),  # 使用类别embedding
                    "pose_classes": pose_data['pose_classes'],  # 保留类别标签用于分析
                    "raw_poses": pose_data['raw_poses'],  # 保留原始pose用于对比
                    "pose_analysis": pose_data['pose_analysis']  # 保留分析信息
                }
                
                break
                
            except Exception as e:
                print(f"ERROR WHEN LOADING: {e}")
                index = random.randrange(len(self.encoded_paths))
                
        return result
    
    def __len__(self):
        return self.steps_per_epoch


class LightningModelForDataProcess(pl.LightningModule):
    def __init__(self, text_encoder_path, vae_path, image_encoder_path=None, tiled=False, tile_size=(34, 34), tile_stride=(18, 16)):
        super().__init__()
        model_path = [text_encoder_path, vae_path]
        if image_encoder_path is not None:
            model_path.append(image_encoder_path)
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        model_manager.load_models(model_path)
        self.pipe = WanVideoReCamMasterPipeline.from_model_manager(model_manager)

        self.tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}
        
    def test_step(self, batch, batch_idx):
        text, video, path = batch["text"][0], batch["video"], batch["path"][0]
        
        self.pipe.device = self.device
        if video is not None:
            pth_path = path + ".recam.pth"
            if not os.path.exists(pth_path):
                # prompt
                prompt_emb = self.pipe.encode_prompt(text)
                # video
                video = video.to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
                latents = self.pipe.encode_video(video, **self.tiler_kwargs)[0]
                # image
                if "first_frame" in batch:
                    first_frame = Image.fromarray(batch["first_frame"][0].cpu().numpy())
                    _, _, num_frames, height, width = video.shape
                    image_emb = self.pipe.encode_image(first_frame, num_frames, height, width)
                else:
                    image_emb = {}
                data = {"latents": latents, "prompt_emb": prompt_emb, "image_emb": image_emb}
                torch.save(data, pth_path)
                print(f"Output: {pth_path}")
            else:
                print(f"File {pth_path} already exists, skipping.")

class LightningModelForTrain(pl.LightningModule):
    def __init__(
        self,
        dit_path,
        learning_rate=1e-5,
        use_gradient_checkpointing=True, use_gradient_checkpointing_offload=False,
        resume_ckpt_path=None
    ):
        super().__init__()
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        if os.path.isfile(dit_path):
            model_manager.load_models([dit_path])
        else:
            dit_path = dit_path.split(",")
            model_manager.load_models([dit_path])
        model_manager.load_models(["models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"])
        self.pipe = WanVideoReCamMasterPipeline.from_model_manager(model_manager)
        self.pipe.scheduler.set_timesteps(1000, training=True)

        dim=self.pipe.dit.blocks[0].self_attn.q.weight.shape[0]
        for block in self.pipe.dit.blocks:
            block.cam_encoder = nn.Linear(512, dim)  # Changed from 12 to 7
            block.projector = nn.Linear(dim, dim)
            block.cam_encoder.weight.data.zero_()
            block.cam_encoder.bias.data.zero_()
            block.projector.weight = nn.Parameter(torch.eye(dim))
            block.projector.bias = nn.Parameter(torch.zeros(dim))
        
        if resume_ckpt_path is not None:
            state_dict = torch.load(resume_ckpt_path, map_location="cpu")
            self.pipe.dit.load_state_dict(state_dict, strict=True)

        self.freeze_parameters()
        for name, module in self.pipe.denoising_model().named_modules():
            if any(keyword in name for keyword in ["cam_encoder", "projector", "self_attn"]):
                print(f"Trainable: {name}")
                for param in module.parameters():
                    param.requires_grad = True

        trainable_params = 0
        seen_params = set()
        for name, module in self.pipe.denoising_model().named_modules():
            for param in module.parameters():
                if param.requires_grad and param not in seen_params:
                    trainable_params += param.numel()
                    seen_params.add(param)
        print(f"Total number of trainable parameters: {trainable_params}")
        
        self.learning_rate = learning_rate
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload

        # 创建可视化目录
        self.vis_dir = "nus/visualizations"
        os.makedirs(self.vis_dir, exist_ok=True)
        
        
    def freeze_parameters(self):
        # Freeze parameters
        self.pipe.requires_grad_(False)
        self.pipe.eval()
        self.pipe.denoising_model().train()

    def visualize_training_results(self, latents, noisy_latents, noise_pred, training_target, step):
        """可视化训练结果"""
        try:
            with torch.no_grad():
                # 分离target和condition部分
                tgt_latent_len = 5
                
                # 提取各部分latents
                target_latents = latents[:, :, tgt_latent_len:, :, :]  # 原始target
                condition_latents = latents[:, :, :tgt_latent_len, :, :]  # condition
                noisy_target_latents = noisy_latents[:, :, tgt_latent_len:, :, :]  # 加噪target
                
                # 解码为视频帧 (取第一个batch)
                # 只可视化前几帧以节省内存
                vis_frames = 10
                
                # 解码condition frames
                condition_sample = condition_latents[:, :, :vis_frames, :, :]
                condition_video = self.pipe.decode_video(condition_sample, tiled=True, tile_size=(34, 34), tile_stride=(18, 16))
                condition_video = condition_video[0].to(torch.float32)  # [C, T, H, W]
                
                # 解码原始target frames
                target_sample = target_latents[:, :, :vis_frames, :, :]
                target_video = self.pipe.decode_video(target_sample, tiled=True, tile_size=(34, 34), tile_stride=(18, 16))
                target_video = target_video[0].to(torch.float32)  # [C, T, H, W]
                
                # 解码加噪target frames
                noisy_target_sample = noisy_target_latents[:, :, :vis_frames, :, :]
                noisy_target_video = self.pipe.decode_video(noisy_target_sample, tiled=True, tile_size=(34, 34), tile_stride=(18, 16))
                noisy_target_video = noisy_target_video[0].to(torch.float32)  # [C, T, H, W]
                
                # 解码预测结果 (从noise_pred重构)
                pred_latents = noisy_target_latents - noise_pred[:, :, 5:, :, :]
                pred_video = self.pipe.decode_video(pred_latents, tiled=True, tile_size=(34, 34), tile_stride=(18, 16))
                pred_video = pred_video[0].to(torch.float32)  # [C, T, H, W]
                
                # 归一化到[0,1]
                condition_video = (condition_video * 0.5 + 0.5).clamp(0, 1)
                target_video = (target_video * 0.5 + 0.5).clamp(0, 1)
                noisy_target_video = (noisy_target_video * 0.5 + 0.5).clamp(0, 1)
                pred_video = (pred_video * 0.5 + 0.5).clamp(0, 1)
                
                # 创建可视化图像
                fig, axes = plt.subplots(4, vis_frames, figsize=(vis_frames * 3, 12))
                if vis_frames == 1:
                    axes = axes.reshape(-1, 1)
                
                for frame_idx in range(vis_frames-1):
                    # Condition frame
                    condition_frame = condition_video[:, frame_idx, :, :].permute(1, 2, 0).cpu().numpy()
                    axes[0, frame_idx].imshow(condition_frame)
                    axes[0, frame_idx].set_title(f'Condition Frame {frame_idx}')
                    axes[0, frame_idx].axis('off')
                    
                    # Original target frame
                    target_frame = target_video[:, frame_idx, :, :].permute(1, 2, 0).cpu().numpy()
                    axes[1, frame_idx].imshow(target_frame)
                    axes[1, frame_idx].set_title(f'Original Target {frame_idx}')
                    axes[1, frame_idx].axis('off')
                    
                    # Noisy target frame
                    noisy_frame = noisy_target_video[:, frame_idx, :, :].permute(1, 2, 0).cpu().numpy()
                    axes[2, frame_idx].imshow(noisy_frame)
                    axes[2, frame_idx].set_title(f'Noisy Target {frame_idx}')
                    axes[2, frame_idx].axis('off')
                    
                    # Predicted frame
                    pred_frame = pred_video[:, frame_idx, :, :].permute(1, 2, 0).cpu().numpy()
                    axes[3, frame_idx].imshow(pred_frame)
                    axes[3, frame_idx].set_title(f'Prediction {frame_idx}')
                    axes[3, frame_idx].axis('off')
                
                plt.tight_layout()
                
                # 保存图像
                save_path = os.path.join(self.vis_dir, f"training_step_{step:06d}.png")
                plt.savefig(save_path, dpi=100, bbox_inches='tight')
                plt.close()
                
                # 记录到wandb
                if wandb.run is not None:
                    wandb.log({
                        "training_visualization": wandb.Image(save_path),
                        "step": step
                    })
                
                print(f"Visualization saved to {save_path}")
                
        except Exception as e:
            print(f"Error during visualization: {e}")

    def training_step(self, batch, batch_idx):
        # Data
        latents = batch["latents"].to(self.device)
        #  裁剪空间尺寸 (例如裁剪到固定的 height 和 width)
        target_height, target_width = 50, 70  # 根据你的需求调整
        current_height, current_width = latents.shape[3], latents.shape[4]
        
        if current_height > target_height or current_width > target_width:
            # 中心裁剪
            h_start = (current_height - target_height) // 2
            w_start = (current_width - target_width) // 2
            latents = latents[:, :, :, 
                            h_start:h_start+target_height, 
                            w_start:w_start+target_width]
        
        prompt_emb = batch["prompt_emb"]
        prompt_emb["context"] = prompt_emb["context"][0].to(self.device)
        image_emb = batch["image_emb"]
        
        if "clip_feature" in image_emb:
            image_emb["clip_feature"] = image_emb["clip_feature"][0].to(self.device)
        if "y" in image_emb:
            image_emb["y"] = image_emb["y"][0].to(self.device)
        
        cam_emb = batch["camera"].to(self.device)

        # Loss
        self.pipe.device = self.device
        noise = torch.randn_like(latents)
        timestep_id = torch.randint(0, self.pipe.scheduler.num_train_timesteps, (1,))
        timestep = self.pipe.scheduler.timesteps[timestep_id].to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
        
        extra_input = self.pipe.prepare_extra_input(latents)
        origin_latents = copy.deepcopy(latents)
        noisy_latents = self.pipe.scheduler.add_noise(latents, noise, timestep)
        tgt_latent_len = 10
        noisy_latents[:, :, :tgt_latent_len, ...] = origin_latents[:, :, :tgt_latent_len, ...]
        training_target = self.pipe.scheduler.training_target(latents, noise, timestep)
        
        # Compute loss
        noise_pred = self.pipe.denoising_model()(
            noisy_latents, timestep=timestep, cam_emb=cam_emb, **prompt_emb, **extra_input, **image_emb,
            use_gradient_checkpointing=self.use_gradient_checkpointing,
            use_gradient_checkpointing_offload=self.use_gradient_checkpointing_offload
        )
        loss = torch.nn.functional.mse_loss(noise_pred[:, :, tgt_latent_len:, ...].float(), training_target[:, :, tgt_latent_len:, ...].float())
        loss = loss * self.pipe.scheduler.training_weight(timestep)

        # 可视化 (每10步一次)
        if self.global_step % 1000 == 500:
            self.visualize_training_results(
                latents=origin_latents,
                noisy_latents=noisy_latents,
                noise_pred=noise_pred,
                training_target=training_target,
                step=self.global_step
            )

        # Record log
        wandb.log({
            "train_loss": loss.item(),
            "timestep": timestep.item(),
            "global_step": self.global_step
        })

        return loss


    def configure_optimizers(self):
        trainable_modules = filter(lambda p: p.requires_grad, self.pipe.denoising_model().parameters())
        optimizer = torch.optim.AdamW(trainable_modules, lr=self.learning_rate)
        return optimizer
    

    def on_save_checkpoint(self, checkpoint):
        checkpoint_dir = "/home/zhuyixuan05/ReCamMaster/nus"
        print(f"Checkpoint directory: {checkpoint_dir}")
        current_step = self.global_step
        print(f"Current step: {current_step}")

        checkpoint.clear()
        trainable_param_names = list(filter(lambda named_param: named_param[1].requires_grad, self.pipe.denoising_model().named_parameters()))
        trainable_param_names = set([named_param[0] for named_param in trainable_param_names])
        state_dict = self.pipe.denoising_model().state_dict()
        torch.save(state_dict, os.path.join(checkpoint_dir, f"step{current_step}_8040.ckpt"))

def data_process(args):
    if args.dataset_type == "nuscenes":
        dataset = NuScenesVideoDataset(
            args.dataset_path,
            height=args.height,
            width=args.width,
            condition_frames=args.condition_frames,
            target_frames=args.target_frames,
            default_text_prompt=args.default_text_prompt,
            is_i2v=args.image_encoder_path is not None
        )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        batch_size=1,
        num_workers=args.dataloader_num_workers
    )
    model = LightningModelForDataProcess(
        text_encoder_path=args.text_encoder_path,
        image_encoder_path=args.image_encoder_path,
        vae_path=args.vae_path,
        tiled=args.tiled,
        tile_size=(args.tile_size_height, args.tile_size_width),
        tile_stride=(args.tile_stride_height, args.tile_stride_width),
    )
    trainer = pl.Trainer(
        accelerator="gpu",
        devices="auto",
        default_root_dir=args.output_path,
    )
    trainer.test(model, dataloader)
    
    
def train(args):
    if args.dataset_type == "nuscenes":
        dataset = NuScenesTensorDataset(
            args.dataset_path,
            steps_per_epoch=args.steps_per_epoch,
            condition_frames=args.condition_frames,
            target_frames=args.target_frames,
        )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        batch_size=1,
        num_workers=args.dataloader_num_workers
    )
    model = LightningModelForTrain(
        dit_path=args.dit_path,
        learning_rate=args.learning_rate,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        resume_ckpt_path=args.resume_ckpt_path,
    )

    if args.use_swanlab:
        project_name = "nuscenes-recam" if args.dataset_type == "nuscenes" else "recam"
        wandb.init(
            project=project_name,
            name=f"{args.dataset_type}-video-generation",
        )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices="auto",
        precision="bf16",
        strategy=args.training_strategy,
        choices=["original", "nuscenes"],
        help="Type of dataset. 'original' for the original format, 'nuscenes' for NuScenes format.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/share_zhuyixuan05/zhuyixuan05/nuscenes_video_generation_3",
        help="The path of the Dataset.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="/share_zhuyixuan05/zhuyixuan05/nus_checkpoint",
        help="Path to save the model.",
    )
    parser.add_argument(
        "--text_encoder_path",
        type=str,
        default="models/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
        help="Path of text encoder.",
    )
    parser.add_argument(
        "--image_encoder_path",
        type=str,
        default=None,
        help="Path of image encoder.",
    )
    parser.add_argument(
        "--vae_path",
        type=str,
        default="models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
        help="Path of VAE.",
    )
    parser.add_argument(
        "--dit_path",
        type=str,
        default="models/Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors",
        help="Path of DiT.",
    )
    parser.add_argument(
        "--tiled",
        default=True,
        action="store_true",
        help="Whether enable tile encode in VAE. This option can reduce VRAM required.",
    )
    parser.add_argument(
        "--tile_size_height",
        type=int,
        default=34,
        help="Tile size (height) in VAE.",
    )
    parser.add_argument(
        "--tile_size_width",
        type=int,
        default=34,
        help="Tile size (width) in VAE.",
    )
    parser.add_argument(
        "--tile_stride_height",
        type=int,
        default=18,
        help="Tile stride (height) in VAE.",
    )
    parser.add_argument(
        "--tile_stride_width",
        type=int,
        default=16,
        help="Tile stride (width) in VAE.",
    )
    parser.add_argument(
        "--steps_per_epoch",
        type=int,
        default=1000,
        help="Number of steps per epoch.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=81,
        help="Number of frames.",
    )
    parser.add_argument(
        "--condition_frames",
        type=int,
        default=20,
        help="Number of condition frames for NuScenes dataset.",
    )
    parser.add_argument(
        "--target_frames",
        type=int,
        default=10,
        help="Number of target frames for NuScenes dataset.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=900,
        help="Image height.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1600,
        help="Image width.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=4,
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate.",
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help="The number of batches in gradient accumulation.",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=2,
        help="Number of epochs.",
    )
    parser.add_argument(
        "--training_strategy",
        type=str,
        default="deepspeed_stage_1",
        choices=["auto", "deepspeed_stage_1", "deepspeed_stage_2", "deepspeed_stage_3"],
        help="Training strategy",
    )
    parser.add_argument(
        "--use_gradient_checkpointing",
        default=False,
        action="store_true",
        help="Whether to use gradient checkpointing.",
    )
    parser.add_argument(
        "--use_gradient_checkpointing_offload",
        default=False,
        action="store_true",
        help="Whether to use gradient checkpointing offload.",
    )
    parser.add_argument(
        "--use_swanlab",
        default=True,
        action="store_true",
        help="Whether to use SwanLab logger.",
    )
    parser.add_argument(
        "--swanlab_mode",
        default="cloud",
        help="SwanLab mode (cloud or local).",
    )
    parser.add_argument(
        "--metadata_file_name",
        type=str,
        default="metadata.csv",
    )
    parser.add_argument(
        "--resume_ckpt_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--default_text_prompt",
        type=str,
        default="A car driving scene captured by front camera",
        help="Default text prompt for NuScenes samples without description.",
    )
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    os.makedirs(os.path.join(args.output_path, "checkpoints"), exist_ok=True)
    if args.task == "data_process":
        data_process(args)
    elif args.task == "train":
        train(args)