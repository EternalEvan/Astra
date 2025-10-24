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

class TextVideoDataset(torch.utils.data.Dataset):
    def __init__(self, base_path, metadata_path, max_num_frames=81, frame_interval=1, num_frames=81, height=480, width=832, is_i2v=False):
        metadata = pd.read_csv(metadata_path)
        self.path = [os.path.join(base_path, "train", file_name) for file_name in metadata["file_name"]]
        self.text = metadata["text"].to_list()
        
        self.max_num_frames = max_num_frames
        self.frame_interval = frame_interval
        self.num_frames = num_frames
        self.height = height
        self.width = width
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


    def load_frames_using_imageio(self, file_path, max_num_frames, start_frame_id, interval, num_frames, frame_process):
        reader = imageio.get_reader(file_path)
        if reader.count_frames() < max_num_frames or reader.count_frames() - 1 < start_frame_id + (num_frames - 1) * interval:
            reader.close()
            return None
        
        frames = []
        first_frame = None
        for frame_id in range(num_frames):
            frame = reader.get_data(start_frame_id + frame_id * interval)
            frame = Image.fromarray(frame)
            frame = self.crop_and_resize(frame)
            if first_frame is None:
                first_frame = np.array(frame)
            frame = frame_process(frame)
            frames.append(frame)
        reader.close()

        frames = torch.stack(frames, dim=0)
        frames = rearrange(frames, "T C H W -> C T H W")

        if self.is_i2v:
            return frames, first_frame
        else:
            return frames


    def load_video(self, file_path):
        start_frame_id = 0
        frames = self.load_frames_using_imageio(file_path, self.max_num_frames, start_frame_id, self.frame_interval, self.num_frames, self.frame_process)
        return frames
    
    
    def is_image(self, file_path):
        file_ext_name = file_path.split(".")[-1]
        if file_ext_name.lower() in ["jpg", "jpeg", "png", "webp"]:
            return True
        return False
    
    
    def load_image(self, file_path):
        frame = Image.open(file_path).convert("RGB")
        frame = self.crop_and_resize(frame)
        first_frame = frame
        frame = self.frame_process(frame)
        frame = rearrange(frame, "C H W -> C 1 H W")
        return frame


    def __getitem__(self, data_id):
        text = self.text[data_id]
        path = self.path[data_id]
        while True:
            try:
                if self.is_image(path):
                    if self.is_i2v:
                        raise ValueError(f"{path} is not a video. I2V model doesn't support image-to-image training.")
                    video = self.load_image(path)
                else:
                    video = self.load_video(path)
                if self.is_i2v:
                    video, first_frame = video
                    data = {"text": text, "video": video, "path": path, "first_frame": first_frame}
                else:
                    data = {"text": text, "video": video, "path": path}
                break
            except:
                data_id += 1
        return data
    

    def __len__(self):
        return len(self.path)



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

class Camera(object):
    def __init__(self, c2w):
        c2w_mat = np.array(c2w).reshape(4, 4)
        self.c2w_mat = c2w_mat
        self.w2c_mat = np.linalg.inv(c2w_mat)



class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, base_path, metadata_path, steps_per_epoch, condition_frames=32, target_frames=32):
        metadata = pd.read_csv(metadata_path)
        self.path = [os.path.join(base_path, "train", file_name) for file_name in metadata["file_name"]]
        print(len(self.path), "videos in metadata.")
        self.path = [i + ".recam.pth" for i in self.path if os.path.exists(i + ".recam.pth")]
        print(len(self.path), "tensors cached in metadata.")
        assert len(self.path) > 0
        self.steps_per_epoch = steps_per_epoch
        self.condition_frames = int(condition_frames)
        self.target_frames = int(target_frames)

    def parse_matrix(self, matrix_str):
        rows = matrix_str.strip().split('] [')
        matrix = []
        for row in rows:
            row = row.replace('[', '').replace(']', '')
            matrix.append(list(map(float, row.split())))
        return np.array(matrix)

    def get_relative_pose(self, pose_prev, pose_curr):
        """计算相对位姿：从pose_prev到pose_curr"""
        pose_prev_inv = np.linalg.inv(pose_prev)
        relative_pose = pose_curr @ pose_prev_inv
        return relative_pose

    def __getitem__(self, index):
        while True:
            try:
                data = {}
                data_id = torch.randint(0, len(self.path), (1,))[0]
                data_id = (data_id + index) % len(self.path)
                
                # 加载单个相机的数据
                path = self.path[data_id]
                video_data = torch.load(path, weights_only=True, map_location="cpu")

                # 获取视频latents
                full_latents = video_data['latents']  # [C, T, H, W]
                total_frames = full_latents.shape[1]
                
                # 检查是否有足够的帧数
                required_frames = self.condition_frames + self.target_frames
                if total_frames < required_frames:
                    continue
                
                # 随机选择起始位置
                max_start = total_frames - required_frames
                start_frame = random.randint(0, max_start) if max_start > 0 else 0
                
                # 提取condition和target段
                condition_latents = full_latents[:, start_frame:start_frame+self.condition_frames, :, :]
                target_latents = full_latents[:, start_frame+self.condition_frames:start_frame+self.condition_frames+self.target_frames, :, :]
                
                # 拼接latents [condition, target] - 注意：训练时condition帧在前，target帧在后
                data['latents'] = torch.cat([condition_latents, target_latents], dim=1)
                
                data['prompt_emb'] = video_data['prompt_emb']
                data['image_emb'] = video_data.get('image_emb', {})

                # 加载相机轨迹数据，生成时序相对位姿
                base_path = path.rsplit('/', 2)[0]
                camera_path = os.path.join(base_path, "cameras", "camera_extrinsics.json")
                
                if not os.path.exists(camera_path):
                    # 如果没有相机数据，生成零向量 - 只为target帧生成
                    pose_embedding = torch.zeros(self.target_frames, 12, dtype=torch.bfloat16)
                else:
                    with open(camera_path, 'r') as file:
                        cam_data = json.load(file)
                    
                    # 提取相机路径（使用相同相机的不同时间点）
                    match = re.search(r'cam(\d+)', path)
                    cam_idx = int(match.group(1)) if match else 1
                    
                    # 为target帧生成相对位姿
                    relative_poses = []
                    
                    # 计算每个target帧相对于condition最后一帧的位姿
                    condition_end_frame_idx = start_frame + self.condition_frames - 1
                    
                    # 获取reference pose（condition段的最后一帧）
                    if f"frame{condition_end_frame_idx}" in cam_data and f"cam{cam_idx:02d}" in cam_data[f"frame{condition_end_frame_idx}"]:
                        reference_matrix_str = cam_data[f"frame{condition_end_frame_idx}"][f"cam{cam_idx:02d}"]
                        reference_pose = self.parse_matrix(reference_matrix_str)
                        if reference_pose.shape == (3, 4):
                            reference_pose = np.vstack([reference_pose, np.array([0, 0, 0, 1.0])])
                    else:
                        reference_pose = np.eye(4, dtype=np.float32)
                    
                    # 🔧 修复：为每个target帧计算相对位姿
                    for i in range(self.target_frames):
                        target_frame_idx = start_frame + self.condition_frames + i
                        
                        if f"frame{target_frame_idx}" in cam_data and f"cam{cam_idx:02d}" in cam_data[f"frame{target_frame_idx}"]:
                            target_matrix_str = cam_data[f"frame{target_frame_idx}"][f"cam{cam_idx:02d}"]
                            target_pose = self.parse_matrix(target_matrix_str)
                            if target_pose.shape == (3, 4):
                                target_pose = np.vstack([target_pose, np.array([0, 0, 0, 1.0])])
                            
                            # 🔧 修复：正确调用get_relative_pose方法
                            relative_pose = self.get_relative_pose(reference_pose, target_pose)
                            relative_poses.append(torch.as_tensor(relative_pose[:3, :]))  # 取前3行
                        else:
                            # 如果没有对应帧的数据，使用单位矩阵
                            relative_poses.append(torch.as_tensor(np.eye(3, 4, dtype=np.float32)))
                    
                    pose_embedding = torch.stack(relative_poses, dim=0)  # [target_frames, 3, 4]
                    pose_embedding = rearrange(pose_embedding, 'b c d -> b (c d)')  # [target_frames, 12]
                
                data['camera'] = pose_embedding.to(torch.bfloat16)
                break
                
            except Exception as e:
                print(f"ERROR WHEN LOADING: {e}")
                index = random.randrange(len(self.path))
        
        return data

    def __len__(self):
        return self.steps_per_epoch


class LightningModelForTrain(pl.LightningModule):
    def __init__(
        self,
        dit_path,
        learning_rate=1e-5,
        use_gradient_checkpointing=True, use_gradient_checkpointing_offload=False,
        resume_ckpt_path=None,
        condition_frames=10,
        target_frames=5,
    ):
        super().__init__()
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        if os.path.isfile(dit_path):
            model_manager.load_models([dit_path])
        else:
            dit_path = dit_path.split(",")
            model_manager.load_models([dit_path])
        self.pipe = WanVideoReCamMasterPipeline.from_model_manager(model_manager)
        self.pipe.scheduler.set_timesteps(1000, training=True)

        dim=self.pipe.dit.blocks[0].self_attn.q.weight.shape[0]
        for block in self.pipe.dit.blocks:
            block.cam_encoder = nn.Linear(12, dim)
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
        self.condition_frames = int(condition_frames)
        self.target_frames = int(target_frames)
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
        
        
    def freeze_parameters(self):
        # Freeze parameters
        self.pipe.requires_grad_(False)
        self.pipe.eval()
        self.pipe.denoising_model().train()
        

    def training_step(self, batch, batch_idx):
        # Data
        latents = batch["latents"].to(self.device)  # [B, C, T, H, W], T = condition_frames + target_frames
        prompt_emb = batch["prompt_emb"]
        prompt_emb["context"] = prompt_emb["context"][0].to(self.device)
        image_emb = batch["image_emb"]
        
        # target_height, target_width = 40, 70
        # current_height, current_width = latents.shape[3], latents.shape[4]
        
        # if current_height > target_height or current_width > target_width:
        #     h_start = (current_height - target_height) // 2
        #     w_start = (current_width - target_width) // 2
        #     latents = latents[:, :, :, 
        #                     h_start:h_start+target_height, 
        #                     w_start:w_start+target_width]
            
        if "clip_feature" in image_emb:
            image_emb["clip_feature"] = image_emb["clip_feature"][0].to(self.device)
        if "y" in image_emb:
            image_emb["y"] = image_emb["y"][0].to(self.device)
        
        cam_emb = batch["camera"].to(self.device)  # [B, target_frames, 12] - 只有target帧的pose

        # Loss
        self.pipe.device = self.device
        noise = torch.randn_like(latents)
        timestep_id = torch.randint(0, self.pipe.scheduler.num_train_timesteps, (1,))
        timestep = self.pipe.scheduler.timesteps[timestep_id].to(dtype=self.pipe.torch_dtype, device=self.pipe.device)

        extra_input = self.pipe.prepare_extra_input(latents)
        origin_latents = copy.deepcopy(latents)
        noisy_latents = self.pipe.scheduler.add_noise(latents, noise, timestep)

        # 🔧 修复：condition段在前，保持clean；target段在后，参与去噪训练
        cond_len = self.condition_frames
        noisy_latents[:, :, :cond_len, ...] = origin_latents[:, :, :cond_len, ...]
        training_target = self.pipe.scheduler.training_target(latents, noise, timestep)
        
        # Compute loss (只对target段计算loss)
        noise_pred = self.pipe.denoising_model()(
            noisy_latents, timestep=timestep, cam_emb=cam_emb, **prompt_emb, **extra_input, **image_emb,
            use_gradient_checkpointing=self.use_gradient_checkpointing,
            use_gradient_checkpointing_offload=self.use_gradient_checkpointing_offload
        )
        
        # 🔧 修复：只对target段（后半部分）计算loss
        target_noise_pred = noise_pred[:, :, cond_len:, ...]
        target_training_target = training_target[:, :, cond_len:, ...]
        
        loss = torch.nn.functional.mse_loss(
            target_noise_pred.float(),
            target_training_target.float()
        )
        loss = loss * self.pipe.scheduler.training_weight(timestep)

        wandb.log({
            "train_loss": loss.item(),
            "condition_frames": cond_len,
            "target_frames": self.target_frames,
        })
        return loss

    def configure_optimizers(self):
        trainable_modules = filter(lambda p: p.requires_grad, self.pipe.denoising_model().parameters())
        optimizer = torch.optim.AdamW(trainable_modules, lr=self.learning_rate)
        return optimizer
    

    def on_save_checkpoint(self, checkpoint):
        checkpoint_dir = "/share_zhuyixuan05/zhuyixuan05/recam_future"
        print(f"Checkpoint directory: {checkpoint_dir}")
        current_step = self.global_step
        print(f"Current step: {current_step}")

        checkpoint.clear()
        trainable_param_names = list(filter(lambda named_param: named_param[1].requires_grad, self.pipe.denoising_model().named_parameters()))
        trainable_param_names = set([named_param[0] for named_param in trainable_param_names])
        state_dict = self.pipe.denoising_model().state_dict()
        torch.save(state_dict, os.path.join(checkpoint_dir, f"step{current_step}.ckpt"))



def parse_args():
    parser = argparse.ArgumentParser(description="Train ReCamMaster")
    parser.add_argument(
        "--task",
        type=str,
        default="train",
        choices=["data_process", "train"],
        help="Task. `data_process` or `train`.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/share_zhuyixuan05/zhuyixuan05/MultiCamVideo-Dataset",
        help="The path of the Dataset.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./",
        help="Path to save the model.",
    )
    parser.add_argument(
        "--text_encoder_path",
        type=str,
        default=None,
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
        default=None,
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
        default=False,
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
        "--height",
        type=int,
        default=480,
        help="Image height.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=832,
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
        "--condition_frames",
        type=int,
        default=10,
        help="Number of condition frames (kept clean).",
    )
    parser.add_argument(
        "--target_frames",
        type=int,
        default=10,
        help="Number of target frames (to be denoised).",
    )
    args = parser.parse_args()
    return args


def data_process(args):
    dataset = TextVideoDataset(
        args.dataset_path,
        os.path.join(args.dataset_path, args.metadata_file_name),
        max_num_frames=args.num_frames,
        frame_interval=1,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
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
    dataset = TensorDataset(
        args.dataset_path,
        os.path.join(args.dataset_path, "metadata.csv"),
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
        condition_frames=args.condition_frames,
        target_frames=args.target_frames,
    )

    if args.use_swanlab:
        wandb.init(
            project="recam",
            name="recam",
        )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices="auto",
        precision="bf16",
        strategy=args.training_strategy,
        default_root_dir=args.output_path,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=[],
    )
    trainer.fit(model, dataloader)


if __name__ == '__main__':
    args = parse_args()
    os.makedirs(os.path.join(args.output_path, "checkpoints"), exist_ok=True)
    if args.task == "data_process":
        data_process(args)
    elif args.task == "train":
        train(args)