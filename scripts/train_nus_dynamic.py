import torch
import torch.nn as nn
import lightning as pl
import wandb
import os
import copy
from diffsynth import WanVideoReCamMasterPipeline, ModelManager
import os
import json
import torch
import numpy as np
from PIL import Image
import imageio
import random
from torchvision.transforms import v2
from einops import rearrange
from pose_classifier import PoseClassifier

class DynamicNuScenesDataset(torch.utils.data.Dataset):
    """支持动态历史长度的NuScenes数据集"""
    
    def __init__(self, base_path, steps_per_epoch, 
                 min_condition_frames=10, max_condition_frames=40,
                 target_frames=10, height=900, width=1600):
        self.base_path = base_path
        self.scenes_path = os.path.join(base_path, "scenes")
        self.min_condition_frames = min_condition_frames
        self.max_condition_frames = max_condition_frames
        self.target_frames = target_frames
        self.height = height
        self.width = width
        self.steps_per_epoch = steps_per_epoch
        self.pose_classifier = PoseClassifier()
        
        # 🔧 新增：VAE时间压缩比例
        self.time_compression_ratio = 4  # VAE将时间维度压缩4倍
        
        # 查找所有处理好的场景
        self.scene_dirs = []
        if os.path.exists(self.scenes_path):
            for item in os.listdir(self.scenes_path):
                scene_dir = os.path.join(self.scenes_path, item)
                if os.path.isdir(scene_dir):
                    scene_info_path = os.path.join(scene_dir, "scene_info.json")
                    if os.path.exists(scene_info_path):
                        # 检查是否有编码的tensor文件
                        encoded_path = os.path.join(scene_dir, "encoded_video-480p.pth")
                        if os.path.exists(encoded_path):
                            self.scene_dirs.append(scene_dir)
        
        # print(f"Found {len(self.scene_dirs)} scenes with encoded data")
        assert len(self.scene_dirs) > 0, "No encoded scenes found!"
        
        # 预处理设置
        self.frame_process = v2.Compose([
            v2.CenterCrop(size=(height, width)),
            v2.Resize(size=(height, width), antialias=True),
            v2.ToTensor(),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def calculate_relative_rotation(self, current_rotation, reference_rotation):
        """计算相对旋转四元数"""
        q_current = torch.tensor(current_rotation, dtype=torch.float32)
        q_ref = torch.tensor(reference_rotation, dtype=torch.float32)

        q_ref_inv = torch.tensor([q_ref[0], -q_ref[1], -q_ref[2], -q_ref[3]])

        w1, x1, y1, z1 = q_ref_inv
        w2, x2, y2, z2 = q_current

        relative_rotation = torch.tensor([
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        ])

        return relative_rotation
    
    def select_dynamic_segment(self, scene_info):
        """动态选择条件帧和目标帧 - 修正版本处理VAE时间压缩"""
        keyframe_indices = scene_info['keyframe_indices']  # 原始帧索引
        total_frames = scene_info['total_frames']  # 原始总帧数
        
        if len(keyframe_indices) < 2:
            print('error1____________')
            return None
        
        # 🔧 计算压缩后的帧数
        compressed_total_frames = total_frames // self.time_compression_ratio
        compressed_keyframe_indices = [idx // self.time_compression_ratio for idx in keyframe_indices]
        
        # print(f"原始总帧数: {total_frames}, 压缩后: {compressed_total_frames}")
        # print(f"原始关键帧: {keyframe_indices[:5]}..., 压缩后: {compressed_keyframe_indices[:5]}...")
        
        # 随机选择条件帧长度（基于压缩后的帧数）

        min_condition_compressed = self.min_condition_frames // self.time_compression_ratio
        max_condition_compressed = self.max_condition_frames // self.time_compression_ratio
        target_frames_compressed = self.target_frames // self.time_compression_ratio
        
        ratio = random.random()
        print('ratio:',ratio)
        if ratio<0.15:
            condition_frames_compressed = 1
        elif 0.15<=ratio<0.3:
            condition_frames_compressed = random.randint(min_condition_compressed, max_condition_compressed)
        else:
            condition_frames_compressed = target_frames_compressed
        
        # 确保有足够的帧数
        min_required_frames = condition_frames_compressed + target_frames_compressed
        if compressed_total_frames < min_required_frames:
            print(f"压缩后帧数不足: {compressed_total_frames} < {min_required_frames}")
            return None
        
        # 随机选择起始位置（基于压缩后的帧数）
        max_start = compressed_total_frames - min_required_frames
        start_frame_compressed = random.randint(0, max_start)
        
        condition_end_compressed = start_frame_compressed + condition_frames_compressed
        target_end_compressed = condition_end_compressed + target_frames_compressed
        
        # 🔧 关键修复：在压缩空间中查找关键帧
        condition_keyframes_compressed = [idx for idx in compressed_keyframe_indices 
                                        if start_frame_compressed <= idx < condition_end_compressed]
        
        target_keyframes_compressed = [idx for idx in compressed_keyframe_indices 
                                     if condition_end_compressed <= idx < target_end_compressed]
        
        if not condition_keyframes_compressed:
            print(f"条件段内无关键帧: {start_frame_compressed}-{condition_end_compressed}")
            return None
        
        # 使用条件段的最后一个关键帧作为reference
        reference_keyframe_compressed = max(condition_keyframes_compressed)
        
        # 🔧 找到对应的原始关键帧索引用于pose查找
        reference_keyframe_original_idx = None
        for i, compressed_idx in enumerate(compressed_keyframe_indices):
            if compressed_idx == reference_keyframe_compressed:
                reference_keyframe_original_idx = i
                break
        
        if reference_keyframe_original_idx is None:
            print(f"无法找到reference关键帧的原始索引")
            return None
        
        # 找到目标段对应的原始关键帧索引
        target_keyframes_original_indices = []
        for compressed_idx in target_keyframes_compressed:
            for i, comp_idx in enumerate(compressed_keyframe_indices):
                if comp_idx == compressed_idx:
                    target_keyframes_original_indices.append(i)
                    break
        
        return {
            'start_frame': start_frame_compressed,  # 压缩后的起始帧
            'condition_frames': condition_frames_compressed,  # 压缩后的条件帧数
            'target_frames': target_frames_compressed,  # 压缩后的目标帧数
            'condition_range': (start_frame_compressed, condition_end_compressed),
            'target_range': (condition_end_compressed, target_end_compressed),
            'reference_keyframe_idx': reference_keyframe_original_idx,  # 原始关键帧索引
            'target_keyframe_indices': target_keyframes_original_indices,  # 原始关键帧索引列表
            'original_condition_frames': condition_frames_compressed * self.time_compression_ratio,  # 用于记录
            'original_target_frames': target_frames_compressed * self.time_compression_ratio,
        }
    

    def create_pose_embeddings(self, scene_info, segment_info):
        """创建pose embeddings - 修正版本，包含condition和target的实际pose"""
        keyframe_poses = scene_info['keyframe_poses']
        reference_keyframe_idx = segment_info['reference_keyframe_idx']
        target_keyframe_indices = segment_info['target_keyframe_indices']
        
        if reference_keyframe_idx >= len(keyframe_poses):
            return None
        
        reference_pose = keyframe_poses[reference_keyframe_idx]
        
        # 🔧 关键修复：pose向量应该包含condition帧和target帧的实际pose数据
        condition_frames = segment_info['condition_frames']  # 压缩后的condition帧数
        target_frames = segment_info['target_frames']        # 压缩后的target帧数
        total_frames = condition_frames + target_frames      # 总帧数，与latent对齐
        
        print(f"创建pose embedding: condition_frames={condition_frames}, target_frames={target_frames}, total_frames={total_frames}")
        
        # 🔧 获取condition段的关键帧索引
        start_frame = segment_info['start_frame']
        condition_end_compressed = start_frame + condition_frames
        
        # 压缩后的关键帧索引
        compressed_keyframe_indices = [idx // self.time_compression_ratio for idx in scene_info['keyframe_indices']]
        
        # 找到condition段的关键帧
        condition_keyframes_compressed = [idx for idx in compressed_keyframe_indices 
                                        if start_frame <= idx < condition_end_compressed]
        
        # 找到对应的原始关键帧索引
        condition_keyframes_original_indices = []
        for compressed_idx in condition_keyframes_compressed:
            for i, comp_idx in enumerate(compressed_keyframe_indices):
                if comp_idx == compressed_idx:
                    condition_keyframes_original_indices.append(i)
                    break
        
        pose_vecs = []
        frame_types = []  # 新增：记录每帧是condition还是target
        
        # 🔧 前面的condition帧使用实际的pose数据
        for i in range(condition_frames):
            if not condition_keyframes_original_indices:
                # 如果condition段没有关键帧，使用reference pose
                translation = torch.zeros(3, dtype=torch.float32)
                rotation = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)  # 单位四元数
            else:
                # 为condition帧分配pose
                if len(condition_keyframes_original_indices) == 1:
                    keyframe_idx = condition_keyframes_original_indices[0]
                else:
                    # 线性插值选择关键帧
                    if condition_frames == 1:
                        keyframe_idx = condition_keyframes_original_indices[0]
                    else:
                        interp_ratio = i / (condition_frames - 1)
                        interp_idx = int(interp_ratio * (len(condition_keyframes_original_indices) - 1))
                        keyframe_idx = condition_keyframes_original_indices[interp_idx]
                
                if keyframe_idx >= len(keyframe_poses):
                    translation = torch.zeros(3, dtype=torch.float32)
                    rotation = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)
                else:
                    condition_pose = keyframe_poses[keyframe_idx]
                    
                    # 计算相对于reference的pose
                    translation = torch.tensor(
                        np.array(condition_pose['translation']) - np.array(reference_pose['translation']),
                        dtype=torch.float32
                    )
                    
                    relative_rotation = self.calculate_relative_rotation(
                        condition_pose['rotation'],
                        reference_pose['rotation']
                    )
                    
                    rotation = relative_rotation
            
            # 🔧 添加frame type embedding：0表示condition
            pose_vec = torch.cat([translation, rotation, torch.tensor([0.0], dtype=torch.float32)], dim=0)  # [3+4+1=8D]
            pose_vecs.append(pose_vec)
            frame_types.append('condition')
        
        # 🔧 后面的target帧使用实际的pose数据
        if not target_keyframe_indices:
            # 如果目标段没有关键帧，target帧使用零向量
            for i in range(target_frames):
                pose_vec = torch.cat([
                    torch.zeros(3, dtype=torch.float32),  # translation
                    torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),  # rotation
                    torch.tensor([1.0], dtype=torch.float32)  # frame type: 1表示target
                ], dim=0)
                pose_vecs.append(pose_vec)
                frame_types.append('target')
        else:
            # 为每个target帧分配pose
            for i in range(target_frames):
                if len(target_keyframe_indices) == 1:
                    # 只有一个关键帧，所有target帧使用相同的pose
                    target_keyframe_idx = target_keyframe_indices[0]
                else:
                    # 多个关键帧，线性插值选择
                    if target_frames == 1:
                        # 只有一帧，使用第一个关键帧
                        target_keyframe_idx = target_keyframe_indices[0]
                    else:
                        # 线性插值
                        interp_ratio = i / (target_frames - 1)
                        interp_idx = int(interp_ratio * (len(target_keyframe_indices) - 1))
                        target_keyframe_idx = target_keyframe_indices[interp_idx]
                
                if target_keyframe_idx >= len(keyframe_poses):
                    pose_vec = torch.cat([
                        torch.zeros(3, dtype=torch.float32),
                        torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
                        torch.tensor([1.0], dtype=torch.float32)  # target
                    ], dim=0)
                else:
                    target_pose = keyframe_poses[target_keyframe_idx]
                    
                    # 计算相对pose
                    relative_translation = torch.tensor(
                        np.array(target_pose['translation']) - np.array(reference_pose['translation']),
                        dtype=torch.float32
                    )
                    
                    relative_rotation = self.calculate_relative_rotation(
                        target_pose['rotation'],
                        reference_pose['rotation']
                    )
                    
                    # 🔧 添加frame type embedding：1表示target
                    pose_vec = torch.cat([
                        relative_translation, 
                        relative_rotation, 
                        torch.tensor([1.0], dtype=torch.float32)
                    ], dim=0)  # [8D]
                
                pose_vecs.append(pose_vec)
                frame_types.append('target')
        
        if not pose_vecs:
            print("❌ 没有生成任何pose向量")
            return None
        
        pose_sequence = torch.stack(pose_vecs, dim=0)  # [total_frames, 8]
        print(f"生成pose序列形状: {pose_sequence.shape}")
        print(f"期望形状: [{total_frames}, 8]")
        print(f"帧类型分布: {frame_types}")
        
        # 🔧 只对target部分进行分类分析（condition部分不需要分类）
        target_pose_sequence = pose_sequence[condition_frames:, :7]  # 只取target部分的前7维
        
        if target_pose_sequence.numel() == 0:
            print("❌ Target pose序列为空")
            return None
        
        # 使用分类器分析target部分
        pose_analysis = self.pose_classifier.analyze_pose_sequence(target_pose_sequence)
        
        # 过滤掉backward样本
        class_distribution = pose_analysis['class_distribution']
        # if 'backward' in class_distribution and class_distribution['backward'] > 0:
        #     print(f"⚠️  检测到backward运动，跳过样本")
        #     return None
        
        # 🔧 创建完整的类别embedding（包含condition和target）
        # condition帧的类别标签设为forward（或者可以设为特殊的"condition"类别）
        condition_classes = torch.full((condition_frames,), 0, dtype=torch.long)  # 0表示forward/condition
        target_classes = pose_analysis['classifications']
        
        # 拼接condition和target的类别
        full_classes = torch.cat([condition_classes, target_classes], dim=0)
        
        # 🔧 创建enhanced class embedding，包含frame type信息
        class_embeddings = self.create_enhanced_class_embedding(
            full_classes, pose_sequence, embed_dim=512
        )
        
        print(f"最终class embedding形状: {class_embeddings.shape}")
        print(f"期望形状: [{total_frames}, 512]")
        
        # 🔧 验证embedding形状是否正确
        if class_embeddings.shape[0] != total_frames:
            print(f"❌ Embedding帧数不匹配: {class_embeddings.shape[0]} != {total_frames}")
            return None
        
        return {
            'raw_poses': pose_sequence,           # [total_frames, 8] 包含condition和target的实际pose + frame type
            'pose_classes': full_classes,         # [total_frames] 包含condition和target的类别
            'class_embeddings': class_embeddings, # [total_frames, 512] 增强的embedding
            'pose_analysis': pose_analysis,       # 只包含target部分的分析
            'condition_frames': condition_frames,
            'target_frames': target_frames,
            'frame_types': frame_types
        }

    def create_enhanced_class_embedding(self, class_labels: torch.Tensor, pose_sequence: torch.Tensor, embed_dim: int = 512) -> torch.Tensor:
        """
        创建增强的类别embedding，包含frame type和pose信息
        Args:
            class_labels: [num_frames] 类别标签
            pose_sequence: [num_frames, 8] pose序列，最后一维是frame type
            embed_dim: embedding维度
        Returns:
            embeddings: [num_frames, embed_dim]
        """
        num_classes = 4
        num_frames = len(class_labels)
        
        # 基础的方向embedding
        direction_vectors = torch.tensor([
            [1.0, 0.0, 0.0, 0.0],  # forward: 主要x分量
            [-1.0, 0.0, 0.0, 0.0], # backward: 负x分量  
            [0.0, 1.0, 0.0, 0.0],  # left_turn: 主要y分量
            [0.0, -1.0, 0.0, 0.0], # right_turn: 负y分量
        ], dtype=torch.float32)
        
        # One-hot编码
        one_hot = torch.zeros(num_frames, num_classes)
        one_hot.scatter_(1, class_labels.unsqueeze(1), 1)
        
        # 基于方向向量的基础embedding
        base_embeddings = one_hot @ direction_vectors  # [num_frames, 4]
        
        # 🔧 添加frame type信息
        frame_types = pose_sequence[:, -1]  # 最后一维是frame type
        frame_type_embeddings = torch.zeros(num_frames, 2)
        frame_type_embeddings[:, 0] = (frame_types == 0).float()  # condition
        frame_type_embeddings[:, 1] = (frame_types == 1).float()  # target
        
        # 🔧 添加pose的几何信息
        translations = pose_sequence[:, :3]  # [num_frames, 3]
        rotations = pose_sequence[:, 3:7]    # [num_frames, 4]
        
        # 组合所有特征
        combined_features = torch.cat([
            base_embeddings,         # [num_frames, 4] 方向特征
            frame_type_embeddings,   # [num_frames, 2] 帧类型特征
            translations,            # [num_frames, 3] 位移特征
            rotations,               # [num_frames, 4] 旋转特征
        ], dim=1)  # [num_frames, 13]
        
        # 扩展到目标维度
        if embed_dim > 13:
            # 使用线性变换扩展
            expand_matrix = torch.randn(13, embed_dim) * 0.1
            # 保持重要特征
            expand_matrix[:13, :13] = torch.eye(13)
            embeddings = combined_features @ expand_matrix
        else:
            embeddings = combined_features[:, :embed_dim]
        
        return embeddings

    def __getitem__(self, index):
        while True:
            try:
                # 随机选择一个场景
                scene_dir = random.choice(self.scene_dirs)
                
                # 加载场景信息
                with open(os.path.join(scene_dir, "scene_info.json"), 'r') as f:
                    scene_info = json.load(f)
                
                # 加载编码的视频数据
                encoded_data = torch.load(
                    os.path.join(scene_dir, "encoded_video-480p.pth"),
                    weights_only=True,
                    map_location="cpu"
                )
                
                # 🔧 验证latent帧数是否符合预期
                full_latents = encoded_data['latents']  # [C, T, H, W]
                expected_latent_frames = scene_info['total_frames'] // self.time_compression_ratio
                actual_latent_frames = full_latents.shape[1]
                
                # print(f"场景 {os.path.basename(scene_dir)}: 原始帧数={scene_info['total_frames']}, "
                #       f"预期latent帧数={expected_latent_frames}, 实际latent帧数={actual_latent_frames}")
                
                if abs(actual_latent_frames - expected_latent_frames) > 2:  # 允许小的舍入误差
                    print(f"⚠️  Latent帧数不匹配，跳过此样本")
                    continue
                
                # 动态选择段落
                segment_info = self.select_dynamic_segment(scene_info)
                if segment_info is None:
                    continue
                
                # 创建pose embeddings
                pose_data = self.create_pose_embeddings(scene_info, segment_info)
                if pose_data is None:
                    continue
                
                # 🔧 使用压缩后的索引提取latent段落
                start_frame = segment_info['start_frame']  # 已经是压缩后的索引
                condition_frames = segment_info['condition_frames']  # 已经是压缩后的帧数
                target_frames = segment_info['target_frames']  # 已经是压缩后的帧数
                
                # print(f"提取latent段落: start={start_frame}, condition={condition_frames}, target={target_frames}")
                # print(f"Full latents shape: {full_latents.shape}")
                
                # 确保索引不越界
                if start_frame + condition_frames + target_frames > full_latents.shape[1]:
                    print(f"索引越界，跳过: {start_frame + condition_frames + target_frames} > {full_latents.shape[1]}")
                    continue
                
                condition_latents = full_latents[:, start_frame:start_frame+condition_frames, :, :]
                
                target_latents = full_latents[:, start_frame+condition_frames:start_frame+condition_frames+target_frames, :, :]
                
                # print(f"Condition latents shape: {condition_latents.shape}")
                # print(f"Target latents shape: {target_latents.shape}")
                
                # 拼接latents [condition, target]
                combined_latents = torch.cat([condition_latents, target_latents], dim=1)
                
                result = {
                    "latents": combined_latents,
                    "prompt_emb": encoded_data["prompt_emb"],
                    "image_emb": encoded_data.get("image_emb", {}),
                    "camera": pose_data['class_embeddings'].to(torch.bfloat16),
                    "pose_classes": pose_data['pose_classes'],
                    "raw_poses": pose_data['raw_poses'],
                    "pose_analysis": pose_data['pose_analysis'],
                    "condition_frames": condition_frames,  # 压缩后的帧数
                    "target_frames": target_frames,  # 压缩后的帧数
                    "scene_name": os.path.basename(scene_dir),
                    # 🔧 新增：记录原始帧数用于调试
                    "original_condition_frames": segment_info['original_condition_frames'],
                    "original_target_frames": segment_info['original_target_frames'],
                }
                
                return result
                
            except Exception as e:
                print(f"Error loading sample: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    def __len__(self):
        return self.steps_per_epoch
    
class DynamicLightningModelForTrain(pl.LightningModule):
    def __init__(
        self,
        dit_path,
        learning_rate=1e-5,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
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

        # 添加相机编码器
        dim = self.pipe.dit.blocks[0].self_attn.q.weight.shape[0]
        for block in self.pipe.dit.blocks:
            block.cam_encoder = nn.Linear(512, dim)
            block.projector = nn.Linear(dim, dim)
            block.cam_encoder.weight.data.zero_()
            block.cam_encoder.bias.data.zero_()
            block.projector.weight = nn.Parameter(torch.eye(dim))
            block.projector.bias = nn.Parameter(torch.zeros(dim))
        
        if resume_ckpt_path is not None:
            state_dict = torch.load(resume_ckpt_path, map_location="cpu")
            self.pipe.dit.load_state_dict(state_dict, strict=True)

        self.freeze_parameters()
        
        # 只训练相机相关和注意力模块
        for name, module in self.pipe.denoising_model().named_modules():
            if any(keyword in name for keyword in ["cam_encoder", "projector", "self_attn"]):
                for param in module.parameters():
                    param.requires_grad = True
        
        self.learning_rate = learning_rate
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        
        # 创建可视化目录
        self.vis_dir = "nus/visualizations_dynamic"
        os.makedirs(self.vis_dir, exist_ok=True)
        
    def freeze_parameters(self):
        self.pipe.requires_grad_(False)
        self.pipe.eval()
        self.pipe.denoising_model().train()

    def training_step(self, batch, batch_idx):
        # 获取动态长度信息（这些已经是压缩后的帧数）
        condition_frames = batch["condition_frames"][0].item()  # 压缩后的condition长度
        target_frames = batch["target_frames"][0].item()       # 压缩后的target长度
        
        # 🔧 获取原始帧数用于日志记录
        original_condition_frames = batch.get("original_condition_frames", [condition_frames * 4])[0]
        original_target_frames = batch.get("original_target_frames", [target_frames * 4])[0]
        
        # Data
        latents = batch["latents"].to(self.device)
        # print(f"压缩后condition帧数: {condition_frames}, target帧数: {target_frames}")
        # print(f"原始condition帧数: {original_condition_frames}, target帧数: {original_target_frames}")
        # print(f"Latents shape: {latents.shape}")
        
        # 裁剪空间尺寸以节省内存
        # target_height, target_width = 50, 70
        # current_height, current_width = latents.shape[3], latents.shape[4]
        
        # if current_height > target_height or current_width > target_width:
        #     h_start = (current_height - target_height) // 2
        #     w_start = (current_width - target_width) // 2
        #     latents = latents[:, :, :, 
        #                     h_start:h_start+target_height, 
        #                     w_start:w_start+target_width]
        
        prompt_emb = batch["prompt_emb"]
        prompt_emb["context"] = prompt_emb["context"][0].to(self.device)
        image_emb = batch["image_emb"]
        # print(f"裁剪后latents shape: {latents.shape}")

        if "clip_feature" in image_emb:
            image_emb["clip_feature"] = image_emb["clip_feature"][0].to(self.device)
        if "y" in image_emb:
            image_emb["y"] = image_emb["y"][0].to(self.device)
        
        cam_emb = batch["camera"].to(self.device)

        # Loss计算
        self.pipe.device = self.device
        noise = torch.randn_like(latents)
        timestep_id = torch.randint(0, self.pipe.scheduler.num_train_timesteps, (1,))
        timestep = self.pipe.scheduler.timesteps[timestep_id].to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
        
        extra_input = self.pipe.prepare_extra_input(latents)
        origin_latents = copy.deepcopy(latents)
        noisy_latents = self.pipe.scheduler.add_noise(latents, noise, timestep)
        
        # 🔧 关键：使用压缩后的condition长度
        # condition部分保持clean，只对target部分加噪
        noisy_latents[:, :, :condition_frames, ...] = origin_latents[:, :, :condition_frames, ...]
        training_target = self.pipe.scheduler.training_target(latents, noise, timestep)
        # print(f"targe尺寸: {training_target.shape}")
        # 预测噪声
        noise_pred = self.pipe.denoising_model()(
            noisy_latents, timestep=timestep, cam_emb=cam_emb, **prompt_emb, **extra_input, **image_emb,
            use_gradient_checkpointing=self.use_gradient_checkpointing,
            use_gradient_checkpointing_offload=self.use_gradient_checkpointing_offload
        )
        # print(f"pred尺寸: {training_target.shape}")
        # 🔧 只对target部分计算loss（使用压缩后的索引）
        target_noise_pred = noise_pred[:, :, condition_frames:condition_frames+target_frames, ...]
        target_training_target = training_target[:, :, condition_frames:condition_frames+target_frames, ...]
        
        loss = torch.nn.functional.mse_loss(target_noise_pred.float(), target_training_target.float())
        loss = loss * self.pipe.scheduler.training_weight(timestep)
        print('--------loss------------:',loss)

        # 记录额外信息
        wandb.log({
            "train_loss": loss.item(),
            "timestep": timestep.item(),
            "condition_frames_compressed": condition_frames,  # 压缩后的帧数000
            "target_frames_compressed": target_frames,
            "condition_frames_original": original_condition_frames,  # 原始帧数
            "target_frames_original": original_target_frames,
            "total_frames_compressed": condition_frames + target_frames,
            "total_frames_original": original_condition_frames + original_target_frames,
            "global_step": self.global_step
        })

        return loss

    def configure_optimizers(self):
        trainable_modules = filter(lambda p: p.requires_grad, self.pipe.denoising_model().parameters())
        optimizer = torch.optim.AdamW(trainable_modules, lr=self.learning_rate)
        return optimizer
    
    def on_save_checkpoint(self, checkpoint):
        checkpoint_dir = "/home/zhuyixuan05/ReCamMaster/nus_dynamic"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        current_step = self.global_step
        checkpoint.clear()
        
        state_dict = self.pipe.denoising_model().state_dict()
        torch.save(state_dict, os.path.join(checkpoint_dir, f"step{current_step}_dynamic.ckpt"))
        print(f"Saved dynamic model checkpoint: step{current_step}_dynamic.ckpt")

def train_dynamic(args):
    """训练支持动态历史长度的模型"""
    dataset = DynamicNuScenesDataset(
        args.dataset_path,
        steps_per_epoch=args.steps_per_epoch,
        min_condition_frames=args.min_condition_frames,
        max_condition_frames=args.max_condition_frames,
        target_frames=args.target_frames,
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        batch_size=1,
        num_workers=args.dataloader_num_workers
    )
    
    model = DynamicLightningModelForTrain(
        dit_path=args.dit_path,
        learning_rate=args.learning_rate,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        resume_ckpt_path=args.resume_ckpt_path,
    )

    wandb.init(
        project="nuscenes-dynamic-recam",
        name=f"dynamic-{args.min_condition_frames}-{args.max_condition_frames}",
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
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Dynamic ReCamMaster")
    parser.add_argument("--dataset_path", type=str, default="/share_zhuyixuan05/zhuyixuan05/nuscenes_video_generation_dynamic")
    parser.add_argument("--dit_path", type=str, default="models/Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors")
    parser.add_argument("--output_path", type=str, default="./")
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--steps_per_epoch", type=int, default=3000)
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--min_condition_frames", type=int, default=10, help="最小条件帧数")
    parser.add_argument("--max_condition_frames", type=int, default=40, help="最大条件帧数")
    parser.add_argument("--target_frames", type=int, default=32, help="目标帧数")
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--training_strategy", type=str, default="deepspeed_stage_1")
    parser.add_argument("--use_gradient_checkpointing", action="store_true")
    parser.add_argument("--use_gradient_checkpointing_offload", action="store_true")
    parser.add_argument("--resume_ckpt_path", type=str, default=None)
    
    args = parser.parse_args()
    
    train_dynamic(args)