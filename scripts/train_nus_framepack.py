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
    """支持FramePack机制的动态历史长度NuScenes数据集"""
    
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
        
        assert len(self.scene_dirs) > 0, "No encoded scenes found!"

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
    
    def select_dynamic_segment_framepack(self, scene_info):
        """🔧 FramePack风格的动态选择条件帧和目标帧"""
        keyframe_indices = scene_info['keyframe_indices']  # 原始帧索引
        total_frames = scene_info['total_frames']  # 原始总帧数
        
        if len(keyframe_indices) < 2:
            print('error1____________')
            return None
        
        # 🔧 计算压缩后的帧数
        compressed_total_frames = total_frames // self.time_compression_ratio
        compressed_keyframe_indices = [idx // self.time_compression_ratio for idx in keyframe_indices]
        
        # 随机选择条件帧长度（基于压缩后的帧数）
        min_condition_compressed = self.min_condition_frames // self.time_compression_ratio
        max_condition_compressed = self.max_condition_frames // self.time_compression_ratio
        target_frames_compressed = self.target_frames // self.time_compression_ratio
        
        # 🔧 FramePack风格的采样策略
        ratio = random.random()
        print('ratio:', ratio)
        if ratio < 0.15:
            condition_frames_compressed = 1
        elif 0.15 <= ratio < 0.9:
            condition_frames_compressed = random.randint(min_condition_compressed, max_condition_compressed)
        else:
            condition_frames_compressed = target_frames_compressed
        
        # 确保有足够的帧数
        min_required_frames = condition_frames_compressed + target_frames_compressed
        if compressed_total_frames < min_required_frames:
            print(f"压缩后帧数不足: {compressed_total_frames} < {min_required_frames}")
            return None
        
        # 随机选择起始位置（基于压缩后的帧数）
        max_start = compressed_total_frames - min_required_frames - 1
        start_frame_compressed = random.randint(0, max_start)
        
        condition_end_compressed = start_frame_compressed + condition_frames_compressed
        target_end_compressed = condition_end_compressed + target_frames_compressed
        
        # 🔧 FramePack风格的索引处理
        latent_indices = torch.arange(condition_end_compressed, target_end_compressed)  # 只预测未来帧
        
        # 1x帧：起始帧 + 最后1帧
        clean_latent_indices_start = torch.tensor([start_frame_compressed])
        clean_latent_1x_indices = torch.tensor([condition_end_compressed - 1])
        clean_latent_indices = torch.cat([clean_latent_indices_start, clean_latent_1x_indices])
        
        # 🔧 2x帧：根据实际condition长度确定
        if condition_frames_compressed >= 2:
            # 取最后2帧（如果有的话）
            clean_latent_2x_start = max(start_frame_compressed, condition_end_compressed - 2-1)
            clean_latent_2x_indices = torch.arange(clean_latent_2x_start, condition_end_compressed-1)
        else:
            # 如果condition帧数不足2帧，创建空索引
            clean_latent_2x_indices = torch.tensor([], dtype=torch.long)
        
        # 🔧 4x帧：根据实际condition长度确定，最多16帧
        if condition_frames_compressed > 3:
            # 取最多16帧的历史（如果有的话）
            clean_4x_start = max(start_frame_compressed, condition_end_compressed - 16-3)
            clean_latent_4x_indices = torch.arange(clean_4x_start, condition_end_compressed-3)
        else:
            clean_latent_4x_indices = torch.tensor([], dtype=torch.long)
        
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
            
            # 🔧 FramePack风格的索引
            'latent_indices': latent_indices,
            'clean_latent_indices': clean_latent_indices,
            'clean_latent_2x_indices': clean_latent_2x_indices,
            'clean_latent_4x_indices': clean_latent_4x_indices,
        }

    def create_pose_embeddings(self, scene_info, segment_info):
        """🔧 为所有帧（condition + target）创建pose embeddings - FramePack风格"""
        keyframe_poses = scene_info['keyframe_poses']
        reference_keyframe_idx = segment_info['reference_keyframe_idx']
        target_keyframe_indices = segment_info['target_keyframe_indices']
        
        if reference_keyframe_idx >= len(keyframe_poses):
            return None
        
        reference_pose = keyframe_poses[reference_keyframe_idx]
        
        # 🔧 为所有帧（condition + target）计算pose embeddings
        start_frame = segment_info['start_frame']
        condition_end_compressed = start_frame + segment_info['condition_frames']
        target_end_compressed = condition_end_compressed + segment_info['target_frames']
        
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
        frame_types = []
        
        # 🔧 为condition帧计算pose
        for i in range(segment_info['condition_frames']):
            if not condition_keyframes_original_indices:
                translation = torch.zeros(3, dtype=torch.float32)
                rotation = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)
            else:
                # 为condition帧分配pose
                if len(condition_keyframes_original_indices) == 1:
                    keyframe_idx = condition_keyframes_original_indices[0]
                else:
                    if segment_info['condition_frames'] == 1:
                        keyframe_idx = condition_keyframes_original_indices[0]
                    else:
                        interp_ratio = i / (segment_info['condition_frames'] - 1)
                        interp_idx = int(interp_ratio * (len(condition_keyframes_original_indices) - 1))
                        keyframe_idx = condition_keyframes_original_indices[interp_idx]
                
                if keyframe_idx >= len(keyframe_poses):
                    translation = torch.zeros(3, dtype=torch.float32)
                    rotation = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)
                else:
                    condition_pose = keyframe_poses[keyframe_idx]
                    
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
            pose_vec = torch.cat([translation, rotation, torch.tensor([0.0], dtype=torch.float32)], dim=0)  # [8D]
            pose_vecs.append(pose_vec)
            frame_types.append('condition')
        
        # 🔧 为target帧计算pose
        if not target_keyframe_indices:
            for i in range(segment_info['target_frames']):
                pose_vec = torch.cat([
                    torch.zeros(3, dtype=torch.float32),
                    torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
                    torch.tensor([1.0], dtype=torch.float32)  # frame type: 1表示target
                ], dim=0)
                pose_vecs.append(pose_vec)
                frame_types.append('target')
        else:
            for i in range(segment_info['target_frames']):
                if len(target_keyframe_indices) == 1:
                    target_keyframe_idx = target_keyframe_indices[0]
                else:
                    if segment_info['target_frames'] == 1:
                        target_keyframe_idx = target_keyframe_indices[0]
                    else:
                        interp_ratio = i / (segment_info['target_frames'] - 1)
                        interp_idx = int(interp_ratio * (len(target_keyframe_indices) - 1))
                        target_keyframe_idx = target_keyframe_indices[interp_idx]
                
                if target_keyframe_idx >= len(keyframe_poses):
                    pose_vec = torch.cat([
                        torch.zeros(3, dtype=torch.float32),
                        torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
                        torch.tensor([1.0], dtype=torch.float32)
                    ], dim=0)
                else:
                    target_pose = keyframe_poses[target_keyframe_idx]
                    
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
                    ], dim=0)
                
                pose_vecs.append(pose_vec)
                frame_types.append('target')
        
        if not pose_vecs:
            print("❌ 没有生成任何pose向量")
            return None
        
        pose_sequence = torch.stack(pose_vecs, dim=0)  # [total_frames, 8]
        
        # 🔧 只对target部分进行分类分析
        target_pose_sequence = pose_sequence[segment_info['condition_frames']:, :7]
        
        if target_pose_sequence.numel() == 0:
            print("❌ Target pose序列为空")
            return None
        
        # 使用分类器分析target部分
        pose_analysis = self.pose_classifier.analyze_pose_sequence(target_pose_sequence)
        
        # 🔧 创建完整的类别embedding
        condition_classes = torch.full((segment_info['condition_frames'],), 0, dtype=torch.long)
        target_classes = pose_analysis['classifications']
        
        full_classes = torch.cat([condition_classes, target_classes], dim=0)
        
        # 🔧 创建enhanced class embedding
        class_embeddings = self.create_enhanced_class_embedding(
            full_classes, pose_sequence, embed_dim=512
        )
        
        return class_embeddings

    def create_enhanced_class_embedding(self, class_labels: torch.Tensor, pose_sequence: torch.Tensor, embed_dim: int = 512) -> torch.Tensor:
        """创建增强的类别embedding"""
        num_classes = 4
        num_frames = len(class_labels)
        
        direction_vectors = torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
        ], dtype=torch.float32)
        
        one_hot = torch.zeros(num_frames, num_classes)
        one_hot.scatter_(1, class_labels.unsqueeze(1), 1)
        
        base_embeddings = one_hot @ direction_vectors
        
        frame_types = pose_sequence[:, -1]
        frame_type_embeddings = torch.zeros(num_frames, 2)
        frame_type_embeddings[:, 0] = (frame_types == 0).float()
        frame_type_embeddings[:, 1] = (frame_types == 1).float()
        
        translations = pose_sequence[:, :3]
        rotations = pose_sequence[:, 3:7]
        
        combined_features = torch.cat([
            base_embeddings,
            frame_type_embeddings,
            translations,
            rotations,
        ], dim=1)
        
        if embed_dim > 13:
            expand_matrix = torch.randn(13, embed_dim) * 0.1
            expand_matrix[:13, :13] = torch.eye(13)
            embeddings = combined_features @ expand_matrix
        else:
            embeddings = combined_features[:, :embed_dim]
        
        return embeddings

    def prepare_framepack_inputs(self, full_latents, segment_info):
        """🔧 准备FramePack风格的多尺度输入"""
        if len(full_latents.shape) == 4:
            full_latents = full_latents.unsqueeze(0)
            B, C, T, H, W = full_latents.shape
        else:
            B, C, T, H, W = full_latents.shape
        
        # 主要latents（用于去噪预测）
        latent_indices = segment_info['latent_indices']
        main_latents = full_latents[:, :, latent_indices, :, :]
        
        # 🔧 1x条件帧（起始帧 + 最后1帧）
        clean_latent_indices = segment_info['clean_latent_indices']
        clean_latents = full_latents[:, :, clean_latent_indices, :, :]
        
        # 🔧 4x条件帧 - 总是16帧，直接用真实索引 + 0填充
        clean_latent_4x_indices = segment_info['clean_latent_4x_indices']
        
        clean_latents_4x = torch.zeros(B, C, 16, H, W, dtype=full_latents.dtype)
        clean_latent_4x_indices_final = torch.full((16,), -1, dtype=torch.long)
        
        if len(clean_latent_4x_indices) > 0:
            actual_4x_frames = len(clean_latent_4x_indices)
            start_pos = max(0, 16 - actual_4x_frames)
            end_pos = 16
            actual_start = max(0, actual_4x_frames - 16)
            
            clean_latents_4x[:, :, start_pos:end_pos, :, :] = full_latents[:, :, clean_latent_4x_indices[actual_start:], :, :]
            clean_latent_4x_indices_final[start_pos:end_pos] = clean_latent_4x_indices[actual_start:]
        
        # 🔧 2x条件帧 - 总是2帧，直接用真实索引 + 0填充
        clean_latent_2x_indices = segment_info['clean_latent_2x_indices']
        
        clean_latents_2x = torch.zeros(B, C, 2, H, W, dtype=full_latents.dtype)
        clean_latent_2x_indices_final = torch.full((2,), -1, dtype=torch.long)
        
        if len(clean_latent_2x_indices) > 0:
            actual_2x_frames = len(clean_latent_2x_indices)
            start_pos = max(0, 2 - actual_2x_frames)
            end_pos = 2
            actual_start = max(0, actual_2x_frames - 2)
            
            clean_latents_2x[:, :, start_pos:end_pos, :, :] = full_latents[:, :, clean_latent_2x_indices[actual_start:], :, :]
            clean_latent_2x_indices_final[start_pos:end_pos] = clean_latent_2x_indices[actual_start:]
        
        # 移除batch维度
        if B == 1:
            main_latents = main_latents.squeeze(0)
            clean_latents = clean_latents.squeeze(0)
            clean_latents_2x = clean_latents_2x.squeeze(0)
            clean_latents_4x = clean_latents_4x.squeeze(0)
        
        return {
            'latents': main_latents,
            'clean_latents': clean_latents,
            'clean_latents_2x': clean_latents_2x,
            'clean_latents_4x': clean_latents_4x,
            'latent_indices': segment_info['latent_indices'],
            'clean_latent_indices': segment_info['clean_latent_indices'],
            'clean_latent_2x_indices': clean_latent_2x_indices_final,
            'clean_latent_4x_indices': clean_latent_4x_indices_final,
        }

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
                
                full_latents = encoded_data['latents']  # [C, T, H, W]
                expected_latent_frames = scene_info['total_frames'] // self.time_compression_ratio
                actual_latent_frames = full_latents.shape[1]
                
                if abs(actual_latent_frames - expected_latent_frames) > 2:
                    print(f"⚠️  Latent帧数不匹配，跳过此样本")
                    continue
                
                # 🔧 使用FramePack风格的段落选择
                segment_info = self.select_dynamic_segment_framepack(scene_info)
                if segment_info is None:
                    continue
                
                # 🔧 创建pose embeddings
                pose_embeddings = self.create_pose_embeddings(scene_info, segment_info)
                if pose_embeddings is None:
                    continue
                
                # 🔧 准备FramePack风格的多尺度输入
                framepack_inputs = self.prepare_framepack_inputs(full_latents, segment_info)
                
                n = segment_info["condition_frames"]
                m = segment_info['target_frames']
                
                # 🔧 添加mask到pose embeddings
                mask = torch.zeros(n+m, dtype=torch.float32)
                mask[:n] = 1.0
                mask = mask.view(-1, 1)
                
                camera_with_mask = torch.cat([pose_embeddings, mask], dim=1)
                
                result = {
                    # 🔧 FramePack风格的多尺度输入
                    "latents": framepack_inputs['latents'],
                    "clean_latents": framepack_inputs['clean_latents'],
                    "clean_latents_2x": framepack_inputs['clean_latents_2x'],
                    "clean_latents_4x": framepack_inputs['clean_latents_4x'],
                    "latent_indices": framepack_inputs['latent_indices'],
                    "clean_latent_indices": framepack_inputs['clean_latent_indices'],
                    "clean_latent_2x_indices": framepack_inputs['clean_latent_2x_indices'],
                    "clean_latent_4x_indices": framepack_inputs['clean_latent_4x_indices'],
                    
                    # Camera数据
                    "camera": camera_with_mask,
                    
                    "prompt_emb": encoded_data["prompt_emb"],
                    "image_emb": encoded_data.get("image_emb", {}),
                    "condition_frames": n,
                    "target_frames": m,
                    "scene_name": os.path.basename(scene_dir),
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

def replace_dit_model_in_manager():
    """在模型加载前替换DiT模型类"""
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
        replace_dit_model_in_manager()  # 🔧 在这里调用
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        if os.path.isfile(dit_path):
            model_manager.load_models([dit_path])
        else:
            dit_path = dit_path.split(",")
            model_manager.load_models([dit_path])
        model_manager.load_models(["models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"])
        
        self.pipe = WanVideoReCamMasterPipeline.from_model_manager(model_manager)
        self.pipe.scheduler.set_timesteps(1000, training=True)

        # 🔧 添加FramePack的clean_x_embedder
        self.add_framepack_components()

        # 添加相机编码器
        dim = self.pipe.dit.blocks[0].self_attn.q.weight.shape[0]
        for block in self.pipe.dit.blocks:
            block.cam_encoder = nn.Linear(513, dim)  # 512 + 1 for mask
            block.projector = nn.Linear(dim, dim)
            block.cam_encoder.weight.data.zero_()
            block.cam_encoder.bias.data.zero_()
            block.projector.weight = nn.Parameter(torch.eye(dim))
            block.projector.bias = nn.Parameter(torch.zeros(dim))
        
        if resume_ckpt_path is not None:
            state_dict = torch.load(resume_ckpt_path, map_location="cpu")
            self.pipe.dit.load_state_dict(state_dict, strict=True)
            print('load checkpoint:', resume_ckpt_path)

        self.freeze_parameters()
        
        # 只训练相机相关和注意力模块以及FramePack相关组件
        for name, module in self.pipe.denoising_model().named_modules():
            if any(keyword in name for keyword in ["cam_encoder", "projector", "self_attn", "clean_x_embedder"]):
                for param in module.parameters():
                    param.requires_grad = True
        
        self.learning_rate = learning_rate
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        
        # 创建可视化目录
        self.vis_dir = "nus/visualizations_dynamic_framepack"
        os.makedirs(self.vis_dir, exist_ok=True)

    def add_framepack_components(self):
        """🔧 添加FramePack相关组件"""
        if not hasattr(self.pipe.dit, 'clean_x_embedder'):
            inner_dim = self.pipe.dit.blocks[0].self_attn.q.weight.shape[0]
            
            class CleanXEmbedder(nn.Module):
                def __init__(self, inner_dim):
                    super().__init__()
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
            
            self.pipe.dit.clean_x_embedder = CleanXEmbedder(inner_dim)
            print("✅ 添加了FramePack的clean_x_embedder组件")
        
    def freeze_parameters(self):
        self.pipe.requires_grad_(False)
        self.pipe.eval()
        self.pipe.denoising_model().train()

    def training_step(self, batch, batch_idx):
        """🔧 使用FramePack风格的训练步骤"""
        condition_frames = batch["condition_frames"][0].item()
        target_frames = batch["target_frames"][0].item()
        
        original_condition_frames = batch.get("original_condition_frames", [condition_frames * 4])[0]
        original_target_frames = batch.get("original_target_frames", [target_frames * 4])[0]

        scene_name = batch.get("scene_name", ["unknown"])[0]
        
        # 🔧 准备FramePack风格的输入
        latents = batch["latents"].to(self.device)
        if len(latents.shape) == 4:
            latents = latents.unsqueeze(0)
        
        clean_latents = batch["clean_latents"].to(self.device) if batch["clean_latents"].numel() > 0 else None
        if clean_latents is not None and len(clean_latents.shape) == 4:
            clean_latents = clean_latents.unsqueeze(0)
        
        clean_latents_2x = batch["clean_latents_2x"].to(self.device) if batch["clean_latents_2x"].numel() > 0 else None
        if clean_latents_2x is not None and len(clean_latents_2x.shape) == 4:
            clean_latents_2x = clean_latents_2x.unsqueeze(0)
        
        clean_latents_4x = batch["clean_latents_4x"].to(self.device) if batch["clean_latents_4x"].numel() > 0 else None
        if clean_latents_4x is not None and len(clean_latents_4x.shape) == 4:
            clean_latents_4x = clean_latents_4x.unsqueeze(0)
        
        # 索引
        latent_indices = batch["latent_indices"].to(self.device)
        clean_latent_indices = batch["clean_latent_indices"].to(self.device) if batch["clean_latent_indices"].numel() > 0 else None
        clean_latent_2x_indices = batch["clean_latent_2x_indices"].to(self.device) if batch["clean_latent_2x_indices"].numel() > 0 else None
        clean_latent_4x_indices = batch["clean_latent_4x_indices"].to(self.device) if batch["clean_latent_4x_indices"].numel() > 0 else None
        
        # Camera embedding
        cam_emb = batch["camera"].to(self.device)
        camera_dropout_prob = 0.1
        if random.random() < camera_dropout_prob:
            cam_emb = torch.zeros_like(cam_emb)
            print("应用camera dropout for CFG training")
        
        prompt_emb = batch["prompt_emb"]
        prompt_emb["context"] = prompt_emb["context"][0].to(self.device)
        image_emb = batch["image_emb"]

        if "clip_feature" in image_emb:
            image_emb["clip_feature"] = image_emb["clip_feature"][0].to(self.device)
        if "y" in image_emb:
            image_emb["y"] = image_emb["y"][0].to(self.device)

        # Loss计算
        self.pipe.device = self.device
        noise = torch.randn_like(latents)
        timestep_id = torch.randint(0, self.pipe.scheduler.num_train_timesteps, (1,))
        timestep = self.pipe.scheduler.timesteps[timestep_id].to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
        
        # 🔧 FramePack风格的噪声处理
        noisy_condition_latents = None
        if clean_latents is not None:
            noisy_condition_latents = copy.deepcopy(clean_latents)
            is_add_noise = random.random()
            if is_add_noise > 0.2:
                noise_cond = torch.randn_like(clean_latents)
                timestep_id_cond = torch.randint(0, self.pipe.scheduler.num_train_timesteps//4*3, (1,))
                timestep_cond = self.pipe.scheduler.timesteps[timestep_id_cond].to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
                noisy_condition_latents = self.pipe.scheduler.add_noise(clean_latents, noise_cond, timestep_cond)

        extra_input = self.pipe.prepare_extra_input(latents)
        origin_latents = copy.deepcopy(latents)
        noisy_latents = self.pipe.scheduler.add_noise(latents, noise, timestep)
        
        training_target = self.pipe.scheduler.training_target(latents, noise, timestep)
        
        # 🔧 使用FramePack风格的forward调用
        noise_pred = self.pipe.denoising_model()(
            noisy_latents, 
            timestep=timestep, 
            cam_emb=cam_emb,
            # FramePack风格的条件输入
            latent_indices=latent_indices,
            clean_latents=noisy_condition_latents if noisy_condition_latents is not None else clean_latents,
            clean_latent_indices=clean_latent_indices,
            clean_latents_2x=clean_latents_2x,
            clean_latent_2x_indices=clean_latent_2x_indices,
            clean_latents_4x=clean_latents_4x,
            clean_latent_4x_indices=clean_latent_4x_indices,
            **prompt_emb, 
            **extra_input, 
            **image_emb,
            use_gradient_checkpointing=self.use_gradient_checkpointing,
            use_gradient_checkpointing_offload=self.use_gradient_checkpointing_offload
        )
        
        # 计算loss
        loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
        loss = loss * self.pipe.scheduler.training_weight(timestep)
        print('--------loss------------:', loss)

        # 记录信息
        wandb.log({
            "train_loss": loss.item(),
            "timestep": timestep.item(),
            "condition_frames_compressed": condition_frames,
            "target_frames_compressed": target_frames,
            "condition_frames_original": original_condition_frames,
            "target_frames_original": original_target_frames,
            "has_clean_latents": clean_latents is not None,
            "has_clean_latents_2x": clean_latents_2x is not None,
            "has_clean_latents_4x": clean_latents_4x is not None,
            "total_frames_compressed": target_frames,
            "total_frames_original": original_target_frames,
            "scene_name": scene_name,
            "global_step": self.global_step
        })

        return loss

    def configure_optimizers(self):
        trainable_modules = filter(lambda p: p.requires_grad, self.pipe.denoising_model().parameters())
        optimizer = torch.optim.AdamW(trainable_modules, lr=self.learning_rate)
        return optimizer
    
    def on_save_checkpoint(self, checkpoint):
        checkpoint_dir = "/home/zhuyixuan05/ReCamMaster/nus_dynamic_framepack"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        current_step = self.global_step
        checkpoint.clear()
        
        state_dict = self.pipe.denoising_model().state_dict()
        torch.save(state_dict, os.path.join(checkpoint_dir, f"step{current_step}_framepack.ckpt"))
        print(f"Saved FramePack model checkpoint: step{current_step}_framepack.ckpt")

def train_dynamic(args):
    """训练支持FramePack机制的动态历史长度模型"""
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
        project="nuscenes-dynamic-framepack-recam",
        name=f"framepack-{args.min_condition_frames}-{args.max_condition_frames}",
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
    
    parser = argparse.ArgumentParser(description="Train FramePack Dynamic ReCamMaster for NuScenes")
    parser.add_argument("--dataset_path", type=str, default="/share_zhuyixuan05/zhuyixuan05/nuscenes_video_generation_dynamic")
    parser.add_argument("--dit_path", type=str, default="models/Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors")
    parser.add_argument("--output_path", type=str, default="./")
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--steps_per_epoch", type=int, default=3000)
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--min_condition_frames", type=int, default=8, help="最小条件帧数")
    parser.add_argument("--max_condition_frames", type=int, default=120, help="最大条件帧数")
    parser.add_argument("--target_frames", type=int, default=32, help="目标帧数")
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--training_strategy", type=str, default="deepspeed_stage_1")
    parser.add_argument("--use_gradient_checkpointing", action="store_true")
    parser.add_argument("--use_gradient_checkpointing_offload", action="store_true")
    parser.add_argument("--resume_ckpt_path", type=str, default=None)
    
    args = parser.parse_args()
    
    print("🔧 使用FramePack风格训练NuScenes数据集:")
    print(f"  - 支持多尺度下采样(1x/2x/4x)")
    print(f"  - 使用WanModelFuture模型")
    print(f"  - 数据集路径: {args.dataset_path}")
    
    train_dynamic(args)