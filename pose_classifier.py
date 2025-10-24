import torch
import numpy as np
from typing import List, Tuple

class PoseClassifier:
    """将pose参数分类为前后左右四个类别，正确使用rotation数据判断转弯"""
    
    def __init__(self):
        # 定义四个方向的类别
        self.FORWARD = 0
        self.BACKWARD = 1
        self.LEFT_TURN = 2
        self.RIGHT_TURN = 3
        
        self.class_names = ['forward', 'backward', 'left_turn', 'right_turn']
        
    def classify_pose_sequence(self, pose_sequence: torch.Tensor) -> torch.Tensor:
        """
        对pose序列进行分类，基于相对于reference的pose变化
        Args:
            pose_sequence: [num_frames, 7] (relative_translation + relative_quaternion)
                          这里的pose都是相对于reference帧的相对变换
        Returns:
            classifications: [num_frames] 类别标签
        """
        # 提取平移部分 [num_frames, 3] 和旋转部分 [num_frames, 4]
        translations = pose_sequence[:, :3]  # 相对于reference的位移
        rotations = pose_sequence[:, 3:7]    # 相对于reference的旋转 [w, x, y, z]
        
        # 分类每一帧 - 都是相对于reference帧的变化
        classifications = []
        for i in range(len(pose_sequence)):
            # 🔧 修改：每一帧都基于相对于reference的变化进行分类
            relative_translation = translations[i]  # 相对于reference的位移
            relative_rotation = rotations[i]         # 相对于reference的旋转
            
            class_label = self._classify_single_pose(relative_translation, relative_rotation)
            classifications.append(class_label)
            
        return torch.tensor(classifications, dtype=torch.long)
    
    def _classify_single_pose(self, relative_translation: torch.Tensor, 
                            relative_rotation: torch.Tensor) -> int:
        """
        对单个pose进行分类，基于相对于reference的变化
        Args:
            relative_translation: [3] 相对于reference的位移变化
            relative_rotation: [4] 相对于reference的旋转四元数 [w, x, y, z]
        """
        # 🔧 关键：从相对旋转四元数提取yaw角度
        yaw_angle = self._quaternion_to_yaw(relative_rotation)
        
        # 🔧 计算前进/后退（主要看x方向的位移）
        forward_movement = -relative_translation[0].item()  # x负方向为前进
        
        # 🔧 设置阈值
        yaw_threshold = 0.05  # 约2.9度，可以调整
        movement_threshold = 0.01  # 位移阈值
        
        # 🔧 优先判断转弯（基于相对于reference的yaw角度）
        if abs(yaw_angle) > yaw_threshold:
            if yaw_angle > 0:
                return self.LEFT_TURN   # 正yaw角度为左转
            else:
                return self.RIGHT_TURN  # 负yaw角度为右转
        
        # 🔧 如果没有明显转弯，判断前进后退（基于相对位移）
        if abs(forward_movement) > movement_threshold:
            if forward_movement > 0:
                return self.FORWARD
            else:
                return self.BACKWARD
        
        # 🔧 如果位移和旋转都很小，判断为前进（静止时的默认状态）
        return self.FORWARD
    
    def _quaternion_to_yaw(self, q: torch.Tensor) -> float:
        """
        从四元数提取yaw角度（绕z轴旋转）
        Args:
            q: [4] 四元数 [w, x, y, z]
        Returns:
            yaw: yaw角度（弧度）
        """
        try:
            # 转换为numpy数组进行计算
            q_np = q.detach().cpu().numpy()
            
            # 🔧 确保四元数是单位四元数
            norm = np.linalg.norm(q_np)
            if norm > 1e-8:
                q_np = q_np / norm
            else:
                # 如果四元数接近零，返回0角度
                return 0.0
            
            w, x, y, z = q_np
            
            # 🔧 计算yaw角度：atan2(2*(w*z + x*y), 1 - 2*(y^2 + z^2))
            yaw = np.arctan2(2.0 * (w*z + x*y), 1.0 - 2.0 * (y*y + z*z))
            
            return float(yaw)
            
        except Exception as e:
            print(f"Error computing yaw from quaternion: {e}")
            return 0.0
    
    def create_class_embedding(self, class_labels: torch.Tensor, embed_dim: int = 512) -> torch.Tensor:
        """
        为类别标签创建embedding
        Args:
            class_labels: [num_frames] 类别标签
            embed_dim: embedding维度
        Returns:
            embeddings: [num_frames, embed_dim]
        """
        num_classes = 4
        num_frames = len(class_labels)
        
        # 🔧 创建更有意义的embedding，不同类别有不同的特征
        # 使用预定义的方向向量
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
        
        # 扩展到目标维度
        if embed_dim > 4:
            # 使用线性变换扩展
            expand_matrix = torch.randn(4, embed_dim) * 0.1
            # 保持方向性
            expand_matrix[:4, :4] = torch.eye(4)
            embeddings = base_embeddings @ expand_matrix
        else:
            embeddings = base_embeddings[:, :embed_dim]
        
        return embeddings
    
    def get_class_name(self, class_id: int) -> str:
        """获取类别名称"""
        return self.class_names[class_id]
    
    def analyze_pose_sequence(self, pose_sequence: torch.Tensor) -> dict:
        """
        分析pose序列，返回详细的统计信息
        Args:
            pose_sequence: [num_frames, 7] (translation + quaternion)
        Returns:
            analysis: 包含统计信息的字典
        """
        classifications = self.classify_pose_sequence(pose_sequence)
        
        # 统计各类别数量
        class_counts = torch.bincount(classifications, minlength=4)
        
        # 计算连续运动段
        motion_segments = []
        if len(classifications) > 0:
            current_class = classifications[0].item()
            segment_start = 0
            
            for i in range(1, len(classifications)):
                if classifications[i].item() != current_class:
                    motion_segments.append({
                        'class': self.get_class_name(current_class),
                        'start_frame': segment_start,
                        'end_frame': i-1,
                        'duration': i - segment_start
                    })
                    current_class = classifications[i].item()
                    segment_start = i
            
            # 添加最后一个段
            motion_segments.append({
                'class': self.get_class_name(current_class),
                'start_frame': segment_start,
                'end_frame': len(classifications)-1,
                'duration': len(classifications) - segment_start
            })
        
        # 计算总体运动信息
        translations = pose_sequence[:, :3]
        if len(translations) > 1:
            # 计算累积距离（相对于reference的总移动距离）
            total_distance = torch.norm(translations[-1] - translations[0])
        else:
            total_distance = torch.tensor(0.0)
        
        analysis = {
            'total_frames': len(pose_sequence),
            'class_distribution': {
                self.get_class_name(i): count.item() 
                for i, count in enumerate(class_counts)
            },
            'motion_segments': motion_segments,
            'total_distance': total_distance.item(),
            'classifications': classifications
        }
        
        return analysis
    
    def debug_single_pose(self, relative_translation: torch.Tensor, 
                         relative_rotation: torch.Tensor) -> dict:
        """
        调试单个pose的分类过程
        Args:
            relative_translation: [3] 相对位移
            relative_rotation: [4] 相对旋转四元数
        Returns:
            debug_info: 调试信息字典
        """
        yaw_angle = self._quaternion_to_yaw(relative_rotation)
        forward_movement = -relative_translation[0].item()
        
        yaw_threshold = 0.05
        movement_threshold = 0.01
        
        classification = self._classify_single_pose(relative_translation, relative_rotation)
        
        debug_info = {
            'relative_translation': relative_translation.tolist(),
            'relative_rotation': relative_rotation.tolist(),
            'yaw_angle_rad': yaw_angle,
            'yaw_angle_deg': np.degrees(yaw_angle),
            'forward_movement': forward_movement,
            'yaw_threshold': yaw_threshold,
            'movement_threshold': movement_threshold,
            'classification': self.get_class_name(classification),
            'classification_id': classification,
            'decision_process': {
                'abs_yaw_exceeds_threshold': abs(yaw_angle) > yaw_threshold,
                'abs_movement_exceeds_threshold': abs(forward_movement) > movement_threshold,
                'yaw_direction': 'left' if yaw_angle > 0 else 'right' if yaw_angle < 0 else 'none',
                'movement_direction': 'forward' if forward_movement > 0 else 'backward' if forward_movement < 0 else 'none'
            }
        }
        
        return debug_info