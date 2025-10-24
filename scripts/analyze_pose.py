import os
import json
import matplotlib.pyplot as plt
import numpy as np
from pose_classifier import PoseClassifier
import torch
from collections import defaultdict

def analyze_turning_patterns_detailed(dataset_path, num_samples=50):
    """详细分析转弯模式，基于相对于reference的pose变化"""
    classifier = PoseClassifier()
    samples_path = os.path.join(dataset_path, "samples")
    
    all_analyses = []
    sample_count = 0
    
    # 用于统计每个类别的样本
    class_samples = defaultdict(list)
    
    print("=== 开始分析样本（基于相对于reference的变化）===")
    
    for item in sorted(os.listdir(samples_path)):  # 排序以便有序输出
        if sample_count >= num_samples:
            break
            
        sample_dir = os.path.join(samples_path, item)
        if os.path.isdir(sample_dir):
            poses_path = os.path.join(sample_dir, "poses.json")
            if os.path.exists(poses_path):
                try:
                    with open(poses_path, 'r') as f:
                        poses_data = json.load(f)
                    
                    target_relative_poses = poses_data['target_relative_poses']
                    
                    if len(target_relative_poses) > 0:
                        # 🔧 创建相对pose向量（已经是相对于reference的）
                        pose_vecs = []
                        for pose_data in target_relative_poses:
                            # 相对位移（已经是相对于reference计算的）
                            translation = torch.tensor(pose_data['relative_translation'], dtype=torch.float32)
                            
                            # 🔧 相对旋转（需要从current和reference计算）
                            current_rotation = torch.tensor(pose_data['current_rotation'], dtype=torch.float32)
                            reference_rotation = torch.tensor(pose_data['reference_rotation'], dtype=torch.float32)
                            
                            # 计算相对旋转：q_relative = q_ref^-1 * q_current
                            relative_rotation = calculate_relative_rotation(current_rotation, reference_rotation)
                            
                            # 组合为7D向量：[relative_translation, relative_rotation]
                            pose_vec = torch.cat([translation, relative_rotation], dim=0)
                            pose_vecs.append(pose_vec)
                        
                        if pose_vecs:
                            pose_sequence = torch.stack(pose_vecs, dim=0)
                            
                            # 🔧 使用新的分析方法
                            analysis = classifier.analyze_pose_sequence(pose_sequence)
                            analysis['sample_name'] = item
                            all_analyses.append(analysis)
                            
                            # 🔧 详细输出每个样本的分类信息
                            print(f"\n--- 样本 {sample_count + 1}: {item} ---")
                            print(f"总帧数: {analysis['total_frames']}")
                            print(f"总距离: {analysis['total_distance']:.4f}")
                            
                            # 分类分布
                            class_dist = analysis['class_distribution']
                            print(f"分类分布:")
                            for class_name, count in class_dist.items():
                                percentage = count / analysis['total_frames'] * 100
                                print(f"  {class_name}: {count} 帧 ({percentage:.1f}%)")
                            
                            # 🔧 调试前几个pose的分类过程
                            print(f"前3帧的详细分类过程:")
                            for i in range(min(3, len(pose_vecs))):
                                debug_info = classifier.debug_single_pose(
                                    pose_vecs[i][:3], pose_vecs[i][3:7]
                                )
                                print(f"  帧{i}: {debug_info['classification']} "
                                      f"(yaw: {debug_info['yaw_angle_deg']:.2f}°, "
                                      f"forward: {debug_info['forward_movement']:.3f})")
                            
                            # 运动段落
                            print(f"运动段落:")
                            for i, segment in enumerate(analysis['motion_segments']):
                                print(f"  段落{i+1}: {segment['class']} (帧 {segment['start_frame']}-{segment['end_frame']}, 持续 {segment['duration']} 帧)")
                            
                            # 🔧 确定主要运动类型
                            dominant_class = max(class_dist.items(), key=lambda x: x[1])
                            dominant_class_name = dominant_class[0]
                            dominant_percentage = dominant_class[1] / analysis['total_frames'] * 100
                            
                            print(f"主要运动类型: {dominant_class_name} ({dominant_percentage:.1f}%)")
                            
                            # 将样本添加到对应类别
                            class_samples[dominant_class_name].append({
                                'name': item,
                                'percentage': dominant_percentage,
                                'analysis': analysis
                            })
                            
                            sample_count += 1
                            
                except Exception as e:
                    print(f"❌ 处理样本 {item} 时出错: {e}")
    
    print("\n" + "="*60)
    print("=== 按类别分组的样本统计（基于相对于reference的变化）===")
    
    # 🔧 按类别输出样本列表
    for class_name in ['forward', 'backward', 'left_turn', 'right_turn']:
        samples = class_samples[class_name]
        print(f"\n🔸 {class_name.upper()} 类样本 (共 {len(samples)} 个):")
        
        if samples:
            # 按主要类别占比排序
            samples.sort(key=lambda x: x['percentage'], reverse=True)
            
            for i, sample_info in enumerate(samples, 1):
                print(f"  {i:2d}. {sample_info['name']} ({sample_info['percentage']:.1f}%)")
                
                # 显示详细的段落信息
                segments = sample_info['analysis']['motion_segments']
                segment_summary = []
                for seg in segments:
                    if seg['duration'] >= 2:  # 只显示持续时间>=2帧的段落
                        segment_summary.append(f"{seg['class']}({seg['duration']})")
                
                if segment_summary:
                    print(f"      段落: {' -> '.join(segment_summary)}")
        else:
            print("  (无样本)")
    
    # 🔧 统计总体模式
    print(f"\n" + "="*60)
    print("=== 总体统计 ===")
    
    total_forward = sum(a['class_distribution']['forward'] for a in all_analyses)
    total_backward = sum(a['class_distribution']['backward'] for a in all_analyses)
    total_left_turn = sum(a['class_distribution']['left_turn'] for a in all_analyses)
    total_right_turn = sum(a['class_distribution']['right_turn'] for a in all_analyses)
    total_frames = total_forward + total_backward + total_left_turn + total_right_turn
    
    print(f"总样本数: {len(all_analyses)}")
    print(f"总帧数: {total_frames}")
    print(f"Forward: {total_forward} 帧 ({total_forward/total_frames*100:.1f}%)")
    print(f"Backward: {total_backward} 帧 ({total_backward/total_frames*100:.1f}%)")
    print(f"Left Turn: {total_left_turn} 帧 ({total_left_turn/total_frames*100:.1f}%)")
    print(f"Right Turn: {total_right_turn} 帧 ({total_right_turn/total_frames*100:.1f}%)")
    
    # 🔧 样本分布统计
    print(f"\n按主要类型的样本分布:")
    for class_name in ['forward', 'backward', 'left_turn', 'right_turn']:
        count = len(class_samples[class_name])
        percentage = count / len(all_analyses) * 100 if all_analyses else 0
        print(f"  {class_name}: {count} 样本 ({percentage:.1f}%)")
    
    return all_analyses, class_samples

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

if __name__ == "__main__":
    dataset_path = "/share_zhuyixuan05/zhuyixuan05/nuscenes_video_generation_2"
    
    print("开始详细分析pose分类（基于相对于reference的变化）...")
    all_analyses, class_samples = analyze_turning_patterns_detailed(dataset_path, num_samples=4000)
    
    print(f"\n🎉 分析完成! 共处理 {len(all_analyses)} 个样本")