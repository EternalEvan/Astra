import os
import torch
from tqdm import tqdm

def analyze_openx_dataset_frame_counts(dataset_path):
    """分析OpenX数据集中的帧数分布"""
    
    print(f"🔧 分析OpenX数据集: {dataset_path}")
    
    if not os.path.exists(dataset_path):
        print(f"  ⚠️ 路径不存在: {dataset_path}")
        return
    
    episode_dirs = []
    total_episodes = 0
    valid_episodes = 0
    
    # 收集所有episode目录
    for item in os.listdir(dataset_path):
        episode_dir = os.path.join(dataset_path, item)
        if os.path.isdir(episode_dir):
            total_episodes += 1
            encoded_path = os.path.join(episode_dir, "encoded_video.pth")
            if os.path.exists(encoded_path):
                episode_dirs.append(episode_dir)
                valid_episodes += 1
    
    print(f"📊 总episode数: {total_episodes}")
    print(f"📊 有效episode数: {valid_episodes}")
    
    if len(episode_dirs) == 0:
        print("❌ 没有找到有效的episode")
        return
    
    # 统计帧数分布
    frame_counts = []
    less_than_10 = 0
    less_than_8 = 0
    less_than_5 = 0
    error_count = 0
    
    print("🔧 开始分析帧数分布...")
    
    for episode_dir in tqdm(episode_dirs, desc="分析episodes"):
        try:
            encoded_data = torch.load(
                os.path.join(episode_dir, "encoded_video.pth"),
                weights_only=False,
                map_location="cpu"
            )
            
            latents = encoded_data['latents']  # [C, T, H, W]
            frame_count = latents.shape[1]  # T维度
            frame_counts.append(frame_count)
            
            if frame_count < 10:
                less_than_10 += 1
            if frame_count < 8:
                less_than_8 += 1
            if frame_count < 5:
                less_than_5 += 1
                
        except Exception as e:
            error_count += 1
            if error_count <= 5:  # 只打印前5个错误
                print(f"❌ 加载episode {os.path.basename(episode_dir)} 时出错: {e}")
    
    # 统计结果
    total_valid = len(frame_counts)
    print(f"\n📈 帧数分布统计:")
    print(f"  总有效episodes: {total_valid}")
    print(f"  错误episodes: {error_count}")
    print(f"  最小帧数: {min(frame_counts) if frame_counts else 0}")
    print(f"  最大帧数: {max(frame_counts) if frame_counts else 0}")
    print(f"  平均帧数: {sum(frame_counts) / len(frame_counts):.2f}" if frame_counts else 0)
    
    print(f"\n🎯 关键统计:")
    print(f"  帧数 < 5:  {less_than_5:6d} episodes ({less_than_5/total_valid*100:.2f}%)")
    print(f"  帧数 < 8:  {less_than_8:6d} episodes ({less_than_8/total_valid*100:.2f}%)")
    print(f"  帧数 < 10: {less_than_10:6d} episodes ({less_than_10/total_valid*100:.2f}%)")
    print(f"  帧数 >= 10: {total_valid-less_than_10:6d} episodes ({(total_valid-less_than_10)/total_valid*100:.2f}%)")
    
    # 详细分布
    frame_counts.sort()
    print(f"\n📊 详细帧数分布:")
    
    # 按范围统计
    ranges = [
        (1, 4, "1-4帧"),
        (5, 7, "5-7帧"),
        (8, 9, "8-9帧"),
        (10, 19, "10-19帧"),
        (20, 49, "20-49帧"),
        (50, 99, "50-99帧"),
        (100, float('inf'), "100+帧")
    ]
    
    for min_f, max_f, label in ranges:
        count = sum(1 for f in frame_counts if min_f <= f <= max_f)
        percentage = count / total_valid * 100
        print(f"  {label:8s}: {count:6d} episodes ({percentage:5.2f}%)")
    
    # 建议的训练配置
    print(f"\n💡 训练配置建议:")
    time_compression_ratio = 4
    min_condition_compressed = 4 // time_compression_ratio  # 1帧
    target_frames_compressed = 32 // time_compression_ratio  # 8帧
    min_required_compressed = min_condition_compressed + target_frames_compressed  # 9帧
    
    usable_episodes = sum(1 for f in frame_counts if f >= min_required_compressed)
    usable_percentage = usable_episodes / total_valid * 100
    
    print(f"  最小条件帧数(压缩后): {min_condition_compressed}")
    print(f"  目标帧数(压缩后): {target_frames_compressed}")
    print(f"  最小所需帧数(压缩后): {min_required_compressed}")
    print(f"  可用于训练的episodes: {usable_episodes} ({usable_percentage:.2f}%)")
    
    # 保存详细统计到文件
    output_file = os.path.join(dataset_path, "frame_count_analysis.txt")
    with open(output_file, 'w') as f:
        f.write(f"OpenX Dataset Frame Count Analysis\n")
        f.write(f"Dataset Path: {dataset_path}\n")
        f.write(f"Analysis Date: {__import__('datetime').datetime.now()}\n\n")
        
        f.write(f"Total Episodes: {total_episodes}\n")
        f.write(f"Valid Episodes: {total_valid}\n")
        f.write(f"Error Episodes: {error_count}\n\n")
        
        f.write(f"Frame Count Statistics:\n")
        f.write(f"  Min Frames: {min(frame_counts) if frame_counts else 0}\n")
        f.write(f"  Max Frames: {max(frame_counts) if frame_counts else 0}\n")
        f.write(f"  Avg Frames: {sum(frame_counts) / len(frame_counts):.2f}\n\n" if frame_counts else "  Avg Frames: 0\n\n")
        
        f.write(f"Key Statistics:\n")
        f.write(f"  < 5 frames:  {less_than_5} ({less_than_5/total_valid*100:.2f}%)\n")
        f.write(f"  < 8 frames:  {less_than_8} ({less_than_8/total_valid*100:.2f}%)\n")
        f.write(f"  < 10 frames: {less_than_10} ({less_than_10/total_valid*100:.2f}%)\n")
        f.write(f"  >= 10 frames: {total_valid-less_than_10} ({(total_valid-less_than_10)/total_valid*100:.2f}%)\n\n")
        
        f.write(f"Detailed Distribution:\n")
        for min_f, max_f, label in ranges:
            count = sum(1 for f in frame_counts if min_f <= f <= max_f)
            percentage = count / total_valid * 100
            f.write(f"  {label}: {count} ({percentage:.2f}%)\n")
        
        f.write(f"\nTraining Configuration Recommendation:\n")
        f.write(f"  Usable Episodes (>= {min_required_compressed} compressed frames): {usable_episodes} ({usable_percentage:.2f}%)\n")
        
        # 写入所有帧数
        f.write(f"\nAll Frame Counts:\n")
        for i, count in enumerate(frame_counts):
            f.write(f"{count}")
            if (i + 1) % 20 == 0:
                f.write("\n")
            else:
                f.write(", ")
    
    print(f"\n💾 详细统计已保存到: {output_file}")
    
    return {
        'total_valid': total_valid,
        'less_than_10': less_than_10,
        'less_than_8': less_than_8,
        'less_than_5': less_than_5,
        'frame_counts': frame_counts,
        'usable_episodes': usable_episodes
    }

def quick_sample_analysis(dataset_path, sample_size=1000):
    """快速采样分析，用于大数据集的初步估计"""
    
    print(f"🚀 快速采样分析 (样本数: {sample_size})")
    
    episode_dirs = []
    for item in os.listdir(dataset_path):
        episode_dir = os.path.join(dataset_path, item)
        if os.path.isdir(episode_dir):
            encoded_path = os.path.join(episode_dir, "encoded_video.pth")
            if os.path.exists(encoded_path):
                episode_dirs.append(episode_dir)
    
    if len(episode_dirs) == 0:
        print("❌ 没有找到有效的episode")
        return
    
    # 随机采样
    import random
    sample_dirs = random.sample(episode_dirs, min(sample_size, len(episode_dirs)))
    
    frame_counts = []
    less_than_10 = 0
    
    for episode_dir in tqdm(sample_dirs, desc="采样分析"):
        try:
            encoded_data = torch.load(
                os.path.join(episode_dir, "encoded_video.pth"),
                weights_only=False,
                map_location="cpu"
            )
            
            frame_count = encoded_data['latents'].shape[1]
            frame_counts.append(frame_count)
            
            if frame_count < 10:
                less_than_10 += 1
                
        except Exception as e:
            continue
    
    total_sample = len(frame_counts)
    percentage_less_than_10 = less_than_10 / total_sample * 100
    
    print(f"📊 采样结果:")
    print(f"  采样数量: {total_sample}")
    print(f"  < 10帧: {less_than_10} ({percentage_less_than_10:.2f}%)")
    print(f"  >= 10帧: {total_sample - less_than_10} ({100 - percentage_less_than_10:.2f}%)")
    print(f"  平均帧数: {sum(frame_counts) / len(frame_counts):.2f}")
    
    # 估算全数据集
    total_episodes = len(episode_dirs)
    estimated_less_than_10 = int(total_episodes * percentage_less_than_10 / 100)
    
    print(f"\n🔮 全数据集估算:")
    print(f"  总episodes: {total_episodes}")
    print(f"  估算 < 10帧: {estimated_less_than_10} ({percentage_less_than_10:.2f}%)")
    print(f"  估算 >= 10帧: {total_episodes - estimated_less_than_10} ({100 - percentage_less_than_10:.2f}%)")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="分析OpenX数据集的帧数分布")
    parser.add_argument("--dataset_path", type=str, 
                       default="/share_zhuyixuan05/zhuyixuan05/openx-fractal-encoded",
                       help="OpenX编码数据集路径")
    parser.add_argument("--quick", action="store_true", help="快速采样分析模式")
    parser.add_argument("--sample_size", type=int, default=1000, help="快速模式的采样数量")
    
    args = parser.parse_args()
    
    if args.quick:
        quick_sample_analysis(args.dataset_path, args.sample_size)
    else:
        analyze_openx_dataset_frame_counts(args.dataset_path)