import os
import json
import numpy as np
from nuscenes.nuscenes import NuScenes
import multiprocessing as mp
from tqdm import tqdm
import cv2
from PIL import Image

# Configuration
VERSION = 'v1.0-trainval'
DATA_ROOT = '/share_zhuyixuan05/public_datasets/nuscenes/nuscenes-download/data'
OUTPUT_DIR = '/share_zhuyixuan05/zhuyixuan05/nuscenes_video_generation_dynamic'
NUM_PROCESSES = 30
PROCESSED_SCENES_FILE = os.path.join(OUTPUT_DIR, 'processed_scenes_dynamic.txt')
CAMERA_CHANNELS = ['CAM_FRONT']

def calculate_relative_pose(pose_current, pose_reference):
    """计算相对于参考pose的相对位置和旋转"""
    trans_ref = np.array(pose_reference['translation'])
    trans_cur = np.array(pose_current['translation'])
    
    # 计算相对位置
    relative_translation = trans_cur - trans_ref
    
    relative_pose = {
        'relative_translation': relative_translation.tolist(),
        'current_rotation': pose_current['rotation'],
        'reference_rotation': pose_reference['rotation'],
        'timestamp': pose_current['timestamp']
    }
    
    return relative_pose

def extract_full_scene_with_keyframes(nusc, scene_token, scene_name, output_dir, channel):
    """提取完整场景并记录关键帧位置"""
    scene_record = nusc.get('scene', scene_token)
    current_sample_token = scene_record['first_sample_token']
    
    # 收集所有sample_data tokens、ego_poses和关键帧标记
    all_sd_tokens = []
    all_ego_poses = []
    keyframe_indices = []  # 记录哪些帧是关键帧
    frame_index = 0
    
    while current_sample_token:
        sample_record = nusc.get('sample', current_sample_token)
        
        if channel in sample_record['data']:
            current_sd_token = sample_record['data'][channel]
            
            # 从keyframe开始，收集所有sample_data
            while current_sd_token:
                sd_record = nusc.get('sample_data', current_sd_token)
                all_sd_tokens.append(current_sd_token)
                
                # 记录ego_pose和关键帧位置
                if sd_record['is_key_frame']:
                    ego_pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])
                    all_ego_poses.append(ego_pose_record)
                    keyframe_indices.append(frame_index)
                else:
                    all_ego_poses.append(None)
                
                frame_index += 1
                current_sd_token = sd_record['next'] if sd_record['next'] != '' else None
            
            break
        
        current_sample_token = sample_record['next'] if sample_record['next'] != '' else None
    
    # 检查是否有足够的帧数和关键帧
    total_frames = len(all_sd_tokens)
    num_keyframes = len(keyframe_indices)
    
    if total_frames < 30 or num_keyframes < 3:  # 至少需要30帧和3个关键帧
        print(f"Scene {scene_name}: Insufficient frames ({total_frames}) or keyframes ({num_keyframes}), skipping...")
        return 0
    
    # 创建场景目录
    scene_dir = os.path.join(output_dir, 'scenes', f"{scene_name}_{channel}")
    os.makedirs(scene_dir, exist_ok=True)
    
    # 渲染完整视频
    video_path = os.path.join(scene_dir, 'full_video.mp4')
    success = render_full_video(nusc, all_sd_tokens, video_path)
    
    if not success:
        print(f"Failed to render video for {scene_name}")
        return 0
    
    # 处理关键帧的poses
    keyframe_poses = []
    valid_keyframes = []
    
    for i, frame_idx in enumerate(keyframe_indices):
        pose = all_ego_poses[frame_idx]
        if pose is not None:
            keyframe_poses.append(pose)
            valid_keyframes.append(frame_idx)
    
    # 保存完整的场景信息
    scene_info = {
        'scene_name': scene_name,
        'channel': channel,
        'total_frames': total_frames,
        'keyframe_indices': valid_keyframes,
        'keyframe_poses': keyframe_poses,
        'sample_data_tokens': all_sd_tokens,
        'video_path': 'full_video.mp4'
    }
    
    with open(os.path.join(scene_dir, 'scene_info.json'), 'w') as f:
        json.dump(scene_info, f, indent=2)
    
    print(f"Processed scene {scene_name}: {total_frames} frames, {len(valid_keyframes)} keyframes")
    return 1

def render_full_video(nusc, sd_tokens, output_path):
    """渲染完整视频序列"""
    if not sd_tokens:
        return False
    
    try:
        # 获取第一帧来确定视频尺寸
        first_sd = nusc.get('sample_data', sd_tokens[0])
        first_image_path = os.path.join(nusc.dataroot, first_sd['filename'])
        first_image = Image.open(first_image_path)
        width, height = first_image.size
        
        # 设置视频编码器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 10.0, (width, height))
        
        for sd_token in sd_tokens:
            sd_record = nusc.get('sample_data', sd_token)
            image_path = os.path.join(nusc.dataroot, sd_record['filename'])
            
            if os.path.exists(image_path):
                image = cv2.imread(image_path)
                if image is not None:
                    out.write(image)
        
        out.release()
        return True
        
    except Exception as e:
        print(f"Error rendering video to {output_path}: {str(e)}")
        return False

def process_scene_dynamic(args):
    """处理单个场景，生成动态长度数据"""
    scene_token, channels = args
    nusc = NuScenes(version=VERSION, dataroot=DATA_ROOT, verbose=False)
    scene_record = nusc.get('scene', scene_token)
    scene_name = scene_record['name']
    
    success_channels = []
    total_scenes = 0
    
    try:
        for channel in channels:
            # 检查是否已经处理过
            scene_dir = os.path.join(OUTPUT_DIR, 'scenes', f"{scene_name}_{channel}")
            if os.path.exists(os.path.join(scene_dir, 'scene_info.json')):
                print(f"Scene {scene_name} {channel} already processed, skipping...")
                success_channels.append(channel)
                continue
            
            # 提取完整场景
            scenes_count = extract_full_scene_with_keyframes(nusc, scene_token, scene_name, OUTPUT_DIR, channel)
            
            if scenes_count > 0:
                success_channels.append(channel)
                total_scenes += scenes_count
            else:
                print(f"Failed to process scene {scene_name} {channel}")
                
    except Exception as e:
        print(f"Error processing {scene_name} ({scene_token}): {str(e)}")
    
    return scene_token, success_channels, total_scenes

def get_processed_scenes():
    """读取处理记录"""
    processed = {}
    if os.path.exists(PROCESSED_SCENES_FILE):
        with open(PROCESSED_SCENES_FILE, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or ':' not in line:
                    continue
                token, channels_str = line.split(':', 1)
                processed[token] = set(channels_str.split(','))
    return processed

def main():
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'scenes'), exist_ok=True)
    
    # 初始化数据集
    nusc = NuScenes(version=VERSION, dataroot=DATA_ROOT, verbose=True)
    all_scenes = {s['token']: s for s in nusc.scene}
    
    # 加载处理记录
    processed = get_processed_scenes()
    
    # 生成任务列表
    tasks = []
    for scene_token in all_scenes:
        processed_channels = processed.get(scene_token, set())
        remaining = [ch for ch in CAMERA_CHANNELS if ch not in processed_channels]
        if remaining:
            tasks.append((scene_token, remaining))
    
    print(f"Total scenes: {len(all_scenes)}")
    print(f"Pending tasks: {len(tasks)}")
    print("Processing full scenes with keyframe tracking...")
    
    if not tasks:
        print("All scenes already processed!")
        return
    
    # 创建进程池
    total_scenes_created = 0
    with mp.Pool(processes=NUM_PROCESSES) as pool:
        results = []
        for res in tqdm(pool.imap_unordered(process_scene_dynamic, tasks),
                         total=len(tasks),
                         desc="Processing Scenes"):
            results.append(res)
        
        # 更新处理记录
        updated = get_processed_scenes()
        for scene_token, success_chs, scenes_count in results:
            if scene_token not in updated:
                updated[scene_token] = set()
            updated[scene_token].update(success_chs)
            total_scenes_created += scenes_count
        
        # 写入最终记录
        with open(PROCESSED_SCENES_FILE, 'w') as f:
            for token, chs in updated.items():
                f.write(f"{token}:{','.join(sorted(chs))}\n")
    
    print(f"\nProcessing completed!")
    print(f"Total scenes created: {total_scenes_created}")
    print(f"Output directory: {OUTPUT_DIR}")

if __name__ == '__main__':
    main()