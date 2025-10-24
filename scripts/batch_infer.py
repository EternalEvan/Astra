import os
import subprocess
import argparse
from pathlib import Path
import glob

def find_video_files(videos_dir):
    """查找视频目录下的所有视频文件"""
    video_extensions = ['.mp4']
    video_files = []
    
    for ext in video_extensions:
        pattern = os.path.join(videos_dir, f"*{ext}")
        video_files.extend(glob.glob(pattern))
    
    return sorted(video_files)

def run_inference(condition_video, direction, dit_path, output_dir):
    """运行单个推理任务"""
    # 构建输出文件名
    input_filename = os.path.basename(condition_video)
    name_parts = os.path.splitext(input_filename)
    output_filename = f"{name_parts[0]}_{direction}{name_parts[1]}"
    output_path = os.path.join(output_dir, output_filename)
    
    # 构建推理命令
    cmd = [
        "python", "infer_nus.py",
        "--condition_video", condition_video,
        "--direction", direction,
        "--dit_path", dit_path,
        "--output_path", output_path,
    ]
    
    print(f"🎬 生成 {direction} 方向视频: {input_filename} -> {output_filename}")
    print(f"   命令: {' '.join(cmd)}")
    
    try:
        # 运行推理
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"✅ 成功生成: {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 生成失败: {e}")
        print(f"   错误输出: {e.stderr}")
        return False

def batch_inference(args):
    """批量推理主函数"""
    videos_dir = args.videos_dir
    output_dir = args.output_dir
    directions = args.directions
    dit_path = args.dit_path
    
    # 检查输入目录
    if not os.path.exists(videos_dir):
        print(f"❌ 视频目录不存在: {videos_dir}")
        return
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 输出目录: {output_dir}")
    
    # 查找所有视频文件
    video_files = find_video_files(videos_dir)
    
    if not video_files:
        print(f"❌ 在 {videos_dir} 中没有找到视频文件")
        return
    
    print(f"🎥 找到 {len(video_files)} 个视频文件:")
    for video in video_files:
        print(f"   - {os.path.basename(video)}")
    
    print(f"🎯 将为每个视频生成以下方向: {', '.join(directions)}")
    print(f"📊 总共将生成 {len(video_files) * len(directions)} 个视频")
    
    # 统计信息
    total_tasks = len(video_files) * len(directions)
    completed_tasks = 0
    failed_tasks = 0
    
    # 批量处理
    for i, video_file in enumerate(video_files, 1):
        print(f"\n{'='*60}")
        print(f"处理视频 {i}/{len(video_files)}: {os.path.basename(video_file)}")
        print(f"{'='*60}")
        
        for j, direction in enumerate(directions, 1):
            print(f"\n--- 方向 {j}/{len(directions)}: {direction} ---")
            
            # 检查输出文件是否已存在
            input_filename = os.path.basename(video_file)
            name_parts = os.path.splitext(input_filename)
            output_filename = f"{name_parts[0]}_{direction}{name_parts[1]}"
            output_path = os.path.join(output_dir, output_filename)
            
            if os.path.exists(output_path) and not args.overwrite:
                print(f"⏭️  文件已存在，跳过: {output_filename}")
                completed_tasks += 1
                continue
            
            # 运行推理
            success = run_inference(
                condition_video=video_file,
                direction=direction,
                dit_path=dit_path,
                output_dir=output_dir,
            )
            
            if success:
                completed_tasks += 1
            else:
                failed_tasks += 1
            
            # 显示进度
            current_progress = completed_tasks + failed_tasks
            print(f"📈 进度: {current_progress}/{total_tasks} "
                  f"(成功: {completed_tasks}, 失败: {failed_tasks})")
    
    # 最终统计
    print(f"\n{'='*60}")
    print(f"🎉 批量推理完成!")
    print(f"📊 总任务数: {total_tasks}")
    print(f"✅ 成功: {completed_tasks}")
    print(f"❌ 失败: {failed_tasks}")
    print(f"📁 输出目录: {output_dir}")
    
    if failed_tasks > 0:
        print(f"⚠️  有 {failed_tasks} 个任务失败，请检查日志")
    
    # 列出生成的文件
    if completed_tasks > 0:
        print(f"\n📋 生成的文件:")
        generated_files = glob.glob(os.path.join(output_dir, "*.mp4"))
        for file_path in sorted(generated_files):
            print(f"   - {os.path.basename(file_path)}")

def main():
    parser = argparse.ArgumentParser(description="批量对nus/videos目录下的所有视频生成不同方向的输出")
    
    parser.add_argument("--videos_dir", type=str, default="/home/zhuyixuan05/ReCamMaster/nus/videos/4032",
                       help="输入视频目录路径")
    
    parser.add_argument("--output_dir", type=str, default="nus/infer_results/batch_dynamic_4032_noise",
                       help="输出视频目录路径")
    
    parser.add_argument("--directions", nargs="+", 
                       default=["left_turn", "right_turn"],
                       choices=["forward", "backward", "left_turn", "right_turn"],
                       help="要生成的方向列表")
    
    parser.add_argument("--dit_path", type=str, default="/home/zhuyixuan05/ReCamMaster/nus_dynamic/step15000_dynamic.ckpt",
                       help="训练好的DiT模型路径")
    
    parser.add_argument("--overwrite", action="store_true",
                       help="是否覆盖已存在的输出文件")
    
    parser.add_argument("--dry_run", action="store_true",
                       help="只显示将要执行的任务，不实际运行")
    
    args = parser.parse_args()
    
    if args.dry_run:
        print("🔍 预览模式 - 只显示任务，不执行")
        videos_dir = args.videos_dir
        video_files = find_video_files(videos_dir)
        
        print(f"📁 输入目录: {videos_dir}")
        print(f"📁 输出目录: {args.output_dir}")
        print(f"🎥 找到视频: {len(video_files)} 个")
        print(f"🎯 生成方向: {', '.join(args.directions)}")
        print(f"📊 总任务数: {len(video_files) * len(args.directions)}")
        
        print(f"\n将要执行的任务:")
        for video in video_files:
            for direction in args.directions:
                input_name = os.path.basename(video)
                name_parts = os.path.splitext(input_name)
                output_name = f"{name_parts[0]}_{direction}{name_parts[1]}"
                print(f"   {input_name} -> {output_name} ({direction})")
    else:
        batch_inference(args)

if __name__ == "__main__":
    main()