import os
import random
import subprocess
import time

src_root = "/share_zhuyixuan05/zhuyixuan05/spatialvid"
dst_root = "/share_zhuyixuan05/zhuyixuan05/New_spatialvid_drone_first"
infer_script = "/home/zhuyixuan05/ReCamMaster/infer_origin.py"  # 修改为你的实际路径

while True:
    # 随机选择一个子文件夹
    subdirs = [d for d in os.listdir(src_root) if os.path.isdir(os.path.join(src_root, d))]
    if not subdirs:
        print("没有可用的子文件夹")
        break
    chosen = random.choice(subdirs)
    chosen_dir = os.path.join(src_root, chosen)
    pth_file = os.path.join(chosen_dir, "encoded_video.pth")
    if not os.path.exists(pth_file):
        print(f"{pth_file} 不存在，跳过")
        continue

    # 生成输出文件名
    out_file = os.path.join(dst_root, f"{chosen}.mp4")
    print(f"开始生成: {pth_file} -> {out_file}")

    # 构造命令
    cmd = [
        "python", infer_script,
        "--condition_pth", pth_file,
        "--output_path", out_file,
        "--prompt", "exploring the world",
        "--modality_type", "sekai",
        "--direction", "right",
        "--dit_path", "/share_zhuyixuan05/zhuyixuan05/ICLR2026/framepack_moe/step25000_first.ckpt",
        "--use_gt_prompt"
    ]

    # 仅使用第二张 GPU
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"

    # 执行推理
    subprocess.run(cmd, env=env)