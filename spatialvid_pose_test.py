
import torch

import os


import os
import json
import torch
import numpy as np

from scipy.spatial.transform import Rotation as R

import pdb

def compute_relative_pose_matrix2(
    pose_a, 
    pose_b 
) -> np.ndarray:
    """
    计算两个相机姿态（7元数组）之间的相对位姿，输出3×4相机矩阵 [R_rel | t_rel]
    
    数学定义：相对位姿描述“从姿态A到姿态B的变换”，即：
    - 若点P在姿态A的相机坐标系下坐标为P_A，在姿态B的相机坐标系下坐标为P_B，
      则满足 P_B = R_rel @ P_A + t_rel（R_rel为相对旋转矩阵，t_rel为相对平移向量）
    
    参数:
    pose_a: 参考姿态A，形状(7,)的数组/list，格式[tx_a, ty_a, tz_a, qx_a, qy_a, qz_a, qw_a]
            - tx_a/ty_a/tz_a: 姿态A在世界坐标系的位置（平移向量）
            - qx_a/qy_a/qz_a/qw_a: 姿态A的朝向（单位四元数，右手坐标系）
    pose_b: 目标姿态B，格式与pose_a完全一致
    
    返回:
    3×4的相对位姿相机矩阵，前3列是3×3相对旋转矩阵R_rel，第4列是3×1相对平移向量t_rel
    
    异常:
    ValueError: 若输入姿态形状/格式不正确，或四元数非单位四元数
    """
    # --------------------------
    # 1. 输入校验（确保数据格式正确）
    # --------------------------
    # 转换为numpy数组并检查形状
    pose_a = np.asarray(pose_a, dtype=np.float64)
    pose_b = np.asarray(pose_b, dtype=np.float64)
    
    if pose_a.shape != (7,):
        raise ValueError(f"姿态A需为(7,)数组，实际输入形状{pose_a.shape}")
    if pose_b.shape != (7,):
        raise ValueError(f"姿态B需为(7,)数组，实际输入形状{pose_b.shape}")
    
    # 分离平移向量和四元数
    t_a = pose_a[:3]  # 姿态A的世界坐标：[tx_a, ty_a, tz_a]
    q_a = pose_a[3:]  # 姿态A的四元数：[qx_a, qy_a, qz_a, qw_a]
    t_b = pose_b[:3]  # 姿态B的世界坐标
    q_b = pose_b[3:]  # 姿态B的四元数
    
    # 检查四元数是否为单位四元数（避免旋转计算错误）
    q_a_norm = np.linalg.norm(q_a)
    q_b_norm = np.linalg.norm(q_b)
    if not np.isclose(q_a_norm, 1.0, atol=1e-4):
        raise ValueError(f"姿态A的四元数非单位四元数，模长为{q_a_norm:.6f}（需接近1.0）")
    if not np.isclose(q_b_norm, 1.0, atol=1e-4):
        raise ValueError(f"姿态B的四元数非单位四元数，模长为{q_b_norm:.6f}（需接近1.0）")
    
    # --------------------------
    # 2. 计算相对旋转矩阵 R_rel
    # --------------------------
    # 将四元数转换为Rotation对象（scipy自动处理右手坐标系）
    rot_a = R.from_quat(q_a)  # 姿态A的旋转矩阵（世界→A相机的旋转）
    rot_b = R.from_quat(q_b)  # 姿态B的旋转矩阵（世界→B相机的旋转）
    
    # 相对旋转 = 姿态B的旋转 × 姿态A旋转的逆（单位旋转矩阵的逆=转置）
    # 数学逻辑：R_rel 描述“A相机坐标系→B相机坐标系”的旋转
    rot_rel = rot_b * rot_a.inv()
    R_rel = rot_rel.as_matrix()  # 转换为3×3矩阵， dtype=np.float64
    
    # --------------------------
    # 3. 计算相对平移向量 t_rel
    # --------------------------
    # 数学推导：t_rel = R_rel @ (-rot_a.inv() @ t_a) + (rot_b.inv() @ t_b)
    # 简化后：t_rel = rot_a.inv().as_matrix().T @ (t_b - t_a)
    # 物理意义：在A相机坐标系下，B相机相对于A相机的位置
    R_a_T = rot_a.inv().as_matrix().T  # 姿态A旋转矩阵的逆=转置（单位矩阵性质）
    t_rel = R_a_T @ (t_b - t_a)  # 3×1相对平移向量
    
    # --------------------------
    # 4. 组合为3×4相机矩阵
    # --------------------------
    # 拼接旋转矩阵（3×3）和平移向量（3×1），形成3×4矩阵
    relative_cam_matrix = np.hstack([R_rel, t_rel.reshape(3, 1)])
    
    return relative_cam_matrix


def compute_relative_pose_matrix(pose1, pose2):
    """
    计算相邻两帧的相对位姿，返回3×4的相机矩阵 [R_rel | t_rel]
    
    参数:
    pose1: 第i帧的相机位姿，形状为(7,)的数组 [tx1, ty1, tz1, qx1, qy1, qz1, qw1]
    pose2: 第i+1帧的相机位姿，形状为(7,)的数组 [tx2, ty2, tz2, qx2, qy2, qz2, qw2]
    
    返回:
    relative_matrix: 3×4的相对位姿矩阵，前3列是旋转矩阵R_rel，第4列是平移向量t_rel
    """
    # 分离平移向量和四元数
    t1 = pose1[:3]  # 第i帧平移 [tx1, ty1, tz1]
    q1 = pose1[3:]  # 第i帧四元数 [qx1, qy1, qz1, qw1]
    t2 = pose2[:3]  # 第i+1帧平移
    q2 = pose2[3:]  # 第i+1帧四元数
    
    # 1. 计算相对旋转矩阵 R_rel
    rot1 = R.from_quat(q1)  # 第i帧旋转
    rot2 = R.from_quat(q2)  # 第i+1帧旋转
    rot_rel = rot2 * rot1.inv()  # 相对旋转 = 后一帧旋转 × 前一帧旋转的逆
    R_rel = rot_rel.as_matrix()  # 转换为3×3矩阵
    
    # 2. 计算相对平移向量 t_rel
    R1_T = rot1.as_matrix().T  # 前一帧旋转矩阵的转置（等价于逆）
    t_rel = R1_T @ (t2 - t1)   # 相对平移 = R1^T × (t2 - t1)
    
    # 3. 组合为3×4矩阵 [R_rel | t_rel]
    relative_matrix = np.hstack([R_rel, t_rel.reshape(3, 1)])
    
    return relative_matrix

encoded_data = torch.load(
                    os.path.join('/share_zhuyixuan05/zhuyixuan05/spatialvid/fdb39216-0d15-5f0f-a78f-c599913a4a2e_0000600_0000900', "encoded_video.pth"),
                    weights_only=False,
                    map_location="cpu"
                )

cam_data_ori = np.load('./poses.npy')

cam_data_seq_ori = cam_data_ori
print(cam_data_seq_ori.shape)
print('---------------------------')
cam_data = encoded_data['cam_emb']

cam_data_seq = cam_data_seq_ori # 
cam_data_seq_inter = cam_data['extrinsic']
print(cam_data_seq_inter.shape)
keyframe_original_idx = list(range(10))

relative_cams = []

for idx in keyframe_original_idx:
    cam_prev = cam_data_seq[idx]
    cam_next = cam_data_seq[idx+1]

    relative_cam = compute_relative_pose_matrix2(cam_prev,cam_next)

    relative_cams.append(torch.as_tensor(relative_cam[:3,:]))
relative_cam = compute_relative_pose_matrix2(cam_data_seq_inter[0],cam_data_seq_inter[-1])

relative_cams.append(torch.as_tensor(relative_cam[:3,:]))

print(relative_cams[-1])

