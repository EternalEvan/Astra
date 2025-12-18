#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =========================
# User Config (edit below)
# =========================
METADATA_CSV      = "/mnt/data/SpatialVID-HQ/data/train/SpatialVID_HQ_metadata.csv"
POSES_OUT_ROOT    = None     # where to save (T,4,4) npz
OUT_JSON          = "/mnt/preprocess_data/SpatialVID-HQ/manifest25.json"       # final output list

# CSV column names
VIDEO_COL         = "video path"
ANNOTATION_COL    = "annotation path"              # folder with poses.npy
POSES7D_FILE      = "poses.npy"                    # NPZ with (T,7) or ('t','q')
SCENEJSON_FILE    = "caption.json"                 # JSON with SceneSummary & SceneDescription
FPS_COL           = "fps"
FRAMES_COL        = "num frames"

# Filters (set to None to ignore)
FPS_MIN           = 25.0                           # or None
FPS_MAX           = None                           # or None
FRAMES_MIN        = 120                            # or None
FRAMES_MAX        = None                           # or None

# =========================

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd

from tqdm import tqdm


# ---------------------- Quaternion & Pose Utilities ----------------------

def _normalize_quat(q: np.ndarray) -> np.ndarray:
    """Normalize quaternion(s). Expected order (x, y, z, w)."""
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q, axis=-1, keepdims=True) + 1e-12
    return q / n

def _quat_xyzw_to_R(q: np.ndarray) -> np.ndarray:
    """
    Convert quaternion(s) (x,y,z,w) to rotation matrix/matrices.
    q: (..., 4)
    returns: (..., 3, 3)
    """
    q = _normalize_quat(q)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z

    R = np.empty(q.shape[:-1] + (3, 3), dtype=np.float64)
    R[..., 0, 0] = 1 - 2 * (yy + zz)
    R[..., 0, 1] = 2 * (xy - wz)
    R[..., 0, 2] = 2 * (xz + wy)

    R[..., 1, 0] = 2 * (xy + wz)
    R[..., 1, 1] = 1 - 2 * (xx + zz)
    R[..., 1, 2] = 2 * (yz - wx)

    R[..., 2, 0] = 2 * (xz - wy)
    R[..., 2, 1] = 2 * (yz + wx)
    R[..., 2, 2] = 1 - 2 * (xx + yy)
    return R

def poses7d_to_T44(poses7d: np.ndarray) -> np.ndarray:
    """
    poses7d: (T,7) with columns (tx,ty,tz,qx,qy,qz,qw)
    returns: (T,4,4) homogeneous transforms
    """
    poses7d = np.asarray(poses7d, dtype=np.float64)
    if poses7d.ndim != 2 or poses7d.shape[1] != 7:
        raise ValueError(f"Expected poses7d as (T,7), got {poses7d.shape}")
    t = poses7d[:, :3]
    q = poses7d[:, 3:7]  # (x,y,z,w)
    Rm = _quat_xyzw_to_R(q)  # (T,3,3)

    T = np.repeat(np.eye(4, dtype=np.float32)[None, ...], poses7d.shape[0], axis=0)
    T[:, :3, :3] = Rm.astype(np.float32)
    T[:, :3, 3] = t.astype(np.float32)
    
    assert T.shape == (poses7d.shape[0], 4, 4)
    return T

def load_poses_7d_from_npz(path: Path) -> np.ndarray:
    """
    Load (T,7) poses from NPZ.
    """
    return np.load(path, allow_pickle=False)


def _compute_indices_by_stride(n_frames: int, stride: int, start: int = 0) -> List[int]:
    """
    Generate indices with a fixed stride: start, start + stride, ...
    until the indices are < n_frames (no out-of-bounds).
    """
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")
    if start < 0 or start >= n_frames:
        return []
    idxs = list(range(start, n_frames, stride))
    return idxs

def expected_by_rule(fps: float, n_frames: int) -> int:
    """
    Compute the expected number of sampled frames according to the rule.
    """
    stride = int(fps / 5)
    idxs = _compute_indices_by_stride(n_frames, stride, 0)
    return len(idxs)


# ---------------------- Scene Prompt Utilities ----------------------

def build_prompt_from_scene_json(scene_json_path: Path) -> Optional[str]:
    """
    Load the scene JSON and concatenate:
      prompt = "<SceneSummary>  <SceneDescription>"
    Return "" if fields missing; return None if file unreadable.
    """
    try:
        with open(scene_json_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        summary = (obj.get("SceneSummary") or "").strip()
        desc = (obj.get("SceneDescription") or "").strip()
        if not summary or not desc:
            raise ValueError("Both summary and description are empty")
        return f"{summary}  {desc}"
    except Exception:
        return None

# ---------------------- Filtering ----------------------

def apply_filters(
    df: pd.DataFrame,
    fps_min: Optional[float],
    fps_max: Optional[float],
    frames_min: Optional[int],
    frames_max: Optional[int],
    fps_col: str = "fps",
    frames_col: str = "num frames",
) -> pd.DataFrame:
    """Apply filters where frame-count filters are evaluated AFTER hypothetically
    downsampling video to target_fps (default 24)."""

    mask = pd.Series(True, index=df.index)

    # ---- Safe conversion of numeric columns ----
    fps_s = pd.to_numeric(df.get(fps_col, pd.Series(index=df.index, dtype=float)), errors="coerce")
    frm_s = pd.to_numeric(df.get(frames_col, pd.Series(index=df.index, dtype=float)), errors="coerce")

    # ---- FPS range filtering (based on original FPS) ----
    if fps_col in df.columns:
        if fps_min is not None:
            mask &= fps_s >= fps_min
        if fps_max is not None:
            mask &= fps_s <= fps_max

    # ---- Compute effective frame count after downsampling to 5 FPS ----
    sample_ratio = (fps_s / 5.0).astype(int)
    frames_after = np.floor(frm_s / sample_ratio) * 4

    # ---- Frame-count filtering based on downsampled frames ----
    if frames_col in df.columns:
        if frames_min is not None:
            mask &= frames_after >= frames_min
        if frames_max is not None:
            mask &= frames_after <= frames_max

    # Optionally attach the computed downsampled frame count for debugging/inspection
    out = df[mask].copy()
    out["frames_after_24fps"] = frames_after[mask].astype("Int64")  # optional
    return out

# ---------------------- Main Pipeline ----------------------

def process_row(
    row: pd.Series,
    poses_out_root: Path,
    video_col: str,
    annotation_col: str,
    scenejson_file: str,
    poses7d_file: str,
    fps_col: str = "fps",
    frames_col: str = "num frames",
) -> Optional[Dict[str, Any]]:
    """
    For one row:
    - read scene JSON and build prompt
    - load (T,7), convert to (T,4,4), save NPZ
    - return dict {video_path, poses_npz, prompt}
    """
    id = Path(str(row[annotation_col])).name
    video_path = Path(str(row[video_col])).as_posix()
    annotation_path = Path(str(row[annotation_col]))
    
    # prompt
    scene_json_path = annotation_path / scenejson_file
    prompt = build_prompt_from_scene_json(scene_json_path)

    # poses
    poses7d_path = annotation_path / poses7d_file
    poses7d = load_poses_7d_from_npz(poses7d_path)
    # T = poses7d_to_T44(poses7d)

    expected_count = expected_by_rule(
        fps=float(row[fps_col]),
        n_frames=int(row[frames_col]),
    )
    assert expected_count == poses7d.shape[0], f"Predicted count {poses7d.shape[0]} does not match expected {expected_count} for video {video_path}"

    # # output path
    # poses_out_root.mkdir(parents=True, exist_ok=True)
    # out_path = poses_out_root / (id + "_T44.npz")
    # np.savez_compressed(out_path, T=poses7d)
    
    return {
        "video_path": video_path,
        "poses_npz": poses7d_path.as_posix(),
        "prompt": prompt,
        'fps': round(float(row[fps_col]),2),
        'num_frames': int(row[frames_col]),
        'num_frames_after_24fps': int(row['frames_after_24fps']),
        'num_poses': poses7d.shape[0],
    }

def run():
    metadata_csv = Path(METADATA_CSV)
    poses_out_root = Path(POSES_OUT_ROOT) if POSES_OUT_ROOT is not None else None
    out_json = Path(OUT_JSON)

    # 1) Read
    df = pd.read_csv(metadata_csv)

    # 2) Filter
    df_f = apply_filters(
        df,
        fps_min=FPS_MIN,
        fps_max=FPS_MAX,
        frames_min=FRAMES_MIN,
        frames_max=FRAMES_MAX,
        fps_col=FPS_COL,
        frames_col=FRAMES_COL,
    )
    
    print(f"[Info] Filtered from {len(df)} to {len(df_f)} items.")

    # 3-4-5) Build outputs
    outputs: List[Dict[str, Any]] = []
    # tqdm shows progress automatically with total=len(df_f)
    for _, row in tqdm(df_f.iterrows(), total=len(df_f), desc="Processing rows", unit="row"):
        try:
            item = process_row(
                row=row,
                poses_out_root=poses_out_root,
                video_col=VIDEO_COL,
                annotation_col=ANNOTATION_COL,
                scenejson_file=SCENEJSON_FILE,
                poses7d_file=POSES7D_FILE,
                fps_col=FPS_COL,
                frames_col=FRAMES_COL,
            )
            if item is not None:
                outputs.append(item)
        except Exception as e:
            tqdm.write(f"[Warn] Skip row due to error: {e}")  # prints safely with tqdm
            # pass
        
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(outputs, f, ensure_ascii=False, indent=2)

    print(f"[Done] Wrote {len(outputs)} items to {out_json}")

if __name__ == "__main__":
    run()
    
