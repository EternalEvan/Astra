import os
import re
import cv2
import glob
import json
import uuid
import torch
import argparse
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from videox_fun.models import AutoencoderKLWan, AutoencoderKLWan3_8, CLIPModel, WanT5EncoderModel
from transformers import AutoTokenizer
from omegaconf import OmegaConf
import torch.distributed as dist

from typing import List, Dict


def init_distributed():
    """initialize torch.distributed"""
    if dist.is_available() and not dist.is_initialized():
        backend = "nccl"
        dist.init_process_group(backend=backend)
    rank = dist.get_rank() if dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_initialized() else 1
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return rank, world, local_rank

def barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

def bcast_object(obj, src=0):
    """Broadcast a Python object (requires PyTorch support for broadcast_object_list)."""
    if not (dist.is_available() and dist.is_initialized()):
        return obj

    data = [obj] if dist.get_rank() == src else [None]
    dist.broadcast_object_list(data, src=src)
    return data[0]

def gather_objects(obj, dst=0):
    """Gather Python objects from all ranks to dst (rank 0)."""
    if not (dist.is_available() and dist.is_initialized()):
        return [obj]
    output = [None for _ in range(dist.get_world_size())] if dist.get_rank() == dst else None
    try:
        dist.gather_object(obj, output, dst=dst)  # new API
        return output if dist.get_rank() == dst else None
    except Exception:
        # Compatibility fallback for older versions:
        # each rank writes a local file, and rank 0 post-processes later
        return None

# ----------------- Image size and cropping -----------------

def center_crop_params(h, w, target_ratio: float):
    in_ratio = w / h
    if in_ratio > target_ratio:
        new_w = int(h * target_ratio)
        new_h = h
        left = (w - new_w) // 2
        top = 0
    else:
        new_w = w
        new_h = int(w / target_ratio)
        left = 0
        top = (h - new_h) // 2
    return top, left, new_h, new_w

# ----------------- Encoder -----------------

class VideoEncoder():
    def __init__(
        self,
        config_path,
        pretrained_model_name_or_path,
        device_str="cuda:0",
        io_threads=None,
    ):
        self.device_str = device_str
        # model_manager = ModelManager(torch_dtype=torch.bfloat16, device=self.device_str)
        # model_manager.load_models([text_encoder_path, vae_path])
        # self.pipe = WanVideoReCamMasterPipeline.from_model_manager(model_manager)
        # self.pipe.device = self.device_str
        weight_dtype = torch.bfloat16
        
        config = OmegaConf.load(config_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(pretrained_model_name_or_path, config['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer')),
        )
        self.text_encoder = WanT5EncoderModel.from_pretrained(
            os.path.join(pretrained_model_name_or_path, config['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')),
            additional_kwargs=OmegaConf.to_container(config['text_encoder_kwargs']),
            low_cpu_mem_usage=True,
            torch_dtype=weight_dtype,
        )
        self.text_encoder = self.text_encoder.eval().to(device=self.device_str, dtype=weight_dtype)
        # Get Vae
        Chosen_AutoencoderKL = {
            "AutoencoderKLWan": AutoencoderKLWan,
            "AutoencoderKLWan3_8": AutoencoderKLWan3_8
        }[config['vae_kwargs'].get('vae_type', 'AutoencoderKLWan')]
        self.vae = Chosen_AutoencoderKL.from_pretrained(
            os.path.join(pretrained_model_name_or_path, config['vae_kwargs'].get('vae_subpath', 'vae')),
            additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
        )
        self.vae = self.vae.eval().to(device=self.device_str, dtype=weight_dtype)

      # using fixed resolution
        self.target_w, self.target_h = 832, 480
        self.target_ratio = self.target_w / self.target_h
        self.io_threads = io_threads or min(8, (os.cpu_count() or 8) // 2)

    # process each frame：BGR->center crop->resize->RGB->[-1,1] -> (C,H,W) float32
    def _process_frame_array(self, bgr: np.ndarray) -> np.ndarray:
        if bgr is None:
            raise ValueError("Empty frame encountered.")
        h, w = bgr.shape[:2]
        top, left, new_h, new_w = center_crop_params(h, w, self.target_ratio)
        im = bgr[top:top+new_h, left:left+new_w]
        im = cv2.resize(im, (self.target_w, self.target_h), interpolation=cv2.INTER_LINEAR)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = im.astype(np.float32) / 255.0
        im = (im - 0.5) / 0.5
        chw = np.transpose(im, (2, 0, 1))
        return chw

    # load frames from a video -> (1,C,T,H,W) bfloat16/cuda
    def load_video_as_tensor(self, video_path: str, device=None, dtype=torch.bfloat16, pin_memory=True):
        device = device or self.device_str
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")

        raw = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            raw.append(frame)
        cap.release()

        if not raw:
            raise ValueError(f"No frames decoded from video: {video_path}")

        T = len(raw)
        H, W = self.target_h, self.target_w
        host = torch.empty((T, 3, H, W), dtype=torch.float32, pin_memory=pin_memory)

        workers = self.io_threads
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(self._process_frame_array, raw[i]): i for i in range(T)}
            for fut in as_completed(futures):
                i = futures[fut]
                chw = fut.result()
                host[i].copy_(torch.from_numpy(chw), non_blocking=False)

        t = host.permute(1, 0, 2, 3).unsqueeze(0).to(device=device, dtype=dtype, non_blocking=True)
        return t

    def encode_whole_video(self, frames_1cthw):
        with torch.no_grad():
            latents = self.vae.encode(frames_1cthw)[0].sample()
        return latents  # cthw
    
    def encode_first_frame(self, frames_1cthw):
        with torch.no_grad():
            first_frame = frames_1cthw[:, :, 0:1, :, :].repeat(1, 1, 4, 1, 1)
            latents = self.vae.encode(first_frame)[0].sample()
        return latents  # cthw
    
    def encode_prompt(self, prompt: str):
        with torch.no_grad():
            prompt_ids = self.tokenizer(
                prompt, 
                padding="max_length", 
                max_length=512, 
                truncation=True, 
                add_special_tokens=True, 
                return_tensors="pt"
            )
            text_input_ids = prompt_ids.input_ids
            prompt_attention_mask = prompt_ids.attention_mask

            seq_lens = prompt_attention_mask.gt(0).sum(dim=1).long()
            prompt_embeds = self.text_encoder(text_input_ids.to(self.device_str), attention_mask=prompt_attention_mask.to(self.device_str))[0]
            prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
                    
            # for k in prompt_emb:
            #     prompt_emb[k] = prompt_emb[k].cpu()
        return prompt_embeds

# ---------- manifest tools ----------

def manifest_path(output_dir: str) -> str:
    return os.path.join(output_dir, "manifest.json")

def load_manifest(output_dir: str) -> dict:
    path = manifest_path(output_dir)
    if not os.path.exists(path):
        return {"entries": []}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        if "entries" not in data:
            data = {"entries": data.get("entries", [])}
        return data

def save_manifest(output_dir: str, manifest: dict):
    path = manifest_path(output_dir)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def build_key(video_path: str, npz_path: str, num_frames: int) -> str:
    return f"{os.path.abspath(video_path)}|{os.path.abspath(npz_path)}|{num_frames}"

def allocate_ranked_filename(dataset: str, output_dir: str, rank: int, seq: int, pad: int = 6) -> str:
    """Avoid multi-process conflicts by embedding the rank and local sequence index in the filename."""
    name = f"{dataset}_r{rank}_{seq:0{pad}d}.pth"
    full = os.path.join(output_dir, name)
    return name, full

def per_rank_jsonl_path(output_dir: str, rank: int) -> str:
    return os.path.join(output_dir, f"manifest_rank{rank}.jsonl")

def append_entry_atomic_jsonl(output_dir: str, rank: int, entry: Dict):
    """Atomically append an entry to the per-rank JSONL file (line-delimited JSON)."""
    path = per_rank_jsonl_path(output_dir, rank)
    line = json.dumps(entry, ensure_ascii=False) + "\n"
    fd = os.open(path, os.O_CREAT | os.O_WRONLY | os.O_APPEND, 0o644)
    try:
        os.write(fd, line.encode("utf-8"))
    finally:
        os.close(fd)

def merge_all_jsonl_to_manifest(output_dir: str):
    """Called only on rank 0: merge all manifest_rank*.jsonl files into manifest.json (deduplicated by key)."""
    # 1) Load the current main manifest (empty if it does not exist)
    manifest = load_manifest(output_dir)
    have = {e.get("key"): i for i, e in enumerate(manifest.get("entries", [])) if e.get("key")}

    # 2) scan all jsonl
    jsonl_files = sorted(glob.glob(os.path.join(output_dir, "manifest_rank*.jsonl")))
    added = 0
    for f in jsonl_files:
        with open(f, "r", encoding="utf-8") as rf:
            for line in rf:
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                except Exception:
                    continue
                k = e.get("key")
                if not k or k in have:
                    continue
                manifest["entries"].append(e)
                have[k] = len(manifest["entries"]) - 1
                added += 1

    if added > 0:
        save_manifest(output_dir, manifest)
    return added

# ----------------- Main pipeline (video + JSON, distributed) -----------------

def encode_videos_from_json(json_path, dataset: str, vae_path, output_dir, io_threads=None,data_dir = None):
    os.makedirs(output_dir, exist_ok=True)

    # Initialize distributed process

    rank, world_size, local_rank = init_distributed()
    device_str = f"cuda:{local_rank}"

    # Only rank 0 loads the historical manifest and broadcasts the
    # "already processed keys" set to all ranks for idempotent skipping
    if rank == 0:
        old_manifest = load_manifest(output_dir)
        done_keys = set(e.get("key") for e in old_manifest.get("entries", []) if e.get("key"))
        existed_pths = [str(e.get("pth")) for e in old_manifest.get("entries", []) if e.get("pth")]
        print(f"[Rank {rank}] Loaded existing manifest with {len(done_keys)} done keys.")
        print(f"[Rank {rank}] Existing encoded files: {len(existed_pths)}")
    else:
        done_keys = None
        existed_pths = None
    done_keys = bcast_object(done_keys, src=0)
    existed_pths = bcast_object(existed_pths, src=0)
    print(existed_pths[:5])

    # Load the task list (all ranks read it for easy partitioning)
    with open(json_path, "r", encoding="utf-8") as f:
        all_items = json.load(f)
    assert isinstance(all_items, list), "JSON 顶层应为列表。"

    # Partition: each rank processes its own share (i % world_size == rank)

    indices = [i for i in range(len(all_items)) if (i % world_size) == rank]
    items = [all_items[i] for i in indices]

    # Each rank initializes its own encoder on its GPU
    encoder = VideoEncoder(
        "./wan_civitai.yaml",
        vae_path,
        device_str=device_str,
        io_threads=io_threads
    )

    local_entries = []
    local_seq = 0  

    iterator = items
    # Rank 0 shows overall progress; other ranks print minimally to avoid stdout contention
    if rank == 0:
        iterator = tqdm(items, desc=f"[Rank {rank}] Encoding videos (world={world_size})")

    for it in iterator:
        # Allocate output filenames (including rank to avoid conflicts)
        pth_name, encoded_path = allocate_ranked_filename(dataset, output_dir, rank, local_seq)

        local_seq += 1
        if os.path.exists(encoded_path):
            print(f"[Rank {rank}] Skipping existing file: {encoded_path}")
            # Use the `continue` statement to skip the remaining code in the current loop iteration
            continue
                       
        video_path = it["video_path"]
        video_path = data_dir + video_path
        npz_path = data_dir + it["poses_npz"]
        prompt = it.get("prompt", "")
        
        fps = it["fps"]
        num_frames = it["num_frames"]
        num_poses = it["num_poses"]
        
        if str(pth_name) in existed_pths:
            if rank == 0:
                print(f"[WARN] already exists: {pth_name}")
            continue
            
        else:
            if rank == 0:
                print( f"[Rank {rank}] Processing: {pth_name} " )
                
            if not os.path.exists(video_path):
                if rank == 0:
                    print(f"[WARN] video not found: {video_path}")
                continue
            if not os.path.exists(npz_path):
                if rank == 0:
                    print(f"[WARN] poses npz not found: {npz_path}")
                continue

            # load video -> (1,C,T,H,W)
            try:
                frames_1cthw = encoder.load_video_as_tensor(
                    video_path, device=device_str, dtype=torch.bfloat16, pin_memory=True
                )
            except Exception as e:
                print(f"[Rank {rank}] [ERR] decode video failed: {video_path}  ({e})")
                continue
            
            try:
                _, C, T_video, H, W = frames_1cthw.shape
                assert abs(T_video - num_frames) < 3, f"Video frame count does not match JSON: {T_video} != {num_frames} ({video_path})"                
                sampled_indices = torch.linspace(0, T_video - 1, steps=int(num_poses * 4 + 4)).long()
                frames_1cthw = frames_1cthw[:, :, sampled_indices, :, :]
                _, C, new_T_video, H, W = frames_1cthw.shape
            except Exception as e:
                print(f"[Rank {rank}] [ERR] process frames failed: {video_path}  ({e})")
                continue
            
            # load camera
            try:
                extrinsics = np.load(npz_path, allow_pickle=True).astype(np.float32)
                # cam = np.load(npz_path, allow_pickle=True)
                # extrinsics = cam['T'].astype(np.float32)
            except Exception as e:
                print(f"[Rank {rank}] [ERR] load npz failed: {npz_path}  ({e})")
                continue

            # skip if this (video, npz, T) key already exists in the historical manifest
            key = build_key(video_path, npz_path, new_T_video)
            if key in done_keys:
                # Already processed -> skip
                continue

            # Encode and save
            try:
                latents = encoder.encode_whole_video(frames_1cthw)
                latents_first = encoder.encode_first_frame(frames_1cthw)
                prompt_emb = encoder.encode_prompt(prompt)
            except Exception as e:
                print(f"[Rank {rank}] [ERR] encode failed: {video_path}  ({e})")
                continue

            encoded_data = {
                "latents": latents.cpu(),
                "latents_first": latents_first.cpu(),
                "cam_emb": {
                    "extrinsic": torch.from_numpy(extrinsics).to(latents.dtype).cpu(),
                    "intrinsic": None
                },
                "prompt_emb": prompt_emb,
            }
            try:
                torch.save(encoded_data, encoded_path)
            except Exception as e:
                print(f"[Rank {rank}] [ERR] save failed: {encoded_path}  ({e})")
                continue
            
            entry = {
                "pth": pth_name,
                "dataset": dataset,
                "video_path": os.path.abspath(video_path),
                "npz_path": os.path.abspath(npz_path),
                "num_frames": int(T_video),
                "compressed_num_frames": int(new_T_video),
                "num_poses": int(num_poses),
                "compressed_frames_1cthw_shape": list(frames_1cthw.shape),
                "fps": float(fps),
                "latents_shape": list(encoded_data["latents"].shape),
                "latents_first_shape": list(encoded_data["latents_first"].shape),
                "npz_shape": list(encoded_data["cam_emb"]["extrinsic"].shape),
                "key": key,
                "prompt": prompt,
                "prompt_present": bool(prompt),
                "rank": rank,
            }
            local_entries.append(entry)
            append_entry_atomic_jsonl(output_dir, rank, entry)
        
    # Synchronize and gather entries from all ranks to rank 0
    barrier()
    # gathered = gather_objects(local_entries, dst=0)

    if rank == 0:
        added = merge_all_jsonl_to_manifest(output_dir)
        manifest = load_manifest(output_dir)
        print(f"[Rank 0] Final merge: +{added} entries. Total: {len(manifest.get('entries', []))}")
        
        print(f"[Rank 0] Done. New total entries: {len(manifest.get('entries', []))}")

    barrier()
    if dist.is_initialized():
        dist.destroy_process_group()

# ----------------- CLI -----------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str,
                        default="/mnt/data/preprocess_data/SpatialVID-HQ/manifest25.json",
                        help="JSON list file with fields: video_path, poses_npz, prompt"
                        )
    parser.add_argument("--output_dir", type=str,
                        default="/mnt/data/preprocessed_data/SpatialVID_Wan21")
    parser.add_argument("--io_threads", type=int, default=0,
                        help="0 = auto; otherwise specify the number of parallel decoding threads "
                        "(mainly useful for image-based pipelines; limited benefit for sequential video decoding)"
                        )
    parser.add_argument("--data_dir", type=str, default="/mnt/data/SpatialVID-HQ/SpatialVid/HQ/",
                        help="path to the dataset"
                        )

    parser.add_argument("--vae_dir", type=str, default="../models/Wan-AI/Wan2.1-T2V-1.3B",
                        help="path to the dataset"
                        )
    args = parser.parse_args()
    encode_videos_from_json(
        args.json_path, "SpatialVID", args.vae_dir, args.output_dir,
        io_threads=(None if args.io_threads == 0 else args.io_threads),data_dir=args.data_dir
    )