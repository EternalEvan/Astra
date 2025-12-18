This README explains how to **filter SpatialVID samples** and run **multi-GPU pretokenization**, then **merge manifests** if the pretokenization is interrupted or finished.

---

## 1) Download & Extract
Download SpatialVID (e.g., [SpatialVID-HQ](https://huggingface.co/datasets/FelixYuan/SpatialVID-HQ)) and extract it to a fixed directory.

Example:
- Data root: `/mnt/data/SpatialVID-HQ/`

Make sure the metadata CSV and annotation files exist (e.g., `SpatialVID_HQ_metadata.csv`, `poses.npy`, `caption.json`).

---

## 2) Build the filtered final JSON (manifest*.json)
Script: `get_final_json.py`

### Edit paths / filters inside the script
Update these (top of file):
- `METADATA_CSV`
- `OUT_JSON`
- Optional filters: `FPS_MIN/FPS_MAX`, `FRAMES_MIN/FRAMES_MAX`

### Run
```bash
python get_final_json.py
```

Output: a filtered JSON list (e.g., `manifest25.json`).

---

## 3) Multi-GPU pretokenize
Script: `pretokenize.py` (torchrun distributed)

### IMPORTANT: update hard-coded paths
`pretokenize.py` contains machine-specific paths (e.g.,  pretrained / dataset / output paths).  
Make sure they match your environment.

### Run (recommended via script)
Script: `pretokenize.sh`

Edit in `pretokenize.sh`:
- `CUDA_VISIBLE_DEVICES=...`
- `--nproc_per_node=...`
- `--master_port=...`
- `--json_path <your manifest*.json>`
- `--output_dir <output folder>`

Run:
```bash
bash pretokenize.sh
```

Outputs in `--output_dir`:
- `SpatialVID_r{rank}_{seq}.pth` (per-sample tensors)
- `manifest_rank{rank}.jsonl` (per-rank manifest shards)
- `manifest.json` (merged final manifest, if completed normally)

---

## 4) If interrupted: merge rank jsonl into one manifest
Script: `merge_json.py`

### Edit output_dir inside mergejson.py
Set:
- `output_dir = "<same as pretokenize --output_dir>"`

### Run
```bash
python merge_json.py
```

Result:
- `output_dir/manifest.json` merged from `manifest_rank*.jsonl` (deduplicated).


