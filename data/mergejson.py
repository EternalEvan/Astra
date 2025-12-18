import json
import os
import glob


def manifest_path(output_dir: str) -> str:
    return os.path.join(output_dir, "manifest.json")

def load_manifest(output_dir: str) -> dict:
    path = manifest_path(output_dir)
    if not os.path.exists(path):
        return {"entries": []}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        # Compatible with legacy structures
        if "entries" not in data:
            data = {"entries": data.get("entries", [])}
        return data

def save_manifest(output_dir: str, manifest: dict):
    path = manifest_path(output_dir)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def merge_all_jsonl_to_manifest(output_dir: str):
    """
    Called only on rank0: Merges all manifest_rank*.jsonl files into 
    manifest.json (deduplicated by 'key').
    """
    # 1) Load the current main manifest (empty if it doesn't exist)
    manifest = load_manifest(output_dir)
    have = {e.get("key"): i for i, e in enumerate(manifest.get("entries", [])) if e.get("key")}

    # 2) Scan all jsonl files
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
        print(f"[mergejson] Merged {added} new entries from {len(jsonl_files)} jsonl files into manifest.")
    else:
        print(f"[mergejson] No new entries to merge from {len(jsonl_files)} jsonl files.")
        

if __name__ == "__main__":
    output_dir = "/mnt/data/louis_crq/preprocessed_data/SpatialVID_Wan22"  # Change this to your output directory
    merge_all_jsonl_to_manifest(output_dir)