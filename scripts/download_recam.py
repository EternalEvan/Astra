from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="KwaiVGI/ReCamMaster-Wan2.1",
    local_dir="models/ReCamMaster/checkpoints",
    resume_download=True  # 支持断点续传
)
