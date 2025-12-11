CUDA_VISIBLE_DEVICES=0 python infer_origin.py \
    --condition_pth ./encoded_video.pth \
    --output_path ./ \
    --prompt "A drone flying scene in a game world" \
    --modality_type sekai

