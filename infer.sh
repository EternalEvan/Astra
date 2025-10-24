CUDA_VISIBLE_DEVICES=0 python infer_origin.py \
    --condition_pth /share_zhuyixuan05/zhuyixuan05/sekai-game-walking/00100100001_0004650_0004950/encoded_video.pth \
    --output_path /home/zhuyixuan05/ReCamMaster/moe/infer_results/sekai.mp4 \
    --prompt "A drone flying scene in a game world" \
    --modality_type sekai

CUDA_VISIBLE_DEVICES=1 python infer_moe.py \
    --condition_pth /share_zhuyixuan05/zhuyixuan05/nuscenes_video_generation_dynamic/scenes/scene-0001_CAM_FRONT/encoded_video-480p.pth \
    --output_path /home/zhuyixuan05/ReCamMaster/moe/infer_results/nuscenes.mp4 \
    --prompt "A car is driving" \
    --modality_type nuscenes

CUDA_VISIBLE_DEVICES=0 python infer_origin.py \
    --condition_pth /share_zhuyixuan05/zhuyixuan05/spatialvid/a9a6d37f-0a6c-548a-a494-7d902469f3f2_0000000_0000300/encoded_video.pth \
    --output_path /home/zhuyixuan05/ReCamMaster/moe/infer_results/spatialvid.mp4 \
    --prompt "A man is entering the room" \
    --modality_type sekai

CUDA_VISIBLE_DEVICES=1 python infer_moe.py \
    --condition_pth /share_zhuyixuan05/zhuyixuan05/openx-fractal-encoded/episode_000001/encoded_video.pth \
    --output_path /home/zhuyixuan05/ReCamMaster/moe/infer_results/openx.mp4 \
    --prompt "A robotic arm is moving the object" \
    --modality_type openx

CUDA_VISIBLE_DEVICES=1 python infer_origin.py \
    --condition_pth /share_zhuyixuan05/zhuyixuan05/sekai-game-drone/00500210001_0012150_0012450/encoded_video.pth \
    --output_path /home/zhuyixuan05/ReCamMaster/moe/infer_results/edit.mp4 \
    --prompt "A drone flying scene in a game world, and it starts to rain" \
    --modality_type sekai


CUDA_VISIBLE_DEVICES=0 python infer_origin.py \
    --condition_pth /share_zhuyixuan05/zhuyixuan05/spatialvid/0268e6b0-f41e-5c2f-bf6b-936e55dc4a05_0000600_0000900/encoded_video.pth \
    --output_path /home/zhuyixuan05/ReCamMaster/moe/infer_results/spatialvid.mp4 \
    --prompt "walking in the city, the weather from day turns to night" \
    --modality_type sekai \
    --direction "right" \
    --initial_condition_frames "1"

    CUDA_VISIBLE_DEVICES=1 python infer_moe.py \
    --condition_pth /share_zhuyixuan05/zhuyixuan05/nuscenes_video_generation_dynamic/scenes/scene-0001_CAM_FRONT/encoded_video-480p.pth \
    --output_path /home/zhuyixuan05/ReCamMaster/moe/infer_results/nuscenes.mp4 \
    --prompt "A car is driving" \
    --modality_type nuscenes
    
    CUDA_VISIBLE_DEVICES=1 python infer_moe.py \
    --condition_pth /share_zhuyixuan05/zhuyixuan05/openx-fractal-encoded/episode_000001/encoded_video.pth \
    --output_path /home/zhuyixuan05/ReCamMaster/moe/infer_results/openx.mp4 \
    --prompt "A robotic arm is moving the object" \
    --modality_type openx

    CUDA_VISIBLE_DEVICES=1 python infer_origin.py \
    --condition_pth /share_zhuyixuan05/zhuyixuan05/sekai-game-walking/00100100001_0004650_0004950/encoded_video.pth \
    --output_path /home/zhuyixuan05/ReCamMaster/moe/infer_results/sekai.mp4 \
    --prompt "A drone flying scene in a game world" \
    --modality_type sekai