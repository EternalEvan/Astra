# Disable NVLink SHARP functionality if NVLS issues are suspected
export NCCL_NVLS_ENABLE=0

# If the above is ineffective, try disabling InfiniBand support 
# (especially in non-InfiniBand or Ethernet environments)
export NCCL_IB_DISABLE=1

# Set detailed debugging information level to help with troubleshooting
export NCCL_DEBUG=INFO

# Set a longer timeout for NCCL operations to prevent failures during slow initialization
export NCCL_TIMEOUT=1800

# GPU device visibility and additional NCCL overrides
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# Path configuration
ROOT=./
cd $ROOT
export PYTHONPATH="$ROOT:$PYTHONPATH"

# Execute multi-GPU pre-tokenization script
torchrun \
  --nproc_per_node=8 \
  --master_port=29600 \
  pretokenize.py
# >> pretokenize_spatialvid2_multigpu.log 2>&1