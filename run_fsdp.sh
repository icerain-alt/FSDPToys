#!/bin/bash

mkdir -p logs

if [ -x "$(command -v nvidia-smi)" ]; then
  echo "[INFO] Running in CUDA environment."
  export CUDA_LAUNCH_BLOCKING=0
  export TORCH_NCCL_AVOID_RECORD_STREAMS=1
  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
else
  echo "[INFO] Running in non-CUDA environment."
  export ASCEND_LAUNCH_BLOCKING=0
  export MULTI_STREAM_MEMORY_REUSE=2
  export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
fi

ps aux | grep train | grep -v grep | awk '{print $2}' | xargs kill -9

torchrun \
  --nnodes=1 \
  --nproc_per_node=8 \
  train_fsdp2.py \
  --batch_size=4 \
  --seq_len=4096 \
  --fsdp_size=8 \
  --gradient_checkpointing \
  --chunk_loss \
  2>&1 | tee "logs/$(date +%Y%m%d%H%M%S).log"