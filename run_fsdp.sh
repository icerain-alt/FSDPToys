mkdir -p logs

export ASCEND_LAUNCH_BLOCKING=0
export MULTI_STREAM_MEMORY_REUSE=2
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

ps aux | grep train | grep -v grep | awk '{print $2}' | xargs kill -9

torchrun \
  --nnodes=1 \
  --nproc_per_node=4 \
  train_fsdp2.py \
  --batch_size=4 \
  --seq_len=2048 \
  --fsdp_size=4 \
  --gradient_checkpointing \
  2>&1 | tee "logs/$(date +%Y%m%d%H%M%S).log"