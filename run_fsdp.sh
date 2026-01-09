mkdir -p logs

export MULTI_STREAM_MEMORY_REUSE=2
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

ps aux | grep train | grep -v grep | awk '{print $2}' | xargs kill -9
torchrun --nproc_per_node=8 train_fsdp2.py --fsdp_size=8 2>&1 | tee logs/$(date +%Y%m%d%H%M%S).log