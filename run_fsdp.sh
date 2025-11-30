mkdir -p .logs
ps aux | grep train | grep -v grep | awk '{print $2}' | xargs kill -9
torchrun --nproc_per_node=8 --master_port=29502 train_fsdp2.py 2>&1 | tee .logs/$(date +%Y%m%d%H%M%S).log