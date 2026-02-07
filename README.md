# FSDPToys - Lightweight FSDP Training Framework

FSDPToys is a lightweight repository for large model training based on PyTorch FSDP (Fully Sharded Data Parallel). It provides simple and universal distributed training code without requiring pre-trained weights, and ensures perfect loss alignment between FSDP1 and FSDP2 implementations.

## 🚀 Features

- **Efficient Distributed Training**: Native PyTorch FSDP/FSDP2 implementation with sharded parameters, gradients, and optimizer states
- **Hybrid Sharding (HSDP)**: Flexible sharding strategies via Device Mesh configuration
- **Memory Optimization**:
  - Gradient Checkpointing
  - CPU Offloading (parameters, gradients, optimizer states)
  - Chunk Loss for reduced peak memory usage
  - Optimizer offload (only for optimizer states)
  - FSDP stream reuse patches for FSDP1/FSDP2
- **Hardware Compatibility**: Supports both NVIDIA GPU (CUDA) and Huawei Ascend NPU
- **Built-in Profiler and Snapshot**: Performance analysis tools for GPU/NPU profiling

## 📁 Directory Structure

```
FSDPToys/
├── accelerate/              # Acceleration components
│   ├── fsdp1_patch.py       
│   ├── fsdp2_patch.py      
│   ├── loss.py              # Chunk Loss
│   ├── offload.py           # Opitimizer Offloading
│   └── recompute.py         
├── models/                  # Model implementations
│   ├── llama2.py            # Llama2 model
│   ├── llama3.py            
│   ├── llama4/              
│   └── qwen3_moe_mini.py    
├── utils/                   # Utility functions
│   ├── profiler.py          # Performance profiler
│   └── utils.py             
├── train_fsdp1.py           # FSDP1 training script
├── train_fsdp2.py           # FSDP2 training script
├── train_simple.py          # Simple single-GPU training script
├── run_fsdp.sh
├── test.py
└── README.md                
```

## ⚙️ Requirements

- **Python**: >= 3.10
- **PyTorch**: >= 2.6
- Torch_npu: https://gitcode.com/Ascend/pytorch

## 🚦Quick Start

### Using Launch Script
The pre-configured script automatically detects hardware and starts training:

```bash
bash run_fsdp.sh
```

### Manual Training
Customize training with `torchrun`:

```bash
# FSDP1 example
torchrun --nnodes=1 --nproc_per_node=4 train_fsdp1.py \
  --batch_size=4 \
  --seq_len=4096 \
  --fsdp_size=8 \
  --gradient_checkpointing \
  --chunk_loss \
  --cpu_offload

# FSDP2 example
torchrun --nnodes=1 --nproc_per_node=8 train_fsdp2.py \
  --batch_size=4\
  --seq_len=4096 \
  --fsdp_size=8 \
  --gradient_checkpointing
```

## 🔧 Key Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--batch_size` | int | 4 | Batch size per GPU |
| `--seq_len` | int | 1024 | Input sequence length |
| `--fsdp_size` | int | 8 | HSDP sharding size |
| `--cpu_offload` | bool | False | Enable CPU offloading |
| `--optimizer_offload` | bool | False | Offload optimizer states only |
| `--gradient_checkpointing` | bool | False | Enable gradient checkpointing |
| `--chunk_loss` | bool | False | Enable chunked loss computation |
| `--profile` | bool | False | Enable PyTorch profiling |
| `--snapshot` | bool | False | Enable PyTorch memory snapshot |

## 💡 Memory Optimization Tips

1. `fsdp_size=world_size`
2. `--gradient_checkpointing`
3. ` --chunk_loss`
4. `--cpu_offload` or `--optimizer_offload`
