# FSDPToys - Lightweight FSDP Training Framework

FSDPToys is a lightweight repository for large model training based on PyTorch FSDP (Fully Sharded Data Parallel). It provides simple and universal distributed training code without requiring pre-trained weights, and ensures perfect loss alignment between FSDP1 and FSDP2 implementations.

## 🚀 Features

- **Efficient Distributed Training**: Native PyTorch FSDP/FSDP2 implementation with sharded parameters, gradients, and optimizer states
- **Hybrid Sharding (HSDP)**: Flexible sharding strategies via Device Mesh configuration
- **Meta Initialization**: Memory-efficient model initialization using PyTorch meta device for large-scale model
- **Memory Optimization**:
  - Gradient Checkpointing
  - CPU Offloading (parameters, gradients, optimizer states)
  - Chunk Loss for reduced peak memory usage
  - Optimizer offload (only for optimizer states)
  - FSDP stream reuse patches for FSDP1/FSDP2
- **Hardware Compatibility**: Supports both NVIDIA GPU (CUDA) and Huawei Ascend NPU
- **Built-in Profiler and Snapshot**: Performance analysis tools for GPU/NPU profiling

## 📊 FSDP1 Performance Comparison

> **Note**: Performance metrics are measured on the Llama2 - 7B (6.7B actual) model with batch_size=1, seq_len=4096, and 8× Ascend A2 NPUs.
>
> **Dependencies**: torch==2.7.1, torch_npu==2.7.1, CANN 8.3.RC1

| Wrapped Module                 | Gradient Checkpointing | Chunked Loss | Optimizer Offloading | CPU Offloading | Step Time (s) | Memory allocated (GB) | Memory Reserved (GB) |
| ------------------------------ | :--------------------: | :----------: | :------------------: | :------------: | :-----------: | :-------------------: | :------------------: |
| Transformer Block              |           ❌            |      ❌       |          ❌           |       ❌        |     1.96      |         9.72          |        44.92         |
| Transformer Block              |           ✅            |      ❌       |          ❌           |       ❌        |     2.30      |         9.72          |        21.90         |
| Transformer Block              |           ✅            |      ✅       |          ❌           |       ❌        |     2.31      |         9.48          |        21.59         |
| Transformer Block              |           ✅            |      ✅       |          ✅           |       ❌        |     3.09      |         3.20          |        20.20         |
| Transformer Block              |           ✅            |      ✅       |          ❌           |       ✅        |     10.44     |         0.06          |        12.31         |
| Transformer Block + Attn + MLP |           ✅            |      ✅       |          ❌           |       ✅        |     10.38     |         0.06          |         9.36         |
| Transformer Block + Attn + MLP |           ❌            |      ❌       |          ❌           |       ❌        |     2.12      |         9.72          |        42.80         |
| Transformer Block + Attn + MLP |           ✅            |      ❌       |          ❌           |       ❌        |     2.78      |         9.72          |        19.85         |

## 📊 FSDP2 Performance Comparison

> **Note**: Performance metrics are measured on the Llama2-7B (6.7B actual) model with batch_size=1, seq_len=4096, and 8× Ascend A2 NPUs.
>
> **Dependencies**: torch==2.7.1, torch_npu==2.7.1. For FSDP2, `patch/fsdp2_patch.py` was applied to fix backward prefetching bug in 2.7.1.

| Wrapped Module                 | Gradient Checkpointing | Chunked Loss | Optimizer Offloading | CPU Offloading | Step Time (s) | Memory allocated (GB) | Memory Reserved (GB) |
| ------------------------------ | :--------------------: | :----------: | :------------------: | :------------: | :-----------: | :-------------------: | :------------------: |
| Transformer Block              |           ❌            |      ❌       |          ❌           |       ❌        |     1.90      |         9.72          |        40.78         |
| Transformer Block              |           ✅            |      ❌       |          ❌           |       ❌        |     2.26      |         9.72          |        18.17         |
| Transformer Block              |           ✅            |      ✅       |          ❌           |       ❌        |     2.27      |         9.48          |        17.78         |
| Transformer Block              |           ✅            |      ✅       |          ✅           |       ❌        |     3.74      |         3.20          |        13.62         |
| Transformer Block              |           ✅            |      ✅       |          ❌           |       ✅        |     10.31     |         0.06          |         8.44         |
| Transformer Block + Attn + MLP |           ✅            |      ✅       |          ❌           |       ✅        |     10.37     |         0.06          |         7.64         |
| Transformer Block + Attn + MLP |           ❌            |      ❌       |          ❌           |       ❌        |     2.02      |         9.72          |        40.36         |
| Transformer Block + Attn + MLP |           ✅            |      ❌       |          ❌           |       ❌        |     2.41      |         9.72          |        17.45         |

> **Note**: All experiments were run on ARM CPUs. The CPU offload policy generally performs better on x86 CPUs.

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
- Torch npu: https://gitcode.com/Ascend/pytorch

## 🚦Quick Start

1. Using Launch Script

The pre-configured script automatically detects hardware and starts training:

```bash
bash run_fsdp.sh
```

2. Customize training with `torchrun`:

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

1. `--fsdp_size=world_size`
2. `--gradient_checkpointing`
3. `--chunk_loss`
4. `--cpu_offload` or `--optimizer_offload`

## 📚 Reference Documents

1. [FullyShardedDataParallel — PyTorch 2.3 documentation](https://pytorch.org/docs/2.3/fsdp.html#)
2. [FSDP Notes — PyTorch 2.3 documentation](https://pytorch.org/docs/2.3/notes/fsdp.html#fsdp-notes)
3. [pytorch/examples: A set of examples around pytorch in Vision, Text, Reinforcement Learning, etc.](https://github.com/pytorch/examples)
4. [Introducing PyTorch Fully Sharded Data Parallel (FSDP) API | PyTorch](https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/)
5. [[1910.02054\] ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)
6. [[2304.11277\] PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel](https://arxiv.org/abs/2304.11277)
7. [Rethinking PyTorch Fully Sharded Data Parallel (FSDP) from First Principles - distributed - PyTorch Developer Mailing List](https://dev-discuss.pytorch.org/t/rethinking-pytorch-fully-sharded-data-parallel-fsdp-from-first-principles/1019)
8. [FSDP & CUDACachingAllocator: an outsider newb perspective - distributed - PyTorch Developer Mailing List](https://dev-discuss.pytorch.org/t/fsdp-cudacachingallocator-an-outsider-newb-perspective/1486)
9. [torchtitan/docs/fsdp.md at main · pytorch/torchtitan · GitHub](https://github.com/pytorch/torchtitan/blob/main/docs/fsdp.md)
10. [CUDA Environment Variables — PyTorch 2.10 documentation](https://docs.pytorch.org/docs/stable/cuda_environment_variables.html)
