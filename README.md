FSDPToys 是一个基于 PyTorch FSDP (Fully Sharded Data Parallel) 的大模型训练框架，专注于提供简单、高效、可扩展的分布式训练实现。

## 功能特性

- **分布式训练**：基于 PyTorch FSDP/FSDP2 实现高效的数据并行训练
- **模型支持**：包含 Llama2、Llama3、Llama4 等多种模型架构
- **优化技术**：支持选择性重计算、CPU Offloading、Chunk Loss等优化策略
- **硬件兼容性**：支持 GPU 和华为 Ascend NPU 硬件加速
- **性能分析**：内置NPU Profiling性能分析工具

## 目录结构

```plainText
├── models/          # 模型实现目录
│   ├── llama2.py    # Llama2 模型实现
│   ├── llama3.py    # Llama3 模型实现
│   └── ...
├── train_fsdp1.py   # FSDP 训练脚本 1
├── train_fsdp2.py   # FSDP 训练脚本 2
├── train_simple.py  # 简单训练脚本
├── loss.py          # 损失函数实现
├── utils.py         # 工具函数集合
├── run_fsdp.sh      # 运行脚本示例
└── logs/            # 日志目录
```

## 快速开始

### 环境要求

- Python 3.10+
- PyTorch 2.6+
- CUDA 11.0+（GPU 支持）
- Ascend NPU SDK（NPU 支持，可选）

### 运行训练

使用提供的 `run_fsdp.sh` 脚本可以快速启动分布式训练：

```bash
bash run_fsdp.sh
```

或者直接使用 `torchrun` 命令：

```bash
torchrun --nnodes=1 --nproc_per_node=4 train_fsdp1.py \
  --batch_size=4 \
  --seq_len=2048 \
  --fsdp_size=4 \
  --gradient_checkpointing
```

## 参数配置

- `--fsdp_size`：控制HSDP，`device_mesh=(world_size//fsdp_size, fsdp_size)`
- `--cpu_offload`：启用FSDP自带的CPU Offloading
- `--gradient_checkpointing`：启用激活值重计算
- `--chunk_loss`：启用Chunk Loss，通过分块计算`cross_entropy`，降低峰值显存

