import os
import time
import argparse
import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as T
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import (
    BackwardPrefetch,
    MixedPrecision,
    ShardingStrategy,
    CPUOffload,
)
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.utils.data.distributed import DistributedSampler

from models.llama2 import Transformer, TransformerBlock, ModelArgs
from utils import (
    format_metrics_to_gb,
    print_model_info,
    seed_all,
    load_fsdp_model,
    save_fsdp_model,
    is_torch_npu_available,
    build_profiler,
)
from accelerate import offload_fsdp_optimizer, load_fsdp_optimizer, chunk_loss_fun

if is_torch_npu_available():
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
    print("CUDA backend not available. Using NPU backend.")
else:
    print("NPU backend not available. Using CUDA backend.")


def get_args():
    parser = argparse.ArgumentParser(description="PyTorch llama FSDP Example")

    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per GPU")
    parser.add_argument(
        "--num_epochs", type=int, default=100, help="Total training epochs"
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of data loading workers"
    )
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument(
        "--seed", type=int, default=421, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--seq_len", type=int, default=1024, help="Input sequence length"
    )
    parser.add_argument(
        "--fsdp_size",
        type=int,
        default=8,
        help="Sharding size for HSDP(Hybrid Sharding)",
    )
    parser.add_argument(
        "--checkpointing_start_index",
        type=int,
        default=0,
        help="Checkpointing start from which layer",
    )
    parser.add_argument(
        "--load_path",
        type=str,
        default="ckpts/fsdp1/llama_checkpoint",
        help="Checkpoint loading path",
    )
    parser.add_argument(
        "--save_path", type=str, default="ckpts/fsdp1", help="Checkpoint saving path"
    )
    parser.add_argument(
        "--save_freq",
        type=int,
        default=1,
        help="Checkpoint saving frequency (in epochs)",
    )
    parser.add_argument(
        "--profile_path",
        type=str,
        default="profile/llama_7b_fsdp1_base",
        help="NPU profiling path",
    )
    parser.add_argument(
        "--cpu_offload",
        action="store_true",
        help="Offload model params, grads and optimizer states to CPU (default: False)",
    )
    parser.add_argument(
        "--optimizer_offload",
        action="store_true",
        help="Offload optimizer states to CPU (default: False); Conflicts with CPU offload.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing (default: False)",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="NPU profiling for performance analysis (default: False)",
    )
    parser.add_argument(
        "--chunk_loss",
        action="store_true",
        help="Enable chunk loss function (default: False)",
    )

    args = parser.parse_args()

    seed_all(args.seed, mode=False, is_npu=is_torch_npu_available())

    return args


def train_one_epoch(model, loader, optimizer, epoch, rank, args):
    model.train()
    total_loss = 0.0

    if args.profile:
        profiler = build_profiler(profile_path=args.profile_path)
        profiler.start()

    for batch_idx, (inputs, labels) in enumerate(loader):
        t0 = time.time()
        inputs = inputs.cuda()
        labels = labels[:, None].cuda().repeat(1, args.seq_len)

        outputs = model(inputs.reshape(-1, args.seq_len))
        if args.chunk_loss:
            loss = chunk_loss_fun(outputs, model.lm_head.weight, labels)
        else:
            logits = model.compute_logits(outputs)
            loss = F.cross_entropy(logits.flatten(0, 1).float(), labels.flatten())

        loss.backward()
        model.clip_grad_norm_(1.0)

        optimizer.step()
        optimizer.zero_grad()

        if args.profile:
            profiler.step()

        # Calculate metrics
        total_loss += loss.item()

        if rank == 0:
            print(
                f"Epoch: {epoch} | Batch: {batch_idx}/{len(loader)} | Elapsed Time: {time.time() - t0:.3f} s | Loss: {loss.item():.4f} | "
                f"Mem_alloc: {format_metrics_to_gb(torch.cuda.memory_allocated())} GB Mem_reserve: {format_metrics_to_gb(torch.cuda.memory_reserved())} GB"
            )

    if args.profile:
        profiler.stop()

    # Sync metrics across devices
    avg_loss = torch.tensor(total_loss / len(loader)).cuda(rank)

    dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)


def main(rank, world_size):
    args = get_args()

    # Prepare dataset
    train_set = datasets.FakeData(
        size=10000,
        image_size=(1, args.seq_len),
        num_classes=1000,
        transform=T.Compose([T.ToTensor(), lambda x: (x * 256).int()]),
    )

    train_sampler = DistributedSampler(
        train_set, num_replicas=world_size, rank=rank, shuffle=True
    )

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )

    # Build model
    simple_llama2_config = ModelArgs(
        n_layers=32,
        vocab_size=100000,
        gradient_checkpointing=args.gradient_checkpointing,
        checkpointing_start_index=args.checkpointing_start_index,
    )

    init_device = "cpu" if rank == 0 else "meta"
    with torch.device(init_device):
        model = Transformer.from_model_args(simple_llama2_config)

    # Device mesh for HSDP
    mesh_2d = init_device_mesh(
        "cuda",
        (world_size // args.fsdp_size, args.fsdp_size),
        mesh_dim_names=["dp", "fsdp"],
    )

    # Load checkpoint on cpu
    load_fsdp_model(model, rank, args.load_path, "fullstate")

    auto_wrap_policy = ModuleWrapPolicy({TransformerBlock})
    model = FSDP(
        model,
        device_id=rank,
        device_mesh=mesh_2d,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=ShardingStrategy.HYBRID_SHARD,  # FULL_SHARD
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
        ),
        forward_prefetch=False,
        limit_all_gathers=True,  # False for ZERO2.
        use_orig_params=False,  # set to True for when some parameters are frozen
        sync_module_states=True,  # broadcast module parameters and buffers from rank 0
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        cpu_offload=CPUOffload(offload_params=args.cpu_offload),
    )
    # Print model info on rank 0
    if rank == 0:
        print_model_info(model)
        print("\n" + "=" * 50, f"{'FSDP Info':^10}", "=" * 50)
        for name, param in model.named_parameters():
            print(
                f"Param name = {name}, shape = {param.size()}, dtype = {param.dtype}, requires_grad = {param.requires_grad}",
                flush=True,
            )

    # Optimizer setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, fused=True)
    if args.optimizer_offload:
        optimizer.register_step_pre_hook(
            lambda optim, args, kwargs: load_fsdp_optimizer(
                optim, torch.cuda.current_device()
            )
        )
        optimizer.register_step_post_hook(
            lambda optim, args, kwargs: offload_fsdp_optimizer(optim)
        )

    # Training loop
    for epoch in range(args.num_epochs):
        train_sampler.set_epoch(epoch)
        train_one_epoch(model, train_loader, optimizer, epoch, rank, args)

        if (epoch + 1) % args.save_freq == 0:
            os.makedirs(
                f"{args.save_path}/llama_checkpoint_epoch{epoch}", exist_ok=True
            )
            save_fsdp_model(
                model,
                rank,
                f"{args.save_path}/llama_checkpoint_epoch{epoch}",
                "fullstate",
            )

    # Cleanup
    dist.destroy_process_group()


if __name__ == "__main__":
    # Env vars auto-set by torchrun
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    main(rank, world_size)
