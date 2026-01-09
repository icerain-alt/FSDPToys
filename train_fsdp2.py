import os
import time
import argparse
import torch
import torchvision.datasets as datasets
import torchvision.transforms as T
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy, CPUOffloadPolicy
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    set_model_state_dict,
)

from llama_model import Transformer, TransformerBlock, ModelArgs
from utils import (
    format_metrics_to_gb,
    print_model_info,
    seed_all,
    save_fsdp2_model,
    load_fsdp2_model,
)

import torch_npu
from torch_npu.contrib import transfer_to_npu

def get_args():
    parser = argparse.ArgumentParser(description="PyTorch llama FSDP2 Example")

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
        "--cpu_offload",
        type=bool,
        default=False,
        help="Offload model params, grads and optimizer states to CPU",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        type=bool,
        default=False,
        help="Enable gradient checkpointing",
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
        default="ckpts/fsdp2/llama_checkpoint",
        help="Checkpoint loading path",
    )
    parser.add_argument(
        "--save_path", type=str, default="ckpts/fsdp2", help="Checkpoint saving path"
    )
    parser.add_argument(
        "--save_freq",
        type=int,
        default=1,
        help="Checkpoint saving frequency (in epochs)",
    )
    parser.add_argument(
        "--profile",
        type=bool,
        default=False,
        help="NPU profiling for performance analysis",
    )
    parser.add_argument(
        "--profile_path",
        type=str,
        default="profile/llama_7b_fsdp2_base",
        help="NPU profiling path",
    )

    args = parser.parse_args()

    seed_all(args.seed, mode=False)

    return args


def train_one_epoch(model, loader, optimizer, epoch, rank, args):
    model.train()
    total_loss = 0.0

    if args.profile:
        experimental_config = torch_npu.profiler._ExperimentalConfig(
            aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
            l2_cache=False,
            data_simplification=True,
        )
        profiler = torch_npu.profiler.profile(
            activities=[
                torch_npu.profiler.ProfilerActivity.CPU,
                torch_npu.profiler.ProfilerActivity.NPU,
            ],
            record_shapes=False,
            profile_memory=True,
            with_stack=False,
            experimental_config=experimental_config,
            schedule=torch_npu.profiler.schedule(
                wait=0, warmup=1, active=1, repeat=1, skip_first=10
            ),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                args.profile_path,
            ),
        )
        profiler.start()

    for batch_idx, (inputs, labels) in enumerate(loader):
        t0 = time.time()
        inputs = inputs.reshape(-1, args.seq_len)

        outputs = model(inputs)
        loss = outputs.mean()

        loss.backward()

        # TODO: fsdp2 cpu offload not support clip grad norm
        # https://github.com/pytorch/pytorch/issues/148532
        if not args.cpu_offload:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

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
        num_classes=10,
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
        vocab_size=32000,
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
    load_fsdp2_model(model, rank, args.load_path, "fullstate")
    full_state_dict = model.state_dict()

    settings = dict(
        mesh=mesh_2d,
        mp_policy=MixedPrecisionPolicy(
            param_dtype=torch.bfloat16, reduce_dtype=torch.float32
        ),
        offload_policy=CPUOffloadPolicy() if args.cpu_offload else None,
    )
    for module in model.modules():
        if isinstance(module, TransformerBlock):
            fully_shard(module, **settings)
    fully_shard(model, **settings)

    # Loads the full state dict (could be only on rank 0) into the sharded model
    options = StateDictOptions(
        full_state_dict=True, cpu_offload=args.cpu_offload, broadcast_from_rank0=True
    )
    set_model_state_dict(model, full_state_dict, options=options)
    del full_state_dict

    # Print model info on rank 0
    if rank == 0:
        print_model_info(model)
        print("\n" + "=" * 50, f"{'FSDP Info':^10}", "=" * 50)
        for name, param in model.named_parameters():
            print(
                f"Param name = {name}, shape = {param.size()}, dtype = {param.dtype}, requires_grad = {param.requires_grad}",
                flush=True,
            )

    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, fused=True)

    # Training loop
    for epoch in range(args.num_epochs):
        train_sampler.set_epoch(epoch)
        train_one_epoch(model, train_loader, optimizer, epoch, rank, args)

        if (epoch + 1) % args.save_freq == 0:
            os.makedirs(
                f"{args.save_path}/llama_checkpoint_epoch{epoch}", exist_ok=True
            )
            save_fsdp2_model(
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
