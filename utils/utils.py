import os
import random
import torch
import numpy as np
from torch.distributed.tensor import DTensor
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
    FullStateDictConfig,  # general model non-sharded, non-flattened params
    ShardedStateDictConfig,  # un-flattened param but shards, usable by other parallel schemes.
)
from torch.utils.checkpoint import checkpoint


g_gigabyte = 1024**3


def is_torch_npu_available() -> bool:
    """Check if Ascend NPU is available for PyTorch operations.

    Attempts to detect NPU availability by checking for the torch.npu module
    and its is_available() function.

    Returns:
        bool: True if NPU is available, False otherwise.
    """
    try:
        if hasattr(torch, "npu") and callable(getattr(torch.npu, "is_available", None)):
            return torch.npu.is_available()
        return False
    except ImportError:
        return False


def print_rank0(msg):
    if torch.distributed.get_rank() == 0:
        print(msg, flush=True)


def seed_all(seed=1234, mode=False, is_npu=False):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(mode)

    if is_npu:
        import torch_npu

        torch_npu.npu.manual_seed_all(seed)
        torch_npu.npu.manual_seed(seed)
        os.environ["HCCL_DETERMINISTIC"] = str(mode)


def format_metrics_to_gb(item):
    """quick function to format numbers to gigabyte and round to 4 digit precision"""
    metric_num = item / g_gigabyte
    metric_num = round(metric_num, ndigits=4)
    return metric_num


def print_model_info(model):
    print("=" * 50, f"{'Model Structure':^10}", "=" * 50)
    print(model)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n" + "=" * 50, f"{'Parameter Info':^10}", "=" * 50)
    print(f"Total Parameters: {total_params / 1e6:.2f}M")
    print(f"Trainable Parameters: {trainable_params / 1e6:.2f}M")


def gradient_checkpointing(module, *args, enabled, **kwargs):
    if enabled:
        return checkpoint(module, *args, use_reentrant=False, **kwargs)
    else:
        return module(*args, **kwargs)


def fsdp2_clip_grad_norm_(parameters, max_norm, norm_type=2.0, error_if_nonfinite=False, foreach=None):
    """torch.nn.utils.clip_grad_norm_ cann't run on cpu parameter DTensor"""
    from torch.nn.utils.clip_grad import _clip_grads_with_norm_, _get_total_norm

    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    else:
        # prevent generators from being exhausted
        parameters = list(parameters)
    grads = [p.grad for p in parameters if p.grad is not None]
    total_norm = _get_total_norm(grads, norm_type, error_if_nonfinite, foreach)
    total_norm = total_norm.to(torch.cuda.current_device(), non_blocking=True)
    _clip_grads_with_norm_(parameters, max_norm, total_norm, foreach)
    return total_norm


def set_modules_to_forward_prefetch(model, num_to_forward_prefetch=1):
    # FSDP2: Explicit prefetching params for forward
    for i, layer in enumerate(model.layers):
        if i >= len(model.layers) - num_to_forward_prefetch:
            break
        layers_to_prefetch = [
            model.layers[i + j] for j in range(1, num_to_forward_prefetch + 1)
        ]
        layer.set_modules_to_forward_prefetch(layers_to_prefetch)


def set_modules_to_backward_prefetch(model, num_to_backward_prefetch=1):
    # FSDP2: Explicit prefetching params for backward
    for i, layer in enumerate(model.layers):
        if i < num_to_backward_prefetch:
            continue
        layers_to_prefetch = [
            model.layers[i - j] for j in range(1, num_to_backward_prefetch + 1)
        ]
        layer.set_modules_to_backward_prefetch(layers_to_prefetch)


def save_fsdp_model(model, rank, save_path, ckpt_type="fullstate"):
    if ckpt_type == "fullstate":
        with FSDP.state_dict_type(
            module=model,
            state_dict_type=StateDictType.FULL_STATE_DICT,
            state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            cpu_state = model.state_dict()
        if rank == 0:
            print("--> saving full model ...")
            torch.save(cpu_state, f"{save_path}/fullstate.pth")
    elif ckpt_type == "shardstate":
        with FSDP.state_dict_type(
            module=model,
            state_dict_type=StateDictType.SHARDED_STATE_DICT,
            state_dict_config=ShardedStateDictConfig(offload_to_cpu=True),
        ):
            state_dict = model.state_dict()
        print_rank0("--> saving sharded model ...")
        torch.save(state_dict, f"{save_path}/rank{rank}.pth")
    else:
        raise ValueError(f"Unknown checkpoint type: {ckpt_type}")

    print_rank0(f"model checkpoint saved at {save_path}\n")


def load_fsdp_model(model, rank, load_path, ckpt_type="fullstate"):
    if os.path.exists(load_path):
        if ckpt_type == "fullstate":
            if rank == 0:
                model.load_state_dict(
                    torch.load(f"{load_path}/fullstate.pth", map_location="cpu")
                )
                print_rank0("model checkpoint loaded to rank0 cpu")
        elif ckpt_type == "shardstate":
            with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
                model.load_state_dict(torch.load(f"{load_path}/rank{rank}.pth"))
            print_rank0(f"--> sharded state loaded on rank {rank}")
        else:
            raise ValueError(f"Unknown checkpoint type: {ckpt_type}")
    else:
        print_rank0(
            f"Checkpoint {load_path} does not exist. Starting training from scratch."
        )


def save_fsdp2_model(model, rank, save_path, ckpt_type="fullstate"):
    if ckpt_type == "fullstate":
        sharded_sd = model.state_dict()
        cpu_state_dict = {}
        for param_name, sharded_param in sharded_sd.items():
            if isinstance(sharded_param, DTensor):
                full_param = sharded_param.full_tensor()
            else:
                full_param = sharded_param
                print_rank0(f"UnSharded param: {param_name}")
            if rank == 0:
                cpu_state_dict[param_name] = full_param.cpu()
            else:
                del full_param
        if rank == 0:
            print("--> saving full model ...")
            torch.save(cpu_state_dict, f"{save_path}/fullstate.pth")

    elif ckpt_type == "shardstate":
        print_rank0("--> saving sharded model ...")
        torch.save(model.state_dict(), f"{save_path}/rank{rank}.pth")
    else:
        raise ValueError(f"Unknown checkpoint type: {ckpt_type}")

    print_rank0(f"model checkpoint saved at {save_path}\n")


def load_fsdp2_model(model, rank, load_path, ckpt_type="fullstate"):
    if os.path.exists(load_path):
        if ckpt_type == "fullstate":
            if rank == 0:
                model.load_state_dict(
                    torch.load(f"{load_path}/fullstate.pth", map_location="cpu")
                )
                print_rank0("model checkpoint loaded to rank0 cpu")
        elif ckpt_type == "shardstate":
            model.load_state_dict(torch.load(f"{load_path}/rank{rank}.pth"))
            print_rank0(f"--> sharded state loaded on rank {rank}")
        else:
            raise ValueError(f"Unknown checkpoint type: {ckpt_type}")
    else:
        print_rank0(
            f"Checkpoint {load_path} does not exist. Starting training from scratch."
        )
