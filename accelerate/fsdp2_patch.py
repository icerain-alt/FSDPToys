import logging
from typing import Any

import torch
import torch.nn as nn
from torch.profiler import record_function
from torch.distributed.fsdp._fully_shard._fsdp_common import (
    compiled_autograd_enabled,
    TrainingState,
)

logger = logging.getLogger("torch.distributed.fsdp.fully_shard")

COMM_CONTEXT = None


def is_bw() -> bool:
    return torch._C._current_graph_task_id() != -1


def copy_fsdp_comm_only_stream(src_comm_ctx, dst_comm_ctx):
    dst_comm_ctx.device_handle = src_comm_ctx.device_handle

    dst_comm_ctx.all_gather_copy_in_stream = src_comm_ctx.all_gather_copy_in_stream
    dst_comm_ctx.all_gather_stream = src_comm_ctx.all_gather_stream
    dst_comm_ctx.reduce_scatter_stream = src_comm_ctx.reduce_scatter_stream
    dst_comm_ctx.all_reduce_stream = src_comm_ctx.all_reduce_stream

    dst_comm_ctx.all_gather_state = None
    dst_comm_ctx.reduce_scatter_state = None
    dst_comm_ctx.post_forward_order = []

    return dst_comm_ctx


def _init_shared_state(self) -> None:
    # communication context is shared among all FSDP instances
    global COMM_CONTEXT
    if COMM_CONTEXT is not None:
        self._comm_ctx = copy_fsdp_comm_only_stream(COMM_CONTEXT, self._comm_ctx)
    else:
        self._comm_ctx.lazy_init(self._device)
        COMM_CONTEXT = self._comm_ctx
    for state in self._state_ctx.all_states:
        state._state_ctx = self._state_ctx
        state._comm_ctx = self._comm_ctx
        if fsdp_param_group := state._fsdp_param_group:
            fsdp_param_group.comm_ctx = self._comm_ctx


def post_forward(self, module: nn.Module, input: Any, output: Any):
    if not compiled_autograd_enabled():
        logger.debug("%s", self._with_fqn("FSDP::post_forward"))

    with record_function(self._with_fqn("FSDP::post_forward")):
        if not compiled_autograd_enabled():
            # for AC(fully_shard(model)), AC runs fsdp's _pre_forward
            # it shouldn't change post_forward_order
            if not is_bw():
                self.reshard()
                self._record_post_forward()
        else:
            self.reshard()
            self._record_post_forward()
        self._training_state = TrainingState.IDLE
        return output


def apply_fully_shard_patch():
    from torch.distributed.fsdp._fully_shard._fsdp_state import FSDPState
    from torch.distributed.fsdp._fully_shard._fsdp_param_group import FSDPParamGroup

    FSDPState._init_shared_state = _init_shared_state
    FSDPParamGroup.post_forward = post_forward
