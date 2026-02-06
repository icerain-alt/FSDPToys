"""
Initialize shared global CUDA streams for FSDP (Fully Sharded Data Parallel) to reduce memory fragmentation.

Background:
Torch allocates a separate memory pool for each CUDA stream, which can lead to severe memory fragmentation
when using multiple FSDP model instances with independent streams.Streams are initialized only once and shared across all FSDP instances.
"""

from typing import no_type_check
from torch.distributed.fsdp._common_utils import _FSDPState
from torch.distributed.fsdp._init_utils import HYBRID_SHARDING_STRATEGIES

COMPUTE_STREAM = None
UNSHARD_STREAM = None
POST_BACKWARD_STREAM = None
PRE_UNSHARD_STREAM = None
ALL_REDUCE_STREAM = None


@no_type_check
def _init_streams(
    state: _FSDPState,
) -> None:
    """
    Initializes CUDA streams for overlapping communication, computation, and
    data transfers. The streams should be shared across FSDP instances.
    """
    assert state._is_root
    assert state._device_handle.is_available()

    global \
        COMPUTE_STREAM, \
        UNSHARD_STREAM, \
        POST_BACKWARD_STREAM, \
        PRE_UNSHARD_STREAM, \
        ALL_REDUCE_STREAM

    uses_hybrid_sharding = any(
        fsdp_state.sharding_strategy in HYBRID_SHARDING_STRATEGIES
        for fsdp_state in state._all_fsdp_states
    )
    # Prioritize all-gathers/reduce-scatters over async all-reduce for HSDP and
    # preserve the default priority of 0 otherwise
    high_priority = -1 if state.limit_all_gathers and uses_hybrid_sharding else 0

    # Default stream for computation
    if COMPUTE_STREAM is None:
        COMPUTE_STREAM = state._device_handle.current_stream()
    state._default_stream = COMPUTE_STREAM

    if state._fsdp_extension is not None:
        # set the compute stream to the FSDP extension
        state._fsdp_extension.compute_stream = state._default_stream

    # Stream for unshard logic, including allocating the all-gather destination
    # tensors and the all-gathers themselves
    if UNSHARD_STREAM is None:
        UNSHARD_STREAM = state._device_handle.Stream(priority=high_priority)
    state._unshard_stream = UNSHARD_STREAM

    # Stream for overlapping gradient reduction with the backward pass gradient
    # computation
    if POST_BACKWARD_STREAM is None:
        POST_BACKWARD_STREAM = state._device_handle.Stream(priority=high_priority)
    state._post_backward_stream = POST_BACKWARD_STREAM

    # Stream for pre-unshard logic, namely allocations and writes for CPU
    # offloading (H2D copy) and mixed precision (low precision cast)
    if PRE_UNSHARD_STREAM is None:
        PRE_UNSHARD_STREAM = state._device_handle.Stream(priority=high_priority)
    state._pre_unshard_stream = PRE_UNSHARD_STREAM

    # Stream to run HSDP's all-reduce as async (if using HSDP)
    if ALL_REDUCE_STREAM is None:
        ALL_REDUCE_STREAM = (
            state._device_handle.Stream()
            if uses_hybrid_sharding
            else state._default_stream
        )
    state._all_reduce_stream = ALL_REDUCE_STREAM


def apply_fsdp_patch():
    """
    Applies the patch to FSDP to initialize shared global CUDA streams.
    """
    from torch.distributed.fsdp import _runtime_utils

    _runtime_utils._init_streams = _init_streams
