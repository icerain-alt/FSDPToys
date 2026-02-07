from torch.utils.checkpoint import checkpoint


def gradient_checkpointing(module, *args, enabled, **kwargs):
    if enabled:
        return checkpoint(module, *args, use_reentrant=False, **kwargs)
    else:
        return module(*args, **kwargs)
