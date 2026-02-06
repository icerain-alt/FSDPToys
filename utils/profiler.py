import torch
from utils import is_torch_npu_available


def build_profiler(profile_path, with_stack=False, with_memory=True):
    if is_torch_npu_available():
        import torch_npu
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
            record_shapes=True,
            profile_memory=with_memory,
            with_stack=with_stack,
            experimental_config=experimental_config,
            schedule=torch_npu.profiler.schedule(
                wait=0, warmup=1, active=1, repeat=1, skip_first=10
            ),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(profile_path)
        )
    else:
        profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU, 
                torch.profiler.ProfilerActivity.CUDA
            ],
            record_shapes=True,
            profile_memory=with_memory,
            with_stack=with_stack,
            schedule=torch.profiler.schedule(
                wait=0, warmup=1, active=1, repeat=1, skip_first=10
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(profile_path)
        )
        
    return profiler
