"""
bench_rmsnorm.py
Throughput / latency benchmark for the RMSNorm CUDA kernel.

Standalone:
    python -m bench.bench_rmsnorm

Via Modal:
    modal run modal_app.py --mode bench --kernel rmsnorm
"""

import sys
import os
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reference.rmsnorm_ref import ref_rmsnorm


class RMSNormFused(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.norm = nn.RMSNorm(dim, eps=eps, elementwise_affine=True,
                               device="cuda", dtype=torch.bfloat16)

    def forward(self, x):
        return self.norm(x)


def _compile_extension():
    from torch.utils.cpp_extension import load

    kernel_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "kernels")
    return load(
        name="rmsnorm_ext",
        sources=[os.path.join(kernel_dir, "rmsnorm.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
        verbose=True,
    )


def bench_ms(fn, warmup=20, iters=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def run_bench(ext=None):
    """Sweep (N, D) configs and print timing + speedup."""
    if ext is None:
        ext = _compile_extension()

    # Kernel is BF16-only and hardcoded to D=2048
    D   = 2048
    eps = 1e-6

    torch_rms = RMSNormFused(D, eps=eps).eval()
    torch.nn.init.ones_(torch_rms.norm.weight)  # match gamma=ones

    print("=" * 90)
    print("Benchmark: RMSNorm BF16  (N = B*T rows, D=2048)")
    print("=" * 90)
    print(f"{'N':>8}  {'cuda (ms)':>10}  {'ref (ms)':>10}  {'torch (ms)':>11}  {'vs ref':>7}  {'vs torch':>9}")
    print("-" * 70)

    gamma = torch.ones(D, device="cuda", dtype=torch.bfloat16)
    for B in [1, 8, 32, 64]:
        for T in [1, 2048, 4096, 8192]:
            N = B * T
            x = torch.randn(N, D, device="cuda", dtype=torch.bfloat16)

            t_cuda  = bench_ms(lambda: ext.forward(x.contiguous(), gamma, eps))
            t_ref   = bench_ms(lambda: ref_rmsnorm(x.contiguous(), gamma, eps))
            def run_fuse():
                with torch.inference_mode():
                    torch_rms(x)
            t_torch = bench_ms(run_fuse)

            print(f"{N:>8}  {t_cuda:>10.4f}  {t_ref:>10.4f}  {t_torch:>11.4f}  {t_ref/t_cuda:>6.2f}x  {t_torch/t_cuda:>8.2f}x")

            del x
            torch.cuda.empty_cache()

    del gamma, torch_rms
    torch.cuda.empty_cache()
    print()


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available — exiting.")
        raise SystemExit(1)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("Compiling CUDA extension …\n")
    run_bench()
