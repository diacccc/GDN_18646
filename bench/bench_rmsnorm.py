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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reference.rmsnorm_ref import ref_rmsnorm


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

    eps = 1e-6

    print("=" * 70)
    print("Benchmark: RMSNorm  (N = B*T rows)")
    print("=" * 70)
    print(f"{'N':>8}  {'D':>6}  {'cuda (ms)':>10}  {'ref (ms)':>10}  {'speedup':>8}")
    print("-" * 55)

    for D in [1024, 2048, 4096]:
        gamma = torch.ones(D, device="cuda", dtype=torch.float32)
        for N in [1, 64, 512, 2048, 8192]:
            x = torch.randn(N, D, device="cuda", dtype=torch.float32)

            t_cuda = bench_ms(lambda: ext.forward(x.contiguous(), gamma, eps))
            t_ref  = bench_ms(lambda: ref_rmsnorm(x.contiguous(), gamma, eps))

            print(f"{N:>8}  {D:>6}  {t_cuda:>10.4f}  {t_ref:>10.4f}  {t_ref/t_cuda:>7.2f}x")

            del x
            torch.cuda.empty_cache()

        del gamma
        print()


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available — exiting.")
        raise SystemExit(1)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("Compiling CUDA extension …\n")
    run_bench()
