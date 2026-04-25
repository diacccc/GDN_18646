"""
bench_fused_proj_conv_silu.py
Throughput / latency benchmark for the fused Proj + Conv1D + SiLU kernel.

Standalone:
    python -m bench.bench_fused_proj_conv_silu

Via Modal:
    modal run modal_app.py --mode bench --kernel fused_proj_conv_silu
"""

import sys
import os
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reference.fused_proj_conv_silu_ref import *


def _compile_extension():
    from torch.utils.cpp_extension import load

    kernel_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "kernels")
    ext = load(
        name="fused_proj_conv_silu_ext",
        sources=[
            os.path.join(kernel_dir, "fused_proj_conv_silu.cu"),
        ],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "-gencode=arch=compute_80,code=sm_80",
            "--expt-relaxed-constexpr",
            "-lineinfo",
        ],
        verbose=True,
    )
    return ext


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
    """Sweep (B, T) configs and print timing + speedup."""
    if ext is None:
        ext = _compile_extension()

    D      = 2048
    D_out  = 2048
    conv_k = 4

    print("=" * 90)
    print(f"Benchmark: D={D}, D_out={D_out}, conv_k={conv_k}")
    print("=" * 90)
    print(f"{'B':>4}  {'T':>6}  {'cuda (ms)':>10}  {'ref (ms)':>10}  {'speedup':>8}")
    print("-" * 50)

    for B in [1, 8, 32, 64]:
        for T in [1, 2048, 4096, 8192]:
            x      = torch.randn(B, T, D, device="cuda", dtype=torch.bfloat16) * 0.02
            weight = torch.randn(D_out, D, device="cuda", dtype=torch.bfloat16) * (D ** -0.5)
            conv_w = torch.randn(D_out, conv_k, device="cuda", dtype=torch.bfloat16) * 0.5

            t_cuda = bench_ms(
                lambda: ext.forward(x.contiguous(), weight.contiguous(), conv_w.contiguous()))
            t_ref  = bench_ms(
                lambda: ref_proj_conv_silu(x, weight, conv_w))

            print(f"{B:>4}  {T:>6}  {t_cuda:>10.4f}  {t_ref:>10.4f}  {t_ref/t_cuda:>7.2f}x")

            del x, weight, conv_w
            torch.cuda.empty_cache()

    # V-path benchmark (D_out = 4096)
    D_out_v = 4096
    print()
    print("=" * 90)
    print(f"Benchmark (V path): D={D}, D_out={D_out_v}, conv_k={conv_k}")
    print("=" * 90)
    print(f"{'B':>4}  {'T':>6}  {'cuda (ms)':>10}  {'ref (ms)':>10}  {'speedup':>8}")
    print("-" * 50)

    for B in [1, 8, 32, 64]:
        for T in [1, 2048, 4096, 8192]:
            x      = torch.randn(B, T, D, device="cuda", dtype=torch.bfloat16) * 0.02
            weight = torch.randn(D_out_v, D, device="cuda", dtype=torch.bfloat16) * (D ** -0.5)
            conv_w = torch.randn(D_out_v, conv_k, device="cuda", dtype=torch.bfloat16) * 0.5

            t_cuda = bench_ms(
                lambda: ext.forward(x.contiguous(), weight.contiguous(), conv_w.contiguous()))
            t_ref  = bench_ms(
                lambda: ref_proj_conv_silu(x, weight, conv_w))

            # t_ref_proj = bench_ms(
            #     lambda: ref_proj(x, weight))
            # x_proj = ref_proj(x, weight)  # reuse for conv and silu benchmarks
            # t_ref_conv = bench_ms(
            #     lambda: ref_conv1d(x_proj, conv_w))
            # y = ref_conv1d(x_proj, conv_w)  # reuse for silu benchmark
            # t_ref_silu = bench_ms(
            #     lambda: ref_silu(y))

            print(f"{B:>4}  {T:>6}  {t_cuda:>10.4f}  {t_ref:>10.4f}  {t_ref/t_cuda:>7.2f}x")
            # print(f"       Breakdown (ms): proj={t_ref_proj:.4f}  conv={t_ref_conv:.4f}  silu={t_ref_silu:.4f}")

            del x, weight, conv_w
            torch.cuda.empty_cache()


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available — exiting.")
        raise SystemExit(1)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("Compiling CUDA extension …\n")
    run_bench()