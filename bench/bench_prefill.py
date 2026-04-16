"""
bench_prefill.py
Throughput / latency benchmark for the prefill recurrence CUDA kernel.

Standalone:
    python -m bench.bench_prefill

Via Modal:
    modal run modal_app.py --mode bench --kernel prefill
"""

import sys
import os
import math
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reference.prefill_ref import ref_prefill


def _compile_extension():
    from torch.utils.cpp_extension import load

    kernel_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "kernels")
    return load(
        name="prefill_ext",
        sources=[os.path.join(kernel_dir, "prefill_unchunked.cu")],
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
    """Sweep (B, T) configs and print timing + speedup."""
    if ext is None:
        ext = _compile_extension()

    Hh   = 16
    dk   = 128
    dv   = 256
    scale = 1.0

    print("=" * 80)
    print(f"Benchmark: Prefill  H={Hh}  dk={dk}  dv={dv}")
    print("=" * 80)
    print(f"{'B':>4}  {'T':>6}  {'cuda (ms)':>10}  {'ref (ms)':>10}  {'speedup':>8}")
    print("-" * 50)

    for B in [1, 4, 8]:
        for T in [1, 64, 256, 512, 1024, 2048]:
            q        = torch.randn(B, T, Hh, dk, device="cuda", dtype=torch.float32) * (dk ** -0.5)
            k        = torch.randn(B, T, Hh, dk, device="cuda", dtype=torch.float32) * (dk ** -0.5)
            v        = torch.randn(B, T, Hh, dv, device="cuda", dtype=torch.float32) * (dk ** -0.5)
            a        = torch.randn(B, T, Hh, device="cuda", dtype=torch.float32)
            b_logits = torch.randn(B, T, Hh, device="cuda", dtype=torch.float32)
            A_log    = torch.zeros(Hh, device="cuda", dtype=torch.float32)
            dt_bias  = torch.zeros(Hh, device="cuda", dtype=torch.float32)
            mask     = torch.ones(B, T, device="cuda", dtype=torch.float32)
            state_in = torch.zeros(B, Hh, dv, dk, device="cuda", dtype=torch.float32)

            def cuda_fn():
                return ext.forward(
                    q.contiguous(), k.contiguous(), v.contiguous(),
                    A_log.contiguous(), a.contiguous(), dt_bias.contiguous(),
                    b_logits.contiguous(), mask.contiguous(),
                    state_in.contiguous(), scale,
                )

            def ref_fn():
                return ref_prefill(
                    q.contiguous(), k.contiguous(), v.contiguous(),
                    A_log.contiguous(), a.contiguous(), dt_bias.contiguous(),
                    b_logits.contiguous(), mask.contiguous(),
                    state_in=state_in, scale=scale,
                )

            t_cuda = bench_ms(cuda_fn)
            t_ref  = bench_ms(ref_fn)

            print(f"{B:>4}  {T:>6}  {t_cuda:>10.4f}  {t_ref:>10.4f}  {t_ref/t_cuda:>7.2f}x")

            del q, k, v, a, b_logits, A_log, dt_bias, mask, state_in
            torch.cuda.empty_cache()
        print()


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available — exiting.")
        raise SystemExit(1)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("Compiling CUDA extension …\n")
    run_bench()
