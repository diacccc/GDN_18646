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
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reference.prefill_ref import ref_prefill


def _compile_extension():
    from torch.utils.cpp_extension import load

    kernel_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "kernels")
    return load(
        name="prefill_ext",
        sources=[os.path.join(kernel_dir, "prefill_chunked.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo", "-arch=sm_80"],
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

    chunk_size = int(ext.C_DIM)
    Hh    = 16
    dk    = 128
    dv    = 256
    scale = 1.0
    dtype = torch.bfloat16

    print("=" * 80)
    print(f"Benchmark: Prefill  H={Hh}  dk={dk}  dv={dv}  chunk_size={chunk_size}")
    print("=" * 80)
    print(f"{'B':>4}  {'T':>6}  {'cuda (ms)':>10}  {'ref (ms)':>10}  {'speedup':>8}")
    print("-" * 50)

    for B in [1, 8, 32, 64]:
        for T in [2048, 4096, 8192]:
            if T % chunk_size != 0:
                print(f"{B:>4}  {T:>6}  skipped (T not divisible by chunk_size={chunk_size})")
                continue

            q        = torch.randn(B, Hh, T, dk, device="cuda", dtype=dtype) * (dk ** -0.5)
            k        = torch.randn(B, Hh, T, dk, device="cuda", dtype=dtype) * (dk ** -0.5)
            v        = torch.randn(B, Hh, T, dv, device="cuda", dtype=dtype) * (dk ** -0.5)
            a        = torch.randn(B, Hh, T, device="cuda", dtype=dtype)
            b_logits = torch.randn(B, Hh, T, device="cuda", dtype=dtype)
            A_log    = torch.zeros(Hh, device="cuda", dtype=torch.float32)
            dt_bias  = torch.tensor(0.0, device="cuda", dtype=torch.float32)
            mask     = torch.ones(B, T, device="cuda", dtype=torch.float32)
            state_in_ext = torch.empty(0, device="cuda", dtype=torch.float32)

            def cuda_fn():
                return ext.prefill(
                    q.contiguous(), k.contiguous(), v.contiguous(),
                    A_log.contiguous(), a.contiguous(),
                    float(dt_bias.item()),
                    b_logits.contiguous(), mask.contiguous(),
                    state_in_ext, scale,
                )

            def ref_fn():
                return ref_prefill(
                    q.contiguous(), k.contiguous(), v.contiguous(),
                    A_log.contiguous(), a.contiguous(), dt_bias.contiguous(),
                    b_logits.contiguous(), mask.contiguous(),
                    state_in=None, scale=scale,
                    chunk_size=chunk_size,
                )

            t_cuda = bench_ms(cuda_fn)
            t_ref  = bench_ms(ref_fn)

            print(f"{B:>4}  {T:>6}  {t_cuda:>10.4f}  {t_ref:>10.4f}  {t_ref/t_cuda:>7.2f}x")

            del q, k, v, a, b_logits, A_log, dt_bias, mask, state_in_ext
            torch.cuda.empty_cache()
        print()


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available — exiting.")
        raise SystemExit(1)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("Compiling CUDA extension …\n")
    run_bench()
