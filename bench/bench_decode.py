"""
bench_decode.py
Throughput / latency benchmark for the decode recurrence CUDA kernel.

Standalone:
    python -m bench.bench_decode

Via Modal:
    modal run modal_app.py --mode bench --kernel decode
"""

import sys
import os
import math
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reference.decode_ref import ref_decode


def _compile_extension():
    from torch.utils.cpp_extension import load

    kernel_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "kernels")
    return load(
        name="decode_ext",
        sources=[os.path.join(kernel_dir, "decode.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
        verbose=True,
    )


def bench_ms(fn, warmup=20, iters=200):
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
    """Sweep batch sizes and print timing + speedup."""
    if ext is None:
        ext = _compile_extension()

    H     = 16
    dk    = 128
    dv    = 256
    scale = 1.0 / math.sqrt(dk)

    print("=" * 70)
    print(f"Benchmark: Decode (T=1)  H={H}  dk={dk}  dv={dv}")
    print("=" * 70)
    print(f"{'B':>6}  {'cuda (ms)':>10}  {'ref (ms)':>10}  {'speedup':>8}")
    print("-" * 45)

    for B in [1, 8, 32, 64, 128, 256]:
        q       = torch.randn(B, 1, H, dk, device="cuda", dtype=torch.bfloat16) * 0.02
        k       = torch.randn(B, 1, H, dk, device="cuda", dtype=torch.bfloat16) * 0.02
        v       = torch.randn(B, 1, H, dv, device="cuda", dtype=torch.bfloat16) * 0.02
        a_param = torch.randn(B, 1, H, device="cuda", dtype=torch.bfloat16) * 0.1
        b_param = torch.randn(B, 1, H, device="cuda", dtype=torch.bfloat16) * 0.1
        A_log   = torch.zeros(H, device="cuda", dtype=torch.float32)
        dt_bias = torch.zeros(H, device="cuda", dtype=torch.float32)
        S_in    = torch.randn(B, H, dk, dv, device="cuda", dtype=torch.float32) * 0.01

        q_k = q.squeeze(1).reshape(B, H, dk).contiguous()
        k_k = k.squeeze(1).reshape(B, H, dk).contiguous()
        v_k = v.squeeze(1).reshape(B, H, dv).contiguous()
        a_k = a_param.squeeze(1).reshape(B, H).contiguous()
        b_k = b_param.squeeze(1).reshape(B, H).contiguous()
        o_out = torch.zeros(B, H, dv, device="cuda", dtype=torch.float32)
        S_out = torch.zeros_like(S_in)

        def cuda_fn():
            ext.forward(
                q_k, k_k, v_k, a_k, b_k,
                A_log, dt_bias, S_in, scale, o_out, S_out,
            )

        def ref_fn():
            ref_decode(q, k, v, S_in, A_log, a_param, dt_bias, b_param, scale)

        t_cuda = bench_ms(cuda_fn)
        t_ref  = bench_ms(ref_fn)

        print(f"{B:>6}  {t_cuda:>10.4f}  {t_ref:>10.4f}  {t_ref/t_cuda:>7.2f}x")

        del q, k, v, a_param, b_param, A_log, dt_bias, S_in, q_k, k_k, v_k, a_k, b_k, o_out, S_out
        torch.cuda.empty_cache()


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available — exiting.")
        raise SystemExit(1)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("Compiling CUDA extension …\n")
    run_bench()
