"""
bench_gdn.py — End-to-end GDN layer: custom CUDA kernels vs PyTorch baseline.

Usage:
    python bench_gdn.py
"""

import os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gdn_layer import GDNLayerBaseline, GDNLayerCustom, make_weights, D, H, DK, DV
from reference.rmsnorm_ref              import ref_rmsnorm
from reference.fused_proj_conv_silu_ref import ref_proj_conv_silu
from reference.prefill_ref              import ref_prefill

KERNEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels")


# ─────────────────────────────────────────────────────────────────────────────
# Kernel compilation
# ─────────────────────────────────────────────────────────────────────────────
def compile_kernels():
    from torch.utils.cpp_extension import load

    def _load(name, src, extra=()):
        return load(
            name=name,
            sources=[os.path.join(KERNEL_DIR, src)],
            extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo", *extra],
            verbose=False,
        )

    print("Compiling kernels …")
    exts = dict(
        rmsnorm = _load("gdn_rmsnorm", "rmsnorm.cu"),
        proj    = _load("gdn_proj",    "fused_proj_conv_silu.cu",
                        ("-gencode=arch=compute_80,code=sm_80", "--expt-relaxed-constexpr")),
        prefill = _load("gdn_prefill", "prefill_chunked.cu", ("-arch=sm_80",)),
    )
    print("Done.\n")
    return exts


# ─────────────────────────────────────────────────────────────────────────────
# Timing helper
# ─────────────────────────────────────────────────────────────────────────────
def bench_ms(fn, warmup=10, iters=50):
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


# ─────────────────────────────────────────────────────────────────────────────
# Correctness check
# ─────────────────────────────────────────────────────────────────────────────
def check_correctness(baseline, custom, chunk_size, B=2, T=128):
    assert T % chunk_size == 0, f"T={T} not divisible by chunk_size={chunk_size}"
    x    = torch.randn(B, T, D, device="cuda", dtype=torch.bfloat16)
    mask = torch.ones(B, T, device="cuda", dtype=torch.float32)

    o_base   = baseline.forward(x, mask)
    o_custom = custom.forward(x, mask)

    diff    = (o_custom.float() - o_base.float()).abs()
    max_err = diff.max().item()
    atol, rtol = 1e-2, 0.05
    passed  = bool((diff < atol + rtol * o_base.float().abs()).all())

    print(f"Correctness check (B={B}, T={T}):  max_err={max_err:.3e}  "
          f"{'PASS' if passed else 'FAIL'}\n")
    return passed


# ─────────────────────────────────────────────────────────────────────────────
# Per-operation breakdown
# ─────────────────────────────────────────────────────────────────────────────
def bench_ops(w, exts, B, T):
    """Print a per-kernel timing breakdown for one (B, T) config."""
    scale = DK ** -0.5
    x     = torch.randn(B, T, D, device="cuda", dtype=torch.bfloat16)
    mask  = torch.ones(B, T,    device="cuda", dtype=torch.float32)

    # -- Shared inputs computed once (not timed) -------------------------
    xn_base   = ref_rmsnorm(x.reshape(B*T, D), w['norm_gamma']).reshape(B, T, D)
    xn_custom = exts['rmsnorm'].forward(
        x.reshape(B*T, D), w['norm_gamma'], 1e-6, 256, 2
    ).reshape(B, T, D)

    def _proj_base(W, cw, dh):
        out = ref_proj_conv_silu(xn_base, W, cw)
        return out.reshape(B, T, H, dh).transpose(1, 2).contiguous()

    def _proj_custom(W, cw, dh):
        out = exts['proj'].forward(xn_custom.contiguous(), W.contiguous(), cw.contiguous())
        return out.reshape(B, T, H, dh).transpose(1, 2).contiguous()

    q_b = _proj_base(w['W_q'], w['conv_q'], DK)
    k_b = _proj_base(w['W_k'], w['conv_k'], DK)
    v_b = _proj_base(w['W_v'], w['conv_v'], DV)
    a_b = (xn_base @ w['W_a'].T).transpose(1, 2).contiguous()
    b_b = (xn_base @ w['W_b'].T).transpose(1, 2).contiguous()

    q_c = _proj_custom(w['W_q'], w['conv_q'], DK)
    k_c = _proj_custom(w['W_k'], w['conv_k'], DK)
    v_c = _proj_custom(w['W_v'], w['conv_v'], DV)
    a_c = (xn_custom @ w['W_a'].T).transpose(1, 2).contiguous()
    b_c = (xn_custom @ w['W_b'].T).transpose(1, 2).contiguous()
    empty_state = torch.empty(0, device="cuda", dtype=torch.float32)

    ops = [
        ("RMSNorm",
            lambda: ref_rmsnorm(x.reshape(B*T, D), w['norm_gamma']),
            lambda: exts['rmsnorm'].forward(x.reshape(B*T, D), w['norm_gamma'], 1e-6, 256, 2)),
        ("Q proj",
            lambda: ref_proj_conv_silu(xn_base, w['W_q'], w['conv_q']),
            lambda: exts['proj'].forward(xn_custom.contiguous(), w['W_q'].contiguous(), w['conv_q'].contiguous())),
        ("K proj",
            lambda: ref_proj_conv_silu(xn_base, w['W_k'], w['conv_k']),
            lambda: exts['proj'].forward(xn_custom.contiguous(), w['W_k'].contiguous(), w['conv_k'].contiguous())),
        ("V proj",
            lambda: ref_proj_conv_silu(xn_base, w['W_v'], w['conv_v']),
            lambda: exts['proj'].forward(xn_custom.contiguous(), w['W_v'].contiguous(), w['conv_v'].contiguous())),
        ("Prefill",
            lambda: ref_prefill(q_b, k_b, v_b, w['A_log'], a_b, w['dt_bias'], b_b, mask,
                                state_in=None, scale=scale),
            lambda: exts['prefill'].prefill(q_c, k_c, v_c, w['A_log'], a_c,
                                            float(w['dt_bias'].item()), b_c, mask,
                                            empty_state, scale)),
    ]

    print(f"  {'Op':<10}  {'baseline (ms)':>14}  {'custom (ms)':>12}  {'speedup':>8}")
    print(f"  {'-'*52}")
    for name, fn_base, fn_cust in ops:
        t_b = bench_ms(fn_base)
        t_c = bench_ms(fn_cust)
        print(f"  {name:<10}  {t_b:>14.4f}  {t_c:>12.4f}  {t_b/t_c:>7.2f}x")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end sweep
# ─────────────────────────────────────────────────────────────────────────────
def bench_e2e(baseline, custom, chunk_size):
    print(f"{'B':>4}  {'T':>6}  {'baseline (ms)':>14}  {'custom (ms)':>12}  {'speedup':>8}")
    print(f"  {'-'*52}")

    for B in [1, 4, 8]:
        for T in [64, 256, 512, 1024]:
            if T % chunk_size != 0:
                continue
            x    = torch.randn(B, T, D, device="cuda", dtype=torch.bfloat16)
            mask = torch.ones(B, T, device="cuda", dtype=torch.float32)

            t_base   = bench_ms(lambda: baseline.forward(x, mask))
            t_custom = bench_ms(lambda: custom.forward(x, mask))
            print(f"{B:>4}  {T:>6}  {t_base:>14.4f}  {t_custom:>12.4f}  {t_base/t_custom:>7.2f}x")
        print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    if not torch.cuda.is_available():
        print("CUDA not available."); raise SystemExit(1)

    print(f"GPU: {torch.cuda.get_device_name(0)}\n")
    exts = compile_kernels()

    w        = make_weights()
    baseline = GDNLayerBaseline(w)
    custom   = GDNLayerCustom(w, exts)
    chunk    = custom.chunk

    # ── Correctness ──────────────────────────────────────────────────────────
    print("=" * 60)
    print("Correctness")
    print("=" * 60)
    check_correctness(baseline, custom, chunk)

    # ── Per-op breakdown (B=1, T=256) ────────────────────────────────────────
    print("=" * 60)
    print(f"Per-operation breakdown  (B=1, T=256)")
    print("=" * 60)
    bench_ops(w, exts, B=1, T=256)

    # ── End-to-end sweep ─────────────────────────────────────────────────────
    print("=" * 60)
    print(f"End-to-end  H={H}  DK={DK}  DV={DV}  chunk={chunk}")
    print("=" * 60)
    bench_e2e(baseline, custom, chunk)


if __name__ == "__main__":
    main()
