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
from reference.decode_ref               import ref_decode

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
        decode  = _load("gdn_decode",  "decode.cu"),
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


def report_diff(name, actual, expected, atol=1e-2, rtol=0.05):
    diff = (actual.float() - expected.float()).abs()
    nan_count = torch.isnan(actual).sum().item() + torch.isnan(expected).sum().item()
    max_err = diff.max().item()
    mean_err = diff.mean().item()
    passed = nan_count == 0 and bool((diff < atol + rtol * expected.float().abs()).all())
    print(f"  {name:<12} max={max_err:.3e}  mean={mean_err:.3e}  "
          f"nan={nan_count:<6} {'PASS' if passed else 'FAIL'}")
    return passed


def report_stats(name, x):
    xf = x.float()
    finite = torch.isfinite(xf)
    if finite.any():
        vals = xf[finite]
        print(f"  {name:<12} absmax={vals.abs().max().item():.3e}  "
              f"mean={vals.mean().item():.3e}  std={vals.std().item():.3e}")
    else:
        print(f"  {name:<12} no finite values")


def debug_prefill_layer(baseline, custom, x, mask, chunk_size):
    B, T, _ = x.shape
    w, exts = baseline.w, custom.exts
    scale = DK ** -0.5

    print("Prefill diagnostic:")
    xn_base = ref_rmsnorm(x.reshape(B*T, D), w['norm_gamma']).reshape(B, T, D)
    xn_custom = exts['rmsnorm'].forward(
        x.reshape(B*T, D), w['norm_gamma'], 1e-6, 256, 2
    ).reshape(B, T, D)
    report_diff("RMSNorm", xn_custom, xn_base, atol=5e-2, rtol=0.02)

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

    print("Activation stats:")
    report_stats("q", q_b)
    report_stats("k", k_b)
    report_stats("v", v_b)
    report_stats("a", a_b)
    report_stats("b", b_b)

    report_diff("Q proj", q_c, q_b, atol=5e-2, rtol=0.1)
    report_diff("K proj", k_c, k_b, atol=5e-2, rtol=0.1)
    report_diff("V proj", v_c, v_b, atol=5e-2, rtol=0.1)
    report_diff("a gate", a_c, a_b)
    report_diff("b gate", b_c, b_b)

    empty_state = torch.empty(0, device=x.device, dtype=torch.float32)
    o_base, _ = ref_prefill(
        q_b, k_b, v_b, w['A_log'], a_b, w['dt_bias'], b_b, mask,
        state_in=None, scale=scale, chunk_size=chunk_size,
    )
    o_custom, _ = exts['prefill'].prefill(
        q_c, k_c, v_c, w['A_log'], a_c, float(w['dt_bias'].item()),
        b_c, mask, empty_state, scale,
    )
    report_diff("Prefill", o_custom, o_base)

    y_base = o_base.to(x.dtype).transpose(1, 2).reshape(B, T, H*DV) @ w['W_o'].T
    y_custom = o_custom.transpose(1, 2).reshape(B, T, H*DV) @ w['W_o'].T
    report_diff("Output", y_custom, y_base, atol=5e-2, rtol=0.1)
    print()


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
    atol, rtol = 5e-2, 0.1
    passed  = bool((diff < atol + rtol * o_base.float().abs()).all())

    print(f"Correctness check (B={B}, T={T}):  max_err={max_err:.3e}  "
          f"{'PASS' if passed else 'FAIL'}\n")
    if not passed:
        debug_prefill_layer(baseline, custom, x, mask, chunk_size)
    return passed


def check_decode_correctness(baseline, custom, B=2, steps=8):
    state_base = baseline.init_decode_state(B)
    state_custom = custom.init_decode_state(B)

    max_err = 0.0
    max_state_err = 0.0
    passed = True
    atol, rtol = 1e-2, 0.05

    for _ in range(steps):
        x = torch.randn(B, 1, D, device="cuda", dtype=torch.bfloat16)
        o_base, state_base = baseline.decode(x, state_base)
        o_custom, state_custom = custom.decode(x, state_custom)

        diff = (o_custom.float() - o_base.float()).abs()
        max_err = max(max_err, diff.max().item())
        passed &= bool((diff < atol + rtol * o_base.float().abs()).all())

        state_diff = (state_custom['recurrent'] - state_base['recurrent']).abs()
        max_state_err = max(max_state_err, state_diff.max().item())
        passed &= bool(state_diff.max().item() < 1e-2)

    print(f"Decode correctness (B={B}, steps={steps}):  "
          f"max_err={max_err:.3e}  max_state_err={max_state_err:.3e}  "
          f"{'PASS' if passed else 'FAIL'}\n")
    return passed


def check_decode_matches_prefill(layer, name, B=1, T=64):
    x = torch.randn(B, T, D, device="cuda", dtype=torch.bfloat16)
    mask = torch.ones(B, T, device="cuda", dtype=torch.float32)

    o_prefill = layer.forward(x, mask)
    state = layer.init_decode_state(B)
    outs = []
    for t in range(T):
        o_t, state = layer.decode(x[:, t:t+1], state)
        outs.append(o_t)
    o_decode = torch.cat(outs, dim=1)

    diff = (o_decode.float() - o_prefill.float()).abs()
    max_err = diff.max().item()
    atol, rtol = 1e-2, 0.05
    passed = bool((diff < atol + rtol * o_prefill.float().abs()).all())

    print(f"{name} decode vs prefill (B={B}, T={T}):  "
          f"max_err={max_err:.3e}  {'PASS' if passed else 'FAIL'}")
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

    q_dec_ref = q_b[:, :, :1, :].transpose(1, 2).contiguous()
    k_dec_ref = k_b[:, :, :1, :].transpose(1, 2).contiguous()
    v_dec_ref = v_b[:, :, :1, :].transpose(1, 2).contiguous()
    a_dec_ref = a_b[:, :, :1].transpose(1, 2).contiguous()
    b_dec_ref = b_b[:, :, :1].transpose(1, 2).contiguous()

    q_dec = q_c[:, :, 0, :].contiguous()
    k_dec = k_c[:, :, 0, :].contiguous()
    v_dec = v_c[:, :, 0, :].contiguous()
    a_dec = a_c[:, :, 0].contiguous()
    b_dec = b_c[:, :, 0].contiguous()
    S_in = torch.randn(B, H, DK, DV, device="cuda", dtype=torch.float32) * 0.01
    o_dec = torch.empty(B, H, DV, device="cuda", dtype=torch.float32)
    S_dec = torch.empty_like(S_in)
    dt_bias_vec = (w['dt_bias'].reshape(1).expand(H).contiguous()
                   if w['dt_bias'].numel() == 1 else w['dt_bias'].contiguous())

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
                                state_in=None, scale=scale,
                                chunk_size=int(exts['prefill'].C_DIM)),
            lambda: exts['prefill'].prefill(q_c, k_c, v_c, w['A_log'], a_c,
                                            float(w['dt_bias'].item()), b_c, mask,
                                            empty_state, scale)),
        ("Decode",
            lambda: ref_decode(q_dec_ref, k_dec_ref, v_dec_ref, S_in, w['A_log'],
                               a_dec_ref, dt_bias_vec, b_dec_ref, scale),
            lambda: exts['decode'].forward(q_dec, k_dec, v_dec, a_dec, b_dec,
                                           w['A_log'], dt_bias_vec, S_in, scale,
                                           o_dec, S_dec)),
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

    for B in [1, 8, 32, 64]:
        for T in [2048, 4096, 8192]:
            if T % chunk_size != 0:
                continue
            x    = torch.randn(B, T, D, device="cuda", dtype=torch.bfloat16)
            mask = torch.ones(B, T, device="cuda", dtype=torch.float32)

            t_base   = bench_ms(lambda: baseline.forward(x, mask))
            t_custom = bench_ms(lambda: custom.forward(x, mask))
            print(f"{B:>4}  {T:>6}  {t_base:>14.4f}  {t_custom:>12.4f}  {t_base/t_custom:>7.2f}x")
        print()


def bench_decode_e2e(baseline, custom):
    print(f"{'B':>4}  {'baseline (ms)':>14}  {'custom (ms)':>12}  {'speedup':>8}")
    print(f"  {'-'*43}")

    for B in [1, 8, 32, 64, 128, 256]:
        x = torch.randn(B, 1, D, device="cuda", dtype=torch.bfloat16)
        state_base = baseline.init_decode_state(B)
        state_custom = custom.init_decode_state(B)

        t_base = bench_ms(lambda: baseline.decode(x, state_base), warmup=5, iters=20)
        t_custom = bench_ms(lambda: custom.decode(x, state_custom), warmup=10, iters=50)
        print(f"{B:>4}  {t_base:>14.4f}  {t_custom:>12.4f}  {t_base/t_custom:>7.2f}x")
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
    custom   = GDNLayerCustom(w, exts)
    chunk    = custom.chunk
    baseline = GDNLayerBaseline(w, chunk_size=chunk)

    # ── Correctness ──────────────────────────────────────────────────────────
    print("=" * 60)
    print("Correctness")
    print("=" * 60)
    check_correctness(baseline, custom, chunk)
    check_decode_correctness(baseline, custom)
    check_decode_matches_prefill(baseline, "Baseline", T=chunk)
    check_decode_matches_prefill(custom, "Custom", T=chunk)
    print()

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

    # ── Decode sweep ─────────────────────────────────────────────────────────
    print("=" * 60)
    print(f"Decode end-to-end  H={H}  DK={DK}  DV={DV}")
    print("=" * 60)
    bench_decode_e2e(baseline, custom)


if __name__ == "__main__":
    main()
