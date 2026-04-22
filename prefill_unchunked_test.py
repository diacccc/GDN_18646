import argparse
import math
import torch
from torch.utils.cpp_extension import load
from torch.nn import functional as F


def load_extension(source_file: str):
    return load(
        name="prefill_ext",
        sources=[source_file],
        verbose=True,
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "-lineinfo",
            "-arch=sm_80",
        ],
    )


@torch.no_grad()
def ref_prefill(
    q, k, v,
    A_log,
    a,
    dt_bias,
    b_logits,
    mask,
    state_in=None,
    scale=0.0,
) -> torch.Tensor:
    """
    Returns:
        output    [B, H, T, V]
        state_out [B, H, K, V]
    """
    B, H, T, K = q.shape
    V = v.shape[-1]

    if scale == 0.0:
        scale = 1.0 / math.sqrt(K)

    # --- Gate computation ---
    x    = a.float() + dt_bias.float()                            # [B, H, T]
    A_log_bht = A_log.float().view(1, H, 1)
    g    = torch.exp(-torch.exp(A_log_bht) * F.softplus(x))  # [B, H, T]
    beta = torch.sigmoid(b_logits.float())                        # [B, H, T]

    # Reshape mask for broadcasting against [B, H, T, V]
    mask_HV = mask[:, None, :, None].float()                      # [B, 1, T, 1]

    # --- Initial state [B, H, K, V] (K-first internally) ---
    if state_in is not None:
        S = state_in.float().clone()                              # [B, H, K, V]
    else:
        S = torch.zeros(B, H, K, V, dtype=torch.float32, device=q.device)

    output = torch.zeros(B, H, T, V, dtype=q.dtype, device=q.device)

    for t in range(T):
        g_t    = g   [:, :, t, None, None]   # [B, H, 1, 1]
        beta_t = beta[:, :, t, None, None]   # [B, H, 1, 1]
        m_t    = mask[:, t,  None, None, None].float()  # [B, 1, 1, 1]

        k_t = k[:, :, t, :, None].float()   # [B, H, K, 1]
        q_t = q[:, :, t, :, None].float()   # [B, H, K, 1]
        v_t = v[:, :, t, None, :].float()   # [B, H, 1, V]

        # 1. Gate
        S = g_t * S                                               # [B, H, K, V]

        # 2. old_v = k^T @ S  ->  [B, H, 1, V]
        old_v = k_t.transpose(-2, -1) @ S                        # [B, H, 1, V]

        # 3. new_v = beta * v + (1 - beta) * old_v
        new_v = beta_t * v_t + (1.0 - beta_t) * old_v           # [B, H, 1, V]

        # 4. S += k @ (new_v - old_v)  [rank-1 update]
        S = S + m_t * (k_t @ (new_v - old_v))                   # [B, H, K, V]

        # 5. output = scale * q^T @ S
        o = scale * (q_t.transpose(-2, -1) @ S)                  # [B, H, 1, V]
        output[:, :, t] = (m_t.squeeze(-1) * o.squeeze(-2)).to(q.dtype)

    state_out = S  # [B, H, K, V]
    return output, state_out


def bench_ms(fn, warmup=5, iters=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    out = fn()
    start.record()
    for _ in range(max(iters - 1, 0)):
        out = fn()
    end.record()
    torch.cuda.synchronize()
    return out, start.elapsed_time(end) / iters


def run_case(ext, B, T, Hh=16, dk=128, dv=256, dtype=torch.bfloat16):
    q = torch.randn(B, Hh, T, dk, device="cuda", dtype=dtype) * (dk ** -0.5)
    k = torch.randn(B, Hh, T, dk, device="cuda", dtype=dtype) * (dk ** -0.5)
    v = torch.randn(B, Hh, T, dv, device="cuda", dtype=dtype) * (dk ** -0.5)
    a = torch.randn(B, Hh, T, device="cuda", dtype=dtype)
    b_logits = torch.randn(B, Hh, T, device="cuda", dtype=dtype)

    A_log = torch.zeros(Hh, device="cuda", dtype=torch.float32)
    dt_bias = torch.tensor(0.0, device="cuda", dtype=torch.float32)
    mask = torch.ones(B, T, device="cuda", dtype=torch.float32)
    state_in_ext = torch.empty(0, device="cuda", dtype=torch.float32)
    scale = 1.0

    with torch.no_grad():
        ref_ret, t_ref = bench_ms(
            lambda: ref_prefill(
                q.contiguous(),
                k.contiguous(),
                v.contiguous(),
                A_log.contiguous(),
                a.contiguous(),
                dt_bias.contiguous(),
                b_logits.contiguous(),
                mask.contiguous(),
                state_in=None,
                scale=scale,
            )
        )
        o_ref, state_ref = ref_ret

        custom_ret, t_custom = bench_ms(
            lambda: ext.prefill(
                q.contiguous(),
                k.contiguous(),
                v.contiguous(),
                A_log.contiguous(),
                a.contiguous(),
                float(dt_bias.item()),
                b_logits.contiguous(),
                mask.contiguous(),
                state_in_ext,
                scale,
            )
        )
        o_custom, state_custom = custom_ret

        max_err = (o_custom.float() - o_ref.float()).abs().max().item()

    print(
        f"B={B:>2} T={T:>5} H={Hh:>3} dtype={str(dtype).split('.')[-1]:>9} "
        f"max_err={max_err:.3e} "
        f"custom={t_custom:8.4f} ms "
        f"ref={t_ref:8.4f} ms"
    )

    del q, k, v, a, b_logits, A_log, dt_bias, mask, state_in_ext, state_ref, state_custom


def parse_args():
    parser = argparse.ArgumentParser(description="GDN unchunked prefill comparison test")
    parser.add_argument("filename", type=str, nargs='?', default="prefill_unchunked.cu", help="CUDA source filename to compile")
    parser.add_argument("--B", type=int, nargs='+', default=[1, 8, 32, 64], help="Batch size")
    parser.add_argument("--T", type=int, nargs='+', default=[64, 1024, 2048, 4096], help="Sequence length")
    return parser.parse_args()


def main():
    if not torch.cuda.is_available():
        print("CUDA is required to run this test.")
        return

    args = parse_args()
    ext = load_extension(args.filename)
    for B in args.B:
        for T in args.T:
            run_case(ext, B, T)

if __name__ == "__main__":
    main()
