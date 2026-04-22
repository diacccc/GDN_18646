import argparse
import math
import torch
from einops import rearrange
from torch.utils.cpp_extension import load
from torch.nn import functional as F


def load_extension(source_file: str):
    module_name = "gdn_prefill"
    return load(
        name=module_name,
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
    chunk_size=None,
):
    """
    Chunked reference: implements the same per-token recurrence as the
    unchunked ref_prefill, computed chunk-by-chunk via WY decomposition.

    Per-token recurrence (identical to unchunked ref_prefill):
        S     ← g_t * S                          # 1. gate state
        old_v ← k_t^T @ S                        # 2. retrieve
        new_v ← beta_t * v_t + (1-beta_t)*old_v  # 3. interpolate
        S     ← S + m_t * k_t @ (new_v - old_v)  # 4. rank-1 update
        o_t   ← m_t * scale * q_t^T @ S          # 5. output
    """
    B, H, T, K = q.shape
    BT = chunk_size or T
    V  = v.shape[-1]

    if scale == 0.0:
        scale = 1.0 / math.sqrt(K)

    # --- Gate computation (identical to unchunked ref_prefill) ---
    x    = a.float() + dt_bias.float()                            # [B, H, T]
    A_log_bht = A_log.float().view(1, H, 1)
    g    = torch.exp(-torch.exp(A_log_bht) * F.softplus(x))  # [B, H, T]
    beta = torch.sigmoid(b_logits.float())                        # [B, H, T]

    # Reshape mask for broadcasting: [B, T] -> [B, 1, T, 1]
    m = mask[:, None, :, None].float()                            # [B, 1, T, 1]

    q, k, v, beta, g = map(
        lambda x: x.contiguous().to(torch.float32),
        [q, k, v, beta, g]
    )
    m = m.expand(-1, H, -1, -1)                                   # [B, H, T, 1]

    # log g for numerically stable cumulative products within chunks
    log_g = g.log()                                               # [B, H, T]

    # --- Pad sequence to a multiple of BT ---
    pad_len = (BT - (T % BT)) % BT
    if pad_len > 0:
        q     = F.pad(q,     (0, 0, 0, pad_len))
        k     = F.pad(k,     (0, 0, 0, pad_len))
        v     = F.pad(v,     (0, 0, 0, pad_len))
        beta  = F.pad(beta,  (0, pad_len))
        log_g = F.pad(log_g, (0, pad_len))
        m     = F.pad(m,     (0, 0, 0, pad_len))

    b, h, l, d_k = q.shape

    q      = q * scale                    # absorb scale into q (as in unchunked)
    # beta-weighted key and masked value — mirror steps 3 & 4 of the recurrence:
    #   new_v - old_v = beta*(v - old_v), so rank-1 key is k*beta, value weight is beta*v
    k_beta = k * beta[..., None] * m      # [B, H, T, K]  (m zeros masked positions)
    v_beta = v * beta[..., None] * m      # [B, H, T, V]

    # --- WY decomposition — computed in parallel over all chunks at once ---
    # Reshape to [B, H, n_chunks, BT, d]
    causal_diag  = torch.triu(torch.ones(BT, BT, dtype=torch.bool, device=q.device), diagonal=0)
    causal_upper = torch.triu(torch.ones(BT, BT, dtype=torch.bool, device=q.device), diagonal=1)
    q, k, k_beta, v_beta, log_g = map(
        lambda x: rearrange(x, 'b h (n c) d -> b h n c d', c=BT),
        [q, k, k_beta, v_beta, log_g.unsqueeze(-1)],
    )

    # Cumulative log-gate within each chunk → used in place of the sequential
    # product g_0*g_1*…*g_t that appears in the unchunked recurrence
    log_g_cum = log_g.squeeze(-1).cumsum(-1)   # [B, H, n, BT]
    g_cum     = log_g_cum.exp()[..., None]      # [B, H, n, BT, 1]

    # G[t, s] = prod(g_{s+1} … g_t) = g_cum[t] / g_cum[s]  (intra-chunk gate ratio)
    # This is the factor by which a rank-1 update written at position s has decayed
    # by the time it is read at position t — equivalent to g_t * … * g_{s+1} in
    # the unchunked loop.
    G = ((log_g_cum.unsqueeze(-1) - log_g_cum.unsqueeze(-2)).tril().exp()).tril()
    # [B, H, n, BT, BT]

    # WY matrix: encodes the triangular intra-chunk delta-rule correction so that
    # (new_v - old_v) at each position accounts for all earlier updates in the chunk
    WY = -((k_beta @ k.transpose(-1, -2)) * G).masked_fill(causal_diag, 0)
    for i in range(1, BT):
        WY[..., i, :i] = (WY[..., i, :i].clone()
                          + (WY[..., i, :i, None].clone()
                             * WY[..., :i, :i].clone()).sum(-2))
    WY = WY + torch.eye(BT, dtype=torch.float, device=q.device)

    # WY @ v_beta  →  effective (new_v - old_v) for each position, intra-chunk only
    new_v_minus_old_v_intra = WY @ v_beta          # [B, H, n, BT, V]

    # WY @ (k_beta * g_cum)  →  used to subtract the cross-chunk S contribution
    # from old_v, so the total (new_v - old_v) is correct when S ≠ 0
    k_beta_gcum = WY @ (k_beta * g_cum)            # [B, H, n, BT, K]

    # --- Sequential loop over chunks — carries S forward, same role as the ---
    # --- timestep loop in unchunked ref_prefill                             ---
    # State S [B, H, K, V] — same convention as unchunked (K-first)
    if state_in is not None:
        S = state_in.float()                        # [B, H, K, V]
    else:
        S = q.new_zeros(b, h, d_k, V)

    o = torch.zeros_like(new_v_minus_old_v_intra)
    for i in range(l // BT):
        q_i = q[:, :, i]       # [B, H, BT, K]
        k_i = k[:, :, i]

        # 2. old_v — cross-chunk portion: what S (entering this chunk, already
        #    cumulatively gated per-position) contributes to the retrieval k^T @ S
        old_v = k_beta_gcum[:, :, i] @ S           # [B, H, BT, V]

        # 3 & 4. new_v - old_v, now corrected for the cross-chunk S contribution
        new_v_minus_old_v = new_v_minus_old_v_intra[:, :, i] - old_v

        # 5. output = m * scale * q^T @ S (cross-chunk) + intra-chunk attention
        #    The m*scale factor is already absorbed into q (via q*=scale) and m
        #    (via k_beta/v_beta), so the masks appear implicitly here.
        o_cross = (q_i * log_g_cum[:, :, i, :, None].exp()) @ S
        attn_i  = (q_i @ k_i.transpose(-1, -2) * G[:, :, i]).masked_fill_(causal_upper, 0)
        o[:, :, i] = o_cross + attn_i @ new_v_minus_old_v

        # 1 & 4. Gate S then accumulate rank-1 updates from every position in chunk
        S = (S * log_g_cum[:, :, i, -1, None, None].exp()
             + (k_i * (log_g_cum[:, :, i, -1, None] - log_g_cum[:, :, i]).exp()[..., None])
             .transpose(-1, -2) @ new_v_minus_old_v)

    o = rearrange(o, 'b h n c d -> b h (n c) d')  # [B, H, T+pad, V]
    o = o[:, :, :T]

    # 5. Apply output mask — mirrors: output[:, t] = m_t * o  in unchunked
    o = o * m[:, :, :T, :]

    output = o  # [B, H, T, V]
    state_out = S               # [B, H, K, V]
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
    chunk_size = int(ext.C_DIM)
    if T % chunk_size != 0:
        print(f"B={B:>2} T={T:>5} skipped (T must be divisible by C={chunk_size})")
        return

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
                chunk_size=chunk_size,
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
    parser = argparse.ArgumentParser(description="GDN chunked prefill comparison test")
    parser.add_argument("filename", type=str, nargs='?', default="prefill_chunked.cu", help="CUDA source filename to compile")
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