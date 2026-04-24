"""
PyTorch reference implementation for the GDN prefill recurrence.
"""

import math
import torch
import torch.nn.functional as F
from einops import rearrange


@torch.no_grad()
def ref_prefill(q, k, v, A_log, a, dt_bias, b_logits, mask,
                state_in=None, scale=0.0, chunk_size=None):
    """
    Chunked reference: implements the same per-token recurrence as the
    unchunked ref_prefill, computed chunk-by-chunk via WY decomposition.

    Per-token recurrence:
        S     <- g_t * S                          # 1. gate state
        old_v <- k_t^T @ S                        # 2. retrieve
        new_v <- beta_t * v_t + (1-beta_t)*old_v  # 3. interpolate
        S     <- S + m_t * k_t @ (new_v - old_v)  # 4. rank-1 update
        o_t   <- m_t * scale * q_t^T @ S          # 5. output

    Args:
        q, k     : [B, H, T, K]  bfloat16 or float32
        v        : [B, H, T, V]  bfloat16 or float32
        A_log    : [H]           float32
        a        : [B, H, T]    bfloat16 or float32
        dt_bias  : scalar float or scalar tensor
        b_logits : [B, H, T]    bfloat16 or float32
        mask     : [B, T]       float32
        state_in : [B, H, K, V] float32 or None
        scale    : float  (0.0 → 1/sqrt(K))
        chunk_size: int or None  (None → single chunk over full T)

    Returns:
        output   : [B, H, T, V]  same dtype as q
        state_out: [B, H, K, V]  float32
    """
    B, H, T, K = q.shape
    BT = chunk_size or T
    V  = v.shape[-1]

    if scale == 0.0:
        scale = 1.0 / math.sqrt(K)

    # --- Gate computation ---
    x         = a.float() + dt_bias.float()                     # [B, H, T]
    A_log_bht = A_log.float().view(1, H, 1)
    g         = torch.exp(-torch.exp(A_log_bht) * F.softplus(x))  # [B, H, T]
    beta      = torch.sigmoid(b_logits.float())                    # [B, H, T]

    # Reshape mask for broadcasting: [B, T] -> [B, H, T, 1]
    m = mask[:, None, :, None].float()                          # [B, 1, T, 1]

    q, k, v, beta, g = map(
        lambda x: x.contiguous().to(torch.float32),
        [q, k, v, beta, g]
    )
    m = m.expand(-1, H, -1, -1)                                 # [B, H, T, 1]

    # log g for numerically stable cumulative products within chunks
    log_g = g.log()                                             # [B, H, T]

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

    q      = q * scale
    k_beta = k * beta[..., None] * m      # [B, H, T, K]
    v_beta = v * beta[..., None] * m      # [B, H, T, V]

    # --- WY decomposition — computed in parallel over all chunks at once ---
    causal_diag  = torch.triu(torch.ones(BT, BT, dtype=torch.bool, device=q.device), diagonal=0)
    causal_upper = torch.triu(torch.ones(BT, BT, dtype=torch.bool, device=q.device), diagonal=1)
    q, k, k_beta, v_beta, log_g = map(
        lambda x: rearrange(x, 'b h (n c) d -> b h n c d', c=BT),
        [q, k, k_beta, v_beta, log_g.unsqueeze(-1)],
    )

    log_g_cum = log_g.squeeze(-1).cumsum(-1)   # [B, H, n, BT]
    g_cum     = log_g_cum.exp()[..., None]      # [B, H, n, BT, 1]

    G = ((log_g_cum.unsqueeze(-1) - log_g_cum.unsqueeze(-2)).tril().exp()).tril()
    # [B, H, n, BT, BT]

    WY = -((k_beta @ k.transpose(-1, -2)) * G).masked_fill(causal_diag, 0)
    for i in range(1, BT):
        WY[..., i, :i] = (WY[..., i, :i].clone()
                          + (WY[..., i, :i, None].clone()
                             * WY[..., :i, :i].clone()).sum(-2))
    WY = WY + torch.eye(BT, dtype=torch.float, device=q.device)

    new_v_minus_old_v_intra = WY @ v_beta          # [B, H, n, BT, V]
    k_beta_gcum = WY @ (k_beta * g_cum)            # [B, H, n, BT, K]

    # --- Sequential loop over chunks ---
    if state_in is not None:
        S = state_in.float()                        # [B, H, K, V]
    else:
        S = q.new_zeros(b, h, d_k, V)

    o = torch.zeros_like(new_v_minus_old_v_intra)
    for i in range(l // BT):
        q_i = q[:, :, i]       # [B, H, BT, K]
        k_i = k[:, :, i]

        old_v = k_beta_gcum[:, :, i] @ S           # [B, H, BT, V]
        new_v_minus_old_v = new_v_minus_old_v_intra[:, :, i] - old_v

        o_cross = (q_i * log_g_cum[:, :, i, :, None].exp()) @ S
        attn_i  = (q_i @ k_i.transpose(-1, -2) * G[:, :, i]).masked_fill_(causal_upper, 0)
        o[:, :, i] = o_cross + attn_i @ new_v_minus_old_v

        S = (S * log_g_cum[:, :, i, -1, None, None].exp()
             + (k_i * (log_g_cum[:, :, i, -1, None] - log_g_cum[:, :, i]).exp()[..., None])
             .transpose(-1, -2) @ new_v_minus_old_v)

    o = rearrange(o, 'b h n c d -> b h (n c) d')  # [B, H, T+pad, V]
    o = o[:, :, :T]
    o = o * m[:, :, :T, :]

    return o, S
