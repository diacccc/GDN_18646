"""
PyTorch reference implementation for the GDN prefill recurrence.
"""

import math
import torch
import torch.nn.functional as F


@torch.no_grad()
def ref_prefill(q, k, v, A_log, a, dt_bias, b_logits, mask,
                state_in=None, scale=0.0):
    """
    Gated Delta Net prefill reference (element-wise recurrence).

    Args:
        q, k     : [B, H, T, K]  float32
        v        : [B, H, T, V]  float32
        A_log    : [H]           float32
        a        : [B, H, T]    float32
        dt_bias  : scalar float or scalar tensor
        b_logits : [B, H, T]    float32
        mask     : [B, T]       float32
        state_in : [B, H, K, V] float32 or None
        scale    : float

    Returns:
        output   : [B, H, T, V] float32
        state_out: [B, H, K, V] float32
    """
    B, H, T, K = q.shape
    V = v.shape[-1]

    if scale == 0.0:
        scale = 1.0 / math.sqrt(K)

    x         = a.float() + dt_bias.float()
    A_log_bht = A_log.float().view(1, H, 1)
    g         = torch.exp(-torch.exp(A_log_bht) * F.softplus(x))  # [B, H, T]
    beta      = torch.sigmoid(b_logits.float())                     # [B, H, T]

    if state_in is not None:
        S = state_in.float().clone()   # [B, H, K, V]
    else:
        S = torch.zeros(B, H, K, V, dtype=torch.float32, device=q.device)

    output = torch.zeros(B, H, T, V, dtype=q.dtype, device=q.device)

    for t in range(T):
        g_t    = g[:, :, t, None, None]            # [B, H, 1, 1]
        beta_t = beta[:, :, t, None, None]         # [B, H, 1, 1]
        m_t    = mask[:, t, None, None, None].float()  # [B, 1, 1, 1]

        k_t = k[:, :, t, :, None].float()          # [B, H, K, 1]
        q_t = q[:, :, t, :, None].float()          # [B, H, K, 1]
        v_t = v[:, :, t, None, :].float()          # [B, H, 1, V]

        S = g_t * S
        old_v = k_t.transpose(-2, -1) @ S          # [B, H, 1, V]
        new_v = beta_t * v_t + (1.0 - beta_t) * old_v
        S = S + m_t * (k_t @ (new_v - old_v))      # [B, H, K, V]
        o = scale * (q_t.transpose(-2, -1) @ S)    # [B, H, 1, V]
        output[:, :, t] = (m_t.squeeze(-1) * o.squeeze(-2)).to(q.dtype)

    return output, S
