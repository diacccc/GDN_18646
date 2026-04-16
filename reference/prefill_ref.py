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
        q, k     : [B, T, H, K]  float32
        v        : [B, T, H, V]  float32
        A_log    : [H]           float32
        a        : [B, T, H]    float32
        dt_bias  : [H]          float32
        b_logits : [B, T, H]    float32
        mask     : [B, T]       float32
        state_in : [B, H, V, K] float32 or None
        scale    : float

    Returns:
        output   : [B, T, H, V] float32
    """
    B, T, H, K = q.shape
    V = v.shape[-1]

    if scale == 0.0:
        scale = 1.0 / math.sqrt(K)

    x    = a.float() + dt_bias.float()
    g    = torch.exp(-torch.exp(A_log.float()) * F.softplus(x))
    beta = torch.sigmoid(b_logits.float())

    mask_HV = mask[:, :, None, None].float()

    if state_in is not None:
        S = state_in.float().transpose(-1, -2).clone()   # [B, H, V, K] -> [B, H, K, V]
    else:
        S = torch.zeros(B, H, K, V, dtype=torch.float32, device=q.device)

    output = torch.zeros(B, T, H, V, dtype=q.dtype, device=q.device)

    for t in range(T):
        g_t    = g   [:, t, :, None, None]
        beta_t = beta[:, t, :, None, None]
        m_t    = mask[:, t,  None, None, None].float()

        k_t = k[:, t, :, :, None].float()
        q_t = q[:, t, :, :, None].float()
        v_t = v[:, t, :, None, :].float()

        S = g_t * S
        old_v = k_t.transpose(-2, -1) @ S
        new_v = beta_t * v_t + (1.0 - beta_t) * old_v
        S = S + m_t * (k_t @ (new_v - old_v))
        o = scale * (q_t.transpose(-2, -1) @ S)
        output[:, t] = (m_t.squeeze(-1) * o.squeeze(-2)).to(q.dtype)

    return output
