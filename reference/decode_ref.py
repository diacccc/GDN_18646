"""
PyTorch reference implementation for the GDN decode recurrence (T=1).
"""

import math
import torch
import torch.nn.functional as F


@torch.no_grad()
def ref_decode(q, k, v, state, A_log, a_param, dt_bias, b_param, scale=None):
    """
    Gated Delta Net decode reference (single token, T=1).

    Args:
        q      : [B, 1, H, K]  bfloat16
        k      : [B, 1, H, K]  bfloat16
        v      : [B, 1, H, V]  bfloat16
        state  : [B, H, K, V]  float32   (K-first layout)
        A_log  : [H]           float32
        a_param: [B, 1, H]    bfloat16
        dt_bias: [H]          float32
        b_param: [B, 1, H]    bfloat16
        scale  : float or None

    Returns:
        output    : [B, H, V] float32
        new_state : [B, H, K, V] float32
    """
    B, T, H, K = q.shape
    V = v.shape[-1]

    if scale is None:
        scale = 1.0 / math.sqrt(K)

    x = a_param.float() + dt_bias.float()
    g = torch.exp(-torch.exp(A_log.float()) * F.softplus(x))      # [B, 1, H]
    beta = torch.sigmoid(b_param.float())                          # [B, 1, H]

    q_f32 = q.squeeze(1).float()     # [B, H, K]
    k_f32 = k.squeeze(1).float()
    v_f32 = v.squeeze(1).float()     # [B, H, V]
    g_f32 = g.squeeze(1).float()     # [B, H]
    beta_f32 = beta.squeeze(1).float()

    state_f32 = state.float().clone()  # [B, H, K, V]

    output = torch.zeros(B, H, V, dtype=torch.float32, device=q.device)
    new_state = torch.zeros_like(state_f32)

    for b in range(B):
        for h in range(H):
            g_val = g_f32[b, h]
            beta_val = beta_f32[b, h]
            k_h = k_f32[b, h]         # [K]
            q_h = q_f32[b, h]         # [K]
            v_h = v_f32[b, h]         # [V]
            S = state_f32[b, h]       # [K, V]

            # Gate
            old_S = g_val * S

            # old_v = k^T S  -> [V]
            old_v = k_h @ old_S

            # Delta rule
            new_v = beta_val * v_h + (1 - beta_val) * old_v

            # Rank-1 update
            S_new = old_S - k_h.unsqueeze(1) @ old_v.unsqueeze(0) \
                          + k_h.unsqueeze(1) @ new_v.unsqueeze(0)

            output[b, h] = scale * (q_h @ S_new)
            new_state[b, h] = S_new

    return output, new_state
