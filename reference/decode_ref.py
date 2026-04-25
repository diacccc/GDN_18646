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
    assert T == 1, f"ref_decode expects T=1, got T={T}"

    if scale is None:
        scale = 1.0 / math.sqrt(K)

    a_scalar = a_param.squeeze(1)
    b_scalar = b_param.squeeze(1)

    raw = a_scalar.float() + dt_bias.float()                     # [B, H]
    g = torch.exp(-torch.exp(A_log.float()) * F.softplus(raw))   # [B, H]
    beta = torch.sigmoid(b_scalar.float())                       # [B, H]

    q_f32 = q.squeeze(1).float()                                 # [B, H, K]
    k_f32 = k.squeeze(1).float()                                 # [B, H, K]
    v_f32 = v.squeeze(1).float()                                 # [B, H, V]
    new_state = state.float().clone()                            # [B, H, K, V]

    # Gated decay.
    new_state = new_state * g[:, :, None, None]

    # Read.
    r = torch.einsum('bhk,bhkd->bhd', k_f32, new_state)          # [B, H, V]

    # Delta and rank-1 update.
    delta = beta[:, :, None] * (v_f32 - r)                       # [B, H, V]
    new_state = new_state + torch.einsum('bhk,bhd->bhkd', k_f32, delta)

    # Output.
    output = scale * torch.einsum('bhk,bhkd->bhd', q_f32, new_state)

    return output, new_state
