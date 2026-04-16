import math
import torch
import torch.nn.functional as F


def matmul(a: torch.Tensor, b: torch.Tensor):
    """Float32 matmul for numerical stability."""
    return a.float() @ b.float()


@torch.no_grad()
def run(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
    """
    Gated Delta Net prefill reference implementation (k-last layout).

    State layout: [H, V, K] (k-last, K dimension at the end)
    """
    total_seq_len, num_q_heads, head_size = q.shape
    num_v_heads = v.shape[1]
    num_k_heads = k.shape[1]
    num_sab_heads = max(num_q_heads, num_v_heads)
    num_seqs = cu_seqlens.size(0) - 1
    device = q.device

    assert num_q_heads == 8
    assert num_k_heads == 8
    assert num_v_heads == 16
    assert head_size == 128

    if scale is None or scale == 0.0:
        scale = 1.0 / math.sqrt(head_size)

    # Compute g and beta from raw parameters
    x = a.float() + dt_bias.float()                              # [S, H]
    g = torch.exp(-torch.exp(A_log.float()) * F.softplus(x))    # [S, H]
    beta = torch.sigmoid(b.float())                              # [S, H]

    q_exp = q.repeat_interleave(num_v_heads // num_q_heads, dim=1)  # [S, H, K]
    k_exp = k.repeat_interleave(num_v_heads // num_k_heads, dim=1)  # [S, H, K]

    # ------------------------------------------------------------------ #
    # Compute per-sequence lengths and the maximum for padding            #
    # ------------------------------------------------------------------ #
    seq_lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()   # list of B ints
    max_len = max(seq_lengths)

    # ------------------------------------------------------------------ #
    # Pack flat [S, H, *] tensors into padded [B, T, H, *] tensors       #
    # ------------------------------------------------------------------ #
    def pad_to_batch(flat, max_len):
        """Scatter flat [S, ...] into padded [B, T, ...] (zeros for padding)."""
        B = num_seqs
        shape = flat.shape[1:]
        out = flat.new_zeros(B, max_len, *shape)
        for seq_idx in range(B):
            s = int(cu_seqlens[seq_idx].item())
            e = int(cu_seqlens[seq_idx + 1].item())
            out[seq_idx, : e - s] = flat[s:e]
        return out

    # [B, T, H, K/V]
    q_b = pad_to_batch(q_exp, max_len)
    k_b = pad_to_batch(k_exp, max_len)
    v_b = pad_to_batch(v,     max_len)
    g_b = pad_to_batch(g,     max_len)    # [B, T, H]
    beta_b = pad_to_batch(beta, max_len)  # [B, T, H]

    # Padding mask: True where the token is real, False where padded
    # [B, T]
    mask = torch.zeros(num_seqs, max_len, dtype=torch.bool, device=device)
    for seq_idx in range(num_seqs):
        mask[seq_idx, : seq_lengths[seq_idx]] = True

    # ------------------------------------------------------------------ #
    # Initialise state: [B, H, K, V]                                     #
    # ------------------------------------------------------------------ #
    if state is not None:
        # Input state is [B, H, V, K] (k-last); transpose to [B, H, K, V]
        state_BHKV = state.clone().float().transpose(-1, -2)
    else:
        state_BHKV = torch.zeros(
            (num_seqs, num_sab_heads, head_size, head_size),
            dtype=torch.float32, device=device
        )

    # Output buffer in padded layout: [B, T, H, V]
    output = torch.zeros(
        (num_seqs, max_len, num_sab_heads, head_size),
        dtype=torch.bfloat16, device=device
    )

    # ------------------------------------------------------------------ #
    # Token loop — all sequences processed in parallel                   #
    # ------------------------------------------------------------------ #
    for i in range(max_len):
        # active[b] is True when sequence b has a real token at step i
        active = mask[:, i]                          # [B]

        # Slice token i for every sequence: [B, H, K/V]
        q_BHK = q_b[:, i].float()                   # [B, H, K]
        k_BHK = k_b[:, i].float()                   # [B, H, K]
        v_BHV = v_b[:, i].float()                   # [B, H, V]
        g_BH  = g_b[:, i].float()                   # [B, H]
        beta_BH = beta_b[:, i].float()               # [B, H]

        # Reshape scalars for broadcasting against [B, H, K, V]
        g_BH11    = g_BH.unsqueeze(-1).unsqueeze(-1)     # [B, H, 1, 1]
        beta_BH11 = beta_BH.unsqueeze(-1).unsqueeze(-1)  # [B, H, 1, 1]

        # Reshape tokens for batched matmul: add a "1" sequence dim
        k_BH1K = k_BHK.unsqueeze(2)   # [B, H, 1, K]
        q_BH1K = q_BHK.unsqueeze(2)   # [B, H, 1, K]
        v_BH1V = v_BHV.unsqueeze(2)   # [B, H, 1, V]

        # Delta rule (same math as the original, now batched over B)
        old_state_BHKV = g_BH11 * state_BHKV                   # [B, H, K, V]
        old_v_BH1V     = matmul(k_BH1K, old_state_BHKV)        # [B, H, 1, V]
        new_v_BH1V     = beta_BH11 * v_BH1V + (1 - beta_BH11) * old_v_BH1V  # [B, H, 1, V]

        # kᵀ @ v  —  einsum over the 1-length "sequence" dim
        state_remove = torch.einsum('bhkl,bhlv->bhkv', k_BH1K.transpose(-1, -2), old_v_BH1V)
        state_update = torch.einsum('bhkl,bhlv->bhkv', k_BH1K.transpose(-1, -2), new_v_BH1V)
        new_state_BHKV = old_state_BHKV - state_remove + state_update  # [B, H, K, V]

        # Only update state for sequences that have a real token at step i
        active_BHKV = active.view(-1, 1, 1, 1).expand_as(state_BHKV)
        state_BHKV = torch.where(active_BHKV, new_state_BHKV, state_BHKV)

        # Output
        o_BH1V = scale * matmul(q_BH1K, state_BHKV)            # [B, H, 1, V]
        output[:, i] = o_BH1V.squeeze(2).to(torch.bfloat16)  # [B, H, V]

    return output