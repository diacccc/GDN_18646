"""
PyTorch reference implementation of Proj + Causal Depthwise Conv1D + SiLU.
Used as ground-truth for correctness testing of the fused CUDA kernel.
"""

import torch
import torch.nn.functional as F

@torch.no_grad()
def ref_proj(input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Reference: Linear projection only (for ablation studies)."""
    return input @ weight.T

@torch.no_grad()
def ref_conv1d(input: torch.Tensor, conv_w: torch.Tensor) -> torch.Tensor:
    """Reference: Causal depthwise conv1d only (for ablation studies)."""
    D_out = conv_w.size(0)
    K = conv_w.size(1)
    input_t = input.transpose(1, 2)                                  # [B, D_out, T]
    input_padded = F.pad(input_t, (K - 1, 0))                       # [B, D_out, T + K-1]
    conv_weight = conv_w.unsqueeze(1)                                  # [D_out, 1, K]
    conv_out = F.conv1d(input_padded, conv_weight, groups=D_out)   # [B, D_out, T]
    conv_out = conv_out.transpose(1, 2)                            # [B, T, D_out]
    return conv_out

@torch.no_grad()
def ref_silu(input: torch.Tensor) -> torch.Tensor:
    """Reference: SiLU activation only (for ablation studies)."""
    return F.silu(input)


@torch.no_grad()
def ref_proj_conv_silu(input: torch.Tensor,
                       weight: torch.Tensor,
                       conv_w: torch.Tensor) -> torch.Tensor:
    """
    Reference: Linear projection -> causal depthwise conv1d -> SiLU.

    Args:
        input  : [B, T, D]        bfloat16
        weight : [D_out, D]       bfloat16
        conv_w : [D_out, CONV_K]  bfloat16   (CONV_K = 4)

    Returns:
        output : [B, T, D_out]    bfloat16
    """
    dtype = input.dtype
    # Upcast to float32 for reference computation
    x  = input
    w  = weight
    cw = conv_w

    # 1. Linear projection
    proj = x @ w.T                                                 # [B, T, D_out]

    # 2. Causal depthwise conv1d
    D_out = w.size(0)
    K = cw.size(1)
    proj_t = proj.transpose(1, 2)                                  # [B, D_out, T]
    proj_padded = F.pad(proj_t, (K - 1, 0))                       # [B, D_out, T + K-1]
    conv_weight = cw.unsqueeze(1)                                  # [D_out, 1, K]
    conv_out = F.conv1d(proj_padded, conv_weight, groups=D_out)   # [B, D_out, T]
    conv_out = conv_out.transpose(1, 2)                            # [B, T, D_out]

    # 3. SiLU activation
    output = F.silu(conv_out)
    return output.to(dtype)