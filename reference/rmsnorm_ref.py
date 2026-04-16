"""
PyTorch reference implementation for RMSNorm.
"""

import torch


@torch.no_grad()
def ref_rmsnorm(x: torch.Tensor, gamma: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Reference RMSNorm: y = x * rsqrt(mean(x^2) + eps) * gamma

    Args:
        x     : [R, D]  float32
        gamma : [D]     float32
        eps   : float

    Returns:
        y     : [R, D]  float32
    """
    x_fp32 = x.float()
    rms = x_fp32.pow(2).mean(dim=-1, keepdim=True)
    x_normed = x_fp32 * torch.rsqrt(rms + eps)
    out = x_normed.to(x.dtype) * gamma
    return out
