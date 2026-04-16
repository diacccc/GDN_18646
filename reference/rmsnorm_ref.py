"""
PyTorch reference implementation for RMSNorm.
"""

import torch


@torch.no_grad()
def ref_rmsnorm(x: torch.Tensor, gamma: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Reference RMSNorm matching the BF16 CUDA kernel behaviour:
      - x and gamma are BF16
      - reduction and normalization are done in FP32 (mirrors __bfloat162float casts in kernel)
      - output is cast back to BF16 (mirrors __float2bfloat16_rn in kernel)

    Args:
        x     : [R, D]  bfloat16
        gamma : [D]     bfloat16
        eps   : float

    Returns:
        y     : [R, D]  bfloat16
    """
    x_fp32 = x.float()
    rms = x_fp32.pow(2).mean(dim=-1, keepdim=True)
    x_normed = x_fp32 * torch.rsqrt(rms + eps)
    out = x_normed * gamma.float()
    return out.to(torch.bfloat16)
