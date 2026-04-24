"""
GDN (Gated Delta Net) layer: PyTorch-baseline and custom-CUDA-kernel versions.

Architecture (prefill mode, T tokens):
  Input [B, T, D]
    └─ RMSNorm
    └─ Q projection  (linear + causal-conv1d + SiLU)  → [B, H, T, DK]
    └─ K projection  (linear + causal-conv1d + SiLU)  → [B, H, T, DK]
    └─ V projection  (linear + causal-conv1d + SiLU)  → [B, H, T, DV]
    └─ a linear                                        → [B, H, T]  (gate)
    └─ b linear                                        → [B, H, T]  (beta)
    └─ Gated Delta-Net prefill recurrence
    └─ Output projection                               → [B, T, D]
"""

import torch

# ── Architecture constants (must match kernel compile-time constants) ──────────
D      = 2048   # model dimension
H      = 16     # number of heads
DK     = 128    # key/query dim per head  (H × DK = 2048)
DV     = 256    # value dim per head      (H × DV = 4096)
CONV_K = 4      # causal conv kernel size (hardcoded in fused_proj kernel)


def make_weights():
    """Randomly initialise a shared weight dict (CUDA, bfloat16 / float32)."""
    def randn_bf16(*shape):
        return torch.randn(*shape, dtype=torch.bfloat16, device="cuda") * shape[0] ** -0.5

    return dict(
        norm_gamma = torch.ones(D, dtype=torch.bfloat16, device="cuda"),

        W_q    = randn_bf16(H*DK, D),
        conv_q = torch.randn(H*DK, CONV_K, dtype=torch.bfloat16, device="cuda") * 0.5,
        W_k    = randn_bf16(H*DK, D),
        conv_k = torch.randn(H*DK, CONV_K, dtype=torch.bfloat16, device="cuda") * 0.5,
        W_v    = randn_bf16(H*DV, D),
        conv_v = torch.randn(H*DV, CONV_K, dtype=torch.bfloat16, device="cuda") * 0.5,

        W_a = randn_bf16(H, D),
        W_b = randn_bf16(H, D),

        A_log   = torch.zeros(H, dtype=torch.float32, device="cuda"),
        dt_bias = torch.tensor(0.0, dtype=torch.float32, device="cuda"),

        W_o = randn_bf16(D, H*DV),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Baseline: pure PyTorch / reference functions
# ─────────────────────────────────────────────────────────────────────────────
class GDNLayerBaseline:
    """Forward pass using pure PyTorch reference implementations."""

    def __init__(self, weights: dict):
        self.w     = weights
        self.scale = DK ** -0.5

    @torch.no_grad()
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """
        x    : [B, T, D]  bfloat16, CUDA
        mask : [B, T]     float32   (1 = keep, 0 = mask); defaults to all-ones
        Returns: [B, T, D] bfloat16
        """
        from reference.rmsnorm_ref              import ref_rmsnorm
        from reference.fused_proj_conv_silu_ref import ref_proj_conv_silu
        from reference.prefill_ref              import ref_prefill

        B, T, _ = x.shape
        w = self.w
        if mask is None:
            mask = x.new_ones(B, T, dtype=torch.float32)

        # 1. RMSNorm  [B*T, D] → [B, T, D]
        xn = ref_rmsnorm(x.reshape(B*T, D), w['norm_gamma']).reshape(B, T, D)

        # 2. Q / K / V projections  →  [B, H, T, D_head]
        def proj(W, cw, dh):
            out = ref_proj_conv_silu(xn, W, cw)            # [B, T, H*dh]
            return out.reshape(B, T, H, dh).transpose(1, 2).contiguous()

        q = proj(w['W_q'], w['conv_q'], DK)
        k = proj(w['W_k'], w['conv_k'], DK)
        v = proj(w['W_v'], w['conv_v'], DV)

        # 3. a / b gate projections  →  [B, H, T]
        a = (xn @ w['W_a'].T).transpose(1, 2).contiguous()
        b = (xn @ w['W_b'].T).transpose(1, 2).contiguous()

        # 4. Prefill recurrence  →  o [B, H, T, DV] float32
        o, _ = ref_prefill(
            q, k, v, w['A_log'], a, w['dt_bias'], b, mask,
            state_in=None, scale=self.scale,
        )

        # 5. Output projection  →  [B, T, D]
        return o.to(x.dtype).transpose(1, 2).reshape(B, T, H*DV) @ w['W_o'].T


# ─────────────────────────────────────────────────────────────────────────────
# Custom: compiled CUDA kernels
# ─────────────────────────────────────────────────────────────────────────────
class GDNLayerCustom:
    """Forward pass using the four compiled CUDA kernels."""

    def __init__(self, weights: dict, exts: dict):
        """
        weights : dict from make_weights()
        exts    : dict with keys 'rmsnorm', 'proj', 'prefill'
        """
        self.w     = weights
        self.exts  = exts
        self.scale = DK ** -0.5
        self.chunk = int(exts['prefill'].C_DIM)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """
        x    : [B, T, D]  bfloat16, CUDA   (T must be divisible by chunk_size)
        mask : [B, T]     float32
        Returns: [B, T, D] bfloat16
        """
        B, T, _ = x.shape
        w, e = self.w, self.exts
        if mask is None:
            mask = x.new_ones(B, T, dtype=torch.float32)

        # 1. RMSNorm  (kernel: [R, D] bf16 → bf16)
        xn = e['rmsnorm'].forward(
            x.reshape(B*T, D), w['norm_gamma'], 1e-6, 256, 2
        ).reshape(B, T, D)

        # 2. Q / K / V projections  →  [B, H, T, D_head]
        def proj(W, cw, dh):
            out = e['proj'].forward(
                xn.contiguous(), W.contiguous(), cw.contiguous()
            )                                               # [B, T, H*dh]
            return out.reshape(B, T, H, dh).transpose(1, 2).contiguous()

        q = proj(w['W_q'], w['conv_q'], DK)
        k = proj(w['W_k'], w['conv_k'], DK)
        v = proj(w['W_v'], w['conv_v'], DV)

        # 3. a / b gate projections  →  [B, H, T]  (plain matmul; no custom kernel)
        a = (xn @ w['W_a'].T).transpose(1, 2).contiguous()
        b = (xn @ w['W_b'].T).transpose(1, 2).contiguous()

        # 4. Prefill recurrence  →  o [B, H, T, DV]
        empty_state = torch.empty(0, device=x.device, dtype=torch.float32)
        o, _ = e['prefill'].prefill(
            q, k, v,
            w['A_log'], a, float(w['dt_bias'].item()),
            b, mask, empty_state, self.scale,
        )

        # 5. Output projection  →  [B, T, D]
        return o.transpose(1, 2).reshape(B, T, H*DV) @ w['W_o'].T
