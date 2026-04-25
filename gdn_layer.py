"""
GDN (Gated Delta Net) layer: PyTorch-baseline and custom-CUDA-kernel versions.

Architecture (prefill mode, T tokens):
  Input [B, T, D]
    └─ RMSNorm
    └─ Q projection  (linear + causal-conv1d + SiLU)  -> [B, H, T, DK]
    └─ K projection  (linear + causal-conv1d + SiLU)  -> [B, H, T, DK]
    └─ V projection  (linear + causal-conv1d + SiLU)  -> [B, H, T, DV]
    └─ a linear                                        -> [B, H, T]  (gate)
    └─ b linear                                        -> [B, H, T]  (beta)
    └─ Gated Delta-Net prefill recurrence
    └─ Output projection                               -> [B, T, D]

Decode mode runs the same blocks for one token and carries:
  recurrent : [B, H, DK, DV] fp32
  conv_q/k/v: [B, H*D_head, CONV_K-1] bf16 projected-token history
  xn        : [B, CONV_K-1, D] bf16 normalized-input history for fused custom decode
"""

import torch

# ── Architecture constants (must match kernel compile-time constants) ──────────
D      = 2048   # model dimension
H      = 16     # number of heads
DK     = 128    # key/query dim per head  (H × DK = 2048)
DV     = 256    # value dim per head      (H × DV = 4096)
CONV_K = 4      # causal conv kernel size (hardcoded in fused_proj kernel)


def _zero_decode_state(weights: dict, B: int, device=None):
    """Create a zero decode cache for the recurrent state and causal convs."""
    device = device or weights['norm_gamma'].device
    return dict(
        recurrent=torch.zeros(B, H, DK, DV, dtype=torch.float32, device=device),
        conv_q=torch.zeros(B, H*DK, CONV_K-1, dtype=torch.bfloat16, device=device),
        conv_k=torch.zeros(B, H*DK, CONV_K-1, dtype=torch.bfloat16, device=device),
        conv_v=torch.zeros(B, H*DV, CONV_K-1, dtype=torch.bfloat16, device=device),
        xn=torch.zeros(B, CONV_K-1, D, dtype=torch.bfloat16, device=device),
    )


def _decode_conv_step(xn: torch.Tensor, W: torch.Tensor, conv_w: torch.Tensor,
                      conv_state: torch.Tensor):
    """
    Linear projection + one-token causal depthwise conv + SiLU.

    conv_state stores the previous CONV_K-1 projected tokens as
    [B, D_out, CONV_K-1] in oldest-to-newest order.
    """
    proj = (xn.squeeze(1) @ W.T).to(xn.dtype)                  # [B, D_out]
    window = torch.cat([conv_state, proj.unsqueeze(-1)], dim=-1)
    conv = (window * conv_w.unsqueeze(0)).sum(dim=-1)          # [B, D_out]
    out = torch.nn.functional.silu(conv).to(xn.dtype)
    return out, window[:, :, 1:].contiguous()


def _decode_fused_proj_step(xn: torch.Tensor, W: torch.Tensor, conv_w: torch.Tensor,
                            xn_state: torch.Tensor, proj_ext):
    """
    Fused projection + causal conv + SiLU for decode.

    The fused kernel does not expose projected-token history, so custom decode
    caches the previous normalized inputs and runs the fused kernel over the
    CONV_K-token window, then keeps only the newest output.
    """
    window = torch.cat([xn_state, xn], dim=1).contiguous()
    out = proj_ext.forward(window, W.contiguous(), conv_w.contiguous())
    return out[:, -1, :].contiguous(), window[:, 1:, :].contiguous()


def _dt_bias_for_decode(dt_bias: torch.Tensor):
    """Decode CUDA expects per-head dt_bias; the prefill kernel uses a scalar."""
    if dt_bias.numel() == 1:
        return dt_bias.reshape(1).expand(H).contiguous()
    return dt_bias.contiguous()


def make_weights():
    """Randomly initialise a shared weight dict (CUDA, bfloat16 / float32)."""
    def randn_bf16(*shape, scale=1.0):
        fan_in = shape[-1]
        return torch.randn(*shape, dtype=torch.bfloat16, device="cuda") * fan_in ** -0.5 * scale

    return dict(
        norm_gamma = torch.ones(D, dtype=torch.bfloat16, device="cuda"),

        W_q    = randn_bf16(H*DK, D, scale=DK ** -0.5),
        conv_q = torch.randn(H*DK, CONV_K, dtype=torch.bfloat16, device="cuda") * CONV_K ** -0.5,
        W_k    = randn_bf16(H*DK, D, scale=DK ** -0.5),
        conv_k = torch.randn(H*DK, CONV_K, dtype=torch.bfloat16, device="cuda") * CONV_K ** -0.5,
        W_v    = randn_bf16(H*DV, D, scale=DK ** -0.5),
        conv_v = torch.randn(H*DV, CONV_K, dtype=torch.bfloat16, device="cuda") * CONV_K ** -0.5,

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

    def __init__(self, weights: dict, chunk_size: int = None):
        self.w          = weights
        self.scale      = DK ** -0.5
        self.chunk_size = chunk_size

    def init_decode_state(self, B: int, device=None):
        return _zero_decode_state(self.w, B, device=device)

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

        # 1. RMSNorm  [B*T, D] -> [B, T, D]
        xn = ref_rmsnorm(x.reshape(B*T, D), w['norm_gamma']).reshape(B, T, D)

        # 2. Q / K / V projections  ->  [B, H, T, D_head]
        def proj(W, cw, dh):
            out = ref_proj_conv_silu(xn, W, cw)            # [B, T, H*dh]
            return out.reshape(B, T, H, dh).transpose(1, 2).contiguous()

        q = proj(w['W_q'], w['conv_q'], DK)
        k = proj(w['W_k'], w['conv_k'], DK)
        v = proj(w['W_v'], w['conv_v'], DV)

        # 3. a / b gate projections  ->  [B, H, T]
        a = (xn @ w['W_a'].T).transpose(1, 2).contiguous()
        b = (xn @ w['W_b'].T).transpose(1, 2).contiguous()

        # 4. Prefill recurrence  ->  o [B, H, T, DV] float32
        o, _ = ref_prefill(
            q, k, v, w['A_log'], a, w['dt_bias'], b, mask,
            state_in=None, scale=self.scale, chunk_size=self.chunk_size,
        )

        # 5. Output projection  ->  [B, T, D]
        return o.to(x.dtype).transpose(1, 2).reshape(B, T, H*DV) @ w['W_o'].T

    @torch.no_grad()
    def decode(self, x: torch.Tensor, state: dict = None):
        """
        Single-token decode.

        x     : [B, 1, D] bfloat16, CUDA
        state : dict from init_decode_state(); defaults to zeros
        Returns: (y [B, 1, D] bfloat16, new_state dict)
        """
        from reference.rmsnorm_ref import ref_rmsnorm
        from reference.decode_ref  import ref_decode

        B, T, _ = x.shape
        assert T == 1, f"decode expects T=1, got T={T}"
        w = self.w
        if state is None:
            state = self.init_decode_state(B, device=x.device)

        xn = ref_rmsnorm(x.reshape(B, D), w['norm_gamma']).reshape(B, 1, D)

        q_proj, conv_q = _decode_conv_step(xn, w['W_q'], w['conv_q'], state['conv_q'])
        k_proj, conv_k = _decode_conv_step(xn, w['W_k'], w['conv_k'], state['conv_k'])
        v_proj, conv_v = _decode_conv_step(xn, w['W_v'], w['conv_v'], state['conv_v'])
        xn_state = torch.cat([state['xn'], xn], dim=1)[:, 1:, :].contiguous()

        q = q_proj.reshape(B, 1, H, DK).contiguous()
        k = k_proj.reshape(B, 1, H, DK).contiguous()
        v = v_proj.reshape(B, 1, H, DV).contiguous()
        a = (xn @ w['W_a'].T).contiguous()      # [B, 1, H]
        b = (xn @ w['W_b'].T).contiguous()      # [B, 1, H]

        o, recurrent = ref_decode(
            q, k, v, state['recurrent'], w['A_log'], a,
            _dt_bias_for_decode(w['dt_bias']), b, self.scale,
        )
        y = o.to(x.dtype).reshape(B, 1, H*DV) @ w['W_o'].T
        return y, dict(
            recurrent=recurrent,
            conv_q=conv_q, conv_k=conv_k, conv_v=conv_v,
            xn=xn_state,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Custom: compiled CUDA kernels
# ─────────────────────────────────────────────────────────────────────────────
class GDNLayerCustom:
    """Forward/decode pass using the compiled CUDA kernels."""

    def __init__(self, weights: dict, exts: dict):
        """
        weights : dict from make_weights()
        exts    : dict with keys 'rmsnorm', 'proj', 'prefill', 'decode'
        """
        self.w     = weights
        self.exts  = exts
        self.scale = DK ** -0.5
        self.chunk = int(exts['prefill'].C_DIM)

    def init_decode_state(self, B: int, device=None):
        return _zero_decode_state(self.w, B, device=device)

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

        # 1. RMSNorm  (kernel: [R, D] bf16 -> bf16)
        xn = e['rmsnorm'].forward(
            x.reshape(B*T, D), w['norm_gamma'], 1e-6, 256, 2
        ).reshape(B, T, D)

        # 2. Q / K / V projections  ->  [B, H, T, D_head]
        def proj(W, cw, dh):
            out = e['proj'].forward(
                xn.contiguous(), W.contiguous(), cw.contiguous()
            )                                               # [B, T, H*dh]
            return out.reshape(B, T, H, dh).transpose(1, 2).contiguous()

        q = proj(w['W_q'], w['conv_q'], DK)
        k = proj(w['W_k'], w['conv_k'], DK)
        v = proj(w['W_v'], w['conv_v'], DV)

        # 3. a / b gate projections  ->  [B, H, T]  (plain matmul; no custom kernel)
        a = (xn @ w['W_a'].T).transpose(1, 2).contiguous()
        b = (xn @ w['W_b'].T).transpose(1, 2).contiguous()

        # 4. Prefill recurrence  ->  o [B, H, T, DV]
        empty_state = torch.empty(0, device=x.device, dtype=torch.float32)
        o, _ = e['prefill'].prefill(
            q, k, v,
            w['A_log'], a, float(w['dt_bias'].item()),
            b, mask, empty_state, self.scale,
        )

        # 5. Output projection  ->  [B, T, D]
        return o.transpose(1, 2).reshape(B, T, H*DV) @ w['W_o'].T

    @torch.no_grad()
    def decode(self, x: torch.Tensor, state: dict = None):
        """
        Single-token decode.

        x     : [B, 1, D] bfloat16, CUDA
        state : dict from init_decode_state(); defaults to zeros
        Returns: (y [B, 1, D] bfloat16, new_state dict)
        """
        B, T, _ = x.shape
        assert T == 1, f"decode expects T=1, got T={T}"
        w, e = self.w, self.exts
        if state is None:
            state = self.init_decode_state(B, device=x.device)

        xn = e['rmsnorm'].forward(
            x.reshape(B, D), w['norm_gamma'], 1e-6, 256, 2
        ).reshape(B, 1, D)

        xn_state = state.get('xn')
        if xn_state is None:
            xn_state = torch.zeros(B, CONV_K-1, D, device=x.device, dtype=x.dtype)

        q_proj, new_xn_state = _decode_fused_proj_step(
            xn, w['W_q'], w['conv_q'], xn_state, e['proj'])
        k_proj, _ = _decode_fused_proj_step(
            xn, w['W_k'], w['conv_k'], xn_state, e['proj'])
        v_proj, _ = _decode_fused_proj_step(
            xn, w['W_v'], w['conv_v'], xn_state, e['proj'])

        q = q_proj.reshape(B, H, DK).contiguous()
        k = k_proj.reshape(B, H, DK).contiguous()
        v = v_proj.reshape(B, H, DV).contiguous()
        a = (xn.squeeze(1) @ w['W_a'].T).contiguous()
        b = (xn.squeeze(1) @ w['W_b'].T).contiguous()

        o = torch.empty(B, H, DV, device=x.device, dtype=torch.float32)
        recurrent = torch.empty_like(state['recurrent'])
        e['decode'].forward(
            q, k, v, a, b,
            w['A_log'].contiguous(), _dt_bias_for_decode(w['dt_bias']),
            state['recurrent'].contiguous(), self.scale, o, recurrent,
        )

        y = o.to(x.dtype).reshape(B, 1, H*DV) @ w['W_o'].T
        return y, dict(
            recurrent=recurrent,
            conv_q=state.get('conv_q', torch.empty(0, device=x.device, dtype=x.dtype)),
            conv_k=state.get('conv_k', torch.empty(0, device=x.device, dtype=x.dtype)),
            conv_v=state.get('conv_v', torch.empty(0, device=x.device, dtype=x.dtype)),
            xn=new_xn_state,
        )
