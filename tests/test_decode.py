"""
test_decode.py — Correctness tests for the decode recurrence CUDA kernel.
"""

import sys, os
import math
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reference.decode_ref import ref_decode


def _compile_extension():
    from torch.utils.cpp_extension import load
    kernel_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "kernels")
    return load(
        name="decode_ext",
        sources=[os.path.join(kernel_dir, "decode.cu")],
        verbose=True,
        extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
    )


def run_case(ext, B, H=16, dk=128, dv=256):
    """Run a single decode test case."""
    scale = 1.0 / math.sqrt(dk)

    q = torch.randn(B, 1, H, dk, device="cuda", dtype=torch.bfloat16) * 0.02
    k = torch.randn(B, 1, H, dk, device="cuda", dtype=torch.bfloat16) * 0.02
    v = torch.randn(B, 1, H, dv, device="cuda", dtype=torch.bfloat16) * 0.02
    a_param = torch.randn(B, 1, H, device="cuda", dtype=torch.bfloat16) * 0.1
    b_param = torch.randn(B, 1, H, device="cuda", dtype=torch.bfloat16) * 0.1
    A_log = torch.zeros(H, device="cuda", dtype=torch.float32)
    dt_bias = torch.zeros(H, device="cuda", dtype=torch.float32)
    S_in = torch.randn(B, H, dk, dv, device="cuda", dtype=torch.float32) * 0.01

    # CUDA kernel: needs (B, H, dk/dv) layout for q/k/v and (B, H) for scalars
    q_k = q.squeeze(1).reshape(B, H, dk).contiguous()
    k_k = k.squeeze(1).reshape(B, H, dk).contiguous()
    v_k = v.squeeze(1).reshape(B, H, dv).contiguous()
    a_k = a_param.squeeze(1).reshape(B, H).contiguous()
    b_k = b_param.squeeze(1).reshape(B, H).contiguous()

    o_out = torch.zeros(B, H, dv, device="cuda", dtype=torch.float32)
    S_out = torch.zeros_like(S_in)

    with torch.no_grad():
        ext.forward(
            q_k.to(torch.bfloat16), k_k.to(torch.bfloat16), v_k.to(torch.bfloat16),
            a_k.to(torch.bfloat16), b_k.to(torch.bfloat16),
            A_log, dt_bias, S_in, scale, o_out, S_out
        )
        o_ref, S_ref = ref_decode(q, k, v, S_in, A_log, a_param, dt_bias, b_param, scale)

    diff_o = (o_out.float() - o_ref.float()).abs()
    diff_s = (S_out.float() - S_ref.float()).abs()

    max_err_o = diff_o.max().item()
    max_err_s = diff_s.max().item()

    rel_err_o = (diff_o / (o_ref.float().abs() + 1e-8)).max().item()
    rel_err_s = (diff_s / (S_ref.float().abs() + 1e-8)).max().item()

    passed = max_err_o < 1e-2 and max_err_s < 1e-2 and rel_err_o < 0.05 and rel_err_s < 0.05

    status = "PASS" if passed else "FAIL"
    print(f"[{status}]  B={B:>2}  H={H:>3}  dk={dk}  dv={dv}  "
          f"max_err_o={max_err_o:.3e}  max_err_S={max_err_s:.3e}  "
          f"rel_err_o={rel_err_o:.3e}  rel_err_S={rel_err_s:.3e}")
    return passed


def run_all(ext=None):
    if ext is None:
        ext = _compile_extension()

    all_passed = True
    print("=" * 70)
    print("Decode Correctness Tests")
    print("=" * 70)
    for B in [1, 8, 32, 64]:
        all_passed &= run_case(ext, B)

    print()
    if all_passed:
        print("All decode tests PASSED.")
    else:
        print("Some decode tests FAILED.")
    return all_passed


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available."); raise SystemExit(1)
    print(f"GPU: {torch.cuda.get_device_name(0)}\n")
    ok = run_all()
    raise SystemExit(0 if ok else 1)
