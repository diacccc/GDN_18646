"""
test_prefill.py — Correctness tests for the prefill recurrence CUDA kernel.
"""

import sys, os
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reference.prefill_ref import ref_prefill


def _compile_extension():
    from torch.utils.cpp_extension import load
    kernel_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "kernels")
    return load(
        name="prefill_ext",
        sources=[os.path.join(kernel_dir, "prefill_chunked.cu")],
        verbose=True,
        extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo", "-arch=sm_80"],
    )


def run_case(ext, B, T, Hh=16, dk=128, dv=256, dtype=torch.bfloat16):
    chunk_size = int(ext.C_DIM)
    if T % chunk_size != 0:
        print(f"[SKIP]  B={B:>2}  T={T:>5}  H={Hh:>3}  (T not divisible by chunk_size={chunk_size})")
        return True

    q        = torch.randn(B, Hh, T, dk, device="cuda", dtype=dtype) * (dk ** -0.5)
    k        = torch.randn(B, Hh, T, dk, device="cuda", dtype=dtype) * (dk ** -0.5)
    v        = torch.randn(B, Hh, T, dv, device="cuda", dtype=dtype) * (dk ** -0.5)
    a        = torch.randn(B, Hh, T, device="cuda", dtype=dtype)
    b_logits = torch.randn(B, Hh, T, device="cuda", dtype=dtype)
    A_log    = torch.zeros(Hh, device="cuda", dtype=torch.float32)
    dt_bias  = torch.tensor(0.0, device="cuda", dtype=torch.float32)
    mask     = torch.ones(B, T, device="cuda", dtype=torch.float32)
    state_in_ext = torch.empty(0, device="cuda", dtype=torch.float32)
    scale = 1.0

    with torch.no_grad():
        o_custom, _ = ext.prefill(
            q.contiguous(), k.contiguous(), v.contiguous(),
            A_log.contiguous(), a.contiguous(),
            float(dt_bias.item()),
            b_logits.contiguous(), mask.contiguous(),
            state_in_ext, scale,
        )
        o_ref, _ = ref_prefill(
            q.contiguous(), k.contiguous(), v.contiguous(),
            A_log.contiguous(), a.contiguous(), dt_bias.contiguous(),
            b_logits.contiguous(), mask.contiguous(),
            state_in=None, scale=scale,
        )

    diff = (o_custom.float() - o_ref.float()).abs()
    max_err = diff.max().item()
    rel_err = (diff / (o_ref.float().abs() + 1e-8)).max().item()
    passed = max_err < 1e-2 and rel_err < 0.05

    status = "PASS" if passed else "FAIL"
    print(f"[{status}]  B={B:>2}  T={T:>5}  H={Hh:>3}  max_err={max_err:.3e}  rel_err={rel_err:.3e}")
    return passed


def run_all(ext=None):
    if ext is None:
        ext = _compile_extension()

    all_passed = True
    print("=" * 60)
    print("Prefill Correctness Tests")
    print("=" * 60)
    for B in [1, 4]:
        for T in [64, 128, 256]:
            all_passed &= run_case(ext, B, T)

    print()
    if all_passed:
        print("All prefill tests PASSED.")
    else:
        print("Some prefill tests FAILED.")
    return all_passed


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available."); raise SystemExit(1)
    print(f"GPU: {torch.cuda.get_device_name(0)}\n")
    ok = run_all()
    raise SystemExit(0 if ok else 1)
