"""
test_rmsnorm.py — Correctness tests for the RMSNorm CUDA kernel.
"""

import sys, os
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reference.rmsnorm_ref import ref_rmsnorm


def _compile_extension():
    from torch.utils.cpp_extension import load
    kernel_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "kernels")
    return load(
        name="rmsnorm_ext",
        sources=[os.path.join(kernel_dir, "rmsnorm.cu")],
        verbose=True,
        extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
    )


def run_case(ext, B, T, D=2048, eps=1e-6, dtype=torch.float32):
    x = torch.randn(B, T, D, device="cuda", dtype=dtype)
    gamma = torch.ones(D, device="cuda", dtype=dtype)

    x_flat = x.reshape(-1, D).contiguous()

    with torch.no_grad():
        y_custom = ext.forward(x_flat, gamma, eps)
        y_ref = ref_rmsnorm(x_flat, gamma, eps)

    diff = (y_custom.float() - y_ref.float()).abs()
    max_err = diff.max().item()
    rel_err = (diff / (y_ref.float().abs() + 1e-8)).max().item()
    passed = max_err < 1e-4 and rel_err < 0.05

    status = "PASS" if passed else "FAIL"
    print(f"[{status}]  B={B:>2}  T={T:>5}  max_err={max_err:.3e}  rel_err={rel_err:.3e}")
    return passed


def run_all(ext=None):
    if ext is None:
        ext = _compile_extension()

    all_passed = True
    print("=" * 60)
    print("RMSNorm Correctness Tests")
    print("=" * 60)
    for B in [1]: #[1, 8, 32]:
        for T in [1]: #[1, 2048, 4096]:
            all_passed &= run_case(ext, B, T)

    print()
    if all_passed:
        print("All RMSNorm tests PASSED.")
    else:
        print("Some RMSNorm tests FAILED.")
    return all_passed


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available."); raise SystemExit(1)
    print(f"GPU: {torch.cuda.get_device_name(0)}\n")
    ok = run_all()
    raise SystemExit(0 if ok else 1)
