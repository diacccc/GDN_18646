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


def run_case(ext, B, T, D=2048, eps=1e-6):
    # Kernel is BF16-only and hardcoded to D=2048
    assert D == 2048, "rmsnorm kernel only supports D=2048"
    x = torch.randn(B, T, D, device="cuda", dtype=torch.bfloat16)
    gamma = torch.ones(D, device="cuda", dtype=torch.bfloat16)

    torch_rms = torch.nn.RMSNorm(D, eps=eps, elementwise_affine=True).to("cuda", torch.bfloat16)
    torch.nn.init.ones_(torch_rms.weight)

    x_flat = x.reshape(-1, D).contiguous()

    with torch.no_grad():
        y_custom = ext.forward(x_flat, gamma, eps)
        y_ref    = ref_rmsnorm(x_flat, gamma, eps)
        y_torch  = torch_rms(x_flat)

    # vs custom ref
    diff_ref = (y_custom.float() - y_ref.float()).abs()
    max_err_ref = diff_ref.max().item()
    rel_err_ref = (diff_ref / (y_ref.float().abs() + 1e-8)).max().item()

    # vs torch.nn.RMSNorm
    diff_torch = (y_custom.float() - y_torch.float()).abs()
    max_err_torch = diff_torch.max().item()
    rel_err_torch = (diff_torch / (y_torch.float().abs() + 1e-8)).max().item()

    # BF16 has ~2 decimal digits of precision; use looser absolute threshold
    passed = max_err_ref < 1e-2 and rel_err_ref < 0.05 and max_err_torch < 1e-2 and rel_err_torch < 0.05

    status = "PASS" if passed else "FAIL"
    print(f"[{status}]  B={B:>2}  T={T:>5}  "
          f"vs_ref: max={max_err_ref:.3e} rel={rel_err_ref:.3e}  "
          f"vs_torch: max={max_err_torch:.3e} rel={rel_err_torch:.3e}")
    return passed


def run_all(ext=None):
    if ext is None:
        ext = _compile_extension()

    all_passed = True
    print("=" * 60)
    print("RMSNorm Correctness Tests")
    print("=" * 60)
    for B in [1, 8, 32]:
        for T in [1, 2048, 4096]:
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
