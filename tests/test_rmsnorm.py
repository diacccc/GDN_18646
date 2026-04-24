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

    # FP32 ground truth: torch.nn.RMSNorm in full float32
    torch_rms_fp32 = torch.nn.RMSNorm(D, eps=eps, elementwise_affine=True,
                                       device="cuda", dtype=torch.float32).eval()
    torch.nn.init.ones_(torch_rms_fp32.weight)

    x_flat = x.reshape(-1, D).contiguous()

    with torch.inference_mode():
        y_custom = ext.forward(x_flat, gamma, eps, 256, 2)  # BF16 output
        y_fp32   = torch_rms_fp32(x_flat.float())           # FP32 ground truth

    # Compare BF16 kernel output against FP32 reference (cast kernel output to fp32)
    diff = (y_custom.float() - y_fp32).abs()
    max_err = diff.max().item()
    rel_err = (diff / (y_fp32.abs() + 1e-8)).max().item()

    # BF16 quantisation introduces ~1/128 relative error (~0.8%); allow up to 2%
    passed = max_err < 5e-2 and rel_err < 0.02

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
