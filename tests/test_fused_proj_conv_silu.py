"""
test_fused_proj_conv_silu.py
Compare fused CUDA kernel against the PyTorch reference for several (B, T) configs.

Can be invoked standalone:
    python -m tests.test_fused_proj_conv_silu

Or via the Modal entrypoint:
    modal run modal_app.py --mode test --kernel fused_proj_conv_silu
"""

import sys
import os
import torch

# Ensure project root is on the path so imports work inside Modal container too
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reference.fused_proj_conv_silu_ref import ref_proj_conv_silu


def _compile_extension():
    """JIT-compile the CUDA extension. Returns the loaded module."""
    from torch.utils.cpp_extension import load

    kernel_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "kernels")
    ext = load(
        name="fused_proj_conv_silu_ext",
        sources=[
            os.path.join(kernel_dir, "fused_proj_conv_silu.cu"),
        ],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "-gencode=arch=compute_80,code=sm_80",
            "--expt-relaxed-constexpr",
            "-lineinfo",
        ],
        verbose=True,
    )
    return ext


def run_case(ext, B, T, D=2048, D_out=2048, conv_k=4):
    """Compare CUDA kernel output against PyTorch reference. Returns True on pass."""
    x      = torch.randn(B, T, D, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(D_out, D, device="cuda", dtype=torch.bfloat16) * (D ** -0.5)
    conv_w = torch.randn(D_out, conv_k, device="cuda", dtype=torch.bfloat16) * 0.5

    with torch.no_grad():
        o_cuda = ext.forward(x.contiguous(), weight.contiguous(), conv_w.contiguous())
        o_ref  = ref_proj_conv_silu(x, weight, conv_w)

    diff     = (o_cuda.float() - o_ref.float()).abs()
    max_err  = diff.max().item()
    mean_err = diff.mean().item()
    rel_err  = (diff / (o_ref.float().abs() + 1e-8)).max().item()
    passed   = max_err < 5e-2 and rel_err < 0.1    # bfloat16 precision (~7 mantissa bits)

    status = "PASS" if passed else "FAIL"
    print(
        f"[{status}]  B={B:>2}  T={T:>5}  D={D:>5}  D_out={D_out:>5}  "
        f"max_err={max_err:.3e}  mean_err={mean_err:.3e}  rel_err={rel_err:.3e}"
    )
    return passed


def run_all(ext=None):
    """Run the full correctness suite. Returns True if every case passes."""
    if ext is None:
        ext = _compile_extension()

    all_passed = True

    print("=" * 80)
    print("Correctness Tests  (Q/K path: D_out = Dk = 2048)")
    print("=" * 80)
    for B in [1, 4]:
        for T in [1, 64, 128, 256]:
            all_passed &= run_case(ext, B, T, D=2048, D_out=2048)

    print()
    print("=" * 80)
    print("Correctness Tests  (V path: D_out = Dv = 4096)")
    print("=" * 80)
    for B in [1, 4]:
        for T in [1, 64, 128]:
            all_passed &= run_case(ext, B, T, D=2048, D_out=4096)

    print()
    if all_passed:
        print("All correctness tests PASSED.")
    else:
        print("Some tests FAILED — check output above.")
    return all_passed


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available — exiting.")
        raise SystemExit(1)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("Compiling CUDA extension …\n")
    ok = run_all()
    raise SystemExit(0 if ok else 1)
