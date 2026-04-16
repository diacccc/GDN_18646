import argparse

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load


def load_extension():
    return load(
        name="rmsnorm_ext",
        sources=["rmsnorm.cu"],
        verbose=True,
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "-lineinfo"
        ],
    )


def ref_rmsnorm(x, gamma, eps=1e-6):
    x_fp32 = x.float()
    rms = x_fp32.pow(2).mean(dim=-1, keepdim=True)
    x_normed = x_fp32 * torch.rsqrt(rms + eps)
    out = x_normed.to(x.dtype) * gamma
    return out


def bench_ms(fn, warmup=20, iters=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def run_case(ext, B, T, D=2048, eps=1e-6, dtype=torch.float32):
    x = torch.randn(B, T, D, device="cuda", dtype=dtype)
    gamma = torch.ones(D, device="cuda", dtype=dtype)

    x_flat = x.reshape(-1, D).contiguous()
    gamma = gamma.contiguous()

    with torch.no_grad():
        y_custom = ext.forward(x_flat, gamma, eps)
        y_ref = ref_rmsnorm(x_flat, gamma, eps)

    max_err = (y_custom.float() - y_ref.float()).abs().max().item()

    torch_mod = nn.RMSNorm(D, eps=eps, elementwise_affine=True, device="cuda", dtype=dtype).eval()
    with torch.no_grad():
        torch_mod.weight.copy_(gamma)

    t_custom = bench_ms(lambda: ext.forward(x_flat, gamma, eps))
    t_ref = bench_ms(lambda: ref_rmsnorm(x_flat, gamma, eps))
    t_torch = bench_ms(lambda: torch_mod(x_flat))

    print(
        f"B={B:>2} T={T:>5} R={B*T:>6} "
        f"max_err={max_err:.3e} "
        f"custom={t_custom:8.4f} ms "
        f"torch={t_torch:8.4f} ms "
        f"ref={t_ref:8.4f} ms"
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--B", type=int, nargs="*", default=[1, 8, 32, 64])
    parser.add_argument("--T", type=int, nargs="*", default=[1, 2048, 4096, 8192])
    parser.add_argument("--eps", type=float, default=1e-6)
    return parser.parse_args()


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Run this on the A100 machine.")
    print(torch.cuda.get_device_name(0))
    print(torch.cuda.get_device_capability(0))
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name}")
    print(f"SMs: {props.multi_processor_count}")

    args = parse_args()
    ext = load_extension()

    for B in args.B:
        for T in args.T:
            run_case(ext, B, T, eps=args.eps)


if __name__ == "__main__":
    main()
