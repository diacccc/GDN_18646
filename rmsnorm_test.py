import argparse

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load

try:
    import pandas as pd
except ImportError:  # pragma: no cover - optional dependency for nicer tables
    pd = None


DEFAULT_D = 2048
DEFAULT_B = [1, 8, 32, 64]
DEFAULT_T = [1, 2048, 4096, 8192]
DEFAULT_WARMUP = 10
DEFAULT_ITERS = 50


def load_extension(source="rmsnorm.cu", name="rmsnorm_ext_bench"):
    return load(
        name=name,
        sources=[source],
        verbose=True,
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "-lineinfo",
            "-gencode=arch=compute_80,code=sm_80",
        ],
    )


def benchmark_fn(fn, warmup=DEFAULT_WARMUP, iters=DEFAULT_ITERS):
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


def throughput_gb_s(bytes_accessed, latency_ms):
    return bytes_accessed / (latency_ms * 1e-3) / 1e9


def rmsnorm_baseline(x, gamma, beta=None, eps=1e-6):
    x_fp32 = x.float()
    gamma_fp32 = gamma.float()
    rms = x_fp32.pow(2).mean(dim=-1, keepdim=True)
    x_normed = x_fp32 * torch.rsqrt(rms + eps)
    out = x_normed * gamma_fp32
    if beta is not None:
        out = out + beta.float()
    return out.to(x.dtype)


class RMSNormFused(nn.Module):
    def __init__(self, dim, eps=1e-6, dtype=torch.bfloat16, device="cuda"):
        super().__init__()
        if not hasattr(nn, "RMSNorm"):
            raise RuntimeError("torch.nn.RMSNorm is not available in this PyTorch build.")
        self.norm = nn.RMSNorm(
            dim,
            eps=eps,
            elementwise_affine=True,
            device=device,
            dtype=dtype,
        )

    def forward(self, x):
        return self.norm(x)


def rmsnorm_custom(ext, x, gamma, eps=1e-6):
    dim = x.shape[-1]
    x_flat = x.reshape(-1, dim).contiguous()
    y_flat = ext.forward(x_flat, gamma.contiguous(), eps)
    return y_flat.reshape_as(x)


def make_gamma(dim, device, dtype, mode):
    if mode == "random":
        return torch.randn(dim, device=device, dtype=dtype)
    return torch.ones(dim, device=device, dtype=dtype)


def benchmark_rmsnorm_suite(
    ext,
    B_vals=DEFAULT_B,
    T_vals=DEFAULT_T,
    dim=DEFAULT_D,
    eps=1e-6,
    dtype=torch.bfloat16,
    warmup=DEFAULT_WARMUP,
    iters=DEFAULT_ITERS,
    gamma_mode="ones",
):
    device = torch.device("cuda")
    elem_size = torch.empty((), dtype=dtype).element_size()

    fused_module = RMSNormFused(dim, eps=eps, dtype=dtype, device=device).eval()
    results = []

    for B in B_vals:
        for T in T_vals:
            x = torch.randn(B, T, dim, device=device, dtype=dtype)
            gamma = make_gamma(dim, device, dtype, gamma_mode)

            with torch.no_grad():
                fused_module.norm.weight.copy_(gamma)
                y_base = rmsnorm_baseline(x, gamma, eps=eps)
                y_fused = fused_module(x)
                y_custom = rmsnorm_custom(ext, x, gamma, eps)

            max_err_fused = (y_fused.float() - y_base.float()).abs().max().item()
            max_err_custom = (y_custom.float() - y_base.float()).abs().max().item()
            max_err_custom_vs_fused = (y_custom.float() - y_fused.float()).abs().max().item()

            def run_base():
                with torch.no_grad():
                    rmsnorm_baseline(x, gamma, eps=eps)

            def run_fused():
                with torch.no_grad():
                    fused_module(x)

            def run_custom():
                with torch.no_grad():
                    rmsnorm_custom(ext, x, gamma, eps)

            lat_base = benchmark_fn(run_base, warmup=warmup, iters=iters)
            lat_fused = benchmark_fn(run_fused, warmup=warmup, iters=iters)
            lat_custom = benchmark_fn(run_custom, warmup=warmup, iters=iters)

            bytes_io = 2 * B * T * dim * elem_size + dim * elem_size
            bw_fused = throughput_gb_s(bytes_io, lat_fused)
            bw_custom = throughput_gb_s(bytes_io, lat_custom)

            row = dict(
                B=B,
                T=T,
                max_err_fused=max_err_fused,
                max_err_custom=max_err_custom,
                max_err_custom_vs_fused=max_err_custom_vs_fused,
                baseline_ms=lat_base,
                fused_ms=lat_fused,
                custom_ms=lat_custom,
                fused_speedup_vs_baseline=lat_base / lat_fused,
                custom_speedup_vs_baseline=lat_base / lat_custom,
                custom_speedup_vs_fused=lat_fused / lat_custom,
                fused_bw_gbs=bw_fused,
                custom_bw_gbs=bw_custom,
            )
            results.append(row)

            print(
                f"B={B:>2}, T={T:>5}  "
                f"baseline={lat_base:8.4f} ms  "
                f"fused={lat_fused:8.4f} ms  "
                f"custom={lat_custom:8.4f} ms  "
                f"custom/fused={lat_fused / lat_custom:5.2f}x  "
                f"max_err={max_err_custom:.3e}"
            )

    if pd is not None:
        return pd.DataFrame(results)
    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark custom RMSNorm CUDA against the gdn_benchmark.ipynb baseline and RMSNormFused."
    )
    parser.add_argument("--source", default="rmsnorm.cu")
    parser.add_argument("--module-name", default="rmsnorm_ext_bench")
    parser.add_argument("--B", type=int, nargs="*", default=DEFAULT_B)
    parser.add_argument("--T", type=int, nargs="*", default=DEFAULT_T)
    parser.add_argument("--D", type=int, default=DEFAULT_D)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    parser.add_argument("--gamma-mode", choices=["ones", "random"], default="ones")
    parser.add_argument("--csv", type=str, default="")
    return parser.parse_args()


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Run this on the A100 machine.")

    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name}")
    print(f"SMs: {props.multi_processor_count}")

    args = parse_args()
    ext = load_extension(source=args.source, name=args.module_name)
    df = benchmark_rmsnorm_suite(
        ext,
        B_vals=args.B,
        T_vals=args.T,
        dim=args.D,
        eps=args.eps,
        warmup=args.warmup,
        iters=args.iters,
        gamma_mode=args.gamma_mode,
    )

    if pd is not None:
        print("\nRMSNorm comparison table:")
        print(df.round(4).to_string(index=False))
        if args.csv:
            df.to_csv(args.csv, index=False)
            print(f"\nSaved results to {args.csv}")
    else:
        print("\nInstall pandas for a tabular summary and CSV export.")


if __name__ == "__main__":
    main()
