"""
Modal entrypoint for GDN CUDA kernels.

Usage (from project root, with Modal installed and `modal setup` done):

    # Run ALL correctness tests on A100
    modal run modal_app.py::test

    # Run a single kernel's test
    modal run modal_app.py::test --kernel fused_proj_conv_silu
    modal run modal_app.py::test --kernel rmsnorm
    modal run modal_app.py::test --kernel prefill
    modal run modal_app.py::test --kernel decode

    # Run performance benchmark on A100
    modal run modal_app.py::bench

    # Profile with Nsight Compute on A100
    modal run modal_app.py --mode profile --kernel fused_proj_conv_silu

    # Drop into an interactive shell on A100
    modal shell modal_app.py::shell

The CUDA extensions are JIT-compiled at container start via torch's cpp_extension.
Builds are cached in a Modal Volume so subsequent runs only recompile when
the .cu / .cpp sources actually change.
"""

from pathlib import Path

import modal

HERE = Path(__file__).parent

# ---------------------------------------------------------------------------
# Container image: CUDA 12.8 devel (has nvcc) + PyTorch 2.10 + ninja.
# CUDA devel image is required because torch.utils.cpp_extension.load needs nvcc.
# ---------------------------------------------------------------------------
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("git", "build-essential")
    .pip_install(
        "torch==2.10.0",
        "numpy",
        "ninja",  # much faster than plain setuptools build
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    # Source code: mounted at container start (fast redeploy on edits).
    .add_local_dir(HERE / "kernels", "/root/kernels")
    .add_local_dir(HERE / "reference", "/root/reference")
    .add_local_dir(HERE / "tests", "/root/tests")
    .add_local_dir(HERE / "bench", "/root/bench")
)

app = modal.App("gdn-kernels", image=image)

# Default GPU. Override with MODAL_GPU env var if you ever want to try elsewhere.
GPU = "A100-80GB"

# Persist torch_extensions build cache across runs so we don't recompile
# from scratch every cold start. Ninja still rebuilds on source change.
build_cache = modal.Volume.from_name("gdn-torch-ext-cache", create_if_missing=True)

VOLUMES = {"/root/.cache/torch_extensions": build_cache}


CUDA_FLAGS = [
    "-O3", "--use_fast_math",
    "--expt-relaxed-constexpr", "-lineinfo",
    "-t0",
]

# Registry: kernel name -> (extension_name, sources, extra_cflags)
_KERNEL_REGISTRY = {
    "fused_proj_conv_silu": (
        "fused_proj_conv_silu_ext",
        ["/root/kernels/fused_proj_conv_silu.cu"],
        [],
    ),
    "rmsnorm": ("rmsnorm_ext", ["/root/kernels/rmsnorm.cu"], []),
    "prefill": ("prefill_ext", ["/root/kernels/prefill_unchunked.cu"], []),
    "decode":  ("decode_ext",  ["/root/kernels/decode.cu"], []),
}

_ext_cache: dict = {}


def _compile_one(kernel_name: str):
    """JIT-compile a single CUDA extension. Returns the loaded module."""
    if kernel_name in _ext_cache:
        return _ext_cache[kernel_name]

    import os, shutil
    import torch
    from torch.utils.cpp_extension import load

    assert torch.cuda.is_available(), "CUDA not available in container"
    print(f"[compile] torch={torch.__version__}  cuda={torch.version.cuda}  "
          f"device={torch.cuda.get_device_name(0)}")

    cache_root = "/root/.cache/torch_extensions"
    os.makedirs(cache_root, exist_ok=True)
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0")

    ext_name, sources, extra_cflags = _KERNEL_REGISTRY[kernel_name]

    # Purge stale cached build so ninja rebuilds from current sources
    stale_dir = os.path.join(cache_root, "py311_cu128", ext_name)
    if os.path.isdir(stale_dir):
        shutil.rmtree(stale_dir)
        print(f"[compile] cleared stale cache: {stale_dir}")

    ext = load(
        name=ext_name,
        sources=sources,
        extra_cuda_cflags=CUDA_FLAGS,
        extra_cflags=extra_cflags if extra_cflags else [],
        verbose=True,
    )
    _ext_cache[kernel_name] = ext
    return ext


# ---------------------------------------------------------------------------
# Entrypoints
# ---------------------------------------------------------------------------
@app.function(gpu=GPU, timeout=1800, volumes=VOLUMES)
def test(kernel: str = "all"):
    """Run correctness tests. Use --kernel to test a single kernel."""
    import sys
    sys.path.insert(0, "/root")

    targets = list(_KERNEL_REGISTRY.keys()) if kernel == "all" else [kernel]
    all_ok = True

    for name in targets:
        ext = _compile_one(name)
        if name == "fused_proj_conv_silu":
            from tests.test_fused_proj_conv_silu import run_all as run_fpcs
            all_ok &= run_fpcs(ext)
        elif name == "rmsnorm":
            from tests.test_rmsnorm import run_all as run_rms
            all_ok &= run_rms(ext)
        elif name == "prefill":
            from tests.test_prefill import run_all as run_pf
            all_ok &= run_pf(ext)
        elif name == "decode":
            from tests.test_decode import run_all as run_dec
            all_ok &= run_dec(ext)

    if not all_ok:
        raise SystemExit(1)


@app.function(gpu=GPU, timeout=1800, volumes=VOLUMES)
def bench(kernel: str = "fused_proj_conv_silu"):
    """Run throughput/latency benchmarks. Use --kernel to benchmark a single kernel."""
    import sys
    sys.path.insert(0, "/root")

    ext = _compile_one(kernel)

    if kernel == "fused_proj_conv_silu":
        from bench.bench_fused_proj_conv_silu import run_bench
    elif kernel == "rmsnorm":
        from bench.bench_rmsnorm import run_bench
    elif kernel == "prefill":
        from bench.bench_prefill import run_bench
    elif kernel == "decode":
        from bench.bench_decode import run_bench
    else:
        raise ValueError(f"Unknown kernel {kernel!r}")

    run_bench(ext)


@app.function(gpu=GPU, timeout=1800, volumes=VOLUMES)
def profile(kernel: str = "fused_proj_conv_silu"):
    """Profile a kernel with torch.profiler (CUPTI). Prints key metrics to stdout."""
    import sys
    sys.path.insert(0, "/root")
    import torch
    from torch.profiler import profile as torch_profile, ProfilerActivity, schedule

    ext = _compile_one(kernel)

    B, T, D = 8, 2048, 2048
    D_out = 4096 if "proj" in kernel else 2048
    conv_k = 4

    x      = torch.randn(B, T, D, device="cuda", dtype=torch.bfloat16) * 0.02
    weight = torch.randn(D_out, D, device="cuda", dtype=torch.bfloat16) * (D ** -0.5)
    conv_w = torch.randn(D_out, conv_k, device="cuda", dtype=torch.bfloat16) * 0.5

    # Warmup
    for _ in range(5):
        ext.forward(x, weight, conv_w)
    torch.cuda.synchronize()

    # Profile with CUPTI
    with torch_profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True,
        with_flops=True,
    ) as prof:
        for _ in range(10):
            ext.forward(x, weight, conv_w)
        torch.cuda.synchronize()

    print("=" * 90)
    print(f"Profile: kernel={kernel}  B={B} T={T} D={D} D_out={D_out}")
    print("=" * 90)

    # Key averages table
    print("\n── CUDA Kernel Summary (sorted by CUDA time) ──────────────────────")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

    # Detailed CUDA events
    print("\n── CUDA Event Trace (top 30 by duration) ──────────────────────────")
    events = sorted(
        [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA],
        key=lambda e: e.cuda_time_total,
        reverse=True,
    )
    print(f"{'Kernel':<60} {'Calls':>6} {'CUDA us':>10} {'Avg us':>10}")
    print("-" * 90)
    seen = set()
    for e in events[:30]:
        key = e.key
        if key in seen:
            continue
        seen.add(key)
        print(f"{key[:59]:<60} {e.count:>6} {e.cuda_time_total:>10.1f} "
              f"{e.cuda_time_total / max(e.count, 1):>10.1f}")

    # ── Register / shared-memory usage via cuobjdump ──────────────────
    import subprocess, glob
    ext_name = _KERNEL_REGISTRY[kernel][0]
    so_files = glob.glob(f"/root/.cache/torch_extensions/py311_cu128/{ext_name}/*.so")
    if so_files:
        res = subprocess.run(
            ["cuobjdump", "-res-usage", so_files[0]],
            capture_output=True, text=True,
        )
        print("\n── Resource Usage (registers, smem, spill) ─────────────────────────")
        print(res.stdout.strip() if res.stdout.strip() else "(no output from cuobjdump)")
        if res.stderr.strip():
            print(res.stderr.strip())
    else:
        print(f"\n[profile] could not find .so for {ext_name}")

    # Export Chrome trace for optional local inspection
    trace_path = "/tmp/profile_trace.json"
    prof.export_chrome_trace(trace_path)
    print(f"\n[profile] Chrome trace saved to {trace_path}")
    print("[profile] (Use `modal shell` to copy it out, or view in chrome://tracing)")


@app.function(gpu=GPU, timeout=3600, volumes=VOLUMES)
def shell():
    """
    Interactive entry. Launch with:  modal shell modal_app.py::shell

    Once inside you have:
      - A mounted A100
      - /root/{kernels,reference,tests,bench} with your code
      - nvcc, ncu (if requested separately), python, torch
    """
    import subprocess
    subprocess.run(["/bin/bash", "-l"])


@app.local_entrypoint()
def main(mode: str = "test", kernel: str = "all"):
    """Dispatcher: `modal run modal_app.py --mode test|bench [--kernel name]`."""
    if mode == "test":
        test.remote(kernel=kernel)
    elif mode == "bench":
        bench.remote(kernel=kernel)
    elif mode == "profile":
        profile.remote(kernel=kernel)
    else:
        raise ValueError(f"Unknown mode {mode!r}; use 'test', 'bench', or 'profile'.")
