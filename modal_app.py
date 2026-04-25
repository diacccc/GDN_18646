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
        "einops",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    # Source code: mounted at container start (fast redeploy on edits).
    .add_local_dir(HERE / "kernels", "/root/kernels")
    .add_local_dir(HERE / "reference", "/root/reference")
    .add_local_dir(HERE / "tests", "/root/tests")
    .add_local_dir(HERE / "bench", "/root/bench")
    .add_local_file(HERE / "gdn_layer.py", "/root/gdn_layer.py")
    .add_local_file(HERE / "bench_gdn.py", "/root/bench_gdn.py")
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
def bench_gdn():
    """Run end-to-end GDN benchmark (custom kernels vs PyTorch baseline)."""
    import sys
    sys.path.insert(0, "/root")
    from bench_gdn import main
    main()


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
    else:
        raise ValueError(f"Unknown mode {mode!r}; use 'test' or 'bench'.")
