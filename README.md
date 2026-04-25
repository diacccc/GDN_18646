# GDN CUDA Kernels

This repository contains CUDA kernels, reference implementations, tests, and benchmarks for a Gated Delta Net (GDN) layer.

Supported components:

- `rmsnorm`
- `fused_proj_conv_silu`
- `prefill`
- `decode`
- end-to-end GDN layer benchmark

## Requirements

These kernels use BF16 tensors, so the GPU must support BF16. The default testing environment uses an NVIDIA A100, which satisfies this requirement.

Tested Modal environment:

- GPU: `A100-80GB`
- Base image: `nvidia/cuda:12.8.1-devel-ubuntu22.04`
- Python: `3.11`
- PyTorch: `2.10.0`
- CUDA wheel index: `https://download.pytorch.org/whl/cu128`
- Additional Python packages: `numpy`, `ninja`, `einops`

## Setup

Install Modal and authenticate:

```bash
pip install modal
modal setup
```

The Modal image installs the CUDA/PyTorch dependencies automatically.

## Correctness Tests

Run all kernel tests:

```bash
modal run modal_app.py --mode test
```

Run one kernel test:

```bash
modal run modal_app.py --mode test --kernel rmsnorm
modal run modal_app.py --mode test --kernel fused_proj_conv_silu
modal run modal_app.py --mode test --kernel prefill
modal run modal_app.py --mode test --kernel decode
```

## Benchmarks

Run one kernel benchmark:

```bash
modal run modal_app.py --mode bench --kernel rmsnorm
modal run modal_app.py --mode bench --kernel fused_proj_conv_silu
modal run modal_app.py --mode bench --kernel prefill
modal run modal_app.py --mode bench --kernel decode
```

Run the end-to-end GDN benchmark, which will run both prefill and decode benchmark: 

```bash
modal run modal_app.py::bench_gdn
```

## Results

All saved test and benchmark results are stored in the `results/` directory.

## Interactive GPU Shell

Start an interactive shell in the Modal environment:

```bash
modal shell modal_app.py::shell
```

## Local Runs

If you have a local CUDA environment with BF16-capable GPU support and the required dependencies, you can run tests and benchmarks directly:

```bash
python -m tests.test_rmsnorm
python -m tests.test_fused_proj_conv_silu
python -m tests.test_prefill
python -m tests.test_decode

python -m bench.bench_rmsnorm
python -m bench.bench_fused_proj_conv_silu
python -m bench.bench_prefill
python -m bench.bench_decode

python bench_gdn.py
```