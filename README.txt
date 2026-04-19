```bash
pip install modal
modal setup
```


```bash
# Test all kernels
modal run modal_app.py --mode test

# Test a single kernel
modal run modal_app.py --mode test --kernel rmsnorm
modal run modal_app.py --mode test --kernel decode
modal run modal_app.py --mode test --kernel prefill
modal run modal_app.py --mode test --kernel fused_proj_conv_silu

# Benchmark
modal run modal_app.py --mode bench --kernel rmsnorm
modal run modal_app.py --mode bench --kernel prefill
modal run modal_app.py --mode bench --kernel decode
modal run modal_app.py --mode bench --kernel fused_proj_conv_silu

# Interactive shell
modal shell modal_app.py::shell
```