---
name: ms-debug
description: Debug MindSpore issues — execution errors, shape mismatches, gradient problems, graph compilation failures, kernel crashes, and device errors. Use when hitting runtime errors, unexpected outputs, pynative vs graph mode differences, NaN/Inf values, or performance issues. Triggers on phrases like "debug", "error", "crash", "wrong output", "shape error", "gradient wrong", "nan/inf", "compilation error", "runtime error", "how to debug".
---

# MindSpore Debugging Guide

## Quick Diagnosis

Start by identifying which phase the error occurs in:

```
Python API call
  → PyNative eager execution?   → kernel error / shape mismatch
  → @ms.jit / Graph mode?       → compilation error OR runtime error
  → Gradient computation?       → autograd error
```

## Environment Variables for Debugging

Set these before running your script:

```bash
# Logging verbosity (0=DEBUG, 1=INFO, 2=WARNING, 3=ERROR)
export GLOG_v=2

# Dump FuncGraph IR at each optimization pass (very verbose)
export MS_DEV_SAVE_GRAPHS=1
export MS_DEV_SAVE_GRAPHS_PATH=/tmp/ms_graphs

# Dump level: 1=basic, 2=more detail, 3=all passes
export MS_DEV_SAVE_GRAPHS=3

# Enable extra runtime checks
export MS_DEV_ENABLE_FALLBACK_RUNTIME=1

# Disable operator fusion (helps isolate kernel bugs)
export MS_GRAPH_KERNEL_FLAGS="--disable_expand_ops=ALL"
```

## Switching to PyNative for Debugging

When graph mode crashes, the first step is to reproduce in pynative mode:

```python
import mindspore as ms

# Switch to pynative for easier debugging
ms.set_context(mode=ms.PYNATIVE_MODE)

# Now you can add print statements anywhere:
def my_net_construct(x):
    out = ops.sin(x)
    print("after sin:", out.shape, out.dtype)  # works in pynative
    out = ops.relu(out)
    return out
```

## Common Errors and Fixes

### Shape Mismatch

```
ValueError: For 'MatMul', the last dim of 'x' must equal the second-to-last dim of 'y'
```

Debug by printing shapes:
```python
ms.set_context(mode=ms.PYNATIVE_MODE)
print(x.shape, y.shape)   # identify the mismatch
```

In graph mode, shapes are often determined at compile time. Use:
```python
@ms.jit
def f(x, y):
    ms.ops.print("x shape:", x.shape)  # ops.print works in graph mode
    return ops.matmul(x, y)
```

### NaN / Inf Values

```python
# Check for NaN after each op (pynative only):
def check_nan(tensor, name):
    if ops.any(ops.isnan(tensor)):
        raise ValueError(f"NaN detected in {name}")

# Or set NaN detection at context level:
ms.set_context(check_bprop=True)  # checks gradient for NaN/Inf
```

Enable NaN kernel detection (Ascend):
```bash
export MS_ASCEND_CHECK_OVERFLOW_MODE=INFNAN_MODE
```

### Graph Mode Compilation Error

```
RuntimeError: Compile graph failed...
```

1. Check the IR dump at `/tmp/ms_graphs/` for the last good graph before failure
2. Simplify: comment out parts of the function until compilation succeeds, then narrow down
3. Common causes:
   - Dynamic Python constructs not supported in graph mode (`if/while` with tensor conditions need special handling)
   - Using `print()` instead of `ops.print()`
   - Returning non-Tensor Python objects

```python
# WRONG in graph mode:
@ms.jit
def f(x):
    if x.sum() > 0:   # tensor condition = error
        return x
    return -x

# RIGHT:
@ms.jit
def f(x):
    mask = (x.sum() > 0).astype(ms.float32)
    return x * mask - x * (1 - mask)
```

### Gradient Errors

```
RuntimeError: Bprop of Primitive ... is not implemented
```

The op doesn't have a gradient rule. Solutions:
1. Check if `bprop_expander` is set correctly in op_def YAML
2. Look for the gradient in `mindspore/ccsrc/frontend/expander/grad/`
3. Use `ops.stop_gradient(x)` to stop gradient flow through that op

For numerical gradient checking:
```python
from mindspore.common.api import _pynative_executor
import numpy as np

def numerical_grad(f, x, eps=1e-4):
    x_np = x.asnumpy()
    grad = np.zeros_like(x_np)
    for i in range(x_np.size):
        x_plus = x_np.copy().flatten()
        x_plus[i] += eps
        x_minus = x_np.copy().flatten()
        x_minus[i] -= eps
        f_plus = f(ms.Tensor(x_plus.reshape(x_np.shape))).asnumpy().sum()
        f_minus = f(ms.Tensor(x_minus.reshape(x_np.shape))).asnumpy().sum()
        grad.flat[i] = (f_plus - f_minus) / (2 * eps)
    return grad

# Compare with autograd:
x = ms.Tensor(np.random.randn(3, 4), ms.float32)
auto_grad = ms.grad(f)(x).asnumpy()
num_grad = numerical_grad(f, x)
print("max diff:", np.max(np.abs(auto_grad - num_grad)))  # should be < 1e-3
```

### Device / Kernel Errors

```
RuntimeError: [DEVICE_NOT_SUPPORTED] Op 'xxx' is not supported on GPU
```

- Check `api_def` YAML: `GPU: None` means not implemented
- Verify backend: `ms.get_context("device_target")`
- For Ascend ACLNN errors: check `MS_ASCEND_GLOBAL_LOG_LEVEL`

```bash
# Ascend debug logging
export MS_ASCEND_GLOBAL_LOG_LEVEL=0   # DEBUG
export ASCEND_GLOBAL_EVENT_ENABLE=1
```

### Memory Errors

```
RuntimeError: Out of memory on device
```

```python
# Check tensor sizes:
print(f"tensor size: {x.nbytes / 1e9:.2f} GB")

# Force synchronous execution (helps pinpoint the op causing OOM):
ms.set_context(mode=ms.PYNATIVE_MODE)

# Enable memory pool debug:
import os
os.environ["MS_MEMORY_POOL_RECYCLE"] = "1"
```

## Inspecting IR Graphs

After setting `MS_DEV_SAVE_GRAPHS=1`, check `/tmp/ms_graphs/`:

```bash
ls /tmp/ms_graphs/
# Files: rank_0/
#   1_parse.ir             -- after Python parse
#   2_symbol_resolve.ir    -- after symbol resolution
#   10_opt_a.ir            -- after optimization pass A
#   ...
#   final.ir               -- final graph before execution
```

Read `.ir` files to see the ANF graph:
```
%1(x) = Load[...]
%2 = Sin(%1)
%3 = ReLU(%2)
return %3
```

## Reproducibility

```python
# Fix random seeds for reproducible runs:
import numpy as np
import mindspore as ms
np.random.seed(42)
ms.set_seed(42)
```

## Pynative vs Graph Mode Difference

If results differ between modes:
```python
x = ms.Tensor(np.random.randn(2, 3), ms.float32)

ms.set_context(mode=ms.PYNATIVE_MODE)
out_pynative = my_func(x)

ms.set_context(mode=ms.GRAPH_MODE)
out_graph = ms.jit(my_func)(x)

print("max diff:", (out_pynative - out_graph).abs().max())
```

If outputs differ, the graph mode likely has a compilation issue — dump the IR and check pass outputs.
