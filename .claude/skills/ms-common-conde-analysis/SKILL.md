---
name: ms-code-analysis
description: Understand, trace, and navigate the MindSpore codebase. Use when tracing how a Python API call maps to a C++ kernel, understanding graph compilation and execution flow, navigating from an op name to its implementation files, comparing pynative vs graph mode behavior, or understanding the ANF IR / FuncGraph. Triggers on phrases like "how does X work", "trace execution", "where is X implemented", "find the kernel for", "how does pynative work", "what is FuncGraph", "understand the code", "how is X dispatched".
---

# MindSpore Code Analysis

## Execution Mode Overview

MindSpore has two execution modes — understanding which you're in is the first step:

| Mode | How to set | Characteristics |
|------|-----------|-----------------|
| **PyNative** (eager) | `ms.set_context(mode=ms.PYNATIVE_MODE)` | Runs ops immediately, Python-level control flow, easier to debug |
| **Graph** (JIT) | `ms.set_context(mode=ms.GRAPH_MODE)` or `@ms.jit` | Compiles to FuncGraph, optimized execution, restricted Python |

## Tracing a Python API Call

### From `ops.xxx(tensor)` to kernel

```
ops.add(x, y)
  → mindspore/python/mindspore/ops/function/math_func.py   (Python wrapper)
  → mindspore/python/mindspore/ops/auto_generate/gen_ops_prim.py  (auto-gen Primitive class)
  → mindspore/ops/api_def/add.yaml                          (dispatch decision)
  → [if pyboost] → C++ pyboost execution path:
      mindspore/ccsrc/pynative/op_function/auto_generate/pyboost_functions.cc
      → mindspore/ops/kernel/{cpu|gpu|ascend}/pyboost/customize/add.cc
  → [if py_method] → tensor_method.py callback
      → eventually calls another op
```

### From `Tensor.xxx()` to kernel

Same as `ops.xxx()` — `Tensor` methods are auto-generated from `api_def/*.yaml` with `interface: tensor`.

### Key files for an operator named `foo`

```bash
# Find all files related to operator 'foo':
find mindspore/ops -name "*foo*"
grep -r "kNameFoo\|\"Foo\"\|prim::kPrimFoo" mindspore/core/ops/ --include="*.h" -l
```

## FuncGraph / ANF IR Concepts

In Graph mode, Python functions are compiled to an ANF (A-Normal Form) graph:

```
FuncGraph
├── Parameter nodes  — inputs
├── ValueNode(s)     — constants (weights, scalars)
└── CNode(s)         — operations: (func, arg1, arg2, ...)
    └── Each CNode has: func (a Primitive or sub-FuncGraph) + inputs
```

### Dump the FuncGraph to inspect it

```python
import os
os.environ["MS_DEV_SAVE_GRAPHS"] = "1"       # Dump IR graphs
os.environ["MS_DEV_SAVE_GRAPHS_PATH"] = "/tmp/ms_graphs"
import mindspore as ms
ms.set_context(mode=ms.GRAPH_MODE)
# ... run your code ...
# Then check /tmp/ms_graphs/ for .dot or .ir files
```

Or via environment:
```bash
export MS_DEV_SAVE_GRAPHS=1
export MS_DEV_SAVE_GRAPHS_PATH=/tmp/ms_graphs
python your_script.py
```

## Directory Structure Reference

```
mindspore/
├── python/mindspore/          Python API (user-facing)
│   ├── ops/                   Operator Python wrappers
│   │   ├── function/          Functional API (ops.add, ops.matmul, ...)
│   │   ├── tensor_method.py   Tensor.xxx() py_method implementations
│   │   └── auto_generate/     Auto-generated from YAML
│   ├── nn/                    Neural network layers (nn.Cell subclasses)
│   ├── graph/                 JIT/graph compilation API
│   └── common/                Tensor, Parameter, dtype
├── ops/                       Operator definitions
│   ├── api_def/               API-level dispatch YAML
│   ├── op_def/yaml/           Op schema YAML
│   ├── infer/ops_func_impl/   InferShape + InferType (C++)
│   └── kernel/                C++ kernel implementations
│       ├── cpu/               CPU kernels
│       ├── gpu/               GPU kernels (CUDA)
│       └── ascend/            Ascend kernels (aclnn)
├── core/ir/                   FuncGraph, ANF nodes (C++ IR)
├── ccsrc/
│   ├── frontend/              JIT compiler, optimizer, grad
│   │   ├── optimizer/irpass/  IR optimization passes
│   │   └── expander/grad/     Gradient expansion rules
│   ├── pynative/              PyNative execution engine
│   │   ├── forward/           Forward execution
│   │   └── backward/          Gradient computation
│   └── backend/               Backend execution (Ascend, GPU, CPU)
└── tests/
    ├── ut/                    C++ unit tests
    └── st/                    Python system tests
```

## Finding Kernel Implementations

Given an op `rsqrt`:

```bash
# 1. Find op_def YAML (schema)
cat mindspore/ops/op_def/yaml/rsqrt_op.yaml
# → tells you C++ dispatch class name: dispatch.Ascend = RsqrtAscend

# 2. Find infer implementation
ls mindspore/ops/infer/ops_func_impl/rsqrt.*

# 3. Find Ascend kernel
find mindspore/ops/kernel/ascend -name "*rsqrt*"

# 4. Find CPU kernel
find mindspore/ops/kernel/cpu -name "*rsqrt*"
```

## Pynative vs Graph Mode Internals

### PyNative (eager)
- Entry: `mindspore/ccsrc/pynative/forward/`
- Each op call → `PyNativeExecutor::RunOp()` → kernel dispatch → immediate execution
- Gradient tape: `mindspore/ccsrc/pynative/backward/grad.cc`

### Graph Mode
- Entry: Python `@ms.jit` decorator or `ms.set_context(mode=GRAPH_MODE)`
- AST parsing: `mindspore/python/mindspore/graph/_parse/`
- FuncGraph construction: `mindspore/core/ir/func_graph.cc`
- Optimization passes: `mindspore/ccsrc/frontend/optimizer/`
- Backend compilation: `mindspore/ccsrc/backend/`

## Gradient / Autograd Tracing

```python
# How grad() works:
from mindspore import grad

def f(x):
    return ops.sin(x)

grad_f = grad(f)   # Returns gradient function
```

- `grad()` → `GradOperation` in `ops/composite/base.py`
- Backward rules: `mindspore/ccsrc/frontend/expander/grad/`
- Backward pass execution: `mindspore/ccsrc/pynative/backward/`

To find the gradient rule for an op like `sin`:
```bash
grep -r "kNameSin\|\"Sin\"" mindspore/ccsrc/frontend/expander/grad/ --include="*.cc" -l
```

## Logging and Debug Output

```bash
# Set log level (0=DEBUG, 1=INFO, 2=WARNING, 3=ERROR)
export GLOG_v=1

# Show all ops being executed in pynative mode
export GLOG_v=2

# Dump IR at each optimization pass
export MS_DEV_SAVE_GRAPHS=3
```
