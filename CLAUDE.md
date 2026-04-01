# MindSpore Development Guide

## Project Overview

MindSpore is an open-source deep learning framework developed by Huawei. It supports three backends: **Ascend** (Huawei NPU), **GPU** (NVIDIA CUDA), and **CPU**.

## Key Directories

```text
mindspore/
├── python/mindspore/          # Python API (user-facing)
│   ├── ops/                   # Operator Python wrappers & functional API
│   │   ├── function/          # ops.xxx() functions
│   │   ├── tensor_method.py   # Tensor.xxx() py_method implementations
│   │   └── auto_generate/     # Auto-generated from YAML
│   ├── nn/                    # Neural network layers (nn.Cell subclasses)
│   ├── graph/                 # JIT compilation (@ms.jit, grad, vmap)
│   └── common/                # Tensor, Parameter, dtype
├── ops/                       # Operator definitions
│   ├── api_def/               # API dispatch YAML (pyboost vs py_method, per backend)
│   ├── op_def/yaml/           # Op schema YAML (args, returns, C++ class names)
│   ├── infer/ops_func_impl/   # InferShape + InferType (C++)
│   └── kernel/                # C++ kernel implementations
│       ├── cpu/               # CPU kernels
│       ├── gpu/               # GPU CUDA kernels
│       └── ascend/            # Ascend aclnn/TBE kernels
├── core/ir/                   # FuncGraph, ANF nodes (C++ IR)
└── ccsrc/
    ├── frontend/              # JIT compiler, optimizer, autograd
    │   ├── optimizer/irpass/  # IR optimization passes
    │   └── expander/grad/     # Gradient expansion rules
    ├── pynative/              # PyNative execution engine
    └── backend/               # Backend execution
```

## Common Development Tasks

### Adding a new operator

1. Create `ops/api_def/{op}.yaml` (dispatch type + backend config)
2. Create `ops/op_def/yaml/{op}_op.yaml` (schema + C++ class names)
3. Create `ops/infer/ops_func_impl/{op}.h/.cc` (InferShape/InferType)
4. Create pyboost kernel files for each enabled backend
5. Run code generation: `cd mindspore/python/mindspore/ops_generate && python gen_ops.py`

See skill: `ms-operations-develop`

### Building

```bash
bash build.sh -e cpu -j8        # CPU only
bash build.sh -e gpu -j8        # GPU
bash build.sh -e ascend -j8     # Ascend
bash build.sh -e gpu -d -j8     # Debug mode
pip install output/mindspore_*.whl --force-reinstall
```

See skills: `ms-common-build`, `ms-common-upgrade-check-linux-cpu`, `ms-common-upgrade-check-macos`

### Debugging

```bash
export GLOG_v=2                          # Verbose logging
export MS_DEV_SAVE_GRAPHS=1              # Dump IR graphs
export MS_DEV_SAVE_GRAPHS_PATH=/tmp/ms  # IR dump location
```

See skill: `ms-common-debug`

## Dispatch Types

Operators use one of two dispatch mechanisms in `api_def/*.yaml`:

- `pyboost`: New fast path; requires C++ customize kernel files in `ops/kernel/{device}/pyboost/customize/`
- `py_method`: Legacy Python path; requires a Python function in `ops/tensor_method.py`

## Execution Modes

```python
import mindspore as ms
ms.set_context(mode=ms.PYNATIVE_MODE)   # Eager (default, debug-friendly)
ms.set_context(mode=ms.GRAPH_MODE)      # JIT compiled (production)
ms.set_context(device_target="GPU")     # or "Ascend", "CPU"
```

## Code Generation

Auto-generated files (don't edit manually):

- `mindspore/python/mindspore/ops/auto_generate/gen_ops_prim.py`
- `mindspore/python/mindspore/ops/auto_generate/gen_ops_def.py`
- `mindspore/ccsrc/pynative/op_function/auto_generate/pyboost_functions.cc`

Regenerate after changing YAML:

```bash
cd mindspore/python/mindspore/ops_generate
python gen_ops.py
```

## Testing

```bash
# System tests (Python)
pytest tests/st/ops/test_{op_name}.py -v -s

# Run with specific backend
pytest tests/st/ops/test_add.py -v --device=CPU

# C++ unit tests (built with -t ut flag)
./build/mindspore/tests/ut/cpp/mindspore_ut
```

Test files live in:

- `tests/st/` — Python system tests (most common)
- `tests/ut/` — C++ unit tests

## Naming Conventions

| Item | Convention | Example |
|------|-----------|---------|
| Python op function | `snake_case` | `ops.rsqrt` |
| Python Tensor method | `snake_case` | `tensor.reshape` |
| C++ class (OpFuncImpl) | `CamelCaseFuncImpl` | `RsqrtFuncImpl` |
| C++ kernel class | `CamelCase{Backend}` | `RsqrtAscend`, `RsqrtCPU` |
| op_def YAML file | `{op_name}_op.yaml` | `rsqrt_op.yaml` |
| api_def YAML file | `{op_name}.yaml` | `rsqrt.yaml` |
| pyboost customize file | `{op_name}.h/.cc` | `rsqrt.h`, `rsqrt.cc` |

## Important Environment Variables

```bash
GLOG_v=2                        # Log level (0=DEBUG..3=ERROR)
MS_DEV_SAVE_GRAPHS=1            # Dump IR graphs
MS_DEV_SAVE_GRAPHS_PATH=/path   # IR dump directory
MS_ASCEND_GLOBAL_LOG_LEVEL=0    # Ascend debug log
MS_MEMORY_POOL_RECYCLE=1        # Memory pool debug
```

## Skills Available

- `ms-common-build` — Build from source
- `ms-common-upgrade-check-linux-cpu` — Verify Linux x86_64 CPU build and CI after system upgrades
- `ms-common-upgrade-check-macos` — Verify macOS Apple Silicon build and CI after system upgrades
- `ms-common-conde-analysis` — Navigate and understand the codebase
- `ms-common-debug` — Debug issues
- `ms-common-document` — Write documentation
- `ms-common-pr-review` — Review pull requests
- `ms-common-write-skills` — Create a new skill for this project
- `ms-api-develop` — Use the Python API
- `ms-jit-fusion-pass` — Add IR optimization/fusion passes
- `ms-operations-develop` — Add/modify operators
