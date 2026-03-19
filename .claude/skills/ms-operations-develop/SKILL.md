---
name: ms-operations
description: Add a new operator or modify an existing operator in MindSpore. Use when adding a new op, changing op signatures, adding new backend support (Ascend/CPU/GPU), migrating py_method dispatch to pyboost, or modifying infer/kernel logic. Triggers on phrases like "add op", "new operator", "implement op", "add CPU support", "migrate to pyboost", "add GPU kernel", "register operator".
---

# MindSpore Operator Development

This skill covers the full lifecycle of adding or modifying a MindSpore operator.

## File Map

Every operator spans multiple files. Start by identifying which already exist:

```
mindspore/ops/api_def/{op}.yaml          # API-level dispatch (pyboost vs py_method, per backend)
mindspore/ops/op_def/yaml/{op}_op.yaml  # Op schema (args, returns, dispatch class names)
mindspore/ops/infer/ops_func_impl/{op}.h/.cc  # InferShape + InferType (C++)
mindspore/ops/kernel/cpu/pyboost/customize/{op}.h/.cc   # CPU pyboost kernel
mindspore/ops/kernel/gpu/pyboost/customize/{op}.h/.cc   # GPU pyboost kernel
mindspore/ops/kernel/ascend/aclnn/pyboost_impl/customize/{op}.h/.cc  # Ascend pyboost kernel
mindspore/ccsrc/include/pynative/utils/pyboost/customize/{op}.h      # Shared customize header
mindspore/ccsrc/pynative/utils/pyboost/customize/{op}.cc             # Shared customize impl
mindspore/python/mindspore/ops/tensor_method.py   # py_method fallback (if py_method dispatch)
```

## Step 1: api_def YAML

File: `mindspore/ops/api_def/{op_name}.yaml`

Controls how the op is dispatched at the API level.

```yaml
{op_name}:
  op_yaml: {op_name}_op.yaml        # references op_def YAML
  py_method: tensor_{op_name}       # Python callback function name
  Ascend: pyboost                   # or py_method
  CPU: pyboost                      # or py_method or None
  GPU: pyboost                      # or py_method or None
  interface: tensor, function       # tensor = Tensor.xxx(), function = ops.xxx()
```

For ops with multiple overloads (scalar vs tensor input), use a list:
```yaml
{op_name}:
  - op_yaml: {op_name}_tensor_op.yaml
    py_method: tensor_{op_name}_tensor
    Ascend: pyboost
    CPU: pyboost
    GPU: pyboost
    interface: tensor, function
  - op_yaml: {op_name}_scalar_op.yaml
    py_method: tensor_{op_name}_scalar
    ...
```

**`py_method` vs `pyboost`:**
- `pyboost` = new fast execution path; requires C++ customize kernel files
- `py_method` = legacy Python dispatch; just needs a Python function in `tensor_method.py`

## Step 2: op_def YAML

File: `mindspore/ops/op_def/yaml/{op_name}_op.yaml`

Defines the op's schema and C++ kernel class names.

```yaml
#operator {op_name}
{op_name}:
  args:
    input:
      dtype: tensor
    weight:
      dtype: float
      default: 1.0
    keepdim:
      dtype: bool
      default: False
  returns:
    output:
      dtype: tensor
  bprop_expander: False          # True if gradient is implemented via expander
  function:
    disable: True                # True for func-only ops (not a Primitive class)
  dispatch:
    enable: True
    Ascend: {OpName}Ascend       # C++ class name, e.g. RsqrtAscend
    CPU: {OpName}CPU             # CamelCase of op_name
    GPU: {OpName}GPU
```

**dtype options:** `tensor`, `float`, `int`, `bool`, `tuple[int]`, `list[int]`, `tuple[tensor]`, `number`

**For optional args:** add `default: None` and use `std::optional<TensorPtr>` in C++

## Step 3: Infer (InferShape + InferType)

File: `mindspore/ops/infer/ops_func_impl/{op_name}.h`

```cpp
#ifndef MINDSPORE_CORE_OPS_OPS_FUNC_IMPL_{OP_NAME_UPPER}_H_
#define MINDSPORE_CORE_OPS_OPS_FUNC_IMPL_{OP_NAME_UPPER}_H_

#include <vector>
#include "ops/ops_func_impl/op_func_impl.h"

namespace mindspore::ops {
class OPS_API {OpName}FuncImpl : public OpFuncImpl {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override;
  TypePtr InferType(const PrimitivePtr &primitive,
                    const std::vector<AbstractBasePtr> &input_args) const override;
  // For simple infer (pyboost path):
  ShapeArray InferShape(const PrimitivePtr &primitive,
                        const ValuePtrList &input_values) const override;
  TypePtrList InferType(const PrimitivePtr &primitive,
                        const ValuePtrList &input_values) const override;
};
}  // namespace mindspore::ops
#endif
```

File: `mindspore/ops/infer/ops_func_impl/{op_name}.cc`

```cpp
#include "infer/ops_func_impl/{op_name}.h"
#include <set>
#include "ops/ops_func_impl/simple_infer.h"

namespace mindspore::ops {

BaseShapePtr {OpName}FuncImpl::InferShape(...) const {
  // Same shape as input (for elementwise ops):
  return input_args[kIndex0]->GetShape()->Clone();
}

TypePtr {OpName}FuncImpl::InferType(...) const {
  // Validate and return output type
  auto input_type = input_args[kIndex0]->GetType();
  static const std::set<TypeId> valid_types = {kNumberTypeFloat32, kNumberTypeFloat16, ...};
  // ... type checking ...
  return input_type;
}

// For simple infer (ValuePtrList path):
TypePtrList {OpName}FuncImpl::InferType(const PrimitivePtr &, const ValuePtrList &input_values) const {
  const auto &x = input_values[kIndex0]->cast<tensor::TensorPtr>();
  return {x->Dtype()};
}

ShapeArray {OpName}FuncImpl::InferShape(const PrimitivePtr &, const ValuePtrList &input_values) const {
  const auto &x = input_values[kIndex0]->cast<tensor::TensorPtr>();
  return {x->shape()};
}

REGISTER_SIMPLE_INFER(kName{OpName}, {OpName}FuncImpl)
}  // namespace mindspore::ops
```

**Common TypeId values:** `kNumberTypeFloat16`, `kNumberTypeFloat32`, `kNumberTypeFloat64`, `kNumberTypeBFloat16`, `kNumberTypeInt32`, `kNumberTypeInt64`, `kNumberTypeBool`

## Step 4: Pyboost Customize Kernel (CPU)

**Header:** `mindspore/ops/kernel/cpu/pyboost/customize/{op_name}.h`

```cpp
#ifndef MINDSPORE_MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_PYBOOST_CUSTOMIZE_{OP_NAME_UPPER}_H_
#define MINDSPORE_MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_PYBOOST_CUSTOMIZE_{OP_NAME_UPPER}_H_

#include <memory>
#include "ir/tensor.h"
#include "include/pynative/utils/pyboost/op_runner.h"

namespace mindspore::kernel::pyboost {
tensor::TensorPtr {OpName}CPUCustomize(const std::shared_ptr<OpRunner> &op,
                                        const TensorPtr &input);
}
#endif
```

**Source:** `mindspore/ops/kernel/cpu/pyboost/customize/{op_name}.cc`

```cpp
#include "mindspore/ops/kernel/cpu/pyboost/customize/{op_name}.h"
#include "include/pynative/utils/pyboost/customize/{op_name}.h"  // shared header

namespace mindspore::kernel::pyboost {
tensor::TensorPtr {OpName}CPUCustomize(const std::shared_ptr<OpRunner> &op,
                                        const TensorPtr &input) {
  return {OpName}Customize(op, input);  // delegate to shared impl
}
}
```

## Step 5: Pyboost Customize Kernel (GPU)

Same pattern as CPU but with `GPU` suffix and `kGPU` device type if using a shared device-aware impl:

```cpp
// Header: mindspore/ops/kernel/gpu/pyboost/customize/{op_name}.h
tensor::TensorPtr {OpName}GPUCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &input);

// Source: mindspore/ops/kernel/gpu/pyboost/customize/{op_name}.cc
tensor::TensorPtr {OpName}GPUCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &input) {
  return {OpName}Customize(op, input);
}
```

## Step 6: Shared Customize Implementation

**Header:** `mindspore/ccsrc/include/pynative/utils/pyboost/customize/{op_name}.h`

```cpp
#include "include/runtime/hardware_abstract/device_context/device_context_manager.h"
#include "include/pynative/utils/pyboost/op_runner.h"

namespace mindspore::kernel::pyboost {
tensor::TensorPtr PYBOOST_API {OpName}Customize(const std::shared_ptr<OpRunner> &op,
                                                 const TensorPtr &input);
}
```

**Source:** `mindspore/ccsrc/pynative/utils/pyboost/customize/{op_name}.cc`

```cpp
#include "include/pynative/utils/pyboost/customize/{op_name}.h"
#include "mindspore/ccsrc/include/pynative/utils/pyboost/functions/auto_generate/functions.h"

namespace mindspore::kernel::pyboost {
tensor::TensorPtr {OpName}Customize(const std::shared_ptr<OpRunner> &op, const TensorPtr &input) {
  // Call existing mint functions or implement using lower-level ops:
  auto out = some_existing_op(input, ...);
  op->set_outputs({out});
  return out;
}
}
```

## Step 7: Python py_method Fallback (if needed)

Only needed when `py_method` dispatch is used (vs `pyboost`).

Add to `mindspore/python/mindspore/ops/tensor_method.py`:

```python
def tensor_{op_name}(input, ...):
    r"""
    For details, please refer to :func:`mindspore.ops.{op_name}`.
    """
    return ops.{op_name}(input, ...)
```

## Common Patterns

### Elementwise op (same shape/type output)
- InferShape: `return input_args[0]->GetShape()->Clone()`
- InferType: `return input_args[0]->GetType()`
- Customize: call the corresponding mint/aten function

### Reduction op (axis parameter, keepdim)
- InferShape: compute reduced shape based on `axis` and `keepdim` args
- Look at existing reductions like `any`, `sum`, `reduce_mean` for reference

### View op (no copy, just metadata change)
- Add `view: True` to op_def YAML
- No kernel needed for reshape-style ops

## Checklist

- [ ] `api_def/{op}.yaml` created with correct dispatch types
- [ ] `op_def/yaml/{op}_op.yaml` created with all args and dispatch class names
- [ ] `infer/ops_func_impl/{op}.h/.cc` with InferShape + InferType
- [ ] CPU pyboost files (if `CPU: pyboost`)
- [ ] GPU pyboost files (if `GPU: pyboost`)
- [ ] Ascend pyboost files (if `Ascend: pyboost`)
- [ ] Shared customize header + impl
- [ ] Python `tensor_method.py` entry (if any backend uses `py_method`)
- [ ] Unit test in `tests/st/`
