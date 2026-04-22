---
name: ms-pr-review
description: Review a MindSpore pull request thoroughly — check code correctness, style, test coverage, API docs, performance, and common pitfalls. Use when reviewing PRs, doing code review, checking if changes are correct, or evaluating a patch. Triggers on phrases like "review PR", "review this code", "check this change", "is this correct", "code review", "review patch", "look at this diff".
---

# MindSpore PR Review Guide

## Review Checklist

### 1. Correctness

**Operator changes:**
- [ ] `api_def` YAML: dispatch type matches what's actually implemented (`pyboost` requires C++ files, `py_method` requires Python function)
- [ ] `op_def` YAML: arg types match C++ signatures (`dtype: tensor` → `TensorPtr`, `dtype: int` → `Int64ImmPtr`, optional → `std::optional<...>`)
- [ ] InferShape returns correct output shape for all valid inputs including edge cases (empty tensor, 1D, broadcast)
- [ ] InferType correctly validates input dtype and returns output dtype
- [ ] Kernel implementation handles all dtypes listed in dispatch

**Gradient/autograd:**
- [ ] If `bprop_expander: False` in op_def but op needs gradients, check that bprop is registered elsewhere
- [ ] Gradient direction is correct (verify with a simple numerical check if uncertain)

**Python API:**
- [ ] `tensor_method.py` callback matches the signature in `api_def` YAML
- [ ] Default args in YAML match Python function defaults

### 2. Code Style

**C++:**
- [ ] Follows the existing naming: `{OpName}FuncImpl`, `{OpName}Ascend`, `{OpName}CPUCustomize`
- [ ] Header guards use the full path: `MINDSPORE_..._H_`
- [ ] Namespaces properly closed with comments: `}  // namespace mindspore::ops`
- [ ] No raw pointers where smart pointers are standard
- [ ] `MS_EXCEPTION_IF_NULL(ptr)` used before dereferencing

**Python:**
- [ ] Follows PEP 8 (spaces not tabs, 4-space indent)
- [ ] Docstring present for public APIs (see ms-document skill for format)
- [ ] Type hints present if existing code uses them

**YAML:**
- [ ] Consistent indentation (2 spaces)
- [ ] `#operator {name}` comment at top of op_def YAML
- [ ] Class names follow CamelCase: `RsqrtAscend`, not `rsqrt_ascend`

### 3. Test Coverage

- [ ] New operator has a test in `tests/st/ops/` or `tests/st/`
- [ ] Test covers: basic correctness, edge cases (empty tensor, boundary dtypes), error cases
- [ ] For multi-backend ops: test runs on Ascend, CPU, GPU (or at least one is covered)
- [ ] If modifying existing op: existing tests still pass

Typical test pattern:
```python
def test_{op_name}_basic():
    x = ms.Tensor([1.0, 2.0, 3.0], ms.float32)
    result = ops.{op_name}(x)
    expected = np.{equivalent}(x.asnumpy())
    np.testing.assert_allclose(result.asnumpy(), expected, rtol=1e-4)
```

### 4. API Documentation

For new ops or changed signatures:
- [ ] Python docstring present with: description, Args section, Returns section, Raises section (if applicable), example code
- [ ] Example in docstring actually works
- [ ] `op_def/yaml/doc/{op_name}_doc.yaml` exists for auto-generated docs (if applicable)

### 5. Common Bugs to Watch For

**Type handling:**
```cpp
// WRONG: unchecked cast
auto tensor_type = input_type->cast<TensorTypePtr>();

// RIGHT: checked cast
auto tensor_type = input_type->cast<TensorTypePtr>();
MS_EXCEPTION_IF_NULL(tensor_type);
```

**Optional arguments in C++:**
```cpp
// WRONG: accessing optional without checking
auto val = opt_arg.value();

// RIGHT:
if (opt_arg.has_value()) {
  auto val = opt_arg.value();
}
// or: opt_arg.value_or(default_val)
```

**Shape inference for broadcast ops:**
- Check that InferShape handles mismatched ranks with proper broadcasting logic
- Don't forget scalar inputs (shape = `{}`)

**YAML op dispatch:**
```yaml
# WRONG: pyboost without kernel files existing
Ascend: pyboost
CPU: pyboost    # This will fail at runtime if no CPU pyboost kernel

# CORRECT if CPU kernel not yet implemented:
CPU: py_method  # or CPU: None
```

### 6. Performance Considerations

- [ ] Is this on the hot path? New Python objects or copies in pynative mode are expensive
- [ ] For pyboost ops: is the kernel calling multiple sub-ops unnecessarily? A single ACLNN call is usually better than composing several
- [ ] Memory: check for unnecessary tensor copies (`contiguous()`, `clone()`)

### 7. Breaking Changes

- [ ] Are existing user APIs changed? If op signature changes, it's a breaking change
- [ ] Does `api_def` change affect tensor method names? (`Tensor.{op_name}()`)
- [ ] If removing `py_method` fallback: was it the only implementation for some backends?

### 8. File Completeness Check

For a new operator, verify all required files are present:
```bash
# Check from the repo root:
op="{op_name}"
ls mindspore/ops/api_def/${op}.yaml
ls mindspore/ops/op_def/yaml/${op}_op.yaml
ls mindspore/ops/infer/ops_func_impl/${op}.{h,cc}
# If CPU: pyboost:
ls mindspore/ops/kernel/cpu/pyboost/customize/${op}.{h,cc}
# If GPU: pyboost:
ls mindspore/ops/kernel/gpu/pyboost/customize/${op}.{h,cc}
```

## Review Comments Templates

**Missing test:**
> This PR adds `{op_name}` but I don't see a corresponding test in `tests/st/`. Please add a basic correctness test.

**Type mismatch:**
> The op_def YAML declares `dtype: int` for `{arg}`, but the Python wrapper passes it as a Python `bool`. Use `dtype: bool` or add a conversion.

**Missing null check:**
> `input_args[0]->GetType()` can return nullptr if abstract is not set. Please add `MS_EXCEPTION_IF_NULL` or an early return.

**pyboost without kernel:**
> `api_def/{op}.yaml` sets `CPU: pyboost`, but I don't see `kernel/cpu/pyboost/customize/{op}.cc`. Either add the kernel or use `py_method`.
