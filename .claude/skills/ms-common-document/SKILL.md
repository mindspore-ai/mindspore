---
name: ms-document
description: Write documentation for MindSpore operators, Python APIs, and modules — including docstrings, API doc YAML, and user guides. Use when adding docstrings to new ops, writing API reference docs, creating examples, documenting nn.Cell subclasses, or following MindSpore doc conventions. Triggers on phrases like "write docs", "add docstring", "document this op", "API docs", "write documentation", "add examples", "how to document".
---

# MindSpore Documentation Guide

## Operator Python Docstring Format

For operators in `mindspore/python/mindspore/ops/function/` or auto-generated classes:

```python
def rsqrt(input):
    r"""
    Returns reciprocal of the square root of a tensor element-wise.

    .. math::

        out_{i} = \frac{1}{\sqrt{input_{i}}}

    Args:
        input (Tensor): The input tensor. Supported dtypes: float16, float32, float64,
            bfloat16. Shape: :math:`(*)` where :math:`*` means any number of dimensions.

    Returns:
        Tensor, has the same shape and dtype as `input`.

    Raises:
        TypeError: If `input` is not a Tensor.
        TypeError: If dtype of `input` is not float16, float32, float64 or bfloat16.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> input = Tensor(np.array([0.25, 4.0, 1.0]), mindspore.float32)
        >>> output = ops.rsqrt(input)
        >>> print(output)
        [2.  0.5 1. ]
    """
```

### Section-by-Section Guide

**Opening description:**
- One-line summary of what the op does
- "element-wise" for elementwise ops
- Include math formula with `.. math::` if helpful

**Args:**
- `name (type): Description.` format
- List supported dtypes explicitly
- Describe shape with `:math:`(*, H, W)`` for fixed dims, `:math:`(*)`  for arbitrary
- For optional args: `name (Union[Tensor, None], optional): Description. Default: ``None``\`.`

**Returns:**
- Full description including shape relation to inputs
- `Tensor, has the same shape and dtype as \`input\`.` for elementwise
- For multiple returns: use numbered list

**Raises:**
- `TypeError`: for wrong input type
- `ValueError`: for wrong value/shape

**Supported Platforms:**
- Use backticks: ` ``Ascend`` ``GPU`` ``CPU`` `
- Only list platforms where `api_def` YAML has non-None dispatch

**Examples:**
- Always runnable (copy-paste should work)
- Import `mindspore`, `numpy`, `Tensor`, `ops`
- Show input creation, the op call, and `print(output)` with expected output

## nn.Cell Docstring Format

```python
class Linear(nn.Cell):
    r"""
    Applies a linear transformation to the input: :math:`y = xA^T + b`.

    Args:
        in_features (int): Size of each input sample.
        out_features (int): Size of each output sample.
        has_bias (bool): If set to False, the layer will not learn an additive bias.
            Default: ``True``.
        dtype (mindspore.dtype): Data type of the weight and bias. Default: ``mindspore.float32``.

    Inputs:
        - **x** (Tensor) - Input of shape :math:`(*, in\_features)`.

    Outputs:
        Tensor of shape :math:`(*, out\_features)`.

    Raises:
        TypeError: If `in_features` or `out_features` is not an int.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, nn
        >>> x = Tensor(np.array([[1.0, 2.0, 3.0]]), mindspore.float32)
        >>> net = nn.Dense(3, 4)
        >>> output = net(x)
        >>> print(output.shape)
        (1, 4)
    """
    def __init__(self, in_features, out_features, has_bias=True, dtype=ms.float32):
        ...
```

Note: For `nn.Cell`, use **Inputs/Outputs** instead of **Args/Returns** for the `construct()` method's parameters.

## op_def doc YAML

File: `mindspore/ops/op_def/yaml/doc/{op_name}_doc.yaml`

```yaml
{op_name}:
  description: |
    Returns reciprocal of the square root of a tensor element-wise.

    .. math::

        out_{i} = \frac{1}{\sqrt{input_{i}}}
  args:
    input:
      type: Tensor
      description: The input tensor. Supported dtypes float16, float32, float64, bfloat16.
  returns:
    output:
      type: Tensor
      description: Has the same shape and dtype as `input`.
  supported_platforms:
    - Ascend
    - GPU
    - CPU
  examples: |
    >>> import mindspore
    >>> import numpy as np
    >>> from mindspore import Tensor, ops
    >>> input = Tensor(np.array([0.25, 4.0]), mindspore.float32)
    >>> output = ops.rsqrt(input)
    >>> print(output)
    [2.  0.5]
```

## Math Notation

Use reStructuredText math directives:

```rst
Inline: :math:`x^2 + y^2`

Block:
.. math::

    \text{out}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}
```

Common patterns:
- Element-wise: `out_{i} = f(input_{i})`
- Matrix: `Y = XA^T + b`
- Reduction: `out = \sum_{i} input_{i}`

## dtype Documentation Patterns

```python
# Common dtype lists for the docstring:

# Float ops:
# Supported dtypes: float16, float32, float64, bfloat16.

# Integer ops:
# Supported dtypes: int8, int16, int32, int64, uint8.

# All numeric:
# Supported dtypes: int8, int16, int32, int64, uint8, uint16, uint32, uint64,
#     float16, float32, float64, bfloat16.

# Complex:
# Supported dtypes: complex64, complex128.
```

## Shape Documentation

```python
# For ops with specific shape requirements:
# - input (Tensor): Shape :math:`(N, C, H, W)`.
# - kernel_size (int or tuple[int]): The size of the sliding window.
#   If int, the height and width will be the same value.

# For broadcast-compatible ops:
# - x (Tensor): Shape :math:`(*)` where :math:`*` means any number of dimensions.
# - y (Tensor): Shape :math:`(*)`, must be broadcastable with `x`.
```

## Version Added

For new APIs, add version info:

```python
def new_op(input):
    """
    ...

    .. versionadded:: 2.3
        The `new_op` function.
    """
```

## Checklist

- [ ] Docstring present on the Python function/class
- [ ] Math formula included for non-obvious ops
- [ ] All args documented with type and description
- [ ] Returns section accurately describes output shape/dtype
- [ ] Raises section lists all explicitly raised exceptions
- [ ] Supported Platforms matches `api_def` YAML dispatch
- [ ] Example is copy-paste runnable and shows expected output
- [ ] Doc YAML in `op_def/yaml/doc/` if op uses auto-doc generation
