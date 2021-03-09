# Copyright 2020-2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Packaged operations based on MindSpore."""

__all__ = [
    'absolute',
    'arange',
    'argmax',
    'argmin',
    'argsort',
    'assign',
    'intersection',
    'matmul',
    'maximum',
    'minimum',
    'mean',
    'mul',
    'sort',
    'sqrt',
    'squeeze',
    'tile',
    'reshape',
    'zeros',
    'zeros_like',
    'softmax',
    'Tensor',
    'summation'
]

from typing import List, Tuple, Union, Callable

import numpy as np

import mindspore
from mindspore import nn
import mindspore.ops.operations as op

_Axis = Union[int, Tuple[int, ...], List[int]]
_Idx = Union[int, mindspore.Tensor, Tuple[int, ...], Tuple[mindspore.Tensor, ...]]
_Number = Union[int, float, np.int, np.float]
_Shape = Union[int, Tuple[int, ...]]
Tensor = mindspore.Tensor


def absolute(inputs: Tensor) -> Tensor:
    """Get the absolute value of a tensor value."""
    abs_op = op.Abs()
    outputs = abs_op(inputs)
    return outputs


def arange(
        start: _Number,
        end: _Number,
        step: _Number = 1,
        dtype: mindspore.dtype = None) -> Tensor:
    """Get the arange value of tensor."""
    nums = np.arange(start=start, stop=end, step=step, dtype=np.int32)
    nums = mindspore.Tensor(nums, dtype=dtype)
    return nums


def argmax(inputs: Tensor, axis: int = -1, keep_dims: bool = False) -> Tensor:
    """Returns the indices of the maximum values along an axis."""
    inputs_np = inputs.asnumpy()
    outputs = np.argmax(inputs_np, axis=axis)

    if keep_dims:
        outputs = np.expand_dims(outputs, axis=axis)

    return mindspore.Tensor(outputs, mindspore.int32)


def argmin(inputs: Tensor, axis: int = -1, keep_dims: bool = False) -> Tensor:
    """Returns the indices of the minimum values along an axis."""
    inputs_np = inputs.asnumpy()
    outputs = np.argmin(inputs_np, axis=axis)

    if keep_dims:
        outputs = np.expand_dims(outputs, axis=axis)

    return mindspore.Tensor(outputs, mindspore.int32)


def argsort(inputs: Tensor, axis: int = -1, descending: bool = False) -> Tensor:
    """Returns the indices that would sort an array."""
    inputs_np = inputs.asnumpy()
    factor = -1 if descending else 1
    indices_np = np.argsort(factor * inputs_np, axis=axis)
    indices = mindspore.Tensor(indices_np, dtype=mindspore.int32)
    return indices


def assign(inputs: Tensor, idx: _Idx, value: Tensor) -> Tensor:
    """Assign a tensor value to the given tensor and index."""
    inputs_np = inputs.asnumpy()
    if isinstance(idx, Tensor):
        idx = idx.asnumpy()
    value_np = value.asnumpy()
    inputs_np[idx] = value_np
    outputs = mindspore.Tensor(inputs_np)
    return outputs


def intersection(*inputs: Tensor) -> Tensor:
    """Get the intersection value by the given tensor list."""
    outputs_np = np.ones_like(inputs[0])
    for inp in inputs:
        outputs_np &= inp.asnumpy()
    outputs = mindspore.Tensor(outputs_np)
    return outputs


def matmul(inputs_x: Tensor, inputs_y: Tensor) -> Tensor:
    """Multiplies matrix `inputs_x` and matrix `inputs_y`."""
    matmul_op = op.MatMul()
    outputs = matmul_op(inputs_x, inputs_y)
    return outputs


def maximum(inputs: Tensor, axis: _Axis = (), keep_dims: bool = False) -> Tensor:
    """Reduces a dimension of a tensor by the maximum value in this dimension."""
    max_op = op.ReduceMax(keep_dims)
    outputs = max_op(inputs, axis)
    return outputs


def minimum(inputs: Tensor, axis: _Axis = (), keep_dims: bool = False) -> Tensor:
    """Reduces a dimension of a tensor by the minimum value in the dimension."""
    max_op = op.ReduceMin(keep_dims)
    outputs = max_op(inputs, axis)
    return outputs


def mean(inputs: Tensor, axis: _Axis = (), keep_dims: bool = False) -> Tensor:
    """Reduces a dimension of a tensor by averaging all elements in the dimension."""
    mean_op = op.ReduceMean(keep_dims)
    outputs = mean_op(inputs, axis)
    return outputs


def mul(inputs_x: Tensor, inputs_y: Tensor) -> Tensor:
    """
    Multiplies two tensors element-wise.

    Inputs of `input_x` and `input_y` comply with the implicit type conversion rules to make the data types consistent.
    The inputs must be two tensors or one tensor and one scalar.
    When the inputs are two tensors,
    dtypes of them cannot be both bool, and the shapes of them could be broadcast.
    When the inputs are one tensor and one scalar,
    the scalar could only be a constant.

    Inputs:
        - **input_x** (Union[Tensor, Number, bool]) - The first input is a number or
          a bool or a tensor whose data type is number or bool.
        - **input_y** (Union[Tensor, Number, bool]) - The second input is a number or
          a bool when the first input is a tensor or a tensor whose data type is number or bool.

    Outputs:
        Tensor, the shape is the same as the one after broadcasting,
        and the data type is the one with higher precision or higher digits among the two inputs.
    """
    mul_op = op.Mul()
    outputs = mul_op(inputs_x, inputs_y)
    return outputs


def sort(inputs: Tensor, axis: _Axis = -1, descending: bool = False) -> Tensor:
    """Return a sorted copy of an array."""
    inputs_np = inputs.asnumpy()
    outputs_np = np.sort(inputs_np, axis=axis)
    if descending:
        outputs_np = np.flip(outputs_np, axis=axis)
    outputs = mindspore.Tensor(outputs_np)
    return outputs


def squeeze(inputs: Tensor, axis: _Axis = ()):
    """Returns a tensor with the same type but dimensions of 1 are removed based on `axis`."""
    squeeze_op = op.Squeeze(axis)
    outputs = squeeze_op(inputs)
    return outputs


def tile(inputs: Tensor, shape: Tuple[int, ...]) -> Tensor:
    """Replicates a tensor with given multiples times."""
    tile_op = op.Tile()
    outputs = tile_op(inputs, shape)
    return outputs


def reshape(inputs: Tensor, shape: _Shape) -> Tensor:
    """Reshapes input tensor with the same values based on a given shape tuple."""
    if isinstance(shape, int):
        shape = (shape,)
    return op.Reshape()(inputs, shape)


def zeros(shape: _Shape, dtype: mindspore.dtype = None) -> Tensor:
    """Return a new array of given shape and type, filled with zeros."""
    outputs = np.zeros(shape)
    return mindspore.Tensor(outputs, dtype=dtype)


def zeros_like(inputs: Tensor, dtype: mindspore.dtype = None) -> Tensor:
    """Return an array of zeros with the same shape and type as a given array."""
    inputs_np = inputs.asnumpy()
    outputs_np = np.zeros_like(inputs_np)
    outputs = mindspore.Tensor(outputs_np, dtype)
    return outputs


def random(shape: _Shape, dtype: mindspore.dtype = None) -> Tensor:
    """Return random floats in the half-open interval [0.0, 1.0)."""
    outputs_np = np.random.random(shape)
    outputs = mindspore.Tensor(outputs_np, dtype)
    return outputs


def randint(low: int, high: int, shape: _Shape, dtype: mindspore.dtype = mindspore.int8) -> Tensor:
    """Return random integers from `low` (inclusive) to `high` (exclusive)."""
    outputs_np = np.random.randint(low, high, size=shape)
    outputs = mindspore.Tensor(outputs_np, dtype=dtype)
    return outputs


def softmax(axis: int = -1) -> Callable:
    """Softmax activation function."""
    func = nn.Softmax(axis=axis)
    return func


def summation(inputs: Tensor, axis: _Axis = (), keep_dims: bool = False) -> Tensor:
    """Reduces a dimension of a tensor by summing all elements in the dimension."""
    sum_op = op.ReduceSum(keep_dims)
    outputs = sum_op(inputs, axis)
    return outputs


def stack(inputs: List[Tensor], axis: int) -> Tensor:
    """Stacks a list of tensors in specified axis."""
    stack_op = op.Stack(axis)
    outputs = stack_op(inputs)
    return outputs


def sqrt(inputs: Tensor) -> Tensor:
    """Returns square root of a tensor element-wise."""
    sqrt_op = op.Sqrt()
    return sqrt_op(inputs)
