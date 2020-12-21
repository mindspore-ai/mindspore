# Copyright 2020 Huawei Technologies Co., Ltd
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
"""internal utility functions"""
from functools import partial

import numpy as onp

import mindspore.context as context
from ..common import Tensor
from ..ops import operations as P
from ..ops import functional as F
from ..ops.primitive import constexpr
from ..common import dtype as mstype

from .dtypes import dtype_tuple, all_types, dtype_map

@constexpr
def _check_shape_compile(shape):
    """check the shape param to match the numpy style inside the graph"""
    if not isinstance(shape, (int, tuple, list)):
        raise TypeError(
            f"only int, tuple and list are allowed for shape, but got {type(shape)}")
    if isinstance(shape, int):
        shape = (shape,)
    if isinstance(shape, list):
        shape = tuple(shape)
    return shape


@constexpr
def _check_is_int(x):
    """Check the type of x is int."""
    if isinstance(x, int):
        return True
    raise TypeError(f"integer argument expected, but got {type(x)}.")


@constexpr
def _check_start_normalize(start, ndim):
    """check and normalize start argument for rollaxis."""
    if start < -ndim or start > ndim:
        raise ValueError(
            f"For rollaxis, start {start} is out of bounds. Ranging from {-ndim} to {ndim} is allowed.")
    if start < 0:
        start = start + ndim
    return start


@constexpr
def _check_axes_range(axes, ndim):
    """
    Check axes are within the number of dimensions of tensor x and normalize the negative axes.
    Args:
        axes (Union[int, tuple(int), list(int)]): Axes of the tensor.
        ndim (int): The number of dimensions of the tensor.
    Return:
        Axes (Union[int, tuple(int)]). If input is integer, return integer, else tuple.
    """
    if not isinstance(axes, int) and not isinstance(axes, tuple) and not isinstance(axes, list):
        raise TypeError(
            f"int, tuple(int) or list(int) expected, but got {type(axes)}.")
    low = -ndim
    up = ndim - 1
    if low > up:
        raise ValueError(
            f"Lower bound {low} and upper bound {up} of axes are not allowed.")
    if isinstance(axes, int):
        if axes < low or axes > up:
            raise ValueError(
                f"axis {axes} is out of bounds for tensor of dimension {ndim}.")
        return axes if axes >= 0 else axes + ndim
    new_axes = []
    for item in axes:
        if not isinstance(item, int):
            raise TypeError(
                f"int in tuple or list expected, but got {type(item)}.")
        if item < low or item > up:
            raise ValueError(
                f"axis {item} in {axes} is out of bounds for tensor of dimension {ndim}.")
        new_axes.append(item if item >= 0 else item + ndim)
    return tuple(new_axes)


def _check_shape_contain_zero(shp):
    """Check whether shape contains 0"""
    if isinstance(shp, int):
        return shp == 0
    if isinstance(shp, (list, tuple)):
        for s in shp:
            if s == 0:
                return True
    return False


def _check_shape(shape):
    """check the shape param to match the numpy style outside the graph"""
    if not isinstance(shape, (int, tuple, list)):
        raise TypeError(
            f"only int, tuple and list are allowed for shape, but got {type(shape)}")
    if isinstance(shape, int):
        shape = (shape,)
    if isinstance(shape, list):
        shape = tuple(shape)
    return shape


def _check_dtype(dtype):
    """check the input dtype and make conversions"""
    # convert the string dtype to mstype.dtype
    if isinstance(dtype, str):
        dtype = dtype.lower()
        dtype = dtype_map[dtype]
    elif isinstance(dtype, type):
        if dtype is int:
            dtype = mstype.int32
        if dtype is float:
            dtype = mstype.float32
        if dtype is bool:
            dtype = mstype.bool_
    if dtype not in dtype_tuple:
        raise TypeError(
            f"only {all_types} are allowed for dtype, but got {type(dtype)}")
    return dtype


def _deep_list(array_like):
    """convert nested tuple/list mixtures to pure nested list"""
    if isinstance(array_like, (list, tuple)):
        return list(map(_deep_list, array_like))
    return array_like


def _deep_tensor_to_nparray(array_like):
    """
    convert a nested list of tensor to nested list of np_array.

    Args:
        array_like(list(tensor)): In any format of nested lists that may contain
        tensors.

    Returns:
        array_like(list(np_array)): Formatted array that can be directly processed
            by numpy.array(), with all tensor elements converted to numpy_array.
    """
    # Recursively check whether each element is a tensor or not, if is tensor,
    # convert it to a numpy array in place
    if isinstance(array_like, Tensor):
        return array_like.asnumpy()

    if isinstance(array_like, list):
        for idx, value in enumerate(array_like):
            array_like[idx] = _deep_tensor_to_nparray(value)

    return array_like


def _check_input_for_asarray(array_like):
    """check whether array_like argument is a valid type for np.asarray conversion"""
    if isinstance(array_like, (Tensor, list, tuple, int, float, bool, onp.ndarray)):
        return True
    raise TypeError(
        "input data must be `int`, `float`, `bool`, `Tensor`, `list`, `tuple`" + \
            f"or numpy.ndarray, but got {type(array_like)}")


def _cast_to(array, dtype):
    """cast the input to specified dtype"""
    cast = P.Cast()
    return cast(array, dtype)


def _is_scalar(shape):
    """check whether input shape is a scalar"""
    return F.shape_mul(shape) == 1


@constexpr
def _get_device_compile():
    """Get the current device (`GPU`, `CPU`, `Ascend`)"""
    return context.get_context('device_target')


def _get_device():
    """Get the current device (`GPU`, `CPU`, `Ascend`)"""
    return context.get_context('device_target')


def _covert_list_tensor_to_tuple_tensor(list_of_tensor):
    """Convert a list of tensor to a tuple of tensor"""
    tuple_of_tensor = ()
    for tensor in list_of_tensor:
        tuple_of_tensor += (tensor,)
    return tuple_of_tensor


def _get_mode():
    """Get the current mode (0 is Graph mode, 1 is PyNative mode)"""
    return context.get_context('mode')


@constexpr
def _reverse_index(idx, arr):
    """
    Returns 1 if shape[idx:] is broadcastable to shape_out[idx:],
    2 situations if the function returns 1:
    - 1. Tensor's shape has 1 at the designated dimension.
    - 2. Tensor's dimension is less than the designated idx. (The Tensor shape
         has been reversed)
    For both cases, 2 tensors are broadcastable.
    otherwise returns the element at position of shape
    """
    if len(arr) <= idx:
        return 1
    return arr[-1 - idx]


@constexpr
def _infer_out_shape(device, *shapes):
    """
    Returns shape of output after broadcasting
    Raises ValueError if shape1 and shape2 cannot be broadcast
    """
    shapes_unbroadcastable = False
    cpu_shapes_different = False
    contains_scalar = any(_is_scalar(shape) for shape in shapes)
    ndim_max = max(map(len, shapes))
    shape_out = [0]*ndim_max
    i = 0
    for i in range(ndim_max):
        shape_out[-1 - i] = max(map(partial(_reverse_index, i), shapes))
        for shape in shapes:
            if _reverse_index(i, shape) != shape_out[-1 - i]:
                if _reverse_index(i, shape) != 1:
                    shapes_unbroadcastable = True
                if device == 'CPU' and not contains_scalar:
                    cpu_shapes_different = True
    if not shapes_unbroadcastable and not cpu_shapes_different:
        return tuple(shape_out)
    if shapes_unbroadcastable:
        raise ValueError(
            f'operands could not be broadcast together with shapes {*shapes,}')
    raise ValueError('broadcasting is currently not supported on CPU. Non-scalar' + \
        f'operands must have the same shape, but got {*shapes,}')


@constexpr
def _check_axis_in_range(axis, ndim):
    """Checks axes are with the bounds of ndim"""
    if -ndim <= axis < ndim:
        return True
    raise ValueError(
        f'axis {axis} is out of bounds for array of dimension {ndim}')


@constexpr
def _check_axis_valid(axes, ndim):
    """
    Checks axes are valid given ndim, and returns axes that can be passed
    to the built-in operator (non-negative, int or tuple)
    """
    if isinstance(axes, int):
        _ = _check_axis_in_range(axes, ndim)
        return (axes % ndim,)
    if isinstance(axes, tuple):
        for axis in axes:
            _ = _check_axis_in_range(axis, ndim)
        axes = tuple(map(lambda x: x % ndim, axes))
        if all(axes.count(el) <= 1 for el in axes):
            return axes
    if axes is None:
        axes = F.make_range(ndim)
        return axes
    raise ValueError('duplicate value in \'axis\'')


@constexpr
def _check_shape_aligned(shape1, shape2):
    """Checks shape1 and shape2 are valid shapes to perform inner product"""
    if shape1[-1] == shape2[-1]:
        return True
    raise ValueError(
        f'shapes {shape1} {shape2} not aligned: {shape1[-1]} (dim 0) != {shape2[-1]} (dim 0)')


@constexpr
def _check_dim_cpu(shape, bound):
    """Checks input shape is upper-bounded by parameter bound"""
    ndim = len(shape)
    if _is_scalar(shape):
        return True
    if ndim <= bound:
        return True
    raise ValueError(
        f'dimension {ndim} larger than {bound} is not supported on CPU')


@constexpr
def _tile_size(shape, out_shape, ndim):
    """Returns tile_size such that shape*tile_size = out_shape"""
    size = [1]*ndim
    for idx, (i, j) in enumerate(zip(shape, out_shape)):
        if i != j:
            size[idx] = j
    return tuple(size)


@constexpr
def _check_core_match(shape1, shape2):
    """Checks shape1 and shape2 are valid shapes to perform matmul"""
    ndim1, ndim2 = len(shape1), len(shape2)
    if ndim1 < 1 or ndim2 < 2:
        return True
    if shape1[-1] == shape2[-2]:
        return True
    raise ValueError(f'mismatch in core dimension of input operands (size {shape1[-1]} ' +
                     f'is different from {shape2[-2]})')


@constexpr
def _cpu_not_support(name):
    """Checks if a function not supported on cpu is executed on cpu device"""
    if _get_device() != 'CPU':
        return True
    raise ValueError(f'{name} is not supported on CPU')


@constexpr
def _check_is_tuple(obj):
    """Check whether obj is a tuple"""
    return isinstance(obj, mstype.tuple_type)


@constexpr
def _check_is_list(obj):
    """Check whether obj is a list"""
    return isinstance(obj, mstype.list_type)


@constexpr
def _check_is_tensor(obj):
    """Check whether obj is a tensor"""
    return isinstance(obj, mstype.tensor_type)
