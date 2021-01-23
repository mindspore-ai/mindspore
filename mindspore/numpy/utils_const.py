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
"""internal graph-compatible utility functions"""
from functools import partial

import mindspore.context as context
from ..ops import functional as F
from ..ops.primitive import constexpr
from ..common import dtype as mstype
from .._c_expression import Tensor as Tensor_
from .._c_expression.typing import Tuple, List

from .dtypes import promotion_rule, dtype_tuple, all_types, dtype_map


@constexpr
def _check_shape(shape):
    """check the shape param to match the numpy style"""
    if not isinstance(shape, (int, tuple, list, Tuple, List)):
        raise TypeError(f"only int, tuple and list are allowed for shape, but got {type(shape)}")
    if isinstance(shape, int):
        shape = (shape,)
    if isinstance(shape, (list, List)):
        shape = tuple(shape)
    return shape


@constexpr
def _check_dtype(dtype):
    """check the input dtype and make conversions"""
    # convert the string dtype to mstype.dtype
    if isinstance(dtype, str):
        dtype = dtype.lower()
        dtype = dtype_map[dtype]
    elif isinstance(dtype, type):
        if dtype is int:
            dtype = mstype.int32
        elif dtype is float:
            dtype = mstype.float32
        else:
            dtype = mstype.pytype_to_dtype(dtype)
    if dtype not in dtype_tuple:
        raise TypeError(f"only {all_types} are allowed for dtype, but got {type(dtype)}")
    return dtype


@constexpr
def _check_shape_contain_zero(shp):
    """Check whether shape contains zero"""
    if isinstance(shp, int):
        return shp == 0
    return F.shape_mul(shp) == 0


@constexpr
def _check_start_normalize(start, ndim):
    """check and normalize start argument for rollaxis."""
    if start < -ndim or start > ndim:
        raise ValueError(f"For rollaxis, start {start} is out of bounds. Ranging from {-ndim} to {ndim} is allowed.")
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
        raise TypeError(f"int, tuple(int) or list(int) expected, but got {type(axes)}.")
    low = -ndim
    up = ndim - 1
    if low > up:
        raise ValueError(f"Lower bound {low} and upper bound {up} of axes are not allowed.")
    if isinstance(axes, int):
        if axes < low or axes > up:
            raise ValueError(f"axis {axes} is out of bounds for tensor of dimension {ndim}.")
        return axes if axes >= 0 else axes + ndim
    new_axes = []
    for item in axes:
        if not isinstance(item, int):
            raise TypeError(f"int in tuple or list expected, but got {type(item)}.")
        if item < low or item > up:
            raise ValueError(f"axis {item} in {axes} is out of bounds for tensor of dimension {ndim}.")
        new_axes.append(item if item >= 0 else item + ndim)
    return tuple(new_axes)


@constexpr
def _get_device_compile():
    """Get the current device (`GPU`, `CPU`, `Ascend`)"""
    return context.get_context('device_target')


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
def _infer_out_shape(*shapes):
    """
    Returns shape of output after broadcasting
    Raises ValueError if shape1 and shape2 cannot be broadcast
    """
    shapes_unbroadcastable = False
    ndim_max = max(map(len, shapes))
    shape_out = [0]*ndim_max
    i = 0
    for i in range(ndim_max):
        shape_out[-1 - i] = max(map(partial(_reverse_index, i), shapes))
        for shape in shapes:
            if _reverse_index(i, shape) != shape_out[-1 - i]:
                if _reverse_index(i, shape) != 1:
                    shapes_unbroadcastable = True
                    break
        if shapes_unbroadcastable:
            break
    if not shapes_unbroadcastable:
        return tuple(shape_out)
    raise ValueError(f'operands could not be broadcast together with shapes {*shapes,}')


@constexpr
def _check_axis_in_range(axis, ndim):
    """Checks axes are with the bounds of ndim"""
    if -ndim <= axis < ndim:
        return True
    raise ValueError(f'axis {axis} is out of bounds for array of dimension {ndim}')


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
    raise ValueError(f'shapes {shape1} {shape2} not aligned: {shape1[-1]} (dim 0) != {shape2[-1]} (dim 0)')


@constexpr
def _tile_size(shape, out_shape, ndim):
    """Returns tile_size such that shape*tile_size = out_shape"""
    size = [1]*ndim
    for idx, (i, j) in enumerate(zip(shape, out_shape)):
        if i != j:
            size[idx] = j
    return tuple(size)


@constexpr
def _check_is_int(obj):
    """Check whether obj is an integer."""
    return isinstance(obj, int)


@constexpr
def _check_is_tuple(obj):
    """Check whether obj is a tuple"""
    return isinstance(obj, (tuple, Tuple))


@constexpr
def _check_is_list(obj):
    """Check whether obj is a list"""
    return isinstance(obj, (list, List))


@constexpr
def _check_is_tensor(obj):
    """Check whether obj is a tensor"""
    return isinstance(obj, mstype.tensor_type)


@constexpr
def _raise_type_error(info, param=None):
    """
    Raise TypeError in both graph/pynative mode

    Args:
        info(str): info string to display
        param(python obj): any object that can be recognized by graph mode. If is
            not None, then param's type information will be extracted and displayed.
            Default is None.
    """
    if param is None:
        raise TypeError(info)
    raise TypeError(info + f"{type(param)}")


@constexpr
def _raise_value_error(info, param=None):
    """
    Raise TypeError in both graph/pynative mode

    Args:
        info(str): info string to display
        param(python obj): any object that can be recognized by graph mode. If is
            not None, then param's value information will be extracted and displayed.
            Default is None.
    """
    if param is None:
        raise ValueError(info)
    raise ValueError(info + f"{param}")


@constexpr
def _empty(dtype, shape):
    """Returns an uninitialized array with dtype and shape."""
    return Tensor_(dtype, shape)


@constexpr
def _promote(dtype1, dtype2):
    if dtype1 == dtype2:
        return dtype1
    if (dtype1, dtype2) in promotion_rule:
        return promotion_rule[dtype1, dtype2]
    return promotion_rule[dtype2, dtype1]


@constexpr
def _max(*args):
    """Returns the maximum value."""
    return max(*args)


@constexpr
def _min(*args):
    """"Returns the minimum value."""
    return min(*args)


@constexpr
def _abs(arg):
    """Returns the absolute value."""
    return abs(arg)


@constexpr
def _check_same_type(dtype1, dtype2):
    return dtype1 == dtype2


@constexpr
def _check_is_float(dtype):
    """Returns whether dtype is float16 or float32."""
    return dtype in (mstype.float16, mstype.float32)


@constexpr
def _check_input_tensor(input_type):
    if not _check_is_tensor(input_type):
        raise TypeError(f'expect Tensor, but got {input_type}')
