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
from __future__ import absolute_import

import math
from itertools import accumulate
import operator

import mindspore.context as context
from mindspore.ops import functional as F
from mindspore.ops.primitive import constexpr
from mindspore.ops.primitive import _primexpr
from mindspore.common import dtype as mstype
from mindspore.common import Tensor
from mindspore._c_expression import Tensor as Tensor_
from mindspore._c_expression import typing
from mindspore._checkparam import Validator as validator

from mindspore.numpy.dtypes import promotion_rule, dtype_tuple, all_types, dtype_map, rule_for_trigonometric

_check_axis_type = constexpr(validator.check_axis_type)


@_primexpr
def _check_shape(shape):
    """check the shape param to match the numpy style"""
    # convert tensor to int/list, use followed if statements to do further conversions
    if isinstance(shape, Tensor):
        shape = shape.asnumpy().tolist()
        # this may return an int, don't change the next if to elif
    if isinstance(shape, int):
        shape = (shape,)
    elif isinstance(shape, (list, typing.List)):
        shape = tuple(shape)
    return shape


@constexpr
def _check_dtype(dtype):
    """check the input dtype and make conversions"""
    # convert the string dtype to mstype.dtype
    if isinstance(dtype, str):
        dtype = dtype.lower()
        dtype = dtype_map.get(dtype, "")
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


@_primexpr
def _is_shape_empty(shp):
    """Check whether shape contains zero"""
    if F.is_sequence_shape_unknown(shp):
        return False
    if isinstance(shp, int):
        return shp == 0
    if isinstance(shp, (tuple, list)):
        return 0 in shp
    return F.shape_mul(shp) == 0


@_primexpr
def _check_start_normalize(start, ndim):
    """check and normalize start argument for rollaxis."""
    if start < 0:
        start = start + ndim
    return start


@_primexpr
def _check_axes_range(axes, ndim):
    """
    Check axes type and normalize the negative axes.

    Args:
        axes: Axes of the tensor.
        ndim (int): The number of dimensions of the tensor.

    Return:
        Axes (Union[int, tuple(int)]). If input is integer, return integer, else tuple.

    Raises:
        TypeError: If the axes are not integer, tuple(int) or list(int).
        ValueError: If duplicate axes exists or some axis is out of bounds.
    """
    axes = _canonicalize_axis(axes, ndim)
    return axes


@constexpr
def _get_device():
    """Get the current device (`GPU`, `CPU`, `Ascend`)"""
    return context.get_context('device_target')


@_primexpr
def _infer_out_shape(*shapes):
    """
    Returns shape of output after broadcasting. Raises ValueError if shapes cannot be broadcast.
    """
    shape_out = list()
    max_len = max([len(it) for it in shapes])
    for i in range(max_len):
        items = [it[i-max_len+len(it)] if i-max_len +
                 len(it) >= 0 else 1 for it in shapes]
        max_size = 0 if 0 in items else max(items)
        shape_out.append(max_size)
    return tuple(shape_out)


@_primexpr
def _can_broadcast(*shapes):
    """
    Returns Ture if shapes can broadcast, False if they cannot.
    """
    max_len = max([len(it) for it in shapes])
    for i in range(max_len):
        items = [it[i-max_len+len(it)] if i-max_len +
                 len(it) >= 0 else 1 for it in shapes]
        max_size = 0 if 0 in items else max(items)
        if any(item not in (1, max_size) for item in items):
            return False
    return True


@_primexpr
def _check_axis_in_range(axis, ndim):
    """Checks axes are with the bounds of ndim"""
    return axis - axis // ndim * ndim


@_primexpr
def _check_axis_valid(axes, ndim):
    """
    Checks axes are valid given ndim, and returns axes that can be passed
    to the built-in operator (non-negative, int or tuple)
    """
    if axes is None:
        axes = F.make_range(ndim)
        return axes
    if isinstance(axes, (tuple, list)):
        axes = tuple(map(lambda x: _check_axis_in_range(x, ndim), axes))
        return axes
    return (_check_axis_in_range(axes, ndim),)


@_primexpr
def _tile_size(shape, out_shape, ndim):
    """Returns tile_size such that shape*tile_size = out_shape"""
    size = [1]*ndim
    for idx, (i, j) in enumerate(zip(shape, out_shape)):
        if i != j:
            size[idx] = j
    return tuple(size)


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
def _raise_runtime_error(info, param=None):
    """
    Raise RuntimeError in both graph/pynative mode

    Args:
        info(str): info string to display
        param(python obj): any object that can be recognized by graph mode. If is
            not None, then param's value information will be extracted and displayed.
            Default is None.
    """
    if param is None:
        raise RuntimeError(info)
    raise RuntimeError(info + f"{param}")


@constexpr
def _raise_unimplemented_error(info, param=None):
    """
    Raise NotImplementedError in both graph/pynative mode

    Args:
        info(str): info string to display
        param(python obj): any object that can be recognized by graph mode. If is
            not None, then param's value information will be extracted and displayed.
            Default is None.
    """
    if param is None:
        raise NotImplementedError(info)
    raise NotImplementedError(info + f"{param}")


@_primexpr
def _empty(dtype, shape):
    """Returns an uninitialized array with dtype and shape."""
    return Tensor_(dtype, shape)


@constexpr
def _promote(dtype1, dtype2):
    if dtype1 == dtype2:
        return dtype1
    if (dtype1, dtype2) in promotion_rule:
        return promotion_rule.get((dtype1, dtype2))
    res = promotion_rule.get((dtype2, dtype1))
    if res is None:
        raise TypeError(f"input dtype: {dtype1}, {dtype2} are not all mindspore datatypes.")
    return res


@constexpr
def _promote_for_trigonometric(dtype):
    res_dtype = rule_for_trigonometric.get(dtype)
    if res_dtype is None:
        raise TypeError(f"input dtype: {dtype} is not a mindspore datatype.")
    return res_dtype


@_primexpr
def _max(*args):
    """Returns the maximum value."""
    return max(*args)


@_primexpr
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
def _check_is_int(dtype):
    return isinstance(dtype, typing.Int)


@_primexpr
def _canonicalize_axis(axis, ndim):
    """
    Check axes are within the number of dimensions of tensor x and normalize the negative axes.
    Args:
        axis (Union[int, tuple(int), list(int)]): Axes of the tensor.
        ndim (int): The number of dimensions of the tensor.
    Return:
        Axis (Union[int, tuple(int)]). If input is integer, return integer, else tuple.
    """
    if isinstance(axis, int):
        axis = [axis]
    for ax in axis:
        _check_axis_in_range(ax, ndim)

    def canonicalizer(ax):
        return ax + ndim if ax < 0 else ax

    def _sort_axis(ax):
        def merge(left, right):
            result = []
            i = j = 0
            while i < len(left) and j < len(right):
                if left[i] <= right[j]:
                    result.append(left[i])
                    i += 1
                else:
                    result.append(right[j])
                    j += 1

            return result + left[i:] + right[j:]

        if len(ax) <= 1:
            return ax

        middle = len(ax) // 2
        left = _sort_axis(ax[:middle])
        right = _sort_axis(ax[middle:])

        return merge(left, right)

    axis = tuple([canonicalizer(axis) for axis in axis])
    return tuple(_sort_axis(axis)) if len(axis) > 1 else axis[0]


@constexpr
def _broadcast_tuples(tup1, tup2):
    """
    Broadcast two 1D tuples to the same length, if inputs are ints, convert to
    tuples first.
    """
    tup1 = (tup1,) if isinstance(tup1, int) else tup1
    tup2 = (tup2,) if isinstance(tup2, int) else tup2
    if not isinstance(tup1, (tuple, list)) or not isinstance(tup2, (tuple, list)):
        raise TypeError("input shift and axis must be tuple or list or int.")
    if len(tup1) == len(tup2):
        return tup1, tup2
    if len(tup1) == 1:
        tup1 *= len(tup2)
    elif len(tup2) == 1:
        tup2 *= len(tup1)
    else:
        raise ValueError("shape mismatch: objects cannot be broadcast to a single shape")
    return tup1, tup2


@_primexpr
def _expanded_shape(ndim, axis_size, axis):
    """
    Returns a shape with size = 1 for all dimensions
    except at axis.
    """
    return tuple([axis_size if i == axis else 1 for i in range(ndim)])


@_primexpr
def _add_unit_axes(shape, ndim, append=False):
    """
    Prepends shape with 1s so that it has the number of dimensions ndim.
    If append is set to True, returns shape appended with 1s instead.
    """
    if isinstance(shape, int):
        shape = (shape,)
    ndim_diff = ndim - len(shape)
    if ndim_diff > 0:
        if append:
            shape = list(shape) + [1] * ndim_diff
        else:
            shape = [1] * ndim_diff + list(shape)
    return tuple(shape)


@constexpr
def  _check_element_int(lst):
    """
    Check whether each element in `lst` is an integer.
    """
    for item in lst:
        if not isinstance(item, int):
            raise TypeError(f"Each element in {lst} should be integer, but got {type(item)}.")
    return True


@constexpr
def _type_convert(force, obj):
    """
    Convert type of `obj` to `force`.
    """
    return force(obj)


@_primexpr
def _list_comprehensions(obj, item=None, return_tuple=False, make_none=False):
    """
    Generates a new list/tuple by list comprehension.

    Args:
        obj (Union[int, list, tuple]):
            If integer, it will be the length of the returned tuple/list.
        item: The value to be filled. Default: None.
            If None, the values in the new list/tuple are the same as obj
            or range(obj) when obj is integer.
        return_tuple(bool): If true, returns tuple, else returns list.

    Returns:
        List or tuple.
    """
    res = []
    lst = obj
    if isinstance(obj, int):
        lst = range(obj)
    if make_none:
        res = [None for _ in lst]
    elif item is None:
        res = list(lst)
    else:
        res = [item for _ in lst]
    if return_tuple:
        return tuple(res)
    return res


@_primexpr
def _tuple_setitem(tup, idx, value):
    """
    Returns a tuple with specified `idx` set to `value`.
    """
    tup = list(tup)
    tup[idx] = value
    return tuple(tup)


@constexpr
def _iota(dtype, num, increasing=True):
    """Creates a 1-D tensor with value: [0,1,...num-1] and dtype."""
    # Change to P.Linspace when the kernel is implemented on CPU.
    if num <= 0:
        raise ValueError("zero shape Tensor is not currently supported.")
    if increasing:
        return Tensor(list(range(int(num))), dtype)
    return Tensor(list(range(int(num)-1, -1, -1)), dtype)


@constexpr
def _ceil(number):
    """Ceils the number in graph mode."""
    return math.ceil(number)


@_primexpr
def _seq_prod(seq1, seq2):
    """Returns the element-wise product of seq1 and seq2."""
    return tuple(map(lambda x, y: x*y, seq1, seq2))


@constexpr
def _make_tensor(val, dtype):
    """Returns the tensor with value `val` and dtype `dtype`."""
    return Tensor(val, dtype)


@_primexpr
def _tuple_slice(tup, start, end):
    """get sliced tuple from start and end."""
    return tup[start:end]


@constexpr
def _isscalar(x):
    """Returns True if x is a scalar type"""
    return isinstance(x, (typing.Number, typing.Int, typing.UInt, typing.Float,
                          typing.Bool, typing.String))


@constexpr
def _cumprod(x):
    return tuple(accumulate(x, operator.mul))


@constexpr
def _in(x, y):
    return x in y


@constexpr
def _callable_const(x):
    """Returns true if x is a function in graph mode."""
    return isinstance(x, typing.Function)


@constexpr
def _check_is_inf(x, negative=False):
    if not negative:
        return x == float('inf')
    return x == float('-inf')
