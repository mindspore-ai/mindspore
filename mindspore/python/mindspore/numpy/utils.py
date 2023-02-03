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
"""internal utility functions"""
from __future__ import absolute_import

import types

from mindspore.common import Tensor
from mindspore.ops import functional as F
from mindspore.common import dtype as mstype

from mindspore.numpy.utils_const import _tile_size, _add_unit_axes, _raise_type_error, _type_convert, \
    _tuple_setitem, _callable_const, _check_is_float, _get_device


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
    if not isinstance(array_like, (Tensor, list, tuple, int, float, bool)):
        _raise_type_error("input data must be `int`, `float`, `bool`, `Tensor`, `list`, `tuple`, but got ", array_like)


def _is_scalar(shape):
    """check whether input shape is a scalar"""
    return F.shape_mul(shape) == 1


def _convert_list_tensor_to_tuple_tensor(list_of_tensor):
    """Convert a list of tensor to a tuple of tensor"""
    if isinstance(list_of_tensor, list):
        tuple_of_tensor = ()
        for tensor in list_of_tensor:
            tuple_of_tensor += (tensor,)
        return tuple_of_tensor
    return list_of_tensor


def _expand(x, ndim, axis=0):
    """Expand x to ndim from axis, which can be 0 or -1."""
    shape = _add_unit_axes(F.shape(x), ndim, axis == -1)
    return F.reshape(x, shape)


def _broadcast_to(x, shape_cur, shape_to, ndim_to):
    """Broadcasts x from shape_cur to shape_to."""
    size = _tile_size(shape_cur, shape_to, ndim_to)
    return F.tile(x, size)


def _broadcast_to_shape(x, shape):
    """Broadcasts x from current shape to shape"""
    ndim_to = len(shape)
    x = _expand(x, ndim_to)
    return _broadcast_to(x, F.shape(x), shape, ndim_to)


def _get_size(x, axis=None):
    """Get the number of elements along the given axis of tensor x."""
    if axis is None or F.tuple_len(axis) == 0:
        axis = F.make_range(x.ndim)
    nums = 1
    for ax in axis:
        nums *= x.shape[ax]
    return nums


def _check_input_tensor(*tensors):
    for tensor in tensors:
        if not isinstance(tensor, Tensor):
            _raise_type_error('expect Tensor, but got ', F.typeof(tensor))
    return True


def _convert_64_to_32(tensor):
    """Convert tensor with float64/int64 types to float32/int32."""
    if tensor.dtype == mstype.float64:
        return tensor.astype("float32")
    if tensor.dtype == mstype.int64:
        return tensor.astype("int32")
    return tensor


def _to_tensor(*args):
    """Returns each input as Tensor"""
    res = ()
    for arg in args:
        if isinstance(arg, (int, float, bool, list, tuple)):
            arg = _convert_64_to_32(_type_convert(Tensor, arg))
        elif not isinstance(arg, Tensor):
            _raise_type_error("Expect input to be array like.")
        res += (arg,)
    if len(res) == 1:
        return res[0]
    return res


def _get_dtype_from_scalar(*input_numbers):
    """
    Get the final dtype from series of input numbers, compared with F.typeof, we
    return int32/float32 for python int/float instead.
    """
    bool_flag = True
    int_flag = True
    for number in input_numbers:
        if number is not None:
            if not isinstance(number, bool):
                bool_flag = False
            if not isinstance(number, int):
                int_flag = False
    if bool_flag:
        return mstype.bool_
    if int_flag:
        return mstype.int32
    return mstype.float32


def _convert_bool_to_int(tensor):
    """Convert tensor with bool type to int32."""
    if tensor.dtype == mstype.bool_:
        return tensor.astype("int32")
    return tensor


def _slice_along_axis(f, axis, slice_start, slice_end):
    """
    Slice a tensor along a given axis

    Args:
        f (Tensor): Input Tensor.
        axis (int): Specified axis.
        slice_start (int): The start of the slice.
        slice_end (int): The end of the slice.

    Returns:
        Sliced tensor.
    """
    index_start = (0,) * f.ndim
    index_end = f.shape
    slice_size = slice_end - slice_start
    index_start = _tuple_setitem(index_start, axis, slice_start)
    index_end = _tuple_setitem(index_end, axis, slice_size)
    return F.tensor_slice(f, index_start, index_end)


def _to_tensor_origin_dtype(*args):
    """Returns each input as Tensor and remains original dtype."""
    res = []
    for arg in args:
        if isinstance(arg, (int, float, bool, list, tuple)):
            arg = _type_convert(Tensor, arg)
        elif not isinstance(arg, Tensor):
            _raise_type_error("Expect input to be array like.")
        res.append(arg)
    if len(res) == 1:
        return res[0]
    return res


def _callable(tensor, obj):
    """Returns True if `obj` is a function."""
    if F.isconstant(tensor):
        return isinstance(obj, types.FunctionType)
    return _callable_const(F.typeof(obj))


def _isnan(x):
    if _get_device() == 'Ascend' and not _check_is_float(F.dtype(x)):
        return F.fill(mstype.bool_, F.shape(x), False)
    return F.isnan(x)
