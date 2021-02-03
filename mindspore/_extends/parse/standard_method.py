# This is the Python adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
#
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
"""standard_method"""

from dataclasses import dataclass

from mindspore import Tensor, Parameter
from mindspore import dtype as mstype

from ..._checkparam import Validator as validator
from ...ops import functional as F
from ...ops import operations as P
from ...ops.composite import tail, core, MultitypeFuncGraph, env_get, hyper_add, \
    zeros_like, ones_like
from ...ops.composite.base import _append
from ...ops.primitive import constexpr

__all__ = ['MultitypeFuncGraph', 'env_get', 'hyper_add', 'zeros_like', 'ones_like']

shape_ = P.Shape()
dtype_ = P.DType()
abs_ = P.Abs()
ndim_ = P.Rank()
size_ = P.Size()

itemsize_map = {mstype.bool_: 1, mstype.int8: 1, mstype.uint8: 1,
                mstype.float16: 2, mstype.int16: 2, mstype.uint16: 2,
                mstype.float32: 4, mstype.int32: 4, mstype.uint32: 4,
                mstype.float64: 8, mstype.int64: 8, mstype.uint64: 8}


def mean(x, axis=(), keep_dims=False):
    """
    Reduces a dimension of a tensor by averaging all elements in the dimension.

    Args:
        axis (Union[None, int, tuple(int)]): Dimensions of reduction,
            when axis is None or empty tuple, reduce all dimensions.
            Default: (), reduce all dimensions.
        keep_dims (bool): Whether to keep the reduced dimensions.
            Default : False, don't keep these reduced dimensions.

    Returns:
        Tensor, has the same data type as x.
    """
    if axis is None:
        axis = ()
    reduce_mean = P.ReduceMean(keep_dims)
    return reduce_mean(x, axis)


def all_(x, axis=(), keep_dims=False):
    """
    Check all array elements along a given axis evaluate to True.

    Args:
        x (Tensor): A Tensor to be reduced.
        axis (Union[None, int, tuple(int)): Dimensions of reduction.
        keep_dims (bool): Whether to keep the reduced dimensions.

    Returns:
        Tensor, has the same data type as x.
    """

    if axis is None:
        axis = ()
    reduce_all = P.ReduceAll(keep_dims)
    return reduce_all(x, axis)


def any_(x, axis=(), keep_dims=False):
    """
    Check any array element along a given axis evaluate to True.

    Args:
        x (Tensor): A Tensor to be reduced.
        axis (Union[None, int, tuple(int)): Dimensions of reduction.
        keep_dims (bool): Whether to keep the reduced dimensions.

    Returns:
        Tensor, has the same data type as x.
    """
    if axis is None:
        axis = ()
    reduce_any = P.ReduceAny(keep_dims)
    return reduce_any(x, axis)


def itemsize_(x):
    """
    Return length of one tensor element in bytes.

    Args:
        x (Tensor): Input tensor.

    Returns:
        itemsize(int).
    """
    return get_itemsize(x.dtype)


def nbytes_(x):
    """
    Return total number of bytes taken by the tensor.

    Args:
        x (Tensor): Input tensor.

    Returns:
        nbytes(int).
    """
    return itemsize_(x) * F.shape_mul(shape_(x))


def strides_(x):
    """
    Return the tuple of bytes to step in each dimension when traversing a tensor.

    Args:
        x (Tensor): Input tensor.

    Returns:
        strides (tuple[int]).
    """
    strides = ()
    ndim = P.Rank()(x)
    tensor_shape = shape_(x)
    for i in F.make_range(0, ndim):
        stride = itemsize_(x)
        for j in F.make_range(i + 1, ndim):
            stride *= tensor_shape[j]
        strides += (stride,)
    return strides


def astype(x, dtype, copy=True):
    """Implementation of `astype`."""
    dtype = check_astype_dtype_const(dtype)
    if not copy and dtype == x.dtype:
        return x
    return F.cast(x, dtype)


def transpose(x, *axis):
    """Implementation of `transpose`."""
    ndim = F.rank(x)
    perm = check_transpose_axis_const(axis, ndim)
    return F.transpose(x, perm)


# `tensor.T` is used as a property in graph mode
T_ = transpose


def reshape(x, *shape):
    """Implementation of `reshape`."""
    new_shape = check_reshape_shp_const(shape)
    return F.reshape(x, new_shape)


def ravel(x):
    """Implementation of `ravel`."""
    return reshape(x, (-1,))


def flatten(x, order='C'):
    """
    Returns a copy of the array collapsed into one dimension.

    Args:
        order (str, optional): Can choose between `C` and `F`. `C` means to
        flatten in row-major (C-style) order. ‘F’ means to flatten in column-major
        (Fortran- style) order. Only `C` and `F` are supported.

    Returns:
        Tensor, has the same data type as x.
    """
    order = check_flatten_order_const(order)
    if order == 'C':
        return F.reshape(x, (-1,))

    perm = F.make_range(0, F.rank(x))
    new_order = F.tuple_reversed(perm)
    return F.reshape(F.transpose(x, new_order), (-1,))


def swapaxes(x, axis1, axis2):
    """
    Interchanges two axes of a tensor.

    Args:
        axis1 (int): First axis.
        axis2 (int): Second axis.

    Returns:
        Transposed tensor, has the same data type as the original tensor x.
    """
    axis1, axis2 = check_swapaxes_axis_const((axis1, axis2), x.ndim)

    if axis1 == axis2:
        return x
    if axis1 > axis2:
        axis1, axis2 = axis2, axis1

    perm = F.make_range(0, x.ndim)
    new_perm = None
    if axis2 + 1 < x.ndim:
        new_perm = perm[0:axis1] + perm[axis2:axis2 + 1] + \
                   perm[axis1 + 1:axis2] + perm[axis1:axis1 + 1] + perm[axis2 + 1:]
    else:
        new_perm = perm[0:axis1] + perm[axis2:axis2 + 1] + \
                   perm[axis1 + 1:axis2] + perm[axis1:axis1 + 1]

    return F.transpose(x, new_perm)


def squeeze(x, axis=None):
    """
    Removes single-dimensional entries from the shape of an tensor.

    Args:
        axis: Union[None, int, list(int), tuple(list)]. Default is None.

    Returns:
        Tensor, with all or a subset of the dimensions of length 1 removed.
    """
    shape = F.shape(x)
    if axis is None:
        return F.squeeze(x)
    # yield squeezed shape based on the axes
    new_shape = prepare_shape_for_squeeze_const(shape, axis)
    return F.reshape(x, new_shape)


def getitem(data, item):
    """Implementation of `getitem`."""
    return data.__getitem__(item)


def setitem(data, item, value):
    """Implementation of `setitem`."""
    return data.__setitem__(item, value)


def ms_iter(xs):
    """Implementation of `iter`."""
    return xs.__ms_iter__()


def ms_next(it):
    """Implementation of `next`."""
    return it.__ms_next__()


def hasnext(it):
    """Implementation of `hasnext`."""
    return it.__ms_hasnext__()


def ms_len(data):
    """Implementation of `len`."""
    return data.__len__()


def floor(x):
    """Implementation of `floor`."""
    return x.__floor__()


def trunc(x):
    """Implementation of `trunc`."""
    return x.__trunc__()


def uadd(x):
    """Implementation of `uadd`."""
    return x.__pos__()


def usub(x):
    """Implementation of `usub`."""
    return x.__neg__()


def scalar_truediv(x, y):
    """Implementation of `scalar_truediv`."""
    return x.__truediv__(y)


def scalar_floordiv(x, y):
    """Implementation of `scalar_floordiv`."""
    return x.__floordiv__(y)


def bool_(x):
    """Implementation of `bool`."""
    return x.__bool__()


def enumerate_(x, start=0):
    """Enumerate list or tuple or tensor."""
    x_type = F.typeof(x)
    ret = ()
    op_name = "enumerate"
    if check_is_tuple_or_list_or_tensor(x_type, op_name, "first input") and check_is_const_int(start, op_name, "start"):
        if check_is_tensor(x_type):
            for i in range(x.shape[0]):
                ret += ((start + i, x[i]),)
        else:
            ret = zip(range(start, start + len(x)), x)
    return ret


def expand_tensor_as(x, y):
    """Expand tensor"""
    broadcast_to = P.BroadcastTo(shape_(y))
    return broadcast_to(x)


def view(x, *shape):
    """Reshape tensor, if shape is -1, reshape tensor into one dimension"""
    shape = check_view_shape(shape)
    return F.reshape(x, shape)


def isinstance_(x, base_type):
    """Determine whether x is an instance of base_type."""
    x_type = F.typeof(x)
    return check_type_same(x_type, base_type)


def while_cond(x):
    """For while condition, if the condition is a tensor, the loop will not be unrolled"""
    if F.issubclass_(F.typeof(x), F.typeof(mstype.tensor)):
        is_cond = check_is_tensor_bool_cond(F.shape(x))
        if is_cond:
            return F.cast(x, mstype.bool_)
    return x


@constexpr
def check_type_same(x_type, base_type):
    """Check x_type is same as base_type."""
    pytype_to_mstype = {
        bool: mstype.Bool,
        int: mstype.Int,
        float: mstype.Float,
        str: mstype.String,
        list: mstype.List,
        tuple: mstype.Tuple,
        dict: mstype.Dict,
        Tensor: mstype.tensor_type,
        Parameter: mstype.ref_type
    }

    has_int = False
    has_tensor = False

    def to_target_type(origin_type):
        try:
            if isinstance(origin_type, type):
                ret_type = pytype_to_mstype[origin_type]
                if ret_type == mstype.Int:
                    nonlocal has_int
                    has_int = True
                if ret_type == mstype.tensor_type:
                    nonlocal has_tensor
                    has_tensor = True
                return (ret_type,)
            if isinstance(origin_type, tuple):
                return tuple(to_target_type(i) for i in origin_type)
            raise TypeError(f"The second arg of 'isinstance' must be a type or a tuple of types, "
                            f"but got a {type(origin_type).__name__}")
        except KeyError:
            raise TypeError(f"The second arg of 'isinstance' should be bool, int, float, str, list, tuple, "
                            f"Tensor, Parameter, or a tuple containing only these types, but got {origin_type}")
    target_type = to_target_type(base_type)
    if (isinstance(x_type, mstype.Bool) and has_int) or (isinstance(x_type, mstype.ref_type) and has_tensor):
        return True
    return isinstance(x_type, target_type)


@constexpr
def get_itemsize(x_type):
    """get itemsize from tensor's dtype."""
    return itemsize_map[x_type]


@constexpr
def check_is_tensor(x):
    """check whether x is tensor."""
    if isinstance(x, mstype.tensor_type):
        return True
    return False


@constexpr
def check_is_tuple_or_list_or_tensor(x, op_name, arg_name):
    """check whether x is list or tuple or tensor."""
    if isinstance(x, (mstype.List, mstype.Tuple, mstype.tensor_type)):
        return True
    raise TypeError(f"For '{op_name}', the '{arg_name}' should be tuple or list or tensor, but got {x}.")


@constexpr
def check_is_const_int(x, op_name, arg_name):
    """check whether x is const int."""
    if x is None:
        raise TypeError(f"For '{op_name}', the '{arg_name}' should be a const int number, but got not const.")
    if not isinstance(x, int):
        raise TypeError(f"For '{op_name}', the '{arg_name}' should be a const int number, but got {x}.")
    return True


@constexpr
def check_is_tensor_bool_cond(shp):
    """check if tensor is a bool condition"""
    if shp in ((), (1,)):
        return True
    raise ValueError("The truth value of an array with several elements is ambiguous.")


@constexpr
def const_tensor_to_bool(x):
    """convert bool tensor to bool condition"""
    if x is None:
        raise ValueError("Only constant tensor bool can be converted to bool")
    x = x.asnumpy()
    if x.shape == ():
        return bool(x)
    if x.shape == (1,):
        return bool(x[0])
    raise ValueError("The truth value of an array with several elements is ambiguous.")


@constexpr
def check_view_shape(x):
    """Check view function input shape"""
    if not x:
        raise ValueError("The shape variable should not be empty")
    if isinstance(x[0], tuple):
        if len(x) != 1:
            raise ValueError(f"Only one tuple is needed, but got {x}")
        x = x[0]
    return x


# convert normal param_check functions to constexpr functions
check_astype_dtype_const = constexpr(validator.check_astype_dtype)
check_transpose_axis_const = constexpr(validator.check_transpose_axis)
check_reshape_shp_const = constexpr(validator.check_reshape_shp)
check_flatten_order_const = constexpr(validator.check_flatten_order)
check_swapaxes_axis_const = constexpr(validator.check_swapaxes_axis)
prepare_shape_for_squeeze_const = constexpr(validator.prepare_shape_for_squeeze)


def tensor_bool(x):
    """tensor as condition, if is constant, return immediate bool value"""
    is_cond = check_is_tensor_bool_cond(F.shape(x))
    if is_cond and F.isconstant(x):
        return const_tensor_to_bool(x)
    return F.cast(x, mstype.bool_)


def and_(x, y):
    """Implementation of `and` (`&`)."""
    return x.__and__(y)


def or_(x, y):
    """Implementation of `or` (`|`)."""
    return x.__or__(y)


def matmul(x, y):
    """Implementation of `matmul` (`@`)."""
    return x.__matmul__(y)


def float_bool(x):
    """Implementation of `float_bool`."""
    return x != 0.0


def int_bool(x):
    """Implementation of `int_bool`."""
    return x != 0


def str_bool(x):
    """Implementation of `str_bool`."""
    if x == "":
        return False
    return True


def list_bool(x):
    """Implementation of `tuple_bool`."""
    return len(x) != 0


def tuple_bool(x):
    """Implementation of `tuple_bool`."""
    return len(x) != 0


def dict_bool(x):
    """Implementation of `dict_bool`."""
    return len(x) != 0


def none_bool(x):
    """Implementation of `none_bool`."""
    return False


def func_bool(x):
    """Implementation of `func_bool`."""
    return True


def float_floordiv(x, y):
    """Implementation of `float_floordiv`."""
    return floor(x / y)


#############
# Iteration #
#############


@dataclass(frozen=True)
class SequenceIterator:
    """
    SequenceIterator is a util dataclass for iterating sequence object.

    Iterator to use for sequences like List, Array.
    """

    idx: int
    seq: list

    @core(ignore_values=True)
    def __ms_hasnext__(self):
        """Whether the index is past the length of the sequence."""
        return self.idx < ms_len(self.seq)

    @core(ignore_values=True)
    def __ms_next__(self):
        """Return the next element and a new iterator."""
        return self.seq[self.idx], SequenceIterator(self.idx + 1, self.seq)


def list_iter(xs):
    """Iterator for List."""
    return SequenceIterator(0, xs)


def array_iter(xs):
    """Iterator for Array."""
    return SequenceIterator(0, xs)


def tuple_next(xs):
    """Next tuple."""
    return xs[0], tail(xs)


def tuple_hasnext(xs):
    """Whether the tuple is empty or not."""
    return len(xs) > 0


def list_next(xs):
    """Next list."""
    return xs[0], tail(xs)


def list_hasnext(xs):
    """Whether the list is empty or not."""
    return len(xs) > 0


def list_append(self_, item):
    return _append(self_, item)


#################
# Array methods #
#################


def to_array(x):
    """Implementation of `to_array`."""
    return x.__ms_to_array__()
