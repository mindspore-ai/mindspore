# This is the Python adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
#
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
"""standard_method"""
from dataclasses import dataclass
from mindspore.common import dtype as mstype
from ...ops import functional as F
from ...ops import operations as P
from ...ops.primitive import constexpr
from ...ops.composite import tail, core, MultitypeFuncGraph, env_get, hyper_add, \
    zeros_like, ones_like
from ...ops.composite.base import _append

__all__ = ['MultitypeFuncGraph', 'env_get', 'hyper_add', 'zeros_like', 'ones_like']

trans = P.Transpose()
shape_ = P.Shape()
reshape_ = P.Reshape()
dtype_ = P.DType()
abs_ = P.Abs()
ndim_ = P.Rank()
size_ = P.Size()

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


def transpose(x, *axis):
    """Implementation of `transpose`."""
    new_order = None
    shape = F.shape(x)
    length = F.tuple_len(shape)
    if not axis:
        perm = F.make_range(0, length)
        new_order = F.tuple_reversed(perm)

    elif len(axis) == 1:
        new_order = convert_list_to_tuple(axis[0])

    elif len(axis) == length:
        new_order = axis

    out = trans(x, new_order)
    return out


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
    return reshape_(x, shape)


def isinstance_(x, base_type):
    """Determine whether x is an instance of base_type."""
    x_type = F.typeof(x)
    return check_type_same(x_type, base_type)


def while_cond(x):
    """For while condtion, if the condition is a tensor, the loop will not be unrolled"""
    if F.issubclass_(F.typeof(x), F.typeof(mstype.tensor)):
        is_cond = check_is_tensor_bool_cond(F.shape(x))
        if is_cond:
            return F.cast(x, mstype.bool_)
    return x


@constexpr
def check_type_same(x_type, base_type):
    """Check x_type is same as base_type."""
    if mstype.issubclass_(x_type, base_type):
        return True
    return False


@constexpr
def check_is_tensor(x):
    """check whether x is tensor."""
    if isinstance(x, mstype.tensor_type):
        return True
    return False


@constexpr
def check_is_tuple_or_list_or_tensor(x, op_name, arg_name):
    """check whether x is list or tuple or tensor."""
    if isinstance(x, (mstype.list_type, mstype.tuple_type, mstype.tensor_type)):
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

@constexpr
def convert_list_to_tuple(shp):
    """Check the type of the shape, if is list, convert to tuple"""
    if not isinstance(shp, (list, tuple)):
        raise ValueError(f"The shape variable should be a list or tuple, but got {type(shp)}")
    if isinstance(shp, list):
        shp = tuple(shp)
    return shp

def tensor_bool(x):
    """tensor as conditon, if is constant, return immediate bool value"""
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
