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
from ...ops import functional as F
from ...ops import operations as P
from ...ops.composite import tail, core, MultitypeFuncGraph, env_get, hyper_add, \
    zeros_like, ones_like
from ...ops.composite.base import _append

__all__ = ['MultitypeFuncGraph', 'env_get', 'hyper_add', 'zeros_like', 'ones_like']

trans = P.Transpose()


def transpose(x):
    """Implementation of `transpose`."""
    shape = F.shape(x)
    length = F.tuple_len(shape)
    perm = F.make_range(0, length)
    revert_perm = F.tuple_reversed(perm)
    out = trans(x, revert_perm)
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


def tensor_bool(x):
    """return immedate x, x is a tensor of bool value"""
    return x


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
