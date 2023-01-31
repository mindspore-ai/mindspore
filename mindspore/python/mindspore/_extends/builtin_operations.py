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
"""builtin_operations"""
from __future__ import absolute_import
import numpy as np
from mindspore.ops import functional as F
from mindspore.ops import composite as C
from mindspore.common.tensor import Tensor
import mindspore.common.dtype as mstype
from mindspore.common.dtype import dtype_to_nptype, get_py_obj_dtype
from mindspore._c_expression import Tensor as CTensor


def ScalarAdd(x, y):
    """Implement `scalar_add`."""
    return x + y


def ScalarMul(x, y):
    """Implement `scalar_mul`."""
    return x * y


def ScalarMod(x, y):
    """Implement `scalar_mul`."""
    return x % y


def ScalarSub(x, y):
    """Implement `scalar_sub`."""
    return x - y


def ScalarUsub(x):
    """Implement `scalar_usub`."""
    return -x


def TupleGetItem(x, index):
    """Implement `tuple_getitem`."""
    if isinstance(x, Tensor):
        x = x.asnumpy()
        y = x[index]
        return Tensor(y)

    if isinstance(x, dict):
        count = 0
        for value in x.values():
            if count == index:
                return value
            count = count + 1

    return x[index]


def ScalarGreater(x, y):
    """Implement `scalar_gt`."""
    return x > y


def scalar_ne(x, y):
    """Implement `scalar_ne`."""
    return x != y


def ScalarEqual(x, y):
    """Implement `scalar_eq`."""
    return x == y


def ScalarLessEqual(x, y):
    """Implement `scalar_le`."""
    return x <= y


def ScalarLess(x, y):
    """Implement `scalar_lt`."""
    return x < y


def identity(x):
    """Implement `identity`."""
    return x


def zeros_like_tensor(x):
    """Implement `zeros_like_tensor`."""
    x = x.asnumpy()
    value = Tensor(np.zeros(x.shape).astype(np.float32))
    return value


def OnesLike(x):
    """Implement `oneslike`."""
    if isinstance(x, (Tensor, CTensor)):
        x = x.asnumpy()
        value = Tensor(np.ones(x.shape).astype(x.dtype))
    else:
        value = Tensor(1.0)
    return value


def Switch(c, x, y):
    """Implement `switch`."""
    return x if c else y


def list_getitem(data, item):
    """Implement `list_getitem`."""
    return data[item]


def bool_not(x):
    """Implement `bool_not`."""
    return not x


def bool_and(x, y):
    """Implement `bool_and`."""
    return x and y


def bool_or(x, y):
    """Implement `bool_or`."""
    return x or y


def make_list(*xs):
    """Implement `make_list`."""
    return list(xs)


def list_len(x):
    """Implement `list_len`."""
    return len(x)


def Depend(value, expr):
    """Implement `Depend`."""
    return value


def UpdateState(monad, *exprs):
    """Implement `UpdateState`."""
    return monad


def Load(value, u=None):
    """Implement `Load`."""
    return value


def scalar_cast(x, t):
    """Implement scalar_cast."""
    np_type = dtype_to_nptype(t)
    value = np_type(x)
    cast_value = np.ndarray.item(value)
    return cast_value


def typeof(x):
    """Implement typeof."""
    return get_py_obj_dtype(x)


def tuple_to_array(x):
    """Implement `tuple_to_array`."""
    return Tensor(np.array(x))


def StopGradient(x):
    """Implement `StopGradient`."""
    return x


def TensorShape(x):
    """Implement `TensorShape`."""
    return Tensor(F.shape(x))


def DynamicBroadcastGradientArgs(x, y):
    """Implement `DynamicBroadcastGradientArgs`."""
    return -1, -1


def StridedSlice(x, begin, end, stride):
    """Implement `StridedSlice`."""
    if isinstance(x, Tensor):
        x = x.asnumpy()
        ret = x
        for i in range(len(x)):
            ret[i:] = x[begin[i], end[i], stride[i]]
        return Tensor(ret)
    ret = x
    for i in range(len(x)):
        ret[i:] = x[begin[i], end[i], stride[i]]
    return ret


hyper_map = C.HyperMap()


def mixed_precision_cast(dst_type, x):
    """Implement `mixed_precision_cast`."""

    def cast_inner(data):
        if isinstance(data, Tensor) and data.dtype in (mstype.float32, mstype.float16, mstype.float64):
            return F.cast(data, dst_type)
        return data

    return hyper_map(cast_inner, x)
