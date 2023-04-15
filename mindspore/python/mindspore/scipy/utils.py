# Copyright 2021 Huawei Technologies Co., Ltd
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
from __future__ import division
from .. import ops
from .. import numpy as mnp
from ..numpy import where, zeros_like, dot, greater
from ..ops import functional as F
from ..common import Tensor, CSRTensor
from ..common import dtype as mstype
from .utils_const import _type_convert, _raise_value_error, _super_check, pack
from ..ops.composite import GradOperation

grad = GradOperation(get_all=False, get_by_list=False, sens_param=False)
_eps_net = ops.Eps()


def _convert_64_to_32(tensor):
    """Convert Tensor with float64/int64 types to float32/int32."""
    if tensor.dtype == mstype.float64:
        return tensor.astype("float32")
    if tensor.dtype == mstype.int64:
        return tensor.astype("int32")
    return tensor


def _to_tensor(*args, dtype=None):
    """Returns each input as Tensor"""
    res = ()
    for arg in args:
        if isinstance(arg, (int, float, bool, list, tuple)):
            arg = _type_convert(Tensor, arg)
            if dtype is None:
                arg = _convert_64_to_32(arg)
            else:
                arg = arg.astype(dtype)
        elif not isinstance(arg, Tensor):
            _raise_value_error("Expect input to be array like.")
        res += (arg,)
    if len(res) == 1:
        return res[0]
    return res


def _to_scalar(arr):
    """Convert a scalar Tensor or ndarray to a scalar."""
    if isinstance(arr, (int, float, bool)):
        return arr
    if isinstance(arr, Tensor):
        if arr.shape:
            return arr
        return arr.asnumpy().item()
    raise ValueError("{} are not supported.".format(type(arr)))


def _eps(x):
    return _eps_net(x[(0,) * x.ndim])


def _safe_normalize(x, threshold=None):
    """Normalize method that cast very small results to zero."""
    x_sum2 = F.reduce_sum(F.pows(x, 2.0))
    norm = F.pows(x_sum2, 1. / 2.0)
    if threshold is None:
        if x.dtype in (mstype.float32, mstype.float64):
            # pick the first element of x to get the eps
            threshold = _eps(x)
        else:
            threshold = 0
    use_norm = greater(norm, threshold)
    x_norm = x / norm
    normalized_x = where(use_norm, x_norm, zeros_like(x))
    norm = where(use_norm, norm, zeros_like(norm))
    return normalized_x, norm


def sparse_dot(a, b):
    """Returns the dot product of CSRTensor and generic Tensor(vector)."""
    b_aligned = F.reshape(b, (b.shape[0], -1))
    res = F.csr_mv(a, b_aligned)
    res = F.reshape(res, a.shape[:-1] + b.shape[1:])
    return res


def _matvec(a, b):
    """Computing matrix-vector products."""
    if isinstance(a, Tensor):
        return dot(a, b)

    if isinstance(a, CSRTensor):
        return sparse_dot(a, b)

    return a(b)


def _norm(x, ord_=None):
    if ord_ == mnp.inf:
        res = mnp.max(mnp.abs(x))
    else:
        res = mnp.sqrt(mnp.sum(x ** 2))
    return res


def _nd_transpose(a):
    dims = a.ndim
    if dims < 2:
        _raise_value_error("to do _nd_transpose for input a's ndim is not greater or equal to 2d, which is invalid.")
    axes = ops.make_range(0, dims)
    axes = axes[:-2] + (axes[-1],) + (axes[-2],)
    return ops.transpose(a, axes)


def _value_check(func_name, arg1, arg2, arg_name='', attr_name='', op="in", fmt="attr", msg=None):
    return _super_check(pack(arg1, arg2), (func_name, arg_name, attr_name), op, fmt, msg, True)


def _type_check(func_name, arg1, arg2, arg_name='', op="isinstance", fmt="type", msg=None):
    return _super_check(pack(arg1, arg2), (func_name, arg_name), op, fmt, msg, False)


def _mstype_check(func_name, arg, arg_mstype, arg_name='a'):
    return _super_check((F.typeof(arg), arg_mstype), pack(arg, arg_mstype, func_name, arg_name), "isinstance", "mstype",
                        None, False)


def _dtype_check(func_name, arg, arg_dtype, arg_name='a'):
    return _super_check((F.dtype(arg), arg_dtype), (func_name, arg_name, "data type"), "in", "attr", None, False)


def _square_check(func_name, arg, arg_name='a'):
    arg_shape = arg.shape
    _super_check((len(arg_shape), 2), (func_name, arg_name, 'dimension'), '==', 'attr', None, True)
    _super_check(arg_shape, (func_name, arg_name), '==', 'square', None, True)
    return arg


def _solve_check(func_name, arg1, arg2, arg1_name='a', arg2_name='b', sparse=False):
    arg1_shape, arg1_dtype = arg1.shape, F.dtype(arg1)
    arg2_shape, arg2_dtype = arg2.shape, F.dtype(arg2)
    _square_check(func_name, arg1, arg1_name)
    _super_check((len(arg2_shape), (1, 2)), (func_name, arg2_name, 'dimension'), 'in', 'attr', None, True)
    _super_check((arg1_shape, arg2_shape), (func_name, arg1_name, arg2_name, sparse), 'solve', 'solve', None, True)
    _super_check((arg1_dtype, arg2_dtype), (func_name, arg1_name, arg2_name, 'data type'), '==', 'match', None, False)
    return arg1, arg2
