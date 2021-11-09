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
import numpy as onp
from .. import nn, ops
from ..numpy import where, isnan, zeros_like
from ..ops import functional as F
from ..common import Tensor
from ..common import dtype as mstype
from .utils_const import _type_convert, _raise_type_error
from ..ops.composite import GradOperation

grad = GradOperation(get_all=False, get_by_list=False, sens_param=False)


def _convert_64_to_32(tensor):
    """Convert tensor with float64/int64 types to float32/int32."""
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
            _raise_type_error("Expect input to be array like.")
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
        arr = arr.asnumpy()
    if isinstance(arr, onp.ndarray):
        if arr.shape:
            return arr
        return arr.item()
    raise ValueError("{} are not supported.".format(type(arr)))


class _SafeNormalize(nn.Cell):
    """Normalize method that cast very small results to zero."""

    def __init__(self):
        """Initialize LineSearch."""
        super(_SafeNormalize, self).__init__()
        self.eps = ops.Eps()

    def construct(self, x, threshold=None):
        x_sum2 = F.reduce_sum(F.pows(x, 2.0))
        norm = F.pows(x_sum2, 1. / 2.0)
        if threshold is None:
            if x.dtype in mstype.float_type:
                # pick the first element of x to get the eps
                threshold = self.eps(x[(0,) * x.ndim])
            else:
                threshold = 0
        normalized_x = where(norm > threshold, x / norm, zeros_like(x))
        normalized_x = where(isnan(normalized_x), 0, normalized_x)
        return normalized_x, norm


_safe_normalize = _SafeNormalize()

_INT_ZERO = _to_tensor(0)
_INT_ONE = _to_tensor(1)
_BOOL_TRUE = _to_tensor(True)
_BOOL_FALSE = _to_tensor(False)
