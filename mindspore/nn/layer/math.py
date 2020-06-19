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
"""math"""
import math
from mindspore.ops import operations as P
from mindspore.ops.operations import _inner_ops as inner
from mindspore.common.tensor import Tensor
from ..cell import Cell
from ...common import dtype as mstype
from ..._checkparam import Validator as validator

__all__ = ['ReduceLogSumExp', 'Range']

class ReduceLogSumExp(Cell):
    r"""
    Reduce a dimension of a tensor by calculating exponential for all elements in the dimension,
    then calculate logarithm of the sum.

    The dtype of the tensor to be reduced is number.

    Args:
        keep_dims (bool): If True, keep these reduced dimensions and the length is 1.
                          If False, don't keep these dimensions.
                          Default : False.

    Inputs:
        - **input_x** (Tensor[Number]) - The input tensor.
        - **axis** (Union[int, tuple(int), list(int)]) - The dimensions to reduce. Default: (), reduce all dimensions.
          Only constant value is allowed.

    Outputs:
        Tensor, has the same dtype as the 'input_x'.

        - If axis is (), and keep_dims is false,
          the output is a 0-D tensor representing the sum of all elements in the input tensor.
        - If axis is int, set as 2, and keep_dims is false,
          the shape of output is :math:`(x_1, x_3, ..., x_R)`.
        - If axis is tuple(int), set as (2, 3), and keep_dims is false,
          the shape of output is :math:`(x_1, x_4, ..., x_R)`.

    Examples:
        >>> input_x = Tensor(np.random.randn(3, 4, 5, 6).astype(np.float32))
        >>> op = P.ReduceLogSumExp(keep_dims=True)
        >>> output = op(input_x, 1)
    """

    def __init__(self, axis, keep_dims=False):
        super(ReduceLogSumExp, self).__init__()
        validator.check_value_type('axis', axis, [int, list, tuple], self.cls_name)
        validator.check_value_type('keep_dims', keep_dims, [bool], self.cls_name)
        self.axis = axis
        self.exp = P.Exp()
        self.sum = P.ReduceSum(keep_dims)
        self.log = P.Log()

    def construct(self, input_x):
        exp = self.exp(input_x)
        sumexp = self.sum(exp, self.axis)
        logsumexp = self.log(sumexp)
        return logsumexp


class Range(Cell):
    r"""
    Creates a sequence of numbers.

    Args:
        start (Union[int, float]): If `limit` is `None`, the value acts as limit in the range and first entry
            defaults to `0`. Otherwise, it acts as first entry in the range.
        limit (Union[int, float]): Acts as upper limit of sequence. If `None`, defaults to the value of `start`
            while set the first entry of the range to `0`. It can not be equal to `start`.
        delta (Union[int, float]): Increment of the range. It can not be equal to zero. Default: 1.

    Outputs:
        Tensor, the dtype is int if the dtype of `start`, `limit` and `delta` all are int. Otherwise, dtype is float.

    Examples:
        >>> net = nn.Range(1, 8, 2)
        >>> out = net()
        [1, 3, 5, 7]
    """

    def __init__(self, start, limit=None, delta=1):
        super(Range, self).__init__()
        validator.check_value_type("start", start, [int, float], self.cls_name)
        validator.check_value_type("delta", delta, [int, float], self.cls_name)
        if delta == 0:
            raise ValueError("The input of `delta` can not be equal to zero.")
        if limit is not None:
            validator.check_value_type("limit", limit, [int, float], self.cls_name)
            if isinstance(start, int) and isinstance(limit, int) and isinstance(delta, int):
                self.dtype = mstype.int32
            else:
                self.dtype = mstype.float32
        else:
            if isinstance(start, int) and isinstance(delta, int):
                self.dtype = mstype.int32
            else:
                self.dtype = mstype.float32
        if isinstance(start, int):
            start = float(start)
        if isinstance(limit, int):
            limit = float(limit)
        if isinstance(delta, int):
            delta = float(delta)
        self.range_x = inner.Range(start, limit, delta)
        if limit is None:
            length_input = math.ceil(start / delta)
        else:
            length_input = math.ceil((limit - start) / delta)
        self.input_tensor = Tensor(list(range(length_input)), self.dtype)

    def construct(self):
        range_out = self.range_x(self.input_tensor)
        return range_out
