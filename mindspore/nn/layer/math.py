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
from mindspore.ops import operations as P
from ..cell import Cell
from ..._checkparam import Validator as validator

__all__ = ['ReduceLogSumExp']

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
