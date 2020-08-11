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
import numpy as np
from mindspore.ops import operations as P
from mindspore.ops.operations import _inner_ops as inner
from mindspore.common.tensor import Tensor
from mindspore.ops.primitive import constexpr
from ..cell import Cell
from ...common import dtype as mstype
from ..._checkparam import Validator as validator
from ..._checkparam import Rel


__all__ = ['ReduceLogSumExp', 'Range', 'LinSpace', 'LGamma']


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
        >>> op = nn.ReduceLogSumExp(keep_dims=True)
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


class LinSpace(Cell):
    r"""
    Generates values in an interval.

    Args:
        start (Union[int, float]): The start of interval. With shape of 0-D.
        stop (Union[int, float]): The end of interval. With shape of 0-D.
        num (int): ticks number in the interval, the ticks include start and stop value. With shape of 0-D.

    Outputs:
        Tensor, With type same as `start`. The shape is 1-D with length of `num`.

    Examples:
        >>> linspace = nn.LinSpace(1, 10, 5)
        >>> output = linspace()
        [1, 3.25, 5.5, 7.75, 10]
    """

    def __init__(self, start, stop, num):
        super(LinSpace, self).__init__()
        validator.check_value_type("start", start, [int, float], self.cls_name)
        validator.check_value_type("stop", stop, [int, float], self.cls_name)
        validator.check_value_type("num", num, [int], self.cls_name)
        validator.check_integer("num", num, 0, Rel.GT, self.cls_name)

        self.is_single = bool(num == 1)
        self.lin_space = inner.LinSpace()
        self.start = Tensor(start, mstype.float32)
        self.stop = Tensor(stop, mstype.float32)
        self.assist = Tensor(list(range(num)), mstype.float32)
        self.num = Tensor(num, mstype.int32)
        self.start_array = Tensor([start], mstype.float32)

    def construct(self):
        if self.is_single:
            return self.start_array

        lin_space_out = self.lin_space(self.assist, self.start, self.stop, self.num)
        return lin_space_out

@constexpr
def check_tensors_dtype_same(data_dtype, value_dtype, op_name):
    """Check tensors data type same."""
    if data_dtype in value_dtype:
        return True
    raise TypeError(f"For '{op_name}', the value data type '{value_dtype}' "
                    f"is not consistent with assigned tensor data type {data_dtype}.")

class LGamma(Cell):
    r"""
    Calculate LGamma using Lanczos' approximation refering to "A Precision Approximationof the Gamma Function".
    The algorithm is:

    .. math::
        lgamma(z + 1) = \frac{(\log(2) + \log(pi))}{2} + (z + 1/2) * log(t(z)) - t(z) + A(z)

        t(z) = z + kLanczosGamma + 1/2

        A(z) = kBaseLanczosCoeff + \sum_{k=1}^n \frac{kLanczosCoefficients[i]}{z + k}

    However, if the input is less than 0.5 use Euler's reflection formula:

    .. math::

        lgamma(x) = \log(pi) - lgamma(1-x) - \log(abs(sin(pi * x)))

    And please note that

    .. math::

        lgamma(+/-inf) = +inf

    Thus, the behaviour of LGamma follows:
    when x > 0.5, return log(Gamma(x))
    when x < 0.5 and is not an interger, return the real part of Log(Gamma(x)) where Log is the complex logarithm
    when x is an integer less or equal to 0, return +inf
    when x = +/- inf, return +inf

    Inputs:
        - **input_x** (Tensor[Number]) - The input tensor. Only float16, float32 are supported.

    Outputs:
        Tensor, has the same shape and dtype as the 'input_x'.

    Examples:
        >>> input_x = Tensor(np.array(2, 3, 4).astype(np.float32))
        >>> op = nn.LGamma()
        >>> output = op(input_x)
    """

    def __init__(self):
        super(LGamma, self).__init__()
        # const numbers
        self.k_lanczos_gamma = 7
        self.k_base_lanczos_coeff = 0.99999999999980993227684700473478
        self.k_lanczos_coefficients = [676.520368121885098567009190444019,
                                       -1259.13921672240287047156078755283,
                                       771.3234287776530788486528258894,
                                       -176.61502916214059906584551354,
                                       12.507343278686904814458936853,
                                       -0.13857109526572011689554707,
                                       9.984369578019570859563e-6,
                                       1.50563273514931155834e-7]
        self.one_half = 0.5
        self.one = 1
        self.two = 2
        self.inf = np.inf
        self.pi = np.pi
        self.log_2 = np.log(self.two)
        self.log_pi = np.log(np.pi)
        self.log_sqrt_two_pi = (self.log_2 + self.log_pi) / self.two
        self.lanczos_gamma_plus_one_half = self.k_lanczos_gamma + 0.5
        self.log_lanczos_gamma_plus_one_half = np.log(self.lanczos_gamma_plus_one_half)

        # operations
        self.log = P.Log()
        self.log1p = P.Log1p()
        self.abs = P.Abs()
        self.shape = P.Shape()
        self.dtype = P.DType()
        self.fill = P.Fill()
        self.floor = P.Floor()
        self.equal = P.Equal()
        self.greater = P.Greater()
        self.less = P.Less()
        self.lessequal = P.LessEqual()
        self.select = P.Select()
        self.sin = P.Sin()
        self.isfinite = P.IsFinite()

    def construct(self, input_x):
        input_dtype = self.dtype(input_x)
        check_tensors_dtype_same(input_dtype, [mstype.float16, mstype.float32], "LGamma")
        infinity = self.fill(input_dtype, self.shape(input_x), self.inf)

        need_to_reflect = self.less(input_x, 0.5)
        neg_input = -input_x
        z = self.select(need_to_reflect, neg_input, input_x - 1)

        @constexpr
        def _calculate_x(z, k_base_lanczos_coeff, k_lanczos_coefficients):
            x = k_base_lanczos_coeff
            for i in range(8):
                product_ = k_lanczos_coefficients[i] / (z + i + 1)
                x = product_ + x
            return  x
        x = _calculate_x(z, self.k_base_lanczos_coeff, self.k_lanczos_coefficients)

        t = z + self.lanczos_gamma_plus_one_half
        log_t = self.log1p(z / self.lanczos_gamma_plus_one_half) + self.log_lanczos_gamma_plus_one_half

        log_y = self.log(x) + (z + self.one_half - t / log_t) * log_t + self.log_sqrt_two_pi

        abs_input = self.abs(input_x)
        abs_frac_input = abs_input - self.floor(abs_input)
        input_x = self.select(self.lessequal(input_x, 0.0),
                              self.select(self.equal(abs_frac_input, 0.0),
                                          infinity, input_x),
                              input_x)
        reduced_frac_input = self.select(self.greater(abs_frac_input, 0.5),
                                         1 - abs_frac_input, abs_frac_input)
        reflection_denom = self.log(self.sin(self.pi * reduced_frac_input))

        reflection = self.select(self.isfinite(reflection_denom),
                                 -reflection_denom - log_y + self.log_pi,
                                 -reflection_denom)

        result = self.select(need_to_reflect, reflection, log_y)

        return self.select(self.isfinite(input_x), result, infinity)
