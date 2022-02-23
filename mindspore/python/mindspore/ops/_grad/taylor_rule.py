# Copyright 2022 Huawei Technologies Co., Ltd
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

"""Define the taylor rules of operations."""

from mindspore import nn
import mindspore as ms
from ..primitive import Primitive
from .. import operations as P
from ..composite.multitype_ops.zeros_like_impl import zeros_like
from .grad_base import taylor_fprop_getters


def _factorial(order):
    """Return [0!, 1!, 2!,..., order!]."""
    range_op = nn.Range(1, order + 1)
    ones_op = P.Ones()
    concat_op = P.Concat()
    factorial_zero = ones_op(1, ms.float32)
    factorial_positive = range_op().astype(ms.float32)
    for i in range(1, order):
        factorial_positive[i] *= factorial_positive[i - 1]
    factorial = concat_op((factorial_zero, factorial_positive))
    return factorial


@taylor_fprop_getters.register(P.Add)
@taylor_fprop_getters.register(P.Sub)
def taylor_add_or_sub(self):
    """Higher order derivatives rule definition for `Add` or `Sub`operation."""
    if isinstance(self, str):
        prim = Primitive(self)
    else:
        prim = self
    def taylor_fprop_add_or_sub(input_x, input_y):
        series = prim(input_x, input_y)
        return series

    return taylor_fprop_add_or_sub


@taylor_fprop_getters.register(P.Mul)
def taylor_mul(self):
    """Higher order derivatives rule definition for `Mul` operation."""
    mul_func = P.Mul()

    def taylor_fprop_mul(input_x, input_y):
        primals = mul_func(input_x[0], input_y[0])
        series_num = len(input_x) - 1
        factorial = _factorial(series_num)
        series = zeros_like(input_x)
        series[0] = primals
        for k in range(1, series_num + 1):
            for i in range(0, k + 1):
                tmp = input_x[i] * input_y[k - i] / (factorial[k - i] * factorial[i])
                series[k] += tmp
            series[k] *= factorial[k]
        return series

    return taylor_fprop_mul


@taylor_fprop_getters.register(P.RealDiv)
def taylor_realdiv(self):
    """Higher order derivatives rule definition for `RealDiv` operation."""
    div_op = P.Div()

    def taylor_fprop_realdiv(input_x, input_y):
        primals = div_op(input_x[0], input_y[0])
        series_num = len(input_x) - 1
        factorial = _factorial(series_num)
        series = zeros_like(input_x)
        series[0] = primals
        for k in range(1, series_num + 1):
            for i in range(0, k):
                tmp = series[i] * input_y[k - i] / (factorial[k - i] * factorial[i])
                series[k] += tmp
            series[k] = (input_x[k] - factorial[k] * series[k]) / input_y[0]
        return series

    return taylor_fprop_realdiv


@taylor_fprop_getters.register(P.Exp)
def taylor_exp(self):
    """Higher order derivatives rule definition for `Exp` operation."""
    exp_ = P.Exp()

    def taylor_fprop_exp(inputs):
        primals = exp_(inputs[0])
        series_num = len(inputs) - 1
        factorial = _factorial(series_num)
        series = zeros_like(inputs)
        series[0] = primals
        for k in range(1, series_num + 1):
            for i in range(1, k + 1):
                tmp = i * inputs[i] * series[k - i] / (factorial[k - i] * factorial[i])
                series[k] += tmp
            series[k] *= factorial[k - 1]
        return series

    return taylor_fprop_exp


@taylor_fprop_getters.register(P.Sin)
def taylor_sin(self):
    """Higher order derivatives rule definition for `Sin` operation."""
    cos = P.Cos()
    sin = P.Sin()

    def taylor_fprop_sin(inputs):
        primal_sin = sin(inputs[0])
        primal_cos = cos(inputs[0])
        series_sin = zeros_like(inputs)
        series_cos = zeros_like(inputs)
        series_sin[0] = primal_sin
        series_cos[0] = primal_cos
        series_num = len(inputs) - 1
        factorial = _factorial(series_num)
        for k in range(1, series_num + 1):
            for i in range(1, k + 1):
                series_sin[k] += i * inputs[i] * series_cos[k - i] / (factorial[i] * factorial[k - i])
                series_cos[k] -= i * inputs[i] * series_sin[k - i] / (factorial[i] * factorial[k - i])
            series_sin[k] *= factorial[k - 1]
            series_cos[k] *= factorial[k - 1]
        return series_sin

    return taylor_fprop_sin


@taylor_fprop_getters.register(P.Cos)
def taylor_cos(self):
    """Higher order derivatives rule definition for `Cos` operation."""
    cos = P.Cos()
    sin = P.Sin()

    def taylor_fprop_cos(inputs):
        primal_cos = cos(inputs[0])
        primal_sin = sin(inputs[0])
        series_cos = zeros_like(inputs)
        series_sin = zeros_like(inputs)
        series_cos[0] = primal_cos
        series_sin[0] = primal_sin
        series_num = len(inputs) - 1
        factorial = _factorial(series_num)
        for k in range(1, series_num + 1):
            for i in range(1, k + 1):
                series_cos[k] -= i * inputs[i] * series_sin[k - i] / (factorial[i] * factorial[k - i])
                series_sin[k] += i * inputs[i] * series_cos[k - i] / (factorial[i] * factorial[k - i])
            series_cos[k] *= factorial[k - 1]
            series_sin[k] *= factorial[k - 1]
        return series_cos

    return taylor_fprop_cos
