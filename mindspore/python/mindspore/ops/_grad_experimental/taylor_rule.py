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

from __future__ import absolute_import
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.ops.primitive import Primitive
from mindspore.ops import operations as P
from mindspore.ops.composite.multitype_ops.zeros_like_impl import zeros_like
from mindspore.ops._grad_experimental.grad_base import taylor_fprop_getters


def _factorial(order):
    """Return [0!, 1!, 2!,..., order!]."""
    range_op = nn.Range(1, order + 1)
    factorial_zero = ops.ones(1, ms.float32)
    factorial_positive = range_op().astype(ms.float32)
    for i in range(1, order):
        factorial_positive[i] *= factorial_positive[i - 1]
    factorial = ops.concat((factorial_zero, factorial_positive))
    return factorial


def _trans_scalar_inputs(x, y):
    """
    If an op accepts multiple inputs and support scalar input by broadcasting it to another input, it requires
    special process(like Mul op) or different broadcast in taylor rule.
    """
    if not x.shape:
        trans_x = zeros_like(y)
        trans_x[0] = x
        return trans_x, y
    if not y.shape:
        trans_y = zeros_like(x)
        trans_y[0] = y
        return x, trans_y
    return x, y


@taylor_fprop_getters.register(P.Add)
@taylor_fprop_getters.register(P.Sub)
def taylor_add_or_sub(self):
    """Higher order derivatives rule definition for `Add` or `Sub`operation."""
    if isinstance(self, str):
        prim = Primitive(self)
    else:
        prim = self

    def taylor_fprop_add_or_sub(input_x, input_y):
        if not input_x.shape and not input_y.shape:
            return prim(input_x, input_y)
        input_x, input_y = _trans_scalar_inputs(input_x, input_y)
        return prim(input_x, input_y)

    return taylor_fprop_add_or_sub


def _taylor_fprop_mul(input_x, input_y):
    """The rule to generate `Mul` taylor rule."""
    if not input_x.shape or not input_y.shape:
        return input_x * input_y
    primals = ops.mul(input_x[0], input_y[0])
    series_num = len(input_x) - 1
    factorial = _factorial(series_num)
    series = zeros_like(input_x)
    series[0] = primals
    for k in range(1, series_num + 1):
        for i in range(0, k + 1):
            tmp = ops.div(input_x[i] * input_y[k - i], (factorial[k - i] * factorial[i]))
            series[k] += tmp
        series[k] *= factorial[k]
    return series


@taylor_fprop_getters.register(P.Mul)
def taylor_mul(self):
    """Higher order derivatives rule definition for `Mul` operation."""

    def taylor_fprop_mul(input_x, input_y):
        series = _taylor_fprop_mul(input_x, input_y)
        return series

    return taylor_fprop_mul


def _taylor_fprop_realdiv(input_x, input_y):
    """The rule to generate `RealDiv` taylor rule."""
    if not input_y.shape:
        return ops.div(input_x, input_y)
    input_x, input_y = _trans_scalar_inputs(input_x, input_y)
    primals = ops.div(input_x[0], input_y[0])
    series_num = len(input_x) - 1
    factorial = _factorial(series_num)
    series = zeros_like(input_x)
    series[0] = primals
    for k in range(1, series_num + 1):
        for i in range(0, k):
            tmp = ops.div(series[i] * input_y[k - i], (factorial[k - i] * factorial[i]))
            series[k] += tmp
        series[k] = ops.div(input_x[k] - factorial[k] * series[k], input_y[0])
    return series


@taylor_fprop_getters.register(P.RealDiv)
def taylor_realdiv(self):
    """Higher order derivatives rule definition for `RealDiv` operation."""

    def taylor_fprop_realdiv(input_x, input_y):
        series = _taylor_fprop_realdiv(input_x, input_y)
        return series

    return taylor_fprop_realdiv


@taylor_fprop_getters.register(P.Exp)
def taylor_exp(self):
    """Higher order derivatives rule definition for `Exp` operation."""

    def taylor_fprop_exp(inputs):
        primals = ops.exp(inputs[0])
        series_num = len(inputs) - 1
        factorial = _factorial(series_num)
        series = zeros_like(inputs)
        series[0] = primals
        for k in range(1, series_num + 1):
            for i in range(1, k + 1):
                tmp = ops.div(i * inputs[i] * series[k - i], (factorial[k - i] * factorial[i]))
                series[k] += tmp
            series[k] *= factorial[k - 1]
        return series

    return taylor_fprop_exp


@taylor_fprop_getters.register(P.Log)
def taylor_log(self):
    """Higher order derivatives rule definition for `Log` operation."""

    def taylor_fprop_log(inputs):
        primals = ops.log(inputs[0])
        series_num = len(inputs) - 1
        factorial = _factorial(series_num)
        series = zeros_like(inputs)
        series[0] = primals
        for k in range(1, series_num + 1):
            for i in range(1, k):
                tmp = ops.div(i * inputs[k - i] * series[i], factorial[k - i] * factorial[i])
                series[k] += tmp
            series[k] = ops.div(inputs[k] - factorial[k - 1] * series[k], inputs[0])
        return series

    return taylor_fprop_log


def _taylor_fprop_sin_cos(inputs):
    """Common rule to generate `Sin` and `Cos` taylor rules."""
    primal_sin = ops.sin(inputs[0])
    primal_cos = ops.cos(inputs[0])
    series_sin = zeros_like(inputs)
    series_cos = zeros_like(inputs)
    series_sin[0] = primal_sin
    series_cos[0] = primal_cos
    series_num = len(inputs) - 1
    factorial = _factorial(series_num)
    for k in range(1, series_num + 1):
        for i in range(1, k + 1):
            series_sin[k] += ops.div(i * inputs[i] * series_cos[k - i], factorial[i] * factorial[k - i])
            series_cos[k] -= ops.div(i * inputs[i] * series_sin[k - i], factorial[i] * factorial[k - i])
        series_sin[k] *= factorial[k - 1]
        series_cos[k] *= factorial[k - 1]
    return series_sin, series_cos


@taylor_fprop_getters.register(P.Sin)
def taylor_sin(self):
    """Higher order derivatives rule definition for `Sin` operation."""

    def taylor_fprop_sin(inputs):
        series_sin = _taylor_fprop_sin_cos(inputs)[0]
        return series_sin

    return taylor_fprop_sin


@taylor_fprop_getters.register(P.Cos)
def taylor_cos(self):
    """Higher order derivatives rule definition for `Cos` operation."""

    def taylor_fprop_cos(inputs):
        series_cos = _taylor_fprop_sin_cos(inputs)[1]
        return series_cos

    return taylor_fprop_cos


@taylor_fprop_getters.register(P.Tan)
def taylor_tan(self):
    """Higher order derivatives rule definition for `Tan` operation."""

    def taylor_fprop_tan(inputs):
        series_sin_cos = _taylor_fprop_sin_cos(inputs)
        series_tan = _taylor_fprop_realdiv(series_sin_cos[0], series_sin_cos[1])
        return series_tan

    return taylor_fprop_tan
