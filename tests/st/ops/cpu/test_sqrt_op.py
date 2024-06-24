# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
from tests.mark_utils import arg_mark

import numpy as np
import pytest
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.ops.operations import _grad_ops as G


class NetSqrtGrad(nn.Cell):
    def __init__(self):
        super(NetSqrtGrad, self).__init__()
        self.sqrt_grad = G.SqrtGrad()

    def construct(self, x, dx):
        return self.sqrt_grad(x, dx)


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.ops = P.Sqrt()

    def construct(self, x):
        return self.ops(x)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_sqrt(dtype):
    """
    Feature: ALL To ALL
    Description: test cases for Sqrt
    Expectation: the result match to numpy
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    x = np.abs(np.random.randn(2, 3, 3, 4)).astype(dtype)
    y_expect = np.sqrt(x)
    net = Net()
    out = net(Tensor(x))
    diff = out.asnumpy() - y_expect
    err = np.ones(shape=y_expect.shape) * 1.0e-5
    assert np.all(diff < err)
    assert out.shape == y_expect.shape


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_sqrt_grad(dtype):
    """
    Feature: ALL To ALL
    Description: test cases for SqrtGrad
    Expectation: the result match to numpy
    """
    x = Tensor(np.array([[[[-1, 1, 10],
                           [5.9, 6.1, 6],
                           [10, 1, -1]]]]).astype(dtype))
    dx = Tensor(np.array([[[[1, 1, 1],
                            [2, 2, 2],
                            [3, 3, 3]]]]).astype(dtype))
    expect = np.array([[[[-0.5, 0.5, 0.05],
                         [0.16949153, 0.16393442, 0.16666667],
                         [0.15, 1.5, -1.5]]]]).astype(dtype)

    error = np.ones(shape=[3, 3]) * 1.0e-6

    sqrt_grad = NetSqrtGrad()
    output = sqrt_grad(x, dx)
    diff = np.abs(output.asnumpy() - expect)
    assert np.all(np.abs(diff) < error)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_sqrt_dy_shape(dtype):
    """
    Feature: ALL To ALL
    Description: test cases for Sqrt dynamic shape
    Expectation: the result match to numpy
    """
    context.set_context(mode=context.GRAPH_MODE)
    input_x_np = np.abs(np.random.randn(2, 3, 3, 4)).astype(dtype)
    benchmark_output = np.sqrt(input_x_np)
    loss = 1e-6
    sqrt_net = Net()
    real_input = Tensor(input_x_np)
    dy_shape = [None for _ in input_x_np.shape]
    input_dyn = Tensor(shape=dy_shape, dtype=real_input.dtype)
    sqrt_net.set_inputs(input_dyn)
    ms_result = sqrt_net(real_input)
    np.testing.assert_allclose(benchmark_output, ms_result.asnumpy(), rtol=loss, atol=loss)
    context.set_context(mode=context.PYNATIVE_MODE)
    ms_result = sqrt_net(real_input)
    np.testing.assert_allclose(benchmark_output, ms_result.asnumpy(), rtol=loss, atol=loss)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_sqrt_grad_dy_shape(dtype):
    """
    Feature: ALL To ALL
    Description: test cases for SqrtGrad dynamic shape
    Expectation: the result match to numpy
    """
    context.set_context(mode=context.GRAPH_MODE)
    input_x_np = np.array([[[[-1, 1, 10],
                             [5.9, 6.1, 6],
                             [10, 1, -1]]]]).astype(dtype)
    dx_np = np.array([[[[1, 1, 1],
                        [2, 2, 2],
                        [3, 3, 3]]]]).astype(dtype)
    benchmark_output = np.array([[[[-0.5, 0.5, 0.05],
                                   [0.16949153, 0.16393442, 0.16666667],
                                   [0.15, 1.5, -1.5]]]]).astype(dtype)
    loss = np.ones(shape=[3, 3]) * 1.0e-6
    sqrt_grad_net = NetSqrtGrad()
    real_input = Tensor(input_x_np)
    real_dx = Tensor(dx_np)
    input_dy_shape = [None for _ in input_x_np.shape]
    dx_dy_shape = [None for _ in dx_np.shape]
    input_dyn = Tensor(shape=input_dy_shape, dtype=real_input.dtype)
    dx_dyn = Tensor(shape=dx_dy_shape, dtype=real_dx.dtype)
    sqrt_grad_net.set_inputs(input_dyn, dx_dyn)
    output = sqrt_grad_net(real_input, real_dx)
    diff = np.abs(output.asnumpy() - benchmark_output)
    assert np.all(np.abs(diff) < loss)
    context.set_context(mode=context.PYNATIVE_MODE)
    output = sqrt_grad_net(real_input, real_dx)
    diff = np.abs(output.asnumpy() - benchmark_output)
    assert np.all(np.abs(diff) < loss)
