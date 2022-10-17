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

import numpy as np
import pytest

import mindspore
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.ops.operations import _grad_ops as G


def soft_shrink_op_np_bencmark(input_x, lambd):
    result = input_x.asnumpy().copy()
    size = input_x.size
    result = result.reshape(size)

    for index in range(size):
        if result[index] > lambd:
            result[index] = result[index] - lambd
        elif result[index] < -lambd:
            result[index] = result[index] + lambd
        else:
            result[index] = 0

    result = result.reshape(input_x.shape)
    return result


class SoftShrinkNet(nn.Cell):
    def __init__(self, lambd):
        super(SoftShrinkNet, self).__init__()
        self.soft_shrink = P.SoftShrink(lambd)

    def construct(self, input_x):
        return self.soft_shrink(input_x)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('dtype', [np.float32])
@pytest.mark.parametrize("data_shape", [(3, 4), (4, 5, 6, 7)])
@pytest.mark.parametrize("lambd", [0.5, 0.75])
def test_soft_shrink(dtype, data_shape, lambd):
    """
    Feature: SoftShrink cpu kernel
    Description: test the rightness of SoftShrink cpu kernel
    Expectation: the output is same as soft_shrink_op_np_bencmark output
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')

    data = np.random.uniform(low=-1, high=1, size=data_shape).astype(dtype)
    input_tensor = Tensor(data)
    benchmark_output = soft_shrink_op_np_bencmark(input_tensor, lambd)

    soft_shrink_net = SoftShrinkNet(lambd)
    output = soft_shrink_net(input_tensor)
    np.testing.assert_array_almost_equal(output.asnumpy(), benchmark_output)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_soft_shrink_dy_shape():
    """
    Feature: test_soft_shrink_dy_shape.
    Description: test cases for dynamic shape.
    Expectation: raise TypeError.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    lambd = 0.5

    np.random.seed(1)
    in_np = np.random.rand(3, 5, 2).astype(np.float32)
    in_tensor = Tensor(in_np)
    in_t_ds = Tensor(shape=[3, 5, None], dtype=mindspore.float32)

    net = SoftShrinkNet(lambd)
    net.set_inputs(in_t_ds)

    output_ms = net(in_tensor)
    output_np = soft_shrink_op_np_bencmark(in_tensor, 0.5)

    np.testing.assert_allclose(output_ms.asnumpy(), output_np, rtol=1e-3)


class ShapeSoftShrinkGradNet(nn.Cell):
    def __init__(self):
        super(ShapeSoftShrinkGradNet, self).__init__()
        self.soft_shrink_grad_op = G.SoftShrinkGrad()

    def construct(self, in_x, grad):
        return self.soft_shrink_grad_op(in_x, grad)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_soft_shrink_grad_ds_shape():
    """
    Feature: test_soft_shrink_dy_shape.
    Description: test cases for dynamic shape.
    Expectation: raise TypeError.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    soft_shrink_net = ShapeSoftShrinkGradNet()

    x = Tensor(np.random.randn(10, 20).astype(np.float32))
    grad = Tensor(np.random.randn(10, 20).astype(np.float32))
    static_output = soft_shrink_net(x, grad)

    x_ds = Tensor(shape=[None, 20], dtype=mindspore.float32)
    grad_ds = Tensor(shape=[None, 20], dtype=mindspore.float32)
    soft_shrink_net.set_inputs(x_ds, grad_ds)
    dynamic_output = soft_shrink_net(x, grad)

    np.testing.assert_allclose(static_output.asnumpy(), dynamic_output.asnumpy(), rtol=1e-3)


def softshrink_grad_op_np_bencmark(grad, input_x, lambd=0.5):
    result = np.zeros_like(grad, dtype=grad.dtype)
    for index, _ in np.ndenumerate(grad):
        if input_x[index] > lambd or input_x[index] < (-1 * lambd):
            result[index] = grad[index]
        else:
            result[index] = 0
    return result


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize("data_shape", [(3, 4), (4, 5, 6, 7)])
def test_softshrink_grad(data_shape):
    """
    Feature: SoftShrinkGrad cpu kernel
    Description: test the rightness of SoftShrinkGrad cpu kernel
    Expectation: the output is same as softshrink_grad_op_np_bencmark output
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    grad_data = np.random.random(data_shape).astype(np.float32)
    input_data = np.random.uniform(
        low=-1, high=1, size=data_shape).astype(np.float32)
    benchmark_output = softshrink_grad_op_np_bencmark(
        grad_data, input_data)
    hshrink_grad = ShapeSoftShrinkGradNet()
    output = hshrink_grad(Tensor(grad_data), Tensor(input_data))
    assert np.allclose(output.asnumpy(), benchmark_output)
