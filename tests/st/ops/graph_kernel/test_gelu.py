# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
import mindspore.context as context
from mindspore import Tensor
from mindspore.nn import Cell
import mindspore.ops.operations as P
import mindspore.ops.operations._grad_ops as G


class GeluNet(Cell):
    def __init__(self):
        super(GeluNet, self).__init__()
        self.gelu = P.GeLU()

    def construct(self, x):
        return self.gelu(x)


class GeluGradNet(Cell):
    def __init__(self):
        super(GeluGradNet, self).__init__()
        self.gelu_grad = G.GeLUGrad()

    def construct(self, dy, x, y):
        return self.gelu_grad(dy, x, y)


def cal_gelu(x):
    tmp = np.sqrt(2.0 / np.pi) * (x + 0.044715 * x * x * x)
    expect = 0.5 * x * (1.0 + np.tanh(tmp))
    return expect

def gelu(x, enable_graph_kernel=False):
    context.set_context(enable_graph_kernel=enable_graph_kernel)
    net = GeluNet()
    result = net(Tensor(x))
    return result

def test_gelu():
    np.random.seed(0)
    input_x = np.random.normal(0, 1, [2, 3, 4, 3]).astype(np.float32)
    expect = gelu(input_x, False)
    result = gelu(input_x, True)
    res = np.allclose(expect.asnumpy(), result.asnumpy(), rtol=1.e-4, atol=1.e-4, equal_nan=True)
    assert res

def cal_gelu_grad():
    tanh_res = np.tanh(0.7978845608 * (input_x + 0.044715 * input_x * input_x * input_x))
    mul_right = 0.7978845608 + 0.1070322244 * input_x * input_x
    dx = 0.5 * (1.0 + tanh_res) + 0.5 * input_x * (1.0 - tanh_res * tanh_res) * mul_right
    expect = input_dy * dx
    return expect

def gelu_grad(input_dy, input_x, input_y, enable_graph_kernel=False):
    context.set_context(enable_graph_kernel=enable_graph_kernel)
    net = GeluGradNet()
    result = net(Tensor(input_dy), Tensor(input_x), Tensor(input_y))
    return result

def test_gelu_grad():
    np.random.seed(0)
    input_dy = np.random.normal(0, 1, [2, 3, 4, 3]).astype(np.float32)
    input_x = np.random.normal(0, 1, [2, 3, 4, 3]).astype(np.float32)
    input_y = cal_gelu(input_x)

    expect = gelu_grad(input_dy, input_x, input_y, False)
    result = gelu_grad(input_dy, input_x, input_y, True)
    res = np.allclose(expect.asnumpy(), result.asnumpy(), rtol=1.e-4, atol=1.e-4, equal_nan=True)
    assert res


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_gelu_gpu():
    context.set_context(mode=context.GRAPH_MODE, enable_graph_kernel=True, device_target="GPU")
    test_gelu()

@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_gelu_ascend():
    context.set_context(mode=context.GRAPH_MODE, enable_graph_kernel=True, device_target="Ascend")
    test_gelu()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_gelu_grad_gpu():
    context.set_context(mode=context.GRAPH_MODE, enable_graph_kernel=True, device_target="GPU")
    test_gelu_grad()

@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_gelu_grad_ascend():
    context.set_context(mode=context.GRAPH_MODE, enable_graph_kernel=True, device_target="Ascend")
    test_gelu_grad()
