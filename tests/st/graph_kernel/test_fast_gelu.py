# Copyright 2024 Huawei Technologies Co., Ltd
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
from tests.mark_utils import arg_mark
import mindspore.context as context
from mindspore import Tensor
from mindspore.nn import Cell
import mindspore.ops.operations as P
import mindspore.ops.operations._grad_ops as G


class GeluNet(Cell):
    def __init__(self):
        super(GeluNet, self).__init__()
        self.FastGeLU = P.FastGeLU()

    def construct(self, x):
        return self.FastGeLU(x)


class GeluGradNet(Cell):
    def __init__(self):
        super(GeluGradNet, self).__init__()
        self.gelu_grad = G.FastGeLUGrad()

    def construct(self, dy, x, y):
        return self.gelu_grad(dy, x)


def cal_gelu(x):
    tmp = np.sqrt(2.0 / np.pi) * (x + 0.044715 * x * x * x)
    expect = 0.5 * x * (1.0 + np.tanh(tmp))
    return expect


def FastGeLU(x, enable_graph_kernel=False):
    context.set_context(enable_graph_kernel=enable_graph_kernel)
    net = GeluNet()
    result = net(Tensor(x))
    return result


def run_gelu():
    np.random.seed(0)
    input_x = np.random.normal(0, 1, [2, 3, 4, 3]).astype(np.float32)
    expect = FastGeLU(input_x, False)
    result = FastGeLU(input_x, True)
    res = np.allclose(expect.asnumpy(), result.asnumpy(), rtol=1.e-4, atol=1.e-4, equal_nan=True)
    assert res


def gelu_grad(input_dy, input_x, input_y, enable_graph_kernel=False):
    context.set_context(enable_graph_kernel=enable_graph_kernel)
    net = GeluGradNet()
    result = net(Tensor(input_dy), Tensor(input_x), Tensor(input_y))
    return result


def run_gelu_grad():
    np.random.seed(0)
    input_dy = np.random.normal(0, 1, [2, 3, 4, 3]).astype(np.float32)
    input_x = np.random.normal(0, 1, [2, 3, 4, 3]).astype(np.float32)
    input_y = cal_gelu(input_x)

    expect = gelu_grad(input_dy, input_x, input_y, False)
    result = gelu_grad(input_dy, input_x, input_y, True)
    res = np.allclose(expect.asnumpy(), result.asnumpy(), rtol=1.e-4, atol=1.e-4, equal_nan=True)
    assert res


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_gelu_ascend():
    """
    Feature: test graph kernel FastGeLU
    Description: run test case on Ascend
    Expectation: the result match with expect
    """
    context.set_context(jit_level='O0')
    context.set_context(mode=context.GRAPH_MODE)
    run_gelu()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_gelu_grad_ascend():
    """
    Feature: test graph kernel FastGeLUGrad
    Description: run test case on Ascend
    Expectation: the result match with expect
    """
    context.set_context(jit_level='O0')
    context.set_context(mode=context.GRAPH_MODE)
    run_gelu_grad()
