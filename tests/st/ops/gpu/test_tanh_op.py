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
from tests.mark_utils import arg_mark

import numpy as np
import pytest

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from mindspore.ops import function as F


class TanhNet(nn.Cell):
    def __init__(self):
        super(TanhNet, self).__init__()
        self.tanh = P.Tanh()

    def construct(self, x):
        return self.tanh(x)


class Grad(nn.Cell):
    def __init__(self, network):
        super(Grad, self).__init__()
        self.grad = C.GradOperation(get_all=True, sens_param=True)
        self.network = network

    def construct(self, input_data, sens):
        gout = self.grad(self.network)(input_data, sens)
        return gout


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize("data_type", [np.float32, np.float64, np.complex64, np.complex128])
def test_tanh(data_type):
    """
    Feature: Tanh
    Description: test cases for Tanh
    Expectation: the result match to numpy
    """
    x_np = np.array(
        [[0.28522366, 0.38033979, 1.54657853, -0.98530175, -0.54365635, 0.12652203, -1.33449938, -0.27737698],
         [2.06282293, 0.84635078, 0.16628414, -0.91823183, -0.72023044, -0.09147043, -0.04166984, -1.5664763],
         [-0.17157249, 0.44260951, -0.6683391, 1.13142613, 1.5536937, -0.32799768, -0.20016545, 0.06773927]],
        dtype=data_type)
    dy_np = np.array(
        [[0.44969849, -0.187879, -0.64300827, 1.36638774, 0.89930276, -0.23835229, -0.67771854, -1.88984999],
         [2.00418801, 2.33336475, 0.00241747, 1.31558685, 0.06768817, -2.23008804, -0.26818366, -1.26873401],
         [1.83694105, 0.5339005, 0.51117424, 0.49202378, -0.83297819, -0.71001219, 0.18913512, 0.65580389]],
        dtype=data_type)

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    x_ms = Tensor(x_np)
    dy_ms = Tensor(dy_np)

    net = TanhNet()
    grad = Grad(net)
    output = grad(x_ms, dy_ms)

    expect = [[0.41501077, -0.16312202, -0.10675912, 0.58678646, 0.67828224, -0.23457714, -0.1643468, -1.75159405],
              [0.12541081, 1.2251587, 0.00235184, 0.62396731, 0.04191568, -2.21153283, -0.26771853, -0.20311764],
              [1.78391056, 0.44159236, 0.33690308, 0.16800483, -0.13651318, -0.63878956, 0.18175511, 0.65280384]]

    assert np.allclose(output[0].asnumpy(), expect)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_tanh_fp16():
    """
    Feature: Tanh
    Description: test cases for Tanh
    Expectation: the result match to numpy
    """
    np.random.seed(42)
    x_np = np.random.randn(5, 3, 6).astype(np.float16)
    dy_np = np.random.randn(5, 3, 6).astype(np.float16)

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    x_ms = Tensor(x_np)
    dy_ms = Tensor(dy_np)

    net = TanhNet()
    grad = Grad(net)
    output = grad(x_ms, dy_ms)

    expect = [[[0.0766, 0.95, -0.474, -0.0568, -0.3713, -1.387],
               [0.04626, 0.1521, 0.004135, -0.1771, -1.149, -0.341],
               [-0.3235, -0.0666, -0.01921, 0.299, 0.7764, 0.1583]],

              [[0.124, -0.0157, -0.3682, -0.0252, 0.05997, 0.51],
               [-0.145, 0.2979, -0.01145, -1.019, 0.8125, 0.6914],
               [0.562, -0.0848, 1.402, -0.5386, 0.318, 0.645]],

              [[-0.9487, -0.04343, 0.02448, -0.4844, -0.939, 0.0666],
               [-1.049, 0.433, -0.1724, 0.9604, -0.6377, -0.1241],
               [0.7246, -0.1364, 0.2051, 1.132, -1.049, 0.1298]],

              [[0.104, 0.3643, -0.6562, -1.202, 0.4688, 0.1294],
               [0.2008, 0.3347, -0.2418, 0.07135, 0.1611, -0.1667],
               [1.856, 0.1979, -1.048, 0.4443, -0.8574, 0.1329]],

              [[1.156, -0.1322, 0.02069, 0.2241, 0.8164, 1.736],
               [-0.2433, -0.05484, -0.848, -0.7197, -0.01453, 0.2637],
               [0.1528, 0.6494, 0.006195, 1.307, -0.2024, 2.113]]]

    assert np.allclose(output[0].asnumpy(), expect, rtol=0.5 * 1e-2, atol=0.5 * 1e-2)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize("data_type", [np.float16, np.float32, np.float64, np.complex64, np.complex128])
def test_func(data_type):
    """
    Feature: Tanh
    Description: test cases for Tanh
    Expectation: the result match to numpy
    """
    x = np.random.randn(2, 3, 3, 4).astype(data_type)
    y_expect = np.tanh(x)

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    tensor = Tensor(x)
    out = F.tanh(tensor)

    assert out.shape == y_expect.shape
    np.allclose(out.asnumpy(), y_expect)

    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    tensor = Tensor(x)
    out = F.tanh(tensor)

    assert out.shape == y_expect.shape
    np.allclose(out.asnumpy(), y_expect)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize("data_type", [np.float16, np.float32, np.float64, np.complex64, np.complex128])
def test_tensor(data_type):
    """
    Feature: Tanh
    Description: test cases for Tanh
    Expectation: the result match to numpy
    """
    x = np.random.randn(2, 3, 3, 4).astype(data_type)
    y_expect = np.tanh(x)

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

    tensor = Tensor(x)
    out = tensor.tanh()

    assert out.shape == y_expect.shape
    np.allclose(out.asnumpy(), y_expect)

    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")

    tensor = Tensor(x)
    out = tensor.tanh()

    assert out.shape == y_expect.shape
    np.allclose(out.asnumpy(), y_expect)
