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

import numpy as np
import pytest

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor, jit, ops
from mindspore.ops import operations as P


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.ops = P.Pow()

    def construct(self, x, y):
        return self.ops(x, y)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_net():
    """
    Feature: ALL To ALL
    Description: test cases for Pow
    Expectation: the result match to numpy
    """
    x0_np = np.random.randint(1, 5, (2, 3, 4, 4)).astype(np.float32)
    y0_np = np.random.randint(1, 5, (2, 3, 4, 4)).astype(np.float32)
    x1_np = np.random.randint(1, 5, (2, 3, 4, 4)).astype(np.float32)
    y1_np = np.array(3).astype(np.float32)
    x2_np = np.random.randint(1, 5, (2, 3, 4, 4)).astype(np.float64)
    y2_np = np.random.randint(1, 5, (2, 3, 4, 4)).astype(np.float64)
    x3_np = np.random.randint(1, 5, (2, 3, 4, 4)).astype(np.float64)
    y3_np = np.array(3).astype(np.float64)

    x0 = Tensor(x0_np)
    y0 = Tensor(y0_np)
    x1 = Tensor(x1_np)
    y1 = Tensor(y1_np)
    x2 = Tensor(x2_np)
    y2 = Tensor(y2_np)
    x3 = Tensor(x3_np)
    y3 = Tensor(y3_np)

    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    net = Net()
    out = net(x0, y0).asnumpy()
    expect = np.power(x0_np, y0_np)
    assert np.all(out == expect)
    assert out.shape == expect.shape

    out = net(x1, y1).asnumpy()
    expect = np.power(x1_np, y1_np)
    assert np.all(out == expect)
    assert out.shape == expect.shape

    out = net(x2, y2).asnumpy()
    expect = np.power(x2_np, y2_np)
    assert np.all(out == expect)
    assert out.shape == expect.shape

    out = net(x3, y3).asnumpy()
    expect = np.power(x3_np, y3_np)
    assert np.all(out == expect)
    assert out.shape == expect.shape


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_tensor_pow_pynative():
    """
    Feature: Tensor interface pow.
    Description: test tensor interface pow in pynative mode.
    Expectation: Success.
    """
    x = Tensor([1, 2, 3])
    y = Tensor([1, 2, 3])
    output = x.pow(y)
    assert np.all(output.asnumpy() == np.array([1, 4, 27]))


@jit
def tensor_pow_func(x, y):
    return x.pow(y)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_tensor_pow_graph():
    """
    Feature: Tensor interface pow.
    Description: test tensor interface pow in graph mode.
    Expectation: Success.
    """
    x = Tensor([1, 2, 3])
    y = Tensor([1, 2, 3])
    output = tensor_pow_func(x, y)
    assert np.all(output.asnumpy() == np.array([1, 4, 27]))


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_pow_functional():
    """
    Feature: Tensor interface pow.
    Description: test functional interface pow.
    Expectation: Success.
    """
    x = Tensor([1, 2, 3])
    y = Tensor([1, 2, 3])
    output = ops.pow(x, y)
    assert np.all(output.asnumpy() == np.array([1, 4, 27]))


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_pow_vmap():
    """
    Feature: Tensor interface pow.
    Description: test pow operation with vmap.
    Expectation: Success.
    """
    x = Tensor([[1, 2, 3], [3, 2, 1]])
    y = Tensor([[1, 2, 3], [3, 2, 1]])
    pow_vmap = ops.vmap(tensor_pow_func, (0, 0), 0)
    output = pow_vmap(x, y)
    assert np.all(output.asnumpy() == np.array([[1, 4, 27], [27, 4, 1]]))
