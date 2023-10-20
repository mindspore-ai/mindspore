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
from mindspore.ops import functional as F
from mindspore.common.api import jit

context.set_context(mode=context.PYNATIVE_MODE, device_target='CPU')


class NetP(nn.Cell):
    def __init__(self, output_size):
        super(NetP, self).__init__()
        self.adaptive_max_pool2d = nn.AdaptiveMaxPool2d(output_size)

    @jit
    def construct(self, x):
        return self.adaptive_max_pool2d(x)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_adaptivemaxpool2d_cpu_net_normal():
    """
    Feature: test adaptivemaxpool2d op in cpu.
    Description: test the ops.
    Expectation: expect correct shape result.
    """
    x = np.random.randn(1, 32, 9, 9)
    net = NetP((3, 5))
    output = net(Tensor(x, mindspore.float32))
    expect_shape = (1, 32, 3, 5)
    assert output.asnumpy().shape == expect_shape


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_adaptivemaxpool2d_cpu_net_h_none():
    """
    Feature: test adaptivemaxpool2d op in cpu.
    Description: test the none value of output_size attr.
    Expectation: expect correct shape result.
    """
    x = np.random.randn(1, 32, 9, 9)
    net = NetP((None, 5))
    output = net(Tensor(x, mindspore.float32))
    expect_shape = (1, 32, 9, 5)
    assert output.asnumpy().shape == expect_shape


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_adaptivemaxpool2d_cpu_net_hxh():
    """
    Feature: test adaptivemaxpool2d op in cpu.
    Description: test the int type of output_size attr.
    Expectation: expect correct shape result.
    """
    x = np.random.randn(1, 32, 9, 9)
    net = NetP((5))
    output = net(Tensor(x, mindspore.float32))
    expect_shape = (1, 32, 5, 5)
    assert output.asnumpy().shape == expect_shape


class NetWithIndices(nn.Cell):
    def __init__(self, output_size):
        super(NetWithIndices, self).__init__()
        self.adaptive_max_pool2d = nn.AdaptiveMaxPool2d(output_size, True)

    @jit
    def construct(self, x):
        return self.adaptive_max_pool2d(x)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_adaptivemaxpool2d_cpu_net_with_indices():
    """
    Feature: test adaptivemaxpool2d op in cpu.
    Description: test the return_indices attr.
    Expectation: expect correct shape result.
    """
    x = np.random.randn(1, 32, 9, 9)
    net = NetWithIndices((3, 5))
    output = net(Tensor(x, mindspore.float32))
    expect_shape = (1, 32, 3, 5)
    assert output[1].asnumpy().shape == expect_shape


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_adaptivemaxpool2d_cpu_net_f():
    """
    Feature: test adaptivemaxpool2d op in cpu.
    Description: test the ops in functional.
    Expectation: expect correct shape result.
    """
    x = np.random.randn(1, 32, 9, 9)
    output = F.adaptive_max_pool2d(Tensor(x, mindspore.float32), (3, 5), True)
    expect_shape = (1, 32, 3, 5)
    assert output[1].asnumpy().shape == expect_shape


class Netnn(nn.Cell):
    def __init__(self):
        super(Netnn, self).__init__()
        self.adaptive_max_pool2d = nn.AdaptiveMaxPool2d((3, 5))

    @jit
    def construct(self, x):
        return self.adaptive_max_pool2d(x)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_adaptivemaxpool2d_cpu_net_nn():
    """
    Feature: test adaptivemaxpool2d op in cpu.
    Description: test the ops in nn.
    Expectation: expect correct shape result.
    """
    x = np.random.randn(1, 32, 9, 9)
    net = Netnn()
    output = net(Tensor(x, mindspore.float32))
    expect_shape = (1, 32, 3, 5)
    assert output.asnumpy().shape == expect_shape


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_adaptivemaxpool2d_cpu_dynamic_shape():
    """
    Feature: test adaptivemaxpool2d op in cpu.
    Description: test the ops in dynamic shape.
    Expectation: expect correct shape result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    net = Netnn()
    x_dyn = Tensor(shape=[1, 32, 9, None], dtype=mindspore.float32)
    net.set_inputs(x_dyn)
    x = np.random.randn(1, 32, 9, 9)
    output = net(Tensor(x, mindspore.float32))
    expect_shape = (1, 32, 3, 5)
    assert output.asnumpy().shape == expect_shape


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_adaptivemaxpool2d_cpu_vmap():
    """
    Feature: test adaptivemaxpool2d op in cpu.
    Description: test the ops vmap_rule.
    Expectation: expect correct shape result.
    """
    input_x = Tensor(np.arange(2 * 2 * 4 * 4).reshape(2, 2, 4, 4), mindspore.float32)
    adaptivemaxpool2d = nn.AdaptiveMaxPool2d((2, None))
    output = adaptivemaxpool2d(input_x)
    expect_output = Tensor([[[[4, 5, 6, 7],
                              [12, 13, 14, 15]],
                             [[20, 21, 22, 23],
                              [28, 29, 30, 31]]],
                            [[[36, 37, 38, 39],
                              [44, 45, 46, 47]],
                             [[52, 53, 54, 55],
                              [60, 61, 62, 63]]]], mindspore.float32)
    assert np.allclose(output.asnumpy(), expect_output.asnumpy(), 0.0001, 0.0001)
