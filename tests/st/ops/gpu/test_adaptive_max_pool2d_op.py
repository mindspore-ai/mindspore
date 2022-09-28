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
from mindspore.common.api import ms_function

context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')


class NetP(nn.Cell):
    def __init__(self, output_size):
        super(NetP, self).__init__()
        self.adaptive_max_pool2d = nn.AdaptiveMaxPool2d(output_size)

    @ms_function
    def construct(self, x):
        return self.adaptive_max_pool2d(x)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_net_normal():
    """
    Feature: test adaptivemaxpool2d op.
    Description: test the ops.
    Expectation: expect correct shape result.
    """
    x = np.random.randn(1, 32, 9, 9)
    net = NetP((3, 5))
    output = net(Tensor(x, mindspore.float32))
    expect_shape = (1, 32, 3, 5)
    assert output.asnumpy().shape == expect_shape


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_net_h_none():
    """
    Feature: test adaptivemaxpool2d op.
    Description: test the none value of output_size attr.
    Expectation: expect correct shape result.
    """
    x = np.random.randn(1, 32, 9, 9)
    net = NetP((None, 5))
    output = net(Tensor(x, mindspore.float32))
    expect_shape = (1, 32, 9, 5)
    assert output.asnumpy().shape == expect_shape


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_net_hxh():
    """
    Feature: test adaptivemaxpool2d op.
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

    @ms_function
    def construct(self, x):
        return self.adaptive_max_pool2d(x)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_net_with_indices():
    """
    Feature: test adaptivemaxpool2d op.
    Description: test the return_indices attr.
    Expectation: expect correct shape result.
    """
    x = np.random.randn(1, 32, 9, 9)
    net = NetWithIndices((3, 5))
    output = net(Tensor(x, mindspore.float32))
    expect_shape = (1, 32, 3, 5)
    assert output[1].asnumpy().shape == expect_shape


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_net_f():
    """
    Feature: test adaptivemaxpool2d op.
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

    @ms_function
    def construct(self, x):
        return self.adaptive_max_pool2d(x)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_net_nn():
    """
    Feature: test adaptivemaxpool2d op.
    Description: test the ops in nn.
    Expectation: expect correct shape result.
    """
    x = np.random.randn(1, 32, 9, 9)
    net = Netnn()
    output = net(Tensor(x, mindspore.float32))
    expect_shape = (1, 32, 3, 5)
    assert output.asnumpy().shape == expect_shape


def test_tensor_interface_pynative():
    """
    Feature: test adaptivemaxpool2d op.
    Description: test the ops in tensor interface in pynative mode.
    Expectation: expect correct shape result.
    """
    x = Tensor(np.random.randn(1, 32, 9, 9), mindspore.float32)
    y = x.adaptive_max_pool2d((3, 5), True)
    expect_shape = (1, 32, 3, 5)
    assert y[1].asnumpy().shape == expect_shape


def test_tensor_interface_graph():
    """
    Feature: test adaptivemaxpool2d op.
    Description: test the ops in tensor interface in graph mode.
    Expectation: expect correct shape result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    x = Tensor(np.random.randn(1, 32, 9, 9), mindspore.float32)
    y = x.adaptive_max_pool2d((3, 5))
    expect_shape = (1, 32, 3, 5)
    assert y.asnumpy().shape == expect_shape


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_dynamic_shape():
    """
    Feature: test adaptivemaxpool2d op.
    Description: test the ops in dynamic shape.
    Expectation: expect correct shape result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    net = Netnn()
    x_dyn = Tensor(shape=[1, 32, 9, None], dtype=mindspore.float32)
    net.set_inputs(x_dyn)
    x = np.random.randn(1, 32, 9, 9)
    output = net(Tensor(x, mindspore.float32))
    expect_shape = (1, 32, 3, 5)
    assert output.asnumpy().shape == expect_shape
