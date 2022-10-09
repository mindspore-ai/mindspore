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
""" test Activations """
import numpy as np
import pytest

import mindspore.nn as nn
from mindspore import Tensor
from mindspore.common.api import _cell_graph_executor
from ..ut_filter import non_graph_engine


class SoftmaxNet(nn.Cell):
    def __init__(self, dim):
        super(SoftmaxNet, self).__init__()
        self.softmax = nn.Softmax(dim)

    def construct(self, x):
        return self.softmax(x)


@non_graph_engine
def test_compile():
    net = SoftmaxNet(0)
    input_tensor = Tensor(np.array([[1.2, 2.1], [2.2, 3.2]], dtype=np.float32))
    net(input_tensor)


@non_graph_engine
def test_compile_axis():
    net = SoftmaxNet(-1)
    prob = 355
    input_data = np.random.randn(4, 16, 1, 1).astype(np.float32) * prob
    input_tensor = Tensor(input_data)
    net(input_tensor)


class Softmax2dNet(nn.Cell):
    def __init__(self):
        super(Softmax2dNet, self).__init__()
        self.softmax2d = nn.Softmax2d()

    def construct(self, x):
        return self.softmax2d(x)


def test_compile_softmax2d():
    """
    Feature: Test Softmax2d.
    Description: Test Softma2d functional.
    Expectation: Success.
    """
    net = Softmax2dNet()
    input_tensor = Tensor(np.random.randn(2, 3, 4, 4).astype(np.float32))
    net(input_tensor)
    _cell_graph_executor.compile(net, input_tensor)


class SoftminNet(nn.Cell):
    """Softmin."""
    def __init__(self, dim):
        super(SoftminNet, self).__init__()
        self.softmin = nn.Softmin(dim)

    def construct(self, x):
        return self.softmin(x)


@non_graph_engine
def test_compile_softmin():
    """
    Feature: Test Softmin.
    Description: Test Softmin functional.
    Expectation: Success.
    """
    net = SoftminNet(0)
    input_tensor = Tensor(np.array([[1.2, 2.1], [2.2, 3.2]], dtype=np.float32))
    net(input_tensor)


@non_graph_engine
def test_compile_axis_softmin():
    """
    Feature: Test Softmin.
    Description: Test Softmin functional.
    Expectation: Success.
    """
    net = SoftminNet(-1)
    prob = 355
    input_data = np.random.randn(4, 16, 1, 1).astype(np.float32) * prob
    input_tensor = Tensor(input_data)
    net(input_tensor)


class LogSoftmaxNet(nn.Cell):
    def __init__(self, dim):
        super(LogSoftmaxNet, self).__init__()
        self.logsoftmax = nn.LogSoftmax(dim)

    def construct(self, x):
        return self.logsoftmax(x)


@non_graph_engine
def test_compile_logsoftmax():
    net = LogSoftmaxNet(0)
    input_tensor = Tensor(np.array([[1.2, 2.1], [2.2, 3.2]], dtype=np.float32))
    net(input_tensor)


class Net1(nn.Cell):
    def __init__(self):
        super(Net1, self).__init__()
        self.relu = nn.ReLU()

    def construct(self, x):
        return self.relu(x)


def test_compile_relu():
    net = Net1()
    input_data = Tensor(np.array([[1.2, 2.1], [2.2, 3.2]], dtype=np.float32))
    _cell_graph_executor.compile(net, input_data)


class NetSiLU(nn.Cell):
    """SiLU."""
    def __init__(self):
        super(NetSiLU, self).__init__()
        self.silu = nn.SiLU()

    def construct(self, x):
        return self.silu(x)


def test_compile_silu():
    """
    Feature: Test SiLU.
    Description: Test SiLU functional.
    Expectation: Success.
    """
    net = NetSiLU()
    input_data = Tensor(np.array([[1.2, 2.1], [2.2, 3.2]], dtype=np.float32))
    _cell_graph_executor.compile(net, input_data)


class NetRReLU(nn.Cell):
    """RReLU"""
    def __init__(self, lower, upper):
        super(NetRReLU, self).__init__()
        self.rrelu = nn.RReLU(lower, upper)

    def construct(self, x):
        return self.rrelu(x)


def test_compile_rrelu():
    """
    Feature: Test RReLU.
    Description: Test the functionality of RReLU.
    Expectation: Success.
    """
    net = NetRReLU(0.1, 0.5)
    input_data = Tensor(np.array([[-1.2, 2.1], [-2.2, 3.2]], dtype=np.float32))
    _cell_graph_executor.compile(net, input_data)


def test_invalid_inputs_rrelu():
    """
    Feature: Test RReLU.
    Description: Test the functionality of RReLU.
    Expectation: Success.
    """
    # case 1: lower/upper is not int or float
    with pytest.raises(TypeError):
        NetRReLU('-1', '-1')

    # case 2: lower > upper
    with pytest.raises(ValueError):
        NetRReLU(lower=0.9, upper=0.1)


class Net_gelu(nn.Cell):
    def __init__(self):
        super(Net_gelu, self).__init__()
        self.gelu = nn.GELU()

    def construct(self, x):
        return self.gelu(x)


def test_compile_gelu():
    net = Net_gelu()
    input_data = Tensor(np.array([[1.2, 2.1], [2.2, 3.2]], dtype=np.float32))
    _cell_graph_executor.compile(net, input_data)


class NetLeakyReLU(nn.Cell):
    def __init__(self, alpha):
        super(NetLeakyReLU, self).__init__()
        self.leaky_relu = nn.LeakyReLU(alpha)

    def construct(self, x):
        return self.leaky_relu(x)


def test_compile_leaky_relu():
    net = NetLeakyReLU(alpha=0.1)
    input_data = Tensor(np.array([[1.6, 0, 0.6], [6, 0, -6]], dtype=np.float32))
    _cell_graph_executor.compile(net, input_data)


class NetSoftsign(nn.Cell):
    def __init__(self):
        super(NetSoftsign, self).__init__()
        self.softsign = nn.Softsign()

    def construct(self, x):
        return self.softsign(x)


def test_compile_softsign():
    """
    Feature: ALL To ALL
    Description: test cases for Softsign
    Expectation: no exception
    """
    net = NetSoftsign()
    x = np.array([0, -1, 2, 30, -30], dtype=np.float32)
    _cell_graph_executor.compile(net, Tensor(x))


class Hardtanh(nn.Cell):
    """Hardtanh."""

    def __init__(self, min_val, max_val):
        super(Hardtanh, self).__init__()
        self.hard_tanh = nn.Hardtanh(min_val, max_val)

    def construct(self, x):
        return self.hard_tanh(x)


def test_hard_tanh():
    """
    Feature: Test Hardtanh.
    Description: Test Hardtanh functional.
    Expectation: Success.
    """
    net = Hardtanh(-1.0, 1.0)
    input_data = Tensor(np.array([[1.6, 0, 0.6], [6, 0, -6]], dtype=np.float32))
    _cell_graph_executor.compile(net, input_data)


class NetThreshold(nn.Cell):
    """Threshold."""
    def __init__(self, threshold, value):
        super(NetThreshold, self).__init__()
        self.threshold = nn.Threshold(threshold, value)

    def construct(self, x):
        return self.threshold(x)


def test_compile_threshold():
    """
    Feature: Test Threshold.
    Description: Test Threshold functional.
    Expectation: Success.
    """
    net = NetThreshold(threshold=0.1, value=1.0)
    input_data = Tensor(np.array([[0.1, 0.2, 0.3], [0.0, 0.1, 0.2]], dtype=np.float32))
    _cell_graph_executor.compile(net, input_data)


class NetTanhshrink(nn.Cell):
    """Tanhshrink"""

    def __init__(self):
        super(NetTanhshrink, self).__init__()
        self.tanhshrink = nn.Tanhshrink()

    def construct(self, x):
        return self.tanhshrink(x)


def test_compile_tanhshrink():
    """
    Feature: Test Tanhshrink
    Description: Test the functionality of tanhshrink
    Expectation: success
    """
    net = NetTanhshrink()
    input_data = Tensor(np.array([1, 2, 3, 2, 1]).astype(np.float16))
    _cell_graph_executor.compile(net, input_data)
