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
