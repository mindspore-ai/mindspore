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


import pytest
import numpy as np

from mindspore import Tensor, context
from mindspore.nn import Cell
from mindspore.ops import composite as C
from mindspore.ops import auto_generate as P
from mindspore.common.api import _cell_graph_executor
from tests.ut.python.ops.test_math_ops import VirtualLoss


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")


x_real = np.arange(4 * 4 * 4).reshape(4, 4, 4).astype(np.float32)
x_imag = np.arange(4 * 4 * 4).reshape(4, 4, 4).astype(np.float32)
x_ = Tensor(x_real + 1j*x_imag)

grad_all = C.GradOperation(get_all=True)


class NetWithLoss(Cell):
    def __init__(self, network):
        super(NetWithLoss, self).__init__()
        self.loss = VirtualLoss()
        self.network = network

    def construct(self, x):
        predict = self.network(x)
        return self.loss(predict)


class GradWrap(Cell):
    def __init__(self, network):
        super(GradWrap, self).__init__()
        self.network = network

    def construct(self, x):
        return grad_all(self.network)(x)


class Net(Cell):
    def __init__(self, func, axes, strategy=None):
        super(Net, self).__init__()
        self.func = func.shard(strategy)
        self.axes = axes

    def construct(self, x):
        return self.func(x, self.axes)


def test_fftshift_auto_parallel():
    """
    Feature: test FFTShift auto parallel
    Description: auto parallel
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="auto_parallel", search_mode="dynamic_programming", device_num=4,
                                      global_rank=0)
    net = GradWrap(NetWithLoss(Net(P.FFTShift(), axes=(2,))))
    net.set_train()
    _cell_graph_executor.compile(net, x_)


def test_fftshift_model_parallel():
    """
    Feature: test FFTShift model parallel
    Description: model parallel
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=4, global_rank=0)
    net = GradWrap(NetWithLoss(Net(P.FFTShift(), axes=(2,), strategy=((2, 2, 1),))))
    net.set_train()
    _cell_graph_executor.compile(net, x_)


def test_fftshift_strategy_error():
    """
    Feature: test invalid strategy for FFTShift
    Description: illegal strategy
    Expectation: raise RuntimeError
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=4, global_rank=0)
    net = GradWrap(NetWithLoss(Net(P.FFTShift(), axes=(1, 2), strategy=((2, 2, 1),))))
    net.set_train()
    with pytest.raises(RuntimeError):
        _cell_graph_executor.compile(net, x_)


def test_ifftshift_auto_parallel():
    """
    Feature: test FFTShift auto parallel
    Description: auto parallel
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="auto_parallel", search_mode="dynamic_programming", device_num=4,
                                      global_rank=0)
    net = GradWrap(NetWithLoss(Net(P.IFFTShift(), axes=(2,))))
    net.set_train()
    _cell_graph_executor.compile(net, x_)


def test_ifftshift_model_parallel():
    """
    Feature: test FFTShift model parallel
    Description: model parallel
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=4, global_rank=0)
    net = GradWrap(NetWithLoss(Net(P.IFFTShift(), axes=(2,), strategy=((2, 2, 1),))))
    net.set_train()
    _cell_graph_executor.compile(net, x_)


def test_ifftshift_strategy_error():
    """
    Feature: test invalid strategy for FFTShift
    Description: illegal strategy
    Expectation: raise RuntimeError
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=4, global_rank=0)
    net = GradWrap(NetWithLoss(Net(P.IFFTShift(), axes=(1, 2), strategy=((2, 2, 1),))))
    net.set_train()
    with pytest.raises(RuntimeError):
        _cell_graph_executor.compile(net, x_)
