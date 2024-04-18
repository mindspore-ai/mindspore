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


x_real = np.arange(4 * 4 * 4 * 4).reshape(4, 4, 4, 4).astype(np.float32)
x_imag = np.arange(4 * 4 * 4 * 4).reshape(4, 4, 4, 4).astype(np.float32)
x_ = Tensor(x_real + 1j*x_imag)
r = np.arange(4 * 4 * 4).reshape(4, 4, 4).astype(np.float32)
r_ = Tensor(r)

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
    def __init__(self, func, n, dim, norm, strategy=None):
        super(Net, self).__init__()
        self.func = func.shard(strategy)
        self.n = n
        self.dim = dim
        self.norm = norm

    def construct(self, x):
        return self.func(x, self.n, self.dim, self.norm)


def test_fft_auto_parallel():
    """
    Feature: test FFT auto parallel
    Description: auto parallel
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="auto_parallel", search_mode="dynamic_programming", device_num=4,
                                      global_rank=0)
    net = GradWrap(NetWithLoss(Net(P.FFT(), n=4, dim=0, norm="backward")))
    net.set_train()
    _cell_graph_executor.compile(net, x_)


def test_fft_model_parallel():
    """
    Feature: test FFT model parallel
    Description: model parallel
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=4, global_rank=0)
    net = GradWrap(NetWithLoss(Net(P.FFT(), n=4, dim=0, norm="backward", strategy=((1, 1, 2, 2),))))
    net.set_train()
    _cell_graph_executor.compile(net, x_)


def test_fft_strategy_error():
    """
    Feature: test invalid strategy for FFT
    Description: illegal strategy
    Expectation: raise RuntimeError
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=4, global_rank=0)
    net = GradWrap(NetWithLoss(Net(P.FFT(), n=4, dim=0, norm="backward", strategy=((2, 2, 1, 1),))))
    net.set_train()
    with pytest.raises(RuntimeError):
        _cell_graph_executor.compile(net, x_)


def test_ifft_auto_parallel():
    """
    Feature: test IFFT auto parallel
    Description: auto parallel
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="auto_parallel", search_mode="dynamic_programming", device_num=4,
                                      global_rank=0)
    net = GradWrap(NetWithLoss(Net(P.IFFT(), n=4, dim=0, norm="backward")))
    net.set_train()
    _cell_graph_executor.compile(net, x_)


def test_ifft_model_parallel():
    """
    Feature: test IFFT model parallel
    Description: model parallel
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=4, global_rank=0)
    net = GradWrap(NetWithLoss(Net(P.IFFT(), n=4, dim=0, norm="backward", strategy=((1, 1, 2, 2),))))
    net.set_train()
    _cell_graph_executor.compile(net, x_)


def test_ifft_strategy_error():
    """
    Feature: test invalid strategy for IFFT
    Description: illegal strategy
    Expectation: raise RuntimeError
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=4, global_rank=0)
    net = GradWrap(NetWithLoss(Net(P.IFFT(), n=4, dim=0, norm="backward", strategy=((2, 2, 1, 1),))))
    net.set_train()
    with pytest.raises(RuntimeError):
        _cell_graph_executor.compile(net, x_)


def test_fft2_auto_parallel():
    """
    Feature: test FFT2 auto parallel
    Description: auto parallel
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="auto_parallel", search_mode="dynamic_programming", device_num=4,
                                      global_rank=0)
    net = GradWrap(NetWithLoss(Net(P.FFT2(), n=(4, 4), dim=(-2, -1), norm="backward")))
    net.set_train()
    _cell_graph_executor.compile(net, x_)


def test_fft2_model_parallel():
    """
    Feature: test FFT2 model parallel
    Description: model parallel
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=4, global_rank=0)
    net = GradWrap(NetWithLoss(Net(P.FFT2(), n=(4, 4), dim=(-2, -1), norm="backward", strategy=((2, 2, 1, 1),))))
    net.set_train()
    _cell_graph_executor.compile(net, x_)


def test_fft2_strategy_error():
    """
    Feature: test invalid strategy for FFT2
    Description: illegal strategy
    Expectation: raise RuntimeError
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=4, global_rank=0)
    net = GradWrap(NetWithLoss(Net(P.FFT2(), n=(4, 4), dim=(-2, -1), norm="backward", strategy=((1, 2, 2, 1),))))
    net.set_train()
    with pytest.raises(RuntimeError):
        _cell_graph_executor.compile(net, x_)


def test_ifft2_auto_parallel():
    """
    Feature: test IFFT2 auto parallel
    Description: auto parallel
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="auto_parallel", search_mode="dynamic_programming", device_num=4,
                                      global_rank=0)
    net = GradWrap(NetWithLoss(Net(P.IFFT2(), n=(4, 4), dim=(-2, -1), norm="backward")))
    net.set_train()
    _cell_graph_executor.compile(net, x_)


def test_ifft2_model_parallel():
    """
    Feature: test IFFT2 model parallel
    Description: model parallel
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=4, global_rank=0)
    net = GradWrap(NetWithLoss(Net(P.IFFT2(), n=(4, 4), dim=(-2, -1), norm="backward", strategy=((2, 2, 1, 1),))))
    net.set_train()
    _cell_graph_executor.compile(net, x_)


def test_ifft2_strategy_error():
    """
    Feature: test invalid strategy for IFFT2
    Description: illegal strategy
    Expectation: raise RuntimeError
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=4, global_rank=0)
    net = GradWrap(NetWithLoss(Net(P.IFFT2(), n=(4, 4), dim=(-2, -1), norm="backward", strategy=((1, 2, 2, 1),))))
    net.set_train()
    with pytest.raises(RuntimeError):
        _cell_graph_executor.compile(net, x_)


def test_fftn_auto_parallel():
    """
    Feature: test FFTN auto parallel
    Description: auto parallel
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="auto_parallel", search_mode="dynamic_programming", device_num=4,
                                      global_rank=0)
    net = GradWrap(NetWithLoss(Net(P.FFTN(), n=(4, 4, 4), dim=(-3, -2, -1), norm="backward")))
    net.set_train()
    _cell_graph_executor.compile(net, x_)


def test_fftn_model_parallel():
    """
    Feature: test FFTN model parallel
    Description: model parallel
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=4, global_rank=0)
    net = GradWrap(NetWithLoss(Net(P.FFTN(), n=(4, 4, 4), dim=(-3, -2, -1), norm="backward", strategy=((2, 1, 1, 1),))))
    net.set_train()
    _cell_graph_executor.compile(net, x_)


def test_fftn_strategy_error():
    """
    Feature: test invalid strategy for FFTShift
    Description: illegal strategy
    Expectation: raise RuntimeError
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=4, global_rank=0)
    net = GradWrap(NetWithLoss(Net(P.FFTN(), n=(4, 4, 4), dim=(-3, -2, -1), norm="backward", strategy=((2, 2, 2, 1),))))
    net.set_train()
    with pytest.raises(RuntimeError):
        _cell_graph_executor.compile(net, x_)


def test_ifftn_auto_parallel():
    """
    Feature: test IFFTN auto parallel
    Description: auto parallel
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="auto_parallel", search_mode="dynamic_programming", device_num=4,
                                      global_rank=0)
    net = GradWrap(NetWithLoss(Net(P.IFFTN(), n=(4, 4, 4), dim=(-3, -2, -1), norm="backward")))
    net.set_train()
    _cell_graph_executor.compile(net, x_)


def test_ifftn_model_parallel():
    """
    Feature: test IFFTN model parallel
    Description: model parallel
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=4, global_rank=0)
    net = GradWrap(NetWithLoss(Net(P.IFFTN(), n=(4, 4, 4), dim=(-3, -2, -1),
                                   norm="backward", strategy=((2, 1, 1, 1),))))
    net.set_train()
    _cell_graph_executor.compile(net, x_)


def test_ifftn_strategy_error():
    """
    Feature: test invalid strategy for IFFTN
    Description: illegal strategy
    Expectation: raise RuntimeError
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=4, global_rank=0)
    net = GradWrap(NetWithLoss(Net(P.IFFTN(), n=(4, 4, 4), dim=(-3, -2, -1),
                                   norm="backward", strategy=((2, 2, 2, 1),))))
    net.set_train()
    with pytest.raises(RuntimeError):
        _cell_graph_executor.compile(net, x_)


def test_rfft_auto_parallel():
    """
    Feature: test RFFT auto parallel
    Description: auto parallel
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="auto_parallel", search_mode="dynamic_programming", device_num=4,
                                      global_rank=0)
    net = GradWrap(NetWithLoss(Net(P.RFFT(), n=4, dim=0, norm="backward")))
    net.set_train()
    _cell_graph_executor.compile(net, r_)


def test_rfft_model_parallel():
    """
    Feature: test RFFT model parallel
    Description: model parallel
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=4, global_rank=0)
    net = GradWrap(NetWithLoss(Net(P.RFFT(), n=4, dim=0, norm="backward", strategy=((1, 2, 2),))))
    net.set_train()
    _cell_graph_executor.compile(net, r_)


def test_rfft_strategy_error():
    """
    Feature: test invalid strategy for RFFT
    Description: illegal strategy
    Expectation: raise RuntimeError
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=4, global_rank=0)
    net = GradWrap(NetWithLoss(Net(P.RFFT(), n=4, dim=0, norm="backward", strategy=((2, 2, 1),))))
    net.set_train()
    with pytest.raises(RuntimeError):
        _cell_graph_executor.compile(net, r_)


def test_irfft_auto_parallel():
    """
    Feature: test IRFFT auto parallel
    Description: auto parallel
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="auto_parallel", search_mode="dynamic_programming", device_num=4,
                                      global_rank=0)
    net = GradWrap(NetWithLoss(Net(P.IRFFT(), n=4, dim=0, norm="backward")))
    net.set_train()
    _cell_graph_executor.compile(net, r_)


def test_irfft_model_parallel():
    """
    Feature: test IRFFT model parallel
    Description: model parallel
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=4, global_rank=0)
    net = GradWrap(NetWithLoss(Net(P.IRFFT(), n=4, dim=0, norm="backward", strategy=((1, 2, 2),))))
    net.set_train()
    _cell_graph_executor.compile(net, r_)


def test_irfft_strategy_error():
    """
    Feature: test invalid strategy for IRFFT
    Description: illegal strategy
    Expectation: raise RuntimeError
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=4, global_rank=0)
    net = GradWrap(NetWithLoss(Net(P.IRFFT(), n=4, dim=0, norm="backward", strategy=((2, 2, 1),))))
    net.set_train()
    with pytest.raises(RuntimeError):
        _cell_graph_executor.compile(net, r_)
