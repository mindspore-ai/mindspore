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

import mindspore as ms
from mindspore import context, Tensor, Parameter
from mindspore.common.api import _cell_graph_executor
from mindspore.nn import Cell, TrainOneStepCell, Momentum
from mindspore.ops import operations as P
from mindspore.ops.auto_generate import SliceExt


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")


class Net(Cell):
    def __init__(self, weight, w2, begin, end, strategy1=None, strategy2=None, is_parameter=True):
        super().__init__()
        self.mul = P.Mul().shard(strategy1)
        self.slice = P.Slice().shard(strategy2)
        if is_parameter:
            self.weight = Parameter(weight, "w1")
        else:
            self.weight = weight
        self.mul2 = P.Mul()
        self.weight2 = Parameter(w2, "w2")
        self.begin = begin
        self.end = end

    def construct(self, x, b):
        out = self.slice(self.weight, self.begin, self.end)
        out = self.mul(x, out)
        out = self.mul2(out, self.weight2)
        return out


class Net2(Cell):
    def __init__(self, weight2, begin, end, strategy1=None, strategy2=None):
        super().__init__()
        self.mul = P.Mul().shard(strategy1)
        self.slice = P.Slice().shard(strategy2)
        self.weight2 = Parameter(weight2, "w2")
        self.begin = begin
        self.end = end

    def construct(self, x, b):
        out = self.mul(x, self.weight2)
        out = self.slice(out, self.begin, self.end)
        return out


class NetSliceExt(Cell):
    def __init__(self, weight2, dim, start, end, step, strategy1=None, strategy2=None):
        super().__init__()
        self.mul = P.Mul().shard(strategy1)
        self.slice = SliceExt().shard(strategy2)
        self.weight2 = Parameter(weight2, name="w2")
        self.dim = dim
        self.start = start
        self.end = end
        self.step = step

    def construct(self, x, b):
        out = self.mul(x, self.weight2)
        out = self.slice(out, self.dim, self.start, self.end, self.step)
        return out


_x = Tensor(np.ones([128, 64, 1]), dtype=ms.float32)
_w1 = Tensor(np.ones([256, 64, 32]), dtype=ms.float32)
_w2 = Tensor(np.ones([128, 64, 1]), dtype=ms.float32)
_b = Tensor(np.ones([128, 64, 32]), dtype=ms.float32)


def compile_net(net):
    optimizer = Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
    train_net = TrainOneStepCell(net, optimizer)
    train_net.set_train()
    _cell_graph_executor.compile(train_net, _x, _b)
    context.reset_auto_parallel_context()


def test_slice_no_fully_fetch_split_error():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((2, 2, 2), (2, 2, 2))
    strategy2 = ((2, 2, 2),)
    net = Net(_w1, _w2, (0, 0, 0), (128, 64, 32), strategy1, strategy2, is_parameter=True)
    with pytest.raises(RuntimeError):
        compile_net(net)


def test_slice_parameter():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((1, 4, 1), (1, 4, 2))
    strategy2 = ((1, 4, 2),)
    net = Net(_w1, _w2, (0, 0, 0), (128, 64, 32), strategy1, strategy2)
    compile_net(net)


def test_slice_tensor():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((1, 4, 1), (1, 4, 2))
    strategy2 = ((1, 4, 2),)
    net = Net(_w1, _w2, (0, 0, 0), (128, 64, 32), strategy1, strategy2, is_parameter=False)
    compile_net(net)


def test_slice_parameter_no_full_split():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((1, 4, 1), (1, 4, 2))
    strategy2 = ((1, 2, 2),)
    net = Net(_w1, _w2, (0, 0, 0), (128, 64, 32), strategy1, strategy2, is_parameter=True)
    compile_net(net)


def test_slice_output():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((1, 8, 1), (1, 8, 1))
    strategy2 = ((1, 8, 1),)
    net = Net2(_w2, (0, 0, 0), (64, 64, 1), strategy1, strategy2)
    compile_net(net)


def test_stridedslice_output_no_full_split():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((1, 8, 1), (1, 8, 1))
    strategy2 = ((1, 4, 1),)
    net = Net2(_w2, (0, 0, 0), (64, 64, 1), strategy1, strategy2)
    compile_net(net)


def test_stridedslice_no_strategy():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((1, 8, 1), (1, 8, 1))
    strategy2 = None
    net = Net2(_w2, (0, 0, 0), (128, 64, 1), strategy1, strategy2)
    compile_net(net)


def test_slice_ext_no_fully_fetch_split_error():
    """
    Feature: distribute operator SliceExt in auto parallel.
    Description: matmul-sliceext net with strategy in semi auto parallel.
    Expectation: the dim split can not be slice.
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((1, 8, 1), (1, 8, 1))
    strategy2 = ((1, 8, 1),)
    net = NetSliceExt(_w2, 1, 0, 8, 1, strategy1, strategy2)
    with pytest.raises(RuntimeError):
        compile_net(net)


def test_slice_ext():
    """
    Feature: distribute operator SliceExt in auto parallel.
    Description: matmul-sliceext net with strategy in semi auto parallel.
    Expectation: compile done without error.
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((1, 8, 1), (1, 8, 1))
    strategy2 = ((1, 8, 1),)
    net = NetSliceExt(_w2, 0, 64, 128, 1, strategy1, strategy2)
    compile_net(net)
