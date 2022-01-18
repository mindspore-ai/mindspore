# Copyright 2021 Huawei Technologies Co., Ltd
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

import numpy as np
import pytest

import mindspore as ms
from mindspore import context, Tensor, Parameter
from mindspore.common.api import _cell_graph_executor
from mindspore.nn import Cell, TrainOneStepCell, Momentum
from mindspore.ops import operations as P


class Net(Cell):
    def __init__(self, conv2d_weight, out_channel, kernel_size, pad_mode, stride,
                 strategy1=None, strategy2=None):
        super().__init__()
        self.conv2d_transpose = P.Conv2DTranspose(out_channel=out_channel, kernel_size=kernel_size,
                                                  pad_mode=pad_mode, stride=stride).shard(strategy1)
        self.neg = P.Neg().shard(strategy2)
        self.weight = Parameter(conv2d_weight, "w1")
        self.add = P.Add()
        self.add_w = Parameter(Tensor(np.ones([32, 8, 8, 8]), dtype=ms.float32), "add_w")

    def construct(self, x, b):
        out = self.add(x, self.add_w)
        out = self.conv2d_transpose(out, self.weight, (32, 16, 8, 8))
        out = self.neg(out)
        return out


class Net2(Cell):
    def __init__(self, conv2d_weight, out_channel, kernel_size, pad_mode, stride, group=1,
                 strategy1=None, strategy2=None):
        super().__init__()
        self.conv2d_transpose = P.Conv2DTranspose(out_channel=out_channel, kernel_size=kernel_size,
                                                  pad_mode=pad_mode, stride=stride, group=group).shard(strategy1)
        self.neg = P.Neg().shard(strategy2)
        self.weight = Parameter(conv2d_weight, "w1")

    def construct(self, x, b):
        out = self.conv2d_transpose(x, self.weight, (32, 16, 16, 16))
        out = self.neg(out)
        return out


_x = Tensor(np.ones([32, 8, 8, 8]), dtype=ms.float32)
_w1 = Tensor(np.ones([8, 16, 2, 2]), dtype=ms.float32)
_w2 = Tensor(np.ones([8, 16, 4, 4]), dtype=ms.float32)
_w3 = Tensor(np.ones([8, 16, 10, 10]), dtype=ms.float32)
_w4 = Tensor(np.ones([8, 16, 3, 3]), dtype=ms.float32)
_w5 = Tensor(np.ones([8, 8, 4, 4]), dtype=ms.float32)
_b = Tensor(np.ones([32, 16, 8, 8]), dtype=ms.float32)


def compile_net(net):
    optimizer = Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
    train_net = TrainOneStepCell(net, optimizer)
    train_net.set_auto_parallel()
    train_net.set_train()
    _cell_graph_executor.compile(train_net, _x, _b)
    context.reset_auto_parallel_context()


def test_conv2d_transpose_data_parallel():
    """
    Feature: test data parallel strategy
    Description: only shard batch dimension
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((8, 1, 1, 1), (1, 1, 1, 1))
    strategy2 = ((8, 1, 1, 1),)
    net = Net(_w1, out_channel=8, kernel_size=2, pad_mode="same", stride=1, strategy1=strategy1, strategy2=strategy2)
    compile_net(net)


def test_conv2d_transpose_group():
    """
    Feature: test group is not 1
    Description: shard n/h/w, and group is 2
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((2, 1, 2, 2), (1, 1, 1, 1))
    strategy2 = ((8, 1, 1, 1),)
    net = Net2(_w5, out_channel=8, kernel_size=4, pad_mode="same", stride=2, group=2, strategy1=strategy1,
               strategy2=strategy2)
    compile_net(net)


def test_conv2d_transpose_model_parallel1():
    """
    Feature: test model parallel strategy
    Description: only shard batch dimension and channel dimension
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((2, 2, 1, 1), (2, 2, 1, 1))
    strategy2 = ((8, 1, 1, 1),)
    net = Net(_w1, out_channel=8, kernel_size=2, pad_mode="same", stride=1, strategy1=strategy1, strategy2=strategy2)
    compile_net(net)


def test_conv2d_transpose_model_parallel2():
    """
    Feature: test model parallel strategy
    Description: shard batch dimension and w dimension
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((2, 1, 1, 4), (1, 1, 1, 1))
    strategy2 = ((2, 1, 1, 4),)
    net = Net2(_w2, out_channel=8, kernel_size=(4, 4), pad_mode="same", stride=2,
               strategy1=strategy1, strategy2=strategy2)
    compile_net(net)


def test_conv2d_transpose_model_parallel3():
    """
    Feature: test model parallel strategy
    Description: shard batch dimension, channel dimension and w dimension
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=16, global_rank=0)
    strategy1 = ((2, 2, 1, 4), (2, 1, 1, 1))
    strategy2 = ((2, 2, 1, 4),)
    net = Net2(_w2, out_channel=8, kernel_size=(4, 4), pad_mode="same", stride=2,
               strategy1=strategy1, strategy2=strategy2)
    compile_net(net)


def test_conv2d_transpose_model_parallel4():
    """
    Feature: test model parallel strategy
    Description: shard batch dimension, channel dimension and w dimension
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=16, global_rank=0)
    strategy1 = ((1, 1, 2, 4), (1, 1, 1, 1))
    strategy2 = ((2, 2, 1, 4),)
    net = Net2(_w2, out_channel=8, kernel_size=(4, 4), pad_mode="same", stride=2,
               strategy1=strategy1, strategy2=strategy2)
    compile_net(net)


def test_conv2d_transpose_all_rank_no_need_overlap():
    """
    Feature: test model parallel strategy
    Description: shard batch dimension, channel dimension and w dimension
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=16, global_rank=0)
    strategy1 = ((2, 2, 1, 4), (2, 1, 1, 1))
    strategy2 = ((2, 2, 1, 4),)
    net = Net2(_w1, out_channel=8, kernel_size=(2, 2), pad_mode="same", stride=2,
               strategy1=strategy1, strategy2=strategy2)
    compile_net(net)


def test_conv2d_transpose_split_h_or_w_in_pad_mode():
    """
    Feature: test pad mode
    Description: shard batch dimension, channel dimension and w dimension in pad mode
    Expectation: compile failed
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=16, global_rank=0)
    strategy1 = ((2, 2, 1, 4), (2, 1, 1, 1))
    strategy2 = ((2, 2, 1, 4),)
    net = Net2(_w1, out_channel=8, kernel_size=(2, 2), pad_mode="pad", stride=2,
               strategy1=strategy1, strategy2=strategy2)
    with pytest.raises(RuntimeError):
        compile_net(net)


def test_conv2d_transpose_split_h_in_same_mode():
    """
    Feature: test split h dimension
    Description: shard h dimension in same mode
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=16, global_rank=0)
    strategy1 = ((2, 2, 4, 1), (2, 1, 1, 1))
    strategy2 = ((2, 2, 4, 1),)
    net = Net2(_w1, out_channel=8, kernel_size=(2, 2), pad_mode="same", stride=2,
               strategy1=strategy1, strategy2=strategy2)
    compile_net(net)


def test_conv2d_transpose_overlap_size_too_large():
    """
    Feature: test overlap size is too large
    Description: shard w dimension and overlap size larger than slice shape
    Expectation: compile failed
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((1, 1, 1, 8), (1, 1, 1, 1))
    strategy2 = ((1, 1, 1, 8),)
    net = Net2(_w3, out_channel=8, kernel_size=(10, 10), pad_mode="same", stride=2,
               strategy1=strategy1, strategy2=strategy2)
    with pytest.raises(RuntimeError):
        compile_net(net)
