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

import mindspore as ms
from mindspore import context, Tensor, Parameter
from mindspore.common.api import _cell_graph_executor
from mindspore.nn import Cell, TrainOneStepCell, Momentum, BatchNorm2d, BatchNorm1d
from mindspore.ops import operations as P
from tests.security_utils import security_off_wrap


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")

class Net(Cell):
    def __init__(self, conv2d_weight, out_channel, kernel_size, pad_mode, stride,
                 strategy1=None, strategy2=None):
        super().__init__()
        self.conv2d = P.Conv2D(out_channel=out_channel, kernel_size=kernel_size,
                               pad_mode=pad_mode, stride=stride).shard(strategy1)
        self.conv2d_weight = Parameter(conv2d_weight, "w1")
        self.bn = BatchNorm2d(8)
        self.bn.bn_train.shard(strategy2)
        self.print = P.Print()

    def construct(self, x, b):
        out = self.conv2d(x, self.conv2d_weight)
        self.print("output is", out)
        out = self.bn(out)
        return out


_x = Tensor(np.ones([32, 16, 8, 8]), dtype=ms.float32)
_w1 = Tensor(np.ones([8, 16, 2, 2]), dtype=ms.float32)
_b = Tensor(np.ones([32, 16, 8, 8]), dtype=ms.float32)


def compile_net(net):
    optimizer = Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
    train_net = TrainOneStepCell(net, optimizer)
    train_net.set_train()
    _cell_graph_executor.compile(train_net, _x, _b)
    context.reset_auto_parallel_context()


@security_off_wrap
def test_batchnorm_data_parallel():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((8, 1, 1, 1), (1, 1, 1, 1))
    strategy2 = ((8, 1, 1, 1), (1,), (1,), (1,), (1,))
    net = Net(_w1, out_channel=8, kernel_size=2, pad_mode="same", stride=1, strategy1=strategy1, strategy2=strategy2)
    compile_net(net)


@security_off_wrap
def test_batchnorm_model_parallel1():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((2, 2, 1, 1), (2, 2, 1, 1))
    strategy2 = ((2, 1, 2, 2), (1,), (1,), (1,), (1,))
    net = Net(_w1, out_channel=8, kernel_size=2, pad_mode="same", stride=1, strategy1=strategy1, strategy2=strategy2)
    compile_net(net)


@security_off_wrap
def test_batchnorm_model_parallel2():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=32, global_rank=0)
    strategy1 = ((2, 2, 2, 2), (2, 2, 1, 1))
    strategy2 = ((1, 8, 1, 1), (8,), (8,), (8,), (8,))
    net = Net(_w1, out_channel=8, kernel_size=2, pad_mode="same", stride=2, strategy1=strategy1, strategy2=strategy2)
    compile_net(net)


class Net2(Cell):
    def __init__(self, strategy1=None, strategy2=None):
        super().__init__()
        self.bn = BatchNorm1d(8)
        self.bn.bn_train.shard(strategy1)
        self.relu = P.ReLU().shard(strategy2)

    def construct(self, x, b):
        out = self.bn(x)
        out = self.relu(out)
        return out


_x1 = Tensor(np.ones([32, 8]), dtype=ms.float32)
_b1 = Tensor(np.ones([32, 8]), dtype=ms.float32)


def compile_net2(net):
    optimizer = Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
    train_net = TrainOneStepCell(net, optimizer)
    train_net.set_train()
    _cell_graph_executor.compile(train_net, _x1, _b1)
    context.reset_auto_parallel_context()


def test_batchnorm1d_data_parallel():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((8, 1), (1,), (1,), (1,), (1,))
    strategy2 = ((8, 1),)
    net = Net2(strategy1=strategy1, strategy2=strategy2)
    compile_net2(net)


def test_batchnorm1d_model_parallel1():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((1, 8), (8,), (8,), (8,), (8,))
    strategy2 = ((1, 8),)
    net = Net2(strategy1=strategy1, strategy2=strategy2)
    compile_net2(net)


def test_batchnorm1d_model_parallel2():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=32, global_rank=0)
    strategy1 = ((2, 4), (4,), (4,), (4,), (4,))
    strategy2 = ((2, 4),)
    net = Net2(strategy1=strategy1, strategy2=strategy2)
    compile_net2(net)
