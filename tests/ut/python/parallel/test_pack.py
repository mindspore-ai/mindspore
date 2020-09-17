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
import mindspore as ms
import mindspore.context as context
from mindspore import Tensor, Parameter
import mindspore.nn as nn
from mindspore.common.api import _executor
from mindspore.nn import TrainOneStepCell, Momentum
from mindspore.ops import operations as P


class Net(nn.Cell):
    def __init__(self, weight1, weight2, axis=0, strategy1=None, strategy2=None, is_parameter=True):
        super(Net, self).__init__()
        self.pack = P.Pack(axis=axis).shard(strategy1)
        self.mul = P.Mul().shard(strategy2)
        if is_parameter:
            self.weight1 = Parameter(weight1, "w1")
        else:
            self.weight1 = weight1
        self.weight2 = Parameter(weight2, "w2")

    def construct(self, x):
        out = self.pack([self.weight1, self.weight2])
        out = self.mul(x, out)
        return out


class Net1(nn.Cell):
    def __init__(self, weight1, weight2, axis=0, strategy1=None, strategy2=None):
        super(Net1, self).__init__()
        self.pack = P.Pack(axis=axis).shard(strategy1)
        self.mul = P.Mul().shard(strategy2)
        self.weight1 = Parameter(weight1, "w1")
        self.weight2 = Parameter(weight2, "w2")

    def construct(self, x):
        out = self.mul(x, self.weight1)
        out = self.pack([out, self.weight2])
        return out


class Net2(nn.Cell):
    def __init__(self, weight1, weight2, weight3, axis=0, strategy1=None, strategy2=None, is_parameter=True):
        super(Net2, self).__init__()
        self.pack = P.Pack(axis=axis).shard(strategy1)
        self.mul = P.Mul().shard(strategy2)
        if is_parameter:
            self.weight1 = Parameter(weight1, "w1")
        else:
            self.weight1 = weight1
        self.weight2 = Parameter(weight2, "w2")
        self.weight3 = Parameter(weight2, "w3")

    def construct(self, x):
        out = self.pack([self.weight1, self.weight2, self.weight3])
        out = self.mul(x, out)
        return out


_w1 = Tensor(np.ones([48, 64]), dtype=ms.float32)
_w2 = Tensor(np.ones([48, 64]), dtype=ms.float32)
_w3 = Tensor(np.ones([48, 64]), dtype=ms.float32)
_x = Tensor(np.ones([2, 48, 64]), dtype=ms.float32)
_x1 = Tensor(np.ones([48, 64]), dtype=ms.float32)
_x2 = Tensor(np.ones([3, 48, 64]), dtype=ms.float32)


def compile_net(net):
    context.set_context(mode=context.GRAPH_MODE, save_graphs=True)
    optimizer = Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
    train_net = TrainOneStepCell(net, optimizer)
    train_net.set_auto_parallel()
    _executor.compile(train_net, _x)
    context.reset_auto_parallel_context()


def compile_net1(net):
    context.set_context(mode=context.GRAPH_MODE, save_graphs=True)
    optimizer = Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
    train_net = TrainOneStepCell(net, optimizer)
    train_net.set_auto_parallel()
    _executor.compile(train_net, _x1)
    context.reset_auto_parallel_context()


def compile_net2(net):
    context.set_context(mode=context.GRAPH_MODE, save_graphs=True)
    optimizer = Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
    train_net = TrainOneStepCell(net, optimizer)
    train_net.set_auto_parallel()
    _executor.compile(train_net, _x2)
    context.reset_auto_parallel_context()


def test_pack_parameter():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((4, 2), (4, 2))
    strategy2 = ((1, 4, 2), (1, 4, 2))
    net = Net(_w1, _w2, 0, strategy1, strategy2)
    compile_net(net)


def test_pack_parameter_no_full_split():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((2, 2), (2, 2))
    strategy2 = ((1, 4, 2), (1, 4, 2))
    net = Net(_w1, _w2, 0, strategy1, strategy2)
    compile_net(net)


def test_pack_tensor_and_parameter():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((4, 2), (4, 2))
    strategy2 = ((1, 4, 2), (1, 4, 2))
    net = Net(_w1, _w2, 0, strategy1, strategy2, False)
    compile_net(net)


def test_pack_output():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((4, 2), (4, 2))
    strategy2 = ((4, 2), (4, 2))
    net = Net1(_w1, _w2, 0, strategy1, strategy2)
    compile_net1(net)


def test_pack_output_axis1():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((4, 2), (4, 2))
    strategy2 = ((4, 2), (4, 2))
    net = Net1(_w1, _w2, 1, strategy1, strategy2)
    compile_net1(net)


def test_pack_output_no_full_split():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((2, 2), (2, 2))
    strategy2 = ((4, 2), (4, 2))
    net = Net1(_w1, _w2, 0, strategy1, strategy2)
    compile_net1(net)


def test_pack_no_strategy():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = None
    strategy2 = ((4, 2), (4, 2))
    net = Net1(_w1, _w2, 0, strategy1, strategy2)
    compile_net1(net)


def test_pack_no_strategy_axis1():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = None
    strategy2 = ((4, 2), (4, 2))
    net = Net1(_w1, _w2, 1, strategy1, strategy2)
    compile_net1(net)


def test_pack_auto_parallel():
    context.set_auto_parallel_context(parallel_mode="auto_parallel", device_num=8, global_rank=0)
    net = Net1(_w1, _w2, 0)
    compile_net1(net)


def test_pack_auto_parallel_axis1():
    context.set_auto_parallel_context(parallel_mode="auto_parallel", device_num=8, global_rank=0)
    net = Net1(_w1, _w2, 1)
    compile_net1(net)


def test_pack_auto_parallel_3_tensor():
    context.set_auto_parallel_context(parallel_mode="auto_parallel", device_num=8, global_rank=0)
    net = Net2(_w1, _w2, _w3)
    compile_net2(net)
