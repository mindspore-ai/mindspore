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
    def __init__(self, mul_weight, axis=0, out_nums=1, strategy1=None, strategy2=None, strategy3=None):
        super(Net, self).__init__()
        self.split = P.Split(axis, out_nums).shard(strategy1)
        self.mul = P.Mul().shard(strategy2)
        self.matmul = P.MatMul(transpose_b=True).shard(strategy2)
        self.matmul2 = P.MatMul().shard(strategy3)
        self.weight = Parameter(mul_weight, "w1")

    def construct(self, x):
        out = self.mul(x, self.weight)
        out1, out2, out3 = self.split(out)
        out = self.matmul(out1, out2)
        out = self.matmul2(out, out3)
        return out


class Net1(nn.Cell):
    def __init__(self, mul_weight, axis=0, out_nums=1, strategy1=None, strategy2=None):
        super(Net1, self).__init__()
        self.split = P.Split(axis, out_nums).shard(strategy1)
        self.mul = P.Mul().shard(strategy2)
        self.weight = Parameter(mul_weight, "w1")

    def construct(self, x):
        out1, out2 = self.split(self.weight)
        out = self.mul(x, out1)
        out = self.mul(out, out2)
        return out


class Net2(nn.Cell):
    def __init__(self, mul_weight, axis=0, out_nums=1, strategy1=None, strategy2=None):
        super(Net2, self).__init__()
        self.split = P.Split(axis, out_nums).shard(strategy1)
        self.mul = P.Mul().shard(strategy2)
        self.weight = Parameter(mul_weight, "w1")

    def construct(self, x):
        out = self.mul(x, self.weight)
        out1, _ = self.split(out)
        return out1


_w = Tensor(np.ones([48, 64]), dtype=ms.float32)
_x = Tensor(np.ones([48, 64]), dtype=ms.float32)

_w1 = Tensor(np.ones([96, 64, 32]), dtype=ms.float32)
_x1 = Tensor(np.ones([48, 64, 32]), dtype=ms.float32)

_w2 = Tensor(np.ones([48, 64, 32]), dtype=ms.float32)

def compile_net(net):
    context.set_context(mode=context.GRAPH_MODE, save_graphs=False)
    optimizer = Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
    train_net = TrainOneStepCell(net, optimizer)
    train_net.set_auto_parallel()
    train_net.set_train()
    _executor.compile(train_net, _x)
    context.reset_auto_parallel_context()


def compile_net1(net):
    context.set_context(mode=context.GRAPH_MODE, save_graphs=False)
    optimizer = Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
    train_net = TrainOneStepCell(net, optimizer)
    train_net.set_auto_parallel()
    train_net.set_train()
    _executor.compile(train_net, _x1)
    context.reset_auto_parallel_context()


def test_split_parameter():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((1, 4, 2),)
    strategy2 = ((1, 4, 2), (1, 4, 2))
    net = Net1(_w1, 0, 2, strategy1, strategy2)
    compile_net1(net)


def test_split_parameter_no_full_split():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((1, 2, 2),)
    strategy2 = ((1, 4, 2), (1, 4, 2))
    net = Net1(_w1, 0, 2, strategy1, strategy2)
    compile_net1(net)


def test_split_tensor():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((1, 8),)
    strategy2 = ((1, 8), (1, 8))
    strategy3 = ((1, 1), (1, 8))
    net = Net(_w, 0, 3, strategy1, strategy2, strategy3)
    compile_net(net)


def test_split_output():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((1, 4, 2),)
    strategy2 = ((1, 4, 2), (1, 4, 2))
    net = Net2(_w2, 0, 2, strategy1, strategy2)
    compile_net1(net)


def test_split_output_no_full_split():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((1, 2, 2),)
    strategy2 = ((1, 4, 2), (1, 4, 2))
    net = Net2(_w2, 0, 2, strategy1, strategy2)
    compile_net1(net)


def test_split_no_strategy():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = None
    strategy2 = ((1, 4, 2), (1, 4, 2))
    net = Net2(_w2, 0, 2, strategy1, strategy2)
    compile_net1(net)


def test_split_auto_parallel():
    context.set_auto_parallel_context(parallel_mode="auto_parallel", device_num=8, global_rank=0)
    net = Net2(_w2, 0, 2)
    compile_net1(net)
