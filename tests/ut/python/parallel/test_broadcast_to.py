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
    def __init__(self, weight1, strategy1=None, strategy2=None, is_parameter=True):
        super(Net, self).__init__()
        self.shape = (8, 48, 64)
        self.broadcast = P.BroadcastTo(self.shape).shard(strategy1)
        self.mul = P.Mul().shard(strategy2)
        if is_parameter:
            self.weight1 = Parameter(weight1, "w1")
        else:
            self.weight1 = weight1

    def construct(self, x):
        out = self.broadcast(self.weight1)
        out = self.mul(x, out)
        return out


class MatMulNet(nn.Cell):
    def __init__(self, weight1, strategy1=None, strategy2=None, strategy3=None, is_parameter=True):
        super(MatMulNet, self).__init__()
        self.shape = (8, 64, 64)
        self.broadcast = P.BroadcastTo(self.shape).shard(strategy1)
        self.matmul = P.BatchMatMul().shard(strategy2)
        self.mul = P.Mul().shard(strategy3)
        if is_parameter:
            self.weight1 = Parameter(weight1, "w1")
        else:
            self.weight1 = weight1

    def construct(self, x1, x2):
        out = self.broadcast(x2)
        out = self.matmul(x1, out)
        out = self.mul(out, self.weight1)
        return out


_w1 = Tensor(np.ones([1, 48, 64]), dtype=ms.float32)
_x1 = Tensor(np.ones([8, 48, 64]), dtype=ms.float32)
_x2 = Tensor(np.ones([64, 64]), dtype=ms.float32)


def compile_net(net):
    context.set_context(mode=context.GRAPH_MODE, save_graphs=False)
    optimizer = Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
    train_net = TrainOneStepCell(net, optimizer)
    train_net.set_auto_parallel()
    train_net.set_train()
    _executor.compile(train_net, _x1)
    context.reset_auto_parallel_context()


def compile_net2(net):
    context.set_context(mode=context.GRAPH_MODE, save_graphs=False)
    optimizer = Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
    train_net = TrainOneStepCell(net, optimizer)
    train_net.set_auto_parallel()
    train_net.set_train()
    _executor.compile(train_net, _x1, _x2)
    context.reset_auto_parallel_context()


def test_BroadcastTo_parameter():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((1, 4, 2),)
    strategy2 = ((1, 4, 2), (1, 4, 2))
    net = Net(_w1, strategy1, strategy2)
    compile_net(net)


def test_BroadcastTo_parameter_no_full():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((1, 2, 2),)
    strategy2 = ((1, 4, 2), (1, 4, 2))
    net = Net(_w1, strategy1, strategy2)
    compile_net(net)


def test_BroadcastTo_auto_parallel():
    context.set_auto_parallel_context(parallel_mode="auto_parallel", device_num=8, global_rank=0)
    net = Net(_w1)
    compile_net(net)


def test_BroadcastTo_matmul():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((2, 4),)
    strategy2 = ((1, 1, 2), (1, 2, 4))
    strategy3 = ((1, 2, 4), (1, 2, 4))
    net = MatMulNet(_w1, strategy1, strategy2, strategy3)
    compile_net2(net)
