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
from mindspore import context, Tensor, Parameter
from mindspore.common.api import _executor
from mindspore.nn import Cell, TrainOneStepCell, Momentum
from mindspore.ops import operations as P


class Net(Cell):
    def __init__(self, weight, weight2, strategy1=None, strategy2=None, is_parameter=True):
        super().__init__()
        self.mul = P.Mul().shard(strategy1)
        self.tile = P.Tile().shard(strategy2)
        if is_parameter:
            self.weight = Parameter(weight, "w1")
        else:
            self.weight = weight
        self.mul2 = P.Mul()
        self.weight2 = Parameter(weight2, "w2")

    def construct(self, x, b):
        out = self.tile(self.weight, (8, 4, 2))
        out = self.mul(x, out)
        out = self.mul2(out, self.weight2)
        return out


class Net2(Cell):
    def __init__(self, weight2, strategy1=None, strategy2=None):
        super().__init__()
        self.mul = P.Mul().shard(strategy1)
        self.tile = P.Tile().shard(strategy2)
        self.weight2 = Parameter(weight2, "w2")

    def construct(self, x, b):
        out = self.mul(x, self.weight2)
        out = self.tile(out, (8, 8, 4, 2))
        return out


_x = Tensor(np.ones([128, 64, 32]), dtype=ms.float32)
_w1 = Tensor(np.ones([16, 16, 16]), dtype=ms.float32)
_w2 = Tensor(np.ones([128, 64, 32]), dtype=ms.float32)
_b = Tensor(np.ones([128, 64, 32]), dtype=ms.float32)


def compile_net(net):
    context.set_context(save_graphs=False)
    optimizer = Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
    train_net = TrainOneStepCell(net, optimizer)
    train_net.set_auto_parallel()
    train_net.set_train()
    _executor.compile(train_net, _x, _b)
    context.reset_auto_parallel_context()


def test_tile_parameter():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((2, 2, 2), (2, 2, 2))
    strategy2 = ((2, 2, 2),)
    net = Net(_w1, _w2, strategy1, strategy2, is_parameter=True)
    compile_net(net)


def test_tile_parameter_no_full_split():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((2, 2, 2), (2, 2, 2))
    strategy2 = ((2, 2, 1),)
    net = Net(_w1, _w2, strategy1, strategy2, is_parameter=True)
    compile_net(net)


def test_tile_tensor():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((2, 2, 2), (2, 2, 2))
    strategy2 = ((2, 2, 2),)
    net = Net(_w1, _w2, strategy1, strategy2, is_parameter=False)
    compile_net(net)


def test_tile_tensor_no_full_split():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((2, 2, 2), (2, 2, 2))
    strategy2 = ((2, 2, 1),)
    net = Net(_w1, _w2, strategy1, strategy2, is_parameter=False)
    compile_net(net)


def test_tile_output():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((2, 2, 2), (2, 2, 2))
    strategy2 = ((1, 2, 2, 2),)
    net = Net2(_w2, strategy1, strategy2)
    compile_net(net)

def test_tile_output_no_full_split():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((2, 2, 2), (2, 2, 2))
    strategy2 = ((1, 2, 1, 2),)
    net = Net2(_w2, strategy1, strategy2)
    compile_net(net)


def test_tile_no_strategy():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((2, 2, 2), (2, 2, 2))
    strategy2 = None
    net = Net2(_w2, strategy1, strategy2)
    compile_net(net)

def test_tile_auto_parallel():
    context.set_auto_parallel_context(parallel_mode="auto_parallel", device_num=8, global_rank=0)
    net = Net2(_w2)
    compile_net(net)
