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

import numpy as np
import pytest
import mindspore as ms
from mindspore import context, Tensor, Parameter
from mindspore.common.api import _cell_graph_executor
from mindspore.nn import Cell, TrainOneStepCell, Momentum
from mindspore.ops import operations as P


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")


class Net(Cell):
    def __init__(self, mul_weight, strategy1=None, strategy2=None):
        super().__init__()
        self.mul = P.Mul().shard(strategy1)
        self.mul2 = P.Mul().shard(strategy2)
        self.mul_weight = Parameter(mul_weight, "w1")

    def construct(self, x, b):
        out = self.mul(x, self.mul_weight)
        out = self.mul2(out, self.mul_weight)
        return out


class Net2(Cell):
    def __init__(self, mul_weight, strategy1=None, strategy2=None):
        super().__init__()
        self.mul = P.Mul().shard(strategy1)
        self.mul2 = P.Mul().shard(strategy2)
        self.mul_weight = Parameter(mul_weight, "w1")

    def construct(self, x, b):
        out = self.mul(x, self.mul_weight)
        out = self.mul2(x, out)
        return out


class Net3(Cell):
    def __init__(self, mul_weight, strategy1=None, strategy2=None):
        super().__init__()
        self.mul = P.MatMul().shard(strategy1)
        self.mul2 = P.MatMul().shard(strategy2)
        self.mul_weight = Parameter(mul_weight, "w1")

    def construct(self, x, b):
        out = self.mul(x, self.mul_weight)
        out = self.mul2(out, self.mul_weight)
        return out


_x = Tensor(np.ones([16, 16]), dtype=ms.float32)
_w = Tensor(np.ones([16, 16]), dtype=ms.float32)
_b = Tensor(np.ones([16, 16]), dtype=ms.float32)


def compile_net(net):
    optimizer = Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
    train_net = TrainOneStepCell(net, optimizer)
    train_net.set_train()
    _cell_graph_executor.compile(train_net, _x, _b)
    context.reset_auto_parallel_context()


def test_parameter_same_split():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=16, global_rank=0)
    strategy1 = ((16, 1), (16, 1))
    strategy2 = ((16, 1), (16, 1))
    net = Net(_w, strategy1, strategy2)
    compile_net(net)


def test_parameter_different_split():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=16, global_rank=0)
    strategy1 = ((16, 1), (16, 1))
    strategy2 = ((4, 4), (4, 4))
    net = Net(_w, strategy1, strategy2)
    with pytest.raises(RuntimeError):
        compile_net(net)


def test_input_same_split():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=16, global_rank=0)
    strategy1 = ((16, 1), (16, 1))
    strategy2 = ((16, 1), (16, 1))
    net = Net(_w, strategy1, strategy2)
    compile_net(net)


def test_parameter_different_group():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=16, global_rank=0)
    strategy1 = ((1, 2), (2, 1))
    strategy2 = ((8, 2), (2, 1))
    net = Net3(_w, strategy1, strategy2)
    with pytest.raises(RuntimeError):
        compile_net(net)
