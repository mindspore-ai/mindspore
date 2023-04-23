# Copyright 2023 Huawei Technologies Co., Ltd
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
from mindspore.common.api import _cell_graph_executor
from mindspore.nn import Cell, TrainOneStepCell, Momentum
from mindspore.ops import operations as P


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")


class Net(Cell):
    def __init__(self, mul_weight, strategy1=None, strategy2=None):
        super().__init__()
        self.mul1 = P.Mul().shard(strategy1)
        self.mul2 = P.Mul()
        self.add = P.Add().shard(strategy2)
        self.relu = P.ReLU()
        self.mul_weight = Parameter(mul_weight, "w1")

    def construct(self, x, b):
        out = self.mul1(x, self.mul_weight)
        out1 = self.mul2(b, self.mul_weight)
        out = self.add(out, out1)
        out = self.relu(out)
        return out


class Net1(Cell):
    def __init__(self, mul_weight, strategy1=None, strategy2=None):
        super().__init__()
        self.mul1 = P.Mul().shard(strategy1)
        self.cast = P.Cast()
        self.mul2 = P.Mul()
        self.add = P.Add().shard(strategy2)
        self.relu = P.ReLU()
        self.mul_weight = Parameter(mul_weight, "w1")

    def construct(self, x, b):
        out = self.mul1(x, self.cast(self.mul_weight, ms.float32))
        out1 = self.mul2(b, self.mul_weight)
        out = self.add(out, out1)
        out = self.relu(out)
        return out


class Net2(Cell):
    def __init__(self, mul_weight, strategy1=None, strategy2=None):
        super().__init__()
        self.mul1 = P.Mul().shard(strategy1)
        self.cast = P.Cast()
        self.mul2 = P.Mul()
        self.add = P.Add().shard(strategy2)
        self.relu = P.ReLU()
        self.mul_weight = Parameter(mul_weight, "w1")

    def construct(self, x, b):
        out = self.mul1(x, self.mul_weight)
        out1 = self.mul2(b, self.cast(self.mul_weight, ms.float32))
        out = self.add(out, out1)
        out = self.relu(out)
        return out


_x = Tensor(np.ones([64, 32]), dtype=ms.float32)
_w1 = Tensor(np.ones([64, 32]), dtype=ms.float32)
_b = Tensor(np.ones([64, 32]), dtype=ms.float32)


def compile_net(net):
    optimizer = Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
    train_net = TrainOneStepCell(net, optimizer)
    train_net.set_train()
    _cell_graph_executor.compile(train_net, _x, _b)
    context.reset_auto_parallel_context()


def test_sp_share_param1():
    """
    Feature: share_param1 case with sharding_propagation
    Description: no cast
    Expectation: success
    """
    context.set_auto_parallel_context(parallel_mode="auto_parallel", device_num=8, global_rank=0,
                                      search_mode="sharding_propagation")
    strategy1 = ((2, 4), (2, 4))
    strategy2 = ((8, 1), (8, 1))
    net = Net(_w1, strategy1, strategy2)
    compile_net(net)


def test_sp_share_param2():
    """
    Feature: share_param1 case with sharding_propagation
    Description: a cast connect to mul
    Expectation: success
    """
    context.set_auto_parallel_context(parallel_mode="auto_parallel", device_num=8, global_rank=0,
                                      search_mode="sharding_propagation")
    strategy1 = ((2, 4), (2, 4))
    strategy2 = ((8, 1), (8, 1))
    net = Net1(_w1, strategy1, strategy2)
    compile_net(net)


def test_sp_share_param3():
    """
    Feature: share_param1 case with sharding_propagation
    Description: a cast connect to mul1
    Expectation: success
    """
    context.set_auto_parallel_context(parallel_mode="auto_parallel", device_num=8, global_rank=0,
                                      search_mode="sharding_propagation")
    strategy1 = ((2, 4), (2, 4))
    strategy2 = ((8, 1), (8, 1))
    net = Net2(_w1, strategy1, strategy2)
    compile_net(net)
