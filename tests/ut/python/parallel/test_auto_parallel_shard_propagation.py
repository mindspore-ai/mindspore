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
import mindspore as ms
from mindspore import context, Tensor, Parameter
from mindspore.common.api import _cell_graph_executor
from mindspore.nn import Cell, TrainOneStepCell, Momentum
from mindspore.ops import operations as P


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")


class Net(Cell):
    def __init__(self, mul_weight, strategy1=None, strategy2=None, strategy3=None):
        super().__init__()
        self.mul = P.Mul().shard(strategy1)
        self.sigmoid = P.Sigmoid().shard(strategy2)
        self.relu = P.ReLU().shard(strategy3)
        self.mul_weight = Parameter(mul_weight, "w1")

    def construct(self, x, b):
        out = self.mul(x, self.mul_weight)
        out = self.sigmoid(out)
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


def test_auto_parallel_activation1():
    context.set_auto_parallel_context(parallel_mode="auto_parallel", device_num=16, global_rank=0,
                                      search_mode="sharding_propagation")
    strategy1 = ((4, 4), (4, 4))
    strategy2 = None
    net = Net(_w1, strategy1, strategy2)
    compile_net(net)


def test_auto_parallel_activation2():
    context.set_auto_parallel_context(parallel_mode="auto_parallel", device_num=16, global_rank=0,
                                      search_mode="sharding_propagation")
    strategy1 = None
    strategy2 = ((4, 4),)
    net = Net(_w1, strategy1, strategy2)
    compile_net(net)


def auto_parallel_activation3():
    context.set_auto_parallel_context(parallel_mode="auto_parallel", device_num=16, global_rank=0,
                                      search_mode="sharding_propagation")
    strategy1 = ((4, 4), (4, 4))
    strategy2 = None
    strategy3 = ((4, 4),)
    net = Net(_w1, strategy1, strategy2, strategy3)
    compile_net(net)


def test_auto_parallel_activation4():
    context.set_auto_parallel_context(parallel_mode="auto_parallel", device_num=16, global_rank=0,
                                      auto_parallel_search_mode="sharding_propagation")
    strategy1 = ((4, 4), (4, 4))
    strategy2 = None
    strategy3 = ((8, 2),)
    net = Net(_w1, strategy1, strategy2, strategy3)
    compile_net(net)
