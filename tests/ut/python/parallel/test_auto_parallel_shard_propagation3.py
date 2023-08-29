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
        self.add = P.TensorAdd()
        self.relu = P.ReLU().shard(strategy1)
        self.max = P.ReduceMax(keep_dims=True).shard(strategy2)
        self.sub = P.Sub().shard(strategy3)
        self.mul_weight = Parameter(mul_weight, "w1")

    def construct(self, x, b):
        out = self.add(x, self.mul_weight)
        out = self.relu(out)
        out2 = self.max(out)
        out = self.sub(out, out2)
        return out


_x = Tensor(np.ones([64, 32000]), dtype=ms.float32)
_w1 = Tensor(np.ones([64, 32000]), dtype=ms.float32)
_b = Tensor(np.ones([64, 32000]), dtype=ms.float32)


def compile_net(net):
    optimizer = Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
    train_net = TrainOneStepCell(net, optimizer)
    train_net.set_train()
    _cell_graph_executor.compile(train_net, _x, _b)
    context.reset_auto_parallel_context()


def test_auto_parallel_activation1():
    context.set_auto_parallel_context(parallel_mode="auto_parallel", device_num=8, global_rank=0,
                                      search_mode="sharding_propagation")
    strategy1 = None
    strategy2 = ((8, 1),)
    strategy3 = ((1, 8), (1, 1))
    net = Net(_w1, strategy1, strategy2, strategy3)
    compile_net(net)


def test_auto_parallel_activation2():
    context.set_auto_parallel_context(parallel_mode="auto_parallel", device_num=8, global_rank=0,
                                      search_mode="sharding_propagation")
    strategy1 = ((1, 8),)
    strategy2 = ((1, 1),)
    strategy3 = ((1, 8), (1, 1))
    net = Net(_w1, strategy1, strategy2, strategy3)
    compile_net(net)
