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

import numpy as np

import mindspore as ms
from mindspore import context, Tensor, Parameter
from mindspore.common.api import _cell_graph_executor
from mindspore.nn import Cell, TrainOneStepCell, Momentum
from mindspore.ops import operations as P
context.set_context(save_graphs=2, save_graphs_path="./")

class Net(Cell):
    def __init__(self, w, strategy1=None):
        super().__init__()
        self.tril = P.Tril().shard(strategy1)
        self.mul = P.Mul()
        self.mul_weight = Parameter(w, "w1")
        self.sigmoid = P.Sigmoid()

    def construct(self, x):
        out = self.mul(x, self.mul_weight)
        out = self.tril(out)
        out = self.sigmoid(out)
        return out


_x = Tensor(np.ones([16, 64, 32]), dtype=ms.float32)
_w = Tensor(np.ones([64, 32]), dtype=ms.float32)


def compile_net(net):
    optimizer = Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
    train_net = TrainOneStepCell(net, optimizer)
    train_net.set_train()
    _cell_graph_executor.compile(train_net, _x)
    context.reset_auto_parallel_context()


def test_auto_parallel_tril():
    """
    Feature: test tril auto parallel
    Description: auto parallel
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="auto_parallel", search_mode="dynamic_programming", device_num=16,
                                      global_rank=2)
    strategy1 = ((16, 1, 1),)
    net = Net(_w, strategy1)
    compile_net(net)


def test_semi_auto_parallel_tril():
    """
    Feature: test tril semi auto parallel
    Description: semi auto parallel
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((8, 1, 1),)
    net = Net(_w, strategy1)
    compile_net(net)
