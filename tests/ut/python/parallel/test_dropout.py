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
from mindspore.nn import Cell, TrainOneStepCell, Momentum
from mindspore.ops import operations as P


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")


class Net(Cell):
    def __init__(self, mul_weight, strategy1=None, strategy2=None):
        super().__init__()
        self.mul = P.Mul().shard(strategy1)
        self.dropout1 = P.Dropout(keep_prob=0.5).shard(strategy2)
        self.relu = P.ReLU().shard(strategy2)
        self.dropout2 = P.Dropout(keep_prob=0.5).shard(strategy2)
        self.relu2 = P.ReLU().shard(strategy2)
        self.mul_weight = Parameter(mul_weight, "w1")

    def construct(self, x, b):
        out = self.mul(x, self.mul_weight)
        out, _ = self.dropout1(out)
        out = self.relu(out)
        out, _ = self.dropout2(out)
        out = self.relu2(out)
        return out


_x = Tensor(np.ones([128, 64]), dtype=ms.float32)
_w1 = Tensor(np.ones([128, 64]), dtype=ms.float32)
_b = Tensor(np.ones([128, 64]), dtype=ms.float32)


def compile_net(net):
    optimizer = Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
    train_net = TrainOneStepCell(net, optimizer)
    train_net.set_train()
    _cell_graph_executor.compile(train_net, _x, _b)
    context.reset_auto_parallel_context()


def test_dropout_data_parallel():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=16, global_rank=0)
    strategy1 = ((16, 1), (16, 1))
    strategy2 = ((16, 1),)
    net = Net(_w1, strategy1, strategy2)
    compile_net(net)


def test_dropout_model_parallel():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=16, global_rank=0)
    strategy1 = ((1, 16), (1, 16))
    strategy2 = ((1, 16),)
    net = Net(_w1, strategy1, strategy2)
    compile_net(net)


def test_dropout_mixed_parallel():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=16, global_rank=0)
    strategy1 = ((4, 4), (4, 4))
    strategy2 = ((4, 4),)
    net = Net(_w1, strategy1, strategy2)
    compile_net(net)


def test_dropout_auto_parallel():
    context.set_auto_parallel_context(parallel_mode="auto_parallel", device_num=16, global_rank=0)
    net = Net(_w1)
    compile_net(net)


def test_dropout_repeat_calc():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=16, global_rank=0)
    strategy1 = ((4, 4), (4, 4))
    strategy2 = ((2, 4),)
    net = Net(_w1, strategy1, strategy2)
    compile_net(net)
