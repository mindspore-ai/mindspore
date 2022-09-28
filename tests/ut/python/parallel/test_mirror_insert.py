# Copyright 2022 Huawei Technologies Co., Ltd
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


class DependNet(Cell):
    def __init__(self, mul_weight, strategy):
        super().__init__()
        self.matmul = P.MatMul().shard(strategy)
        self.cast = P.Cast()
        self.sigmoid = P.Sigmoid()
        self.relu = P.ReLU()
        self.weight = Parameter(mul_weight, "w1")
        self.denpend_tensor = Tensor(0.0, dtype=ms.float32)
        self.depend = P.Depend()

    def construct(self, x):
        u = self.relu(self.denpend_tensor)
        weight = self.depend(self.weight, u)
        out = self.matmul(self.cast(x, ms.float16), self.cast(weight, ms.float16))
        out = self.sigmoid(out)
        out = self.relu(out)
        return out


class LoadNet(Cell):
    def __init__(self, mul_weight, strategy):
        super().__init__()
        self.matmul = P.MatMul().shard(strategy)
        self.cast = P.Cast()
        self.sigmoid = P.Sigmoid()
        self.relu = P.ReLU()
        self.weight = Parameter(mul_weight, "w1")

    def construct(self, x):
        out = self.matmul(self.cast(x, ms.float16), self.cast(self.weight, ms.float16))
        out = self.sigmoid(out)
        out = self.relu(out)
        return out


_x = Tensor(np.ones([64, 32]), dtype=ms.float32)
_w1 = Tensor(np.ones([32, 64]), dtype=ms.float32)


def compile_net(net):
    optimizer = Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
    train_net = TrainOneStepCell(net, optimizer)
    train_net.set_train()
    _cell_graph_executor.compile(train_net, _x)
    context.reset_auto_parallel_context()


def test_mirror_insert1():
    """
    Feature: Test find parameter and insert mirror before Depend.
    Description: simulate auto_monad case: param->Depend
    Expectation: Successful graph compilation.
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy = ((8, 1), (1, 1))
    net = DependNet(_w1, strategy)
    compile_net(net)


def test_mirror_insert2():
    """
    Feature: Test find parameter and insert mirror before Load.
    Description: simulate auto_monad case: param->Load
    Expectation: Successful graph compilation.
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy = ((8, 1), (1, 1))
    net = LoadNet(_w1, strategy)
    compile_net(net)
