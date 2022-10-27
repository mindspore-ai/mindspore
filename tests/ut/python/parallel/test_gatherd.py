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
    def __init__(self, dim, index, strategy1=None, strategy2=None):
        super().__init__()
        self.gatherd = P.GatherD().shard(strategy1)
        self.neg = P.Neg().shard(strategy2)
        self.input = Parameter(index, "w1")
        self.dim = dim

    def construct(self, x, b):
        out = self.gatherd(self.input, self.dim, x)
        out = self.neg(out)
        return out


_x = Tensor(np.ones([16, 32, 64]), dtype=ms.int32)
_w1 = Tensor(np.ones([16, 32, 64]), dtype=ms.float32)
_b = Tensor(np.ones([16, 32, 64]), dtype=ms.float32)


def compile_net(net):
    optimizer = Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
    train_net = TrainOneStepCell(net, optimizer)
    train_net.set_train()
    _cell_graph_executor.compile(train_net, _x, _b)
    context.reset_auto_parallel_context()


def test_gathernd_dim0():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=16, global_rank=0)
    strategy1 = ((1, 2, 8), (1, 2, 8))
    strategy2 = ((1, 2, 8),)
    net = Net(0, _w1, strategy1, strategy2)
    compile_net(net)


def test_gathernd_dim2():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=16, global_rank=0)
    strategy1 = ((2, 8, 1), (2, 8, 1))
    strategy2 = ((2, 8, 1),)
    net = Net(2, _w1, strategy1, strategy2)
    compile_net(net)


def test_gathernd_dim2_default_batch_parallel():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=16, global_rank=0)
    strategy1 = None
    strategy2 = ((2, 8, 1),)
    net = Net(2, _w1, strategy1, strategy2)
    compile_net(net)


def test_gathernd_repeat_calc():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=16, global_rank=0)
    strategy1 = ((1, 2, 4), (1, 2, 4))
    strategy2 = ((1, 2, 4),)
    net = Net(0, _w1, strategy1, strategy2)
    compile_net(net)
