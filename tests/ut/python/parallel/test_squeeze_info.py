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
from mindspore import context, Tensor
from mindspore.common.api import _cell_graph_executor
from mindspore.nn import Cell
from mindspore.ops import operations as P


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")


class Net(Cell):
    def __init__(self, strategy1=None, strategy2=None, axis=()):
        super().__init__()
        self.squeeze = P.Squeeze(axis=axis).shard(strategy1)
        self.mul = P.Mul().shard(strategy2)

    def construct(self, x, b):
        out = self.squeeze(x)
        out = self.mul(out, b)
        return out


_x = Tensor(np.ones([64, 1, 32, 1]), dtype=ms.float32)
_b = Tensor(np.ones([64, 32]), dtype=ms.float32)


def compile_net(net):
    net.set_train()
    _cell_graph_executor.compile(net, _x, _b)
    context.reset_auto_parallel_context()


def test_squeeze_data_parallel():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=16, global_rank=0)
    strategy1 = ((16, 1, 1, 1),)
    strategy2 = ((16, 1), (16, 1))
    net = Net(strategy1, strategy2)
    compile_net(net)


def test_squeeze_model_parallel():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=16, global_rank=0)
    strategy1 = ((1, 1, 16, 1),)
    strategy2 = ((1, 16), (1, 16))
    net = Net(strategy1, strategy2)
    compile_net(net)


def test_squeeze_specified_axis():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=16, global_rank=0)
    strategy1 = ((4, 1, 4, 1),)
    strategy2 = ((8, 2), (8, 2))
    net = Net(strategy1, strategy2, (1, 3))
    compile_net(net)


def test_squeeze_auto_parallel():
    context.set_auto_parallel_context(parallel_mode="auto_parallel", device_num=16, global_rank=0)
    net = Net()
    compile_net(net)


def test_squeeze_repeat_calc():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=16, global_rank=0)
    strategy1 = ((1, 1, 8, 1),)
    strategy2 = ((2, 8), (2, 8))
    net = Net(strategy1, strategy2)
    compile_net(net)
