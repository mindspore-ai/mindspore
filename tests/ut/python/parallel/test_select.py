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
# ============================================================================
import numpy as np
import pytest

import mindspore as ms
from mindspore import context, Tensor, Parameter
from mindspore.nn import Cell, Momentum
from mindspore.ops import operations as P
from mindspore.train import Model
from tests.dataset_mock import MindData


class Dataset(MindData):
    def __init__(self, predict, label, length=3):
        super(Dataset, self).__init__(size=length)
        self.predict = predict
        self.label = label
        self.index = 0
        self.length = length

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= self.length:
            raise StopIteration
        self.index += 1
        return self.predict, self.label

    def reset(self):
        self.index = 0


class Net(Cell):
    def __init__(self, w1, w2, strategy1=None, strategy2=None):
        super().__init__()
        self.less = P.Less().shard(strategy1)
        self.w1 = Parameter(w1, "w1")
        self.w2 = Parameter(w2, "w2")
        self.select = P.Select().shard(strategy2)

    def construct(self, x, b):
        out = self.less(x, b)
        out = self.select(out, self.w1, self.w2)
        return out


_x = Tensor(np.ones([16, 64, 32]), dtype=ms.float32)
_b = Tensor(np.ones([16, 64, 32]), dtype=ms.float32)
_w1 = Tensor(np.ones([128, 64, 32]), dtype=ms.float32)
_w2 = Tensor(np.ones([128, 64, 32]), dtype=ms.float32)


def compile_net(net):
    learning_rate = 0.1
    momentum = 0.9
    epoch_size = 2
    dataset = Dataset(_x, _b)
    opt = Momentum(net.trainable_params(), learning_rate, momentum)
    model = Model(net, optimizer=opt)
    model.train(epoch_size, dataset, dataset_sink_mode=False)
    context.reset_auto_parallel_context()


def test_select_data_parallel():
    context.set_auto_parallel_context(
        parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((8, 1, 1), (8, 1, 1))
    strategy2 = ((8, 1, 1), (8, 1, 1), (8, 1, 1))
    net = Net(_w1, _w2, strategy1, strategy2)
    compile_net(net)


def test_select_model_parallel():
    context.set_auto_parallel_context(
        parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((2, 2, 2), (2, 2, 2))
    strategy2 = ((2, 2, 2), (2, 2, 2), (2, 2, 2))
    net = Net(_w1, _w2, strategy1, strategy2)
    compile_net(net)


def test_select_mirror():
    context.set_auto_parallel_context(
        parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((1, 2, 2), (1, 2, 2))
    strategy2 = ((1, 2, 2), (1, 2, 2), (1, 2, 2))
    net = Net(_w1, _w2, strategy1, strategy2)
    compile_net(net)


def test_select_auto_parallel():
    context.set_auto_parallel_context(
        parallel_mode="auto_parallel", device_num=8, global_rank=0)
    net = Net(_w1, _w2)
    compile_net(net)


def test_select_strategy_error():
    context.set_auto_parallel_context(
        parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((2, 2, 2), (2, 2, 2))
    strategy2 = ((8, 1, 1), (2, 2, 2), (2, 2, 2))
    net = Net(_w1, _w2, strategy1, strategy2)
    with pytest.raises(RuntimeError):
        compile_net(net)
