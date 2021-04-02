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
# ============================================================================
import numpy as np

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
    def __init__(self, weight, weight2, strategy1=None, strategy2=None, is_parameter=True):
        super().__init__()
        self.concat = P.Concat(axis=0).shard(strategy1)
        if is_parameter:
            self.weight = Parameter(weight, "w1")
        else:
            self.weight = weight
        self.mul = P.Mul().shard(strategy2)
        self.weight2 = Parameter(weight2, "w2")

    def construct(self, x, b):
        out = self.concat((self.weight, self.weight2))
        out = self.mul(x, out)
        return out


class Net2(Cell):
    def __init__(self, weight, strategy1=None, strategy2=None, axis=0):
        super().__init__()
        self.mul = P.Mul().shard(strategy1)
        self.concat = P.Concat(axis=axis).shard(strategy2)
        self.weight = Parameter(weight, "w")

    def construct(self, x, b):
        out = self.mul(x, x)
        out = self.concat((out, self.weight))
        return out


class Net3(Cell):
    def __init__(self, weight, weight2, weight3, strategy1=None, strategy2=None, is_parameter=True):
        super().__init__()
        self.concat = P.Concat(axis=0).shard(strategy1)
        if is_parameter:
            self.weight = Parameter(weight, "w1")
        else:
            self.weight = weight
        self.mul = P.Mul().shard(strategy2)
        self.weight2 = Parameter(weight2, "w2")
        self.weight3 = Parameter(weight3, "w3")

    def construct(self, x, b):
        out = self.concat((self.weight, self.weight2, self.weight3))
        out = self.mul(x, out)
        return out


_x = Tensor(np.ones([16, 64, 32]), dtype=ms.float32)
_b = Tensor(np.ones([16, 64, 32, 32]), dtype=ms.int32)
_w1 = Tensor(np.ones([96, 64, 32]), dtype=ms.float32)
_w2 = Tensor(np.ones([32, 64, 32]), dtype=ms.float32)
_w3 = Tensor(np.ones([128, 16, 32]), dtype=ms.float32)

w1 = Tensor(np.ones([48, 64, 32]), dtype=ms.float32)
w2 = Tensor(np.ones([16, 64, 32]), dtype=ms.float32)
w3 = Tensor(np.ones([64, 64, 32]), dtype=ms.float32)


def compile_net(net):
    context.set_context(save_graphs=False)
    learning_rate = 0.1
    momentum = 0.9
    epoch_size = 2
    dataset = Dataset(_x, _b)
    opt = Momentum(net.trainable_params(), learning_rate, momentum)
    model = Model(net, optimizer=opt, amp_level="O2")
    model.train(epoch_size, dataset, dataset_sink_mode=False)
    context.reset_auto_parallel_context()


def test_concat_parameter():
    context.set_auto_parallel_context(
        parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((1, 4, 2), (1, 4, 2))
    strategy2 = ((1, 4, 2), (1, 4, 2))
    net = Net(_w1, _w2, strategy1, strategy2, is_parameter=True)
    compile_net(net)


def test_concat_parameter_no_full_split():
    context.set_auto_parallel_context(
        parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((1, 2, 2), (1, 2, 2))
    strategy2 = ((1, 4, 2), (1, 4, 2))
    net = Net(_w1, _w2, strategy1, strategy2, is_parameter=True)
    compile_net(net)


def test_concat_tensor_and_parameter():
    context.set_auto_parallel_context(
        parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((1, 2, 2), (1, 2, 2))
    strategy2 = ((1, 4, 2), (1, 4, 2))
    net = Net(_w1, _w2, strategy1, strategy2, is_parameter=False)
    compile_net(net)


def test_concat_output():
    context.set_auto_parallel_context(
        parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((2, 2, 2), (2, 2, 2))
    strategy2 = ((1, 4, 2), (1, 4, 2))
    net = Net2(_w1, strategy1, strategy2)
    compile_net(net)


def test_concat_output_no_full_split():
    context.set_auto_parallel_context(
        parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((2, 2, 2), (2, 2, 2))
    strategy2 = ((1, 2, 2), (1, 2, 2))
    net = Net2(_w1, strategy1, strategy2)
    compile_net(net)


def test_concat_no_strategy():
    context.set_auto_parallel_context(
        parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((2, 2, 2), (2, 2, 2))
    strategy2 = None
    net = Net2(_w3, strategy1, strategy2, axis=1)
    compile_net(net)


def test_concat_auto_parallel():
    context.set_auto_parallel_context(
        parallel_mode="auto_parallel", device_num=8, global_rank=0)
    net = Net2(_w2)
    compile_net(net)


def test_concat_auto_parallel2():
    context.set_auto_parallel_context(
        parallel_mode="auto_parallel", device_num=8, global_rank=0)
    strategy1 = None
    strategy2 = None
    net = Net2(_w3, strategy1, strategy2, axis=1)
    compile_net(net)


def test_concat_auto_parallel_3_tensor():
    context.set_auto_parallel_context(
        parallel_mode="auto_parallel", device_num=8, global_rank=0)
    net = Net3(w1, w2, w3)
    compile_net(net)
