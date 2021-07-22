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
    def __init__(self, w1_shape, indices_shape, strategy1=None, strategy2=None, strategy3=None):
        super().__init__()
        self.mul = P.Mul().shard(strategy1)
        self.w1 = Parameter(Tensor(np.ones(w1_shape), dtype=ms.float32), "w1")
        self.indices = Tensor(np.ones(indices_shape), dtype=ms.int32)
        self.gathernd = P.GatherNd().shard(strategy2)
        self.relu = P.ReLU().shard(strategy3)

    def construct(self, x, b):
        out = self.mul(x, self.w1)
        out = self.gathernd(out, self.indices)
        out = self.relu(out)
        return out


class Net2(Cell):
    def __init__(self, w1_shape, indices_shape, strategy1=None, strategy2=None, strategy3=None):
        super().__init__()
        self.mul = P.Mul().shard(strategy1)
        self.w1 = Parameter(Tensor(np.ones(w1_shape), dtype=ms.float32), "w1")
        self.indices = Tensor(np.ones(indices_shape), dtype=ms.int32)
        self.gathernd = P.GatherNd().shard(strategy2)
        self.relu = P.ReLU().shard(strategy3)

    def construct(self, x, b):
        out = self.mul(x, self.w1)
        out = self.gathernd(out, self.indices)
        return out


class Net3(Cell):
    def __init__(self, w1_shape, indices_shape, strategy1=None, strategy2=None, strategy3=None):
        super().__init__()
        self.mul = P.Mul().shard(strategy1)
        self.w1 = Parameter(Tensor(np.ones(w1_shape), dtype=ms.float32), "w1")
        self.indices = Tensor(np.ones(indices_shape), dtype=ms.int32)
        self.gathernd = P.GatherNd().shard(strategy2)
        self.relu = P.ReLU().shard(strategy3)

    def construct(self, x, b):
        out = self.gathernd(x, self.indices)
        out = self.relu(out)
        out = self.mul(out, self.w1)
        return out


# full_batch = false
_x = Tensor(np.ones([1, 16, 32]), dtype=ms.float32)
_b = Tensor(np.ones([1, 16, 32]), dtype=ms.float32)


def compile_net(net):
    learning_rate = 0.1
    momentum = 0.9
    epoch_size = 2
    dataset = Dataset(_x, _b)
    opt = Momentum(net.trainable_params(), learning_rate, momentum)
    model = Model(net, optimizer=opt)
    model.train(epoch_size, dataset, dataset_sink_mode=False)
    context.reset_auto_parallel_context()


def test_gathernd_data_parallel():
    context.set_auto_parallel_context(
        parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    w1_shape = [8, 16, 32]
    indices_shape = [8, 4, 2, 1]
    strategy1 = ((8, 1, 1), (8, 1, 1))
    strategy2 = ((1, 1, 1), (8, 1, 1, 1))
    strategy3 = ((8, 1, 1, 1, 1),)
    net = Net(w1_shape, indices_shape, strategy1, strategy2, strategy3)
    compile_net(net)


def test_gathernd_data_parallel2():
    context.set_auto_parallel_context(
        parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    w1_shape = [8, 16, 32]
    indices_shape = [8, 4, 2, 2]
    strategy1 = ((8, 1, 1), (8, 1, 1))
    strategy2 = ((1, 1, 1), (8, 1, 1, 1))
    strategy3 = ((8, 1, 1, 1),)
    net = Net(w1_shape, indices_shape, strategy1, strategy2, strategy3)
    compile_net(net)


def test_gathernd_data_parallel3():
    context.set_auto_parallel_context(
        parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    w1_shape = [8, 16, 32]
    indices_shape = [8, 4, 2, 3]
    strategy1 = ((8, 1, 1), (8, 1, 1))
    strategy2 = ((1, 1, 1), (8, 1, 1, 1))
    strategy3 = ((8, 1, 1),)
    net = Net(w1_shape, indices_shape, strategy1, strategy2, strategy3)
    compile_net(net)


def test_gathernd_data_parallel4():
    context.set_auto_parallel_context(
        parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    w1_shape = [8, 16, 32]
    indices_shape = [8, 4, 2, 1]
    strategy1 = ((8, 1, 1), (8, 1, 1))
    strategy2 = ((1, 1, 1), (8, 1, 1, 1))
    strategy3 = ((8, 1, 1, 1, 1),)
    net = Net2(w1_shape, indices_shape, strategy1, strategy2, strategy3)
    compile_net(net)


def test_gathernd_data_parallel5():
    context.set_auto_parallel_context(
        parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    w1_shape = [8, 16, 32]
    indices_shape = [8, 4, 2, 2]
    strategy1 = ((8, 1, 1), (8, 1, 1))
    strategy2 = ((1, 1, 1), (8, 1, 1, 1))
    strategy3 = ((8, 1, 1, 1),)
    net = Net2(w1_shape, indices_shape, strategy1, strategy2, strategy3)
    compile_net(net)


def test_gathernd_data_parallel6():
    context.set_auto_parallel_context(
        parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    w1_shape = [8, 16, 32]
    indices_shape = [8, 4, 2, 3]
    strategy1 = ((8, 1, 1), (8, 1, 1))
    strategy2 = ((1, 1, 1), (8, 1, 1, 1))
    strategy3 = ((8, 1, 1),)
    net = Net2(w1_shape, indices_shape, strategy1, strategy2, strategy3)
    compile_net(net)


def test_gathernd_data_parallel7():
    context.set_auto_parallel_context(
        parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    w1_shape = [8, 4, 2, 16, 32]
    indices_shape = [8, 4, 2, 1]
    strategy1 = ((8, 1, 1, 1, 1), (8, 1, 1, 1, 1))
    strategy2 = ((1, 1, 1), (8, 1, 1, 1))
    strategy3 = ((8, 1, 1, 1, 1),)
    net = Net3(w1_shape, indices_shape, strategy1, strategy2, strategy3)
    compile_net(net)


def test_gathernd_data_parallel8():
    context.set_auto_parallel_context(
        parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    w1_shape = [8, 4, 2, 32]
    indices_shape = [8, 4, 2, 2]
    strategy1 = ((8, 1, 1, 1), (8, 1, 1, 1))
    strategy2 = ((1, 1, 1), (8, 1, 1, 1))
    strategy3 = ((8, 1, 1, 1),)
    net = Net3(w1_shape, indices_shape, strategy1, strategy2, strategy3)
    compile_net(net)


def test_gathernd_data_parallel9():
    context.set_auto_parallel_context(
        parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    w1_shape = [8, 4, 2]
    indices_shape = [8, 4, 2, 3]
    strategy1 = ((8, 1, 1), (8, 1, 1))
    strategy2 = ((1, 1, 1), (8, 1, 1, 1))
    strategy3 = ((8, 1, 1),)
    net = Net3(w1_shape, indices_shape, strategy1, strategy2, strategy3)
    compile_net(net)


def test_gathernd_model_parallel():
    context.set_auto_parallel_context(
        parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    w1_shape = [8, 16, 32]
    indices_shape = [8, 4, 2, 1]
    strategy1 = ((8, 1, 1), (8, 1, 1))
    strategy2 = ((1, 1, 1), (2, 2, 2, 1))
    strategy3 = ((8, 1, 1, 1, 1),)
    net = Net(w1_shape, indices_shape, strategy1, strategy2, strategy3)
    compile_net(net)


def test_gathernd_model_parallel2():
    context.set_auto_parallel_context(
        parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    w1_shape = [8, 16, 32]
    indices_shape = [8, 4, 2, 2]
    strategy1 = ((8, 1, 1), (8, 1, 1))
    strategy2 = ((1, 1, 1), (2, 2, 2, 1))
    strategy3 = ((8, 1, 1, 1),)
    net = Net(w1_shape, indices_shape, strategy1, strategy2, strategy3)
    compile_net(net)


def test_gathernd_model_parallel3():
    context.set_auto_parallel_context(
        parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    w1_shape = [8, 16, 32]
    indices_shape = [8, 4, 2, 3]
    strategy1 = ((8, 1, 1), (8, 1, 1))
    strategy2 = ((1, 1, 1), (2, 2, 2, 1))
    strategy3 = ((8, 1, 1),)
    net = Net(w1_shape, indices_shape, strategy1, strategy2, strategy3)
    compile_net(net)


def test_gathernd_model_parallel4():
    context.set_auto_parallel_context(
        parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    w1_shape = [8, 16, 32]
    indices_shape = [8, 4, 2, 1]
    strategy1 = ((8, 1, 1), (8, 1, 1))
    strategy2 = ((1, 1, 1), (2, 2, 2, 1))
    strategy3 = ((8, 1, 1, 1, 1),)
    net = Net2(w1_shape, indices_shape, strategy1, strategy2, strategy3)
    compile_net(net)


def test_gathernd_model_parallel5():
    context.set_auto_parallel_context(
        parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    w1_shape = [8, 16, 32]
    indices_shape = [8, 4, 2, 2]
    strategy1 = ((8, 1, 1), (8, 1, 1))
    strategy2 = ((1, 1, 1), (2, 2, 2, 1))
    strategy3 = ((8, 1, 1, 1),)
    net = Net2(w1_shape, indices_shape, strategy1, strategy2, strategy3)
    compile_net(net)


def test_gathernd_model_parallel6():
    context.set_auto_parallel_context(
        parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    w1_shape = [8, 16, 32]
    indices_shape = [8, 4, 2, 3]
    strategy1 = ((8, 1, 1), (8, 1, 1))
    strategy2 = ((1, 1, 1), (2, 2, 2, 1))
    strategy3 = ((8, 1, 1),)
    net = Net2(w1_shape, indices_shape, strategy1, strategy2, strategy3)
    compile_net(net)


def test_gathernd_model_parallel7():
    context.set_auto_parallel_context(
        parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    w1_shape = [8, 4, 2, 16, 32]
    indices_shape = [8, 4, 2, 1]
    strategy1 = ((8, 1, 1, 1, 1), (8, 1, 1, 1, 1))
    strategy2 = ((1, 1, 1), (2, 2, 2, 1))
    strategy3 = ((8, 1, 1, 1, 1),)
    net = Net3(w1_shape, indices_shape, strategy1, strategy2, strategy3)
    compile_net(net)


def test_gathernd_model_parallel8():
    context.set_auto_parallel_context(
        parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    w1_shape = [8, 4, 2, 32]
    indices_shape = [8, 4, 2, 2]
    strategy1 = ((8, 1, 1, 1), (8, 1, 1, 1))
    strategy2 = ((1, 1, 1), (2, 2, 2, 1))
    strategy3 = ((8, 1, 1, 1),)
    net = Net3(w1_shape, indices_shape, strategy1, strategy2, strategy3)
    compile_net(net)


def test_gathernd_model_parallel9():
    context.set_auto_parallel_context(
        parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    w1_shape = [8, 4, 2]
    indices_shape = [8, 4, 2, 3]
    strategy1 = ((8, 1, 1), (8, 1, 1))
    strategy2 = ((1, 1, 1), (2, 2, 2, 1))
    strategy3 = ((8, 1, 1),)
    net = Net3(w1_shape, indices_shape, strategy1, strategy2, strategy3)
    compile_net(net)

def test_gathernd_auto_parallel():
    context.set_auto_parallel_context(
        parallel_mode="auto_parallel", device_num=8, global_rank=0)
    w1_shape = [8, 16, 32]
    indices_shape = [8, 4, 2, 1]
    net = Net(w1_shape, indices_shape)
    compile_net(net)


def test_gathernd_auto_parallel2():
    context.set_auto_parallel_context(
        parallel_mode="auto_parallel", device_num=8, global_rank=0)
    w1_shape = [8, 16, 32]
    indices_shape = [8, 4, 2, 2]
    net = Net(w1_shape, indices_shape)
    compile_net(net)


def test_gathernd_auto_parallel3():
    context.set_auto_parallel_context(
        parallel_mode="auto_parallel", device_num=8, global_rank=0)
    w1_shape = [8, 16, 32]
    indices_shape = [8, 4, 2, 3]
    net = Net(w1_shape, indices_shape)
    compile_net(net)


def test_gathernd_strategy_error():
    context.set_auto_parallel_context(
        parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    w1_shape = [8, 16, 32]
    indices_shape = [8, 4, 2, 3]
    strategy1 = ((8, 1, 1), (8, 1, 1))
    strategy2 = ((2, 1, 1), (1, 2, 2, 1))
    strategy3 = ((8, 1, 1),)
    net = Net(w1_shape, indices_shape, strategy1, strategy2, strategy3)
    with pytest.raises(RuntimeError):
        compile_net(net)


def test_gathernd_strategy_error2():
    context.set_auto_parallel_context(
        parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    w1_shape = [8, 16, 32]
    indices_shape = [8, 4, 2, 3]
    strategy1 = ((8, 1, 1), (8, 1, 1))
    strategy2 = ((1, 1, 1), (1, 2, 2, 2))
    strategy3 = ((8, 1, 1),)
    net = Net(w1_shape, indices_shape, strategy1, strategy2, strategy3)
    with pytest.raises(RuntimeError):
        compile_net(net)
