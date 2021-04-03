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
from mindspore.common import dtype as mstype
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
    def __init__(self, weight, start, limit, delta, strategy1=None, strategy2=None, strategy3=None):
        super().__init__()
        self.mul = P.Mul().shard(strategy1)
        if isinstance(start, float):
            self.type = mstype.float32
        else:
            self.type = mstype.int32
        self.start = Tensor(start, self.type)
        self.limit = Tensor(limit, self.type)
        self.delta = Tensor(delta, self.type)
        self.range = P.Range()
        self.range.shard(strategy2)
        self.mul2 = P.Mul().shard(strategy3)
        self.weight = Parameter(weight, "w")

    def construct(self, x, b):
        r_out = self.range(self.start, self.limit, self.delta)
        out = self.mul(x, self.weight)
        out = self.mul2(out, r_out)
        return out


dev_num = 4
_x = Tensor(np.ones([64 // dev_num, 8]), dtype=ms.float32)
_b = Tensor(np.ones([8]), dtype=ms.float32)
_w1 = Tensor(np.ones([64, 8]), dtype=ms.float32)


def compile_net(net):
    context.set_context(save_graphs=False)
    learning_rate = 0.1
    momentum = 0.9
    epoch_size = 2
    dataset = Dataset(_x, _b)
    opt = Momentum(net.trainable_params(), learning_rate, momentum)
    model = Model(net, optimizer=opt)
    model.train(epoch_size, dataset, dataset_sink_mode=False)
    context.reset_auto_parallel_context()


def test_range():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=dev_num, global_rank=2)
    strategy1 = ((2, 2), (2, 2))
    strategy2 = ((2,),)
    strategy3 = ((2, 2), (2,))
    net = Net(_w1, 0, 8, 1, strategy1, strategy2, strategy3)
    compile_net(net)


def test_range2():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=dev_num, global_rank=0)
    strategy1 = ((4, 1), (4, 1))
    strategy2 = ((1,),)
    strategy3 = ((4, 1), (1,))
    net = Net(_w1, 0.0, 4.0, 0.5, strategy1, strategy2, strategy3)
    compile_net(net)


def test_range3():
    context.set_auto_parallel_context(parallel_mode="auto_parallel", device_num=dev_num, global_rank=2)
    net = Net(_w1, 0.0, 4.0, 0.5)
    compile_net(net)
