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
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
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
    def __init__(self, mul_weight, strategy1=None, strategy2=None):
        super().__init__()
        self.mul = P.Mul().shard(strategy1)
        self.neg = P.Neg().shard(strategy2)
        self.mul_weight = Parameter(mul_weight, "w1")

    def construct(self, x):
        out = self.mul(x, self.mul_weight)
        out = self.neg(out)
        return out


_x = Tensor(np.ones([32, 128]), dtype=ms.float32)
_b = Tensor(np.ones([32]), dtype=ms.int32)
_w1 = Tensor(np.ones([512, 128]), dtype=ms.float32)


def compile_net(net):
    context.set_context(save_graphs=False)
    learning_rate = 0.1
    momentum = 0.9
    epoch_size = 2
    dataset = Dataset(_x, _b)
    loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    opt = Momentum(net.trainable_params(), learning_rate, momentum)
    model = Model(net, loss, optimizer=opt, amp_level="O2")
    model.train(epoch_size, dataset, dataset_sink_mode=False)
    context.reset_auto_parallel_context()


def test_neg_data_parallel():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=16, global_rank=0)
    strategy1 = ((16, 1), (16, 1))
    strategy2 = ((16, 1),)
    net = Net(_w1, strategy1, strategy2)
    compile_net(net)


def test_neg_model_parallel():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=16, global_rank=0)
    strategy1 = ((1, 16), (1, 16))
    strategy2 = ((1, 16),)
    net = Net(_w1, strategy1, strategy2)
    compile_net(net)


def test_neg_hybrid_parallel():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=16, global_rank=0)
    strategy1 = ((4, 4), (4, 4))
    strategy2 = ((4, 4),)
    net = Net(_w1, strategy1, strategy2)
    compile_net(net)


def test_neg_auto_parallel():
    context.set_auto_parallel_context(parallel_mode="auto_parallel", device_num=16, global_rank=0)
    net = Net(_w1)
    compile_net(net)


def test_neg_repeat_calc():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=16, global_rank=0)
    strategy1 = ((4, 4), (4, 4))
    strategy2 = ((2, 2),)
    net = Net(_w1, strategy1, strategy2)
    compile_net(net)


def test_neg_repeat_calc2():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=16, global_rank=0)
    strategy1 = ((4, 2), (4, 2))
    strategy2 = ((4, 4),)
    net = Net(_w1, strategy1, strategy2)
    compile_net(net)


def test_parallel_optimizer_with_mix_precision():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=16, global_rank=0,
                                      enable_parallel_optimizer=True)
    strategy1 = ((8, 1), (8, 1))
    strategy2 = ((8, 1),)
    net = Net(_w1, strategy1, strategy2)
    compile_net(net)
