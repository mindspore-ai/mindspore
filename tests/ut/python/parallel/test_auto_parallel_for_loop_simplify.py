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
import mindspore.nn as nn
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


class SubNet(Cell):
    def __init__(self, index):
        super().__init__()
        self.matmul = P.MatMul()
        self.relu = P.ReLU()
        self.weight = Parameter(Tensor(np.ones([128, 128]), dtype=ms.float32), "matmul_w"+str(index))

    def construct(self, x):
        out = self.matmul(x, self.weight)
        out = self.relu(out)
        return out


class Net(Cell):
    def __init__(self, mul_weight, num_layers, strategy1=None, strategy2=None):
        super().__init__()
        self.mul = P.Mul().shard(strategy1)
        self.neg = P.Neg().shard(strategy2)
        self.mul_weight = Parameter(mul_weight, "w1")
        self.num_layers = num_layers
        self.layers = nn.CellList()
        for i in range(num_layers):
            self.layers.append(SubNet(i))

    def construct(self, x):
        for i in range(self.num_layers):
            x = self.layers[i](x)
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
    model = Model(net, loss, optimizer=opt)
    model.train(epoch_size, dataset, dataset_sink_mode=False)
    context.reset_auto_parallel_context()


def test_auto_parallel():
    context.set_auto_parallel_context(parallel_mode="auto_parallel", device_num=16, global_rank=0)
    net = Net(_w1, 3)
    compile_net(net)
