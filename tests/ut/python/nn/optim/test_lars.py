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
from collections import Counter
import numpy as np

import mindspore.nn as nn
from mindspore import Tensor, Parameter
from mindspore.common import dtype as mstype
from mindspore.common.api import _cell_graph_executor
from mindspore.nn import TrainOneStepCell, WithLossCell
from mindspore.nn.optim import LARS, Momentum
from mindspore.ops import operations as P


def multisteplr(total_steps, milestone, base_lr=0.9, gamma=0.1, dtype=mstype.float32):
    lr = []
    milestone = Counter(milestone)

    for step in range(total_steps):
        base_lr = base_lr * gamma ** milestone[step]
        lr.append(base_lr)
    return Tensor(np.array(lr), dtype)


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.weight = Parameter(Tensor(np.ones([64, 10]).astype(np.float32)), name="weight")
        self.bias = Parameter(Tensor(np.ones([10]).astype((np.float32))), name="bias")
        self.matmul = P.MatMul()
        self.biasAdd = P.BiasAdd()

    def construct(self, x):
        x = self.biasAdd(self.matmul(x, self.weight), self.bias)
        return x


def test_lars_multi_step_lr():
    inputs = Tensor(np.ones([1, 64]).astype(np.float32))
    label = Tensor(np.zeros([1, 10]).astype(np.float32))
    net = Net()
    net.set_train()
    loss = nn.SoftmaxCrossEntropyWithLogits()

    lr = multisteplr(10, [2, 6])
    SGD = Momentum(net.trainable_params(), lr, 0.9)
    optimizer = LARS(SGD, epsilon=1e-08, coefficient=0.02, use_clip=True,
                     lars_filter=lambda x: 'bn' not in x.name)

    net_with_loss = WithLossCell(net, loss)
    train_network = TrainOneStepCell(net_with_loss, optimizer)
    _cell_graph_executor.compile(train_network, inputs, label)


def test_lars_float_lr():
    inputs = Tensor(np.ones([1, 64]).astype(np.float32))
    label = Tensor(np.zeros([1, 10]).astype(np.float32))
    net = Net()
    net.set_train()
    loss = nn.SoftmaxCrossEntropyWithLogits()

    lr = 0.1
    SGD = Momentum(net.trainable_params(), lr, 0.9)
    optimizer = LARS(SGD, epsilon=1e-08, coefficient=0.02,
                     lars_filter=lambda x: 'bn' not in x.name)

    net_with_loss = WithLossCell(net, loss)
    train_network = TrainOneStepCell(net_with_loss, optimizer)
    _cell_graph_executor.compile(train_network, inputs, label)
