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
""" test adam """
import numpy as np
import pytest

import mindspore.nn as nn
from mindspore import Tensor
from mindspore.common.api import _executor
from mindspore.nn import TrainOneStepCell, WithLossCell
from mindspore.nn.optim import Adam, AdamWeightDecay, AdamWeightDecayDynamicLR, Lamb
from mindspore.ops import operations as P
from mindspore.parallel._auto_parallel_context import auto_parallel_context
from mindspore import context


class Net(nn.Cell):
    """Net definition"""

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Dense(128, 768, activation='relu')
        self.fc2 = nn.Dense(128, 768, activation='relu')
        self.fc3 = nn.Dense(128, 768, activation='relu')
        self.fc4 = nn.Dense(768, 768, activation='relu')
        self.relu4 = nn.ReLU()
        self.relu5 = nn.ReLU()
        self.transpose = P.Transpose()
        self.matmul1 = P.MatMul()
        self.matmul2 = P.MatMul()

    def construct(self, x):
        q = self.fc1(x)
        k = self.fc2(x)
        v = self.fc3(x)
        k = self.transpose(k, (1, 0))
        c = self.relu4(self.matmul1(q, k))
        s = self.relu5(self.matmul2(c, v))
        s = self.fc4(s)
        return s


def test_AdamWeightDecayDynamicLR():
    """ test_AdamWeightDecayDynamicLR """
    auto_parallel_context().set_enable_parallel_optimizer(True)
    context.set_auto_parallel_context(parallel_mode="data_parallel", device_num=2)
    inputs = Tensor(np.ones([32, 128]).astype(np.float32))
    label = Tensor(np.zeros([32, 768]).astype(np.float32))
    net = Net()
    net.set_train()
    loss = nn.SoftmaxCrossEntropyWithLogits()
    optimizer = AdamWeightDecayDynamicLR(net.trainable_params(), decay_steps=20, learning_rate=0.1)

    net_with_loss = WithLossCell(net, loss)
    train_network = TrainOneStepCell(net_with_loss, optimizer)
    _executor.compile(train_network, inputs, label)


def test_AdamWeightDecay():
    """ test_AdamWeightDecayDynamicLR """
    auto_parallel_context().set_enable_parallel_optimizer(True)
    context.set_auto_parallel_context(parallel_mode="data_parallel", device_num=2)
    inputs = Tensor(np.ones([32, 128]).astype(np.float32))
    label = Tensor(np.zeros([32, 768]).astype(np.float32))
    net = Net()
    net.set_train()
    loss = nn.SoftmaxCrossEntropyWithLogits()
    optimizer = AdamWeightDecay(net.trainable_params(), learning_rate=0.1)

    net_with_loss = WithLossCell(net, loss)
    train_network = TrainOneStepCell(net_with_loss, optimizer)
    _executor.compile(train_network, inputs, label)


def test_lamb_compile():
    """ test_Lamb_compile """
    auto_parallel_context().set_enable_parallel_optimizer(True)
    context.set_auto_parallel_context(parallel_mode="auto_parallel", device_num=2)
    inputs = Tensor(np.ones([32, 128]).astype(np.float32))
    label = Tensor(np.zeros([32, 768]).astype(np.float32))
    net = Net()
    net.set_train()
    loss = nn.SoftmaxCrossEntropyWithLogits()
    optimizer = Lamb(net.trainable_params(), decay_steps=10)

    net_with_loss = WithLossCell(net, loss)
    train_network = TrainOneStepCell(net_with_loss, optimizer)
    _executor.compile(train_network, inputs, label)


def test_edge_case():
    """ test_edge_case """
    auto_parallel_context().set_enable_parallel_optimizer(True)
    net = Net()
    with pytest.raises(RuntimeError):
        context.set_auto_parallel_context(parallel_mode="stand_alone")
        Lamb(net.trainable_params(), decay_steps=10)
    with pytest.raises(RuntimeError):
        Adam(net.trainable_params(), learning_rate=0.1)
    with pytest.raises(RuntimeError):
        context.set_auto_parallel_context(device_num=16)
        Lamb(net.trainable_params(), decay_steps=10)
