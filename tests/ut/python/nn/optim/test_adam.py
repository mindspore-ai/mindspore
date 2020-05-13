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
from mindspore import Tensor, Parameter
from mindspore.common.api import _executor
from mindspore.nn import TrainOneStepCell, WithLossCell
from mindspore.nn.optim import AdamWeightDecay, AdamWeightDecayDynamicLR
from mindspore.ops import operations as P


class Net(nn.Cell):
    """ Net definition """

    def __init__(self):
        super(Net, self).__init__()
        self.weight = Parameter(Tensor(np.ones([64, 10]).astype(np.float32)), name="weight")
        self.bias = Parameter(Tensor(np.ones([10]).astype((np.float32))), name="bias")
        self.matmul = P.MatMul()
        self.biasAdd = P.BiasAdd()

    def construct(self, x):
        x = self.biasAdd(self.matmul(x, self.weight), self.bias)
        return x


class NetWithoutWeight(nn.Cell):
    def __init__(self):
        super(NetWithoutWeight, self).__init__()
        self.matmul = P.MatMul()

    def construct(self, x):
        x = self.matmul(x, x)
        return x


def test_adamwithoutparam():
    net = NetWithoutWeight()
    net.set_train()
    with pytest.raises(ValueError, match=r"Optimizer got an empty parameter list"):
        AdamWeightDecay(net.trainable_params(), learning_rate=0.1)


def test_adamw_compile():
    """ test_adamw_compile """
    inputs = Tensor(np.ones([1, 64]).astype(np.float32))
    label = Tensor(np.zeros([1, 10]).astype(np.float32))
    net = Net()
    net.set_train()

    loss = nn.SoftmaxCrossEntropyWithLogits()
    optimizer = AdamWeightDecay(net.trainable_params(), learning_rate=0.1)

    net_with_loss = WithLossCell(net, loss)
    train_network = TrainOneStepCell(net_with_loss, optimizer)
    _executor.compile(train_network, inputs, label)


def test_AdamWeightDecay_beta1():
    net = Net()
    print("**********", net.get_parameters())
    with pytest.raises(ValueError):
        AdamWeightDecay(net.get_parameters(), beta1=1.0, learning_rate=0.1)


def test_AdamWeightDecay_beta2():
    net = Net()
    with pytest.raises(ValueError):
        AdamWeightDecay(net.get_parameters(), beta2=1.0, learning_rate=0.1)


def test_AdamWeightDecay_e():
    net = Net()
    with pytest.raises(ValueError):
        AdamWeightDecay(net.get_parameters(), eps=-0.1, learning_rate=0.1)


def test_AdamWeightDecayDynamicLR():
    """ test_AdamWeightDecayDynamicLR """
    inputs = Tensor(np.ones([1, 64]).astype(np.float32))
    label = Tensor(np.zeros([1, 10]).astype(np.float32))
    net = Net()
    net.set_train()
    loss = nn.SoftmaxCrossEntropyWithLogits()
    optimizer = AdamWeightDecayDynamicLR(net.trainable_params(), decay_steps=20, learning_rate=0.1)

    net_with_loss = WithLossCell(net, loss)
    train_network = TrainOneStepCell(net_with_loss, optimizer)
    _executor.compile(train_network, inputs, label)


def test_adam_mindspore_flatten():
    net = nn.Flatten()
    with pytest.raises(ValueError, match=r"Optimizer got an empty parameter list"):
        AdamWeightDecay(net.get_parameters())
