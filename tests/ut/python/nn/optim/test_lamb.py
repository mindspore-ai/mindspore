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
""" test lamb """
import numpy as np
import pytest

import mindspore.nn as nn
from mindspore import Tensor, Parameter
from mindspore.common.api import _executor
from mindspore.nn import TrainOneStepCell, WithLossCell
from mindspore.nn.optim import Lamb
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
    """ NetWithoutWeight definition """

    def __init__(self):
        super(NetWithoutWeight, self).__init__()
        self.matmul = P.MatMul()

    def construct(self, x):
        x = self.matmul(x, x)
        return x


def test_lamb_compile():
    """ test_Lamb_compile """
    inputs = Tensor(np.ones([1, 64]).astype(np.float32))
    label = Tensor(np.zeros([1, 10]).astype(np.float32))
    net = Net()
    net.set_train()
    loss = nn.SoftmaxCrossEntropyWithLogits()
    optimizer = Lamb(net.trainable_params(), decay_steps=10)

    net_with_loss = WithLossCell(net, loss)
    train_network = TrainOneStepCell(net_with_loss, optimizer)
    _executor.compile(train_network, inputs, label)


def test_lamb_error():
    net = Net()
    with pytest.raises(TypeError):
        Lamb(net.get_parameters(), decay_steps=6, warmup_steps=5.0)

    with pytest.raises(TypeError):
        Lamb(net.get_parameters(), decay_steps=1.0)

    with pytest.raises(ValueError):
        Lamb(net.get_parameters(), decay_steps=0)
