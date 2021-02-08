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
from mindspore import Tensor, Parameter, context
from mindspore.common.api import _executor
from mindspore.nn import TrainOneStepCell, WithLossCell
from mindspore.nn.optim import Adam, AdamWeightDecay
from mindspore.ops import operations as P

@pytest.fixture(scope="module", autouse=True)
def setup_teardown():
    context.set_context(enable_sparse=True)
    yield
    context.set_context(enable_sparse=False)

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


class NetWithSparseGatherV2(nn.Cell):
    """ NetWithSparseGatherV2 definition """
    def __init__(self):
        super(NetWithSparseGatherV2, self).__init__()
        self.weight1 = Parameter(Tensor(np.ones([3, 1, 2]).astype(np.float32)), name="weight1")
        self.weight2 = Parameter(Tensor(np.ones([2, 1, 2]).astype((np.float32))), name="weight2")
        self.axis = 0
        self.gather = P.SparseGatherV2()

    def construct(self, indices, label):
        return self.gather(self.weight1, indices, self.axis) + self.weight2


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


def test_adam_compile():
    """ test adam compile """
    inputs = Tensor(np.ones([1, 64]).astype(np.float32))
    label = Tensor(np.zeros([1, 10]).astype(np.float32))
    net = Net()
    net.set_train()

    loss = nn.SoftmaxCrossEntropyWithLogits()
    optimizer = Adam(net.trainable_params(), learning_rate=0.1, weight_decay=0.9)

    net_with_loss = WithLossCell(net, loss)
    train_network = TrainOneStepCell(net_with_loss, optimizer)
    _executor.compile(train_network, inputs, label)


def test_sparse_adam_compile():
    """ test_sparse_adam_compile """
    indices = Tensor(np.array([0, 1]).astype(np.int32))
    label = Tensor(np.zeros([2, 1, 2]).astype(np.float32))
    net = NetWithSparseGatherV2()
    net.set_train()

    optimizer = Adam(net.trainable_params(), learning_rate=0.1, loss_scale=1024.0, weight_decay=0.9)
    optimizer.target = 'CPU'
    train_network = TrainOneStepCell(net, optimizer)
    _executor.compile(train_network, indices, label)


def test_sparse_adam():
    """ test_sparse_adam """
    indices = Tensor(np.array([0, 1]).astype(np.int32))
    label = Tensor(np.zeros([2, 1, 2]).astype(np.float32))
    net = NetWithSparseGatherV2()
    net.set_train()

    optimizer = Adam(net.trainable_params(), learning_rate=0.1, loss_scale=1024.0, weight_decay=0.9)
    train_network = TrainOneStepCell(net, optimizer)
    _executor.compile(train_network, indices, label)


def test_adam_group1():
    """ test_adam_group_lr_and_weight_decay """
    inputs = Tensor(np.ones([1, 64]).astype(np.float32))
    label = Tensor(np.zeros([1, 10]).astype(np.float32))
    net = Net()
    net.set_train()
    loss = nn.SoftmaxCrossEntropyWithLogits()
    net_with_loss = WithLossCell(net, loss)
    all_params = net.trainable_params()

    poly_decay_lr = nn.polynomial_decay_lr(0.01, 0.0001, total_step=10, step_per_epoch=1, decay_epoch=3, power=1.0)

    group_params = [{'params': [all_params[0]], 'lr': poly_decay_lr, 'weight_decay': 0.9},
                    {'params': [all_params[1]]}]
    optimizer = nn.Adam(group_params, learning_rate=0.1)

    train_network = TrainOneStepCell(net_with_loss, optimizer)
    _executor.compile(train_network, inputs, label)


def test_adam_group2():
    """ test_adam_group_lr_and_weight_decay """
    inputs = Tensor(np.ones([1, 64]).astype(np.float32))
    label = Tensor(np.zeros([1, 10]).astype(np.float32))
    net = Net()
    net.set_train()
    loss = nn.SoftmaxCrossEntropyWithLogits()
    net_with_loss = WithLossCell(net, loss)
    all_params = net.trainable_params()

    schedule_lr = nn.PolynomialDecayLR(0.01, 0.0001, 3, power=1.0)
    group_params = [{'params': [all_params[0]], 'lr': 0.02, 'weight_decay': 0.9},
                    {'params': [all_params[1]]}]
    optimizer = nn.Adam(group_params, learning_rate=schedule_lr)
    train_network = TrainOneStepCell(net_with_loss, optimizer)
    _executor.compile(train_network, inputs, label)


def test_adamweightdecay_group():
    """ test_adam_group_lr_and_weight_decay """
    inputs = Tensor(np.ones([1, 64]).astype(np.float32))
    label = Tensor(np.zeros([1, 10]).astype(np.float32))
    net = Net()
    net.set_train()
    loss = nn.SoftmaxCrossEntropyWithLogits()
    net_with_loss = WithLossCell(net, loss)
    all_params = net.trainable_params()

    schedule_lr = nn.PolynomialDecayLR(0.01, 0.0001, 3, power=1.0)
    group_params = [{'params': [all_params[0]], 'lr': 0.02, 'weight_decay': 0.9},
                    {'params': [all_params[1]]}]
    optimizer = nn.AdamWeightDecay(group_params, learning_rate=schedule_lr)
    train_network = TrainOneStepCell(net_with_loss, optimizer)
    _executor.compile(train_network, inputs, label)


def test_adamoffload_group():
    """ test_adam_group_lr_and_weight_decay """
    inputs = Tensor(np.ones([1, 64]).astype(np.float32))
    label = Tensor(np.zeros([1, 10]).astype(np.float32))
    net = Net()
    net.set_train()
    loss = nn.SoftmaxCrossEntropyWithLogits()
    net_with_loss = WithLossCell(net, loss)
    all_params = net.trainable_params()

    schedule_lr = nn.PolynomialDecayLR(0.01, 0.0001, 3, power=1.0)
    group_params = [{'params': [all_params[0]], 'lr': 0.02, 'weight_decay': 0.9},
                    {'params': [all_params[1]]}]
    optimizer = nn.AdamOffload(group_params, learning_rate=schedule_lr)
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


def test_adam_mindspore_with_empty_params():
    net = nn.Flatten()
    with pytest.raises(ValueError, match=r"Optimizer got an empty parameter list"):
        AdamWeightDecay(net.get_parameters())
