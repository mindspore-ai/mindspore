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
""" test adafactor """
import numpy as np

import mindspore.nn as nn
from mindspore import Tensor, Parameter, context
from mindspore.common.api import _cell_graph_executor
from mindspore.nn import TrainOneStepCell, WithLossCell
from mindspore.nn.optim.adafactor import AdaFactor
from mindspore.ops import operations as P


context.set_context(mode=context.GRAPH_MODE)


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


def test_adafactor_compile1():
    """ test adafactor compile """
    inputs = Tensor(np.ones([1, 64]).astype(np.float32))
    label = Tensor(np.zeros([1, 10]).astype(np.float32))
    net = Net()
    net.set_train()

    loss = nn.SoftmaxCrossEntropyWithLogits()
    optimizer = AdaFactor(net.trainable_params(), learning_rate=0.1, weight_decay=0.9, relative_step=False)

    net_with_loss = WithLossCell(net, loss)
    train_network = TrainOneStepCell(net_with_loss, optimizer)
    _cell_graph_executor.compile(train_network, inputs, label)


def test_adafactor_compile2():
    """ test adafactor compile """
    inputs = Tensor(np.ones([1, 64]).astype(np.float32))
    label = Tensor(np.zeros([1, 10]).astype(np.float32))
    net = Net()
    net.set_train()

    loss = nn.SoftmaxCrossEntropyWithLogits()
    optimizer = AdaFactor(net.trainable_params(), learning_rate=None, weight_decay=0.9)

    net_with_loss = WithLossCell(net, loss)
    train_network = TrainOneStepCell(net_with_loss, optimizer)
    _cell_graph_executor.compile(train_network, inputs, label)


def test_adafactor_compile3():
    """ test adafactor compile """
    inputs = Tensor(np.ones([1, 64]).astype(np.float32))
    label = Tensor(np.zeros([1, 10]).astype(np.float32))
    net = Net()
    net.set_train()

    loss = nn.SoftmaxCrossEntropyWithLogits()
    optimizer = AdaFactor(net.trainable_params(), learning_rate=None, weight_decay=0.9,
                          scale_parameter=True, relative_step=True,
                          warmup_init=False, compression=False)
    net_with_loss = WithLossCell(net, loss)
    train_network = TrainOneStepCell(net_with_loss, optimizer)
    _cell_graph_executor.compile(train_network, inputs, label)


def test_adafactor_compile4():
    """ test adafactor compile """
    inputs = Tensor(np.ones([1, 64]).astype(np.float32))
    label = Tensor(np.zeros([1, 10]).astype(np.float32))
    net = Net()
    net.set_train()

    loss = nn.SoftmaxCrossEntropyWithLogits()
    scale_parameter = False
    relative_step = True
    warmup_init = False
    compression = False
    optimizer = AdaFactor(net.trainable_params(), learning_rate=None, weight_decay=0.9,
                          scale_parameter=scale_parameter, relative_step=relative_step,
                          warmup_init=warmup_init, compression=compression)
    net_with_loss = WithLossCell(net, loss)
    train_network = TrainOneStepCell(net_with_loss, optimizer)
    _cell_graph_executor.compile(train_network, inputs, label)


def test_adafactor_compile5():
    """ test adafactor compile """
    inputs = Tensor(np.ones([1, 64]).astype(np.float32))
    label = Tensor(np.zeros([1, 10]).astype(np.float32))
    net = Net()
    net.set_train()

    loss = nn.SoftmaxCrossEntropyWithLogits()
    scale_parameter = False
    relative_step = True
    warmup_init = True
    compression = True
    optimizer = AdaFactor(net.trainable_params(), learning_rate=None, weight_decay=0.9,
                          scale_parameter=scale_parameter, relative_step=relative_step,
                          warmup_init=warmup_init, compression=compression)
    net_with_loss = WithLossCell(net, loss)
    train_network = TrainOneStepCell(net_with_loss, optimizer)
    _cell_graph_executor.compile(train_network, inputs, label)


def test_adafactor_compile6():
    """ test adafactor compile """
    inputs = Tensor(np.ones([1, 64]).astype(np.float32))
    label = Tensor(np.zeros([1, 10]).astype(np.float32))
    net = Net()
    net.set_train()

    loss = nn.SoftmaxCrossEntropyWithLogits()
    scale_parameter = True
    relative_step = True
    warmup_init = True
    compression = True
    optimizer = AdaFactor(net.trainable_params(), learning_rate=None, weight_decay=0.9,
                          scale_parameter=scale_parameter, relative_step=relative_step,
                          warmup_init=warmup_init, compression=compression)
    net_with_loss = WithLossCell(net, loss)
    train_network = TrainOneStepCell(net_with_loss, optimizer)
    _cell_graph_executor.compile(train_network, inputs, label)


def test_adafactor_group1():
    """ test_adafactor_group_lr_and_weight_decay """
    inputs = Tensor(np.ones([1, 64]).astype(np.float32))
    label = Tensor(np.zeros([1, 10]).astype(np.float32))
    net = Net()
    net.set_train()
    loss = nn.SoftmaxCrossEntropyWithLogits()
    net_with_loss = WithLossCell(net, loss)
    all_params = net.trainable_params()

    poly_decay_lr = nn.polynomial_decay_lr(0.01, 0.0001, total_step=10, step_per_epoch=1, decay_epoch=3, power=1.0)

    group_params = [{'params': [all_params[0]]}, {'params': [all_params[1]]}]
    optimizer = AdaFactor(group_params, learning_rate=poly_decay_lr, relative_step=False)

    train_network = TrainOneStepCell(net_with_loss, optimizer)
    _cell_graph_executor.compile(train_network, inputs, label)


def test_adafactor_group2():
    """ test_adafactor_group_lr_and_weight_decay """
    inputs = Tensor(np.ones([1, 64]).astype(np.float32))
    label = Tensor(np.zeros([1, 10]).astype(np.float32))
    net = Net()
    net.set_train()
    loss = nn.SoftmaxCrossEntropyWithLogits()
    net_with_loss = WithLossCell(net, loss)
    all_params = net.trainable_params()

    schedule_lr = nn.PolynomialDecayLR(0.01, 0.0001, 3, power=1.0)
    group_params = [{'params': [all_params[0]]},
                    {'params': [all_params[1]]}]
    optimizer = AdaFactor(group_params, learning_rate=schedule_lr, relative_step=False)
    train_network = TrainOneStepCell(net_with_loss, optimizer)
    _cell_graph_executor.compile(train_network, inputs, label)


def test_adafactor_group3():
    """ test_adafactor_group_lr_and_weight_decay """
    inputs = Tensor(np.ones([1, 64]).astype(np.float32))
    label = Tensor(np.zeros([1, 10]).astype(np.float32))
    net = Net()
    net.set_train()
    loss = nn.SoftmaxCrossEntropyWithLogits()
    net_with_loss = WithLossCell(net, loss)
    all_params = net.trainable_params()

    group_params = [{'params': [all_params[0]]}, {'params': [all_params[1]]}]
    optimizer = AdaFactor(group_params, learning_rate=None)

    train_network = TrainOneStepCell(net_with_loss, optimizer)
    _cell_graph_executor.compile(train_network, inputs, label)


def test_adafactor_group4():
    """ test_adafactor_group_lr_and_weight_decay """
    inputs = Tensor(np.ones([1, 64]).astype(np.float32))
    label = Tensor(np.zeros([1, 10]).astype(np.float32))
    net = Net()
    net.set_train()
    loss = nn.SoftmaxCrossEntropyWithLogits()
    net_with_loss = WithLossCell(net, loss)
    all_params = net.trainable_params()

    group_params = [{'params': [all_params[0]]},
                    {'params': [all_params[1]]}]
    optimizer = AdaFactor(group_params, learning_rate=None)
    train_network = TrainOneStepCell(net_with_loss, optimizer)
    _cell_graph_executor.compile(train_network, inputs, label)


def test_adafactor_group5():
    """ test_adafactor_group_lr_and_weight_decay """
    inputs = Tensor(np.ones([1, 64]).astype(np.float32))
    label = Tensor(np.zeros([1, 10]).astype(np.float32))
    net = Net()
    net.set_train()
    loss = nn.SoftmaxCrossEntropyWithLogits()
    net_with_loss = WithLossCell(net, loss)
    all_params = net.trainable_params()

    group_params = [{'params': [all_params[0]]},
                    {'params': [all_params[1]]}]
    optimizer = AdaFactor(group_params, learning_rate=None, beta1=0.1)
    train_network = TrainOneStepCell(net_with_loss, optimizer)
    _cell_graph_executor.compile(train_network, inputs, label)


def test_adafactor_group6():
    """ test_adafactor_group_lr_and_weight_decay """
    inputs = Tensor(np.ones([1, 64]).astype(np.float32))
    label = Tensor(np.zeros([1, 10]).astype(np.float32))
    net = Net()
    net.set_train()
    loss = nn.SoftmaxCrossEntropyWithLogits()
    net_with_loss = WithLossCell(net, loss)
    all_params = net.trainable_params()

    group_params = [{'params': [all_params[0]]},
                    {'params': [all_params[1]]}]
    optimizer = AdaFactor(group_params, learning_rate=None, beta1=0.2)
    train_network = TrainOneStepCell(net_with_loss, optimizer)
    _cell_graph_executor.compile(train_network, inputs, label)
