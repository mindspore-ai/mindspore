# Copyright 2022 Huawei Technologies Co., Ltd
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
""" test apply_adam_with_amsgrad """
import os
import pytest
import numpy as np

import mindspore.nn as nn
from mindspore import Tensor, Parameter, context
from mindspore.common.api import _cell_graph_executor
from mindspore.nn import TrainOneStepCell, WithLossCell
from mindspore.nn.optim import Adam
from mindspore.ops import operations as P


class Net(nn.Cell):
    """ Net definition """

    def __init__(self):
        super(Net, self).__init__()
        self.weight = Parameter(Tensor(np.ones([64, 10]).astype(np.float32)), name="weight")
        self.bias = Parameter(Tensor(np.ones([10]).astype((np.float32))), name="bias")
        self.matmul = P.MatMul()
        self.biasadd = P.BiasAdd()

    def construct(self, x):
        x = self.biasadd(self.matmul(x, self.weight), self.bias)
        return x


def test_apply_adam_with_amsgrad_compile():
    """
    Feature: test apply_adam_with_amsgrad compile
    Description: test with graph and pynative mode on gpu and cpu platforms
    Expectation: success
    """
    inputs = Tensor(np.ones([1, 64]).astype(np.float32))
    label = Tensor(np.zeros([1, 10]).astype(np.float32))
    net = Net()
    net.set_train()

    loss = nn.SoftmaxCrossEntropyWithLogits()
    optimizer = Adam(net.trainable_params(), learning_rate=0.1, weight_decay=0.9, use_amsgrad=True)
    net_with_loss = WithLossCell(net, loss)
    train_network = TrainOneStepCell(net_with_loss, optimizer)

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    _cell_graph_executor.compile(train_network, inputs, label)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    _cell_graph_executor.compile(train_network, inputs, label)
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    _cell_graph_executor.compile(train_network, inputs, label)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    _cell_graph_executor.compile(train_network, inputs, label)
    context.set_context(mode=context.GRAPH_MODE)


def test_apply_adam_with_amsgrad_group1():
    """
    Feature: test_apply_adam_with_amsgrad_group_lr_and_weight_decay
    Description: test with graph and pynative mode on gpu and cpu platforms
    Expectation: success
    """
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
    optimizer = Adam(group_params, learning_rate=0.1, use_amsgrad=True)

    train_network = TrainOneStepCell(net_with_loss, optimizer)

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    _cell_graph_executor.compile(train_network, inputs, label)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    _cell_graph_executor.compile(train_network, inputs, label)
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    _cell_graph_executor.compile(train_network, inputs, label)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    _cell_graph_executor.compile(train_network, inputs, label)
    context.set_context(mode=context.GRAPH_MODE)


def test_apply_adam_with_amsgrad_group2():
    """
    Feature: test_apply_adam_with_amsgrad_group_lr_and_weight_decay
    Description: test with graph and pynative mode on gpu and cpu platforms
    Expectation: success
    """
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
    optimizer = nn.Adam(group_params, learning_rate=schedule_lr, use_amsgrad=True)
    train_network = TrainOneStepCell(net_with_loss, optimizer)

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    _cell_graph_executor.compile(train_network, inputs, label)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    _cell_graph_executor.compile(train_network, inputs, label)
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    _cell_graph_executor.compile(train_network, inputs, label)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    _cell_graph_executor.compile(train_network, inputs, label)
    context.set_context(mode=context.GRAPH_MODE)


class NetWithSparseGatherV2(nn.Cell):
    """ Feature: NetWithSparseGatherV2 definition """

    def __init__(self):
        super(NetWithSparseGatherV2, self).__init__()
        self.weight1 = Parameter(Tensor(np.ones([3, 1, 2]).astype(np.float32)), name="weight1")
        self.weight2 = Parameter(Tensor(np.ones([2, 1, 2]).astype((np.float32))), name="weight2")
        self.axis = 0
        self.gather = P.SparseGatherV2()

    def construct(self, indices, label):
        return self.gather(self.weight1, indices, self.axis) + self.weight2


def test_sparse_apply_adam_with_amsgrad():
    """
    Feature: test_sparse_apply_adam_with_amsgrad
    Description: test with graph and pynative mode on gpu and cpu platforms
    Expectation: raise exceptions
    """
    indices = Tensor(np.array([0, 1]).astype(np.int32))
    label = Tensor(np.zeros([2, 1, 2]).astype(np.float32))
    net = NetWithSparseGatherV2()
    net.set_train()

    optimizer = Adam(net.trainable_params(), learning_rate=0.1, loss_scale=1024.0, weight_decay=0.9, use_amsgrad=True)
    train_network = TrainOneStepCell(net, optimizer)

    os.environ['MS_DEV_JIT_SYNTAX_LEVEL'] = '0'
    with pytest.raises(Exception):
        context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
        _cell_graph_executor.compile(train_network, indices, label)
    with pytest.raises(Exception):
        context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
        _cell_graph_executor.compile(train_network, indices, label)
    with pytest.raises(Exception):
        context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
        _cell_graph_executor.compile(train_network, indices, label)
    with pytest.raises(Exception):
        context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
        _cell_graph_executor.compile(train_network, indices, label)
    context.set_context(mode=context.GRAPH_MODE)
    os.environ['MS_DEV_JIT_SYNTAX_LEVEL'] = '2'
