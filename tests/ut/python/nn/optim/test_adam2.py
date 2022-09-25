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
import pytest

import mindspore.nn as nn
from mindspore import Tensor, Parameter, context
from mindspore.common.api import _cell_graph_executor
from mindspore.nn import TrainOneStepCell, WithLossCell
from mindspore.nn.optim import Adam
from mindspore.ops import operations as P

context.set_context(mode=context.GRAPH_MODE)


class Net(nn.Cell):
    """ Net definition """

    def __init__(self):
        super(Net, self).__init__()
        self.weight = Parameter(Tensor(np.ones([64, 10]).astype(np.float32)), name="weight")
        self.bias = Parameter(Tensor(np.ones([10]).astype(np.float32)), name="bias")
        self.matmul = P.MatMul()
        self.bias_add = P.BiasAdd()

    def construct(self, x):
        x = self.bias_add(self.matmul(x, self.weight), self.bias)
        return x


def test_adam_offload_group():
    """
    Feature: Adam optimizer
    Description: Verify AdamOffload
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
    optimizer = nn.Adam(group_params, learning_rate=schedule_lr, use_offload=True)
    train_network = TrainOneStepCell(net_with_loss, optimizer)
    _cell_graph_executor.compile(train_network, inputs, label)


def test_adam_with_both_lazy_and_offload():
    """
    Feature: Adam optimizer
    Description: Verify if the error message is correct
    Expectation: success
    """
    net = Net()
    with pytest.raises(ValueError, match=r"For 'Adam', 'use_lazy' and 'use_offload' can not both be True"):
        Adam(net.trainable_params(), use_lazy=True, use_offload=True)


def test_lazy_with_amsgrad():
    """
    Feature: Adam optimizer with lazy=True
    Description: Verify if the error message is correct
    Expectation: success
    """
    net = Net()
    with pytest.raises(ValueError, match=r"For lazy Adam and Adam with offload, there is no parameter named "
                                         r"'use_amsgrad'."):
        Adam(net.trainable_params(), use_lazy=True, use_amsgrad=True)


def test_offload_with_amsgrad():
    """
    Feature: Adam optimizer with offload=True
    Description: Verify if the error message is correct
    Expectation: success
    """
    net = Net()
    with pytest.raises(ValueError, match=r"For lazy Adam and Adam with offload, there is no parameter named "
                                         r"'use_amsgrad'."):
        Adam(net.trainable_params(), use_offload=True, use_amsgrad=True)
