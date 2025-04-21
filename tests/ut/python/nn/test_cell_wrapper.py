# Copyright 2019 Huawei Technologies Co., Ltd
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
from mindspore import Tensor, Parameter
from mindspore.common import dtype as mstype
from mindspore.common.api import _cell_graph_executor
from mindspore.nn import TrainOneStepCell, WithLossCell, ParameterUpdate
from mindspore.nn.optim import Momentum
from mindspore.ops import operations as P


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


def test_parameter_update_int32_and_tensor():
    """ test_parameter_update """
    net = Net()
    loss = nn.SoftmaxCrossEntropyWithLogits()
    optimizer = Momentum(net.get_parameters(), Tensor(np.array([0.1, 0.01, 0.001]), mstype.float32), 0.001)

    net_with_loss = WithLossCell(net, loss)
    train_network = TrainOneStepCell(net_with_loss, optimizer)

    # compile train graph
    train_network.set_train()
    inputs = Tensor(np.ones([1, 64]).astype(np.float32))
    label = Tensor(np.zeros([1, 10]).astype(np.float32))
    _cell_graph_executor.compile(train_network, inputs, label)

    # test tensor
    param_lr = train_network.parameters_dict()['learning_rate']
    update_network = ParameterUpdate(param_lr)
    update_network.phase = 'update_param'

    input_lr = Tensor(np.array([0.2, 0.02, 0.002]), mstype.float32)
    _cell_graph_executor.compile(update_network, input_lr)

    # test int32
    param_step = train_network.parameters_dict()['global_step']
    update_global_step = ParameterUpdate(param_step)

    input_step = Tensor(np.array([1000]), mstype.int32)
    _cell_graph_executor.compile(update_global_step, input_step)


def test_parameter_update_float32():
    """ test_parameter_update """
    net = Net()
    loss = nn.SoftmaxCrossEntropyWithLogits()
    optimizer = Momentum(net.get_parameters(), 0.01, 0.001)

    net_with_loss = WithLossCell(net, loss)
    train_network = TrainOneStepCell(net_with_loss, optimizer)

    # compile train graph
    train_network.set_train()
    inputs = Tensor(np.ones([1, 64]).astype(np.float32))
    label = Tensor(np.zeros([1, 10]).astype(np.float32))
    _cell_graph_executor.compile(train_network, inputs, label)

    # construct and compile update graph
    param_lr = train_network.parameters_dict()['learning_rate']
    update_network = ParameterUpdate(param_lr)
    update_network.phase = 'update_param'

    input_lr = Tensor(0.0001, mstype.float32)
    _cell_graph_executor.compile(update_network, input_lr)


def test_parameter_update_error():
    """ test_parameter_update """
    input_np = np.array([1])

    with pytest.raises(TypeError):
        ParameterUpdate(input_np)
