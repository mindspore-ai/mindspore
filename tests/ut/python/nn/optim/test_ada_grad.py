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
""" test ADA_GRAD """

import pytest
import numpy as np

import mindspore.nn as nn
from mindspore import Tensor, Parameter, context
from mindspore.common.api import _executor
from mindspore.nn import TrainOneStepCell, WithLossCell
from mindspore.nn.optim import Adagrad
from mindspore.ops import operations as P


@pytest.fixture(scope="module", autouse=True)
def setup_teardown():
    context.set_context(enable_sparse=True)
    yield
    context.set_context(enable_sparse=False)


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.weight = Parameter(Tensor(np.ones([64, 10]).astype(np.float32)), name='weight')
        self.bias = Parameter(Tensor(np.ones([10]).astype(np.float32)), name='bias')
        self.matmul = P.MatMul()
        self.biasAdd = P.BiasAdd()

    def construct(self, x):
        x = self.biasAdd(self.matmul(x, self.weight), self.bias)
        return x


def test_ada_grad():
    """ test_ada_grad """
    inputs = Tensor(np.ones([1, 64]).astype(np.float32))
    label = Tensor(np.zeros([1, 10]).astype(np.float32))
    net = Net()
    net.set_train()
    loss = nn.SoftmaxCrossEntropyWithLogits()
    optimizer = Adagrad(net.trainable_params(), weight_decay=0.9, loss_scale=1024.0)
    net_with_loss = WithLossCell(net, loss)
    train_network = TrainOneStepCell(net_with_loss, optimizer)
    _executor.compile(train_network, inputs, label)
