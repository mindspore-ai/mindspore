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
""" test adadelta """
import pytest
import numpy as np

import mindspore.nn as nn
from mindspore import Tensor, Parameter, context
from mindspore.common.api import _cell_graph_executor
from mindspore.nn import TrainOneStepCell, WithLossCell
from mindspore.nn.optim import Adadelta
from mindspore.ops import operations as P

context.set_context(mode=context.GRAPH_MODE)
skip_flag = context.get_context("device_target") == "GPU"


class Net(nn.Cell):
    """Net."""

    def __init__(self):
        super(Net, self).__init__()
        self.weight = Parameter(Tensor(np.ones([64, 10]).astype(np.float32)), name='weight')
        self.bias = Parameter(Tensor(np.ones([10]).astype(np.float32)), name='bias')
        self.matmul = P.MatMul()
        self.bias_add = P.BiasAdd()

    def construct(self, x):
        x = self.bias_add(self.matmul(x, self.weight), self.bias)
        return x


@pytest.mark.skipif(skip_flag, reason="not support running in GPU")
def test_adadelta():
    """
    Feature: Test Adadelta.
    Description: Test Adadelta functional.
    Expectation: Success.
    """
    inputs = Tensor(np.ones([1, 64]).astype(np.float32))
    label = Tensor(np.zeros([1, 10]).astype(np.float32))
    net = Net()
    net.set_train()
    loss = nn.SoftmaxCrossEntropyWithLogits()
    optimizer = Adadelta(net.trainable_params(), weight_decay=0.9, loss_scale=1024.0)
    net_with_loss = WithLossCell(net, loss)
    train_network = TrainOneStepCell(net_with_loss, optimizer)
    _cell_graph_executor.compile(train_network, inputs, label)
