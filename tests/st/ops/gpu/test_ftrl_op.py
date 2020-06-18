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

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.nn import Dense
from mindspore.nn import TrainOneStepCell, WithLossCell
from mindspore.nn.optim import FTRL
from mindspore.ops import operations as P

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")


class NetFtrl(nn.Cell):
    def __init__(self):
        super(NetFtrl, self).__init__()
        self.batch_size = 1
        self.reshape = P.Reshape()
        weight = Tensor(np.ones([10, 16]).astype(np.float32) * 0.01)
        self.fc1 = Dense(16, 10, weight_init=weight)

    def construct(self, input_x):
        output = self.reshape(input_x, (self.batch_size, -1))
        output = self.fc1(output)
        return output


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_ftrl():
    epoch = 3
    net = NetFtrl()
    optimizer = FTRL(filter(lambda x: x.requires_grad,
                            net.get_parameters()), learning_rate=0.01)
    criterion = nn.SoftmaxCrossEntropyWithLogits(is_grad=False, sparse=True)
    net_with_criterion = WithLossCell(net, criterion)
    train_network = TrainOneStepCell(
        net_with_criterion, optimizer)
    train_network.set_train()

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    losses1 = []
    for _ in range(epoch):
        data = Tensor(np.arange(0, 16).reshape(
            1, 1, 4, 4).astype(np.float32) * 0.01)
        label = Tensor(np.array([0]).astype(np.int32))
        loss = train_network(data, label)
        losses1.append(loss.asnumpy())
    assert losses1[0] > losses1[1]
    assert losses1[1] > losses1[2]

    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    losses2 = []
    for _ in range(epoch):
        data = Tensor(np.arange(0, 16).reshape(
            1, 1, 4, 4).astype(np.float32) * 0.01)
        label = Tensor(np.array([0]).astype(np.int32))
        loss = train_network(data, label)
        losses2.append(loss.asnumpy())
    assert losses2[0] > losses2[1]
    assert losses2[1] > losses2[2]
