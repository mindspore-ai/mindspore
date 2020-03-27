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

import pytest
import numpy as np
import mindspore.nn as nn
from mindspore.nn.optim import Momentum
from mindspore.ops import operations as P
from mindspore.nn import TrainOneStepCell, WithLossCell
from mindspore.nn import Dense
from mindspore import Tensor
from mindspore.common.initializer import initializer
from mindspore.common import dtype as mstype
import mindspore.context as context

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
class LeNet(nn.Cell):
    def __init__(self):
        super(LeNet, self).__init__()
        self.relu = P.ReLU()
        self.batch_size = 1
        weight1 = Tensor(np.ones([6, 3, 5, 5]).astype(np.float32) * 0.01)
        weight2 = Tensor(np.ones([16, 6, 5, 5]).astype(np.float32) * 0.01)
        self.conv1 = nn.Conv2d(3, 6, (5, 5), weight_init=weight1, stride=1, padding=0, pad_mode='valid')
        self.conv2 = nn.Conv2d(6, 16, (5, 5), weight_init=weight2, pad_mode='valid', stride=1, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode="valid")

        self.reshape = P.Reshape()
        self.reshape1 = P.Reshape()

        self.fc1 = Dense(400, 120)
        self.fc2 = Dense(120, 84)
        self.fc3 = Dense(84, 10)

    def construct(self, input_x):
        output = self.conv1(input_x)
        output = self.relu(output)
        output = self.pool(output)
        output = self.conv2(output)
        output = self.relu(output)
        output = self.pool(output)
        output = self.reshape(output, (self.batch_size, -1))
        output = self.fc1(output)
        output = self.fc2(output)
        output = self.fc3(output)
        return output


def multisteplr(total_steps, gap, base_lr=0.9, gamma=0.1, dtype=mstype.float32):
    lr = []
    for step in range(total_steps):
        lr_ = base_lr * gamma ** (step//gap)
        lr.append(lr_)
    return Tensor(np.array(lr), dtype)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_train_lenet():
    epoch = 100
    net = LeNet()
    momentum = initializer(Tensor(np.array([0.9]).astype(np.float32)), [1])
    learning_rate = multisteplr(epoch, 30)

    optimizer = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), learning_rate, momentum)
    criterion = nn.SoftmaxCrossEntropyWithLogits(is_grad=False, sparse=True)
    net_with_criterion = WithLossCell(net, criterion)
    train_network = TrainOneStepCell(net_with_criterion, optimizer)  # optimizer
    train_network.set_train()
    losses = []
    for i in range(epoch):
        data = Tensor(np.ones([net.batch_size, 3 ,32, 32]).astype(np.float32) * 0.01)
        label = Tensor(np.ones([net.batch_size]).astype(np.int32))
        loss = train_network(data, label)
        losses.append(loss)
    print(losses)
