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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pytest

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.nn import TrainOneStepCell, WithLossCell
from mindspore.nn.optim import Momentum

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")


class AlexNet(nn.Cell):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.batch_size = 32
        self.conv1 = nn.Conv2d(3, 96, 11, stride=4, pad_mode="valid")
        self.conv2 = nn.Conv2d(96, 256, 5, stride=1, pad_mode="same")
        self.conv3 = nn.Conv2d(256, 384, 3, stride=1, pad_mode="same")
        self.conv4 = nn.Conv2d(384, 384, 3, stride=1, pad_mode="same")
        self.conv5 = nn.Conv2d(384, 256, 3, stride=1, pad_mode="same")
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="valid")
        self.flatten = nn.Flatten()
        self.fc1 = nn.Dense(6 * 6 * 256, 4096)
        self.fc2 = nn.Dense(4096, 4096)
        self.fc3 = nn.Dense(4096, num_classes)

    def construct(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_trainTensor(num_classes=10, epoch=15, batch_size=32):
    net = AlexNet(num_classes)
    lr = 0.1
    momentum = 0.9
    optimizer = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), lr, momentum, weight_decay=0.0001)
    criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    net_with_criterion = WithLossCell(net, criterion)
    train_network = TrainOneStepCell(net_with_criterion, optimizer)
    train_network.set_train()
    losses = []
    for i in range(0, epoch):
        data = Tensor(np.ones([batch_size, 3, 227, 227]).astype(np.float32) * 0.01)
        label = Tensor(np.ones([batch_size]).astype(np.int32))
        loss = train_network(data, label).asnumpy()
        losses.append(loss)
    assert losses[-1] < 0.01
