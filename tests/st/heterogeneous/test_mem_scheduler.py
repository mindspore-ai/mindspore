# Copyright 2021 Huawei Technologies Co., Ltd
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

import os
import numpy as np
import pytest

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.nn import TrainOneStepCell, WithLossCell
from mindspore.nn.optim import Momentum
from mindspore.ops import operations as P
from tests.st.networks.models.resnetv1_5 import resnet50

context.set_context(mode=context.GRAPH_MODE, memory_optimize_level='O0')


class LeNet(nn.Cell):
    def __init__(self):
        super(LeNet, self).__init__()
        self.relu = P.ReLU()
        self.batch_size = 32

        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0, has_bias=False, pad_mode='valid')
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0, has_bias=False, pad_mode='valid')
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.reshape = P.Reshape()
        self.fc1 = nn.Dense(400, 120)
        self.fc2 = nn.Dense(120, 84)
        self.fc3 = nn.Dense(84, 10)


    def construct(self, input_x):
        output = self.conv1(input_x)
        output = self.relu(output)
        output = self.pool(output)
        output = self.conv2(output)
        output = self.relu(output)
        output = self.pool(output)
        output = self.reshape(output, (self.batch_size, -1))
        output = self.fc1(output)
        output = self.relu(output)
        output = self.fc2(output)
        output = self.relu(output)
        output = self.fc3(output)
        return output


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_resnet():
    '''
    Feature: MemScheduler
    Description: Test MemScheduler
    Expectation: Run resnet success
    '''
    os.environ['ENABLE_MEM_SCHEDULER'] = '1'
    num_classes = 10
    epoch = 8
    batch_size = 1
    net = resnet50(batch_size, num_classes)
    lr = 0.1
    momentum = 0.9
    optimizer = Momentum(filter(lambda x: x.requires_grad,
                                net.get_parameters()), lr, momentum)
    criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    net_with_criterion = WithLossCell(net, criterion)
    train_network = TrainOneStepCell(
        net_with_criterion, optimizer)  # optimizer
    train_network.set_train()
    losses = []
    for _ in range(0, epoch):
        data = Tensor(np.ones([batch_size, 3, 224, 224]
                              ).astype(np.float32) * 0.01)
        label = Tensor(np.ones([batch_size]).astype(np.int32))
        loss = train_network(data, label)
        losses.append(loss)
    assert losses[-1].asnumpy() < 1
    os.environ['ENABLE_MEM_SCHEDULER'] = '0'


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_lenet():
    '''
    Feature: MemScheduler
    Description: Test MemScheduler
    Expectation: Run lenet success
    '''
    os.environ['ENABLE_MEM_SCHEDULER'] = '1'
    data = Tensor(np.ones([32, 1, 32, 32]).astype(np.float32) * 0.01)
    label = Tensor(np.ones([32]).astype(np.int32))
    net = LeNet()
    learning_rate = 0.01
    momentum = 0.9

    optimizer = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), learning_rate, momentum)
    criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True)
    net_with_criterion = WithLossCell(net, criterion)
    train_network = TrainOneStepCell(net_with_criterion, optimizer)  # optimizer
    train_network.set_train()
    res = train_network(data, label)
    diff = res.asnumpy()[0] - 2.3025851
    assert np.all(diff < 1.e-6)
    os.environ['ENABLE_MEM_SCHEDULER'] = '0'


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_lenet_manual_offload():
    '''
    Feature: MemScheduler
    Description: Test set offload strategy
    Expectation: Run lenet success
    '''
    os.environ['ENABLE_MEM_SCHEDULER'] = '1'
    data = Tensor(np.ones([32, 1, 32, 32]).astype(np.float32) * 0.01)
    label = Tensor(np.ones([32]).astype(np.int32))
    net = LeNet()
    net.relu.add_prim_attr("Offload", True)
    learning_rate = 0.01
    momentum = 0.9

    optimizer = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), learning_rate, momentum)
    criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True)
    net_with_criterion = WithLossCell(net, criterion)
    train_network = TrainOneStepCell(net_with_criterion, optimizer)  # optimizer
    train_network.set_train()
    res = train_network(data, label)
    diff = res.asnumpy()[0] - 2.3025851
    assert np.all(diff < 1.e-6)
    os.environ['ENABLE_MEM_SCHEDULER'] = '0'


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_1024_batch_size_resnet():
    """
    Feature: Memory offload
    Description: Test memory offload.
    Expectation: Run resnet with 1024 batch size successfully.
    """
    os.environ['GRAPH_OP_RUN'] = '1'
    num_classes = 10
    epoch = 6
    batch_size = 1024
    context.set_context(memory_offload='ON')
    net = resnet50(batch_size, num_classes)
    lr = 0.1
    momentum = 0.9
    optimizer = Momentum(filter(lambda x: x.requires_grad,
                                net.get_parameters()), lr, momentum)
    criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    net_with_criterion = WithLossCell(net, criterion)
    train_network = TrainOneStepCell(net_with_criterion, optimizer)  # optimizer
    train_network.set_train()
    losses = []
    for _ in range(0, epoch):
        data = Tensor(np.ones([batch_size, 3, 224, 224]
                              ).astype(np.float32) * 0.01)
        label = Tensor(np.ones([batch_size]).astype(np.int32))
        loss = train_network(data, label)
        losses.append(loss)
    assert losses[-1].asnumpy() < 1.5
    os.environ['GRAPH_OP_RUN'] = '0'
