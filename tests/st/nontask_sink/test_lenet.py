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
import os
import time
import numpy as np
import pytest

import mindspore.nn as nn
from mindspore import context, Tensor, ParameterTuple
from mindspore.common import dtype as mstype
from mindspore.common.initializer import TruncatedNormal
from mindspore.nn.optim import Momentum
from mindspore.nn.wrap.cell_wrapper import WithLossCell
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.ops import operations as P

np.random.seed(1)
grad_by_list = C.GradOperation(get_by_list=True)


def weight_variable():
    """weight initial"""
    return TruncatedNormal(0.02)


def conv(in_channels, out_channels, kernel_size, stride=1, padding=0):
    """weight initial for conv layer"""
    weight = weight_variable()
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=kernel_size, stride=stride, padding=padding,
                     weight_init=weight, has_bias=False, pad_mode="valid")


def fc_with_initialize(input_channels, out_channels):
    """weight initial for fc layer"""
    weight = weight_variable()
    bias = weight_variable()
    return nn.Dense(input_channels, out_channels, weight, bias)


class LeNet(nn.Cell):
    """
    Lenet network
    Args:
        num_class (int): Num classes, Default: 10.
    Returns:
        Tensor, output tensor
    Examples:
        >>> LeNet(num_class=10)
    """

    def __init__(self, num_class=10):
        super(LeNet, self).__init__()
        self.num_class = num_class
        self.batch_size = 32
        self.conv1 = conv(1, 6, 5)
        self.conv2 = conv(6, 16, 5)
        self.fc1 = fc_with_initialize(16 * 5 * 5, 120)
        self.fc2 = fc_with_initialize(120, 84)
        self.fc3 = fc_with_initialize(84, self.num_class)
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.reshape = P.Reshape()

    def construct(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.reshape(x, (self.batch_size, -1))
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


class CrossEntropyLoss(nn.Cell):
    """
    Define loss for network
    """

    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.cross_entropy = P.SoftmaxCrossEntropyWithLogits()
        self.mean = P.ReduceMean()
        self.one_hot = P.OneHot()
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)
        self.num = Tensor(32.0, mstype.float32)

    def construct(self, logits, label):
        label = self.one_hot(label, F.shape(logits)[1], self.on_value, self.off_value)
        loss = self.cross_entropy(logits, label)[0]
        loss = P.RealDiv()(P.ReduceSum()(loss, -1), self.num)
        return loss


class GradWrap(nn.Cell):
    """
    GradWrap definition
    """

    def __init__(self, network):
        super(GradWrap, self).__init__()
        self.network = network
        self.weights = ParameterTuple(filter(lambda x: x.requires_grad, network.get_parameters()))

    def construct(self, x, label):
        weights = self.weights
        return grad_by_list(self.network, weights)(x, label)


def test_ascend_lenet():
    epoch_size = 20
    batch_size = 32
    inputs = Tensor(np.ones([batch_size, 1, 32, 32]).astype(np.float32))
    labels = Tensor(np.ones([batch_size]).astype(np.int32))

    net = LeNet()
    criterion = CrossEntropyLoss()
    optimizer = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), 0.1, 0.9)

    net_with_criterion = WithLossCell(net, criterion)
    train_network = GradWrap(net_with_criterion)
    train_network.set_train()
    total_time = 0

    for epoch in range(0, epoch_size):
        start_time = time.time()
        fw_output = net(inputs)
        loss_output = criterion(fw_output, labels)
        grads = train_network(inputs, labels)
        optimizer(grads)
        end_time = time.time()
        cost_time = end_time - start_time
        total_time = total_time + cost_time

        print("======epoch: ", epoch, " loss: ", loss_output.asnumpy(), " cost time: ", cost_time)
    return loss_output


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_ascend_lenet1():
    os.environ['GRAPH_OP_RUN'] = str(1)
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    loss_output = test_ascend_lenet()
    assert loss_output.asnumpy() < 0.004
    assert loss_output.asnumpy() > 0.003

@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_ascend_lenet2():
    os.environ['GRAPH_OP_RUN'] = str(1)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    loss_output = test_ascend_lenet()
    assert loss_output.asnumpy() < 0.004
    assert loss_output.asnumpy() > 0.003


class GradWrapTuple(nn.Cell):
    """
    GradWrapTuple definition
    """

    def __init__(self, network):
        super(GradWrapTuple, self).__init__()
        self.network = network
        self.weights = tuple(filter(lambda x: x.requires_grad, network.get_parameters()))

    def construct(self, x, label):
        weights = self.weights
        return grad_by_list(self.network, weights)(x, label)


def test_ascend_lenet_grad_by_list_tuple():
    """
    Feature: GradOperation get_by_list pass tuple/list
    Description: Grad with Parameters as input type and fv. list or tuple as fv of grad.
    Expectation: No exception.
    """
    epoch_size = 20
    batch_size = 32
    inputs = Tensor(np.ones([batch_size, 1, 32, 32]).astype(np.float32))
    labels = Tensor(np.ones([batch_size]).astype(np.int32))

    net = LeNet()
    criterion = CrossEntropyLoss()
    optimizer = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), 0.1, 0.9)

    net_with_criterion = WithLossCell(net, criterion)
    train_network = GradWrapTuple(net_with_criterion)
    train_network.set_train()
    total_time = 0

    for epoch in range(0, epoch_size):
        start_time = time.time()
        fw_output = net(inputs)
        loss_output = criterion(fw_output, labels)
        grads = train_network(inputs, labels)
        optimizer(grads)
        end_time = time.time()
        cost_time = end_time - start_time
        total_time = total_time + cost_time

        print("======epoch: ", epoch, " loss: ", loss_output.asnumpy(), " cost time: ", cost_time)
    return loss_output


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_ascend_lenet_grad_by_list_tuple1():
    """
    Feature: GradOperation get_by_list pass tuple/list
    Description: Grad with Parameters as input type and fv. list or tuple as fv of grad.
    Expectation: No exception.
    """
    os.environ['GRAPH_OP_RUN'] = str(1)
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    loss_output = test_ascend_lenet_grad_by_list_tuple()
    assert loss_output.asnumpy() < 0.004
    assert loss_output.asnumpy() > 0.003
