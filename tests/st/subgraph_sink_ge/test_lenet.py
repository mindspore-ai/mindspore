# Copyright 2023 Huawei Technologies Co., Ltd
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
import random
import numpy as np

import mindspore.nn as nn
from mindspore import context, Tensor
from mindspore.common import dtype as mstype
from mindspore.common.initializer import TruncatedNormal
from mindspore.nn.optim import Momentum
from mindspore.nn.wrap.cell_wrapper import WithLossCell, TrainOneStepCell
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore.common import set_seed
from mindspore._extends import cell_attr_register
from tests.mark_utils import arg_mark


def seed_set():
    set_seed(1)
    np.random.seed(1)
    random.seed(1)


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


class LeNetConvBlock(nn.Cell):
    @cell_attr_register
    def __init__(self, in_channels, out_channels, kernel_size):
        super(LeNetConvBlock, self).__init__()
        self.conv = conv(in_channels, out_channels, kernel_size)
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)

    def construct(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        return x


class LeNetFcBlock(nn.Cell):
    @cell_attr_register
    def __init__(self, in_channels, out_channels):
        super(LeNetFcBlock, self).__init__()
        self.fc = fc_with_initialize(in_channels, out_channels)
        self.relu = nn.ReLU()

    def construct(self, x):
        x = self.fc(x)
        x = self.relu(x)
        return x


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
        self.conv1 = LeNetConvBlock(1, 6, 5)
        self.conv2 = LeNetConvBlock(6, 16, 5)
        self.fc1 = LeNetFcBlock(16 * 5 * 5, 120)
        self.fc2 = LeNetFcBlock(120, 84)
        self.fc3 = fc_with_initialize(84, self.num_class)
        self.reshape = P.Reshape()

    def construct(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.reshape(x, (self.batch_size, -1))
        x = self.fc1(x)
        x = self.fc2(x)
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


def train_ascend_lenet():
    epoch_size = 20
    batch_size = 32
    inputs = Tensor(np.ones([batch_size, 1, 32, 32]).astype(np.float32))
    labels = Tensor(np.ones([batch_size]).astype(np.int32))

    net = LeNet()
    criterion = CrossEntropyLoss()
    optimizer = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), 0.1, 0.9)

    net_with_criterion = WithLossCell(net, criterion)
    train_network = TrainOneStepCell(net_with_criterion, optimizer)
    train_network.set_train()
    total_time = 0

    for epoch in range(0, epoch_size):
        start_time = time.time()
        loss = train_network(inputs, labels)
        end_time = time.time()
        cost_time = end_time - start_time
        total_time = total_time + cost_time

        print("======epoch: ", epoch, " loss: ", loss.asnumpy(), " cost time: ", cost_time)
    return loss


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_ascend_lenet_cell():
    """
    Feature: test ge ascend lenet with cell reuse.
    Description: subgraph sink with ge.
    Expectation: the result match with expect
    """
    seed_set()
    os.environ['MS_FORMAT_MODE'] = str(1)
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    loss_output = train_ascend_lenet()
    del os.environ['MS_FORMAT_MODE']
    assert loss_output.asnumpy() < 0.004
    assert loss_output.asnumpy() > 0.003


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_ascend_lenet_no_cell():
    """
    Feature: test ge ascend lenet with no cell reuse.
    Description: multi-graph sink with ge.
    Expectation: the result match with expect
    """
    seed_set()
    os.environ['MS_FORMAT_MODE'] = str(1)
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    loss_output = train_ascend_lenet()
    del os.environ['MS_FORMAT_MODE']
    assert loss_output.asnumpy() < 0.004
    assert loss_output.asnumpy() > 0.003
