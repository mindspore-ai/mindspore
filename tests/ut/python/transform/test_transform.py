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
"""
@File  : test_adapter.py
@Author:
@Date  : 2019-03-20
@Desc  : test mindspore compile method
"""
import logging
import numpy as np

import mindspore.nn as nn
from mindspore import Tensor, Parameter
from mindspore.ops import operations as P

log = logging.getLogger("test")
log.setLevel(level=logging.ERROR)


def conv3x3(in_channels, out_channels, stride=1, padding=1):
    """3x3 convolution """
    weight = Tensor(np.ones([out_channels, in_channels, 3, 3]).astype(np.float32))
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=3, stride=stride,
                     padding=padding, weight_init=weight)


def conv1x1(in_channels, out_channels, stride=1, padding=0):
    """1x1 convolution"""
    weight = Tensor(np.ones([out_channels, in_channels, 1, 1]).astype(np.float32))
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=1, stride=stride,
                     padding=padding, weight_init=weight)


class ResidualBlock(nn.Cell):
    """
    residual Block
    """
    expansion = 4

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 down_sample=False):
        super(ResidualBlock, self).__init__()

        out_chls = out_channels // self.expansion
        self.conv1 = conv1x1(in_channels, out_chls, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_chls)

        self.conv2 = conv3x3(out_chls, out_chls, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_chls)

        self.conv3 = conv1x1(out_chls, out_channels, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU()
        self.downsample = down_sample

        if self.downsample:
            self.conv_down_sample = conv1x1(in_channels, out_channels,
                                            stride=stride, padding=0)
            self.bn_down_sample = nn.BatchNorm2d(out_channels)
        self.add = P.Add()

    def construct(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample:
            identity = self.conv_down_sample(identity)
            identity = self.bn_down_sample(identity)

        out = self.add(out, identity)
        out = self.relu(out)

        return out


class ResNet(nn.Cell):
    """ ResNet definition """

    def __init__(self, tensor):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.weight = Parameter(tensor, name='w')

    def construct(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        return x


class LeNet(nn.Cell):
    """ LeNet definition """

    def __init__(self):
        super(LeNet, self).__init__()
        self.relu = nn.ReLU()
        weight1 = Tensor(np.ones([6, 1, 5, 5]).astype(np.float32) * 0.01)
        weight2 = Tensor(np.ones([16, 6, 5, 5]).astype(np.float32) * 0.01)
        self.conv1 = nn.Conv2d(1, 6, (5, 5), weight_init=weight1, stride=1, padding=0, pad_mode='valid')
        self.conv2 = nn.Conv2d(6, 16, (5, 5), weight_init=weight2, pad_mode='valid')
        self.pool = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        fcweight1 = Tensor(np.ones([120, 16 * 5 * 5]).astype(np.float32) * 0.01)
        fcweight2 = Tensor(np.ones([84, 120]).astype(np.float32) * 0.01)
        fcweight3 = Tensor(np.ones([10, 84]).astype(np.float32) * 0.01)
        self.fc1 = nn.Dense(16 * 5 * 5, 120, weight_init=fcweight1)
        self.fc2 = nn.Dense(120, 84, weight_init=fcweight2)
        self.fc3 = nn.Dense(84, 10, weight_init=fcweight3)

    def construct(self, input_x):
        output = self.conv1(input_x)
        output = self.relu(output)
        output = self.pool(output)
        output = self.conv2(output)
        output = self.relu(output)
        output = self.pool(output)
        output = self.flatten(output)
        output = self.fc1(output)
        output = self.fc2(output)
        output = self.fc3(output)
        return output


def loss_func(x):
    return x


def optimizer(x):
    return x


class Net(nn.Cell):
    """ Net definition """

    def __init__(self, dim):
        super(Net, self).__init__()
        self.softmax = nn.Softmax(dim)

    def construct(self, input_x):
        return self.softmax(input_x)
