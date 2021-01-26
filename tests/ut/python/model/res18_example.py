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
resnet50 example
"""
import numpy as np

import mindspore.nn as nn  # pylint: disable=C0414
from mindspore import Tensor
from mindspore.common.api import _executor
from mindspore.ops.operations import Add
from ...train_step_wrap import train_step_with_loss_warp


def conv3x3(in_channels, out_channels, stride=1, padding=1, pad_mode='pad'):
    """3x3 convolution """
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=3, stride=stride, padding=padding, pad_mode=pad_mode)


def conv1x1(in_channels, out_channels, stride=1, padding=0, pad_mode='pad'):
    """1x1 convolution"""
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=1, stride=stride, padding=padding, pad_mode=pad_mode)


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

        self.conv_down_sample = conv1x1(in_channels, out_channels,
                                        stride=stride, padding=0)
        self.bn_down_sample = nn.BatchNorm2d(out_channels)
        self.add = Add()

    def construct(self, x):
        """
        :param x:
        :return:
        """
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


class ResNet18(nn.Cell):
    """
    resnet nn.Cell
    """

    def __init__(self, block, num_classes=100):
        super(ResNet18, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, pad_mode='pad')
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same')

        self.layer1 = self.MakeLayer(
            block, 2, in_channels=64, out_channels=256, stride=1)
        self.layer2 = self.MakeLayer(
            block, 2, in_channels=256, out_channels=512, stride=2)
        self.layer3 = self.MakeLayer(
            block, 2, in_channels=512, out_channels=1024, stride=2)
        self.layer4 = self.MakeLayer(
            block, 2, in_channels=1024, out_channels=2048, stride=2)

        self.avgpool = nn.AvgPool2d(7, 1)
        self.flatten = nn.Flatten()
        self.fc = nn.Dense(512 * block.expansion, num_classes)

    def MakeLayer(self, block, layer_num, in_channels, out_channels, stride):
        """
        make block layer
        :param block:
        :param layer_num:
        :param in_channels:
        :param out_channels:
        :param stride:
        :return:
        """
        layers = []
        resblk = block(in_channels, out_channels,
                       stride=stride, down_sample=True)
        layers.append(resblk)

        for _ in range(1, layer_num):
            resblk = block(out_channels, out_channels, stride=1)
            layers.append(resblk)

        return nn.SequentialCell(layers)

    def construct(self, x):
        """
        :param x:
        :return:
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)

        return x


class ResNet9(nn.Cell):
    """
    resnet nn.Cell
    """

    def __init__(self, block, num_classes=100):
        super(ResNet9, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)

        self.layer1 = self.MakeLayer(
            block, 1, in_channels=64, out_channels=256, stride=1)
        self.layer2 = self.MakeLayer(
            block, 1, in_channels=256, out_channels=512, stride=2)
        self.layer3 = self.MakeLayer(
            block, 1, in_channels=512, out_channels=1024, stride=2)
        self.layer4 = self.MakeLayer(
            block, 1, in_channels=1024, out_channels=2048, stride=2)

        self.avgpool = nn.AvgPool2d(7, 1)
        self.flatten = nn.Flatten()
        self.fc = nn.Dense(512 * block.expansion, num_classes)

    def MakeLayer(self, block, layer_num, in_channels, out_channels, stride):
        """
        make block layer
        :param block:
        :param layer_num:
        :param in_channels:
        :param out_channels:
        :param stride:
        :return:
        """
        layers = []
        resblk = block(in_channels, out_channels,
                       stride=stride, down_sample=True)
        layers.append(resblk)

        for _ in range(1, layer_num):
            resblk = block(out_channels, out_channels, stride=1)
            layers.append(resblk)

        return nn.SequentialCell(layers)

    def construct(self, x):
        """
        :param x:
        :return:
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)

        return x


def resnet18():
    return ResNet18(ResidualBlock, 10)


def resnet9():
    return ResNet9(ResidualBlock, 10)


def test_compile():
    net = resnet18()
    input_data = Tensor(np.ones([1, 3, 224, 224]))
    _executor.compile(net, input_data)


def test_train_step():
    net = train_step_with_loss_warp(resnet9())
    input_data = Tensor(np.ones([1, 3, 224, 224]))
    label = Tensor(np.zeros([1, 10]))
    _executor.compile(net, input_data, label)


def test_train_step_training():
    net = train_step_with_loss_warp(resnet9())
    input_data = Tensor(np.ones([1, 3, 224, 224]))
    label = Tensor(np.zeros([1, 10]))
    net.set_train()
    _executor.compile(net, input_data, label)
