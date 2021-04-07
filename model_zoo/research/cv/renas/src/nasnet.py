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
"""NASNet."""
import numpy as np
import mindspore.nn as nn
from mindspore import ops
from mindspore.ops import operations as P
from mindspore.common.tensor import Tensor


def _weight_variable(shape, factor=0.01):
    init_value = np.random.randn(*shape).astype(np.float32) * factor
    return Tensor(init_value)


def _conv3x3(in_channel, out_channel, stride=1):
    weight_shape = (out_channel, in_channel, 3, 3)
    weight = _weight_variable(weight_shape)
    return nn.Conv2d(in_channel, out_channel,
                     kernel_size=3, stride=stride, padding=1, pad_mode='pad', weight_init=weight)


def _conv1x1(in_channel, out_channel, stride=1):
    weight_shape = (out_channel, in_channel, 1, 1)
    weight = _weight_variable(weight_shape)
    return nn.Conv2d(in_channel, out_channel,
                     kernel_size=1, stride=stride, padding=0, pad_mode='same', weight_init=weight)


def _conv7x7(in_channel, out_channel, stride=1):
    weight_shape = (out_channel, in_channel, 7, 7)
    weight = _weight_variable(weight_shape)
    return nn.Conv2d(in_channel, out_channel,
                     kernel_size=7, stride=stride, padding=3, pad_mode='pad', weight_init=weight)


def _bn(channel):
    return nn.BatchNorm2d(channel)


def _bn_last(channel):
    return nn.BatchNorm2d(channel)


def _fc(in_channel, out_channel):
    weight_shape = (out_channel, in_channel)
    weight = _weight_variable(weight_shape)
    return nn.Dense(in_channel, out_channel, has_bias=True, weight_init=weight, bias_init=0)


class BasicCell(nn.Cell):
    """
    NASNet basic cell definition.

    Args:
        None.
    Returns:
        Tensor, output tensor.
    """
    expansion = 4

    def __init__(self):
        super(BasicCell, self).__init__()

        self.conv3x3_1 = _conv3x3(128, 128)
        self.bn3x3_1 = _bn(128)
        self.conv3x3_2 = _conv3x3(128, 128)
        self.bn3x3_2 = _bn(128)
        self.conv3x3_3 = _conv3x3(128, 128)
        self.bn3x3_3 = _bn(128)
        self.mp = nn.MaxPool2d(kernel_size=3, stride=1, pad_mode="same")

        self.proj1 = _conv1x1(128, 64)
        self.bn1 = _bn(64)
        self.proj2 = _conv1x1(128, 64)
        self.bn2 = _bn(64)
        self.proj3 = _conv1x1(128, 64)
        self.bn3 = _bn(64)
        self.proj4 = _conv1x1(128, 64)
        self.bn4 = _bn(64)
        self.proj5 = _conv1x1(128, 64)
        self.bn5 = _bn(64)
        self.proj6 = _conv1x1(128, 64)
        self.bn6 = _bn(64)

        self.relu = P.ReLU()
        self.concat = ops.Concat(axis=1)

    def construct(self, x):
        o1 = self.mp(x)
        o1 = self.concat((self.relu(self.bn1(self.proj1(o1))), self.relu(self.bn2(self.proj2(x)))))
        o2 = self.relu(self.bn3x3_1(self.conv3x3_1(o1)))
        o2 = self.concat((self.relu(self.bn3(self.proj3(o2))), self.relu(self.bn4(self.proj4(x)))))
        o3 = self.relu(self.bn3x3_2(self.conv3x3_2(o2)))
        o4 = self.relu(self.bn3x3_3(self.conv3x3_3(x)))
        out = self.concat((self.relu(self.bn5(self.proj5(o3))), self.relu(self.bn6(self.proj6(o4)))))
        return out


class NasBenchNet(nn.Cell):
    """
    NASNet architecture.

    Args:
        cell (Cell): Cell for network.
        num_classes (int): The number of classes that the training images are belonging to.
    Returns:
        Tensor, output tensor.
    """
    def __init__(self,
                 cell,
                 num_classes=10):
        super(NasBenchNet, self).__init__()

        self.conv1 = _conv3x3(3, 128)
        self.bn1 = _bn(128)
        self.mp = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode="valid")
        self.block1 = self._make_block(cell)
        self.block2 = self._make_block(cell)
        self.block3 = self._make_block(cell)
        self.linear = _fc(128, num_classes)
        self.ap = nn.AvgPool2d(kernel_size=8, pad_mode='valid')
        self.relu = P.ReLU()
        self.flatten = nn.Flatten()

    def _make_block(self, cell):
        layers = []
        for _ in range(3):
            layers.append(cell())
        return nn.SequentialCell(layers)

    def construct(self, x):
        """construct"""
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.block1(out)
        out = self.mp(out)
        out = self.block2(out)
        out = self.mp(out)
        out = self.block3(out)
        out = self.ap(out)
        out = self.flatten(out)
        out = self.linear(out)
        return out


def nasbenchnet():
    return NasBenchNet(BasicCell)
