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
"""WideResNet"""

import mindspore.nn as nn
import mindspore.ops as ops


class WideBasic(nn.Cell):
    """
    WideBasic
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(WideBasic, self).__init__()

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, has_bias=True)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(keep_prob=0.7)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, has_bias=True)

        self.shortcut = nn.SequentialCell()

        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.SequentialCell(
                [nn.Conv2d(in_channels, out_channels, 1, stride=stride, has_bias=True)]
            )

    def construct(self, x):
        """
        basic construct
        """

        identity = x

        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv2(x)

        shortcut = self.shortcut(identity)

        return x + shortcut


class WideResNet(nn.Cell):
    """
    WideReNet
    """
    def __init__(self, num_classes, block, depth=50, widen_factor=1):
        """
        classes, block, depth, widen_factor
        """
        super(WideResNet, self).__init__()

        self.depth = depth
        k = widen_factor
        n = int((depth - 4) / 6)
        self.in_channels = 16
        self.conv1 = nn.Conv2d(3, self.in_channels, 3, 1, padding=0, pad_mode='same')
        self.conv2 = self._make_layer(block, 16 * k, n, 1)
        self.conv3 = self._make_layer(block, 32 * k, n, 2)
        self.conv4 = self._make_layer(block, 64 * k, n, 2)
        self.bn = nn.BatchNorm2d(64 * k, momentum=0.9)
        self.relu = nn.ReLU()
        self.mean = ops.ReduceMean(keep_dims=True)
        self.flatten = nn.Flatten()
        self.linear = nn.Dense(64 * k, num_classes, has_bias=True)

        self.bn1 = nn.BatchNorm2d(16)


    def construct(self, x):
        """
        WideResNet construct
        """

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.mean(x, (2, 3))
        x = self.flatten(x)
        x = self.linear(x)

        return x

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """
        make layer
        """

        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, s))
            self.in_channels = out_channels

        return nn.SequentialCell(*layers)


def wideresnet(depth=40, widen_factor=10):
    net = WideResNet(10, WideBasic, depth=depth, widen_factor=widen_factor)
    return net
