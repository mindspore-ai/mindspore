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
"""mini_alexnet."""

import mindspore.nn as nn
from mindspore.ops import operations as P


def conv(in_channels, out_channels, kernel_size, stride=1, padding=0, pad_mode="valid", has_bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                     has_bias=has_bias, pad_mode=pad_mode)


def fc_with_initialize(input_channels, out_channels, has_bias=True):
    return nn.Dense(input_channels, out_channels, has_bias=has_bias)


class AlexNet(nn.Cell):
    """
    Alexnet
    """
    def __init__(self, num_classes=10, channel=1, phase='train', include_top=True):
        super().__init__()
        self.conv1 = conv(channel, 12, 11, stride=2, pad_mode="same", has_bias=True)
        self.conv2 = conv(12, 20, 3, pad_mode="same", has_bias=True)
        self.relu = P.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='valid')
        self.include_top = include_top
        if self.include_top:
            dropout_ratio = 0.65
            if phase == 'test':
                dropout_ratio = 1.0
            self.flatten = nn.Flatten()
            self.fc1 = fc_with_initialize(20*3*3, 1024)
            self.fc2 = fc_with_initialize(1024, 1024)
            self.fc3 = fc_with_initialize(1024, num_classes)
            self.dropout = nn.Dropout(p=1-dropout_ratio)

    def construct(self, x):
        """define network"""
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        if not self.include_top:
            return x
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x
