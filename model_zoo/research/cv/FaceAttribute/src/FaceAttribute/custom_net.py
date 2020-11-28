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
"""Face attribute network unit."""
import mindspore.nn as nn
from mindspore.nn import Dense


class Cut(nn.Cell):
    def construct(self, x):
        return x


def bn_with_initialize(out_channels):
    bn = nn.BatchNorm2d(out_channels, momentum=0.9, eps=1e-5)
    return bn


def fc_with_initialize(input_channels, out_channels):
    return Dense(input_channels, out_channels)


def conv3x3(in_channels, out_channels, stride=1, groups=1, dilation=1, pad_mode="pad", padding=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                     pad_mode=pad_mode, group=groups, has_bias=False, dilation=dilation, padding=padding)


def conv1x1(in_channels, out_channels, pad_mode="pad", stride=1, padding=0):
    """1x1 convolution"""
    return nn.Conv2d(in_channels, out_channels, pad_mode=pad_mode, kernel_size=1, stride=stride, has_bias=False,
                     padding=padding)
