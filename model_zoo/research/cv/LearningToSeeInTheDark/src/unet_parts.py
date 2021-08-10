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
"""Unet Components"""
import mindspore.nn as nn
import mindspore.ops.operations as F
from mindspore.ops import Maximum
from mindspore.ops import DepthToSpace as dts
from mindspore.common.initializer import TruncatedNormal
from mindspore.common.initializer import XavierUniform
import mindspore as ms
ms.set_seed(1212)


class LRelu(nn.Cell):
    """ activation function """
    def __init__(self):
        super(LRelu, self).__init__()
        self.max = Maximum()

    def construct(self, x):
        """ construct of lrelu activation """
        return self.max(x * 0.2, x)


class DoubleConv(nn.Cell):
    """conv2d for two times with lrelu activation"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.kernel_init = XavierUniform()
        self.double_conv = nn.SequentialCell(
            [nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, pad_mode="same",
                       weight_init=self.kernel_init), LRelu(),
             nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1, pad_mode="same",
                       weight_init=self.kernel_init), LRelu()])

    def construct(self, x):
        """ construct of double conv2d """
        return self.double_conv(x)


class Down(nn.Cell):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.SequentialCell(
            [nn.MaxPool2d(kernel_size=2, stride=2, pad_mode="same"),
             DoubleConv(in_channels, out_channels)]
        )

    def construct(self, x):
        """ construct of down cell """
        return self.maxpool_conv(x)


class Up(nn.Cell):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.concat = F.Concat(axis=1)
        self.kernel_init = TruncatedNormal(0.02)
        self.conv = DoubleConv(in_channels, out_channels)
        self.up = nn.Conv2dTranspose(in_channels, in_channels // 2, kernel_size=2, stride=2,
                                     pad_mode='same', weight_init=self.kernel_init)

    def construct(self, x1, x2):
        """ construct of up cell """
        x1 = self.up(x1)
        x = self.concat((x1, x2))
        return self.conv(x)


class OutConv(nn.Cell):
    """trans data into RGB channels"""
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.kernel_init = XavierUniform()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, pad_mode='same', weight_init=self.kernel_init)
        self.DtS = dts(block_size=2)

    def construct(self, x):
        """ construct of last conv """
        x = self.conv(x)
        x = self.DtS(x)
        return x
