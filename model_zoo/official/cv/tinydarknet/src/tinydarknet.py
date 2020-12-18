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
"""TinydarkNet"""
import mindspore.nn as nn
from mindspore.common.initializer import TruncatedNormal
from mindspore.ops import operations as P


def weight_variable():
    """Weight variable."""
    return TruncatedNormal(0.02)


class Conv1x1dBlock(nn.Cell):
    """
     Basic convolutional block
     Args:
         in_channles (int): Input channel.
         out_channels (int): Output channel.
         kernel_size (int): Input kernel size. Default: 1
         stride (int): Stride size for the first convolutional layer. Default: 1.
         padding (int): Implicit paddings on both sides of the input. Default: 0.
         pad_mode (str): Padding mode. Optional values are "same", "valid", "pad". Default: "same".
      Returns:
          Tensor, output tensor.
    """

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, pad_mode="same"):
        super(Conv1x1dBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                              padding=padding, pad_mode=pad_mode, weight_init=weight_variable())
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
        self.leakyrelu = nn.LeakyReLU()

    def construct(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.leakyrelu(x)
        return x

class Conv3x3dBlock(nn.Cell):
    """
     Basic convolutional block
     Args:
         in_channles (int): Input channel.
         out_channels (int): Output channel.
         kernel_size (int): Input kernel size. Default: 1
         stride (int): Stride size for the first convolutional layer. Default: 1.
         padding (int): Implicit paddings on both sides of the input. Default: 0.
         pad_mode (str): Padding mode. Optional values are "same", "valid", "pad". Default: "same".
      Returns:
          Tensor, output tensor.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, pad_mode="pad"):
        super(Conv3x3dBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                              padding=padding, pad_mode=pad_mode, weight_init=weight_variable())
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
        self.leakyrelu = nn.LeakyReLU()

    def construct(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.leakyrelu(x)
        return x



class TinyDarkNet(nn.Cell):
    """
    Tinydarknet architecture
    """

    def __init__(self, num_classes, include_top=True):
        super(TinyDarkNet, self).__init__()
        self.conv1 = Conv3x3dBlock(3, 16)
        self.conv2 = Conv3x3dBlock(16, 32)
        self.conv3 = Conv1x1dBlock(32, 16)
        self.conv4 = Conv3x3dBlock(16, 128)
        self.conv5 = Conv1x1dBlock(128, 16)
        self.conv6 = Conv3x3dBlock(16, 128)
        self.conv7 = Conv1x1dBlock(128, 32)
        self.conv8 = Conv3x3dBlock(32, 256)
        self.conv9 = Conv1x1dBlock(256, 32)
        self.conv10 = Conv3x3dBlock(32, 256)
        self.conv11 = Conv1x1dBlock(256, 64)
        self.conv12 = Conv3x3dBlock(64, 512)
        self.conv13 = Conv1x1dBlock(512, 64)
        self.conv14 = Conv3x3dBlock(64, 512)
        self.conv15 = Conv1x1dBlock(512, 128)
        self.conv16 = Conv1x1dBlock(128, 1000)

        self.maxpool2d = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode="same")
        self.avgpool2d = P.ReduceMean(keep_dims=True)

        self.flatten = nn.Flatten()


    def construct(self, x):
        """construct"""
        x = self.conv1(x)
        x = self.maxpool2d(x)

        x = self.conv2(x)
        x = self.maxpool2d(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.maxpool2d(x)

        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.maxpool2d(x)

        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.conv14(x)
        x = self.conv15(x)
        x = self.conv16(x)

        x = self.avgpool2d(x, (2, 3))
        x = self.flatten(x)

        return x
