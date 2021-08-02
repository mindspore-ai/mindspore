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
"""DarkNet model."""
import mindspore.nn as nn
from mindspore.ops import operations as P


def conv_block(in_channels,
               out_channels,
               kernel_size,
               stride,
               dilation=1):
    """Get a conv2d batchnorm and relu layer"""
    pad_mode = 'same'
    padding = 0

    return nn.SequentialCell(
        [nn.Conv2d(in_channels,
                   out_channels,
                   kernel_size=kernel_size,
                   stride=stride,
                   padding=padding,
                   dilation=dilation,
                   pad_mode=pad_mode),
         nn.BatchNorm2d(out_channels, momentum=0.9),
         nn.LeakyReLU(alpha=0.1)]
    )


class Tiny(nn.Cell):
    """
    DarkNet V1 network.

    Args:
        block: Cell. Block for network.
        layer_nums: List. Numbers of different layers.
        in_channels: Integer. Input channel.
        out_channels: Integer. Output channel.
        detect: Bool. Whether detect or not. Default:False.

    Returns:
        Tuple, tuple of output tensor,(f1,f2,f3,f4,f5).

    Examples:

    """

    def __init__(self, detect=False):
        super(Tiny, self).__init__()

        self.detect = detect

        self.conv0 = conv_block(3, 16, kernel_size=3, stride=1)
        self.maxpool_k2_s2 = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode="valid")
        self.maxpool_k2_s1 = nn.MaxPool2d(kernel_size=2, stride=1, pad_mode="same")
        self.conv1 = conv_block(16, 32, kernel_size=3, stride=1)
        self.conv2 = conv_block(32, 64, kernel_size=3, stride=1)
        self.conv3 = conv_block(64, 128, kernel_size=3, stride=1)
        self.conv4 = conv_block(128, 256, kernel_size=3, stride=1)
        self.conv5 = conv_block(256, 512, kernel_size=3, stride=1)
        self.conv6 = conv_block(512, 1024, kernel_size=3, stride=1)
        self.conv7 = conv_block(1024, 256, kernel_size=1, stride=1)
        self.conv8 = conv_block(256, 512, kernel_size=3, stride=1)
        self.conv9 = conv_block(256, 128, kernel_size=1, stride=1)
        self.conv10 = conv_block(384, 256, kernel_size=3, stride=1)
        self.concat = P.Concat(axis=1)

    def construct(self, x):
        """Construction function"""
        img_hight = P.Shape()(x)[2]
        img_width = P.Shape()(x)[3]

        c0 = self.conv0(x)
        mp0 = self.maxpool_k2_s2(c0)
        c1 = self.conv1(mp0)
        mp1 = self.maxpool_k2_s2(c1)
        c2 = self.conv2(mp1)
        mp2 = self.maxpool_k2_s2(c2)
        c3 = self.conv3(mp2)
        mp3 = self.maxpool_k2_s2(c3)
        c4 = self.conv4(mp3)

        mp4 = self.maxpool_k2_s2(c4)
        c5 = self.conv5(mp4)
        mp5 = self.maxpool_k2_s1(c5)
        c6 = self.conv6(mp5)
        c7 = self.conv7(c6)

        c8 = self.conv8(c7)
        c9 = self.conv9(c7)
        ups = P.ResizeNearestNeighbor((img_hight / 16, img_width / 16))(c9)
        concat = self.concat((ups, c4))
        c10 = self.conv10(concat)

        return c8, c10
