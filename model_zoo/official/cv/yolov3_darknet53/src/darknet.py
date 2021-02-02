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
         nn.BatchNorm2d(out_channels, momentum=0.1),
         nn.ReLU()]
    )


class ResidualBlock(nn.Cell):
    """
    DarkNet V1 residual block definition.

    Args:
        in_channels: Integer. Input channel.
        out_channels: Integer. Output channel.

    Returns:
        Tensor, output tensor.
    Examples:
        ResidualBlock(3, 208)
    """
    expansion = 4

    def __init__(self,
                 in_channels,
                 out_channels):

        super(ResidualBlock, self).__init__()
        out_chls = out_channels//2
        self.conv1 = conv_block(in_channels, out_chls, kernel_size=1, stride=1)
        self.conv2 = conv_block(out_chls, out_channels, kernel_size=3, stride=1)
        self.add = P.Add()

    def construct(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.add(out, identity)

        return out


class DarkNet(nn.Cell):
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
        DarkNet(ResidualBlock,
               [1, 2, 8, 8, 4],
               [32, 64, 128, 256, 512],
               [64, 128, 256, 512, 1024],
               100)
    """
    def __init__(self,
                 block,
                 layer_nums,
                 in_channels,
                 out_channels,
                 detect=False):
        super(DarkNet, self).__init__()

        self.outchannel = out_channels[-1]
        self.detect = detect

        if not len(layer_nums) == len(in_channels) == len(out_channels) == 5:
            raise ValueError("the length of layer_num, inchannel, outchannel list must be 5!")
        self.conv0 = conv_block(3,
                                in_channels[0],
                                kernel_size=3,
                                stride=1)
        self.conv1 = conv_block(in_channels[0],
                                out_channels[0],
                                kernel_size=3,
                                stride=2)
        self.layer1 = self._make_layer(block,
                                       layer_nums[0],
                                       in_channel=out_channels[0],
                                       out_channel=out_channels[0])
        self.conv2 = conv_block(in_channels[1],
                                out_channels[1],
                                kernel_size=3,
                                stride=2)
        self.layer2 = self._make_layer(block,
                                       layer_nums[1],
                                       in_channel=out_channels[1],
                                       out_channel=out_channels[1])
        self.conv3 = conv_block(in_channels[2],
                                out_channels[2],
                                kernel_size=3,
                                stride=2)
        self.layer3 = self._make_layer(block,
                                       layer_nums[2],
                                       in_channel=out_channels[2],
                                       out_channel=out_channels[2])
        self.conv4 = conv_block(in_channels[3],
                                out_channels[3],
                                kernel_size=3,
                                stride=2)
        self.layer4 = self._make_layer(block,
                                       layer_nums[3],
                                       in_channel=out_channels[3],
                                       out_channel=out_channels[3])
        self.conv5 = conv_block(in_channels[4],
                                out_channels[4],
                                kernel_size=3,
                                stride=2)
        self.layer5 = self._make_layer(block,
                                       layer_nums[4],
                                       in_channel=out_channels[4],
                                       out_channel=out_channels[4])

    def _make_layer(self, block, layer_num, in_channel, out_channel):
        """
        Make Layer for DarkNet.

        :param block: Cell. DarkNet block.
        :param layer_num: Integer. Layer number.
        :param in_channel: Integer. Input channel.
        :param out_channel: Integer. Output channel.

        Examples:
            _make_layer(ConvBlock, 1, 128, 256)
        """
        layers = []
        darkblk = block(in_channel, out_channel)
        layers.append(darkblk)

        for _ in range(1, layer_num):
            darkblk = block(out_channel, out_channel)
            layers.append(darkblk)

        return nn.SequentialCell(layers)

    def construct(self, x):
        c1 = self.conv0(x)
        c2 = self.conv1(c1)
        c3 = self.layer1(c2)
        c4 = self.conv2(c3)
        c5 = self.layer2(c4)
        c6 = self.conv3(c5)
        c7 = self.layer3(c6)
        c8 = self.conv4(c7)
        c9 = self.layer4(c8)
        c10 = self.conv5(c9)
        c11 = self.layer5(c10)
        if self.detect:
            return c7, c9, c11

        return c11

    def get_out_channels(self):
        return self.outchannel


def darknet53():
    """
    Get DarkNet53 neural network.

    Returns:
        Cell, cell instance of DarkNet53 neural network.

    Examples:
        darknet53()
    """
    return DarkNet(ResidualBlock, [1, 2, 8, 8, 4],
                   [32, 64, 128, 256, 512],
                   [64, 128, 256, 512, 1024])
