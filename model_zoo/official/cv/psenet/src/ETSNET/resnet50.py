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
"""ResNet."""
import mindspore.nn as nn
from mindspore.ops import operations as P

from .base import _conv, _bn

class ResidualBlock(nn.Cell):
    """
    ResNet V1 residual block definition.

    Args:
        in_channels: Integer. Input channel.
        out_channels: Integer. Output channel.
        stride: Integer. Stride size for the initial convolutional layer. Default:1.
        momentum: Float. Momentum for batchnorm layer. Default:0.1.

    Returns:
        Tensor, output tensor.

    Examples:
        ResidualBlock(3,256,stride=2,down_sample=True)
    """
    expansion = 4

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 momentum=0.1):
        super(ResidualBlock, self).__init__()

        out_chls = out_channels // self.expansion
        self.conv1 = _conv(in_channels, out_chls, kernel_size=1, stride=1)
        self.bn1 = _bn(out_chls, momentum=momentum)

        self.conv2 = _conv(out_chls, out_chls, kernel_size=3, stride=stride, padding=1, pad_mode='pad')
        self.bn2 = _bn(out_chls, momentum=momentum)

        self.conv3 = _conv(out_chls, out_channels, kernel_size=1, stride=1)
        self.bn3 = _bn(out_channels, momentum=momentum)

        self.relu = P.ReLU()
        self.downsample = (in_channels != out_channels)
        self.stride = stride
        if self.downsample or self.stride != 1:
            self.conv_down_sample = _conv(in_channels, out_channels,
                                          kernel_size=1, stride=stride)
            self.bn_down_sample = _bn(out_channels, momentum=momentum)

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

        if self.downsample or self.stride != 1:
            identity = self.conv_down_sample(identity)
            identity = self.bn_down_sample(identity)

        out = self.add(out, identity)
        out = self.relu(out)

        return out


class ResNet(nn.Cell):
    """
    ResNet V1 network.

    Args:
        block: Cell. Block for network.
        layer_nums: List. Numbers of different layers.
        in_channels: Integer. Input channel.
        out_channels: Integer. Output channel.
        num_classes: Integer. Class number. Default:100.

    Returns:
        Tensor, output tensor.

    Examples:
        ResNet(ResidualBlock,
               [3, 4, 6, 3],
               [64, 256, 512, 1024],
               [256, 512, 1024, 2048],
               100)
    """

    def __init__(self,
                 block,
                 layer_nums,
                 in_channels,
                 out_channels):
        super(ResNet, self).__init__()

        if not len(layer_nums) == len(in_channels) == len(out_channels) == 4:
            raise ValueError("the length of "
                             "layer_num, inchannel, outchannel list must be 4!")

        self.conv1 = _conv(3, 64, kernel_size=7, stride=2, padding=3, pad_mode='pad')
        self.bn1 = _bn(64)
        self.relu = P.ReLU()
        self.pad = P.Pad(((0, 0), (0, 0), (1, 1), (1, 1)))
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='valid')

        self.layer1 = self._make_layer(block,
                                       layer_nums[0],
                                       in_channel=in_channels[0],
                                       out_channel=out_channels[0],
                                       stride=1)
        self.layer2 = self._make_layer(block,
                                       layer_nums[1],
                                       in_channel=in_channels[1],
                                       out_channel=out_channels[1],
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       layer_nums[2],
                                       in_channel=in_channels[2],
                                       out_channel=out_channels[2],
                                       stride=2)
        self.layer4 = self._make_layer(block,
                                       layer_nums[3],
                                       in_channel=in_channels[3],
                                       out_channel=out_channels[3],
                                       stride=2)

    def _make_layer(self, block, layer_num, in_channel, out_channel, stride):
        """
        Make Layer for ResNet.

        Args:
            block: Cell. Resnet block.
            layer_num: Integer. Layer number.
            in_channel: Integer. Input channel.
            out_channel: Integer. Output channel.
            stride:Integer. Stride size for the initial convolutional layer.

        Returns:
            SequentialCell, the output layer.

        Examples:
            _make_layer(BasicBlock, 3, 128, 256, 2)
        """
        layers = []

        resblk = block(in_channel, out_channel, stride=stride)
        layers.append(resblk)

        for _ in range(1, layer_num):
            resblk = block(out_channel, out_channel, stride=1)
            layers.append(resblk)

        return nn.SequentialCell(layers)

    def construct(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pad(x)
        c1 = self.maxpool(x)

        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        return c2, c3, c4, c5
