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
"""Resnet50 backbone."""
import mindspore.nn as nn

BN_MOMENTUM = 0.9


class ResidualBlock(nn.Cell):
    """
    ResNet V1 residual block definition.

    Args:
        in_channel (int) - Input channel.
        out_channel (int) - Output channel.
        stride (int) - Stride size for the initial convolutional layer. Default: 1.
        downsample (func) - the downsample in block. Default: None.

    Returns:
        Tensor, output tensor.

    Examples:
        >>> ResidualBlock(64, 256, stride=2, downsample=None)
    """
    expansion = 4

    def __init__(self,
                 in_channel,
                 out_channel,
                 stride=1,
                 downsample=None):
        super(ResidualBlock, self).__init__()
        self.stride = stride
        channel = out_channel // self.expansion
        self.conv1 = nn.Conv2d(in_channel, channel, kernel_size=1, has_bias=False)
        self.bn1 = nn.BatchNorm2d(channel, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(channel, channel, kernel_size=3, stride=stride,
                               pad_mode='pad', padding=1, has_bias=False)
        self.bn2 = nn.BatchNorm2d(channel, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(channel, out_channel, kernel_size=1, has_bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.downsample = downsample

    def construct(self, x):
        """Defines the computation performed."""
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNetFea(nn.Cell):
    """
    ResNet architecture.

    Args:
        block (Cell): Block for network.
        layer_nums (list): Numbers of block in different layers.
        in_channels (list): Input channel in each layer.
        out_channels (list): Output channel in each layer.
        strides (list):  Stride size in each layer.
    Returns:
        Tensor, output tensor.

    Examples:
        >>> ResNetFea(ResidualBlock,
        >>>        [3, 4, 6, 3],
        >>>        [64, 256, 512, 1024],
        >>>        [256, 512, 1024, 2048],
        >>>        [1, 2, 2, 2])
    """

    def __init__(self,
                 block,
                 layer_nums,
                 in_channels,
                 out_channels,
                 strides):
        self.cin = 64
        super(ResNetFea, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, pad_mode='pad', padding=3, has_bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same')

        self.layer1 = self._make_layer(block,
                                       layer_nums[0],
                                       in_channel=in_channels[0],
                                       out_channel=out_channels[0],
                                       stride=strides[0])

        self.layer2 = self._make_layer(block,
                                       layer_nums[1],
                                       in_channel=in_channels[1],
                                       out_channel=out_channels[1],
                                       stride=strides[1])

        self.layer3 = self._make_layer(block,
                                       layer_nums[2],
                                       in_channel=in_channels[2],
                                       out_channel=out_channels[2],
                                       stride=strides[2])

        self.layer4 = self._make_layer(block,
                                       layer_nums[3],
                                       in_channel=in_channels[3],
                                       out_channel=out_channels[3],
                                       stride=strides[3])
        self.cin = out_channels[3]
        self.deconv_layers = self._make_deconv_layer(
            num_layers=3,
            num_filters=[256, 128, 64],
            num_kernels=[4, 4, 4],
        )

    def _make_layer(self, block, layer_num, in_channel, out_channel, stride=1):
        """
        Make stage network of ResNet.

        Args:
            block (Cell): Resnet block.
            layer_num (int): Layer number.
            in_channel (int): Input channel.
            out_channel (int): Output channel.
            stride (int): Stride size for the first convolutional layer.
        Returns:
            SequentialCell, the output layer.

        Examples:
            >>> _make_layer(ResidualBlock, 3, 128, 256, 2)
        """
        downsample = None
        if stride != 1 or in_channel != out_channel:
            downsample = nn.SequentialCell(
                nn.Conv2d(in_channel, out_channel,
                          kernel_size=1, stride=stride, has_bias=False),
                nn.BatchNorm2d(out_channel, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(in_channel, out_channel, stride, downsample))
        for _ in range(1, layer_num):
            layers.append(block(out_channel, out_channel))

        return nn.SequentialCell(*layers)

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        """
        Deconvolution for upsampling
        """
        layers = []
        for i in range(num_layers):
            kernel = num_kernels[i]
            cout = num_filters[i]

            up = nn.Conv2dTranspose(in_channels=self.cin, out_channels=cout,
                                    kernel_size=kernel, stride=2,
                                    pad_mode='pad', padding=1)
            layers.append(up)
            layers.append(nn.BatchNorm2d(cout, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU())
            self.cin = cout

        return nn.SequentialCell(*layers)

    def construct(self, x):
        """Defines the computation performed."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.deconv_layers(x)

        return x
