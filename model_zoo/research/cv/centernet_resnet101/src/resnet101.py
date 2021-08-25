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
"""
ResNet101 backbone
"""
import math
import mindspore.nn as nn
from mindspore import Parameter
from mindspore.common.initializer import initializer
from mindspore.common.initializer import HeUniform

BN_MOMENTUM = 0.9


class Bottleneck(nn.Cell):
    """
    ResNet basic block.

    Args:
        cin(int): Input channel.
        cout(int): Output channel.
        stride(int): Stride size for the initial convolutional layer. Default:1.
        downsample(Cell): Downsample convolution block. Default:None.

    Returns:
        Tensor, output tensor.

    """
    expansion = 4

    def __init__(self, cin, cout, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(cin, cout, kernel_size=1, has_bias=False)
        self.bn1 = nn.BatchNorm2d(cout, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(cout, cout, kernel_size=3, stride=stride,
                               pad_mode='pad', padding=1, has_bias=False)
        self.bn2 = nn.BatchNorm2d(cout, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(cout, cout * self.expansion, kernel_size=1, has_bias=False)
        self.bn3 = nn.BatchNorm2d(cout * self.expansion)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

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


class ResNet101(nn.Cell):
    """
    ResNet architecture.

    Args:
        block (Cell): Block for network.
        layer (list): Numbers of block in different layers.
        heads (dict): The number of heatmap,width and height,offset.
        head_conv(int): Input convolution dimension.

    Returns:
        Tensor, output tensor.

    """
    def __init__(self, block, layers, heads, head_conv):
        self.cin = head_conv
        self.heads = heads
        super(ResNet101, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, pad_mode='pad', padding=3, has_bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same')
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.deconv_layers = self._make_deconv_layer(
            num_layers=3,
            num_filters=[256, 128, 64],
            num_kernels=[4, 4, 4],
        )

    def _make_layer(self, block, cout, blocks, stride=1):
        """
        Make stage network of ResNet.

        Args:
            block (Cell): Resnet block.
            cout (int): Output channel.
            blocks(int): Layer number.
            stride (int): Stride size for the first convolutional layer.

        Returns:
            SequentialCell, the output layer.
        """
        downsample = None
        if stride != 1 or self.cin != cout * block.expansion:
            downsample = nn.SequentialCell(
                nn.Conv2d(self.cin, cout * block.expansion,
                          kernel_size=1, stride=stride, has_bias=False),
                nn.BatchNorm2d(cout * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.cin, cout, stride, downsample))
        self.cin = cout * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.cin, cout))

        return nn.SequentialCell(*layers)

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        """
        Make deconvolution network of ResNet.

        Args:
            num_layer(int): Layer number.
            num_filters (list): Convolution dimension.
            num_kernels (list): The size of convolution kernel .

        Returns:
            SequentialCell, the output layer.
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


def weights_init(net):
    """Initialize the weight."""
    for _, cell in net.cells_and_names():
        if isinstance(cell, nn.Conv2d):
            cell.weight = Parameter(initializer(HeUniform(negative_slope=math.sqrt(5)),
                                                cell.weight.shape, cell.weight.dtype), name=cell.weight.name)
