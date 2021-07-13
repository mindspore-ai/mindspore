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
WideResNet building blocks.
"""

import mindspore.nn as nn
from mindspore.ops import operations as P


def _get_optional_avg_pool(stride):
    """
    Create an average pool op if stride is larger than 1, return None otherwise.

    Args:
        stride (int): Stride size, a positive integer.

    Returns:
        nn.AvgPool2d or None.
    """
    if stride == 1:
        return None
    return nn.AvgPool2d(kernel_size=stride, stride=stride)


def _get_optional_pad(in_channels, out_channels):
    """
    Create a zero-pad op if out_channels is larger than in_channels, return None
    otherwise.

    Args:
        in_channels (int): The input channel size.
        out_channels (int): The output channel size (must not be smaller than
                            in_channels).

    Returns:
        nn.Pad or None.
    """
    if in_channels == out_channels:
        return None
    pad_left = (out_channels - in_channels) // 2
    pad_right = out_channels - in_channels - pad_left
    return nn.Pad((
        (0, 0),
        (pad_left, pad_right),
        (0, 0),
        (0, 0),
    ))


class ResidualBlock(nn.Cell):
    """
    ResidualBlock is the basic building block for wide-resnet.

    Args:
        in_channels (int): The input channel size.
        out_channels (int): The output channel size.
        stride (int): The stride size used in the first convolution layer.
        activate_before_residual (bool): Whether to apply bn and relu before
                                         residual.
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            stride,
            activate_before_residual=False,
    ):
        super(ResidualBlock, self).__init__()
        self.activate_before_residual = activate_before_residual
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, stride=stride)

        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=1)

        self.avg_pool = _get_optional_avg_pool(stride)
        self.pad = _get_optional_pad(in_channels, out_channels)

        self.relu = nn.ReLU()
        self.add = P.Add()

    def construct(self, x):
        """Construct the forward network."""
        if self.activate_before_residual:
            out = self.bn1(x)
            out = self.relu(out)
            orig_x = out
        else:
            orig_x = x
            out = self.bn1(x)
            out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        if self.avg_pool is not None:
            orig_x = self.avg_pool(orig_x)
        if self.pad is not None:
            orig_x = self.pad(orig_x)
        return self.add(out, orig_x)


class ResidualGroup(nn.Cell):
    """
    ResidualGroup gathers a group of ResidualBlocks (default: 4).

    Args:
        in_channels (int): The input channel size.
        out_channels (int): The output channel size.
        stride (int): The stride size used in the first ResidualBlock.
        activate_before_residual (bool): Whether to apply bn and relu before
                                         residual in the first ResidualBlock.
        num_blocks (int): Number of ResidualBlocks in the group.
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            stride,
            activate_before_residual=False,
            num_blocks=4,
    ):
        super(ResidualGroup, self).__init__()

        self.rb = ResidualBlock(in_channels, out_channels, stride,
                                activate_before_residual)
        self.rbs = nn.SequentialCell([
            ResidualBlock(out_channels, out_channels, 1)
            for _ in range(num_blocks - 1)
        ])

        self.avg_pool = _get_optional_avg_pool(stride)
        self.pad = _get_optional_pad(in_channels, out_channels)

        self.add = P.Add()

    def construct(self, x):
        """Construct the forward network."""
        orig_x = x

        out = self.rb(x)
        out = self.rbs(out)

        if self.avg_pool is not None:
            orig_x = self.avg_pool(orig_x)
        if self.pad is not None:
            orig_x = self.pad(orig_x)
        return self.add(out, orig_x)


class WRN(nn.Cell):
    """
    WRN is short for Wide-ResNet.

    Args:
        wrn_size (int): Wide-ResNet size.
        in_channels (int): The input channel size.
        num_classes (int): Number of classes to predict.
    """

    def __init__(self, wrn_size, in_channels, num_classes):
        super(WRN, self).__init__()

        sizes = [
            min(wrn_size, 16),
            wrn_size,
            wrn_size * 2,
            wrn_size * 4,
        ]
        strides = [1, 2, 2]

        self.conv1 = nn.Conv2d(in_channels, sizes[0], 3)

        self.rg1 = ResidualGroup(sizes[0], sizes[1], strides[0], True)
        self.rg2 = ResidualGroup(sizes[1], sizes[2], strides[1], False)
        self.rg3 = ResidualGroup(sizes[2], sizes[3], strides[2], False)

        final_stride = 1
        for s in strides:
            final_stride *= s
        self.avg_pool = _get_optional_avg_pool(final_stride)
        self.pad = _get_optional_pad(sizes[0], sizes[-1])

        self.bn = nn.BatchNorm2d(sizes[-1])
        self.fc = nn.Dense(sizes[-1], num_classes)

        self.mean = P.ReduceMean()
        self.relu = nn.ReLU()
        self.add = P.Add()

    def construct(self, x):
        """Construct the forward network."""
        out = self.conv1(x)
        orig_x = out

        out = self.rg1(out)
        out = self.rg2(out)
        out = self.rg3(out)

        if self.avg_pool is not None:
            orig_x = self.avg_pool(orig_x)
        if self.pad is not None:
            orig_x = self.pad(orig_x)
        out = self.add(out, orig_x)

        out = self.bn(out)
        out = self.relu(out)
        out = self.mean(out, (2, 3))
        out = self.fc(out)

        return out
