# Copyright 2025 Huawei Technologies Co., Ltd
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
"""Common Network."""
from mindspore import nn
from mindspore.ops import operations as P


def _conv3x3(in_channels, out_channels, stride=1, padding=0, pad_mode='same', init=0.01, bias_init="zero"):
    """Get a conv2d layer with 3x3 kernel size."""
    return nn.Conv2d(in_channels, out_channels,
        kernel_size=3, stride=stride, padding=padding, pad_mode=pad_mode, weight_init=init, bias_init=bias_init)


def _conv1x1(in_channels, out_channels, stride=1, padding=0, pad_mode='same', init=0.01, bias_init="zero"):
    """Get a conv2d layer with 1x1 kernel size."""
    return nn.Conv2d(in_channels, out_channels,
        kernel_size=1, stride=stride, padding=padding, pad_mode=pad_mode, weight_init=init, bias_init=bias_init)


def _dense(in_channels, out_channels, init=0.01, strategy=None, bias_init="zero"):
    dense = nn.Dense(in_channels, out_channels, weight_init=init, has_bias=True, bias_init=bias_init)
    if strategy is not None:
        dense.matmul.shard(in_strategy=strategy)
    return dense


class ResidualBlock(nn.Cell):
    """
    ResNet V1 residual block definition.

    Args:
        in_channels: Integer. Input channel.
        out_channels: Integer. Output channel.
        stride: Integer. Stride size for the initial convolutional layer. Default:1.
        down_sample: Boolean. If to do the downsample in block. Default:False.
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
                 down_sample=False,
                 momentum=0.1,
                 init=0.01):
        super().__init__()

        out_chls = out_channels // self.expansion
        self.conv1 = _conv1x1(in_channels, out_chls, stride=1, init=init)
        self.bn1 = nn.BatchNorm2d(out_chls, momentum=momentum)

        self.conv2 = _conv3x3(out_chls, out_chls, stride=stride, init=init)
        self.bn2 = nn.BatchNorm2d(out_chls, momentum=momentum)

        self.conv3 = _conv1x1(out_chls, out_channels, stride=1, init=init)
        self.bn3 = nn.BatchNorm2d(out_channels, momentum=momentum)

        self.relu = P.ReLU()
        self.downsample = down_sample
        if self.downsample:
            self.conv_down_sample = _conv1x1(in_channels, out_channels,
                stride=stride, init=init)
            self.bn_down_sample = nn.BatchNorm2d(out_channels, momentum=momentum)
        self.add = P.TensorAdd()

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

        if self.downsample:
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
    def __init__(self, # pylint: disable=dangerous-default-value
                 block,
                 layer_nums,
                 in_channels,
                 out_channels,
                 strides=[1, 2, 2, 2],
                 num_classes=100,
                 init=0.01,
                 strategy=None):
        super().__init__()

        if not len(layer_nums) == len(in_channels) == len(out_channels) == 4:
            raise ValueError("the length of "
                             "layer_num, inchannel, outchannel list must be 4!")

        self.init = init
        self.conv1 = nn.Conv2d(3,
            64,
            kernel_size=7,
            stride=2, weight_init="normal")
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = P.ReLU()
        self.maxpool = P.MaxPool(kernel_size=3,
            strides=2,
            pad_mode='same')

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

        self.mean = P.ReduceMean(keep_dims=False)
        self.end_point = _dense(out_channels[3], num_classes, init=init, strategy=strategy)
        self.flatten = nn.Flatten()
        self.squeeze = P.Squeeze()
        self.cast = P.Cast()

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
        down_sample = False
        if stride != 1 or in_channel != out_channel:
            down_sample = True
        resblk = block(in_channel,
            out_channel,
            stride=stride,
            down_sample=down_sample, init=self.init)
        layers.append(resblk)

        for _ in range(1, layer_num):
            resblk = block(out_channel, out_channel, stride=1, init=self.init)
            layers.append(resblk)

        return nn.SequentialCell(layers)

    def construct(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        c1 = self.maxpool(x)

        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        out = self.mean(c5, (2, 3))
        out = self.end_point(out)

        return out


def resnet50(class_num=10, init=0.01, strategy=None):
    """
    Get ResNet50 neural network.

    Args:
        class_num: Integer. Class number.
        init: Union[Tensor, str, Initializer, numbers.Number].
        strategy: Tuple.

    Returns:
        Cell, cell instance of ResNet50 neural network.

    Examples:
        resnet50(100)
    """
    return ResNet(ResidualBlock,
        [3, 4, 6, 3],
        [64, 256, 512, 1024],
        [256, 512, 1024, 2048],
        [2, 2, 2, 1],
        class_num,
        init=init,
        strategy=strategy)


class Conv2dReduceMean(nn.Cell):
    def __init__(self, in_channel=3, out_channel=12, kernel_size=1, stride_size=1,
                 kernel_me="ones", has_bias=False, bias=None):
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride_size,
                              padding=0, has_bias=has_bias, weight_init=kernel_me,
                              bias_init=bias)
        self.mean = P.ReduceMean(keep_dims=False)

    def construct(self, x):
        x = self.conv(x)
        x = self.mean(x, (2, 3))
        return x
