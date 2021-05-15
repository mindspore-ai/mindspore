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
"""ResNet."""
import numpy as np
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.common.tensor import Tensor
from mindspore.common.initializer import initializer
import mindspore.common.dtype as mstype


# from mindspore.ops import resolved_ops as RO

def _weight_variable(shape):
    """weight_variable"""
    return initializer('HeUniform', shape=shape, dtype=mstype.float32)


def _weight_variable_(shape, factor=0.01):
    init_value = np.random.randn(*shape).astype(np.float32) * factor
    return Tensor(init_value)


def _conv3x3(in_channel, out_channel, stride=1):
    return nn.Conv2d(in_channel, out_channel,
                     kernel_size=3, stride=stride, padding=0, pad_mode='same', weight_init="HeUniform")


def _conv1x1(in_channel, out_channel, stride=1):
    return nn.Conv2d(in_channel, out_channel,
                     kernel_size=1, stride=stride, padding=0, pad_mode='same', weight_init="HeUniform")


def _conv7x7(in_channel, out_channel, stride=1):
    return nn.Conv2d(in_channel, out_channel,
                     kernel_size=7, stride=stride, padding=0, pad_mode='same', weight_init="HeUniform")


def _bn(channel, training=True):
    if training:
        return nn.BatchNorm2d(channel, eps=1e-4, momentum=0.9,
                              gamma_init=1, beta_init=0, moving_mean_init=0, moving_var_init=1)
    return nn.BatchNorm2d(channel, eps=1e-4, momentum=0.9,
                          gamma_init=1, beta_init=0, moving_mean_init=0, moving_var_init=1,
                          use_batch_statistics=training)


def _bn_last(channel, training=True):
    if training:
        return nn.BatchNorm2d(channel, eps=1e-4, momentum=0.9,
                              gamma_init=1, beta_init=0, moving_mean_init=0, moving_var_init=1)
    return nn.BatchNorm2d(channel, eps=1e-4, momentum=0.9,
                          gamma_init=0, beta_init=0, moving_mean_init=0, moving_var_init=1,
                          use_batch_statistics=training)


def _fc(in_channel, out_channel):
    return nn.Dense(in_channel, out_channel, has_bias=True, weight_init="HeUniform", bias_init=0)


class BasicBlock(nn.Cell):
    """
    basic block for resnet18 and resnet34
    """
    expansion = 1

    def __init__(self,
                 in_channel,
                 out_channel,
                 stride=1,
                 training=True
                 ):
        super(BasicBlock, self).__init__()
        channel = out_channel // self.expansion
        self.conv1 = _conv3x3(in_channel, channel, stride=1)
        self.bn1 = _bn(channel, training)

        self.conv2 = _conv3x3(channel, out_channel, stride=stride)
        self.bn2 = _bn_last(channel, training)

        self.relu = nn.ReLU()

        self.down_sample = False

        if stride != 1 or in_channel != out_channel:
            self.down_sample = True
        self.down_sample_layer = None

        if self.down_sample:
            self.down_sample_layer = nn.SequentialCell([_conv1x1(in_channel, out_channel, stride),
                                                        _bn(out_channel, training)])
        self.add = P.TensorAdd()

    def construct(self, x):
        """forward function"""

        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.down_sample:
            identity = self.down_sample_layer(identity)

        out = self.add(out, identity)
        out = self.relu(out)

        return out


class ResidualBlock(nn.Cell):
    """
    ResNet V1 residual block definition.

    Args:
        in_channel (int): Input channel.
        out_channel (int): Output channel.
        stride (int): Stride size for the first convolutional layer. Default: 1.

    Returns:
        Tensor, output tensor.

    Examples:
        >>> ResidualBlock(3, 256, stride=2)
    """
    expansion = 4

    def __init__(self,
                 in_channel,
                 out_channel,
                 stride=1,
                 training=True
                 ):
        super(ResidualBlock, self).__init__()

        channel = out_channel // self.expansion
        self.conv1 = _conv1x1(in_channel, channel, stride=1)
        self.bn1 = _bn(channel, training)

        self.conv2 = _conv3x3(channel, channel, stride=stride)
        self.bn2 = _bn(channel, training)

        self.conv3 = _conv1x1(channel, out_channel, stride=1)
        self.bn3 = _bn_last(out_channel, training)

        self.relu = nn.ReLU()

        self.down_sample = False

        if stride != 1 or in_channel != out_channel:
            self.down_sample = True
        self.down_sample_layer = None

        if self.down_sample:
            self.down_sample_layer = nn.SequentialCell([_conv1x1(in_channel, out_channel, stride),
                                                        _bn(out_channel, training)])
        self.add = P.TensorAdd()

    def construct(self, x):
        """forward function"""

        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.down_sample:
            identity = self.down_sample_layer(identity)

        out = self.add(out, identity)
        out = self.relu(out)

        return out


class ResNet(nn.Cell):
    """
    ResNet architecture.

    Args:
        block (Cell): Block for network.
        layer_nums (list): Numbers of block in different layers.
        in_channels (list): Input channel in each layer.
        out_channels (list): Output channel in each layer.
        strides (list):  Stride size in each layer.
        low_dims (int): The dimension of outputThe.
    Returns:
        Tensor, output tensor.

    Examples:
        >>> ResNet(ResidualBlock,
        >>>        [3, 4, 6, 3],
        >>>        [64, 256, 512, 1024],
        >>>        [256, 512, 1024, 2048],
        >>>        [1, 2, 2, 2],
        >>>        128)
    """

    def __init__(self,
                 block,
                 layer_nums,
                 in_channels,
                 out_channels,
                 strides,
                 low_dims,
                 training_mode=True,
                 use_MLP=False):
        super(ResNet, self).__init__()

        if not len(layer_nums) == len(in_channels) == len(out_channels) == 4:
            raise ValueError("the length of layer_num, in_channels, out_channels list must be 4!")

        self.use_MLP = use_MLP
        self.training_mode = training_mode
        self.concat = P.Concat()
        self.split = P.Split(0, 3)
        self.l2norm = P.L2Normalize(axis=1)

        self.conv1 = _conv3x3(3, 64, stride=1)
        self.bn1 = _bn(64, training_mode)
        self.relu = nn.ReLU()

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

        self.mean = P.ReduceMean(keep_dims=True)
        self.flatten = nn.Flatten()
        self.end_point = _fc(block.expansion * 512, low_dims)
        self.mlp_layer1 = _fc(block.expansion * 512, block.expansion * 512)
        self.mlp_layer2 = _fc(block.expansion * 512, low_dims)

    def _make_layer(self, block, layer_num, in_channel, out_channel, stride):
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
        layers = []

        resnet_block = block(in_channel, out_channel, stride=stride, training=self.training_mode)
        layers.append(resnet_block)

        for _ in range(1, layer_num):
            resnet_block = block(out_channel, out_channel, stride=1, training=self.training_mode)
            layers.append(resnet_block)

        return nn.SequentialCell(layers)

    def construct(self, x3, x2, x1):
        """forward function"""

        x = self.concat((x3, x2, x1))
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        out = self.mean(c5, (2, 3))
        out = self.flatten(out)

        if not self.training_mode:
            out = self.l2norm(out)
            out3, out2, out1 = self.split(out)
            return out3

        if self.use_MLP:
            out = self.mlp_layer1(out)
            out = self.mlp_layer2(out)
        else:
            out = self.end_point(out)

        out = self.l2norm(out)

        out3, out2, out1 = self.split(out)
        return out3, out2, out1


def resnet50(low_dims=128, training_mode=True, use_MLP=False):
    """
    Get ResNet50 neural network.

    Args:
        class_num (int): Class number.

    Returns:
        Cell, cell instance of ResNet50 neural network.

    Examples:
        >>> net = resnet50(128)
    """
    return ResNet(ResidualBlock,
                  [3, 4, 6, 3],
                  [64, 256, 512, 1024],
                  [256, 512, 1024, 2048],
                  [1, 2, 2, 2],
                  low_dims,
                  training_mode,
                  use_MLP)


def resnet101(low_dims=128, training_mode=True, use_MLP=False):
    """
    Get ResNet101 neural network.

    Args:
        class_num (int): Class number.

    Returns:
        Cell, cell instance of ResNet101 neural network.

    Examples:
        >>> net = resnet101(128)
    """
    return ResNet(ResidualBlock,
                  [3, 4, 23, 3],
                  [64, 256, 512, 1024],
                  [256, 512, 1024, 2048],
                  [1, 2, 2, 2],
                  low_dims,
                  training_mode,
                  use_MLP)


def resnet18(low_dims=128, training_mode=True, use_MLP=False):
    """
    Get ResNet18 neural network.
    Returns:
        Cell, cell instance of ResNet18 neural network.
    Examples:
        >>> net = resnet18(128)
    """
    return ResNet(BasicBlock,
                  [2, 2, 2, 2],
                  [64, 64, 128, 256],
                  [64, 128, 256, 512],
                  [1, 2, 2, 2],
                  low_dims,
                  training_mode,
                  use_MLP)
