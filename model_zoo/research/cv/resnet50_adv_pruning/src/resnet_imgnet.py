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
import math
import numpy as np
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.common.tensor import Tensor


def _weight_variable(shape, factor=0.01):
    init_value = np.random.randn(*shape).astype(np.float32) * factor
    return Tensor(init_value)


def _conv3x3(in_channel, out_channel, stride=1):
    weight_shape = (out_channel, in_channel, 3, 3)
    weight = _weight_variable(weight_shape)
    return nn.Conv2d(in_channel, out_channel,
                     kernel_size=3, stride=stride, padding=1, pad_mode='pad', weight_init=weight)


def _conv1x1(in_channel, out_channel, stride=1):
    weight_shape = (out_channel, in_channel, 1, 1)
    weight = _weight_variable(weight_shape)
    return nn.Conv2d(in_channel, out_channel,
                     kernel_size=1, stride=stride, padding=0, pad_mode='same', weight_init=weight)


def _conv7x7(in_channel, out_channel, stride=1):
    weight_shape = (out_channel, in_channel, 7, 7)
    weight = _weight_variable(weight_shape)
    return nn.Conv2d(in_channel, out_channel,
                     kernel_size=7, stride=stride, padding=3, pad_mode='pad', weight_init=weight)


def _bn(channel):
    return nn.BatchNorm2d(channel)


def _bn_last(channel):
    return nn.BatchNorm2d(channel)


def _fc(in_channel, out_channel):
    weight_shape = (out_channel, in_channel)
    weight = _weight_variable(weight_shape)
    return nn.Dense(in_channel, out_channel, has_bias=True, weight_init=weight, bias_init=0)


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
                 rate,
                 stride=1,
                 num=0,
                 index=None):
        super(ResidualBlock, self).__init__()

        channel = out_channel // self.expansion
        prune_channel = math.ceil(channel*rate)
        self.conv1 = _conv1x1(in_channel, prune_channel, stride=1)
        self.bn1 = _bn(prune_channel)

        self.conv2 = _conv3x3(prune_channel, prune_channel, stride=stride)
        self.bn2 = _bn(prune_channel)

        self.conv3 = _conv1x1(prune_channel, math.ceil(
            channel*rate*self.expansion), stride=1)
        self.bn3 = _bn_last(math.ceil(channel*rate*self.expansion))

        self.relu = nn.ReLU()

        self.down_sample = False

        if stride != 1 or in_channel != out_channel:
            self.down_sample = True
        self.down_sample_layer = None

        if self.down_sample:
            self.down_sample_layer = nn.SequentialCell([_conv1x1(in_channel, out_channel, stride),
                                                        _bn(out_channel)])
        self.add = P.Add()

        self.op = P.ScatterNd()
        self.transpose = P.Transpose()
        self.idx = None
        self.shape = None
        if out_channel == 256:
            self.shape = (out_channel, 1, 56, 56)
            if num == 0:
                self.idx = index[0]
            elif num == 1:
                self.idx = index[1]
            elif num == 2:
                self.idx = index[2]
        elif out_channel == 512:
            self.shape = (out_channel, 1, 28, 28)
            if num == 0:
                self.idx = index[3]
            elif num == 1:
                self.idx = index[4]
            elif num == 2:
                self.idx = index[5]
            elif num == 3:
                self.idx = index[6]
        elif out_channel == 1024:
            self.shape = (out_channel, 1, 14, 14)
            if num == 0:
                self.idx = index[7]
            elif num == 1:
                self.idx = index[8]
            elif num == 2:
                self.idx = index[9]
            elif num == 3:
                self.idx = index[10]
            elif num == 4:
                self.idx = index[11]
            elif num == 5:
                self.idx = index[12]
        elif out_channel == 2048:
            self.shape = (out_channel, 1, 7, 7)
            if num == 0:
                self.idx = index[13]
            elif num == 1:
                self.idx = index[14]
            elif num == 2:
                self.idx = index[15]

    def construct(self, x):
        r"""construct of ResidualBlock"""
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

        output = self.transpose(out, (1, 0, 2, 3))
        output = self.op(self.idx, output, self.shape)
        output = self.transpose(output, (1, 0, 2, 3))

        output = self.add(identity, output)
        output = self.relu(output)

        return output


class ResNet(nn.Cell):
    """
    ResNet architecture.

    Args:
        block (Cell): Block for network.
        layer_nums (list): Numbers of block in different layers.
        in_channels (list): Input channel in each layer.
        out_channels (list): Output channel in each layer.
        strides (list):  Stride size in each layer.
        num_classes (int): The number of classes that the training images are belonging to.
    Returns:
        Tensor, output tensor.

    Examples:
        >>> ResNet(ResidualBlock,
        >>>        [3, 4, 6, 3],
        >>>        [64, 256, 512, 1024],
        >>>        [256, 512, 1024, 2048],
        >>>        [1, 2, 2, 2],
        >>>        10)
    """

    def __init__(self,
                 rate,
                 block,
                 layer_nums,
                 in_channels,
                 out_channels,
                 strides,
                 num_classes,
                 index):
        super(ResNet, self).__init__()

        if not len(layer_nums) == len(in_channels) == len(out_channels) == 4:
            raise ValueError(
                "the length of layer_num, in_channels, out_channels list must be 4!")

        self.conv1 = _conv7x7(3, 64, stride=2)
        self.bn1 = _bn(64)
        self.relu = P.ReLU()
        self.pad_op = P.Pad(((0, 0), (0, 0), (1, 1), (1, 1)))
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="valid")

        self.layer1 = self._make_layer(block,
                                       rate,
                                       layer_nums[0],
                                       in_channel=in_channels[0],
                                       out_channel=out_channels[0],
                                       stride=strides[0],
                                       index=index)
        self.layer2 = self._make_layer(block,
                                       rate,
                                       layer_nums[1],
                                       in_channel=in_channels[1],
                                       out_channel=out_channels[1],
                                       stride=strides[1],
                                       index=index)
        self.layer3 = self._make_layer(block,
                                       rate,
                                       layer_nums[2],
                                       in_channel=in_channels[2],
                                       out_channel=out_channels[2],
                                       stride=strides[2],
                                       index=index)
        self.layer4 = self._make_layer(block,
                                       rate,
                                       layer_nums[3],
                                       in_channel=in_channels[3],
                                       out_channel=out_channels[3],
                                       stride=strides[3],
                                       index=index)

        self.mean = P.ReduceMean(keep_dims=True)
        self.flatten = nn.Flatten()
        self.end_point = _fc(out_channels[3], num_classes)

        self._initialize_weights()

    def _make_layer(self, block, rate, layer_num, in_channel, out_channel, stride, index):
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

        resnet_block = block(in_channel, out_channel, rate,
                             stride=stride, num=0, index=index)
        layers.append(resnet_block)

        for _ in range(1, layer_num):
            resnet_block = block(out_channel, out_channel,
                                 rate, stride=1, num=_, index=index)
            layers.append(resnet_block)

        return nn.SequentialCell(layers)

    def construct(self, x):
        r"""construct of ResNet"""
        x1 = self.conv1(x)
        x2 = self.bn1(x1)
        x3 = self.relu(x2)
        x4 = self.pad_op(x3)
        c1 = self.maxpool(x4)

        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        out = self.mean(c5, (2, 3))
        out = self.flatten(out)
        out = self.end_point(out)
        return out

    def _initialize_weights(self):
        """
        Initialize weights.

        Args:

        Returns:
            None.

        Examples:
            >>> _initialize_weights()
        """
        self.init_parameters_data()
        for _, m in self.cells_and_names():
            if isinstance(m, (nn.Conv2d)):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.set_data(Tensor(np.random.normal(0, np.sqrt(2. / n),
                                                          m.weight.data.shape).astype("float32")))
                if m.bias is not None:
                    m.bias.set_data(
                        Tensor(np.zeros(m.bias.data.shape, dtype="float32")))
            elif isinstance(m, nn.BatchNorm2d):
                m.gamma.set_data(
                    Tensor(np.ones(m.gamma.data.shape, dtype="float32")))
                m.beta.set_data(
                    Tensor(np.zeros(m.beta.data.shape, dtype="float32")))
            elif isinstance(m, nn.Dense):
                m.weight.set_data(Tensor(np.random.normal(
                    0, 0.01, m.weight.data.shape).astype("float32")))
                if m.bias is not None:
                    m.bias.set_data(
                        Tensor(np.zeros(m.bias.data.shape, dtype="float32")))


def resnet50(rate=1, class_num=10, index=None):
    """
    Get ResNet50 neural network.

    Args:
        class_num (int): Class number.

    Returns:
        Cell, cell instance of ResNet50 neural network.

    Examples:
        >>> net = resnet50(10)
    """
    return ResNet(rate,
                  ResidualBlock,
                  [3, 4, 6, 3],
                  [64, 256, 512, 1024],
                  [256, 512, 1024, 2048],
                  [1, 2, 2, 2],
                  class_num,
                  index)


def resnet101(rate=1, class_num=10, index=None):
    """
    Get ResNet101 neural network.

    Args:
        class_num (int): Class number.

    Returns:
        Cell, cell instance of ResNet101 neural network.

    Examples:
        >>> net = resnet101(1001)
    """
    return ResNet(rate,
                  ResidualBlock,
                  [3, 4, 23, 3],
                  [64, 256, 512, 1024],
                  [256, 512, 1024, 2048],
                  [1, 2, 2, 2],
                  class_num,
                  index)
