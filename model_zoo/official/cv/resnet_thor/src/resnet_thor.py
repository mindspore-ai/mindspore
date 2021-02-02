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
from mindspore.common.tensor import Tensor
from mindspore.ops import operations as P
from mindspore import context

from src.thor_layer import Conv2d_Thor, Dense_Thor, Conv2d_Thor_GPU, Dense_Thor_GPU


def calculate_gain(nonlinearity, param=None):
    """calculate_gain"""
    linear_fns = ['linear', 'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d', 'conv_transpose2d', 'conv_transpose3d']
    res = 0
    if nonlinearity in linear_fns or nonlinearity == 'sigmoid':
        res = 1
    elif nonlinearity == 'tanh':
        res = 5.0 / 3
    elif nonlinearity == 'relu':
        res = math.sqrt(2.0)
    elif nonlinearity == 'leaky_relu':
        if param is None:
            negative_slope = 0.01
        elif not isinstance(param, bool) and isinstance(param, int) or isinstance(param, float):
            # True/False are instances of int, hence check above
            negative_slope = param
        else:
            raise ValueError("negative_slope {} not a valid number".format(param))
        res = math.sqrt(2.0 / (1 + negative_slope ** 2))
    else:
        raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))
    return res


def _calculate_fan_in_and_fan_out(tensor):
    """_calculate_fan_in_and_fan_out"""
    dimensions = len(tensor)
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")
    if dimensions == 2:  # Linear
        fan_in = tensor[1]
        fan_out = tensor[0]
    else:
        num_input_fmaps = tensor[1]
        num_output_fmaps = tensor[0]
        receptive_field_size = 1
        if dimensions > 2:
            receptive_field_size = tensor[2] * tensor[3]
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size
    return fan_in, fan_out


def _calculate_correct_fan(tensor, mode):
    mode = mode.lower()
    valid_modes = ['fan_in', 'fan_out']
    if mode not in valid_modes:
        raise ValueError("Mode {} not supported, please use one of {}".format(mode, valid_modes))
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    return fan_in if mode == 'fan_in' else fan_out


def kaiming_normal(inputs_shape, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    fan = _calculate_correct_fan(inputs_shape, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    return np.random.normal(0, std, size=inputs_shape).astype(np.float32)


def kaiming_uniform(inputs_shape, a=0., mode='fan_in', nonlinearity='leaky_relu'):
    fan = _calculate_correct_fan(inputs_shape, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    return np.random.uniform(-bound, bound, size=inputs_shape).astype(np.float32)


def _weight_variable(shape, factor=0.01):
    init_value = np.random.randn(*shape).astype(np.float32) * factor
    return Tensor(init_value)


def _conv3x3(in_channel, out_channel, stride=1, damping=0.03, loss_scale=1, frequency=278, batch_size=32):
    weight_shape = (out_channel, in_channel, 3, 3)
    weight = Tensor(kaiming_normal(weight_shape, mode="fan_out", nonlinearity='relu'))
    if context.get_context('device_target') == "Ascend":
        layer = Conv2d_Thor(in_channel, out_channel,
                            kernel_size=3, stride=stride, padding=0, pad_mode='same', weight_init=weight,
                            damping=damping, loss_scale=loss_scale, frequency=frequency, batch_size=batch_size)
    else:
        layer = Conv2d_Thor_GPU(in_channel, out_channel,
                                kernel_size=3, stride=stride, padding=0, pad_mode='same', weight_init=weight,
                                damping=damping, loss_scale=loss_scale, frequency=frequency, batch_size=batch_size)
    return layer


def _conv1x1(in_channel, out_channel, stride=1, damping=0.03, loss_scale=1, frequency=278, batch_size=32):
    weight_shape = (out_channel, in_channel, 1, 1)
    weight = Tensor(kaiming_normal(weight_shape, mode="fan_out", nonlinearity='relu'))
    if context.get_context('device_target') == "Ascend":
        layer = Conv2d_Thor(in_channel, out_channel,
                            kernel_size=1, stride=stride, padding=0, pad_mode='same', weight_init=weight,
                            damping=damping, loss_scale=loss_scale, frequency=frequency, batch_size=batch_size)
    else:
        layer = Conv2d_Thor_GPU(in_channel, out_channel,
                                kernel_size=1, stride=stride, padding=0, pad_mode='same', weight_init=weight,
                                damping=damping, loss_scale=loss_scale, frequency=frequency, batch_size=batch_size)
    return layer


def _conv7x7(in_channel, out_channel, stride=1, damping=0.03, loss_scale=1, frequency=278, batch_size=32):
    weight_shape = (out_channel, in_channel, 7, 7)
    weight = Tensor(kaiming_normal(weight_shape, mode="fan_out", nonlinearity='relu'))
    if context.get_context('device_target') == "Ascend":
        layer = Conv2d_Thor(in_channel, out_channel,
                            kernel_size=7, stride=stride, padding=0, pad_mode='same', weight_init=weight,
                            damping=damping, loss_scale=loss_scale, frequency=frequency, batch_size=batch_size)
    else:
        layer = Conv2d_Thor_GPU(in_channel, out_channel,
                                kernel_size=7, stride=stride, padding=0, pad_mode='same', weight_init=weight,
                                damping=damping, loss_scale=loss_scale, frequency=frequency, batch_size=batch_size)
    return layer


def _bn(channel):
    return nn.BatchNorm2d(channel, eps=1e-4, momentum=0.9,
                          gamma_init=1, beta_init=0, moving_mean_init=0, moving_var_init=1)


def _bn_last(channel):
    return nn.BatchNorm2d(channel, eps=1e-4, momentum=0.9,
                          gamma_init=0, beta_init=0, moving_mean_init=0, moving_var_init=1)


def _fc(in_channel, out_channel, damping, loss_scale, frequency, batch_size=32):
    weight_shape = (out_channel, in_channel)
    weight = Tensor(kaiming_uniform(weight_shape, a=math.sqrt(5)))
    if context.get_context('device_target') == "Ascend":
        layer = Dense_Thor(in_channel, out_channel, has_bias=False, weight_init=weight,
                           bias_init=0, damping=damping, loss_scale=loss_scale, frequency=frequency,
                           batch_size=batch_size)
    else:
        layer = Dense_Thor_GPU(in_channel, out_channel, has_bias=False, weight_init=weight,
                               bias_init=0, damping=damping, loss_scale=loss_scale, frequency=frequency,
                               batch_size=batch_size)
    return layer


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
                 damping=0.03,
                 loss_scale=1,
                 frequency=278,
                 batch_size=32):
        super(ResidualBlock, self).__init__()

        channel = out_channel // self.expansion
        self.conv1 = _conv1x1(in_channel, channel, stride=1, damping=damping, loss_scale=loss_scale,
                              frequency=frequency, batch_size=batch_size)
        self.bn1 = _bn(channel)

        self.conv2 = _conv3x3(channel, channel, stride=stride, damping=damping, loss_scale=loss_scale,
                              frequency=frequency, batch_size=batch_size)
        self.bn2 = _bn(channel)

        self.conv3 = _conv1x1(channel, out_channel, stride=1, damping=damping, loss_scale=loss_scale,
                              frequency=frequency, batch_size=batch_size)
        self.bn3 = _bn_last(out_channel)

        self.relu = nn.ReLU()

        self.down_sample = False

        if stride != 1 or in_channel != out_channel:
            self.down_sample = True
        self.down_sample_layer = None

        if self.down_sample:
            self.down_sample_layer = nn.SequentialCell([_conv1x1(in_channel, out_channel, stride,
                                                                 damping=damping, loss_scale=loss_scale,
                                                                 frequency=frequency,
                                                                 batch_size=batch_size),
                                                        _bn(out_channel)])
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
                 block,
                 layer_nums,
                 in_channels,
                 out_channels,
                 strides,
                 num_classes,
                 damping,
                 loss_scale,
                 frequency,
                 batch_size,
                 include_top=True):
        super(ResNet, self).__init__()

        if not len(layer_nums) == len(in_channels) == len(out_channels) == 4:
            raise ValueError("the length of layer_num, in_channels, out_channels list must be 4!")

        self.conv1 = _conv7x7(3, 64, stride=2, damping=damping, loss_scale=loss_scale,
                              frequency=frequency, batch_size=batch_size)
        self.bn1 = _bn(64)
        self.relu = P.ReLU()

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="same")

        self.layer1 = self._make_layer(block,
                                       layer_nums[0],
                                       in_channel=in_channels[0],
                                       out_channel=out_channels[0],
                                       stride=strides[0],
                                       damping=damping,
                                       loss_scale=loss_scale,
                                       frequency=frequency,
                                       batch_size=batch_size)
        self.layer2 = self._make_layer(block,
                                       layer_nums[1],
                                       in_channel=in_channels[1],
                                       out_channel=out_channels[1],
                                       stride=strides[1],
                                       damping=damping,
                                       loss_scale=loss_scale,
                                       frequency=frequency,
                                       batch_size=batch_size)
        self.layer3 = self._make_layer(block,
                                       layer_nums[2],
                                       in_channel=in_channels[2],
                                       out_channel=out_channels[2],
                                       stride=strides[2], damping=damping,
                                       loss_scale=loss_scale,
                                       frequency=frequency,
                                       batch_size=batch_size)
        self.layer4 = self._make_layer(block,
                                       layer_nums[3],
                                       in_channel=in_channels[3],
                                       out_channel=out_channels[3],
                                       stride=strides[3],
                                       damping=damping,
                                       loss_scale=loss_scale,
                                       frequency=frequency,
                                       batch_size=batch_size)
        self.include_top = include_top
        if self.include_top:
            self.mean = P.ReduceMean(keep_dims=True)
            self.flatten = nn.Flatten()
            self.end_point = _fc(out_channels[3], num_classes, damping=damping, loss_scale=loss_scale,
                                 frequency=frequency, batch_size=batch_size)

    def _make_layer(self, block, layer_num, in_channel, out_channel, stride,
                    damping, loss_scale, frequency, batch_size):
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

        resnet_block = block(in_channel, out_channel, stride=stride,
                             damping=damping, loss_scale=loss_scale, frequency=frequency,
                             batch_size=batch_size)
        layers.append(resnet_block)

        for _ in range(1, layer_num):
            resnet_block = block(out_channel, out_channel, stride=1,
                                 damping=damping, loss_scale=loss_scale, frequency=frequency,
                                 batch_size=batch_size)
            layers.append(resnet_block)

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

        if not self.include_top:
            return x

        out = self.mean(c5, (2, 3))
        out = self.flatten(out)
        out = self.end_point(out)

        return out


def resnet50(class_num=10, damping=0.03, loss_scale=1, frequency=278, batch_size=32, include_top=True):
    """
    Get ResNet50 neural network.

    Args:
        class_num (int): Class number.

    Returns:
        Cell, cell instance of ResNet50 neural network.

    Examples:
        >>> net = resnet50(10)
    """
    return ResNet(ResidualBlock,
                  [3, 4, 6, 3],
                  [64, 256, 512, 1024],
                  [256, 512, 1024, 2048],
                  [1, 2, 2, 2],
                  class_num,
                  damping,
                  loss_scale,
                  frequency,
                  batch_size,
                  include_top)
