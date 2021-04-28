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
"""GENet."""

import math
import numpy as np
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.common.tensor import Tensor
from src.GEBlock import GEBlock

def calculate_gain(nonlinearity, param=None):
    """calculate_gain"""
    linear_fns = ['linear', 'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d',
                  'conv_transpose2d', 'conv_transpose3d']
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
    """
    _calculate_fan_in_and_fan_out
    """
    dimensions = len(tensor)
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor"
                         " with fewer than 2 dimensions")
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
    """
        for pylint.
    """
    mode = mode.lower()
    valid_modes = ['fan_in', 'fan_out']
    if mode not in valid_modes:
        raise ValueError("Mode {} not supported, please use one of {}".format(mode, valid_modes))
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    return fan_in if mode == 'fan_in' else fan_out


def kaiming_normal(inputs_shape, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    """
        for pylint.
    """
    fan = _calculate_correct_fan(inputs_shape, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    return np.random.normal(0, std, size=inputs_shape).astype(np.float32)


def kaiming_uniform(inputs_shape, a=0., mode='fan_in', nonlinearity='leaky_relu'):
    """
        for pylint.
    """
    fan = _calculate_correct_fan(inputs_shape, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    return np.random.uniform(-bound, bound, size=inputs_shape).astype(np.float32)


def _conv3x3(in_channel, out_channel, stride=1):

    weight_shape = (out_channel, in_channel, 3, 3)
    weight = Tensor(kaiming_normal(weight_shape, mode="fan_out", nonlinearity='relu'))

    return nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride,
                     padding=0, pad_mode='same', weight_init=weight)


def _conv1x1(in_channel, out_channel, stride=1):

    weight_shape = (out_channel, in_channel, 1, 1)
    weight = Tensor(kaiming_normal(weight_shape, mode="fan_out", nonlinearity='relu'))

    return nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride,
                     padding=0, pad_mode='same', weight_init=weight)


def _conv7x7(in_channel, out_channel, stride=1):

    weight_shape = (out_channel, in_channel, 7, 7)
    weight = Tensor(kaiming_normal(weight_shape, mode="fan_out", nonlinearity='relu'))
    return nn.Conv2d(in_channel, out_channel,
                     kernel_size=7, stride=stride, padding=0, pad_mode='same', weight_init=weight)


def _bn(channel):
    return nn.BatchNorm2d(channel, eps=1e-4, momentum=0.95,
                          gamma_init=1, beta_init=0, moving_mean_init=0, moving_var_init=1)


def _bn_last(channel):
    return nn.BatchNorm2d(channel, eps=1e-4, momentum=0.95,
                          gamma_init=0, beta_init=0, moving_mean_init=0, moving_var_init=1)


def _fc(in_channel, out_channel):
    weight_shape = (out_channel, in_channel)
    weight = Tensor(kaiming_uniform(weight_shape, a=math.sqrt(5)))
    return nn.Dense(in_channel, out_channel, has_bias=True, weight_init=weight, bias_init=0)


class GENet(nn.Cell):
    """
    GENet architecture.

    Args:
        block (Cell): Block for network.
        layer_nums (list): Numbers of block in different layers.
        in_channels (list): Input channel in each layer.
        out_channels (list): Output channel in each layer.
        strides (list):  Stride size in each layer.
        spatial(list):   Numbers of output spatial size of different groups.
        num_classes (int): The number of classes that the training images are belonging to.
        extra_params(bool)    : Whether to use DW Conv to down-sample
        mlp(bool)      : Whether to combine SENet (using 1*1 conv)

    Returns:
        Tensor, output tensor.

    Examples:
        >>> GENet(GEBlock,
        >>>        [3, 4, 6, 3],
        >>>        [64, 256, 512, 1024],
        >>>        [256, 512, 1024, 2048],
        >>>        [1, 2, 2, 2],
        >>>        [56,28,14,7]
        >>>        1001,True,True)
    """

    def __init__(self,
                 block,
                 layer_nums,
                 in_channels,
                 out_channels,
                 strides,
                 spatial,
                 num_classes,
                 extra_params,
                 mlp):
        super(GENet, self).__init__()

        if not len(layer_nums) == len(in_channels) == len(out_channels) == 4:
            raise ValueError("the length of layer_num, in_channels, out_channels list must be 4!")
        self.extra = extra_params

        # initial stage
        self.conv1 = _conv7x7(3, 64, stride=2)
        self.bn1 = _bn(64)
        self.relu = P.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="same")

        self.layer1 = self._make_layer(block=block,
                                       layer_num=layer_nums[0],
                                       in_channel=in_channels[0],
                                       out_channel=out_channels[0],
                                       stride=strides[0],
                                       spatial=spatial[0],
                                       extra_params=extra_params,
                                       mlp=mlp)
        self.layer2 = self._make_layer(block=block,
                                       layer_num=layer_nums[1],
                                       in_channel=in_channels[1],
                                       out_channel=out_channels[1],
                                       stride=strides[1],
                                       spatial=spatial[1],
                                       extra_params=extra_params,
                                       mlp=mlp)
        self.layer3 = self._make_layer(block=block,
                                       layer_num=layer_nums[2],
                                       in_channel=in_channels[2],
                                       out_channel=out_channels[2],
                                       stride=strides[2],
                                       spatial=spatial[2],
                                       extra_params=extra_params,
                                       mlp=mlp)
        self.layer4 = self._make_layer(block=block,
                                       layer_num=layer_nums[3],
                                       in_channel=in_channels[3],
                                       out_channel=out_channels[3],
                                       stride=strides[3],
                                       spatial=spatial[3],
                                       extra_params=extra_params,
                                       mlp=mlp)

        self.mean = P.ReduceMean(keep_dims=True)
        self.flatten = nn.Flatten()
        self.end_point = _fc(out_channels[3], num_classes)

    def _make_layer(self, block, layer_num, in_channel, out_channel,
                    stride, spatial, extra_params, mlp):
        """
        Make stage network of GENet.

        Args:
            block (Cell): GENet block.
            layer_num (int): Layer number.
            in_channel (int): Input channel.
            out_channel (int): Output channel.
            stride (int): Stride size for the first convolutional layer.
            spatial(int):   output spatial size of every block in same group.
            extra_params(bool)    : Whether to use DW Conv to down-sample
            mlp(bool)      : Whether to combine SENet (using 1*1 conv)
        Returns:
            SequentialCell, the output layer.

        """
        layers = []

        ge_block = block(in_channel=in_channel,
                         out_channel=out_channel,
                         stride=stride,
                         spatial=spatial,
                         extra_params=extra_params,
                         mlp=mlp)
        layers.append(ge_block)
        for _ in range(1, layer_num):
            ge_block = block(in_channel=out_channel,
                             out_channel=out_channel,
                             stride=1,
                             spatial=spatial,
                             extra_params=extra_params,
                             mlp=mlp)
            layers.append(ge_block)
        return nn.SequentialCell(layers)

    def construct(self, x):
        """
        Args:
            x : input Tensor.
        """
        # initial stage
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        c1 = self.maxpool(x)

        # four groups
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        out = self.mean(c5, (2, 3))
        out = self.flatten(out)
        out = self.end_point(out)

        return out

def GE_resnet50(class_num=1000, extra=True, mlp=True):
    """
    Get GE-ResNet50 neural network.
    Default : GE Theta+ version (best)

    Args:
        class_num (int): Class number.
        extra(bool)    : Whether to use DW Conv to down-sample
        mlp(bool)      : Whether to combine SENet (using 1*1 conv)
    Returns:
        Cell, cell instance of GENet-ResNet50 neural network.

    Examples:
        >>> net = GE_resnet50(1000)
    """

    return GENet(block=GEBlock,
                 layer_nums=[3, 4, 6, 3],
                 in_channels=[64, 256, 512, 1024],
                 out_channels=[256, 512, 1024, 2048],
                 strides=[1, 2, 2, 2],
                 spatial=[56, 28, 14, 7],
                 num_classes=class_num,
                 extra_params=extra,
                 mlp=mlp)
