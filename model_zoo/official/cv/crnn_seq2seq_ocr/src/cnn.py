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
#" ============================================================================
"""
CRN-Seq2Seq-OCR CNN model.

"""

import math
import numpy as np

import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.common.tensor import Tensor


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
    if dimensions == 2:
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


def kaiming_normal(inputs_shape, gain_param=0, mode='fan_in', nonlinearity='leaky_relu'):
    fan = _calculate_correct_fan(inputs_shape, mode)
    gain = calculate_gain(nonlinearity, gain_param)
    std = gain / math.sqrt(fan)
    return np.random.normal(0, std, size=inputs_shape).astype(np.float32)


class ConvRelu(nn.Cell):
    """
    Convolution Layer followed by Relu Layer

    """
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1):
        super(ConvRelu, self).__init__()
        shape = (out_channels, in_channels, kernel_size[0], kernel_size[1])
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size,
                              stride,
                              weight_init=Tensor(kaiming_normal(shape)))
        self.relu = nn.ReLU()

    def construct(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


class ConvBNRelu(nn.Cell):
    """
    Convolution Layer followed by Batch Normalization and Relu Layer

    """
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, pad_mode='same'):
        super(ConvBNRelu, self).__init__()
        shape = (out_channels, in_channels, kernel_size[0], kernel_size[1])
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size, stride,
                              pad_mode=pad_mode,
                              weight_init=Tensor(kaiming_normal(shape)))
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def construct(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class CNN(nn.Cell):
    """
    CNN Class for OCR

    """

    def __init__(self, conv_out_dim):
        super(CNN, self).__init__()
        self.convRelu1 = ConvRelu(3, 64, (3, 3))
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.convRelu2 = ConvRelu(64, 128, (3, 3))
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))

        self.convBNRelu1 = ConvBNRelu(128, 256, (3, 3))
        self.convRelu3 = ConvRelu(256, 256, (3, 3))
        self.maxpool3 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))

        self.convBNRelu2 = ConvBNRelu(256, 384, (3, 3))
        self.convRelu4 = ConvRelu(384, 384, (3, 3))
        self.maxpool4 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))

        self.convBNRelu3 = ConvBNRelu(384, 384, (3, 3))
        self.convRelu5 = ConvRelu(384, 384, (3, 3))
        self.maxpool5 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))

        self.convBNRelu4 = ConvBNRelu(384, 384, (3, 3))
        self.convRelu6 = ConvRelu(384, 384, (3, 3))
        self.maxpool6 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))

        self.pad = nn.Pad(paddings=((0, 0), (0, 0), (0, 0), (0, 1)))
        self.convBNRelu5 = ConvBNRelu(384, conv_out_dim, (2, 2), pad_mode='valid')
        self.dropout = nn.Dropout(keep_prob=0.5)

        self.squeeze = P.Squeeze(2)
        self.cast = P.Cast()

    def construct(self, x):
        x = self.convRelu1(x)
        x = self.maxpool1(x)

        x = self.convRelu2(x)
        x = self.maxpool2(x)

        x = self.convBNRelu1(x)
        x = self.convRelu3(x)
        x = self.maxpool3(x)

        x = self.convBNRelu2(x)
        x = self.convRelu4(x)
        x = self.maxpool4(x)

        x = self.convBNRelu3(x)
        x = self.convRelu5(x)
        x = self.maxpool5(x)

        x = self.convBNRelu4(x)
        x = self.convRelu6(x)
        x = self.maxpool6(x)

        x = self.pad(x)
        x = self.convBNRelu5(x)
        x = self.dropout(x)
        x = self.squeeze(x)

        return x
