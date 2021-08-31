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
"""network layers initialization"""

import math

import mindspore.nn as nn
from mindspore.common.initializer import initializer, HeUniform, Uniform


def calculate_fan_in_and_fan_out(shape):
    """
    calculate fan_in and fan_out

    Args:
        shape (tuple): input shape.

    Returns:
        Tuple, a tuple with two elements, the first element is `n_in` and the second element is `n_out`.
    """
    dimensions = len(shape)
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")
    if dimensions == 2:  # Linear
        fan_in = shape[1]
        fan_out = shape[0]
    else:
        num_input_fmaps = shape[1]
        num_output_fmaps = shape[0]
        receptive_field_size = 1
        if dimensions > 2:
            receptive_field_size = shape[2] * shape[3]
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size
    return fan_in, fan_out


class Dense(nn.Dense):
    """Dense"""

    def __init__(self, in_channels, out_channels, has_bias=True, activation=None):
        super(Dense, self).__init__(in_channels, out_channels, weight_init='normal', bias_init='zeros',
                                    has_bias=has_bias, activation=activation)
        self.reset_parameters()

    def reset_parameters(self):
        """reset parameters"""
        self.weight.set_data(initializer(HeUniform(math.sqrt(5)), self.weight.shape))
        if self.has_bias:
            fan_in, _ = calculate_fan_in_and_fan_out(self.weight.shape)
            bound = 1 / math.sqrt(fan_in)
            self.bias.set_data(initializer(Uniform(bound), [self.out_channels]))


class Conv2d(nn.Conv2d):
    """Conv2d"""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, pad_mode='same', padding=0, dilation=1,
                 group=1, has_bias=True):
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride, pad_mode, padding, dilation, group,
                                     has_bias, weight_init='normal', bias_init='zeros')
        self.reset_parameters()

    def reset_parameters(self):
        """reset parameters"""
        self.weight.set_data(initializer(HeUniform(math.sqrt(5)), self.weight.shape))
        if self.has_bias:
            fan_in, _ = calculate_fan_in_and_fan_out(self.weight.shape)
            bound = 1 / math.sqrt(fan_in)
            self.bias.set_data(initializer(Uniform(bound), [self.out_channels]))
