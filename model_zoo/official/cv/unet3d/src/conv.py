# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# less required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import mindspore.nn as nn
from mindspore import Parameter
from mindspore import dtype as mstype
from mindspore.ops import operations as P
from mindspore.ops.operations import nn_ops as nps
from mindspore.common.initializer import initializer

def weight_variable(shape):
    init_value = initializer('Normal', shape, mstype.float32)
    return Parameter(init_value)

class Conv3D(nn.Cell):
    def __init__(self,
                 in_channel,
                 out_channel,
                 kernel_size,
                 mode=1,
                 pad_mode="valid",
                 pad=0,
                 stride=1,
                 dilation=1,
                 group=1,
                 data_format="NCDHW",
                 bias_init="zeros",
                 has_bias=True):
        super().__init__()
        self.weight_shape = (out_channel, in_channel, kernel_size[0], kernel_size[1], kernel_size[2])
        self.weight = weight_variable(self.weight_shape)
        self.conv = nps.Conv3D(out_channel=out_channel, kernel_size=kernel_size, mode=mode, \
                               pad_mode=pad_mode, pad=pad, stride=stride, dilation=dilation, \
                               group=group, data_format=data_format)
        self.bias_init = bias_init
        self.has_bias = has_bias
        self.bias_add = P.BiasAdd(data_format=data_format)
        if self.has_bias:
            self.bias = Parameter(initializer(self.bias_init, [out_channel]), name='bias')

    def construct(self, x):
        output = self.conv(x, self.weight)
        if self.has_bias:
            output = self.bias_add(output, self.bias)
        return output

class Conv3DTranspose(nn.Cell):
    def __init__(self,
                 in_channel,
                 out_channel,
                 kernel_size,
                 mode=1,
                 pad=0,
                 stride=1,
                 dilation=1,
                 group=1,
                 output_padding=0,
                 data_format="NCDHW",
                 bias_init="zeros",
                 has_bias=True):
        super().__init__()
        self.weight_shape = (in_channel, out_channel, kernel_size[0], kernel_size[1], kernel_size[2])
        self.weight = weight_variable(self.weight_shape)
        self.conv_transpose = nps.Conv3DTranspose(in_channel=in_channel, out_channel=out_channel,\
                                                  kernel_size=kernel_size, mode=mode, pad=pad, stride=stride, \
                                                  dilation=dilation, group=group, output_padding=output_padding, \
                                                  data_format=data_format)
        self.bias_init = bias_init
        self.has_bias = has_bias
        self.bias_add = P.BiasAdd(data_format=data_format)
        if self.has_bias:
            self.bias = Parameter(initializer(self.bias_init, [out_channel]), name='bias')

    def construct(self, x):
        output = self.conv_transpose(x, self.weight)
        if self.has_bias:
            output = self.bias_add(output, self.bias)
        return output
