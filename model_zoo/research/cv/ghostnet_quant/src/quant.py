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
"""Quantization define"""
import mindspore as ms
import mindspore.nn as nn
from mindspore import Parameter, Tensor
from mindspore.ops import operations as P
from mindspore.ops import composite as C
from mindspore.common.initializer import initializer

#------weight symmetric, activation asymmetric------#


class QuanConv(nn.Conv2d):
    r"""Conv for quantization"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, pad_mode='same',
                 padding=0, dilation=1, group=1, has_bias=True):
        super(QuanConv, self).__init__(in_channels, out_channels,
                                       kernel_size, stride, pad_mode, padding, dilation, group, has_bias)
        self.floor = P.Floor()
        self.expand_dims = P.ExpandDims()
        self.x_lower_bound = Tensor(0, ms.float32)
        self.x_upper_bound = Tensor(2 ** 8 - 1, ms.float32)
        self.w_lower_bound = Tensor(-2 ** 7 - 1, ms.float32)
        self.w_upper_bound = Tensor(2 ** 7, ms.float32)
        self.scale_a = Parameter(initializer('ones', [1]))
        self.scale_w = Parameter(initializer(
            'ones', [out_channels]))
        self.zp_a = Parameter(initializer('ones', [1]))

    def construct(self, in_data):
        r"""construct of QuantConv"""
        x = self.floor(in_data / self.scale_a - self.zp_a + 0.5)
        x = C.clip_by_value(x, self.x_lower_bound, self.x_upper_bound)
        x = (x + self.zp_a) * self.scale_a

        exp_dim_scale_w = self.scale_w
        exp_dim_scale_w = self.expand_dims(exp_dim_scale_w, 1)
        exp_dim_scale_w = self.expand_dims(exp_dim_scale_w, 2)
        exp_dim_scale_w = self.expand_dims(exp_dim_scale_w, 3)
        w = self.floor(self.weight / exp_dim_scale_w + 0.5)
        w = C.clip_by_value(w, self.w_lower_bound, self.w_upper_bound)
        w = w * exp_dim_scale_w

        # forward
        output = self.conv2d(x, w)
        if self.has_bias:
            output = self.bias_add(output, self.bias)
        return output
