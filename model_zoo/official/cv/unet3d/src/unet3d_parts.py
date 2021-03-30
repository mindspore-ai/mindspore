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
from mindspore import dtype as mstype
from mindspore.ops import operations as P

class BatchNorm3d(nn.Cell):
    def __init__(self, num_features):
        super().__init__()
        self.reshape = P.Reshape()
        self.shape = P.Shape()
        self.bn2d = nn.BatchNorm2d(num_features, data_format="NCHW")

    def construct(self, x):
        x_shape = self.shape(x)
        x = self.reshape(x, (x_shape[0], x_shape[1], x_shape[2] * x_shape[3], x_shape[4]))
        bn2d_out = self.bn2d(x)
        bn3d_out = self.reshape(bn2d_out, x_shape)
        return bn3d_out

class ResidualUnit(nn.Cell):
    def __init__(self, in_channel, out_channel, stride=2, kernel_size=(3, 3, 3), down=True, is_output=False):
        super().__init__()
        self.stride = stride
        self.down = down
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.down_conv_1 = nn.Conv3d(in_channel, out_channel, kernel_size=(3, 3, 3), \
                                     pad_mode="pad", stride=self.stride, padding=1)
        self.is_output = is_output
        if not is_output:
            self.batchNormal1 = BatchNorm3d(num_features=self.out_channel)
            self.relu1 = nn.PReLU()
        if self.down:
            self.down_conv_2 = nn.Conv3d(out_channel, out_channel, kernel_size=(3, 3, 3), \
                                         pad_mode="pad", stride=1, padding=1)
            self.relu2 = nn.PReLU()
            if kernel_size[0] == 1:
                self.residual = nn.Conv3d(in_channel, out_channel, kernel_size=(1, 1, 1), \
                                          pad_mode="valid", stride=self.stride)
            else:
                self.residual = nn.Conv3d(in_channel, out_channel, kernel_size=(3, 3, 3), \
                                       pad_mode="pad", stride=self.stride, padding=1)
            self.batchNormal2 = BatchNorm3d(num_features=self.out_channel)


    def construct(self, x):
        out = self.down_conv_1(x)
        if self.is_output:
            return out
        out = self.batchNormal1(out)
        out = self.relu1(out)
        if self.down:
            out = self.down_conv_2(out)
            out = self.batchNormal2(out)
            out = self.relu2(out)
            res = self.residual(x)
        else:
            res = x
        return out + res

class Down(nn.Cell):
    def __init__(self, in_channel, out_channel, stride=2, kernel_size=(3, 3, 3), dtype=mstype.float16):
        super().__init__()
        self.stride = stride
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.down_conv = ResidualUnit(self.in_channel, self.out_channel, stride, kernel_size).to_float(dtype)

    def construct(self, x):
        x = self.down_conv(x)
        return x


class Up(nn.Cell):
    def __init__(self, in_channel, down_in_channel, out_channel, stride=2, is_output=False, dtype=mstype.float16):
        super().__init__()
        self.in_channel = in_channel
        self.down_in_channel = down_in_channel
        self.out_channel = out_channel
        self.stride = stride
        self.conv3d_transpose = nn.Conv3dTranspose(in_channels=self.in_channel + self.down_in_channel, \
                                                   out_channels=self.out_channel, kernel_size=(3, 3, 3), \
                                                   pad_mode="pad", stride=self.stride, \
                                                   output_padding=(1, 1, 1), padding=1)

        self.concat = P.Concat(axis=1)
        self.conv = ResidualUnit(self.out_channel, self.out_channel, stride=1, down=False, \
                                 is_output=is_output).to_float(dtype)
        self.batchNormal1 = BatchNorm3d(num_features=self.out_channel)
        self.relu = nn.PReLU()

    def construct(self, input_data, down_input):
        x = self.concat((input_data, down_input))
        x = self.conv3d_transpose(x)
        x = self.batchNormal1(x)
        x = self.relu(x)
        x = self.conv(x)
        return x
