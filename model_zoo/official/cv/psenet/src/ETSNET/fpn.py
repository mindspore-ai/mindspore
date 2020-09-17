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


import mindspore.nn as nn
import mindspore.ops.operations as P

from .base import _conv, _bn

class FPN(nn.Cell):
    def __init__(self, in_channels, out_channel, long_size):
        super(FPN, self).__init__()

        self.long_size = long_size

        # reduce layers
        self.reduce_conv_c2 = _conv(in_channels[0], out_channel, kernel_size=1, has_bias=True)
        self.reduce_bn_c2 = _bn(out_channel)
        self.reduce_relu_c2 = nn.ReLU()

        self.reduce_conv_c3 = _conv(in_channels[1], out_channel, kernel_size=1, has_bias=True)
        self.reduce_bn_c3 = _bn(out_channel)
        self.reduce_relu_c3 = nn.ReLU()

        self.reduce_conv_c4 = _conv(in_channels[2], out_channel, kernel_size=1, has_bias=True)
        self.reduce_bn_c4 = _bn(out_channel)
        self.reduce_relu_c4 = nn.ReLU()

        self.reduce_conv_c5 = _conv(in_channels[3], out_channel, kernel_size=1, has_bias=True)
        self.reduce_bn_c5 = _bn(out_channel)
        self.reduce_relu_c5 = nn.ReLU()

        # smooth layers
        self.smooth_conv_p4 = _conv(out_channel, out_channel, kernel_size=3, has_bias=True)
        self.smooth_bn_p4 = _bn(out_channel)
        self.smooth_relu_p4 = nn.ReLU()

        self.smooth_conv_p3 = _conv(out_channel, out_channel, kernel_size=3, has_bias=True)
        self.smooth_bn_p3 = _bn(out_channel)
        self.smooth_relu_p3 = nn.ReLU()

        self.smooth_conv_p2 = _conv(out_channel, out_channel, kernel_size=3, has_bias=True)
        self.smooth_bn_p2 = _bn(out_channel)
        self.smooth_relu_p2 = nn.ReLU()

        self._upsample_p4 = P.ResizeBilinear((long_size // 16, long_size // 16), align_corners=True)
        self._upsample_p3 = P.ResizeBilinear((long_size // 8, long_size // 8), align_corners=True)
        self._upsample_p2 = P.ResizeBilinear((long_size // 4, long_size // 4), align_corners=True)

        self.concat = P.Concat(axis=1)

    def construct(self, c2, c3, c4, c5):
        p5 = self.reduce_conv_c5(c5)
        p5 = self.reduce_relu_c5(self.reduce_bn_c5(p5))

        c4 = self.reduce_conv_c4(c4)
        c4 = self.reduce_relu_c4(self.reduce_bn_c4(c4))
        p4 = self._upsample_p4(p5) + c4
        p4 = self.smooth_conv_p4(p4)
        p4 = self.smooth_relu_p4(self.smooth_bn_p4(p4))

        c3 = self.reduce_conv_c3(c3)
        c3 = self.reduce_relu_c3(self.reduce_bn_c3(c3))
        p3 = self._upsample_p3(p4) + c3
        p3 = self.smooth_conv_p3(p3)
        p3 = self.smooth_relu_p3(self.smooth_bn_p3(p3))

        c2 = self.reduce_conv_c2(c2)
        c2 = self.reduce_relu_c2(self.reduce_bn_c2(c2))
        p2 = self._upsample_p2(p3) + c2
        p2 = self.smooth_conv_p2(p2)
        p2 = self.smooth_relu_p2(self.smooth_bn_p2(p2))

        p3 = self._upsample_p2(p3)
        p4 = self._upsample_p2(p4)
        p5 = self._upsample_p2(p5)

        out = self.concat((p2, p3, p4, p5))

        return out
