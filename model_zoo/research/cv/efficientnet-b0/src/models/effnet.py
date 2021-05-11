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
"""efficientnet model define"""
import math
import mindspore.nn as nn

from src.models.layers import conv_bn_act
from src.models.layers import AdaptiveAvgPool2d
from src.models.layers import Flatten
from src.models.layers import SEModule
from src.models.layers import DropConnect


class MBConv(nn.Cell):
    """MBConv"""
    def __init__(self, in_, out_, expand,
                 kernel_size, stride, skip,
                 se_ratio, dc_ratio=0.2):
        super().__init__()
        mid_ = in_ * expand
        self.expand = expand
        self.expand_conv = conv_bn_act(in_, mid_, kernel_size=1, bias=False)

        self.depth_wise_conv = conv_bn_act(mid_, mid_,
                                           kernel_size=kernel_size, stride=stride,
                                           groups=mid_, bias=False)

        self.se = SEModule(mid_, int(in_ * se_ratio))

        self.project_conv = nn.SequentialCell([
            nn.Conv2d(mid_, out_, kernel_size=1, stride=1, has_bias=False),
            nn.BatchNorm2d(num_features=out_, eps=0.001, momentum=0.99)
        ])
        self.skip = skip and (stride == 1) and (in_ == out_)

        # DropConnect
        self.dropconnect = DropConnect(dc_ratio)

    def construct(self, inputs):
        """MBConv"""
        if self.expand != 1:
            expand = self.expand_conv(inputs)
        else:
            expand = inputs
        x = self.depth_wise_conv(expand)
        x = self.se(x)
        x = self.project_conv(x)
        if self.skip:
            x = x + inputs
        return x


class MBBlock(nn.Cell):
    """MBBlock"""
    def __init__(self, in_, out_, expand, kernel, stride, num_repeat, skip, se_ratio, drop_connect_ratio=0.2):
        super().__init__()
        layers = [MBConv(in_, out_, expand, kernel, stride, skip, se_ratio, drop_connect_ratio)]
        for _ in range(1, num_repeat):
            layers.append(MBConv(out_, out_, expand, kernel, 1, skip, se_ratio, drop_connect_ratio))
        self.layers = nn.SequentialCell([*layers])

    def construct(self, x):
        return self.layers(x)


class EfficientNet(nn.Cell):
    """efficientnet model"""
    def __init__(self, width_coeff, depth_coeff,
                 depth_div=8, min_depth=None,
                 dropout_rate=0.2, drop_connect_rate=0.2,
                 num_classes=1000):
        super().__init__()
        min_depth = min_depth or depth_div
        dropout_rate = 1 - dropout_rate

        def renew_ch(x):
            if not width_coeff:
                return x

            x *= width_coeff
            new_x = max(min_depth, int(x + depth_div / 2) // depth_div * depth_div)
            if new_x < 0.9 * x:
                new_x += depth_div
            return int(new_x)

        def renew_repeat(x):
            return int(math.ceil(x * depth_coeff))

        self.stem = conv_bn_act(3, renew_ch(32), kernel_size=3, stride=2, bias=False)

        self.blocks = nn.SequentialCell([
            #       input channel  output    expand  k  s                   skip  se
            MBBlock(renew_ch(32), renew_ch(16), 1, 3, 1, renew_repeat(1), True, 0.25, drop_connect_rate),
            MBBlock(renew_ch(16), renew_ch(24), 6, 3, 2, renew_repeat(2), True, 0.25, drop_connect_rate),
            MBBlock(renew_ch(24), renew_ch(40), 6, 5, 2, renew_repeat(2), True, 0.25, drop_connect_rate),
            MBBlock(renew_ch(40), renew_ch(80), 6, 3, 2, renew_repeat(3), True, 0.25, drop_connect_rate),
            MBBlock(renew_ch(80), renew_ch(112), 6, 5, 1, renew_repeat(3), True, 0.25, drop_connect_rate),
            MBBlock(renew_ch(112), renew_ch(192), 6, 5, 2, renew_repeat(4), True, 0.25, drop_connect_rate),
            MBBlock(renew_ch(192), renew_ch(320), 6, 3, 1, renew_repeat(1), True, 0.25, drop_connect_rate)
        ])

        self.head = nn.SequentialCell([
            *conv_bn_act(renew_ch(320), renew_ch(1280), kernel_size=1, bias=False),
            AdaptiveAvgPool2d(),
            nn.Dropout(dropout_rate),
            Flatten(),
            nn.Dense(renew_ch(1280), num_classes)
        ])

    def construct(self, inputs):
        stem = self.stem(inputs)
        x = self.blocks(stem)
        x = self.head(x)
        return x
