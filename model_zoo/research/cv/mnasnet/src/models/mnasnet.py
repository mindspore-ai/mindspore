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
"""MNASNet model definition"""
import mindspore.nn as nn


class ConvBlock(nn.Cell):
    """ConvBlock"""

    def __init__(self,
                 in_channel,
                 out_channel,
                 kernel_size=3,
                 stride=1,
                 pad_mode='pad',
                 padding=0,
                 group=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channel,
                              out_channel,
                              kernel_size=kernel_size,
                              stride=stride,
                              pad_mode=pad_mode,
                              padding=padding,
                              group=group,
                              has_bias=True)
        self.bn = nn.BatchNorm2d(out_channel)
        self.activation = nn.ReLU()

    def construct(self, x):
        """construct"""
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class SepConv(nn.Cell):
    """SepConv"""

    def __init__(self,
                 in_channels,
                 out_channels):
        super(SepConv, self).__init__()

        sequence = [nn.Conv2d(in_channels,
                              in_channels,
                              kernel_size=3,
                              stride=1,
                              pad_mode="pad",
                              padding=1,
                              group=in_channels,
                              has_bias=False),
                    nn.BatchNorm2d(in_channels),
                    nn.ReLU(),
                    nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size=3,
                              stride=1,
                              pad_mode="pad",
                              padding=1,
                              group=1,
                              has_bias=False),
                    nn.BatchNorm2d(out_channels),
                    ]

        self.sequence = nn.SequentialCell(sequence)

    def construct(self, x):
        """construct"""
        output = self.sequence(x)
        return output


class InvertedResidual(nn.Cell):
    """InvertedResidual"""

    def __init__(self, inp, oup, stride, expand_ratio, kernel):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = self.stride == 1 and inp == oup

        self.conv = nn.SequentialCell(
            # pw
            nn.Conv2d(inp, inp * expand_ratio, 1, 1, "pad", 0),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU(),
            # dw
            nn.Conv2d(inp * expand_ratio, inp * expand_ratio, kernel, stride, "pad", kernel // 2,
                      group=inp * expand_ratio),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU(),
            # pw-linear
            nn.Conv2d(inp * expand_ratio, oup, 1, 1, "pad", 0),
            nn.BatchNorm2d(oup),
        )

    def construct(self, x):
        """construct"""
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


class Mnasnet(nn.Cell):
    """Mnasnet"""

    def __init__(self):
        super(Mnasnet, self).__init__()

        self.interverted_residual_setting = [
            # t, c, n, s, k
            [3, 24, 3, 2, 3],  # -> 56x56
            [3, 40, 3, 2, 5],  # -> 28x28
            [6, 80, 3, 2, 5],  # -> 14x14
            [6, 96, 2, 1, 3],  # -> 14x14
            [6, 192, 4, 2, 5],  # -> 7x7
            [6, 320, 1, 1, 3],  # -> 7x7
        ]
        width_mult = 1.
        features = [ConvBlock(3, 32, kernel_size=3, stride=2, padding=1), SepConv(32, 16)]
        input_channel = 16
        for t, c, n, s, k in self.interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    features.append(InvertedResidual(input_channel, output_channel, s, t, k))
                else:
                    features.append(InvertedResidual(input_channel, output_channel, 1, t, k))
                input_channel = output_channel
        self.features = nn.SequentialCell(features)
        self.conv1 = nn.SequentialCell(nn.Conv2d(input_channel, 1280, 1, 1, "pad", 0), nn.BatchNorm2d(1280), nn.ReLU())
        self.avg_pool = nn.AvgPool2d(7, pad_mode='valid')
        self.flatten = nn.Flatten()
        self.fc = nn.Dense(1280, 1000, has_bias=True)

    def construct(self, x):
        """construct"""
        x = self.features(x)
        x = self.conv1(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        output = self.fc(x)
        return output
