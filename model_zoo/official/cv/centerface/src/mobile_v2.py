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
"""modified mobilenet_v2 backbone"""

import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.ops.operations import Add

from src.var_init import KaimingNormal

__all__ = ['MobileNetV2', 'mobilenet_v2']

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Cell):
    """
    Convolution and batchnorm and relu
    """
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__()
        if groups == 1:
            conv = nn.Conv2d(in_planes, out_planes, kernel_size, stride, pad_mode="pad", padding=padding,
                             has_bias=False)
        else:
            conv = nn.Conv2d(in_planes, out_planes, kernel_size, stride, pad_mode="pad", padding=padding,
                             has_bias=False, group=groups, weight_init=KaimingNormal(mode='fan_out'))

        layers = [conv, nn.BatchNorm2d(out_planes).add_flags_recursive(fp32=True), nn.ReLU6()]  #, momentum=0.9
        self.features = nn.SequentialCell(layers)
        self.in_planes = in_planes
        self.print = P.Print()

    def construct(self, x):
        x = self.features(x)
        return x


class InvertedResidual(nn.Cell):
    """
    Inverted residual module
    """
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, kernel_size=1, stride=1, has_bias=False),
            nn.BatchNorm2d(oup).add_flags_recursive(fp32=True)
        ])

        self.conv = nn.SequentialCell(layers)
        self.add = Add()
        self.cast = P.Cast()

    def construct(self, x):
        identity = x
        x = self.conv(x)
        if self.use_res_connect:
            return self.add(identity, x)

        return x


class MobileNetV2(nn.Cell):
    """
    MobileNet V2 main class, backbone

    Args:
        width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
        inverted_residual_setting: Network structure
        round_nearest (int): Round the number of channels in each layer to be a multiple of this number
        Set to 1 to turn off rounding
    """
    def __init__(self, width_mult=1.0, inverted_residual_setting=None, round_nearest=8):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]
        self.feat_id = [1, 2, 4, 6]
        self.feat_channel = []

        # only check the first element, assuming user knows t,c,n,s are required
        if inverted_residual_setting is None or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        features = [ConvBNReLU(3, input_channel, stride=2)]

        for index, (t, c, n, s) in enumerate(inverted_residual_setting):
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel

            if index == 1:
                self.need_fp1 = nn.SequentialCell(features)
                self.feat_channel.append(output_channel)
                features = []
            elif index == 2:
                self.need_fp2 = nn.SequentialCell(features)
                self.feat_channel.append(output_channel)
                features = []
            elif index == 4:
                self.need_fp3 = nn.SequentialCell(features)
                self.feat_channel.append(output_channel)
                features = []
            elif index == 6:
                self.need_fp4 = nn.SequentialCell(features)
                self.feat_channel.append(output_channel)
                features = []


    def construct(self, x):
        x1 = self.need_fp1(x)
        x2 = self.need_fp2(x1)
        x3 = self.need_fp3(x2)
        x4 = self.need_fp4(x3)
        return x1, x2, x3, x4

def mobilenet_v2(**kwargs):
    return MobileNetV2(**kwargs)
