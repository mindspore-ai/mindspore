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
"""effnet."""

import numpy as np
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.common.initializer import TruncatedNormal
from mindspore import Tensor

def weight_variable():
    """weight initial"""
    return TruncatedNormal(0.02)


def _make_value_divisible(value, factor, min_value=None):
    """
    It ensures that all layers have a channel number that is divisible by 8
    :param v: value to process
    :param factor: divisor
    :param min_value: new value always greater than the min_value
    :return: new value
    """
    if min_value is None:
        min_value = factor
    new_value = max(int(value + factor / 2) // factor * factor, min_value)
    if new_value < value * 0.9:
        new_value += factor
    return new_value

class Swish(nn.Cell):
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()

    def construct(self, x):
        s = self.sigmoid(x)
        m = x*s
        return m
        #return x * (1/(1+self.exp(-x)))


class AdaptiveAvgPool(nn.Cell):
    def __init__(self, output_size=None):
        super().__init__()
        self.mean = P.ReduceMean(keep_dims=True)
        self.output_size = output_size

    def construct(self, x):
        return self.mean(x, (2, 3)) ## This is not a general case

class SELayer(nn.Cell):
    """SELayer"""
    def __init__(self, channel, reduction=4):
        super().__init__()
        reduced_chs = _make_value_divisible(channel/reduction, 1)
        self.avg_pool = AdaptiveAvgPool(output_size=(1, 1))
        weight = weight_variable()
        self.conv_reduce = nn.Conv2d(in_channels=channel, out_channels=reduced_chs, kernel_size=1, has_bias=True,
                                     weight_init=weight)
        self.act1 = Swish()
        self.conv_expand = nn.Conv2d(in_channels=reduced_chs, out_channels=channel, kernel_size=1, has_bias=True)
        self.act2 = nn.Sigmoid()

    def construct(self, x):
        #b, c, _, _ = x.shape()
        o = self.avg_pool(x) #.view(b,c)
        o = self.conv_reduce(o)
        o = self.act1(o)
        o = self.conv_expand(o)
        o = self.act2(o) #.view(b, c, 1,1)
        return x * o

class DepthwiseSeparableConv(nn.Cell):
    """DepthwiseSeparableConv"""
    def __init__(self, in_chs, out_chs, dw_kernel_size=3, stride=1, noskip=False, se_ratio=0.0, drop_connect_rate=0.0):
        super().__init__()
        assert stride in [1, 2]
        self.has_residual = (stride == 1 and in_chs == out_chs) and not noskip
        self.drop_connect_rate = drop_connect_rate

        self.conv_dw = nn.Conv2d(in_channels=in_chs, out_channels=in_chs, kernel_size=dw_kernel_size, stride=stride,
                                 pad_mode="pad", padding=1, has_bias=False, group=in_chs)
        self.bn1 = nn.BatchNorm2d(in_chs, eps=0.001) #,momentum=0.1)
        self.act1 = Swish()

       # Squeeze-and-excitation
        if se_ratio is not None and se_ratio > 0.:
            self.se = SELayer(in_chs, reduction=se_ratio)
        else:
            print("ERRRRRORRRR -- not prepared for this one\n")

        self.conv_pw = nn.Conv2d(in_channels=in_chs, out_channels=out_chs, kernel_size=1, stride=stride, has_bias=False)
        self.bn2 = nn.BatchNorm2d(out_chs, eps=0.001) #,momentum=0.1)

    def construct(self, x):
        """construct"""
        residual = x

        x = self.conv_dw(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.se(x)

        x = self.conv_pw(x)
        x = self.bn2(x)

        if self.has_residual:
            # if self.drop_connect_rate > 0.:
                # x = x
                # x = drop_connect(x, self.training, self.drop_connect_rate)
            x += residual
        return x

def conv_3x3_bn(inp, oup, stride):
    weight = weight_variable()
    return nn.SequentialCell([
        nn.Conv2d(in_channels=inp, out_channels=oup, kernel_size=3, stride=stride, padding=1, weight_init=weight,
                  has_bias=False, pad_mode='pad'),
        nn.BatchNorm2d(oup, eps=0.001),  #, momentum=0.1),
        nn.HSwish()])


def conv_1x1_bn(inp, oup):
    weight = weight_variable()
    return nn.SequentialCell([
        nn.Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, stride=1, padding=0, weight_init=weight,
                  has_bias=False),
        nn.BatchNorm2d(oup, eps=0.001),
        nn.HSwish()])


class InvertedResidual(nn.Cell):
    """InvertedResidual"""
    def __init__(self, in_chs, out_chs, kernel_size, stride, padding, expansion, se_ratio):
        super().__init__()
        assert stride in [1, 2]
        mid_chs: int = _make_value_divisible(in_chs * expansion, 1)
        self.has_residual = (in_chs == out_chs and stride == 1)
        self.drop_connect_rate = 0

        # Point-wise expansion
        self.conv_pw = nn.Conv2d(in_channels=in_chs, out_channels=mid_chs, kernel_size=1, stride=1, has_bias=False)
        self.bn1 = nn.BatchNorm2d(mid_chs, eps=0.001)
        self.act1 = Swish()

        # Depth-wise convolution
        if stride > 1:
            self.conv_dw = nn.Conv2d(in_channels=mid_chs, out_channels=mid_chs, kernel_size=kernel_size, stride=stride,
                                     padding=padding, has_bias=False, group=mid_chs, pad_mode='same')
        else:
            self.conv_dw = nn.Conv2d(in_channels=mid_chs, out_channels=mid_chs, kernel_size=kernel_size, stride=stride,
                                     padding=padding, has_bias=False, group=mid_chs, pad_mode='pad')
        self.bn2 = nn.BatchNorm2d(mid_chs, eps=0.001)
        self.act2 = Swish()

        # Squeeze-and-excitation
        if se_ratio is not None and se_ratio > 0.:
            self.se = SELayer(mid_chs, reduction=se_ratio)
        else:
            print("ERRRRRORRRR -- not prepared for this one\n")

        # Point-wise linear projection
        self.conv_pwl = nn.Conv2d(in_channels=mid_chs, out_channels=out_chs, kernel_size=1, stride=1, has_bias=False)
        self.bn3 = nn.BatchNorm2d(out_chs, eps=0.001)

    def construct(self, x):
        """construct"""
        residual = x

        # Point-wise expansion
        x = self.conv_pw(x)
        x = self.bn1(x)
        x = self.act1(x)

        # Depth-wise convolution
        x = self.conv_dw(x)
        x = self.bn2(x)
        x = self.act2(x)

        # Squeeze-and-excitation
        x = self.se(x)

        # Point-wise linear projection
        x = self.conv_pwl(x)
        x = self.bn3(x)

        if self.has_residual:
            # if self.drop_connect_rate > 0.:
            #    x = x
            x += residual
        return x


class EfficientNet(nn.Cell):
    """EfficientNet"""
    def __init__(self, cfgs, num_classes=1000):
        super().__init__()
        # setting of inverted residual blocks
        self.cfgs = cfgs
        stem_size = 32
        self.num_classes_ = num_classes
        self.num_features_ = 1280

        self.conv_stem = nn.Conv2d(in_channels=3, out_channels=stem_size, kernel_size=3, stride=2, has_bias=False)

        self.bn1 = nn.BatchNorm2d(stem_size, eps=0.001) #momentum=0.1)
        self.act1 = Swish()
        in_chs = stem_size

        layers = [nn.SequentialCell([DepthwiseSeparableConv(in_chs, 16, 3, 1, se_ratio=4)]),

                  nn.SequentialCell([InvertedResidual(16, 24, 3, 2, 0, 6, se_ratio=24),
                                     InvertedResidual(24, 24, 3, 1, 1, 6, se_ratio=24)]),

                  nn.SequentialCell([InvertedResidual(24, 40, 5, 2, 0, 6, se_ratio=24),
                                     InvertedResidual(40, 40, 5, 1, 2, 6, se_ratio=24)]),

                  nn.SequentialCell([InvertedResidual(40, 80, 3, 2, 0, 6, se_ratio=24),
                                     InvertedResidual(80, 80, 3, 1, 1, 6, se_ratio=24),
                                     InvertedResidual(80, 80, 3, 1, 1, 6, se_ratio=24)]),

                  nn.SequentialCell([InvertedResidual(80, 112, 5, 1, 2, 6, se_ratio=24),
                                     InvertedResidual(112, 112, 5, 1, 2, 6, se_ratio=24),
                                     InvertedResidual(112, 112, 5, 1, 2, 6, se_ratio=24)]),

                  nn.SequentialCell([InvertedResidual(112, 192, 5, 2, 0, 6, se_ratio=24),
                                     InvertedResidual(192, 192, 5, 1, 2, 6, se_ratio=24),
                                     InvertedResidual(192, 192, 5, 1, 2, 6, se_ratio=24),
                                     InvertedResidual(192, 192, 5, 1, 2, 6, se_ratio=24)]),

                  nn.SequentialCell([InvertedResidual(192, 320, 3, 1, 1, 6, se_ratio=24)])
                 ]
        self.blocks = nn.SequentialCell(layers)

        self.conv_head = nn.Conv2d(in_channels=320, out_channels=self.num_features_, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(self.num_features_, eps=0.001) #,momentum=0.1)
        self.act2 = Swish()
        self.global_pool = AdaptiveAvgPool(output_size=(1, 1))
        self.classifier = nn.Dense(self.num_features_, num_classes)

        self._initialize_weights()

    def construct(self, x):
        """construct"""
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.blocks(x)
        x = self.conv_head(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.global_pool(x)
        x = P.Reshape()(x, (-1, self.num_features_))
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        """_initialize_weights"""
        def init_linear_weight(m):
            m.weight.set_data(Tensor(np.random.normal(0, 0.01, m.weight.data.shape).astype("float32")))
            if m.bias is not None:
                m.bias.set_data(Tensor(np.zeros(m.bias.data.shape, dtype="float32")))

        for m in self.cells():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.set_data(Tensor(np.random.normal(0, np.sqrt(2. / n), m.weight.data.shape).astype("float32")))
                if m.bias is not None:
                    m.bias.data.zero_()
                m.weight.requires_grad = True
            elif isinstance(m, nn.BatchNorm2d):
                m.gamma.set_data(Tensor(np.ones(m.gamma.data.shape, dtype="float32")))
                m.beta.set_data(Tensor(np.zeros(m.beta.data.shape, dtype="float32")))
            elif isinstance(m, nn.Dense):
                init_linear_weight(m)


def effnet(**kwargs):
    """
    Constructs a EfficientNet model
    """
    cfgs = [
        # k, t, c, SE, HS, s
        [3, 1, 16, 1, 0, 2],
        [3, 4.5, 24, 0, 0, 2],
        [3, 3.67, 24, 0, 0, 1],
        [5, 4, 40, 1, 1, 2],
        [5, 6, 40, 1, 1, 1],
        [5, 6, 40, 1, 1, 1],
        [5, 3, 48, 1, 1, 1],
        [5, 3, 48, 1, 1, 1],
        [5, 6, 96, 1, 1, 2],
        [5, 6, 96, 1, 1, 1],
        [5, 6, 96, 1, 1, 1],
    ]

    return EfficientNet(cfgs, **kwargs)
