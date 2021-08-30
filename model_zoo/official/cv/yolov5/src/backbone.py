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
"""DarkNet model."""
import mindspore.nn as nn
from mindspore.ops import operations as P


class Bottleneck(nn.Cell):
    # Standard bottleneck
    # ch_in, ch_out, shortcut, groups, expansion
    def __init__(self, c1, c2, shortcut=True, e=0.5):
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.conv1 = Conv(c1, c_, 1, 1)
        self.conv2 = Conv(c_, c2, 3, 1)
        self.add = shortcut and c1 == c2

    def construct(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        out = c2
        if self.add:
            out = x + out
        return out


class BottleneckCSP(nn.Cell):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, e=0.5):
        super(BottleneckCSP, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.conv1 = Conv(c1, c_, 1, 1)
        self.conv2 = Conv(c1, c_, 1, 1)
        self.conv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.SequentialCell(
            [Bottleneck(c_, c_, shortcut, e=1.0) for _ in range(n)])
        self.concat = P.Concat(axis=1)

    def construct(self, x):
        c1 = self.conv1(x)
        c2 = self.m(c1)
        c3 = self.conv2(x)
        c4 = self.concat((c2, c3))
        c5 = self.conv3(c4)

        return c5


class SPP(nn.Cell):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.conv1 = Conv(c1, c_, 1, 1)
        self.conv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)

        self.maxpool1 = nn.MaxPool2d(kernel_size=5, stride=1, pad_mode='same')
        self.maxpool2 = nn.MaxPool2d(kernel_size=9, stride=1, pad_mode='same')
        self.maxpool3 = nn.MaxPool2d(kernel_size=13, stride=1, pad_mode='same')
        self.concat = P.Concat(axis=1)

    def construct(self, x):
        c1 = self.conv1(x)
        m1 = self.maxpool1(c1)
        m2 = self.maxpool2(c1)
        m3 = self.maxpool3(c1)
        c4 = self.concat((c1, m1, m2, m3))
        c5 = self.conv2(c4)
        return c5


class Focus(nn.Cell):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, act=True):
        super(Focus, self).__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, act)

    def construct(self, x):
        c1 = self.conv(x)
        return c1


class SiLU(nn.Cell):
    def __init__(self):
        super(SiLU, self).__init__()
        self.sigmoid = P.Sigmoid()

    def construct(self, x):
        return x * self.sigmoid(x)


def auto_pad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Cell):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None,
                 dilation=1,
                 alpha=0.1,
                 momentum=0.97,
                 eps=1e-3,
                 pad_mode="same",
                 act=True):  # ch_in, ch_out, kernel, stride, padding
        super(Conv, self).__init__()
        self.padding = auto_pad(k, p)
        self.pad_mode = None
        if self.padding == 0:
            self.pad_mode = 'same'
        elif self.padding == 1:
            self.pad_mode = 'pad'
        self.conv = nn.Conv2d(
            c1,
            c2,
            k,
            s,
            padding=self.padding,
            pad_mode=self.pad_mode,
            has_bias=False)
        self.bn = nn.BatchNorm2d(c2, momentum=momentum, eps=eps)
        self.act = SiLU() if act is True else (
            act if isinstance(act, nn.Cell) else P.Identity())

    def construct(self, x):
        return self.act(self.bn(self.conv(x)))


class YOLOv5Backbone(nn.Cell):
    def __init__(self, shape):
        super(YOLOv5Backbone, self).__init__()
        self.focus = Focus(shape[0], shape[1], k=3, s=1)
        self.conv1 = Conv(shape[1], shape[2], k=3, s=2)
        self.CSP1 = BottleneckCSP(shape[2], shape[2], n=1 * shape[6])
        self.conv2 = Conv(shape[2], shape[3], k=3, s=2)
        self.CSP2 = BottleneckCSP(shape[3], shape[3], n=3 * shape[6])
        self.conv3 = Conv(shape[3], shape[4], k=3, s=2)
        self.CSP3 = BottleneckCSP(shape[4], shape[4], n=3 * shape[6])
        self.conv4 = Conv(shape[4], shape[5], k=3, s=2)
        self.spp = SPP(shape[5], shape[5], k=[5, 9, 13])
        self.CSP4 = BottleneckCSP(shape[5], shape[5], n=1 * shape[6], shortcut=False)

    def construct(self, x):
        """construct method"""
        c1 = self.focus(x)
        c2 = self.conv1(c1)
        c3 = self.CSP1(c2)
        c4 = self.conv2(c3)
        # out
        c5 = self.CSP2(c4)
        c6 = self.conv3(c5)
        # out
        c7 = self.CSP3(c6)
        c8 = self.conv4(c7)
        c9 = self.spp(c8)
        # out
        c10 = self.CSP4(c9)
        return c5, c7, c10
