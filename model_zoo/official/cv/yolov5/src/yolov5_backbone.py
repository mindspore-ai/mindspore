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


class Concat(nn.Cell):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension
        self.concat = P.Concat(self.d)

    def forward(self, x):
        return self.concat


class Bottleneck(nn.Cell):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1)
        self.add = shortcut and c1 == c2

    def construct(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Cell):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSP, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, has_bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, has_bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_, momentum=0.9, eps=1e-5)  # applied to cat(cv2, cv3)
        self.act = nn.LeakyReLU(0.1)
        self.m = nn.SequentialCell([Bottleneck(c_, c_, shortcut, e=1.0) for _ in range(n)])
        self.concat = P.Concat(1)

    def construct(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        concat2 = self.concat((y1, y2))
        return self.cv4(self.act(self.bn(concat2)))


class C3(nn.Cell):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(C3, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.SequentialCell([Bottleneck(c_, c_, shortcut, e=1.0) for _ in range(n)])
        self.concat = P.Concat(1)

    def construct(self, x):
        y1 = self.m(self.cv1(x))
        y2 = self.cv2(x)
        concat2 = self.concat((y1, y2))
        return self.cv3(concat2)


class SPP(nn.Cell):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)

        self.maxpool1 = nn.MaxPool2d(kernel_size=5, stride=1, pad_mode='same')
        self.maxpool2 = nn.MaxPool2d(kernel_size=9, stride=1, pad_mode='same')
        self.maxpool3 = nn.MaxPool2d(kernel_size=13, stride=1, pad_mode='same')
        self.concat = P.Concat(1)

    def construct(self, x):
        x = self.cv1(x)
        m1 = self.maxpool1(x)
        m2 = self.maxpool2(x)
        m3 = self.maxpool3(x)
        concatm = self.concat((x, m1, m2, m3))
        return self.cv2(concatm)


class Focus(nn.Cell):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, act=True):
        super(Focus, self).__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, act)
        self.concat = P.Concat(1)

    def construct(self, x):
        w = P.Shape()(x)[2]
        h = P.Shape()(x)[3]
        concat4 = self.concat((x[..., 0:w:2, 0:h:2], x[..., 1:w:2, 0:h:2], x[..., 0:w:2, 1:h:2], x[..., 1:w:2, 1:h:2]))
        return self.conv(concat4)


class Focusv2(nn.Cell):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, act=True):
        super(Focusv2, self).__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, act)

    def construct(self, x):
        return self.conv(x)


class SiLU(nn.Cell):
    def __init__(self):
        super(SiLU, self).__init__()
        self.sigmoid = P.Sigmoid()

    def construct(self, x):
        return x * self.sigmoid(x)


def autopad(k, p=None):  # kernel, padding
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
        self.padding = autopad(k, p)
        self.pad_mode = None
        if self.padding == 0:
            self.pad_mode = 'same'
        elif self.padding == 1:
            self.pad_mode = 'pad'
        self.conv = nn.Conv2d(c1, c2, k, s, padding=self.padding, pad_mode=self.pad_mode, has_bias=False)
        self.bn = nn.BatchNorm2d(c2, momentum=momentum, eps=eps)
        self.act = SiLU() if act is True else (act if isinstance(act, nn.Cell) else P.Identity())

    def construct(self, x):
        return self.act(self.bn(self.conv(x)))


class YOLOv5Backbone(nn.Cell):

    def __init__(self):
        super(YOLOv5Backbone, self).__init__()

        # self.outchannel = 1024
        # self.concat = P.Concat(axis=1)
        # self.add = P.TensorAdd()

        self.focusv2 = Focusv2(3, 32, k=3, s=1)
        self.conv1 = Conv(32, 64, k=3, s=2)
        self.C31 = C3(64, 64, n=1)
        self.conv2 = Conv(64, 128, k=3, s=2)
        self.C32 = C3(128, 128, n=3)
        self.conv3 = Conv(128, 256, k=3, s=2)
        self.C33 = C3(256, 256, n=3)
        self.conv4 = Conv(256, 512, k=3, s=2)
        self.spp = SPP(512, 512, k=[5, 9, 13])
        self.C34 = C3(512, 512, n=1, shortcut=False)

    def construct(self, x):
        """construct method"""
        fcs = self.focusv2(x)
        cv1 = self.conv1(fcs)
        bcsp1 = self.C31(cv1)
        cv2 = self.conv2(bcsp1)
        bcsp2 = self.C32(cv2)
        cv3 = self.conv3(bcsp2)
        bcsp3 = self.C33(cv3)
        cv4 = self.conv4(bcsp3)
        spp1 = self.spp(cv4)
        bcsp4 = self.C34(spp1)
        return bcsp2, bcsp3, bcsp4
