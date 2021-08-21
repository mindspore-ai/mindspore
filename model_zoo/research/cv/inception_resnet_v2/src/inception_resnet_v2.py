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
"""Inception_ResNet_v2"""
import mindspore.nn as nn
from mindspore.ops import operations as P

class Avgpool(nn.Cell):
    """Avgpool"""
    def __init__(self, kernel_size, stride=1, pad_mode='same'):
        super(Avgpool, self).__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size=kernel_size, stride=stride, pad_mode=pad_mode)

    def construct(self, x):
        x = self.avg_pool(x)
        return x


class Conv2d(nn.Cell):
    """
    Set the default configuration for Conv2dBnAct
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, pad_mode='valid', padding=0,
                 has_bias=False, weight_init="XavierUniform", bias_init='zeros'):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2dBnAct(in_channels, out_channels, kernel_size, stride=stride, pad_mode=pad_mode,
                                   padding=padding, weight_init=weight_init, bias_init=bias_init, has_bias=has_bias,
                                   has_bn=True, eps=0.001, momentum=0.9, activation="relu")

    def construct(self, x):
        x = self.conv(x)
        return x


class Mixed_5b(nn.Cell):
    """
    Mixed_5b
    """
    def __init__(self):
        super(Mixed_5b, self).__init__()

        self.branch0 = Conv2d(192, 96, kernel_size=1, stride=1)

        self.branch1 = nn.SequentialCell(
            Conv2d(192, 48, kernel_size=1, stride=1),
            Conv2d(48, 64, kernel_size=5, stride=1, padding=2, pad_mode='pad')
        )

        self.branch2 = nn.SequentialCell(
            Conv2d(192, 64, kernel_size=1, stride=1),
            Conv2d(64, 96, kernel_size=3, stride=1, padding=1, pad_mode='pad'),
            Conv2d(96, 96, kernel_size=3, stride=1, padding=1, pad_mode='pad')
        )

        self.branch3 = nn.SequentialCell(
            nn.AvgPool2d(3, stride=1, pad_mode='same'),
            Conv2d(192, 64, kernel_size=1, stride=1)
        )

        self.concat = P.Concat(1)

    def construct(self, x):
        '''
        construct
        '''
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = self.concat((x0, x1, x2, x3))
        return out


class Stem(nn.Cell):
    """
    Inceptionv resnet v2 stem

    """
    def __init__(self, in_channels):
        super(Stem, self).__init__()
        self.conv2d_1a = Conv2d(in_channels, 32, kernel_size=3, stride=2)
        self.conv2d_2a = Conv2d(32, 32, kernel_size=3, stride=1)
        self.conv2d_2b = Conv2d(32, 64, kernel_size=3, stride=1, padding=1, pad_mode='pad')
        self.maxpool_3a = nn.MaxPool2d(3, stride=2)
        self.conv2d_3b = Conv2d(64, 80, kernel_size=1, stride=1)
        self.conv2d_4a = Conv2d(80, 192, kernel_size=3, stride=1)
        self.maxpool_5a = nn.MaxPool2d(3, stride=2)
        self.mixed_5b = Mixed_5b()

    def construct(self, x):
        """construct"""
        x = self.conv2d_1a(x)
        x = self.conv2d_2a(x)
        x = self.conv2d_2b(x)
        x = self.maxpool_3a(x)
        x = self.conv2d_3b(x)
        x = self.conv2d_4a(x)
        x = self.maxpool_5a(x)
        x = self.mixed_5b(x)
        return x


class InceptionA(nn.Cell):
    """InceptionA"""
    def __init__(self, scale):
        super(InceptionA, self).__init__()
        self.scale = scale
        self.branch0 = Conv2d(320, 32, kernel_size=1, stride=1)
        self.branch1 = nn.SequentialCell(
            Conv2d(320, 32, kernel_size=1, stride=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, pad_mode='pad')
        )

        self.branch2 = nn.SequentialCell(
            Conv2d(320, 32, kernel_size=1, stride=1),
            Conv2d(32, 48, kernel_size=3, stride=1, padding=1, pad_mode='pad'),
            Conv2d(48, 64, kernel_size=3, stride=1, padding=1, pad_mode='pad')
        )

        self.conv2d = nn.Conv2d(128, 320, kernel_size=1, stride=1)
        self.relu = nn.ReLU()
        self.concat = P.Concat(1)

    def construct(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = self.concat((x0, x1, x2))
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out


class ReductionA(nn.Cell):
    '''
    ReductionA
    '''
    def __init__(self):
        super(ReductionA, self).__init__()

        self.branch0 = Conv2d(320, 384, kernel_size=3, stride=2)

        self.branch1 = nn.SequentialCell(
            Conv2d(320, 256, kernel_size=1, stride=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, pad_mode='pad'),
            Conv2d(256, 384, kernel_size=3, stride=2)
        )

        self.branch2 = nn.MaxPool2d(3, stride=2)
        self.concat = P.Concat(1)

    def construct(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = self.concat((x0, x1, x2))
        return out


class InceptionB(nn.Cell):
    """
    InceptionB
    """
    def __init__(self, scale=1.0):
        super(InceptionB, self).__init__()
        self.scale = scale
        self.branch0 = Conv2d(1088, 192, kernel_size=1, stride=1)
        self.branch1 = nn.SequentialCell(
            Conv2d(1088, 128, kernel_size=1, stride=1),
            Conv2d(128, 160, kernel_size=(1, 7), stride=1, pad_mode='same'),
            Conv2d(160, 192, kernel_size=(7, 1), stride=1, pad_mode='same')
        )
        self.conv2d = nn.Conv2d(384, 1088, kernel_size=1, stride=1)
        self.relu = nn.ReLU()
        self.concat = P.Concat(1)

    def construct(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = self.concat((x0, x1))
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out


class ReductionB(nn.Cell):
    """
    ReductionB
    """
    def __init__(self):
        super(ReductionB, self).__init__()
        self.branch0 = nn.SequentialCell(
            Conv2d(1088, 256, kernel_size=1, stride=1),
            Conv2d(256, 384, kernel_size=3, stride=2)
        )
        self.branch1 = nn.SequentialCell(
            Conv2d(1088, 256, kernel_size=1, stride=1),
            Conv2d(256, 288, kernel_size=3, stride=2)
        )
        self.branch2 = nn.SequentialCell(
            Conv2d(1088, 256, kernel_size=1, stride=1),
            Conv2d(256, 288, kernel_size=3, stride=1, pad_mode='pad', padding=1),
            Conv2d(288, 320, kernel_size=3, stride=2)
        )
        self.branch3 = nn.MaxPool2d(3, stride=2)
        self.concat = P.Concat(1)

    def construct(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = self.concat((x0, x1, x2, x3))
        return out


class InceptionC(nn.Cell):
    """
    InceptionC
    """
    def __init__(self, scale=1.0, noReLU=False):
        super(InceptionC, self).__init__()
        self.scale = scale
        self.noReLU = noReLU
        self.branch0 = Conv2d(2080, 192, kernel_size=1, stride=1)
        self.branch1 = nn.SequentialCell(
            Conv2d(2080, 192, kernel_size=1, stride=1),
            Conv2d(192, 224, kernel_size=(1, 3), stride=1, pad_mode='same'),
            Conv2d(224, 256, kernel_size=(3, 1), stride=1, pad_mode='same')
        )
        self.conv2d = nn.Conv2d(448, 2080, kernel_size=1, stride=1)
        self.concat = P.Concat(1)
        if not self.noReLU:
            self.relu = nn.ReLU()
        self.print = P.Print()

    def construct(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = self.concat((x0, x1))
        out = self.conv2d(out)
        out = out * self.scale + x
        if not self.noReLU:
            out = self.relu(out)
        return out


class Inception_resnet_v2(nn.Cell):
    """
    Inception_resnet_v2 architecture
    Args.
        is_train : in train mode, turn on the dropout.
    """
    def __init__(self, in_channels=3, classes=1000, k=192, l=224, m=256, n=384, is_train=True):
        super(Inception_resnet_v2, self).__init__()
        blocks = []
        blocks.append(Stem(in_channels))
        for _ in range(10):
            blocks.append(InceptionA(scale=0.17))
        blocks.append(ReductionA())
        for _ in range(20):
            blocks.append(InceptionB(scale=0.10))
        blocks.append(ReductionB())
        for _ in range(9):
            blocks.append(InceptionC(scale=0.20))
        self.features = nn.SequentialCell(blocks)
        self.block8 = InceptionC(noReLU=True)
        self.conv2d_7b = Conv2d(2080, 1536, kernel_size=1, stride=1)
        self.avgpool = P.ReduceMean(keep_dims=False)
        self.softmax = nn.DenseBnAct(
            1536, classes, weight_init="XavierUniform", has_bias=True, has_bn=True, activation="logsoftmax")
        if is_train:
            self.dropout = nn.Dropout(0.8)
        else:
            self.dropout = nn.Dropout(1.0)

    def construct(self, x):
        x = self.features(x)
        x = self.block8(x)
        x = self.conv2d_7b(x)
        x = self.avgpool(x, (2, 3))
        x = self.dropout(x)
        x = self.softmax(x)
        return x
