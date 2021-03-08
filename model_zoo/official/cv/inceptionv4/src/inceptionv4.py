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
"""InceptionV4"""
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
                                   has_bn=True, activation="relu")

    def construct(self, x):
        x = self.conv(x)
        return x

class Stem(nn.Cell):
    """
    Inceptionv4 stem

    """
    def __init__(self, in_channels):
        super(Stem, self).__init__()
        self.conv2d_1a_3x3 = Conv2d(
            in_channels, 32, 3, stride=2, padding=0, has_bias=False)

        self.conv2d_2a_3x3 = Conv2d(
            32, 32, 3, stride=1, padding=0, has_bias=False)
        self.conv2d_2b_3x3 = Conv2d(
            32, 64, 3, stride=1, pad_mode='pad', padding=1, has_bias=False)

        self.mixed_3a_branch_0 = nn.MaxPool2d(3, stride=2)
        self.mixed_3a_branch_1 = Conv2d(
            64, 96, 3, stride=2, padding=0, has_bias=False)

        self.mixed_4a_branch_0 = nn.SequentialCell([
            Conv2d(160, 64, 1, stride=1, padding=0, has_bias=False),
            Conv2d(64, 96, 3, stride=1, padding=0, pad_mode='valid', has_bias=False)])

        self.mixed_4a_branch_1 = nn.SequentialCell([
            Conv2d(160, 64, 1, stride=1, padding=0, has_bias=False),
            Conv2d(64, 64, (1, 7), pad_mode='same', stride=1, has_bias=False),
            Conv2d(64, 64, (7, 1), pad_mode='same', stride=1, has_bias=False),
            Conv2d(64, 96, 3, stride=1, padding=0, pad_mode='valid', has_bias=False)])



        self.mixed_5a_branch_0 = Conv2d(
            192, 192, 3, stride=2, padding=0, has_bias=False)
        self.mixed_5a_branch_1 = nn.MaxPool2d(3, stride=2)
        self.concat0 = P.Concat(1)
        self.concat1 = P.Concat(1)
        self.concat2 = P.Concat(1)

    def construct(self, x):
        """construct"""
        x = self.conv2d_1a_3x3(x)  # 149 x 149 x 32
        x = self.conv2d_2a_3x3(x)  # 147 x 147 x 32
        x = self.conv2d_2b_3x3(x)  # 147 x 147 x 64

        x0 = self.mixed_3a_branch_0(x)
        x1 = self.mixed_3a_branch_1(x)
        x = self.concat0((x0, x1))  # 73 x 73 x 160

        x0 = self.mixed_4a_branch_0(x)
        x1 = self.mixed_4a_branch_1(x)
        x = self.concat1((x0, x1))  # 71 x 71 x 192

        x0 = self.mixed_5a_branch_0(x)
        x1 = self.mixed_5a_branch_1(x)
        x = self.concat2((x0, x1))  # 35 x 35 x 384
        return x

class InceptionA(nn.Cell):
    """InceptionA"""
    def __init__(self, in_channels):
        super(InceptionA, self).__init__()
        self.branch_0 = Conv2d(
            in_channels, 96, 1, stride=1, padding=0, has_bias=False)
        self.branch_1 = nn.SequentialCell([
            Conv2d(in_channels, 64, 1, stride=1, padding=0, has_bias=False),
            Conv2d(64, 96, 3, stride=1, pad_mode='pad', padding=1, has_bias=False)])

        self.branch_2 = nn.SequentialCell([
            Conv2d(in_channels, 64, 1, stride=1, padding=0, has_bias=False),
            Conv2d(64, 96, 3, stride=1, pad_mode='pad',
                   padding=1, has_bias=False),
            Conv2d(96, 96, 3, stride=1, pad_mode='pad', padding=1, has_bias=False)])

        self.branch_3 = nn.SequentialCell([
            Avgpool(kernel_size=3, stride=1, pad_mode='same'),
            Conv2d(384, 96, 1, stride=1, padding=0, has_bias=False)])

        self.concat = P.Concat(1)

    def construct(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        x3 = self.branch_3(x)
        x4 = self.concat((x0, x1, x2, x3))
        return x4

class InceptionB(nn.Cell):
    """InceptionB"""
    def __init__(self, in_channels):
        super(InceptionB, self).__init__()
        self.branch_0 = Conv2d(in_channels, 384, 1,
                               stride=1, padding=0, has_bias=False)
        self.branch_1 = nn.SequentialCell([
            Conv2d(in_channels, 192, 1, stride=1, padding=0, has_bias=False),
            Conv2d(192, 224, (1, 7), pad_mode='same',
                   stride=1, has_bias=False),
            Conv2d(224, 256, (7, 1), pad_mode='same',
                   stride=1, has_bias=False),
        ])
        self.branch_2 = nn.SequentialCell([
            Conv2d(in_channels, 192, 1, stride=1, padding=0, has_bias=False),
            Conv2d(192, 192, (7, 1), pad_mode='same',
                   stride=1, has_bias=False),
            Conv2d(192, 224, (1, 7), pad_mode='same',
                   stride=1, has_bias=False),
            Conv2d(224, 224, (7, 1), pad_mode='same',
                   stride=1, has_bias=False),
            Conv2d(224, 256, (1, 7), pad_mode='same', stride=1, has_bias=False)
        ])
        self.branch_3 = nn.SequentialCell([
            Avgpool(kernel_size=3, stride=1, pad_mode='same'),
            Conv2d(in_channels, 128, 1, stride=1, padding=0, has_bias=False)
        ])
        self.concat = P.Concat(1)

    def construct(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        x3 = self.branch_3(x)
        x4 = self.concat((x0, x1, x2, x3))
        return x4

class ReductionA(nn.Cell):
    """ReductionA"""
    def __init__(self, in_channels, k, l, m, n):
        super(ReductionA, self).__init__()
        self.branch_0 = Conv2d(in_channels, n, 3, stride=2, padding=0)
        self.branch_1 = nn.SequentialCell([
            Conv2d(in_channels, k, 1, stride=1, padding=0, has_bias=False),
            Conv2d(k, l, 3, stride=1, pad_mode='pad',
                   padding=1, has_bias=False),
            Conv2d(l, m, 3, stride=2, padding=0, has_bias=False),
        ])
        self.branch_2 = nn.MaxPool2d(3, stride=2)
        self.concat = P.Concat(1)

    def construct(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        x3 = self.concat((x0, x1, x2))
        return x3         # 17 x 17 x 1024

class ReductionB(nn.Cell):
    """ReductionB"""
    def __init__(self, in_channels):
        super(ReductionB, self).__init__()
        self.branch_0 = nn.SequentialCell([
            Conv2d(in_channels, 192, 1, stride=1, padding=0, has_bias=False),
            Conv2d(192, 192, 3, stride=2, padding=0, has_bias=False),
        ])
        self.branch_1 = nn.SequentialCell([
            Conv2d(in_channels, 256, 1, stride=1, padding=0, has_bias=False),
            Conv2d(256, 256, (1, 7), pad_mode='same',
                   stride=1, has_bias=False),
            Conv2d(256, 320, (7, 1), pad_mode='same',
                   stride=1, has_bias=False),
            Conv2d(320, 320, 3, stride=2, padding=0, has_bias=False)
        ])
        self.branch_2 = nn.MaxPool2d(3, stride=2)
        self.concat = P.Concat(1)

    def construct(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        x3 = self.concat((x0, x1, x2))
        return x3     # 8 x 8 x 1536

class InceptionC(nn.Cell):
    """InceptionC"""
    def __init__(self, in_channels):
        super(InceptionC, self).__init__()
        self.branch_0 = Conv2d(in_channels, 256, 1,
                               stride=1, padding=0, has_bias=False)

        self.branch_1 = Conv2d(in_channels, 384, 1,
                               stride=1, padding=0, has_bias=False)
        self.branch_1_1 = Conv2d(
            384, 256, (1, 3), pad_mode='same', stride=1, has_bias=False)
        self.branch_1_2 = Conv2d(
            384, 256, (3, 1), pad_mode='same', stride=1, has_bias=False)

        self.branch_2 = nn.SequentialCell([
            Conv2d(in_channels, 384, 1, stride=1, padding=0, has_bias=False),
            Conv2d(384, 448, (3, 1), pad_mode='same',
                   stride=1, has_bias=False),
            Conv2d(448, 512, (1, 3), pad_mode='same',
                   stride=1, has_bias=False),
        ])
        self.branch_2_1 = Conv2d(
            512, 256, (1, 3), pad_mode='same', stride=1, has_bias=False)
        self.branch_2_2 = Conv2d(
            512, 256, (3, 1), pad_mode='same', stride=1, has_bias=False)

        self.branch_3 = nn.SequentialCell([
            Avgpool(kernel_size=3, stride=1, pad_mode='same'),
            Conv2d(in_channels, 256, 1, stride=1, padding=0, has_bias=False)
        ])
        self.concat0 = P.Concat(1)
        self.concat1 = P.Concat(1)
        self.concat2 = P.Concat(1)

    def construct(self, x):
        """construct"""
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x1_1 = self.branch_1_1(x1)
        x1_2 = self.branch_1_2(x1)
        x1 = self.concat0((x1_1, x1_2))
        x2 = self.branch_2(x)
        x2_1 = self.branch_2_1(x2)
        x2_2 = self.branch_2_2(x2)
        x2 = self.concat1((x2_1, x2_2))
        x3 = self.branch_3(x)
        return self.concat2((x0, x1, x2, x3))  # 8 x 8 x 1536

class Inceptionv4(nn.Cell):
    """
    Inceptionv4 architecture

    Args.
        is_train : in train mode, turn on the dropout.

    """
    def __init__(self, in_channels=3, classes=1000, k=192, l=224, m=256, n=384, is_train=True):
        super(Inceptionv4, self).__init__()
        blocks = []
        blocks.append(Stem(in_channels))
        for _ in range(4):
            blocks.append(InceptionA(384))
        blocks.append(ReductionA(384, k, l, m, n))
        for _ in range(7):
            blocks.append(InceptionB(1024))
        blocks.append(ReductionB(1024))
        for _ in range(3):
            blocks.append(InceptionC(1536))
        self.features = nn.SequentialCell(blocks)

        self.avgpool = P.ReduceMean(keep_dims=False)
        self.softmax = nn.DenseBnAct(
            1536, classes, weight_init="XavierUniform", has_bias=True, has_bn=True, activation="logsoftmax")
        if is_train:
            self.dropout = nn.Dropout(0.20)
        else:
            self.dropout = nn.Dropout(1)
        self.bn0 = nn.BatchNorm1d(1536, eps=0.001, momentum=0.1)


    def construct(self, x):
        x = self.features(x)
        x = self.avgpool(x, (2, 3))
        x = self.bn0(x)
        x = self.dropout(x)
        x = self.softmax(x)
        return x
