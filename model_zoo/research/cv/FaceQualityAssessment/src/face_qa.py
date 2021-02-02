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
"""Face Quality Assessment backbone."""
import mindspore.nn as nn
from mindspore.ops.operations import Add
from mindspore.ops import operations as P
from mindspore.nn import Dense, Cell


class Cut(nn.Cell):



    def construct(self, x):
        return x


def bn_with_initialize(out_channels):
    bn = nn.BatchNorm2d(out_channels, momentum=0.9, eps=1e-5)
    return bn


def fc_with_initialize(input_channels, out_channels):
    return Dense(input_channels, out_channels)


def conv3x3(in_channels, out_channels, stride=1, groups=1, dilation=1, pad_mode="pad", padding=1):
    """3x3 convolution with padding"""

    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                     pad_mode=pad_mode, group=groups, has_bias=False, dilation=dilation, padding=padding)


def conv1x1(in_channels, out_channels, pad_mode="pad", stride=1, padding=0):
    """1x1 convolution"""
    return nn.Conv2d(in_channels, out_channels, pad_mode=pad_mode, kernel_size=1, stride=stride, has_bias=False,
                     padding=padding)


def conv4x4(in_channels, out_channels, stride=1, groups=1, dilation=1, pad_mode="pad", padding=1):
    """4x4 convolution with padding"""

    return nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride,
                     pad_mode=pad_mode, group=groups, has_bias=False, dilation=dilation, padding=padding)


class Block1(Cell):
    '''Block1'''
    def __init__(self):
        super(Block1, self).__init__()

        self.bk1_conv0 = conv3x3(64, 64, stride=1, padding=1)
        self.bk1_bn0 = bn_with_initialize(64)
        self.bk1_relu0 = P.ReLU()
        self.bk1_conv1 = conv3x3(64, 64, stride=1, padding=1)
        self.bk1_bn1 = bn_with_initialize(64)
        self.bk1_conv2 = conv1x1(64, 64, stride=1, padding=0)
        self.bk1_bn2 = bn_with_initialize(64)
        self.bk1_relu1 = P.ReLU()

        self.bk1_conv3 = conv3x3(64, 64, stride=1, padding=1)
        self.bk1_bn3 = bn_with_initialize(64)
        self.bk1_relu3 = P.ReLU()
        self.bk1_conv4 = conv3x3(64, 64, stride=1, padding=1)
        self.bk1_bn4 = bn_with_initialize(64)
        self.bk1_relu4 = P.ReLU()

        self.cast = P.Cast()
        self.add = Add()

    def construct(self, x):
        '''construct'''
        identity = x
        out = self.bk1_conv0(x)
        out = self.bk1_bn0(out)
        out = self.bk1_relu0(out)
        out = self.bk1_conv1(out)
        out = self.bk1_bn1(out)

        identity = self.bk1_conv2(identity)
        identity = self.bk1_bn2(identity)

        out = self.add(out, identity)
        out = self.bk1_relu1(out)

        identity = out
        out = self.bk1_conv3(out)
        out = self.bk1_bn3(out)
        out = self.bk1_relu3(out)
        out = self.bk1_conv4(out)
        out = self.bk1_bn4(out)

        out = self.add(out, identity)
        out = self.bk1_relu4(out)
        return out


class Block2(Cell):
    '''Block2'''
    def __init__(self):
        super(Block2, self).__init__()

        self.bk2_conv0 = conv3x3(64, 128, stride=2, padding=1)
        self.bk2_bn0 = bn_with_initialize(128)
        self.bk2_relu0 = P.ReLU()
        self.bk2_conv1 = conv3x3(128, 128, stride=1, padding=1)
        self.bk2_bn1 = bn_with_initialize(128)
        self.bk2_conv2 = conv1x1(64, 128, stride=2, padding=0)
        self.bk2_bn2 = bn_with_initialize(128)
        self.bk2_relu1 = P.ReLU()

        self.bk2_conv3 = conv3x3(128, 128, stride=1, padding=1)
        self.bk2_bn3 = bn_with_initialize(128)
        self.bk2_relu3 = P.ReLU()
        self.bk2_conv4 = conv3x3(128, 128, stride=1, padding=1)
        self.bk2_bn4 = bn_with_initialize(128)
        self.bk2_relu4 = P.ReLU()

        self.cast = P.Cast()
        self.add = Add()

    def construct(self, x):
        '''construct'''
        identity = x
        out = self.bk2_conv0(x)
        out = self.bk2_bn0(out)
        out = self.bk2_relu0(out)
        out = self.bk2_conv1(out)
        out = self.bk2_bn1(out)

        identity = self.bk2_conv2(identity)
        identity = self.bk2_bn2(identity)

        out = self.add(out, identity)
        out = self.bk2_relu1(out)

        identity = out
        out = self.bk2_conv3(out)
        out = self.bk2_bn3(out)
        out = self.bk2_relu3(out)
        out = self.bk2_conv4(out)
        out = self.bk2_bn4(out)

        out = self.add(out, identity)
        out = self.bk2_relu4(out)
        return out


class FaceQABackbone(Cell):
    '''FaceQABackbone'''
    def __init__(self):
        super(FaceQABackbone, self).__init__()
        self.shape = P.Shape()
        self.reshape = P.Reshape()
        self.cast = P.Cast()

        self.conv0 = conv3x3(3, 64, stride=2, padding=1)
        self.bn0 = bn_with_initialize(64)
        self.relu0 = P.ReLU()
        self.conv1 = conv3x3(64, 64, stride=2, padding=1)
        self.bn1 = bn_with_initialize(64)
        self.relu1 = P.ReLU()
        self.backbone = nn.SequentialCell([
            Block1(),
            Block2()
        ])

        # branch euler
        self.euler_conv = conv3x3(128, 128, stride=2, padding=1)
        self.euler_bn = bn_with_initialize(128)
        self.euler_relu = P.ReLU()
        self.euler_fc1 = fc_with_initialize(128*6*6, 256)
        self.euler_relu1 = P.ReLU()
        self.euler_fc2 = fc_with_initialize(256, 128)
        self.euler_relu2 = P.ReLU()
        self.euler_fc3 = fc_with_initialize(128, 3)

        # branch heatmap
        self.kps_deconv = nn.Conv2dTranspose(128, 5, 4, stride=2, pad_mode='pad', group=1, dilation=1, padding=1,
                                             has_bias=False)
        self.kps_up = nn.Conv2dTranspose(5, 5, 4, stride=2, pad_mode='pad', group=1, dilation=1, padding=1,
                                         has_bias=False)

    def construct(self, x):
        '''construct'''
        # backbone
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.relu0(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.backbone(x)

        # branch euler
        out1 = self.euler_conv(x)
        out1 = self.euler_bn(out1)
        out1 = self.euler_relu(out1)
        b, _, _, _ = self.shape(out1)
        out1 = self.reshape(out1, (b, -1))
        out1 = self.euler_fc1(out1)
        out1 = self.euler_relu1(out1)
        out1 = self.euler_fc2(out1)
        out1 = self.euler_relu2(out1)
        out1 = self.euler_fc3(out1)

        # branch kps
        out2 = self.kps_deconv(x)
        out2 = self.kps_up(out2)

        return out1, out2


class BuildTrainNetwork(nn.Cell):
    '''BuildTrainNetwork'''
    def __init__(self, network, criterion):
        super(BuildTrainNetwork, self).__init__()
        self.network = network
        self.criterion = criterion

    def construct(self, input_data, label):
        out_eul, out_kps = self.network(input_data)
        loss = self.criterion(out_eul, out_kps, label)

        return loss
