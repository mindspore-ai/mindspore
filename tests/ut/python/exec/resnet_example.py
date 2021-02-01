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
"""
resnet50 example
"""
import numpy as np

import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P
from ..ut_filter import non_graph_engine


def conv3x3(in_channels, out_channels, stride=1, padding=1):
    """3x3 convolution """
    weight = Tensor(np.ones([out_channels, in_channels, 3, 3]).astype(np.float32) * 0.01)
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=3, stride=stride, padding=padding, weight_init=weight)


def conv1x1(in_channels, out_channels, stride=1, padding=0):
    """1x1 convolution"""
    weight = Tensor(np.ones([out_channels, in_channels, 1, 1]).astype(np.float32) * 0.01)
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=1, stride=stride, padding=padding, weight_init=weight)


def bn_with_initialize(out_channels):
    shape = (out_channels)
    mean = Tensor(np.ones(shape).astype(np.float32) * 0.01)
    var = Tensor(np.ones(shape).astype(np.float32) * 0.01)
    beta = Tensor(np.ones(shape).astype(np.float32) * 0.01)
    gamma = Tensor(np.ones(shape).astype(np.float32) * 0.01)
    return nn.BatchNorm2d(num_features=out_channels,
                          beta_init=beta,
                          gamma_init=gamma,
                          moving_mean_init=mean,
                          moving_var_init=var)


class ResidualBlock(nn.Cell):
    """
    residual Block
    """
    expansion = 4

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 down_sample=False):
        super(ResidualBlock, self).__init__()

        out_chls = out_channels // self.expansion
        self.conv1 = conv1x1(in_channels, out_chls, stride=stride, padding=0)
        self.bn1 = bn_with_initialize(out_chls)

        self.conv2 = conv3x3(out_chls, out_chls, stride=1, padding=1)
        self.bn2 = bn_with_initialize(out_chls)

        self.conv3 = conv1x1(out_chls, out_channels, stride=1, padding=0)
        self.bn3 = bn_with_initialize(out_channels)

        self.relu = nn.ReLU()
        self.downsample = down_sample

        self.conv_down_sample = conv1x1(in_channels, out_channels,
                                        stride=stride, padding=0)
        self.bn_down_sample = bn_with_initialize(out_channels)
        self.add = P.Add()

    def construct(self, x):
        """
        :param x:
        :return:
        """
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample:
            identity = self.conv_down_sample(identity)
            identity = self.bn_down_sample(identity)

        out = self.add(out, identity)
        out = self.relu(out)

        return out


class MakeLayer3(nn.Cell):
    """
    make resnet50 3 layers
    """

    def __init__(self, block, in_channels, out_channels, stride):
        super(MakeLayer3, self).__init__()
        self.block_down_sample = block(in_channels, out_channels,
                                       stride=stride, down_sample=True)
        self.block1 = block(out_channels, out_channels, stride=1)
        self.block2 = block(out_channels, out_channels, stride=1)

    def construct(self, x):
        x = self.block_down_sample(x)
        x = self.block1(x)
        x = self.block2(x)

        return x


class MakeLayer4(nn.Cell):
    """
    make resnet50 4 layers
    """

    def __init__(self, block, in_channels, out_channels, stride):
        super(MakeLayer4, self).__init__()
        self.block_down_sample = block(in_channels, out_channels,
                                       stride=stride, down_sample=True)
        self.block1 = block(out_channels, out_channels, stride=1)
        self.block2 = block(out_channels, out_channels, stride=1)
        self.block3 = block(out_channels, out_channels, stride=1)

    def construct(self, x):
        x = self.block_down_sample(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        return x


class MakeLayer6(nn.Cell):
    """
    make resnet50 6 layers

    """

    def __init__(self, block, in_channels, out_channels, stride):
        super(MakeLayer6, self).__init__()
        self.block_down_sample = block(in_channels, out_channels,
                                       stride=stride, down_sample=True)
        self.block1 = block(out_channels, out_channels, stride=1)
        self.block2 = block(out_channels, out_channels, stride=1)
        self.block3 = block(out_channels, out_channels, stride=1)
        self.block4 = block(out_channels, out_channels, stride=1)
        self.block5 = block(out_channels, out_channels, stride=1)

    def construct(self, x):
        x = self.block_down_sample(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)

        return x


class ResNet50(nn.Cell):
    """
    resnet nn.Cell
    """

    def __init__(self, block, num_classes=100):
        super(ResNet50, self).__init__()

        weight_conv = Tensor(np.ones([64, 3, 7, 7]).astype(np.float32) * 0.01)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, weight_init=weight_conv)
        self.bn1 = bn_with_initialize(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)

        self.layer1 = MakeLayer3(
            block, in_channels=64, out_channels=256, stride=1)
        self.layer2 = MakeLayer4(
            block, in_channels=256, out_channels=512, stride=2)
        self.layer3 = MakeLayer6(
            block, in_channels=512, out_channels=1024, stride=2)
        self.layer4 = MakeLayer3(
            block, in_channels=1024, out_channels=2048, stride=2)

        self.avgpool = nn.AvgPool2d(7, 1)
        self.flatten = nn.Flatten()

        weight_fc = Tensor(np.ones([num_classes, 512 * block.expansion]).astype(np.float32) * 0.01)
        bias_fc = Tensor(np.ones([num_classes]).astype(np.float32) * 0.01)
        self.fc = nn.Dense(512 * block.expansion, num_classes, weight_init=weight_fc, bias_init=bias_fc)

    def construct(self, x):
        """
        :param x:
        :return:
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)

        return x


def resnet50():
    return ResNet50(ResidualBlock, 10)


@non_graph_engine
def test_compile():
    net = resnet50()
    input_data = Tensor(np.ones([1, 3, 224, 224]).astype(np.float32) * 0.01)

    output = net(input_data)
    print(output.asnumpy())
