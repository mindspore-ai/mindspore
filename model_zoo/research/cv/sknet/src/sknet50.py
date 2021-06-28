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
'''sknet
The sample can be run on Ascend 910 AI processor.
'''

import numpy as np
import mindspore.nn as nn
from mindspore import Tensor
import mindspore.ops as ops
from src.util import GroupConv

def weight_variable_0(shape):
    """weight_variable_0"""
    zeros = np.zeros(shape).astype(np.float32)
    return Tensor(zeros)


def weight_variable_1(shape):
    """weight_variable_1"""
    ones = np.ones(shape).astype(np.float32)
    return Tensor(ones)


def conv3x3(in_channels, out_channels, stride=1, padding=0, pad_mode="same"):
    """3x3 convolution """
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=3, stride=stride, padding=padding,
                     has_bias=False, pad_mode=pad_mode)


def conv1x1(in_channels, out_channels, stride=1, padding=0):
    """1x1 convolution"""
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=1, stride=stride, padding=padding,
                     has_bias=False, pad_mode="same")


def conv7x7(in_channels, out_channels, stride=1, padding=0, pad_mode="same"):
    """7x7 convolution"""
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=7, stride=stride, padding=padding,
                     has_bias=False, pad_mode=pad_mode)


def bn_with_initialize(out_channels):
    """bn_with_initialize"""
    shape = (out_channels)
    mean = weight_variable_0(shape)
    var = weight_variable_1(shape)
    beta = weight_variable_0(shape)
    bn = nn.BatchNorm2d(out_channels, momentum=0.99, eps=0.00001, gamma_init='Uniform',
                        beta_init=beta, moving_mean_init=mean, moving_var_init=var)
    return bn


def bn_with_initialize_last(out_channels):
    """bn_with_initialize_last"""
    shape = (out_channels)
    mean = weight_variable_0(shape)
    var = weight_variable_1(shape)
    beta = weight_variable_0(shape)
    bn = nn.BatchNorm2d(out_channels, momentum=0.99, eps=0.00001, gamma_init='Uniform',
                        beta_init=beta, moving_mean_init=mean, moving_var_init=var)
    return bn


def fc_with_initialize(input_channels, out_channels):
    """fc_with_initialize"""
    return nn.Dense(input_channels, out_channels)


class SKConv(nn.Cell):
    """SKConv"""
    def __init__(self, features, G, M, r, stride=1, L=32):
        super(SKConv, self).__init__()
        d = max(int(features / r), L)
        self.M = M
        self.features = features
        self.G = G
        self.convs = nn.CellList([])
        for i in range(M):
            self.convs.append(nn.SequentialCell(
                GroupConv(features, features, kernel_size=3 + 2 * i, stride=stride, pad_mode='pad', pad=1 + i,
                          groups=G),
                nn.BatchNorm2d(features),
                nn.ReLU()
            ))
        self.fc = nn.Dense(features, d)
        self.fcs = nn.CellList([])
        for i in range(M):
            self.fcs.append(
                nn.Dense(d, features)
            )
        self.softmax = nn.Softmax(axis=1)
        self.feas = Tensor(0.1)
        self.attention_vectors = Tensor(0.1)
    def construct(self, x):
        """SKConv construct"""
        feas = self.feas
        for i in range(self.M):
            fea = self.convs[i](x)
            expand_dims = ops.ExpandDims()
            fea = expand_dims(fea, 1)
            if i == 0:
                feas = fea
            else:
                op = ops.Concat(1)
                feas = op((feas, fea))
        op = ops.ReduceSum()
        fea_U = op(feas, 1)
        fea_s = fea_U.mean(-1)
        fea_s = fea_s.mean(-1)
        fea_z = self.fc(fea_s)
        attention_vectors = self.attention_vectors
        for i in range(self.M):
            vector = self.fcs[i](fea_z)
            expand_dims = ops.ExpandDims()
            vector = expand_dims(vector, 1)
            if i == 0:
                attention_vectors = vector
            else:
                op = ops.Concat(1)
                attention_vectors = op((attention_vectors, vector))
        attention_vectors = self.softmax(attention_vectors)
        expand_dims = ops.ExpandDims()
        attention_vectors = expand_dims(attention_vectors, -1)
        attention_vectors = expand_dims(attention_vectors, -1)
        fea_v = feas * attention_vectors
        op = ops.ReduceSum()
        fea_v = op(fea_v, 1)
        return fea_v


class SKUnit(nn.Cell):
    """SKUnit"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels=None,
                 stride=1,
                 G=32, M=2, r=16, L=32, platform="Ascend"):
        super(SKUnit, self).__init__()

        if mid_channels is None:
            mid_channels = int(out_channels / 2)
        self.feas = nn.SequentialCell([conv1x1(in_channels, mid_channels, stride=1, padding=0),
                                       bn_with_initialize(mid_channels),
                                       SKConv(mid_channels, G=G, M=M, r=r, stride=stride),
                                       bn_with_initialize(mid_channels),
                                       conv1x1(mid_channels, out_channels, stride=1, padding=0),
                                       bn_with_initialize_last(out_channels)
                                       ])
        if in_channels == out_channels:
            self.shortcut = nn.SequentialCell()
        else:
            self.shortcut = nn.SequentialCell([
                conv1x1(in_channels, out_channels, stride=stride, padding=0),
                bn_with_initialize(out_channels)
            ])
        self.relu = ops.ReLU()

    def construct(self, x):
        """construct"""
        out = ops.tensor_add(self.feas(x), self.shortcut(x))
        out = self.relu(out)
        return out

class SKNet(nn.Cell):
    """SKNet"""

    def __init__(self, block, layer_nums, in_channels, out_channels,
                 strides, num_classes=10):
        """init"""
        super(SKNet, self).__init__()
        self.num_classes = num_classes
        self.basic_conv = nn.SequentialCell([conv7x7(3, 64, stride=2, padding=3, pad_mode='pad'),
                                             bn_with_initialize(64)
                                             ])
        self.relu = ops.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="same")
        self.squeeze = ops.Squeeze()
        self.layer1 = self._make_layer(block,
                                       layer_nums[0],
                                       in_channel=in_channels[0],
                                       out_channel=out_channels[0],
                                       stride=strides[0],
                                       )
        self.layer2 = self._make_layer(block,
                                       layer_nums[1],
                                       in_channel=in_channels[1],
                                       out_channel=out_channels[1],
                                       stride=strides[1],
                                       )
        self.layer3 = self._make_layer(block,
                                       layer_nums[2],
                                       in_channel=in_channels[2],
                                       out_channel=out_channels[2],
                                       stride=strides[2],
                                       )
        self.layer4 = self._make_layer(block,
                                       layer_nums[3],
                                       in_channel=in_channels[3],
                                       out_channel=out_channels[3],
                                       stride=strides[3],
                                       )
        self.pool = ops.ReduceMean(keep_dims=True)
        self.fc = fc_with_initialize(out_channels[3], num_classes)

    def construct(self, x):
        """construct"""
        x = self.basic_conv(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x, (2, 3))
        x = self.squeeze(x)
        x = self.fc(x)
        return x

    def _make_layer(self, block, blocks_num, in_channel, out_channel, stride=1, platform="Ascend"):
        """_make_layer"""
        layers = []
        layers.append(block(in_channel,
                            out_channel,
                            stride=stride))
        for _ in range(1, blocks_num):
            layers.append(block(out_channel, out_channel))

        return nn.SequentialCell(layers)


def sknet50(class_num=10):
    """
    Get sknet50 neural network.

    Args:
        class_num (int): Class number.

    Returns:
        Cell, cell instance of ResNet50 neural network.

    Examples:

    """
    return SKNet(SKUnit,
                 [3, 4, 6, 3],
                 [64, 256, 512, 1024],
                 [256, 512, 1024, 2048],
                 [1, 2, 2, 2],
                 class_num)
