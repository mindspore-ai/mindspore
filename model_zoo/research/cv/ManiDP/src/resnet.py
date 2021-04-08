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
"""ResNet."""
import numpy as np
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore import ops
from mindspore.common.tensor import Tensor
import mindspore.common.dtype as mstype


def _weight_variable(shape, factor=0.01):
    init_value = np.random.randn(*shape).astype(np.float32) * factor
    return Tensor(init_value)


def _conv3x3(in_channel, out_channel, stride=1):
    weight_shape = (out_channel, in_channel, 3, 3)
    weight = _weight_variable(weight_shape)
    return nn.Conv2d(in_channel, out_channel,
                     kernel_size=3, stride=stride, padding=1, pad_mode='pad', weight_init=weight)


def _conv1x1(in_channel, out_channel, stride=1):
    weight_shape = (out_channel, in_channel, 1, 1)
    weight = _weight_variable(weight_shape)
    return nn.Conv2d(in_channel, out_channel,
                     kernel_size=1, stride=stride, padding=0, pad_mode='same', weight_init=weight)


def _conv7x7(in_channel, out_channel, stride=1):
    weight_shape = (out_channel, in_channel, 7, 7)
    weight = _weight_variable(weight_shape)
    return nn.Conv2d(in_channel, out_channel,
                     kernel_size=7, stride=stride, padding=3, pad_mode='pad', weight_init=weight)


def _bn(channel):
    return nn.BatchNorm2d(channel)


def _bn_last(channel):
    return nn.BatchNorm2d(channel)


def _fc(in_channel, out_channel, bias=True):
    weight_shape = (out_channel, in_channel)
    weight = _weight_variable(weight_shape)
    return nn.Dense(in_channel, out_channel, has_bias=bias, weight_init=weight, bias_init=0)


class MaskBlock(nn.Cell):
    """
    ResNet basic mask block definition.

    Args:
        in_channels (int): number of input channels.
        out_channels (int): number of output channels.
        num (int): layer number.
        thres (list): threshold of layers.
    Returns:
        Tensor, output tensor.
    """

    def __init__(self, in_channels, out_channels, num, thres=None):
        super(MaskBlock, self).__init__()
        #self.target_pruning_rate = gate_factor
        self.clamp_min = Tensor(0, mstype.float32)
        self.clamp_max = Tensor(1000, mstype.float32)

        if out_channels < 80:
            squeeze_rate = 1
        else:
            squeeze_rate = 2

        self.avg_pool = P.ReduceMean(keep_dims=False)
        self.fc1 = _fc(in_channels, out_channels // squeeze_rate, bias=False)
        self.fc2 = _fc(out_channels // squeeze_rate, out_channels, bias=True)
        self.relu = P.ReLU()

        self.thre = thres[num]
        self.print = P.Print()

    def construct(self, x):
        """construct"""
        x_averaged = self.avg_pool(x, (2, 3))
        y = self.fc1(x_averaged)
        y = self.relu(y)
        y = self.fc2(y)

        mask_before = self.relu(y)
        mask_before = ops.clip_by_value(mask_before, self.clamp_min, self.clamp_max)
        tmp = ops.Greater()(mask_before, self.thre)
        mask = mask_before * tmp

        return mask


class MaskedBasicblock(nn.Cell):
    """
    ResNet basic mask block definition.

    Args:
        inplanes (int): number of input channels.
        planes (int): number of output channels.
        stride (int): convolution kernel stride.
        downsample (Cell): downsample layer.
        num (int): layer number.
        thres (list): threshold of layers.
    Returns:
        Tensor, output tensor.
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, num=0, thres=None):
        super(MaskedBasicblock, self).__init__()

        self.conv_a = _conv3x3(inplanes, planes, stride=stride)
        self.bn_a = _bn(planes)

        self.conv_b = _conv3x3(planes, planes, stride=1)
        self.bn_b = _bn(planes)

        self.downsample = downsample

        self.mb1 = MaskBlock(inplanes, planes, num*2, thres)
        self.mb2 = MaskBlock(planes, planes, num*2+1, thres)
        self.relu = P.ReLU()
        self.expand_dims = ops.ExpandDims()

    def construct(self, x):
        """construct"""
        residual = x

        mask1 = self.mb1(x)

        basicblock = self.conv_a(x)
        basicblock = self.bn_a(basicblock)
        basicblock = self.relu(basicblock)

        basicblock = basicblock * self.expand_dims(self.expand_dims(mask1, -1), -1)
        mask2 = self.mb2(basicblock)

        basicblock = self.conv_b(basicblock)
        basicblock = self.bn_b(basicblock)


        basicblock = basicblock* self.expand_dims(self.expand_dims(mask2, -1), -1)


        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu(residual + basicblock)


class CifarResNet(nn.Cell):
    """
    ResNet architecture.

    Args:
        block (Cell): block for network.
        depth (int): network depth.
        num_classes (int): The number of classes that the training images are belonging to.
        thres (list): threshold of layers.
    Returns:
        Tensor, output tensor.
    """
    def __init__(self, block, depth, num_classes, thres):
        super(CifarResNet, self).__init__()

        layer_blocks = (depth - 2) // 6
        self.num_classes = num_classes

        self.conv_1_3x3 = _conv3x3(3, 16, stride=1)
        self.bn_1 = _bn(16)
        self.relu = P.ReLU()

        self.inplanes = 16
        self.stage_1 = self._make_layer(block, 16, layer_blocks, 1, s_num=0, thres=thres)
        self.stage_2 = self._make_layer(block, 32, layer_blocks, 2, s_num=1, thres=thres)
        self.stage_3 = self._make_layer(block, 64, layer_blocks, 2, s_num=2, thres=thres)
        self.avgpool = nn.AvgPool2d(8)
        self.classifier = _fc(64 * block.expansion, num_classes)
        self.flatten = nn.Flatten()

    def _make_layer(self, block, planes, blocks, stride=1, s_num=0, thres=None):
        """make layer"""
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.SequentialCell([_conv1x1(self.inplanes, planes * block.expansion, stride=stride)])
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, num=s_num*3+0, thres=thres))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, num=s_num*3+i, thres=thres))

        return nn.SequentialCell(layers)

    def construct(self, x):
        """construct"""
        x = self.conv_1_3x3(x)
        x = self.relu(self.bn_1(x))
        x = self.stage_1(x)
        x = self.stage_2(x)
        x = self.stage_3(x)

        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x


def resnet20(num_classes=10, thres=None):
    model = CifarResNet(MaskedBasicblock, 20, num_classes, thres)
    return model

def resnet56(num_classes=10):
    model = CifarResNet(MaskedBasicblock, 56, num_classes)
    return model
