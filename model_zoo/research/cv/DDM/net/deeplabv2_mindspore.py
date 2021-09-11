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

""" architecture of deeplabv2. """

import mindspore.nn as nn
from mindspore.common.initializer import Normal
from mindspore.ops import operations as P
from mindspore.ops import Shape

AFFINE_PAR = True

class Bottleneck(nn.Cell):
    """build bottleneck module"""
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, freeze_bn_affine=True):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride,
                               has_bias=False, weight_init=Normal(0.01))
        self.bn1 = nn.BatchNorm2d(planes, affine=AFFINE_PAR, use_batch_statistics=None)

        if freeze_bn_affine:
            for i in self.bn1.parameters_dict().values():
                i.requires_grad = False

        padding = dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=padding, has_bias=False, dilation=dilation,
                               weight_init=Normal(0.01), pad_mode="pad")
        self.bn2 = nn.BatchNorm2d(planes, affine=AFFINE_PAR, use_batch_statistics=None)

        if freeze_bn_affine:
            for i in self.bn2.parameters_dict().values():
                i.requires_grad = False

        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, has_bias=False, weight_init=Normal(0.01))
        self.bn3 = nn.BatchNorm2d(planes * 4, affine=AFFINE_PAR, use_batch_statistics=None)

        if freeze_bn_affine:
            for i in self.bn3.parameters_dict().values():
                i.requires_grad = False

        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride
        self.add = P.Add()

    def construct(self, x):
        """construct bottleneck module"""
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.add(out, residual)
        out = self.relu(out)
        return out

class ClassifierModule(nn.Cell):
    """build classify module"""
    def __init__(self, inplanes, dilation_series, padding_series, num_classes):
        super(ClassifierModule, self).__init__()
        self.conv2d_list = nn.CellList()

        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(
                nn.Conv2d(inplanes, num_classes, kernel_size=3, stride=1, padding=padding, pad_mode="pad",
                          dilation=dilation, has_bias=True, weight_init=Normal(0.01)))

    def construct(self, x):
        """construct classify module"""
        out = self.conv2d_list[0](x)

        for i in range(1, len(self.conv2d_list)):
            out += self.conv2d_list[i](x)

        return out


class ResNetMulti(nn.Cell):
    """build resnet"""
    def __init__(self, block, layers, num_classes, multi_level, freeze_bn_affine=True):
        self.multi_level = multi_level
        self.inplanes = 64
        super(ResNetMulti, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               has_bias=False, weight_init=Normal(0.01), pad_mode="pad")
        self.bn1 = nn.BatchNorm2d(64, affine=AFFINE_PAR, use_batch_statistics=None)
        self.freeze_bn_affine = freeze_bn_affine

        if self.freeze_bn_affine:
            for i in self.bn1.parameters_dict().values():
                i.requires_grad = False

        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="same")
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)

        if self.multi_level:
            self.layer5 = ClassifierModule(1024, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)

        self.layer6 = ClassifierModule(2048, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)
        self.shape = Shape()
        self.pad = nn.Pad(((0, 0), (0, 0), (1, 1), (1, 1)), "CONSTANT")

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        """define layers"""
        downsample = None

        if (stride != 1
                or self.inplanes != planes * block.expansion
                or dilation == 2
                or dilation == 4):
            downsample = nn.SequentialCell([
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, has_bias=False, weight_init=Normal(0.01)),
                nn.BatchNorm2d(planes * block.expansion, affine=AFFINE_PAR, use_batch_statistics=None)])

        if self.freeze_bn_affine:
            downsample = nn.SequentialCell([
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, has_bias=False, weight_init=Normal(0.01)),
                nn.BatchNorm2d(planes * block.expansion, affine=False, use_batch_statistics=None)])
            # for i in downsample._cells['1'].parameters_dict().values():
            #     i.requires_grad = False

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample,
                  freeze_bn_affine=self.freeze_bn_affine))
        self.inplanes = planes * block.expansion

        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation,
                                freeze_bn_affine=self.freeze_bn_affine))
            print(i)

        return nn.SequentialCell(layers)

    def construct(self, x):
        """construct resnet"""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pad(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        if self.multi_level:
            x1 = self.layer5(x)
        else:
            x1 = None

        x2 = self.layer4(x)
        x2 = self.layer6(x2)
        return x1, x2

    def freeze_batchnorm(self):
        """freeze batchnorm"""
        self.apply(freeze_bn_module)

def freeze_bn_module(m):
    """Freeze bn module.
    param m: a torch module
    """
    classname = type(m).__name__

    if classname.find('BatchNorm') != -1:
        m.eval()

def get_deeplab_v2(num_classes=19, multi_level=True, freeze_bn_affine=True):
    """get deeplabv2 net"""
    model = ResNetMulti(Bottleneck, [3, 4, 23, 3], num_classes, multi_level,
                        freeze_bn_affine=freeze_bn_affine)
    return model
