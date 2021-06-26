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
"""
python resnet_ibn.py
"""
import mindspore.nn as nn
import mindspore.ops as ops

from src.modules import IBN


class BasicBlock_IBN(nn.Cell):
    """
    BasicBlock_IBN

    Args:
        x (Tensor): input tensor
    """
    expansion = 1

    def __init__(self, inplanes, planes, ibn=None, stride=1, downsample=None):
        super(BasicBlock_IBN, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=0, has_bias=False)
        if ibn == 'a':
            self.bn1 = IBN(planes)
        else:
            self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, pad_mode='pad', padding=1, has_bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.IN = nn.GroupNorm(planes, planes, affine=True) if ibn == 'b' else None
        self.downsample = downsample
        self.stride = stride

    def construct(self, x):
        """
        construct
        """
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if self.IN is not None:
            out = self.IN(out)
        out = self.relu(out)

        return out


class Bottleneck_IBN(nn.Cell):
    """
    Bottleneck_IBN

    Args:
        x (Tensor): input tensor
    """
    expansion = 4

    def __init__(self, inplanes, planes, ibn=None, stride=1, downsample=None):
        super(Bottleneck_IBN, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, has_bias=False)
        if ibn == 'a':
            self.bn1 = IBN(planes)
        else:
            self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=0, has_bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, has_bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.IN = nn.GroupNorm(planes * 4, planes * 4, affine=True) if ibn == 'b' else None
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def construct(self, x):
        """
        construct
        """
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

        out += residual
        if self.IN is not None:
            out = self.IN(out)
        out = self.relu(out)

        return out


class ResNet_IBN(nn.Cell):
    """
    ResNet_IBN

    Args:
        x (Tensor): input tensor
    """

    def __init__(self,
                 block,
                 layers,
                 ibn_cfg=('a', 'a', 'a', None),
                 num_classes=1000):
        self.inplanes = 64
        super(ResNet_IBN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, pad_mode='pad',
                               has_bias=False)
        if ibn_cfg[0] == 'b':
            self.bn1 = nn.GroupNorm(64, 64, affine=True)
        else:
            self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same')
        self.layer1 = self._make_layer(block, 64, layers[0], ibn=ibn_cfg[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, ibn=ibn_cfg[1])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, ibn=ibn_cfg[2])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, ibn=ibn_cfg[3])
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=7)
        self.fc = nn.Dense(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1, ibn=None):
        """
        _make_layer
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.SequentialCell([
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, has_bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            ])

        layers = []
        layers.append(block(self.inplanes, planes,
                            None if ibn == 'b' else ibn,
                            stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                None if (ibn == 'b' and i < blocks - 1) else ibn))

        return nn.SequentialCell(*layers)

    def construct(self, x):
        """
        construct
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
        reshape = ops.Reshape()
        x = reshape(x, (x.shape[0], -1))
        x = self.fc(x)

        return x


def resnet18_ibn_a(**kwargs):
    """
    Constructs a ResNet-18-IBN-a model.
    """
    model = ResNet_IBN(block=BasicBlock_IBN,
                       layers=[2, 2, 2, 2],
                       ibn_cfg=('a', 'a', 'a', None),
                       **kwargs)
    return model


def resnet34_ibn_a(**kwargs):
    """
    Constructs a ResNet-34-IBN-a model.
    """
    model = ResNet_IBN(block=BasicBlock_IBN,
                       layers=[3, 4, 6, 3],
                       ibn_cfg=('a', 'a', 'a', None),
                       **kwargs)
    return model


def resnet50_ibn_a(**kwargs):
    """
    Constructs a ResNet-50-IBN-a model.
    """
    model = ResNet_IBN(block=Bottleneck_IBN,
                       layers=[3, 4, 6, 3],
                       ibn_cfg=('a', 'a', 'a', None),
                       **kwargs)
    return model


def resnet101_ibn_a(**kwargs):
    """
    Constructs a ResNet-101-IBN-a model.
    """
    model = ResNet_IBN(block=Bottleneck_IBN,
                       layers=[3, 4, 23, 3],
                       ibn_cfg=('a', 'a', 'a', None),
                       **kwargs)
    return model


def resnet152_ibn_a(**kwargs):
    """
    Constructs a ResNet-152-IBN-a model.
    """
    model = ResNet_IBN(block=Bottleneck_IBN,
                       layers=[3, 8, 36, 3],
                       ibn_cfg=('a', 'a', 'a', None),
                       **kwargs)
    return model


def resnet18_ibn_b(**kwargs):
    """
    Constructs a ResNet-18-IBN-b model.
    """
    model = ResNet_IBN(block=BasicBlock_IBN,
                       layers=[2, 2, 2, 2],
                       ibn_cfg=('b', 'b', None, None),
                       **kwargs)
    return model


def resnet34_ibn_b(**kwargs):
    """
    Constructs a ResNet-34-IBN-b model.
    """
    model = ResNet_IBN(block=BasicBlock_IBN,
                       layers=[3, 4, 6, 3],
                       ibn_cfg=('b', 'b', None, None),
                       **kwargs)
    return model


def resnet50_ibn_b(**kwargs):
    """
    Constructs a ResNet-50-IBN-b model.
    """
    model = ResNet_IBN(block=Bottleneck_IBN,
                       layers=[3, 4, 6, 3],
                       ibn_cfg=('b', 'b', None, None),
                       **kwargs)
    return model


def resnet101_ibn_b(**kwargs):
    """
    Constructs a ResNet-101-IBN-b model.
    """
    model = ResNet_IBN(block=Bottleneck_IBN,
                       layers=[3, 4, 23, 3],
                       ibn_cfg=('b', 'b', None, None),
                       **kwargs)
    return model


def resnet152_ibn_b(**kwargs):
    """
    Constructs a ResNet-152-IBN-b model.
    """
    model = ResNet_IBN(block=Bottleneck_IBN,
                       layers=[3, 8, 36, 3],
                       ibn_cfg=('b', 'b', None, None),
                       **kwargs)
    return model
