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
# =============================================================================
""" resnetv2.py """
import mindspore.nn as nn
from mindspore.ops import operations as P

class PreActBottleNeck(nn.Cell):
    """ PreActBottleNeck """
    expansion = 4

    def __init__(self,
                 in_planes,
                 planes,
                 stride=1):
        super(PreActBottleNeck, self).__init__()

        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm2d(in_planes, eps=1e-5, momentum=0.9)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-5, momentum=0.9)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, pad_mode='pad')
        self.bn3 = nn.BatchNorm2d(planes, eps=1e-5, momentum=0.9)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, stride=1)

        self.downtown = False
        if stride != 1 or in_planes != self.expansion*planes:
            self.downtown = True
            self.shortcut = nn.SequentialCell([nn.Conv2d(in_planes, self.expansion*planes,
                                                         kernel_size=1, stride=stride)])

        self.add = P.TensorAdd()

    def construct(self, x):
        """ construct network """
        out = self.bn1(x)
        out = self.relu(out)
        if self.downtown:
            identity = self.shortcut(out)
        else:
            identity = x
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        out = self.add(out, identity)
        return out


class PreActResNet(nn.Cell):
    """ PreActResNet """
    def __init__(self,
                 block,
                 num_blocks,
                 in_planes,
                 planes,
                 strides,
                 low_memory,
                 num_classes=10):
        super(PreActResNet, self).__init__()
        self.in_planes = in_planes
        self.low_memory = low_memory

        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=7, stride=2)
        self.conv2 = nn.Conv2d(3, self.in_planes, kernel_size=3, stride=1, pad_mode='pad', padding=1)

        self.layer1 = self._make_layer(block,
                                       planes=planes[0],
                                       num_blocks=num_blocks[0],
                                       stride=strides[0])
        self.layer2 = self._make_layer(block,
                                       planes=planes[1],
                                       num_blocks=num_blocks[1],
                                       stride=strides[1])
        self.layer3 = self._make_layer(block,
                                       planes=planes[2],
                                       num_blocks=num_blocks[2],
                                       stride=strides[2])
        self.layer4 = self._make_layer(block,
                                       planes=planes[3],
                                       num_blocks=num_blocks[3],
                                       stride=strides[3])
        self.mean = P.ReduceMean(keep_dims=True)
        self.flatten = nn.Flatten()
        self.linear = nn.Dense(planes[3]*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        layers = []
        strides = [stride] + [1]*(num_blocks-1)

        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion

        return nn.SequentialCell(layers)

    def construct(self, x):
        """ construct network """
        if self.low_memory:
            out = self.conv1(x)
        else:
            out = self.conv2(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.mean(out, (2, 3))
        out = out.view(out.shape[0], -1)
        out = self.linear(out)

        return out


def PreActResNet50(class_num=10, low_memory=False):
    return PreActResNet(PreActBottleNeck,
                        num_blocks=[3, 4, 6, 3],
                        in_planes=64,
                        planes=[64, 128, 256, 512],
                        strides=[1, 2, 2, 2],
                        low_memory=low_memory,
                        num_classes=class_num)


def PreActResNet101(class_num=10, low_memory=False):
    return PreActResNet(PreActBottleNeck,
                        num_blocks=[3, 4, 23, 3],
                        in_planes=64,
                        planes=[64, 128, 256, 512],
                        strides=[1, 2, 2, 2],
                        low_memory=low_memory,
                        num_classes=class_num)


def PreActResNet152(class_num=10, low_memory=False):
    return PreActResNet(PreActBottleNeck,
                        num_blocks=[3, 8, 36, 3],
                        in_planes=64,
                        planes=[64, 128, 256, 512],
                        strides=[1, 2, 2, 2],
                        low_memory=low_memory,
                        num_classes=class_num)
