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
"""Face attribute resnet18 backbone."""
import mindspore.nn as nn
from mindspore.ops.operations import Add
from mindspore.ops import operations as P
from mindspore.nn import Cell

from src.FaceAttribute.custom_net import Cut, bn_with_initialize, conv1x1, conv3x3
from src.FaceAttribute.head_factory_softmax import get_attri_head

__all__ = ['get_resnet18']


class IRBlock(Cell):
    '''IRBlock.'''
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(IRBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride=stride)
        self.bn1 = bn_with_initialize(planes)
        self.relu1 = P.ReLU()
        self.conv2 = conv3x3(planes, planes, stride=1)
        self.bn2 = bn_with_initialize(planes)

        if downsample is None:
            self.downsample = Cut()
        else:
            self.downsample = downsample

        self.add = Add()
        self.cast = P.Cast()
        self.relu2 = P.ReLU()

    def construct(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        identity = self.downsample(x)
        out = self.add(out, identity)
        out = self.relu2(out)
        return out


class DownSample(Cell):
    def __init__(self, inplanes, planes, expansion, stride):
        super(DownSample, self).__init__()
        self.conv1 = conv1x1(inplanes, planes * expansion, stride=stride, pad_mode="valid")
        self.bn1 = bn_with_initialize(planes * expansion)

    def construct(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        return out


class MakeLayer(Cell):
    '''Make layer function.'''
    def __init__(self, block, inplanes, planes, blocks, stride=1):
        super(MakeLayer, self).__init__()

        self.inplanes = inplanes
        self.downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            self.downsample = DownSample(self.inplanes, planes, block.expansion, stride)

        self.layers = []
        self.layers.append(block(self.inplanes, planes, stride, self.downsample))
        self.inplanes = planes
        for _ in range(1, blocks):
            self.layers.append(block(self.inplanes, planes))
        self.layers = nn.CellList(self.layers)

    def construct(self, x):
        for block in self.layers:
            x = block(x)
        return x


class AttriResNet(Cell):
    '''Resnet for attribute.'''
    def __init__(self, block, layers, flat_dim, fc_dim, attri_num_list):
        super(AttriResNet, self).__init__()

        # resnet18
        self.inplanes = 32
        self.conv1 = conv3x3(3, self.inplanes, stride=1)
        self.bn1 = bn_with_initialize(self.inplanes)
        self.relu = P.ReLU()
        self.layer1 = MakeLayer(block, inplanes=32, planes=64, blocks=layers[0], stride=2)
        self.layer2 = MakeLayer(block, inplanes=64, planes=128, blocks=layers[1], stride=2)
        self.layer3 = MakeLayer(block, inplanes=128, planes=256, blocks=layers[2], stride=2)
        self.layer4 = MakeLayer(block, inplanes=256, planes=512, blocks=layers[3], stride=2)

        # avg global pooling
        self.mean = P.ReduceMean(keep_dims=True)
        self.shape = P.Shape()
        self.reshape = P.Reshape()
        self.head = get_attri_head(flat_dim, fc_dim, attri_num_list)

    def construct(self, x):
        '''Construct function.'''
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.mean(x, (2, 3))
        b, c, _, _ = self.shape(x)
        x = self.reshape(x, (b, c))
        return self.head(x)


def get_resnet18(args):
    '''Build resnet18 for attribute.'''
    flat_dim = args.flat_dim
    fc_dim = args.fc_dim
    str_classes = args.classes.strip().split(',')
    if args.attri_num != len(str_classes):
        print('args warning: attri_num != classes num')
        return None
    attri_num_list = []
    for i, _ in enumerate(str_classes):
        attri_num_list.append(int(str_classes[i]))

    attri_resnet18 = AttriResNet(IRBlock, (2, 2, 2, 2), flat_dim, fc_dim, attri_num_list)

    return attri_resnet18
