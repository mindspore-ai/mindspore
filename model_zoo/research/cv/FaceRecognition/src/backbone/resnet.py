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
"""Face Recognition backbone."""
import math
import numpy as np
import mindspore.nn as nn
from mindspore.nn import Cell
from mindspore.ops.operations import Add
from mindspore.ops import operations as P
from mindspore.common.initializer import initializer
from mindspore.common import dtype as mstype

from src.backbone.head import get_head
from src import me_init
from src.custom_net import Cut, bn_with_initialize, fc_with_initialize, conv1x1, conv3x3


__all__ = ['get_backbone']

class Sigmoid(Cell):
    def __init__(self):
        super(Sigmoid, self).__init__()
        self.sigmoid = P.Sigmoid()

    def construct(self, x):
        out = self.sigmoid(x)
        return out


class SEBlock(Cell):
    '''SEBlock'''
    def __init__(self, channel, reduction=16, act_type='relu'):
        super(SEBlock, self).__init__()

        self.fc1 = fc_with_initialize(channel, channel // reduction)
        self.act_layer = nn.PReLU(
            channel // reduction) if act_type == 'prelu' else P.ReLU()
        self.fc2 = fc_with_initialize(channel // reduction, channel)
        self.sigmoid = Sigmoid().add_flags_recursive(fp32=True)
        self.reshape = P.Reshape()
        self.shape = P.Shape()
        self.reduce_mean = P.ReduceMean(True)
        self.cast = P.Cast()

    def construct(self, x):
        '''construct'''
        b, c, _, _ = self.shape(x)
        y = self.reduce_mean(x, (2, 3))
        y = self.reshape(y, (b, c))
        y = self.fc1(y)
        y = self.act_layer(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
        y = self.cast(y, mstype.float16)
        y = self.reshape(y, (b, c, 1, 1))

        return x * y


class IRBlock(Cell):
    '''IRBlock'''
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_se=1, pre_bn=1, use_inference=0,
                 act_type='relu'):
        super(IRBlock, self).__init__()

        if pre_bn == 1:
            self.bn1 = bn_with_initialize(inplanes, use_inference=use_inference)
        else:
            self.bn1 = Cut()
        self.conv1 = conv3x3(inplanes, planes, stride=1)
        self.bn2 = bn_with_initialize(planes, use_inference=use_inference)
        self.act_layer = nn.PReLU(
            planes) if act_type == 'prelu' else P.ReLU()
        self.conv2 = conv3x3(planes, planes, stride=stride)
        self.bn3 = bn_with_initialize(planes, use_inference=use_inference)

        if downsample is None:
            self.downsample = Cut()
        else:
            self.downsample = downsample

        self.use_se = use_se
        if use_se == 1:
            self.se = SEBlock(planes, act_type=act_type)
        self.add = Add()
        self.cast = P.Cast()

    def construct(self, x):
        '''construct'''
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.act_layer(out)
        out = self.conv2(out)
        out = self.bn3(out)

        if self.use_se == 1:
            out = self.se(out)
        identity = self.downsample(x)

        identity = self.cast(identity, mstype.float16)
        out = self.cast(out, mstype.float16)
        out = self.add(out, identity)

        return out


class DownSample(Cell):
    '''DownSample'''
    def __init__(self, inplanes, planes, expansion, stride, use_inference=0):
        super(DownSample, self).__init__()
        self.conv1 = conv1x1(inplanes, planes * expansion,
                             stride=stride, pad_mode="valid")
        self.bn1 = bn_with_initialize(planes * expansion, use_inference=use_inference)

    def construct(self, x):
        out = self.conv1(x)
        out = self.bn1(out)

        return out


class MakeLayer(Cell):
    '''MakeLayer'''
    def __init__(self, block, inplanes, planes, blocks, args, stride=1):
        super(MakeLayer, self).__init__()

        self.inplanes = inplanes
        self.downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            self.downsample = DownSample(
                self.inplanes, planes, block.expansion, stride, use_inference=args.inference)

        self.layers = []
        self.layers.append(block(self.inplanes, planes, stride, self.downsample, use_se=args.use_se, pre_bn=args.pre_bn,
                                 use_inference=args.inference, act_type=args.act_type))
        self.inplanes = planes
        for _ in range(1, blocks):
            self.layers.append(block(self.inplanes, planes, use_se=args.use_se, pre_bn=args.pre_bn,
                                     use_inference=args.inference, act_type=args.act_type))
        self.layers = nn.CellList(self.layers)

    def construct(self, x):
        for block in self.layers:
            x = block(x)
        return x


class FaceResNet(Cell):
    '''FaceResNet'''
    def __init__(self, block, layers, args):
        super(FaceResNet, self).__init__()

        self.act_type = args.act_type
        self.inplanes = 64
        self.use_se = args.use_se

        self.conv1 = conv3x3(3, 64, stride=1)
        self.bn1 = bn_with_initialize(64, use_inference=args.inference)
        self.prelu = nn.PReLU(64) if self.act_type == 'prelu' else P.ReLU()
        self.layer1 = MakeLayer(
            block, planes=64, inplanes=self.inplanes, blocks=layers[0], stride=2, args=args)
        self.inplanes = 64
        self.layer2 = MakeLayer(
            block, planes=128, inplanes=self.inplanes, blocks=layers[1], stride=2, args=args)
        self.inplanes = 128
        self.layer3 = MakeLayer(
            block, planes=256, inplanes=self.inplanes, blocks=layers[2], stride=2, args=args)
        self.inplanes = 256
        self.layer4 = MakeLayer(
            block, planes=512, inplanes=self.inplanes, blocks=layers[3], stride=2, args=args)
        self.head = get_head(args)

        np.random.seed(1)
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                cell.weight.set_data(initializer(me_init.ReidKaimingUniform(a=math.sqrt(5), mode='fan_out'),
                                                 cell.weight.shape))
                if cell.bias is not None:
                    cell.bias.set_data(initializer('zeros', cell.bias.shape))
            elif isinstance(cell, nn.Dense):
                cell.weight.set_data(initializer(me_init.ReidKaimingNormal(a=math.sqrt(5), mode='fan_out'),
                                                 cell.weight.shape))
                if cell.bias is not None:
                    cell.bias.set_data(initializer('zeros', cell.bias.shape))
            elif isinstance(cell, (nn.BatchNorm2d, nn.BatchNorm1d)):
                # defulat gamma 1 and beta 0, and if you set should be careful for the IRBlock gamma value
                pass
        for _, cell in self.cells_and_names():
            if isinstance(cell, IRBlock):
                # be careful for bn3 Do not change the name unless IRBlock last bn change name
                cell.bn3.gamma.set_data(initializer('zeros', cell.bn3.gamma.shape))

    def construct(self, x):
        '''construct'''
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.head(x)

        return x

def _faceresnet(block, layers, args):
    model = FaceResNet(block, layers, args)
    return model

def get_faceresnet(num_layers, args):
    '''get_faceresnet'''
    if num_layers == 9:
        units = [1, 1, 1, 1]
    elif num_layers == 18:
        units = [2, 2, 2, 2]
    elif num_layers == 34:
        units = [3, 4, 6, 3]
    elif num_layers == 49:
        units = [3, 4, 14, 3]
    elif num_layers == 50:
        units = [3, 4, 14, 3]
    elif num_layers == 74:
        units = [3, 6, 24, 3]
    elif num_layers == 90:
        units = [3, 8, 30, 3]
    elif num_layers == 100:
        units = [3, 13, 30, 3]
    elif num_layers == 101:
        units = [3, 4, 23, 3]
    elif num_layers == 152:
        units = [3, 8, 36, 3]
    elif num_layers == 200:
        units = [3, 24, 36, 3]
    elif num_layers == 269:
        units = [3, 30, 48, 8]
    else:
        raise ValueError(
            "no experiments done on num_layers {}, you can do it yourself".format(num_layers))
    return _faceresnet(IRBlock, units, args)

def get_backbone_faceres(args):
    backbone_type = args.backbone
    layer_num = int(backbone_type[1:])
    return get_faceresnet(layer_num, args)

def get_backbone(args):
    return get_backbone_faceres(args)
