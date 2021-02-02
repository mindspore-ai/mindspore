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

import mindspore.nn as nn
from mindspore.ops.operations import Add
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.nn import Dense, Cell
from mindspore.common import dtype as mstype
from mindspore.common.initializer import initializer
from mindspore import Tensor, Parameter

from src import me_init


class Cut(nn.Cell):



    def construct(self, x):
        return x


def bn_with_initialize(out_channels):
    bn = nn.BatchNorm2d(out_channels, momentum=0.9, eps=1e-5).add_flags_recursive(fp32=True)
    return bn


def fc_with_initialize(input_channels, out_channels):
    return Dense(input_channels, out_channels)


def conv3x3(in_channels, out_channels, stride=1, groups=1, dilation=1, pad_mode="pad", padding=1, bias=True):
    """3x3 convolution with padding"""

    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                     pad_mode=pad_mode, group=groups, has_bias=bias, dilation=dilation, padding=padding)


def conv1x1(in_channels, out_channels, pad_mode="pad", stride=1, padding=0, bias=True):
    """1x1 convolution"""
    return nn.Conv2d(in_channels, out_channels, pad_mode=pad_mode, kernel_size=1, stride=stride, has_bias=bias,
                     padding=padding)


def conv4x4(in_channels, out_channels, stride=1, groups=1, dilation=1, pad_mode="pad", padding=1, bias=True):
    """4x4 convolution with padding"""

    return nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride,
                     pad_mode=pad_mode, group=groups, has_bias=bias, dilation=dilation, padding=padding)


class BaseBlock(Cell):
    '''BaseBlock'''
    def __init__(self, channels):
        super(BaseBlock, self).__init__()

        self.conv1 = conv3x3(channels, channels, stride=1, padding=1, bias=False)
        self.bn1 = bn_with_initialize(channels)
        self.relu1 = P.ReLU()
        self.conv2 = conv3x3(channels, channels, stride=1, padding=1, bias=False)
        self.bn2 = bn_with_initialize(channels)
        self.relu2 = P.ReLU()

        self.cast = P.Cast()
        self.add = Add()

    def construct(self, x):
        '''Construct function.'''
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        # hand cast
        identity = self.cast(identity, mstype.float16)
        out = self.cast(out, mstype.float16)

        out = self.add(out, identity)
        return out


class MakeLayer(Cell):
    '''MakeLayer'''
    def __init__(self, block, inplanes, planes, blocks, stride=2):
        super(MakeLayer, self).__init__()

        self.conv = conv3x3(inplanes, planes, stride=stride, padding=1, bias=True)
        self.bn = bn_with_initialize(planes)
        self.relu = P.ReLU()

        self.layers = []

        for _ in range(0, blocks):
            self.layers.append(block(planes))
        self.layers = nn.CellList(self.layers)

    def construct(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        for block in self.layers:
            x = block(x)
        return x

class SphereNet(Cell):
    '''SphereNet'''
    def __init__(self, num_layers=36, feature_dim=128, shape=(96, 64)):
        super(SphereNet, self).__init__()
        assert num_layers in [12, 20, 36, 64], 'SphereNet num_layers should be 12, 20 or 64'
        if num_layers == 12:
            layers = [1, 1, 1, 1]
            filter_list = [3, 16, 32, 64, 128]
            fc_size = 128 * 6 * 4
        elif num_layers == 20:
            layers = [1, 2, 4, 1]
            filter_list = [3, 64, 128, 256, 512]
            fc_size = 512 * 6 * 4
        elif num_layers == 36:
            layers = [2, 4, 4, 2]
            filter_list = [3, 32, 64, 128, 256]
            fc_size = 256 * 6 * 4
        elif num_layers == 64:
            layers = [3, 7, 16, 3]
            filter_list = [3, 64, 128, 256, 512]
            fc_size = 512 * 6 * 4
        else:
            raise ValueError('sphere' + str(num_layers) + " IS NOT SUPPORTED! (sphere20 or sphere64)")
        self.shape = P.Shape()
        self.reshape = P.Reshape()
        self.arg_shape = shape
        block = BaseBlock

        self.layer1 = MakeLayer(block, filter_list[0], filter_list[1], layers[0], stride=2)
        self.layer2 = MakeLayer(block, filter_list[1], filter_list[2], layers[1], stride=2)
        self.layer3 = MakeLayer(block, filter_list[2], filter_list[3], layers[2], stride=2)
        self.layer4 = MakeLayer(block, filter_list[3], filter_list[4], layers[3], stride=2)

        self.fc = fc_with_initialize(fc_size, feature_dim)
        self.last_bn = nn.BatchNorm1d(feature_dim, momentum=0.9).add_flags_recursive(fp32=True)
        self.cast = P.Cast()
        self.l2norm = P.L2Normalize(axis=1)

        for _, cell in self.cells_and_names():
            if isinstance(cell, (nn.Conv2d, nn.Dense)):
                if cell.bias is not None:
                    cell.weight.set_data(initializer(me_init.ReidKaimingUniform(a=math.sqrt(5), mode='fan_out'),
                                                     cell.weight.shape))
                    cell.bias.set_data(initializer('zeros', cell.bias.shape))
                else:
                    cell.weight.set_data(initializer(me_init.ReidXavierUniform(), cell.weight.shape))

    def construct(self, x):
        '''Construct function.'''
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        b, _, _, _ = self.shape(x)
        x = self.reshape(x, (b, -1))
        x = self.fc(x)
        x = self.last_bn(x)
        x = self.cast(x, mstype.float16)
        x = self.l2norm(x)
        x = self.cast(x, mstype.float32)

        return x


class CombineMarginFC(nn.Cell):
    '''CombineMarginFC'''
    def __init__(self, embbeding_size=128, classnum=270762, s=32, a=1.0, m=0.3, b=0.2):
        super(CombineMarginFC, self).__init__()
        weight_shape = [classnum, embbeding_size]
        weight_init = initializer(me_init.ReidXavierUniform(), weight_shape)
        self.weight = Parameter(weight_init, name='weight')
        self.m = m
        self.s = s
        self.a = a
        self.b = b
        self.m_const = Tensor(self.m, dtype=mstype.float32)
        self.a_const = Tensor(self.a, dtype=mstype.float32)
        self.b_const = Tensor(self.b, dtype=mstype.float32)
        self.s_const = Tensor(self.s, dtype=mstype.float32)
        self.m_const_zero = Tensor(0.0, dtype=mstype.float32)
        self.a_const_one = Tensor(1.0, dtype=mstype.float32)
        self.normalize = P.L2Normalize(axis=1)
        self.fc = P.MatMul(transpose_b=True)
        self.onehot = P.OneHot()
        self.transpose = P.Transpose()
        self.acos = P.ACos()
        self.cos = P.Cos()
        self.cast = P.Cast()
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)

    def construct(self, x, label):
        '''Construct function.'''
        w = self.normalize(self.weight)
        cosine = self.fc(self.cast(x, mstype.float16), self.cast(w, mstype.float16))
        cosine = self.cast(cosine, mstype.float32)
        cosine_shape = F.shape(cosine)

        one_hot_float = self.onehot(
            self.cast(label, mstype.int32), cosine_shape[1], self.on_value, self.off_value)
        theta = self.acos(cosine)
        theta = self.a_const * theta
        theta = self.m_const + theta
        body = self.cos(theta)
        body = body - self.b_const
        cos_mask = F.scalar_to_array(1.0) - one_hot_float
        output = body * one_hot_float + cosine * cos_mask
        output = output * self.s_const
        return output, cosine


class CombineMarginFCFp16(nn.Cell):
    '''CombineMarginFCFp16'''
    def __init__(self, embbeding_size=128, classnum=270762, s=32, a=1.0, m=0.3, b=0.2):
        super(CombineMarginFCFp16, self).__init__()
        weight_shape = [classnum, embbeding_size]
        weight_init = initializer(me_init.ReidXavierUniform(), weight_shape)
        self.weight = Parameter(weight_init, name='weight')

        self.m = m
        self.s = s
        self.a = a
        self.b = b
        self.m_const = Tensor(self.m, dtype=mstype.float16)
        self.a_const = Tensor(self.a, dtype=mstype.float16)
        self.b_const = Tensor(self.b, dtype=mstype.float16)
        self.s_const = Tensor(self.s, dtype=mstype.float16)
        self.m_const_zero = Tensor(0, dtype=mstype.float16)
        self.a_const_one = Tensor(1, dtype=mstype.float16)
        self.normalize = P.L2Normalize(axis=1)
        self.fc = P.MatMul(transpose_b=True)

        self.onehot = P.OneHot()
        self.transpose = P.Transpose()
        self.acos = P.ACos()
        self.cos = P.Cos()
        self.cast = P.Cast()
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)

    def construct(self, x, label):
        '''Construct function.'''
        w = self.normalize(self.weight)
        cosine = self.fc(x, w)
        cosine_shape = F.shape(cosine)

        one_hot_float = self.onehot(
            self.cast(label, mstype.int32), cosine_shape[1], self.on_value, self.off_value)
        one_hot_float = self.cast(one_hot_float, mstype.float16)
        theta = self.acos(cosine)
        theta = self.a_const * theta
        theta = self.m_const + theta
        body = self.cos(theta)
        body = body - self.b_const
        cos_mask = self.cast(F.scalar_to_array(1.0), mstype.float16) - one_hot_float
        output = body * one_hot_float + cosine * cos_mask
        output = output * self.s_const

        return output, cosine


class BuildTrainNetwork(Cell):
    def __init__(self, network, criterion):
        super(BuildTrainNetwork, self).__init__()
        self.network = network
        self.criterion = criterion

    def construct(self, input_data, label):
        output = self.network(input_data)
        loss = self.criterion(output, label)
        return loss


class BuildTrainNetworkWithHead(nn.Cell):
    '''Build TrainNetwork With Head.'''
    def __init__(self, model, head, criterion):
        super(BuildTrainNetworkWithHead, self).__init__()
        self.model = model
        self.head = head
        self.criterion = criterion

    def construct(self, input_data, labels):
        embeddings = self.model(input_data)
        thetas, _ = self.head(embeddings, labels)
        loss = self.criterion(thetas, labels)

        return loss
