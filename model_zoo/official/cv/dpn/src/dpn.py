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
from collections import OrderedDict
import mindspore.nn as nn
import mindspore.ops.operations as F

__all__ = ['DPN', 'dpn92', 'dpn98', 'dpn131', 'dpn107', 'dpns']


def dpn92(num_classes=1000):
    return DPN(num_init_features=64, k_R=96, G=32, k_sec=(3, 4, 20, 3), inc_sec=(16, 32, 24, 128),
               num_classes=num_classes)


def dpn98(num_classes=1000):
    return DPN(num_init_features=96, k_R=160, G=40, k_sec=(3, 6, 20, 3), inc_sec=(16, 32, 32, 128),
               num_classes=num_classes)


def dpn131(num_classes=1000):
    return DPN(num_init_features=128, k_R=160, G=40, k_sec=(4, 8, 28, 3), inc_sec=(16, 32, 32, 128),
               num_classes=num_classes)


def dpn107(num_classes=1000):
    return DPN(num_init_features=128, k_R=200, G=50, k_sec=(4, 8, 20, 3), inc_sec=(20, 64, 64, 128),
               num_classes=num_classes)


dpns = {
    'dpn92': dpn92,
    'dpn98': dpn98,
    'dpn107': dpn107,
    'dpn131': dpn131,
}


class BottleBlock(nn.Cell):
    def __init__(self, in_chs, num_1x1_a, num_3x3_b, num_1x1_c, inc, G, key_stride):
        super(BottleBlock, self).__init__()
        self.G = G
        self.bn1 = nn.BatchNorm2d(in_chs, eps=1e-3, momentum=0.9)
        self.conv1 = nn.Conv2d(in_chs, num_1x1_a, 1, stride=1)
        self.bn2 = nn.BatchNorm2d(num_1x1_a, eps=1e-3, momentum=0.9)
        self.conv2 = nn.CellList()
        for _ in range(G):
            self.conv2.append(nn.Conv2d(num_1x1_a // G, num_3x3_b // G, 3, key_stride, pad_mode='pad', padding=1))
        self.bn3 = nn.BatchNorm2d(num_3x3_b, eps=1e-3, momentum=0.9)
        self.conv3_r = nn.Conv2d(num_3x3_b, num_1x1_c, 1, stride=1)
        self.conv3_d = nn.Conv2d(num_3x3_b, inc, 1, stride=1)

        self.relu = nn.ReLU()
        self.concat = F.Concat(axis=1)
        self.split = F.Split(axis=1, output_num=G)

    def construct(self, x):
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.relu(x)
        group_x = ()
        input_x = self.split(x)
        for i in range(self.G):
            group_x = group_x + (self.conv2[i](input_x[i]),)
        x = self.concat(group_x)
        x = self.bn3(x)
        x = self.relu(x)
        return (self.conv3_r(x), self.conv3_d(x))


class DualPathBlock(nn.Cell):
    def __init__(self, in_chs, num_1x1_a, num_3x3_b, num_1x1_c, inc, G, _type='normal', cat_input=True):
        super(DualPathBlock, self).__init__()
        self.num_1x1_c = num_1x1_c

        if _type == 'proj':
            key_stride = 1
            self.has_proj = True
        if _type == 'down':
            key_stride = 2
            self.has_proj = True
        if _type == 'normal':
            key_stride = 1
            self.has_proj = False

        self.cat_input = cat_input

        if self.has_proj:
            self.c1x1_w_bn = nn.BatchNorm2d(in_chs, eps=1e-3, momentum=0.9)
            self.c1x1_w_relu = nn.ReLU()
            self.c1x1_w_r = self.Conv1x1(in_chs=in_chs, out_chs=num_1x1_c, stride=key_stride)
            self.c1x1_w_d = self.Conv1x1(in_chs=in_chs, out_chs=2 * inc, stride=key_stride)

        self.layers = BottleBlock(in_chs, num_1x1_a, num_3x3_b, num_1x1_c, inc, G, key_stride)
        self.concat = F.Concat(axis=1)
        self.add = F.TensorAdd()

    def Conv1x1(self, in_chs, out_chs, stride):
        return nn.Conv2d(in_chs, out_chs, kernel_size=1, stride=stride, pad_mode='pad', padding=0)

    def construct(self, x):
        if self.cat_input:
            data_in = self.concat(x)
        else:
            data_in = x

        if self.has_proj:
            data_o = self.c1x1_w_bn(data_in)
            data_o = self.c1x1_w_relu(data_o)
            data_o1 = self.c1x1_w_r(data_o)
            data_o2 = self.c1x1_w_d(data_o)
        else:
            data_o1 = x[0]
            data_o2 = x[1]

        out = self.layers(data_in)
        summ = self.add(data_o1, out[0])
        dense = self.concat((data_o2, out[1]))
        return (summ, dense)


class DPN(nn.Cell):

    def __init__(self, num_init_features=64, k_R=96, G=32,
                 k_sec=(3, 4, 20, 3), inc_sec=(16, 32, 24, 128), num_classes=1000):

        super(DPN, self).__init__()
        blocks = OrderedDict()

        # conv1
        blocks['conv1'] = nn.SequentialCell(OrderedDict([
            ('conv', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, pad_mode='pad', padding=3)),
            ('norm', nn.BatchNorm2d(num_init_features, eps=1e-3, momentum=0.9)),
            ('relu', nn.ReLU()),
            ('maxpool', nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same')),
        ]))

        # conv2
        bw = 256
        inc = inc_sec[0]
        R = int((k_R * bw) / 256)
        blocks['conv2_1'] = DualPathBlock(num_init_features, R, R, bw, inc, G, 'proj', False)
        in_chs = bw + 3 * inc
        for i in range(2, k_sec[0] + 1):
            blocks['conv2_{}'.format(i)] = DualPathBlock(in_chs, R, R, bw, inc, G, 'normal')
            in_chs += inc

        # conv3
        bw = 512
        inc = inc_sec[1]
        R = int((k_R * bw) / 256)
        blocks['conv3_1'] = DualPathBlock(in_chs, R, R, bw, inc, G, 'down')
        in_chs = bw + 3 * inc
        for i in range(2, k_sec[1] + 1):
            blocks['conv3_{}'.format(i)] = DualPathBlock(in_chs, R, R, bw, inc, G, 'normal')
            in_chs += inc

        # conv4
        bw = 1024
        inc = inc_sec[2]
        R = int((k_R * bw) / 256)
        blocks['conv4_1'] = DualPathBlock(in_chs, R, R, bw, inc, G, 'down')
        in_chs = bw + 3 * inc
        for i in range(2, k_sec[2] + 1):
            blocks['conv4_{}'.format(i)] = DualPathBlock(in_chs, R, R, bw, inc, G, 'normal')
            in_chs += inc

        # conv5
        bw = 2048
        inc = inc_sec[3]
        R = int((k_R * bw) / 256)
        blocks['conv5_1'] = DualPathBlock(in_chs, R, R, bw, inc, G, 'down')
        in_chs = bw + 3 * inc
        for i in range(2, k_sec[3] + 1):
            blocks['conv5_{}'.format(i)] = DualPathBlock(in_chs, R, R, bw, inc, G, 'normal')
            in_chs += inc

        self.features = nn.SequentialCell(blocks)
        self.concat = F.Concat(axis=1)
        self.conv5_x = nn.SequentialCell(OrderedDict([
            ('norm', nn.BatchNorm2d(in_chs, eps=1e-3, momentum=0.9)),
            ('relu', nn.ReLU()),
        ]))
        self.avgpool = F.ReduceMean(False)
        self.classifier = nn.Dense(in_chs, num_classes)

    def construct(self, x):
        x = self.features(x)
        x = self.concat(x)
        x = self.conv5_x(x)
        x = self.avgpool(x, (2, 3))
        x = self.classifier(x)
        return x
