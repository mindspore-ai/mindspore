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
"""Xception."""
import mindspore.nn as nn
import mindspore.ops.operations as P

class SeparableConv2d(nn.Cell):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0):
        super(SeparableConv2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, group=in_channels, pad_mode='pad',
                               padding=padding, weight_init='xavier_uniform')
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, pad_mode='valid',
                                   weight_init='xavier_uniform')

    def construct(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Cell):
    def __init__(self, in_filters, out_filters, reps, strides=1, start_with_relu=True, grow_first=True):
        super(Block, self).__init__()

        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv2d(in_filters, out_filters, 1, stride=strides, pad_mode='valid', has_bias=False,
                                  weight_init='xavier_uniform')
            self.skipbn = nn.BatchNorm2d(out_filters, momentum=0.9)
        else:
            self.skip = None

        self.relu = nn.ReLU()
        rep = []
        filters = in_filters
        if grow_first:
            rep.append(nn.ReLU())
            rep.append(SeparableConv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1))
            rep.append(nn.BatchNorm2d(out_filters, momentum=0.9))
            filters = out_filters

        for _ in range(reps - 1):
            rep.append(nn.ReLU())
            rep.append(SeparableConv2d(filters, filters, kernel_size=3, stride=1, padding=1))
            rep.append(nn.BatchNorm2d(filters, momentum=0.9))

        if not grow_first:
            rep.append(nn.ReLU())
            rep.append(SeparableConv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1))
            rep.append(nn.BatchNorm2d(out_filters, momentum=0.9))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU()

        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, pad_mode="same"))
        self.rep = nn.SequentialCell(*rep)
        self.add = P.Add()

    def construct(self, inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp
        x = self.add(x, skip)
        return x


class Xception(nn.Cell):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/abs/1610.02357.pdf
    """

    def __init__(self, num_classes=1000):
        """ Constructor
        Args:
            num_classes: number of classes.
        """
        super(Xception, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3, 32, 3, 2, pad_mode='valid', weight_init='xavier_uniform')
        self.bn1 = nn.BatchNorm2d(32, momentum=0.9)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, 3, pad_mode='valid', weight_init='xavier_uniform')
        self.bn2 = nn.BatchNorm2d(64, momentum=0.9)

        # Entry flow
        self.block1 = Block(64, 128, 2, 2, start_with_relu=False, grow_first=True)
        self.block2 = Block(128, 256, 2, 2, start_with_relu=True, grow_first=True)
        self.block3 = Block(256, 728, 2, 2, start_with_relu=True, grow_first=True)

        # Middle flow
        self.block4 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block5 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block6 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block7 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)

        self.block8 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block9 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block10 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block11 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)

        # Exit flow
        self.block12 = Block(728, 1024, 2, 2, start_with_relu=True, grow_first=False)
        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(1536, momentum=0.9)

        self.conv4 = SeparableConv2d(1536, 2048, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(2048, momentum=0.9)

        self.avg_pool = nn.AvgPool2d(10)
        self.dropout = nn.Dropout()
        self.fc = nn.Dense(2048, num_classes)

    def construct(self, x):
        shape = P.Shape()
        reshape = P.Reshape()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.avg_pool(x)
        x = self.dropout(x)

        x = reshape(x, (shape(x)[0], -1))
        x = self.fc(x)

        return x


def xception(class_num=1000):
    """
    Get Xception neural network.

    Args:
        class_num (int): Class number.

    Returns:
        Cell, cell instance of Xception neural network.

    Examples:
        >>> net = xception(1000)
    """
    return Xception(class_num)
