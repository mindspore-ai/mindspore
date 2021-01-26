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
"""ShuffleNetV1"""
import mindspore.nn as nn
import mindspore.ops.operations as P
from mindspore import dtype as mstype


class GroupConv(nn.Cell):
    """
    group convolution operation.

    Args:
        in_channels (int): Input channels of feature map.
        out_channels (int): Output channels of feature map.
        kernel_size (int): Size of convolution kernel.
        stride (int): Stride size for the group convolution layer.

    Returns:
        tensor, output tensor.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, pad_mode="pad", pad=0, groups=1, has_bias=False):
        super(GroupConv, self).__init__()
        assert in_channels % groups == 0 and out_channels % groups == 0
        self.groups = groups
        self.convs = nn.CellList()
        self.op_split = P.Split(axis=1, output_num=self.groups)
        self.op_concat = P.Concat(axis=1)
        self.cast = P.Cast()
        for _ in range(groups):
            self.convs.append(nn.Conv2d(in_channels // groups, out_channels // groups,
                                        kernel_size=kernel_size, stride=stride, has_bias=has_bias,
                                        padding=pad, pad_mode=pad_mode, group=1, weight_init='xavier_uniform'))

    def construct(self, x):
        features = self.op_split(x)
        outputs = ()
        for i in range(self.groups):
            outputs = outputs + (self.convs[i](self.cast(features[i], mstype.float32)),)
        out = self.op_concat(outputs)
        return out


class ShuffleV1Block(nn.Cell):
    def __init__(self, inp, oup, group, first_group, mid_channels, ksize, stride):
        super(ShuffleV1Block, self).__init__()
        self.stride = stride

        pad = ksize // 2
        self.group = group

        if stride == 2:
            outputs = oup - inp
        else:
            outputs = oup

        self.relu = nn.ReLU()
        self.add = P.Add()
        self.concat = P.Concat(1)
        self.shape = P.Shape()
        self.transpose = P.Transpose()
        self.reshape = P.Reshape()

        branch_main_1 = [
            # pw
            GroupConv(in_channels=inp, out_channels=mid_channels, kernel_size=1, stride=1, pad_mode="pad", pad=0,
                      groups=1 if first_group else group),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
        ]

        branch_main_2 = [
            # dw
            nn.Conv2d(mid_channels, mid_channels, kernel_size=ksize, stride=stride, pad_mode='pad', padding=pad,
                      group=mid_channels, weight_init='xavier_uniform', has_bias=False),
            nn.BatchNorm2d(mid_channels),
            # pw
            GroupConv(in_channels=mid_channels, out_channels=outputs, kernel_size=1, stride=1, pad_mode="pad", pad=0,
                      groups=group),
            nn.BatchNorm2d(outputs),
        ]
        self.branch_main_1 = nn.SequentialCell(branch_main_1)
        self.branch_main_2 = nn.SequentialCell(branch_main_2)
        if stride == 2:
            self.branch_proj = nn.AvgPool2d(kernel_size=3, stride=2, pad_mode='same')

    def construct(self, old_x):
        left = old_x
        right = old_x
        out = old_x
        right = self.branch_main_1(right)
        if self.group > 1:
            right = self.channel_shuffle(right)
        right = self.branch_main_2(right)
        if self.stride == 1:
            out = self.relu(self.add(left, right))
        elif self.stride == 2:
            left = self.branch_proj(left)
            out = self.concat((left, right))
            out = self.relu(out)
        return out

    def channel_shuffle(self, x):
        batchsize, num_channels, height, width = self.shape(x)
        group_channels = num_channels // self.group
        x = self.reshape(x, (batchsize, group_channels, self.group, height, width))
        x = self.transpose(x, (0, 2, 1, 3, 4))
        x = self.reshape(x, (batchsize, num_channels, height, width))
        return x


class ShuffleNetV1(nn.Cell):
    def __init__(self, n_class=1000, model_size='2.0x', group=3):
        super(ShuffleNetV1, self).__init__()
        print('model size is ', model_size)

        self.stage_repeats = [4, 8, 4]
        self.model_size = model_size
        if group == 3:
            if model_size == '0.5x':
                self.stage_out_channels = [-1, 12, 120, 240, 480]
            elif model_size == '1.0x':
                self.stage_out_channels = [-1, 24, 240, 480, 960]
            elif model_size == '1.5x':
                self.stage_out_channels = [-1, 24, 360, 720, 1440]
            elif model_size == '2.0x':
                self.stage_out_channels = [-1, 48, 480, 960, 1920]
            else:
                raise NotImplementedError
        elif group == 8:
            if model_size == '0.5x':
                self.stage_out_channels = [-1, 16, 192, 384, 768]
            elif model_size == '1.0x':
                self.stage_out_channels = [-1, 24, 384, 768, 1536]
            elif model_size == '1.5x':
                self.stage_out_channels = [-1, 24, 576, 1152, 2304]
            elif model_size == '2.0x':
                self.stage_out_channels = [-1, 48, 768, 1536, 3072]
            else:
                raise NotImplementedError

        # building first layer
        input_channel = self.stage_out_channels[1]
        self.first_conv = nn.SequentialCell(
            nn.Conv2d(3, input_channel, 3, 2, 'pad', 1, weight_init='xavier_uniform', has_bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU(),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same')

        features = []
        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage + 2]

            for i in range(numrepeat):
                stride = 2 if i == 0 else 1
                first_group = idxstage == 0 and i == 0
                features.append(ShuffleV1Block(input_channel, output_channel,
                                               group=group, first_group=first_group,
                                               mid_channels=output_channel // 4, ksize=3, stride=stride))
                input_channel = output_channel

        self.features = nn.SequentialCell(features)
        self.globalpool = nn.AvgPool2d(7)
        self.classifier = nn.Dense(self.stage_out_channels[-1], n_class)
        self.reshape = P.Reshape()

    def construct(self, x):
        x = self.first_conv(x)
        x = self.maxpool(x)
        x = self.features(x)
        x = self.globalpool(x)
        x = self.reshape(x, (-1, self.stage_out_channels[-1]))
        x = self.classifier(x)
        return x
