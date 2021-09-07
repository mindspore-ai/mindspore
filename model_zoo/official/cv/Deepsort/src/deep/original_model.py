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
import mindspore
import mindspore.nn as nn
import mindspore.ops as F

class BasicBlock(nn.Cell):
    def __init__(self, c_in, c_out, is_downsample=False):
        super(BasicBlock, self).__init__()
        self.add = mindspore.ops.Add()
        self.ReLU = F.ReLU()
        self.is_downsample = is_downsample
        if is_downsample:
            self.conv1 = nn.Conv2d(c_in, c_out, 3, stride=2, pad_mode='pad', padding=1,\
                 has_bias=False, weight_init='HeUniform')
        else:
            self.conv1 = nn.Conv2d(c_in, c_out, 3, stride=1, pad_mode='same',\
                 has_bias=False, weight_init='HeUniform')
        self.bn1 = nn.BatchNorm2d(c_out, momentum=0.9)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(c_out, c_out, 3, stride=1, pad_mode='pad', padding=1,\
             has_bias=False, weight_init='HeUniform')
        self.bn2 = nn.BatchNorm2d(c_out, momentum=0.9)
        if is_downsample:
            self.downsample = nn.SequentialCell(
                [nn.Conv2d(c_in, c_out, 1, stride=2, pad_mode='same', has_bias=False, weight_init='HeUniform'),
                 nn.BatchNorm2d(c_out, momentum=0.9)]
            )
        elif c_in != c_out:
            self.downsample = nn.SequentialCell(
                [nn.Conv2d(c_in, c_out, 1, stride=1, pad_mode='pad', has_bias=False, weight_init='HeUniform'),
                 nn.BatchNorm2d(c_out, momentum=0.9)]
            )
            self.is_downsample = True
    def construct(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.bn2(y)
        if self.is_downsample:
            x = self.downsample(x)
        y = self.add(x, y)
        y = self.ReLU(y)
        return y

def make_layers(c_in, c_out, repeat_times, is_downsample=False):
    blocks = []
    for i in range(repeat_times):
        if i == 0:
            blocks.append(BasicBlock(c_in, c_out, is_downsample=is_downsample))
        else:
            blocks.append(BasicBlock(c_out, c_out))
    return nn.SequentialCell(blocks)


class Net(nn.Cell):
    def __init__(self, num_classes=751, reid=False, ascend=False):
        super(Net, self).__init__()
        # 3 128 64
        self.conv = nn.SequentialCell(
            [nn.Conv2d(3, 32, 3, stride=1, pad_mode='same', has_bias=True, weight_init='HeUniform'),
             nn.BatchNorm2d(32, momentum=0.9),
             nn.ELU(),
             nn.Conv2d(32, 32, 3, stride=1, pad_mode='same', has_bias=True, weight_init='HeUniform'),
             nn.BatchNorm2d(32, momentum=0.9),
             nn.ELU(),
             nn.MaxPool2d(3, 2, pad_mode='same')]
        )
        #]
        # 32 64 32
        self.layer1 = make_layers(32, 32, 2, False)
        # 32 64 32
        self.layer2 = make_layers(32, 64, 2, True)
        # 64 32 16
        self.layer3 = make_layers(64, 128, 2, True)
        # 128 16 8
        self.dp = nn.Dropout(keep_prob=0.6)
        self.dense = nn.Dense(128*16*8, 128)
        self.bn1 = nn.BatchNorm1d(128, momentum=0.9)
        self.elu = nn.ELU()
        # 256 1 1
        self.reid = reid
        self.ascend = ascend
        #self.flatten = nn.Flatten()
        self.div = F.Div()
        self.batch_norm = nn.BatchNorm1d(128, momentum=0.9)
        self.classifier = nn.Dense(128, num_classes)
        self.Norm = nn.Norm(axis=0, keep_dims=True)

    def construct(self, x):
        x = self.conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        #x = self.flatten(x)
        x = x.view((x.shape[0], -1))
        if self.reid:
            x = self.dp(x)
            x = self.dense(x)
            if self.ascend:
                x = self.bn1(x)
            else:
                f = self.Norm(x)
                x = self.div(x, f)
            return x
        x = self.dp(x)
        x = self.dense(x)
        x = self.bn1(x)
        x = self.elu(x)
        x = self.classifier(x)
        return x
