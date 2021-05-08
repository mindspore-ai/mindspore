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
"""resnet extractor"""
import mindspore.nn as nn
from .resnet import resnet50

def conv_bn_relu(in_channel, out_channel, kernel_size, stride, depthwise, activation='relu6'):
    output = []
    output.append(nn.Conv2d(in_channel, out_channel, kernel_size, stride, pad_mode="same",
                            group=1 if not depthwise else in_channel))
    output.append(nn.BatchNorm2d(out_channel))
    if activation:
        output.append(nn.get_activation(activation))
    return nn.SequentialCell(output)

class ExtraLayer(nn.Cell):
    """
    extra feature extractor
    """
    def __init__(self, levels, res_channels, channels, kernel_size, stride):
        super(ExtraLayer, self).__init__()
        self.levels = levels
        self.Channel_cover = conv_bn_relu(512, channels, kernel_size, 1, False)
        bottom_up_cells = [
            conv_bn_relu(channels, channels, kernel_size, stride, False) for x in range(self.levels)
        ]
        self.blocks = nn.CellList(bottom_up_cells)

    def construct(self, features):
        """
        Forward
        """
        mid_feature = self.Channel_cover(features[-1])
        features = features + (self.blocks[0](mid_feature),)
        features = features + (self.blocks[1](features[-1]),)
        return features


class resnet50_extra(nn.Cell):
    """
    ResNet with extra feature.
    """
    def __init__(self):
        super(resnet50_extra, self).__init__()
        self.resnet = resnet50()
        self.extra = ExtraLayer(2, 512, 256, 3, 2)
        self.Channel_cover = conv_bn_relu(2048, 512, 3, 1, False)

    def construct(self, x):
        """
        Forward
        """
        _, _, c3, c4, c5 = self.resnet(x)
        c5 = self.Channel_cover(c5)
        features = self.extra((c3, c4, c5))
        return features
