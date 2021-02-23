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

import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from .mobilenet_v1 import conv_bn_relu, MobileNetV1
from .resnet import resnet50


class FpnTopDown(nn.Cell):
    """
    Fpn to extract features
    """
    def __init__(self, in_channel_list, out_channels):
        super(FpnTopDown, self).__init__()
        self.lateral_convs_list_ = []
        self.fpn_convs_ = []
        for channel in in_channel_list:
            l_conv = nn.Conv2d(channel, out_channels, kernel_size=1, stride=1,
                               has_bias=True, padding=0, pad_mode='same')
            fpn_conv = conv_bn_relu(out_channels, out_channels, kernel_size=3, stride=1, depthwise=False)
            self.lateral_convs_list_.append(l_conv)
            self.fpn_convs_.append(fpn_conv)
        self.lateral_convs_list = nn.layer.CellList(self.lateral_convs_list_)
        self.fpn_convs_list = nn.layer.CellList(self.fpn_convs_)
        self.num_layers = len(in_channel_list)

    def construct(self, inputs):
        image_features = ()
        for i, feature in enumerate(inputs):
            image_features = image_features + (self.lateral_convs_list[i](feature),)

        features = (image_features[-1],)
        for i in range(len(inputs) - 1):
            top = len(inputs) - i - 1
            down = top - 1
            size = F.shape(inputs[down])
            top_down = P.ResizeBilinear((size[2], size[3]))(features[-1])
            top_down = top_down + image_features[down]
            features = features + (top_down,)

        extract_features = ()
        num_features = len(features)
        for i in range(num_features):
            extract_features = extract_features + (self.fpn_convs_list[i](features[num_features - i - 1]),)

        return extract_features


class BottomUp(nn.Cell):
    """
    Bottom Up feature extractor
    """
    def __init__(self, levels, channels, kernel_size, stride):
        super(BottomUp, self).__init__()
        self.levels = levels
        bottom_up_cells = [
            conv_bn_relu(channels, channels, kernel_size, stride, False) for x in range(self.levels)
        ]
        self.blocks = nn.CellList(bottom_up_cells)

    def construct(self, features):
        for block in self.blocks:
            features = features + (block(features[-1]),)
        return features


class FeatureSelector(nn.Cell):
    """
    Select specific layers from an entire feature list
    """
    def __init__(self, feature_idxes):
        super(FeatureSelector, self).__init__()
        self.feature_idxes = feature_idxes

    def construct(self, feature_list):
        selected = ()
        for i in self.feature_idxes:
            selected = selected + (feature_list[i],)
        return selected


class MobileNetV1Fpn(nn.Cell):
    """
    MobileNetV1 with FPN as SSD backbone.
    """
    def __init__(self, config):
        super(MobileNetV1Fpn, self).__init__()
        self.mobilenet_v1 = MobileNetV1(features_only=True)

        self.selector = FeatureSelector([10, 22, 26])

        self.layer_indexs = [10, 22, 26]
        self.fpn = FpnTopDown([256, 512, 1024], 256)
        self.bottom_up = BottomUp(2, 256, 3, 2)

    def construct(self, x):
        features = self.mobilenet_v1(x)
        features = self.selector(features)
        features = self.fpn(features)
        features = self.bottom_up(features)
        return features

class ResNetV1Fpn(nn.Cell):
    """
    ResNet with FPN as SSD backbone.
    """
    def __init__(self, resnet):
        super(ResNetV1Fpn, self).__init__()
        self.resnet = resnet
        self.fpn = FpnTopDown([512, 1024, 2048], 256)
        self.bottom_up = BottomUp(2, 256, 3, 2)

    def construct(self, x):
        _, _, c3, c4, c5 = self.resnet(x)
        features = self.fpn((c3, c4, c5))
        features = self.bottom_up(features)
        return features


def mobilenet_v1_fpn(config):
    return MobileNetV1Fpn(config)

def resnet50_fpn():
    resnet = resnet50()
    return ResNetV1Fpn(resnet)
