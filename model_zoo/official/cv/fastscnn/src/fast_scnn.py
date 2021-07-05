# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Fast Segmentation Convolutional Neural Network"""
import mindspore.nn as nn
from mindspore.ops import operations as P
from src.loss import MixSoftmaxCrossEntropyLoss

__all__ = ['FastSCNN', 'FastSCNNWithLossCell']

class FastSCNN(nn.Cell):
    '''FastSCNN'''
    def __init__(self, num_classes, aux=False):
        super(FastSCNN, self).__init__()
        self.aux = aux
        self.learning_to_downsample = LearningToDownsample(32, 48, 64)
        self.global_feature_extractor = GlobalFeatureExtractor(64, \
                                      [64, 96, 128], 128, 6, [3, 3, 3])
        self.feature_fusion = FeatureFusionModule(64, 128, 128)
        self.classifier = Classifier(128, num_classes)
        if self.aux:
            self.auxlayer1 = nn.SequentialCell(
                [nn.Conv2d(64, 32, 3, pad_mode='pad', padding=1, has_bias=False),
                 nn.BatchNorm2d(32),
                 nn.ReLU(),
                 nn.Dropout(0.9),#1-0.9=0.1
                 nn.Conv2d(32, num_classes, 1, has_bias=True)]
                )
            self.auxlayer2 = nn.SequentialCell(
                [nn.Conv2d(128, 32, 3, pad_mode='pad', padding=1, has_bias=False),
                 nn.BatchNorm2d(32),
                 nn.ReLU(),
                 nn.Dropout(0.9),#1-0.9=0.1
                 nn.Conv2d(32, num_classes, 1, has_bias=True)]
                )
        self.ResizeBilinear = nn.ResizeBilinear()
    def construct(self, x):
        '''construct'''
        size = x.shape[2:]
        higher_res_features = self.learning_to_downsample(x)
        lower_res_features = self.global_feature_extractor(higher_res_features)
        x = self.feature_fusion(higher_res_features, lower_res_features)
        x = self.classifier(x)

        x = self.ResizeBilinear(x, size, align_corners=True)
        if self.aux:
            auxout = self.auxlayer1(higher_res_features)
            auxout = self.ResizeBilinear(auxout, size, align_corners=True)
            auxout2 = self.auxlayer2(lower_res_features)
            auxout2 = self.ResizeBilinear(auxout2, size, align_corners=True)
            return x, auxout, auxout2
        return x

class _ConvBNReLU(nn.Cell):
    """Conv-BN-ReLU"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        super(_ConvBNReLU, self).__init__()
        self.conv = nn.SequentialCell(
            [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, \
                       kernel_size=kernel_size, stride=stride, padding=padding, has_bias=False),
             nn.BatchNorm2d(out_channels),
             nn.ReLU()]
        )
    def construct(self, x):
        return self.conv(x)

class _DSConv(nn.Cell):
    """Depthwise Separable Convolutions"""
    def __init__(self, dw_channels, out_channels, stride=1):
        super(_DSConv, self).__init__()
        self.conv = nn.SequentialCell(
            [nn.Conv2d(in_channels=dw_channels, out_channels=dw_channels, \
                                   kernel_size=3, stride=stride, pad_mode="pad", \
                                   padding=1, group=dw_channels, has_bias=False),
             nn.BatchNorm2d(dw_channels),
             nn.ReLU(),
             nn.Conv2d(in_channels=dw_channels, out_channels=out_channels, \
                      kernel_size=1, has_bias=False),
             nn.BatchNorm2d(out_channels),
             nn.ReLU()]
        )
    def construct(self, x):
        return self.conv(x)

class _DWConv(nn.Cell):
    '''_DWConv'''
    def __init__(self, dw_channels, out_channels, stride=1):
        super(_DWConv, self).__init__()
        self.conv = nn.SequentialCell(
            [nn.Conv2d(in_channels=dw_channels, out_channels=out_channels, \
                      kernel_size=3, stride=stride, pad_mode="pad", padding=1, \
                      group=dw_channels, has_bias=False),
             nn.BatchNorm2d(out_channels),
             nn.ReLU()]
        )
    def construct(self, x):
        return self.conv(x)

class LinearBottleneck(nn.Cell):
    """LinearBottleneck used in MobileNetV2"""
    def __init__(self, in_channels, out_channels, t=6, stride=2):
        super(LinearBottleneck, self).__init__()
        self.use_shortcut = (stride == 1) and (in_channels == out_channels)
        self.block = nn.SequentialCell([
            # pw
            _ConvBNReLU(in_channels, in_channels * t, 1),
            # dw
            _DWConv(in_channels * t, in_channels * t, stride),
            # pw-linear
            nn.Conv2d(in_channels * t, out_channels, 1, has_bias=False),
            nn.BatchNorm2d(out_channels)])

    def construct(self, x):
        out = self.block(x)
        if self.use_shortcut:
            out = x + out
        return out

class PyramidPooling(nn.Cell):
    """Pyramid pooling module"""

    def __init__(self, in_channels, out_channels):
        super(PyramidPooling, self).__init__()
        inter_channels = int(in_channels / 4)
        self.conv1 = _ConvBNReLU(in_channels, inter_channels, 1)
        self.conv2 = _ConvBNReLU(in_channels, inter_channels, 1)
        self.conv3 = _ConvBNReLU(in_channels, inter_channels, 1)
        self.conv4 = _ConvBNReLU(in_channels, inter_channels, 1)
        self.out = _ConvBNReLU(in_channels * 2, out_channels, 1)
        self.concat = P.Concat(axis=1)
        self.pool24 = nn.AvgPool2d(kernel_size=24, stride=24)
        self.pool12 = nn.AvgPool2d(kernel_size=12, stride=12)
        self.pool8 = nn.AvgPool2d(kernel_size=8, stride=8)
        self.pool6 = nn.AvgPool2d(kernel_size=4, stride=4)
        self.resizeBilinear = nn.ResizeBilinear()

    def _AdaptiveAvgPool2d(self, x, output_size):
        #NCHW,  for NCx24x24 and size in (1,2,3,6) only
        if output_size == 1:
            return self.pool24(x)
        if output_size == 2:
            return self.pool12(x)
        if output_size == 3:
            return self.pool8(x)
        return self.pool6(x)

    def upsample(self, x, size):
        return self.resizeBilinear(x, size, align_corners=True)

    def construct(self, x):
        size = x.shape[2:]
        feat1 = self.upsample(self.conv1(self._AdaptiveAvgPool2d(x, 1)), size)
        feat2 = self.upsample(self.conv2(self._AdaptiveAvgPool2d(x, 2)), size)
        feat3 = self.upsample(self.conv3(self._AdaptiveAvgPool2d(x, 3)), size)
        feat4 = self.upsample(self.conv4(self._AdaptiveAvgPool2d(x, 6)), size)
        x = self.concat((x, feat1, feat2, feat3, feat4))
        x = self.out(x)
        return x

class LearningToDownsample(nn.Cell):
    """Learning to downsample module"""

    def __init__(self, dw_channels1=32, dw_channels2=48, out_channels=64):
        super(LearningToDownsample, self).__init__()
        self.conv = _ConvBNReLU(3, dw_channels1, 3, 2)
        self.dsconv1 = _DSConv(dw_channels1, dw_channels2, 2)
        self.dsconv2 = _DSConv(dw_channels2, out_channels, 2)
    def construct(self, x):
        x = self.conv(x)
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        return x

class GlobalFeatureExtractor(nn.Cell):
    """Global feature extractor module"""

    def __init__(self, in_channels=64, block_channels=(64, 96, 128),
                 out_channels=128, t=6, num_blocks=(3, 3, 3)):
        super(GlobalFeatureExtractor, self).__init__()
        self.bottleneck1 = self._make_layer(LinearBottleneck, in_channels, \
                                            block_channels[0], num_blocks[0], t, 2)
        self.bottleneck2 = self._make_layer(LinearBottleneck, block_channels[0], \
                                            block_channels[1], num_blocks[1], t, 2)
        self.bottleneck3 = self._make_layer(LinearBottleneck, block_channels[1], \
                                            block_channels[2], num_blocks[2], t, 1)
        self.ppm = PyramidPooling(block_channels[2], out_channels)

    def _make_layer(self, block, inplanes, planes, blocks, t=6, stride=1):
        layers = []
        layers.append(block(inplanes, planes, t, stride))
        for _ in range(1, blocks):
            layers.append(block(planes, planes, t, 1))
        return nn.SequentialCell(layers)

    def construct(self, x):
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)
        x = self.bottleneck3(x)
        x = self.ppm(x)
        return x

class FeatureFusionModule(nn.Cell):
    """Feature fusion module"""

    def __init__(self, highter_in_channels, lower_in_channels, out_channels, scale_factor=4):
        super(FeatureFusionModule, self).__init__()
        self.scale_factor = scale_factor
        self.dwconv = _DWConv(lower_in_channels, out_channels, 1)
        self.conv_lower_res = nn.SequentialCell([
            nn.Conv2d(out_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels)])
        self.conv_higher_res = nn.SequentialCell([
            nn.Conv2d(highter_in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels)])
        self.relu = nn.ReLU()
        self.ResizeBilinear = nn.ResizeBilinear()
    def construct(self, higher_res_feature, lower_res_feature):
        lower_res_feature = self.ResizeBilinear(lower_res_feature, \
                                                scale_factor=4, align_corners=True)
        lower_res_feature = self.dwconv(lower_res_feature)
        lower_res_feature = self.conv_lower_res(lower_res_feature)
        higher_res_feature = self.conv_higher_res(higher_res_feature)
        out = higher_res_feature + lower_res_feature
        return self.relu(out)

class Classifier(nn.Cell):
    """Classifier"""
    def __init__(self, dw_channels, num_classes, stride=1, **kwargs):
        super(Classifier, self).__init__()
        self.dsconv1 = _DSConv(dw_channels, dw_channels, stride)
        self.dsconv2 = _DSConv(dw_channels, dw_channels, stride)
        self.conv = nn.SequentialCell([
            nn.Dropout(0.9), # 1-0.9=0.1
            nn.Conv2d(dw_channels, num_classes, 1)])

    def construct(self, x):
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        x = self.conv(x)
        return x

class FastSCNNWithLossCell(nn.Cell):
    """FastSCNN loss, MixSoftmaxCrossEntropyLoss."""
    def __init__(self, network, args):
        super(FastSCNNWithLossCell, self).__init__()
        self.network = network
        self.aux = args.aux
        self.loss = MixSoftmaxCrossEntropyLoss(args, aux=args.aux, aux_weight=args.aux_weight)
    def construct(self, images, targets):
        outputs = self.network(images)
        if self.aux:
            return self.loss(outputs[0], outputs[1], outputs[2], targets)
        return self.loss(outputs, targets)
