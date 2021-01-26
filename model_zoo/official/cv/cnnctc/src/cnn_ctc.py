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
"""cnn_ctc network define"""

import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.common.initializer import TruncatedNormal, initializer
import mindspore.common.dtype as mstype

class CNNCTC_Model(nn.Cell):

    def __init__(self, num_class, hidden_size, final_feature_width):
        super(CNNCTC_Model, self).__init__()

        self.num_class = num_class
        self.hidden_size = hidden_size
        self.final_feature_width = final_feature_width

        self.FeatureExtraction = ResNet_FeatureExtractor()
        self.Prediction = nn.Dense(self.hidden_size, self.num_class)

        self.transpose = P.Transpose()
        self.reshape = P.Reshape()

    def construct(self, x):
        x = self.FeatureExtraction(x)
        x = self.transpose(x, (0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]

        x = self.reshape(x, (-1, self.hidden_size))
        x = self.Prediction(x)
        x = self.reshape(x, (-1, self.final_feature_width, self.num_class))

        return x


class WithLossCell(nn.Cell):

    def __init__(self, backbone, loss_fn):
        super(WithLossCell, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self._loss_fn = loss_fn

    def construct(self, img, label_indices, text, sequence_length):
        model_predict = self._backbone(img)
        return self._loss_fn(model_predict, label_indices, text, sequence_length)

    @property
    def backbone_network(self):
        return self._backbone

class ctc_loss(nn.Cell):

    def __init__(self):
        super(ctc_loss, self).__init__()

        self.loss = P.CTCLoss(preprocess_collapse_repeated=False,
                              ctc_merge_repeated=True,
                              ignore_longer_outputs_than_inputs=False)

        self.mean = P.ReduceMean()
        self.transpose = P.Transpose()
        self.reshape = P.Reshape()

    def construct(self, inputs, labels_indices, labels_values, sequence_length):
        inputs = self.transpose(inputs, (1, 0, 2))

        loss, _ = self.loss(inputs, labels_indices, labels_values, sequence_length)

        loss = self.mean(loss)
        return loss


class ResNet_FeatureExtractor(nn.Cell):
    def __init__(self):
        super(ResNet_FeatureExtractor, self).__init__()
        self.ConvNet = ResNet(3, 512, BasicBlock, [1, 2, 5, 3])

    def construct(self, featuremap):
        return self.ConvNet(featuremap)


class ResNet(nn.Cell):
    def __init__(self, input_channel, output_channel, block, layers):
        super(ResNet, self).__init__()

        self.output_channel_block = [int(output_channel / 4), int(output_channel / 2), output_channel, output_channel]

        self.inplanes = int(output_channel / 8)
        self.conv0_1 = ms_conv3x3(input_channel, int(output_channel / 16), stride=1, padding=1, pad_mode='pad')
        self.bn0_1 = ms_fused_bn(int(output_channel / 16))
        self.conv0_2 = ms_conv3x3(int(output_channel / 16), self.inplanes, stride=1, padding=1, pad_mode='pad')
        self.bn0_2 = ms_fused_bn(self.inplanes)
        self.relu = P.ReLU()

        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode='valid')
        self.layer1 = self._make_layer(block, self.output_channel_block[0], layers[0])
        self.conv1 = ms_conv3x3(self.output_channel_block[0], self.output_channel_block[0], stride=1, padding=1,
                                pad_mode='pad')
        self.bn1 = ms_fused_bn(self.output_channel_block[0])

        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode='valid')
        self.layer2 = self._make_layer(block, self.output_channel_block[1], layers[1])
        self.conv2 = ms_conv3x3(self.output_channel_block[1], self.output_channel_block[1], stride=1, padding=1,
                                pad_mode='pad')
        self.bn2 = ms_fused_bn(self.output_channel_block[1])

        self.pad = P.Pad(((0, 0), (0, 0), (0, 0), (2, 2)))
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=(2, 1), pad_mode='valid')
        self.layer3 = self._make_layer(block, self.output_channel_block[2], layers[2])
        self.conv3 = ms_conv3x3(self.output_channel_block[2], self.output_channel_block[2], stride=1, padding=1,
                                pad_mode='pad')
        self.bn3 = ms_fused_bn(self.output_channel_block[2])

        self.layer4 = self._make_layer(block, self.output_channel_block[3], layers[3])
        self.conv4_1 = ms_conv2x2(self.output_channel_block[3], self.output_channel_block[3], stride=(2, 1),
                                  pad_mode='valid')
        self.bn4_1 = ms_fused_bn(self.output_channel_block[3])

        self.conv4_2 = ms_conv2x2(self.output_channel_block[3], self.output_channel_block[3], stride=1, padding=0,
                                  pad_mode='valid')
        self.bn4_2 = ms_fused_bn(self.output_channel_block[3])

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.SequentialCell(
                [ms_conv1x1(self.inplanes, planes * block.expansion, stride=stride),
                 ms_fused_bn(planes * block.expansion)]
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.SequentialCell(layers)

    def construct(self, x):
        x = self.conv0_1(x)
        x = self.bn0_1(x)
        x = self.relu(x)
        x = self.conv0_2(x)
        x = self.bn0_2(x)
        x = self.relu(x)

        x = self.maxpool1(x)
        x = self.layer1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.maxpool2(x)
        x = self.layer2(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.maxpool3(x)
        x = self.layer3(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.layer4(x)
        x = self.pad(x)
        x = self.conv4_1(x)
        x = self.bn4_1(x)
        x = self.relu(x)
        x = self.conv4_2(x)
        x = self.bn4_2(x)
        x = self.relu(x)

        return x


class BasicBlock(nn.Cell):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        self.conv1 = ms_conv3x3(inplanes, planes, stride=stride, padding=1, pad_mode='pad')
        self.bn1 = ms_fused_bn(planes)
        self.conv2 = ms_conv3x3(planes, planes, stride=stride, padding=1, pad_mode='pad')
        self.bn2 = ms_fused_bn(planes)
        self.relu = P.ReLU()
        self.downsample = downsample
        self.add = P.Add()

    def construct(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out = self.add(out, residual)
        out = self.relu(out)

        return out


def weight_variable(shape, factor=0.1, half_precision=False):
    if half_precision:
        return initializer(TruncatedNormal(0.02), shape, dtype=mstype.float16)

    return TruncatedNormal(0.02)


def ms_conv3x3(in_channels, out_channels, stride=1, padding=0, pad_mode='same', has_bias=False):
    """Get a conv2d layer with 3x3 kernel size."""
    init_value = weight_variable((out_channels, in_channels, 3, 3))
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=3, stride=stride, padding=padding, pad_mode=pad_mode, weight_init=init_value,
                     has_bias=has_bias)


def ms_conv1x1(in_channels, out_channels, stride=1, padding=0, pad_mode='same', has_bias=False):
    """Get a conv2d layer with 1x1 kernel size."""
    init_value = weight_variable((out_channels, in_channels, 1, 1))
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=1, stride=stride, padding=padding, pad_mode=pad_mode, weight_init=init_value,
                     has_bias=has_bias)


def ms_conv2x2(in_channels, out_channels, stride=1, padding=0, pad_mode='same', has_bias=False):
    """Get a conv2d layer with 2x2 kernel size."""
    init_value = weight_variable((out_channels, in_channels, 1, 1))
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=2, stride=stride, padding=padding, pad_mode=pad_mode, weight_init=init_value,
                     has_bias=has_bias)


def ms_fused_bn(channels, momentum=0.1):
    """Get a fused batchnorm"""
    return nn.BatchNorm2d(channels, momentum=momentum)
