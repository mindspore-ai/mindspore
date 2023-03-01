# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the License);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# httpwww.apache.orglicensesLICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""DeepLabv3."""

import numpy as np
import mindspore.nn as nn
from mindspore.ops import operations as P
from .backbone.resnet_deeplab import _conv_bn_relu, resnet50_dl, _deep_conv_bn_relu, \
    DepthwiseConv2dNative, SpaceToBatch, BatchToSpace


class ASPPSampleBlock(nn.Cell):
    """ASPP sample block."""
    def __init__(self, feature_shape, scale_size, output_stride):
        super(ASPPSampleBlock, self).__init__()
        sample_h = (feature_shape[0] * scale_size + 1) / output_stride + 1
        sample_w = (feature_shape[1] * scale_size + 1) / output_stride + 1
        self.sample = P.ResizeBilinear((int(sample_h), int(sample_w)), align_corners=True)

    def construct(self, x):
        return self.sample(x)


class ASPP(nn.Cell):
    """
    ASPP model for DeepLabv3.

    Args:
        channel (int): Input channel.
        depth (int): Output channel.
        feature_shape (list): The shape of feature,[h,w].
        scale_sizes (list): Input scales for multi-scale feature extraction.
        atrous_rates (list): Atrous rates for atrous spatial pyramid pooling.
        output_stride (int): 'The ratio of input to output spatial resolution.'
        fine_tune_batch_norm (bool): 'Fine tune the batch norm parameters or not'

    Returns:
        Tensor, output tensor.

    Examples:
        >>> ASPP(channel=2048,256,[14,14],[1],[6],16)
    """
    def __init__(self, channel, depth, feature_shape, scale_sizes,
                 atrous_rates, output_stride, fine_tune_batch_norm=False):
        super(ASPP, self).__init__()
        self.aspp0 = _conv_bn_relu(channel,
                                   depth,
                                   ksize=1,
                                   stride=1,
                                   use_batch_statistics=fine_tune_batch_norm)
        self.atrous_rates = []
        if atrous_rates is not None:
            self.atrous_rates = atrous_rates
            self.aspp_pointwise = _conv_bn_relu(channel,
                                                depth,
                                                ksize=1,
                                                stride=1,
                                                use_batch_statistics=fine_tune_batch_norm)
            self.aspp_depth_depthwiseconv = DepthwiseConv2dNative(channel,
                                                                  channel_multiplier=1,
                                                                  kernel_size=3,
                                                                  stride=1,
                                                                  dilation=1,
                                                                  pad_mode="valid")
            self.aspp_depth_bn = nn.BatchNorm2d(1 * channel, use_batch_statistics=fine_tune_batch_norm)
            self.aspp_depth_relu = nn.ReLU()
            self.aspp_depths = []
            self.aspp_depth_spacetobatchs = []
            self.aspp_depth_batchtospaces = []

            for scale_size in scale_sizes:
                aspp_scale_depth_size = np.ceil((feature_shape[0]*scale_size)/16)
                if atrous_rates is None:
                    break
                for rate in atrous_rates:
                    padding = 0
                    for j in range(100):
                        padded_size = rate * j
                        if padded_size >= aspp_scale_depth_size + 2 * rate:
                            padding = padded_size - aspp_scale_depth_size - 2 * rate
                            break
                    paddings = [[rate, rate + int(padding)],
                                [rate, rate + int(padding)]]
                    self.aspp_depth_spacetobatch = SpaceToBatch(rate, paddings)
                    self.aspp_depth_spacetobatchs.append(self.aspp_depth_spacetobatch)
                    crops = [[0, int(padding)], [0, int(padding)]]
                    self.aspp_depth_batchtospace = BatchToSpace(rate, crops)
                    self.aspp_depth_batchtospaces.append(self.aspp_depth_batchtospace)
            self.aspp_depths = nn.CellList(self.aspp_depths)
            self.aspp_depth_spacetobatchs = nn.CellList(self.aspp_depth_spacetobatchs)
            self.aspp_depth_batchtospaces = nn.CellList(self.aspp_depth_batchtospaces)

        self.global_pooling = nn.AvgPool2d(kernel_size=(int(feature_shape[0]), int(feature_shape[1])))
        self.global_poolings = []
        for scale_size in scale_sizes:
            pooling_h = np.ceil((feature_shape[0]*scale_size)/output_stride)
            pooling_w = np.ceil((feature_shape[0]*scale_size)/output_stride)
            self.global_poolings.append(nn.AvgPool2d(kernel_size=(int(pooling_h), int(pooling_w))))
        self.global_poolings = nn.CellList(self.global_poolings)
        self.conv_bn = _conv_bn_relu(channel,
                                     depth,
                                     ksize=1,
                                     stride=1,
                                     use_batch_statistics=fine_tune_batch_norm)
        self.samples = []
        for scale_size in scale_sizes:
            self.samples.append(ASPPSampleBlock(feature_shape, scale_size, output_stride))
        self.samples = nn.CellList(self.samples)
        self.feature_shape = feature_shape
        self.concat = P.Concat(axis=1)

    def construct(self, x, scale_index=0):
        aspp0 = self.aspp0(x)
        aspp1 = self.global_poolings[scale_index](x)
        aspp1 = self.conv_bn(aspp1)
        aspp1 = self.samples[scale_index](aspp1)
        output = self.concat((aspp1, aspp0))

        for i in range(len(self.atrous_rates)):
            aspp_i = self.aspp_depth_spacetobatchs[i + scale_index * len(self.atrous_rates)](x)
            aspp_i = self.aspp_depth_depthwiseconv(aspp_i)
            aspp_i = self.aspp_depth_batchtospaces[i + scale_index * len(self.atrous_rates)](aspp_i)
            aspp_i = self.aspp_depth_bn(aspp_i)
            aspp_i = self.aspp_depth_relu(aspp_i)
            aspp_i = self.aspp_pointwise(aspp_i)
            output = self.concat((output, aspp_i))
        return output


class DecoderSampleBlock(nn.Cell):
    """Decoder sample block."""
    def __init__(self, feature_shape, scale_size=1.0, decoder_output_stride=4):
        super(DecoderSampleBlock, self).__init__()
        sample_h = (feature_shape[0] * scale_size + 1) / decoder_output_stride + 1
        sample_w = (feature_shape[1] * scale_size + 1) / decoder_output_stride + 1
        self.sample = P.ResizeBilinear((int(sample_h), int(sample_w)), align_corners=True)

    def construct(self, x):
        return self.sample(x)


class Decoder(nn.Cell):
    """
    Decode module for DeepLabv3.
    Args:
        low_level_channel (int): Low level input channel
        channel (int): Input channel.
        depth (int): Output channel.
        feature_shape (list): 'Input image shape, [N,C,H,W].'
        scale_sizes (list): 'Input scales for multi-scale feature extraction.'
        decoder_output_stride (int): 'The ratio of input to output spatial resolution'
        fine_tune_batch_norm (bool): 'Fine tune the batch norm parameters or not'
    Returns:
        Tensor, output tensor.
    Examples:
        >>> Decoder(256, 100, [56,56])
    """
    def __init__(self,
                 low_level_channel,
                 channel,
                 depth,
                 feature_shape,
                 scale_sizes,
                 decoder_output_stride,
                 fine_tune_batch_norm):
        super(Decoder, self).__init__()
        self.feature_projection = _conv_bn_relu(low_level_channel, 48, ksize=1, stride=1,
                                                pad_mode="same", use_batch_statistics=fine_tune_batch_norm)
        self.decoder_depth0 = _deep_conv_bn_relu(channel + 48,
                                                 channel_multiplier=1,
                                                 ksize=3,
                                                 stride=1,
                                                 pad_mode="same",
                                                 dilation=1,
                                                 use_batch_statistics=fine_tune_batch_norm)
        self.decoder_pointwise0 = _conv_bn_relu(channel + 48,
                                                depth,
                                                ksize=1,
                                                stride=1,
                                                use_batch_statistics=fine_tune_batch_norm)
        self.decoder_depth1 = _deep_conv_bn_relu(depth,
                                                 channel_multiplier=1,
                                                 ksize=3,
                                                 stride=1,
                                                 pad_mode="same",
                                                 dilation=1,
                                                 use_batch_statistics=fine_tune_batch_norm)
        self.decoder_pointwise1 = _conv_bn_relu(depth,
                                                depth,
                                                ksize=1,
                                                stride=1,
                                                use_batch_statistics=fine_tune_batch_norm)
        self.depth = depth
        self.concat = P.Concat(axis=1)
        self.samples = []
        for scale_size in scale_sizes:
            self.samples.append(DecoderSampleBlock(feature_shape, scale_size, decoder_output_stride))
        self.samples = nn.CellList(self.samples)

    def construct(self, x, low_level_feature, scale_index):
        low_level_feature = self.feature_projection(low_level_feature)
        low_level_feature = self.samples[scale_index](low_level_feature)
        x = self.samples[scale_index](x)
        output = self.concat((x, low_level_feature))
        output = self.decoder_depth0(output)
        output = self.decoder_pointwise0(output)
        output = self.decoder_depth1(output)
        output = self.decoder_pointwise1(output)
        return output


class SingleDeepLabV3(nn.Cell):
    """
    DeepLabv3 Network.
    Args:
        num_classes (int): Class number.
        feature_shape (list): Input image shape, [N,C,H,W].
        backbone (Cell): Backbone Network.
        channel (int): Resnet output channel.
        depth (int): ASPP block depth.
        scale_sizes (list): Input scales for multi-scale feature extraction.
        atrous_rates (list): Atrous rates for atrous spatial pyramid pooling.
        decoder_output_stride (int): 'The ratio of input to output spatial resolution'
        output_stride (int): 'The ratio of input to output spatial resolution.'
        fine_tune_batch_norm (bool): 'Fine tune the batch norm parameters or not'
    Returns:
        Tensor, output tensor.
    Examples:
        >>> SingleDeepLabV3(num_classes=10,
            >>>           feature_shape=[1,3,224,224],
            >>>           backbone=resnet50_dl(),
            >>>           channel=2048,
            >>>           depth=256)
            >>>           scale_sizes=[1.0])
            >>>           atrous_rates=[6])
            >>>           decoder_output_stride=4)
            >>>           output_stride=16)
        """

    def __init__(self,
                 num_classes,
                 feature_shape,
                 backbone,
                 channel,
                 depth,
                 scale_sizes,
                 atrous_rates,
                 decoder_output_stride,
                 output_stride,
                 fine_tune_batch_norm=False):
        super(SingleDeepLabV3, self).__init__()
        self.num_classes = num_classes
        self.channel = channel
        self.depth = depth
        self.scale_sizes = []
        for scale_size in np.sort(scale_sizes):
            self.scale_sizes.append(scale_size)
        self.net = backbone
        self.aspp = ASPP(channel=self.channel,
                         depth=self.depth,
                         feature_shape=[feature_shape[2],
                                        feature_shape[3]],
                         scale_sizes=self.scale_sizes,
                         atrous_rates=atrous_rates,
                         output_stride=output_stride,
                         fine_tune_batch_norm=fine_tune_batch_norm)

        atrous_rates_len = 0
        if atrous_rates is not None:
            atrous_rates_len = len(atrous_rates)
        self.fc1 = _conv_bn_relu(depth * (2 + atrous_rates_len), depth,
                                 ksize=1,
                                 stride=1,
                                 use_batch_statistics=fine_tune_batch_norm)
        self.fc2 = nn.Conv2d(depth,
                             num_classes,
                             kernel_size=1,
                             stride=1,
                             has_bias=True)
        self.upsample = P.ResizeBilinear((int(feature_shape[2]),
                                          int(feature_shape[3])),
                                         align_corners=True)
        self.samples = []
        for scale_size in self.scale_sizes:
            self.samples.append(SampleBlock(feature_shape, scale_size))
        self.samples = nn.CellList(self.samples)
        self.feature_shape = [float(feature_shape[0]), float(feature_shape[1]), float(feature_shape[2]),
                              float(feature_shape[3])]

        self.pad = P.Pad(((0, 0), (0, 0), (1, 1), (1, 1)))
        self.dropout = nn.Dropout(p=0.1)
        self.shape = P.Shape()
        self.decoder_output_stride = decoder_output_stride
        if decoder_output_stride is not None:
            self.decoder = Decoder(low_level_channel=depth,
                                   channel=depth,
                                   depth=depth,
                                   feature_shape=[feature_shape[2],
                                                  feature_shape[3]],
                                   scale_sizes=self.scale_sizes,
                                   decoder_output_stride=decoder_output_stride,
                                   fine_tune_batch_norm=fine_tune_batch_norm)

    def construct(self, x, scale_index=0):
        x = (2.0 / 255.0) * x - 1.0
        x = self.pad(x)
        low_level_feature, feature_map = self.net(x)
        for scale_size in self.scale_sizes:
            if scale_size * self.feature_shape[2] + 1.0 >= self.shape(x)[2] - 2:
                output = self.aspp(feature_map, scale_index)
                output = self.fc1(output)
                if self.decoder_output_stride is not None:
                    output = self.decoder(output, low_level_feature, scale_index)
                output = self.fc2(output)
                output = self.samples[scale_index](output)
                return output
            scale_index += 1
        return feature_map


class SampleBlock(nn.Cell):
    """Sample block."""
    def __init__(self,
                 feature_shape,
                 scale_size=1.0):
        super(SampleBlock, self).__init__()
        sample_h = np.ceil(float(feature_shape[2]) * scale_size)
        sample_w = np.ceil(float(feature_shape[3]) * scale_size)
        self.sample = P.ResizeBilinear((int(sample_h), int(sample_w)), align_corners=True)

    def construct(self, x):
        return self.sample(x)


class DeepLabV3(nn.Cell):
    """DeepLabV3 model."""
    def __init__(self, num_classes, feature_shape, backbone, channel, depth, infer_scale_sizes, atrous_rates,
                 decoder_output_stride, output_stride, fine_tune_batch_norm, image_pyramid):
        super(DeepLabV3, self).__init__()
        self.infer_scale_sizes = []
        if infer_scale_sizes is not None:
            self.infer_scale_sizes = infer_scale_sizes

        self.infer_scale_sizes = infer_scale_sizes
        if image_pyramid is None:
            image_pyramid = [1.0]

        self.image_pyramid = image_pyramid
        scale_sizes = []
        for pyramid in image_pyramid:
            scale_sizes.append(pyramid)
        for scale in infer_scale_sizes:
            scale_sizes.append(scale)
        self.samples = []
        for scale_size in scale_sizes:
            self.samples.append(SampleBlock(feature_shape, scale_size))
        self.samples = nn.CellList(self.samples)
        self.deeplabv3 = SingleDeepLabV3(num_classes=num_classes,
                                         feature_shape=feature_shape,
                                         backbone=resnet50_dl(fine_tune_batch_norm),
                                         channel=channel,
                                         depth=depth,
                                         scale_sizes=scale_sizes,
                                         atrous_rates=atrous_rates,
                                         decoder_output_stride=decoder_output_stride,
                                         output_stride=output_stride,
                                         fine_tune_batch_norm=fine_tune_batch_norm)
        self.softmax = P.Softmax(axis=1)
        self.concat = P.Concat(axis=2)
        self.expand_dims = P.ExpandDims()
        self.reduce_mean = P.ReduceMean()
        self.sample_common = P.ResizeBilinear((int(feature_shape[2]),
                                               int(feature_shape[3])),
                                              align_corners=True)

    def construct(self, x):
        logits = ()
        if self.training:
            if len(self.image_pyramid) >= 1:
                if self.image_pyramid[0] == 1:
                    logits = self.deeplabv3(x)
                else:
                    x1 = self.samples[0](x)
                    logits = self.deeplabv3(x1)
                    logits = self.sample_common(logits)
                logits = self.expand_dims(logits, 2)
                for i in range(len(self.image_pyramid) - 1):
                    x_i = self.samples[i + 1](x)
                    logits_i = self.deeplabv3(x_i)
                    logits_i = self.sample_common(logits_i)
                    logits_i = self.expand_dims(logits_i, 2)
                    logits = self.concat((logits, logits_i))
            logits = self.reduce_mean(logits, 2)
            return logits
        if len(self.infer_scale_sizes) >= 1:
            infer_index = len(self.image_pyramid)
            x1 = self.samples[infer_index](x)
            logits = self.deeplabv3(x1)
            logits = self.sample_common(logits)
            logits = self.softmax(logits)
            logits = self.expand_dims(logits, 2)
            for i in range(len(self.infer_scale_sizes) - 1):
                x_i = self.samples[i + 1 + infer_index](x)
                logits_i = self.deeplabv3(x_i)
                logits_i = self.sample_common(logits_i)
                logits_i = self.softmax(logits_i)
                logits_i = self.expand_dims(logits_i, 2)
                logits = self.concat((logits, logits_i))
        logits = self.reduce_mean(logits, 2)
        return logits


def deeplabv3_resnet50(num_classes, feature_shape, image_pyramid,
                       infer_scale_sizes, atrous_rates=None, decoder_output_stride=None,
                       output_stride=16, fine_tune_batch_norm=False):
    """
    ResNet50 based DeepLabv3 network.

    Args:
        num_classes (int): Class number.
        feature_shape (list): Input image shape, [N,C,H,W].
        image_pyramid (list): Input scales for multi-scale feature extraction.
        atrous_rates (list): Atrous rates for atrous spatial pyramid pooling.
        infer_scale_sizes (list): 'The scales to resize images for inference.
        decoder_output_stride (int): 'The ratio of input to output spatial resolution'
        output_stride (int): 'The ratio of input to output spatial resolution.'
        fine_tune_batch_norm (bool): 'Fine tune the batch norm parameters or not'

    Returns:
        Cell, cell instance of ResNet50 based DeepLabv3 neural network.

    Examples:
        >>> deeplabv3_resnet50(100, [1,3,224,224],[1.0],[1.0])
    """
    return DeepLabV3(num_classes=num_classes,
                     feature_shape=feature_shape,
                     backbone=resnet50_dl(fine_tune_batch_norm),
                     channel=2048,
                     depth=256,
                     infer_scale_sizes=infer_scale_sizes,
                     atrous_rates=atrous_rates,
                     decoder_output_stride=decoder_output_stride,
                     output_stride=output_stride,
                     fine_tune_batch_norm=fine_tune_batch_norm,
                     image_pyramid=image_pyramid)
