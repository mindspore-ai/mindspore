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
"""ResNet based DeepLab."""
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.common.initializer import initializer
from mindspore._checkparam import twice
from mindspore.common.parameter import Parameter


def _conv_bn_relu(in_channel,
                  out_channel,
                  ksize,
                  stride=1,
                  padding=0,
                  dilation=1,
                  pad_mode="pad",
                  use_batch_statistics=False):
    """Get a conv2d -> batchnorm -> relu layer"""
    return nn.SequentialCell(
        [nn.Conv2d(in_channel,
                   out_channel,
                   kernel_size=ksize,
                   stride=stride,
                   padding=padding,
                   dilation=dilation,
                   pad_mode=pad_mode),
         nn.BatchNorm2d(out_channel, use_batch_statistics=use_batch_statistics),
         nn.ReLU()]
    )


def _deep_conv_bn_relu(in_channel,
                       channel_multiplier,
                       ksize,
                       stride=1,
                       padding=0,
                       dilation=1,
                       pad_mode="pad",
                       use_batch_statistics=False):
    """Get a spacetobatch -> conv2d -> batchnorm -> relu -> batchtospace layer"""
    return nn.SequentialCell(
        [DepthwiseConv2dNative(in_channel,
                               channel_multiplier,
                               kernel_size=ksize,
                               stride=stride,
                               padding=padding,
                               dilation=dilation,
                               pad_mode=pad_mode),
         nn.BatchNorm2d(channel_multiplier * in_channel, use_batch_statistics=use_batch_statistics),
         nn.ReLU()]
    )


def _stob_deep_conv_btos_bn_relu(in_channel,
                                 channel_multiplier,
                                 ksize,
                                 space_to_batch_block_shape,
                                 batch_to_space_block_shape,
                                 paddings,
                                 crops,
                                 stride=1,
                                 padding=0,
                                 dilation=1,
                                 pad_mode="pad",
                                 use_batch_statistics=False):
    """Get a spacetobatch -> conv2d -> batchnorm -> relu -> batchtospace layer"""
    return nn.SequentialCell(
        [SpaceToBatch(space_to_batch_block_shape, paddings),
         DepthwiseConv2dNative(in_channel,
                               channel_multiplier,
                               kernel_size=ksize,
                               stride=stride,
                               padding=padding,
                               dilation=dilation,
                               pad_mode=pad_mode),
         BatchToSpace(batch_to_space_block_shape, crops),
         nn.BatchNorm2d(channel_multiplier * in_channel, use_batch_statistics=use_batch_statistics),
         nn.ReLU()]
    )


def _stob_conv_btos_bn_relu(in_channel,
                            out_channel,
                            ksize,
                            space_to_batch_block_shape,
                            batch_to_space_block_shape,
                            paddings,
                            crops,
                            stride=1,
                            padding=0,
                            dilation=1,
                            pad_mode="pad",
                            use_batch_statistics=False):
    """Get a spacetobatch -> conv2d -> batchnorm -> relu -> batchtospace layer"""
    return nn.SequentialCell([SpaceToBatch(space_to_batch_block_shape, paddings),
                              nn.Conv2d(in_channel,
                                        out_channel,
                                        kernel_size=ksize,
                                        stride=stride,
                                        padding=padding,
                                        dilation=dilation,
                                        pad_mode=pad_mode),
                              BatchToSpace(batch_to_space_block_shape, crops),
                              nn.BatchNorm2d(out_channel, use_batch_statistics=use_batch_statistics),
                              nn.ReLU()]
                             )


def _make_layer(block,
                in_channels,
                out_channels,
                num_blocks,
                stride=1,
                rate=1,
                multi_grads=None,
                output_stride=None,
                g_current_stride=2,
                g_rate=1):
    """Make layer for DeepLab-ResNet network."""
    if multi_grads is None:
        multi_grads = [1] * num_blocks
    # (stride == 2, num_blocks == 4 --> strides == [1, 1, 1, 2])
    strides = [1] * (num_blocks - 1) + [stride]
    blocks = []
    if output_stride is not None:
        if output_stride % 4 != 0:
            raise ValueError('The output_stride needs to be a multiple of 4.')
        output_stride //= 4
    for i_stride, _ in enumerate(strides):
        if output_stride is not None and g_current_stride > output_stride:
            raise ValueError('The target output_stride cannot be reached.')
        if output_stride is not None and g_current_stride == output_stride:
            b_rate = g_rate
            b_stride = 1
            g_rate *= strides[i_stride]
        else:
            b_rate = rate
            b_stride = strides[i_stride]
            g_current_stride *= strides[i_stride]
        blocks.append(block(in_channels=in_channels,
                            out_channels=out_channels,
                            stride=b_stride,
                            rate=b_rate,
                            multi_grad=multi_grads[i_stride]))
        in_channels = out_channels
    layer = nn.SequentialCell(blocks)
    return layer, g_current_stride, g_rate


class Subsample(nn.Cell):
    """
    Subsample for DeepLab-ResNet.
    Args:
        factor (int): Sample factor.
    Returns:
        Tensor, the sub sampled tensor.
    Examples:
        >>> Subsample(2)
    """
    def __init__(self, factor):
        super(Subsample, self).__init__()
        self.factor = factor
        self.pool = nn.MaxPool2d(kernel_size=1,
                                 stride=factor)

    def construct(self, x):
        if self.factor == 1:
            return x
        return self.pool(x)


class SpaceToBatch(nn.Cell):
    def __init__(self, block_shape, paddings):
        super(SpaceToBatch, self).__init__()
        self.space_to_batch = P.SpaceToBatch(block_shape, paddings)
        self.bs = block_shape
        self.pd = paddings

    def construct(self, x):
        return self.space_to_batch(x)


class BatchToSpace(nn.Cell):
    def __init__(self, block_shape, crops):
        super(BatchToSpace, self).__init__()
        self.batch_to_space = P.BatchToSpace(block_shape, crops)
        self.bs = block_shape
        self.cr = crops

    def construct(self, x):
        return self.batch_to_space(x)


class _DepthwiseConv2dNative(nn.Cell):
    """Depthwise Conv2D Cell."""
    def __init__(self,
                 in_channels,
                 channel_multiplier,
                 kernel_size,
                 stride,
                 pad_mode,
                 padding,
                 dilation,
                 group,
                 weight_init):
        super(_DepthwiseConv2dNative, self).__init__()
        self.in_channels = in_channels
        self.channel_multiplier = channel_multiplier
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad_mode = pad_mode
        self.padding = padding
        self.dilation = dilation
        self.group = group
        if not (isinstance(in_channels, int) and in_channels > 0):
            raise ValueError('Attr \'in_channels\' of \'DepthwiseConv2D\' Op passed '
                             + str(in_channels) + ', should be a int and greater than 0.')
        if (not isinstance(kernel_size, tuple)) or len(kernel_size) != 2 or \
            (not isinstance(kernel_size[0], int)) or (not isinstance(kernel_size[1], int)) or \
                kernel_size[0] < 1 or kernel_size[1] < 1:
            raise ValueError('Attr \'kernel_size\' of \'DepthwiseConv2D\' Op passed '
                             + str(self.kernel_size) + ', should be a int or tuple and equal to or greater than 1.')
        self.weight = Parameter(initializer(weight_init, [1, in_channels // group, *kernel_size]),
                                name='weight')

    def construct(self, *inputs):
        """Must be overridden by all subclasses."""
        raise NotImplementedError


class DepthwiseConv2dNative(_DepthwiseConv2dNative):
    """Depthwise Conv2D Cell."""
    def __init__(self,
                 in_channels,
                 channel_multiplier,
                 kernel_size,
                 stride=1,
                 pad_mode='same',
                 padding=0,
                 dilation=1,
                 group=1,
                 weight_init='normal'):
        kernel_size = twice(kernel_size)
        super(DepthwiseConv2dNative, self).__init__(
            in_channels,
            channel_multiplier,
            kernel_size,
            stride,
            pad_mode,
            padding,
            dilation,
            group,
            weight_init)
        self.depthwise_conv2d_native = P.DepthwiseConv2dNative(channel_multiplier=self.channel_multiplier,
                                                               kernel_size=self.kernel_size,
                                                               mode=3,
                                                               pad_mode=self.pad_mode,
                                                               pad=self.padding,
                                                               stride=self.stride,
                                                               dilation=self.dilation,
                                                               group=self.group)

    def shard(self, strategy):
        self.depthwise_conv2d_native.shard(strategy)
        return self

    def construct(self, x):
        return self.depthwise_conv2d_native(x, self.weight)


class BottleneckV1(nn.Cell):
    """
    ResNet V1 BottleneckV1 block definition.
    Args:
        in_channels (int): Input channel.
        out_channels (int): Output channel.
        stride (int): Stride size for the initial convolutional layer. Default: 1.
        rate (int): Rate for convolution. Default: 1.
        multi_grad (int): Employ a rate within network. Default: 1.
    Returns:
        Tensor, the ResNet unit's output.
    Examples:
        >>> BottleneckV1(3,256,stride=2)
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 use_batch_statistics=False,
                 use_batch_to_stob_and_btos=False):
        super(BottleneckV1, self).__init__()
        expansion = 4
        mid_channels = out_channels // expansion
        self.conv_bn1 = _conv_bn_relu(in_channels,
                                      mid_channels,
                                      ksize=1,
                                      stride=1,
                                      use_batch_statistics=use_batch_statistics)
        self.conv_bn2 = _conv_bn_relu(mid_channels,
                                      mid_channels,
                                      ksize=3,
                                      stride=stride,
                                      padding=1,
                                      dilation=1,
                                      use_batch_statistics=use_batch_statistics)
        if use_batch_to_stob_and_btos:
            self.conv_bn2 = _stob_conv_btos_bn_relu(mid_channels,
                                                    mid_channels,
                                                    ksize=3,
                                                    stride=stride,
                                                    padding=0,
                                                    dilation=1,
                                                    space_to_batch_block_shape=2,
                                                    batch_to_space_block_shape=2,
                                                    paddings=[[2, 3], [2, 3]],
                                                    crops=[[0, 1], [0, 1]],
                                                    pad_mode="valid",
                                                    use_batch_statistics=use_batch_statistics)

        self.conv3 = nn.Conv2d(mid_channels,
                               out_channels,
                               kernel_size=1,
                               stride=1)
        self.bn3 = nn.BatchNorm2d(out_channels, use_batch_statistics=use_batch_statistics)
        if in_channels != out_channels:
            conv = nn.Conv2d(in_channels,
                             out_channels,
                             kernel_size=1,
                             stride=stride)
            bn = nn.BatchNorm2d(out_channels, use_batch_statistics=use_batch_statistics)
            self.downsample = nn.SequentialCell([conv, bn])
        else:
            self.downsample = Subsample(stride)
        self.add = P.Add()
        self.relu = nn.ReLU()
        self.Reshape = P.Reshape()

    def construct(self, x):
        out = self.conv_bn1(x)
        out = self.conv_bn2(out)
        out = self.bn3(self.conv3(out))
        out = self.add(out, self.downsample(x))
        out = self.relu(out)
        return out


class BottleneckV2(nn.Cell):
    """
    ResNet V2 Bottleneck variance V2 block definition.
    Args:
        in_channels (int): Input channel.
        out_channels (int): Output channel.
        stride (int): Stride size for the initial convolutional layer. Default: 1.
    Returns:
        Tensor, the ResNet unit's output.
    Examples:
        >>> BottleneckV2(3,256,stride=2)
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 use_batch_statistics=False,
                 use_batch_to_stob_and_btos=False,
                 dilation=1):
        super(BottleneckV2, self).__init__()
        expansion = 4
        mid_channels = out_channels // expansion
        self.conv_bn1 = _conv_bn_relu(in_channels,
                                      mid_channels,
                                      ksize=1,
                                      stride=1,
                                      use_batch_statistics=use_batch_statistics)
        self.conv_bn2 = _conv_bn_relu(mid_channels,
                                      mid_channels,
                                      ksize=3,
                                      stride=stride,
                                      padding=1,
                                      dilation=dilation,
                                      use_batch_statistics=use_batch_statistics)
        if use_batch_to_stob_and_btos:
            self.conv_bn2 = _stob_conv_btos_bn_relu(mid_channels,
                                                    mid_channels,
                                                    ksize=3,
                                                    stride=stride,
                                                    padding=0,
                                                    dilation=1,
                                                    space_to_batch_block_shape=2,
                                                    batch_to_space_block_shape=2,
                                                    paddings=[[2, 3], [2, 3]],
                                                    crops=[[0, 1], [0, 1]],
                                                    pad_mode="valid",
                                                    use_batch_statistics=use_batch_statistics)
        self.conv3 = nn.Conv2d(mid_channels,
                               out_channels,
                               kernel_size=1,
                               stride=1)
        self.bn3 = nn.BatchNorm2d(out_channels, use_batch_statistics=use_batch_statistics)
        if in_channels != out_channels:
            conv = nn.Conv2d(in_channels,
                             out_channels,
                             kernel_size=1,
                             stride=stride)
            bn = nn.BatchNorm2d(out_channels, use_batch_statistics=use_batch_statistics)
            self.downsample = nn.SequentialCell([conv, bn])
        else:
            self.downsample = Subsample(stride)
        self.add = P.Add()
        self.relu = nn.ReLU()

    def construct(self, x):
        out = self.conv_bn1(x)
        out = self.conv_bn2(out)
        out = self.bn3(self.conv3(out))
        out = self.add(out, x)
        out = self.relu(out)
        return out


class BottleneckV3(nn.Cell):
    """
    ResNet V1 Bottleneck variance V1 block definition.
    Args:
        in_channels (int): Input channel.
        out_channels (int): Output channel.
        stride (int): Stride size for the initial convolutional layer. Default: 1.
    Returns:
        Tensor, the ResNet unit's output.
    Examples:
        >>> BottleneckV3(3,256,stride=2)
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 use_batch_statistics=False):
        super(BottleneckV3, self).__init__()
        expansion = 4
        mid_channels = out_channels // expansion
        self.conv_bn1 = _conv_bn_relu(in_channels,
                                      mid_channels,
                                      ksize=1,
                                      stride=1,
                                      use_batch_statistics=use_batch_statistics)
        self.conv_bn2 = _conv_bn_relu(mid_channels,
                                      mid_channels,
                                      ksize=3,
                                      stride=stride,
                                      padding=1,
                                      dilation=1,
                                      use_batch_statistics=use_batch_statistics)
        self.conv3 = nn.Conv2d(mid_channels,
                               out_channels,
                               kernel_size=1,
                               stride=1)
        self.bn3 = nn.BatchNorm2d(out_channels, use_batch_statistics=use_batch_statistics)

        if in_channels != out_channels:
            conv = nn.Conv2d(in_channels,
                             out_channels,
                             kernel_size=1,
                             stride=stride)
            bn = nn.BatchNorm2d(out_channels, use_batch_statistics=use_batch_statistics)
            self.downsample = nn.SequentialCell([conv, bn])
        else:
            self.downsample = Subsample(stride)
        self.downsample = Subsample(stride)
        self.add = P.Add()
        self.relu = nn.ReLU()

    def construct(self, x):
        out = self.conv_bn1(x)
        out = self.conv_bn2(out)
        out = self.bn3(self.conv3(out))
        out = self.add(out, self.downsample(x))
        out = self.relu(out)
        return out


class ResNetV1(nn.Cell):
    """
    ResNet V1 for DeepLab.
    Args:
    Returns:
        Tuple, output tensor tuple, (c2,c5).
    Examples:
        >>> ResNetV1(False)
    """
    def __init__(self, fine_tune_batch_norm=False):
        super(ResNetV1, self).__init__()
        self.layer_root = nn.SequentialCell(
            [RootBlockBeta(fine_tune_batch_norm),
             nn.MaxPool2d(kernel_size=(3, 3),
                          stride=(2, 2),
                          pad_mode='same')])
        self.layer1_1 = BottleneckV1(128, 256, stride=1, use_batch_statistics=fine_tune_batch_norm)
        self.layer1_2 = BottleneckV2(256, 256, stride=1, use_batch_statistics=fine_tune_batch_norm)
        self.layer1_3 = BottleneckV3(256, 256, stride=2, use_batch_statistics=fine_tune_batch_norm)
        self.layer2_1 = BottleneckV1(256, 512, stride=1, use_batch_statistics=fine_tune_batch_norm)
        self.layer2_2 = BottleneckV2(512, 512, stride=1, use_batch_statistics=fine_tune_batch_norm)
        self.layer2_3 = BottleneckV2(512, 512, stride=1, use_batch_statistics=fine_tune_batch_norm)
        self.layer2_4 = BottleneckV3(512, 512, stride=2, use_batch_statistics=fine_tune_batch_norm)
        self.layer3_1 = BottleneckV1(512, 1024, stride=1, use_batch_statistics=fine_tune_batch_norm)
        self.layer3_2 = BottleneckV2(1024, 1024, stride=1, use_batch_statistics=fine_tune_batch_norm)
        self.layer3_3 = BottleneckV2(1024, 1024, stride=1, use_batch_statistics=fine_tune_batch_norm)
        self.layer3_4 = BottleneckV2(1024, 1024, stride=1, use_batch_statistics=fine_tune_batch_norm)
        self.layer3_5 = BottleneckV2(1024, 1024, stride=1, use_batch_statistics=fine_tune_batch_norm)
        self.layer3_6 = BottleneckV2(1024, 1024, stride=1, use_batch_statistics=fine_tune_batch_norm)

        self.layer4_1 = BottleneckV1(1024, 2048, stride=1, use_batch_to_stob_and_btos=True,
                                     use_batch_statistics=fine_tune_batch_norm)
        self.layer4_2 = BottleneckV2(2048, 2048, stride=1, use_batch_to_stob_and_btos=True,
                                     use_batch_statistics=fine_tune_batch_norm)
        self.layer4_3 = BottleneckV2(2048, 2048, stride=1, use_batch_to_stob_and_btos=True,
                                     use_batch_statistics=fine_tune_batch_norm)

    def construct(self, x):
        x = self.layer_root(x)
        x = self.layer1_1(x)
        c2 = self.layer1_2(x)
        x = self.layer1_3(c2)
        x = self.layer2_1(x)
        x = self.layer2_2(x)
        x = self.layer2_3(x)
        x = self.layer2_4(x)
        x = self.layer3_1(x)
        x = self.layer3_2(x)
        x = self.layer3_3(x)
        x = self.layer3_4(x)
        x = self.layer3_5(x)
        x = self.layer3_6(x)

        x = self.layer4_1(x)
        x = self.layer4_2(x)
        c5 = self.layer4_3(x)
        return c2, c5


class RootBlockBeta(nn.Cell):
    """
    ResNet V1 beta root block definition.
    Returns:
        Tensor, the block unit's output.
    Examples:
        >>> RootBlockBeta()
    """
    def __init__(self, fine_tune_batch_norm=False):
        super(RootBlockBeta, self).__init__()
        self.conv1 = _conv_bn_relu(3, 64, ksize=3, stride=2, padding=0, pad_mode="valid",
                                   use_batch_statistics=fine_tune_batch_norm)
        self.conv2 = _conv_bn_relu(64, 64, ksize=3, stride=1, padding=0, pad_mode="same",
                                   use_batch_statistics=fine_tune_batch_norm)
        self.conv3 = _conv_bn_relu(64, 128, ksize=3, stride=1, padding=0, pad_mode="same",
                                   use_batch_statistics=fine_tune_batch_norm)

    def construct(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


def resnet50_dl(fine_tune_batch_norm=False):
    return ResNetV1(fine_tune_batch_norm)
