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

"""build FPN"""
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.ops import functional as F


def _make_divisible(v, divisor, min_value=None):
    """nsures that all layers have a channel number that is divisible by 8."""
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def _conv2d(in_channel, out_channel, kernel_size=3, stride=1, group=1, pad_mod='same'):
    return nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride,
                     padding=0, pad_mode=pad_mod, has_bias=True, group=1)

def _bn(channel):
    return nn.BatchNorm2d(channel, eps=1e-3, momentum=0.97,
                          gamma_init=1, beta_init=0, moving_mean_init=0, moving_var_init=1)

def _last_conv2d(in_channel, out_channel, kernel_size=3, stride=1, pad_mod='same', pad=0):
    in_channels = in_channel
    out_channels = in_channel
    depthwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad_mode='same',
                               padding=pad, group=in_channels)
    conv = _conv2d(in_channel, out_channel, kernel_size=1)
    return nn.SequentialCell([depthwise_conv, _bn(in_channel), nn.ReLU6(), conv])

def conv_bn_relu(in_channel, out_channel, kernel_size, stride, depthwise, activation='relu6'):
    output = []
    output.append(nn.Conv2d(in_channel, out_channel, kernel_size, stride, pad_mode="same",
                            group=1 if not depthwise else in_channel))
    output.append(nn.BatchNorm2d(out_channel))
    if activation:
        output.append(nn.get_activation(activation))
    return nn.SequentialCell(output)

class ConvBNReLU(nn.Cell):
    """
    Convolution/Depthwise fused with Batchnorm and ReLU block definition.

    Args:
        in_planes (int): Input channel.
        out_planes (int): Output channel.
        kernel_size (int): Input kernel size.
        stride (int): Stride size for the first convolutional layer. Default: 1.
        groups (int): channel group. Convolution is 1 while Depthiwse is input channel. Default: 1.
        shared_conv(Cell): Use the weight shared conv, default: None.

    Returns:
        Tensor, output tensor.

    Examples:
        >>> ConvBNReLU(16, 256, kernel_size=1, stride=1, groups=1)
    """
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, shared_conv=None):
        super(ConvBNReLU, self).__init__()
        padding = 0
        in_channels = in_planes
        out_channels = out_planes
        if shared_conv is None:
            if groups == 1:
                conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad_mode='same', padding=padding)
            else:
                out_channels = in_planes
                conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad_mode='same',
                                 padding=padding, group=in_channels)
            layers = [conv, _bn(out_planes), nn.ReLU6()]
        else:
            layers = [shared_conv, _bn(out_planes), nn.ReLU6()]
        self.features = nn.SequentialCell(layers)

    def construct(self, x):
        output = self.features(x)
        return output
class InvertedResidual(nn.Cell):
    """
    Residual block definition.

    Args:
        inp (int): Input channel.
        oup (int): Output channel.
        stride (int): Stride size for the first convolutional layer. Default: 1.
        expand_ratio (int): expand ration of input channel

    Returns:
        Tensor, output tensor.

    Examples:
        >>> ResidualBlock(3, 256, 1, 1)
    """
    def __init__(self, inp, oup, stride, expand_ratio, last_relu=False):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, kernel_size=1, stride=1, has_bias=False),
            _bn(oup),
        ])
        self.conv = nn.SequentialCell(layers)
        self.cast = P.Cast()
        self.last_relu = last_relu
        self.relu = nn.ReLU6()

    def construct(self, x):
        identity = x
        x = self.conv(x)
        if self.use_res_connect:
            x = identity + x
        if self.last_relu:
            x = self.relu(x)
        return x

class SSDWithMobileNetV2(nn.Cell):
    """
    MobileNetV2 architecture for SSD backbone.

    Args:
        width_mult (int): Channels multiplier for round to 8/16 and others. Default is 1.
        inverted_residual_setting (list): Inverted residual settings. Default is None
        round_nearest (list): Channel round to. Default is 8
    Returns:
        Tensor,
        Tensor,
        Tensor

    Examples:
        >>> SSDWithMobileNetV2()
    """
    def __init__(self, width_mult=1.0, inverted_residual_setting=None, round_nearest=8):
        super(SSDWithMobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]
        if len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        #building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU(3, input_channel, stride=2)]
        # building inverted residual blocks
        layer_index = 0
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                if layer_index == 6:
                    hidden_dim = input_channel
                    self.expand_layer_conv_6 = ConvBNReLU(input_channel, hidden_dim, kernel_size=1)
                if layer_index == 13:
                    hidden_dim = input_channel
                    self.expand_layer_conv_13 = ConvBNReLU(input_channel, hidden_dim, kernel_size=1)
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
                layer_index += 1
        # building last several layers
        features.append(ConvBNReLU(input_channel, last_channel, kernel_size=1))

        self.features_1 = nn.SequentialCell(features[:7])
        self.features_2 = nn.SequentialCell(features[7:14])
        self.features_3 = nn.SequentialCell(features[14:19])

    def construct(self, x):
        out = self.features_1(x)
        expand_layer_conv_6 = self.expand_layer_conv_6(out)
        out = self.features_2(out)
        expand_layer_conv_13 = self.expand_layer_conv_13(out)
        out = self.features_3(out)
        return expand_layer_conv_6, expand_layer_conv_13, out


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
            fpn_conv = conv_bn_relu(out_channels, out_channels, kernel_size=3, stride=1, depthwise=True)
            self.lateral_convs_list_.append(l_conv)
            self.fpn_convs_.append(fpn_conv)
        self.lateral_convs_list = nn.layer.CellList(self.lateral_convs_list_)
        self.fpn_convs_list = nn.layer.CellList(self.fpn_convs_)
        self.num_layers = len(in_channel_list)

    def construct(self, inputs):
        """extract features"""
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
            conv_bn_relu(channels, channels, kernel_size, stride, True) for x in range(self.levels)
        ]
        self.blocks = nn.CellList(bottom_up_cells)

    def construct(self, features):
        for block in self.blocks:
            features = features + (block(features[-1]),)
        return features



class MobileNetV2Fpn(nn.Cell):
    """
    MobileNetV2 with FPN as SSD backbone.
    """
    def __init__(self, config):
        super(MobileNetV2Fpn, self).__init__()
        self.mobilenet_v2 = SSDWithMobileNetV2(width_mult=1.0, inverted_residual_setting=None, round_nearest=8)
        self.fpn = FpnTopDown([32, 96, 1280], 32)
        self.bottom_up = BottomUp(2, 32, 3, 2)

    def construct(self, x):
        features = self.mobilenet_v2(x)
        features = self.fpn(features)
        features = self.bottom_up(features)
        return features


def mobilenet_v2_fpn(config):
    return MobileNetV2Fpn(config)
