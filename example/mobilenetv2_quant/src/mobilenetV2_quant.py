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
"""MobileNetV2 Quant model define"""
import numpy as np
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.ops.operations import TensorAdd
from mindspore import Parameter, Tensor
from mindspore.common.initializer import initializer

__all__ = ['mobilenet_v2_quant']

_ema_decay = 0.999
_symmetric = False

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class GlobalAvgPooling(nn.Cell):
    """
    Global avg pooling definition.

    Args:

    Returns:
        Tensor, output tensor.

    Examples:
        >>> GlobalAvgPooling()
    """

    def __init__(self):
        super(GlobalAvgPooling, self).__init__()
        self.mean = P.ReduceMean(keep_dims=False)

    def construct(self, x):
        x = self.mean(x, (2, 3))
        return x


class DepthwiseConv(nn.Cell):
    """
    Depthwise Convolution warpper definition.

    Args:
        in_planes (int): Input channel.
        kernel_size (int): Input kernel size.
        stride (int): Stride size.
        pad_mode (str): pad mode in (pad, same, valid)
        channel_multiplier (int): Output channel multiplier
        has_bias (bool): has bias or not

    Returns:
        Tensor, output tensor.

    Examples:
        >>> DepthwiseConv(16, 3, 1, 'pad', 1, channel_multiplier=1)
    """

    def __init__(self, in_planes, kernel_size, stride, pad_mode, pad, channel_multiplier=1, has_bias=False):
        super(DepthwiseConv, self).__init__()
        self.has_bias = has_bias
        self.in_channels = in_planes
        self.channel_multiplier = channel_multiplier
        self.out_channels = in_planes * channel_multiplier
        self.kernel_size = (kernel_size, kernel_size)
        self.depthwise_conv = P.DepthwiseConv2dNative(channel_multiplier=channel_multiplier,
                                                      kernel_size=self.kernel_size,
                                                      stride=stride, pad_mode=pad_mode, pad=pad)
        self.bias_add = P.BiasAdd()
        weight_shape = [channel_multiplier, in_planes, *self.kernel_size]
        self.weight = Parameter(initializer('ones', weight_shape), name='weight')

        if has_bias:
            bias_shape = [channel_multiplier * in_planes]
            self.bias = Parameter(initializer('zeros', bias_shape), name='bias')
        else:
            self.bias = None

    def construct(self, x):
        output = self.depthwise_conv(x, self.weight)
        if self.has_bias:
            output = self.bias_add(output, self.bias)
        return output


class ConvBNReLU(nn.Cell):
    """
    Convolution/Depthwise fused with Batchnorm and ReLU block definition.

    Args:
        in_planes (int): Input channel.
        out_planes (int): Output channel.
        kernel_size (int): Input kernel size.
        stride (int): Stride size for the first convolutional layer. Default: 1.
        groups (int): channel group. Convolution is 1 while Depthiwse is input channel. Default: 1.

    Returns:
        Tensor, output tensor.

    Examples:
        >>> ConvBNReLU(16, 256, kernel_size=1, stride=1, groups=1)
    """

    def __init__(self, platform, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        super(ConvBNReLU, self).__init__()
        padding = (kernel_size - 1) // 2
        if groups == 1:
            conv = nn.Conv2d(in_planes, out_planes, kernel_size, stride, pad_mode='pad', padding=padding)
        else:
            if platform == "Ascend":
                conv = DepthwiseConv(in_planes, kernel_size, stride, pad_mode='pad', pad=padding)
            elif platform == "GPU":
                conv = nn.Conv2d(in_planes, out_planes, kernel_size, stride,
                                 group=in_planes, pad_mode='pad', padding=padding)

        layers = [conv, nn.BatchNorm2d(out_planes), nn.ReLU6()]
        self.features = nn.SequentialCell(layers)
        self.fake = nn.FakeQuantWithMinMax(in_planes, ema=True, ema_decay=_ema_decay, symmetric=_symmetric)

    def construct(self, x):
        output = self.features(x)
        output = self.fake(output)
        return output


class InvertedResidual(nn.Cell):
    """
    Mobilenetv2 residual block definition.

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

    def __init__(self, platform, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(platform, inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(platform, hidden_dim, hidden_dim,
                       stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv2dBatchNormQuant(hidden_dim, oup, kernel_size=1, stride=1, pad_mode='pad', padding=0, group=1),
            nn.FakeQuantWithMinMax(oup, ema=True, ema_decay=_ema_decay, symmetric=_symmetric)
        ])
        self.conv = nn.SequentialCell(layers)
        self.add = TensorAdd()
        self.add_fake = nn.FakeQuantWithMinMax(oup, ema=True, ema_decay=_ema_decay, symmetric=_symmetric)
        self.cast = P.Cast()

    def construct(self, x):
        identity = x
        x = self.conv(x)
        if self.use_res_connect:
            x = self.add(identity, x)
            x = self.add_fake(x)
        return x


class MobileNetV2Quant(nn.Cell):
    """
    MobileNetV2Quant architecture.

    Args:
        class_num (Cell): number of classes.
        width_mult (int): Channels multiplier for round to 8/16 and others. Default is 1.
        has_dropout (bool): Is dropout used. Default is false
        inverted_residual_setting (list): Inverted residual settings. Default is None
        round_nearest (list): Channel round to . Default is 8
    Returns:
        Tensor, output tensor.

    Examples:
        >>> MobileNetV2Quant(num_classes=1000)
    """

    def __init__(self, platform, num_classes=1000, width_mult=1.,
                 has_dropout=False, inverted_residual_setting=None, round_nearest=8):
        super(MobileNetV2Quant, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        # setting of inverted residual blocks
        self.cfgs = inverted_residual_setting
        if inverted_residual_setting is None:
            self.cfgs = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.out_channels = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU(platform, 3, input_channel, stride=2)]
        # building inverted residual blocks
        for t, c, n, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(platform, input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(platform, input_channel, self.out_channels, kernel_size=1))
        # make it nn.CellList
        self.features = nn.SequentialCell(features)
        # mobilenet head
        head = ([GlobalAvgPooling(), nn.Dense(self.out_channels, num_classes, has_bias=True)] if not has_dropout else
                [GlobalAvgPooling(), nn.Dropout(0.2), nn.Dense(self.out_channels, num_classes, has_bias=True)])
        self.head = nn.SequentialCell(head)

        self._initialize_weights()

    def construct(self, x):
        x = self.features(x)
        x = self.head(x)
        return x

    def _initialize_weights(self):
        """
        Initialize weights.

        Args:

        Returns:
            None.

        Examples:
            >>> _initialize_weights()
        """
        for _, m in self.cells_and_names():
            if isinstance(m, (nn.Conv2d, DepthwiseConv)):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.set_parameter_data(Tensor(np.random.normal(0, np.sqrt(2. / n),
                                                                    m.weight.data.shape()).astype("float32")))
                if m.bias is not None:
                    m.bias.set_parameter_data(
                        Tensor(np.zeros(m.bias.data.shape(), dtype="float32")))
            elif isinstance(m, nn.BatchNorm2d):
                m.gamma.set_parameter_data(
                    Tensor(np.ones(m.gamma.data.shape(), dtype="float32")))
                m.beta.set_parameter_data(
                    Tensor(np.zeros(m.beta.data.shape(), dtype="float32")))
            elif isinstance(m, nn.Dense):
                m.weight.set_parameter_data(Tensor(np.random.normal(
                    0, 0.01, m.weight.data.shape()).astype("float32")))
                if m.bias is not None:
                    m.bias.set_parameter_data(
                        Tensor(np.zeros(m.bias.data.shape(), dtype="float32")))


def mobilenet_v2_quant(**kwargs):
    """
    Constructs a MobileNet V2 model
    """
    return MobileNetV2Quant(**kwargs)
