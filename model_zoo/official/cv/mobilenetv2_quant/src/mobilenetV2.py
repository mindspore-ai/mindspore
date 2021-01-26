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
from mindspore import Tensor

__all__ = ['mobilenetV2']


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

    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        super(ConvBNReLU, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2dBnAct(in_planes, out_planes, kernel_size,
                                   stride=stride,
                                   pad_mode='pad',
                                   padding=padding,
                                   group=groups,
                                   has_bn=True,
                                   activation='relu')

    def construct(self, x):
        x = self.conv(x)
        return x


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

    def __init__(self, inp, oup, stride, expand_ratio):
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
            nn.Conv2dBnAct(hidden_dim, oup, kernel_size=1, stride=1, pad_mode='pad', padding=0, group=1, has_bn=True)
        ])
        self.conv = nn.SequentialCell(layers)
        self.add = P.Add()

    def construct(self, x):
        out = self.conv(x)
        if self.use_res_connect:
            out = self.add(out, x)
        return out


class mobilenetV2(nn.Cell):
    """
    mobilenetV2 fusion architecture.

    Args:
        class_num (Cell): number of classes.
        width_mult (int): Channels multiplier for round to 8/16 and others. Default is 1.
        has_dropout (bool): Is dropout used. Default is false
        inverted_residual_setting (list): Inverted residual settings. Default is None
        round_nearest (list): Channel round to . Default is 8
    Returns:
        Tensor, output tensor.

    Examples:
        >>> mobilenetV2(num_classes=1000)
    """

    def __init__(self, num_classes=1000, width_mult=1.,
                 has_dropout=False, inverted_residual_setting=None, round_nearest=8):
        super(mobilenetV2, self).__init__()
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

        features = [ConvBNReLU(3, input_channel, stride=2)]
        # building inverted residual blocks
        for t, c, n, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(input_channel, self.out_channels, kernel_size=1))
        # make it nn.CellList
        self.features = nn.SequentialCell(features)
        # mobilenet head
        head = ([GlobalAvgPooling(),
                 nn.DenseBnAct(self.out_channels, num_classes, has_bias=True, has_bn=False)
                 ] if not has_dropout else
                [GlobalAvgPooling(),
                 nn.Dropout(0.2),
                 nn.DenseBnAct(self.out_channels, num_classes, has_bias=True, has_bn=False)
                 ])
        self.head = nn.SequentialCell(head)

        # init weights
        self.init_parameters_data()
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
        self.init_parameters_data()
        for _, m in self.cells_and_names():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                w = Tensor(np.random.normal(0, np.sqrt(2. / n), m.weight.data.shape).astype("float32"))
                m.weight.set_data(w)
                if m.bias is not None:
                    m.bias.set_data(Tensor(np.zeros(m.bias.data.shape, dtype="float32")))
            elif isinstance(m, nn.Conv2dBnAct):
                n = m.conv.kernel_size[0] * m.conv.kernel_size[1] * m.conv.out_channels
                w = Tensor(np.random.normal(0, np.sqrt(2. / n), m.conv.weight.data.shape).astype("float32"))
                m.conv.weight.set_data(w)
                if m.conv.bias is not None:
                    m.conv.bias.set_data(Tensor(np.zeros(m.conv.bias.data.shape, dtype="float32")))
            elif isinstance(m, nn.BatchNorm2d):
                m.gamma.set_data(Tensor(np.ones(m.gamma.data.shape, dtype="float32")))
                m.beta.set_data(Tensor(np.zeros(m.beta.data.shape, dtype="float32")))
            elif isinstance(m, nn.Dense):
                m.weight.set_data(Tensor(np.random.normal(0, 0.01, m.weight.data.shape).astype("float32")))
                if m.bias is not None:
                    m.bias.set_data(Tensor(np.zeros(m.bias.data.shape, dtype="float32")))
            elif isinstance(m, nn.DenseBnAct):
                m.dense.weight.set_data(
                    Tensor(np.random.normal(0, 0.01, m.dense.weight.data.shape).astype("float32")))
                if m.dense.bias is not None:
                    m.dense.bias.set_data(Tensor(np.zeros(m.dense.bias.data.shape, dtype="float32")))
