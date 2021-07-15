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
"""GhostNet model define"""
from functools import partial
import math
import numpy as np
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore import Tensor


__all__ = ['ghostnet']


def _make_divisible(x, divisor=4):
    return int(np.ceil(x * 1. / divisor) * divisor)


class MyHSigmoid(nn.Cell):
    """
    Hard Sigmoid definition.

    Args:

    Returns:
        Tensor, output tensor.

    Examples:
        >>> MyHSigmoid()
    """

    def __init__(self):
        super(MyHSigmoid, self).__init__()
        self.relu6 = nn.ReLU6()

    def construct(self, x):
        """ construct """
        return self.relu6(x + 3.) * 0.16666667


class Activation(nn.Cell):
    """
    Activation definition.

    Args:
        act_func(string): activation name.

    Returns:
         Tensor, output tensor.
    """

    def __init__(self, act_func):
        super(Activation, self).__init__()
        if act_func == 'relu':
            self.act = nn.ReLU()
        elif act_func == 'relu6':
            self.act = nn.ReLU6()
        elif act_func in ('hsigmoid', 'hard_sigmoid'):
            self.act = MyHSigmoid()
        elif act_func in ('hswish', 'hard_swish'):
            self.act = nn.HSwish()
        else:
            raise NotImplementedError

    def construct(self, x):
        """ construct """
        return self.act(x)


class GlobalAvgPooling(nn.Cell):
    """
    Global avg pooling definition.

    Args:

    Returns:
        Tensor, output tensor.

    Examples:
        >>> GlobalAvgPooling()
    """

    def __init__(self, keep_dims=False):
        super(GlobalAvgPooling, self).__init__()
        self.mean = P.ReduceMean(keep_dims=keep_dims)

    def construct(self, x):
        """ construct """
        x = self.mean(x, (2, 3))
        return x


class SE(nn.Cell):
    """
    SE warpper definition.

    Args:
        num_out (int): Output channel.
        ratio (int): middle output ratio.

    Returns:
        Tensor, output tensor.

    Examples:
        >>> SE(4)
    """

    def __init__(self, num_out, ratio=4):
        super(SE, self).__init__()
        num_mid = _make_divisible(num_out // ratio)
        self.pool = GlobalAvgPooling(keep_dims=True)
        self.conv_reduce = nn.Conv2d(in_channels=num_out, out_channels=num_mid,
                                     kernel_size=1, has_bias=True, pad_mode='pad')
        self.act1 = Activation('relu')
        self.conv_expand = nn.Conv2d(in_channels=num_mid, out_channels=num_out,
                                     kernel_size=1, has_bias=True, pad_mode='pad')
        self.act2 = Activation('hsigmoid')
        self.mul = P.Mul()

    def construct(self, x):
        """ construct of SE module """
        out = self.pool(x)
        out = self.conv_reduce(out)
        out = self.act1(out)
        out = self.conv_expand(out)
        out = self.act2(out)
        out = self.mul(x, out)
        return out


class ConvUnit(nn.Cell):
    """
    ConvUnit warpper definition.

    Args:
        num_in (int): Input channel.
        num_out (int): Output channel.
        kernel_size (int): Input kernel size.
        stride (int): Stride size.
        padding (int): Padding number.
        num_groups (int): Output num group.
        use_act (bool): Used activation or not.
        act_type (string): Activation type.

    Returns:
        Tensor, output tensor.

    Examples:
        >>> ConvUnit(3, 3)
    """

    def __init__(self, num_in, num_out, kernel_size=1, stride=1, padding=0, num_groups=1,
                 use_act=True, act_type='relu'):
        super(ConvUnit, self).__init__()
        self.conv = nn.Conv2d(in_channels=num_in,
                              out_channels=num_out,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              group=num_groups,
                              has_bias=False,
                              pad_mode='pad')
        self.bn = nn.BatchNorm2d(num_out)
        self.use_act = use_act
        self.act = Activation(act_type) if use_act else None

    def construct(self, x):
        """ construct of conv unit """
        out = self.conv(x)
        out = self.bn(out)
        if self.use_act:
            out = self.act(out)
        return out


class GhostModule(nn.Cell):
    """
    GhostModule warpper definition.

    Args:
        num_in (int): Input channel.
        num_out (int): Output channel.
        kernel_size (int): Input kernel size.
        stride (int): Stride size.
        padding (int): Padding number.
        ratio (int): Reduction ratio.
        dw_size (int): kernel size of cheap operation.
        use_act (bool): Used activation or not.
        act_type (string): Activation type.

    Returns:
        Tensor, output tensor.

    Examples:
        >>> GhostModule(3, 3)
    """

    def __init__(self, num_in, num_out, kernel_size=1, stride=1, padding=0, ratio=2, dw_size=3,
                 use_act=True, act_type='relu'):
        super(GhostModule, self).__init__()
        init_channels = math.ceil(num_out / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = ConvUnit(num_in, init_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                     num_groups=1, use_act=use_act, act_type=act_type)
        self.cheap_operation = ConvUnit(init_channels, new_channels, kernel_size=dw_size, stride=1,
                                        padding=dw_size // 2, num_groups=init_channels,
                                        use_act=use_act, act_type=act_type)
        self.concat = P.Concat(axis=1)

    def construct(self, x):
        """ ghost module construct """
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        return self.concat((x1, x2))


class GhostBottleneck(nn.Cell):
    """
    GhostBottleneck warpper definition.

    Args:
        num_in (int): Input channel.
        num_mid (int): Middle channel.
        num_out (int): Output channel.
        kernel_size (int): Input kernel size.
        stride (int): Stride size.
        act_type (str): Activation type.
        use_se (bool): Use SE warpper or not.

    Returns:
        Tensor, output tensor.

    Examples:
        >>> GhostBottleneck(16, 3, 1, 1)
    """

    def __init__(self, num_in, num_mid, num_out, kernel_size, stride=1, act_type='relu', use_se=False):
        super(GhostBottleneck, self).__init__()
        self.ghost1 = GhostModule(num_in, num_mid, kernel_size=1,
                                  stride=1, padding=0, act_type=act_type)

        self.use_dw = stride > 1
        self.dw = None
        if self.use_dw:
            self.dw = ConvUnit(num_mid, num_mid, kernel_size=kernel_size, stride=stride,
                               padding=self._get_pad(kernel_size), act_type=act_type, num_groups=num_mid, use_act=False)

        self.use_se = use_se
        if use_se:
            self.se = SE(num_mid)

        self.ghost2 = GhostModule(num_mid, num_out, kernel_size=1, stride=1,
                                  padding=0, act_type=act_type, use_act=False)

        self.down_sample = False
        if num_in != num_out or stride != 1:
            self.down_sample = True
        self.shortcut = None
        if self.down_sample:
            self.shortcut = nn.SequentialCell([
                ConvUnit(num_in, num_in, kernel_size=kernel_size, stride=stride,
                         padding=self._get_pad(kernel_size), num_groups=num_in, use_act=False),
                ConvUnit(num_in, num_out, kernel_size=1, stride=1,
                         padding=0, num_groups=1, use_act=False),
            ])
        self.add = P.TensorAdd()

    def construct(self, x):
        """ construct of ghostnet """
        shortcut = x
        out = self.ghost1(x)
        if self.use_dw:
            out = self.dw(out)
        if self.use_se:
            out = self.se(out)
        out = self.ghost2(out)
        if self.down_sample:
            shortcut = self.shortcut(shortcut)
        out = self.add(shortcut, out)
        return out

    def _get_pad(self, kernel_size):
        """set the padding number"""
        pad = 0
        if kernel_size == 1:
            pad = 0
        elif kernel_size == 3:
            pad = 1
        elif kernel_size == 5:
            pad = 2
        elif kernel_size == 7:
            pad = 3
        else:
            raise NotImplementedError
        return pad


class GhostNet(nn.Cell):
    """
    GhostNet architecture.

    Args:
        model_cfgs (Cell): number of classes.
        num_classes (int): Output number classes.
        multiplier (int): Channels multiplier for round to 8/16 and others. Default is 1.
        final_drop (float): Dropout number.
        round_nearest (list): Channel round to . Default is 8.
    Returns:
        Tensor, output tensor.

    Examples:
        >>> GhostNet(num_classes=1000)
    """

    def __init__(self, model_cfgs, num_classes=1000, multiplier=1., final_drop=0.):
        super(GhostNet, self).__init__()
        self.cfgs = model_cfgs['cfg']
        self.inplanes = 16
        first_conv_in_channel = 3
        first_conv_out_channel = _make_divisible(multiplier * self.inplanes)

        self.conv_stem = nn.Conv2d(in_channels=first_conv_in_channel,
                                   out_channels=first_conv_out_channel,
                                   kernel_size=3, padding=1, stride=2,
                                   has_bias=False, pad_mode='pad')
        self.bn1 = nn.BatchNorm2d(first_conv_out_channel)
        self.act1 = Activation('relu')

        self.blocks = []
        for layer_cfg in self.cfgs:
            self.blocks.append(self._make_layer(kernel_size=layer_cfg[0],
                                                exp_ch=_make_divisible(
                                                    multiplier * layer_cfg[1]),
                                                out_channel=_make_divisible(
                                                    multiplier * layer_cfg[2]),
                                                use_se=layer_cfg[3],
                                                act_func=layer_cfg[4],
                                                stride=layer_cfg[5]))
        output_channel = _make_divisible(
            multiplier * model_cfgs["cls_ch_squeeze"])
        self.blocks.append(ConvUnit(_make_divisible(multiplier * self.cfgs[-1][2]), output_channel,
                                    kernel_size=1, stride=1, padding=0, num_groups=1, use_act=True))
        self.blocks = nn.SequentialCell(self.blocks)

        self.global_pool = GlobalAvgPooling(keep_dims=True)
        self.conv_head = nn.Conv2d(in_channels=output_channel,
                                   out_channels=model_cfgs['cls_ch_expand'],
                                   kernel_size=1, padding=0, stride=1,
                                   has_bias=True, pad_mode='pad')
        self.act2 = Activation('relu')
        self.squeeze = P.Flatten()
        self.final_drop = final_drop
        if self.final_drop > 0:
            self.dropout = nn.Dropout(self.final_drop)

        self.classifier = nn.Dense(
            model_cfgs['cls_ch_expand'], num_classes, has_bias=True)

        self._initialize_weights()

    def construct(self, x):
        """ construct of GhostNet """
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.blocks(x)
        x = self.global_pool(x)
        x = self.conv_head(x)
        x = self.act2(x)
        x = self.squeeze(x)
        if self.final_drop > 0:
            x = self.dropout(x)
        x = self.classifier(x)
        return x

    def _make_layer(self, kernel_size, exp_ch, out_channel, use_se, act_func, stride=1):
        mid_planes = exp_ch
        out_planes = out_channel
        layer = GhostBottleneck(self.inplanes, mid_planes, out_planes,
                                kernel_size, stride=stride, act_type=act_func, use_se=use_se)
        self.inplanes = out_planes
        return layer

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
            if isinstance(m, (nn.Conv2d)):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.set_data(Tensor(np.random.normal(0, np.sqrt(2. / n),
                                                          m.weight.data.shape).astype("float32")))
                if m.bias is not None:
                    m.bias.set_data(
                        Tensor(np.zeros(m.bias.data.shape, dtype="float32")))
            elif isinstance(m, nn.BatchNorm2d):
                m.gamma.set_data(
                    Tensor(np.ones(m.gamma.data.shape, dtype="float32")))
                m.beta.set_data(
                    Tensor(np.zeros(m.beta.data.shape, dtype="float32")))
            elif isinstance(m, nn.Dense):
                m.weight.set_data(Tensor(np.random.normal(
                    0, 0.01, m.weight.data.shape).astype("float32")))
                if m.bias is not None:
                    m.bias.set_data(
                        Tensor(np.zeros(m.bias.data.shape, dtype="float32")))


def ghostnet(model_name, **kwargs):
    """
    Constructs a GhostNet model
    """
    model_cfgs = {
        "1x": {
            "cfg": [
                # k, exp, c,  se,     nl,  s,
                # stage1
                [3, 16, 16, False, 'relu', 1],
                # stage2
                [3, 48, 24, False, 'relu', 2],
                [3, 72, 24, False, 'relu', 1],
                # stage3
                [5, 72, 40, True, 'relu', 2],
                [5, 120, 40, True, 'relu', 1],
                # stage4
                [3, 240, 80, False, 'relu', 2],
                [3, 200, 80, False, 'relu', 1],
                [3, 184, 80, False, 'relu', 1],
                [3, 184, 80, False, 'relu', 1],
                [3, 480, 112, True, 'relu', 1],
                [3, 672, 112, True, 'relu', 1],
                # stage5
                [5, 672, 160, True, 'relu', 2],
                [5, 960, 160, False, 'relu', 1],
                [5, 960, 160, True, 'relu', 1],
                [5, 960, 160, False, 'relu', 1],
                [5, 960, 160, True, 'relu', 1]],
            "cls_ch_squeeze": 960,
            "cls_ch_expand": 1280,
        },

        "nose_1x": {
            "cfg": [
                # k, exp, c,  se,     nl,  s,
                # stage1
                [3, 16, 16, False, 'relu', 1],
                # stage2
                [3, 48, 24, False, 'relu', 2],
                [3, 72, 24, False, 'relu', 1],
                # stage3
                [5, 72, 40, False, 'relu', 2],
                [5, 120, 40, False, 'relu', 1],
                # stage4
                [3, 240, 80, False, 'relu', 2],
                [3, 200, 80, False, 'relu', 1],
                [3, 184, 80, False, 'relu', 1],
                [3, 184, 80, False, 'relu', 1],
                [3, 480, 112, False, 'relu', 1],
                [3, 672, 112, False, 'relu', 1],
                # stage5
                [5, 672, 160, False, 'relu', 2],
                [5, 960, 160, False, 'relu', 1],
                [5, 960, 160, False, 'relu', 1],
                [5, 960, 160, False, 'relu', 1],
                [5, 960, 160, False, 'relu', 1]],
            "cls_ch_squeeze": 960,
            "cls_ch_expand": 1280,
        }
    }

    return GhostNet(model_cfgs[model_name], **kwargs)


ghostnet_1x = partial(ghostnet, model_name="1x", final_drop=0.8)
ghostnet_nose_1x = partial(ghostnet, model_name="nose_1x", final_drop=0.8)
