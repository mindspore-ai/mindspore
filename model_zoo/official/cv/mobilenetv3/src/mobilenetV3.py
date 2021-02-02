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
"""MobileNetV3 model define"""
from functools import partial
import numpy as np
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore import Tensor


__all__ = ['mobilenet_v3_large',
           'mobilenet_v3_small']


def _make_divisible(x, divisor=8):
    return int(np.ceil(x * 1. / divisor) * divisor)


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
            self.act = nn.HSigmoid()
        elif act_func in ('hswish', 'hard_swish'):
            self.act = nn.HSwish()
        else:
            raise NotImplementedError

    def construct(self, x):
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
        x = self.mean(x, (2, 3))
        return x


class SE(nn.Cell):
    """
    SE warpper definition.

    Args:
        num_out (int): Numbers of output channels.
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
        self.conv1 = nn.Conv2d(in_channels=num_out, out_channels=num_mid,
                               kernel_size=1, has_bias=True, pad_mode='pad')
        self.act1 = Activation('relu')
        self.conv2 = nn.Conv2d(in_channels=num_mid, out_channels=num_out,
                               kernel_size=1, has_bias=True, pad_mode='pad')
        self.act2 = Activation('hsigmoid')
        self.mul = P.Mul()

    def construct(self, x):
        out = self.pool(x)
        out = self.conv1(out)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.act2(out)
        out = self.mul(x, out)
        return out


class Unit(nn.Cell):
    """
    Unit warpper definition.

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
        >>> Unit(3, 3)
    """

    def __init__(self, num_in, num_out, kernel_size=1, stride=1, padding=0, num_groups=1,
                 use_act=True, act_type='relu'):
        super(Unit, self).__init__()
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
        out = self.conv(x)
        out = self.bn(out)
        if self.use_act:
            out = self.act(out)
        return out


class ResUnit(nn.Cell):
    """
    ResUnit warpper definition.

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
        >>> ResUnit(16, 3, 1, 1)
    """
    def __init__(self, num_in, num_mid, num_out, kernel_size, stride=1, act_type='relu', use_se=False):
        super(ResUnit, self).__init__()
        self.use_se = use_se
        self.first_conv = (num_out != num_mid)
        self.use_short_cut_conv = True

        if self.first_conv:
            self.expand = Unit(num_in, num_mid, kernel_size=1,
                               stride=1, padding=0, act_type=act_type)
        else:
            self.expand = None
        self.conv1 = Unit(num_mid, num_mid, kernel_size=kernel_size, stride=stride,
                          padding=self._get_pad(kernel_size), act_type=act_type, num_groups=num_mid)
        if use_se:
            self.se = SE(num_mid)
        self.conv2 = Unit(num_mid, num_out, kernel_size=1, stride=1,
                          padding=0, act_type=act_type, use_act=False)
        if num_in != num_out or stride != 1:
            self.use_short_cut_conv = False
        self.add = P.Add() if self.use_short_cut_conv else None

    def construct(self, x):
        """construct"""
        if self.first_conv:
            out = self.expand(x)
        else:
            out = x
        out = self.conv1(out)
        if self.use_se:
            out = self.se(out)
        out = self.conv2(out)
        if self.use_short_cut_conv:
            out = self.add(x, out)
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


class MobileNetV3(nn.Cell):
    """
    MobileNetV3 architecture.

    Args:
        model_cfgs (Cell): number of classes.
        num_classes (int): Output number classes.
        multiplier (int): Channels multiplier for round to 8/16 and others. Default is 1.
        final_drop (float): Dropout number.
        round_nearest (list): Channel round to . Default is 8.
    Returns:
        Tensor, output tensor.

    Examples:
        >>> MobileNetV3(num_classes=1000)
    """

    def __init__(self, model_cfgs, num_classes=1000, multiplier=1., final_drop=0.,
                 round_nearest=8, include_top=True, activation="None"):
        super(MobileNetV3, self).__init__()
        self.cfgs = model_cfgs['cfg']
        self.inplanes = 16
        self.features = []
        first_conv_in_channel = 3
        first_conv_out_channel = _make_divisible(multiplier * self.inplanes)

        self.features.append(nn.Conv2d(in_channels=first_conv_in_channel,
                                       out_channels=first_conv_out_channel,
                                       kernel_size=3, padding=1, stride=2,
                                       has_bias=False, pad_mode='pad'))
        self.features.append(nn.BatchNorm2d(first_conv_out_channel))
        self.features.append(Activation('hswish'))
        for layer_cfg in self.cfgs:
            self.features.append(self._make_layer(kernel_size=layer_cfg[0],
                                                  exp_ch=_make_divisible(multiplier * layer_cfg[1]),
                                                  out_channel=_make_divisible(multiplier * layer_cfg[2]),
                                                  use_se=layer_cfg[3],
                                                  act_func=layer_cfg[4],
                                                  stride=layer_cfg[5]))
        output_channel = _make_divisible(multiplier * model_cfgs["cls_ch_squeeze"])
        self.features.append(nn.Conv2d(in_channels=_make_divisible(multiplier * self.cfgs[-1][2]),
                                       out_channels=output_channel,
                                       kernel_size=1, padding=0, stride=1,
                                       has_bias=False, pad_mode='pad'))
        self.features.append(nn.BatchNorm2d(output_channel))
        self.features.append(Activation('hswish'))
        self.features.append(GlobalAvgPooling(keep_dims=True))
        self.features.append(nn.Conv2d(in_channels=output_channel,
                                       out_channels=model_cfgs['cls_ch_expand'],
                                       kernel_size=1, padding=0, stride=1,
                                       has_bias=False, pad_mode='pad'))
        self.features.append(Activation('hswish'))
        if final_drop > 0:
            self.features.append((nn.Dropout(final_drop)))

        # make it nn.CellList
        self.features = nn.SequentialCell(self.features)
        self.include_top = include_top
        self.need_activation = False
        if self.include_top:
            self.output = nn.Conv2d(in_channels=model_cfgs['cls_ch_expand'],
                                    out_channels=num_classes,
                                    kernel_size=1, has_bias=True, pad_mode='pad')
            self.squeeze = P.Squeeze(axis=(2, 3))
            if activation != "None":
                self.need_activation = True
                if activation == "Sigmoid":
                    self.activation = P.Sigmoid()
                elif activation == "Softmax":
                    self.activation = P.Softmax()
                else:
                    raise NotImplementedError(f"The activation {activation} not in [Sigmoid, Softmax].")

        self._initialize_weights()

    def construct(self, x):
        x = self.features(x)
        if self.include_top:
            x = self.output(x)
            x = self.squeeze(x)
            if self.need_activation:
                x = self.activation(x)
        return x


    def _make_layer(self, kernel_size, exp_ch, out_channel, use_se, act_func, stride=1):
        mid_planes = exp_ch
        out_planes = out_channel

        layer = ResUnit(self.inplanes, mid_planes, out_planes,
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


def mobilenet_v3(model_name, **kwargs):
    """
    Constructs a MobileNet V2 model
    """
    model_cfgs = {
        "large": {
            "cfg": [
                # k, exp, c,  se,     nl,  s,
                [3, 16, 16, False, 'relu', 1],
                [3, 64, 24, False, 'relu', 2],
                [3, 72, 24, False, 'relu', 1],
                [5, 72, 40, True, 'relu', 2],
                [5, 120, 40, True, 'relu', 1],
                [5, 120, 40, True, 'relu', 1],
                [3, 240, 80, False, 'hswish', 2],
                [3, 200, 80, False, 'hswish', 1],
                [3, 184, 80, False, 'hswish', 1],
                [3, 184, 80, False, 'hswish', 1],
                [3, 480, 112, True, 'hswish', 1],
                [3, 672, 112, True, 'hswish', 1],
                [5, 672, 160, True, 'hswish', 2],
                [5, 960, 160, True, 'hswish', 1],
                [5, 960, 160, True, 'hswish', 1]],
            "cls_ch_squeeze": 960,
            "cls_ch_expand": 1280,
        },
        "small": {
            "cfg": [
                # k, exp, c,  se,     nl,  s,
                [3, 16, 16, True, 'relu', 2],
                [3, 72, 24, False, 'relu', 2],
                [3, 88, 24, False, 'relu', 1],
                [5, 96, 40, True, 'hswish', 2],
                [5, 240, 40, True, 'hswish', 1],
                [5, 240, 40, True, 'hswish', 1],
                [5, 120, 48, True, 'hswish', 1],
                [5, 144, 48, True, 'hswish', 1],
                [5, 288, 96, True, 'hswish', 2],
                [5, 576, 96, True, 'hswish', 1],
                [5, 576, 96, True, 'hswish', 1]],
            "cls_ch_squeeze": 576,
            "cls_ch_expand": 1280,
        }
    }
    return MobileNetV3(model_cfgs[model_name], **kwargs)


mobilenet_v3_large = partial(mobilenet_v3, model_name="large")
mobilenet_v3_small = partial(mobilenet_v3, model_name="small")
