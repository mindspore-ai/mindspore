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
"""
utils.py
"""

from inspect import isfunction

import mindspore.nn as nn

conv_weight_init = 'HeNormal'


class ConvBlock(nn.Cell):
    """
    Standard convolution block with Batch normalization and activation.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple/list of 2 int
        Convolution window size.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    padding : int, or tuple/list of 2 int, or tuple/list of 4 int
        Padding value for convolution layer.
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    group : int, default 1
        Number of group.
    has_bias : bool, default False
        Whether the layer uses a has_bias vector.
    use_bn : bool, default True
        Whether to use BatchNorm layer.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    activation : function or str or None, default nn.ReLU()
        Activation function or name of activation function.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 dilation=1,
                 group=1,
                 has_bias=False,
                 use_bn=True,
                 bn_eps=1e-5,
                 activation=nn.ReLU()):
        super(ConvBlock, self).__init__()
        self.activate = (activation is not None)
        self.use_bn = use_bn
        self.use_pad = padding

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            pad_mode='pad',
            padding=padding,
            dilation=dilation,
            group=group,
            has_bias=has_bias,
            weight_init=conv_weight_init)
        if self.use_bn:
            self.bn = nn.BatchNorm2d(
                num_features=out_channels,
                momentum=0.9,
                eps=bn_eps)
        if self.activate:
            self.active = get_activation_layer(activation)

    def construct(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.activate:
            x = self.active(x)
        return x


def conv1x1_block(in_channels,
                  out_channels,
                  stride=1,
                  padding=0,
                  group=1,
                  has_bias=False,
                  use_bn=True,
                  bn_eps=1e-5,
                  activation=nn.ReLU()):
    """
    1x1 version of the standard convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    padding : int, or tuple/list of 2 int, or tuple/list of 4 int, default 0
        Padding value for convolution layer.
    group : int, default 1
        Number of group.
    has_bias : bool, default False
        Whether the layer uses a has_bias vector.
    use_bn : bool, default True
        Whether to use BatchNorm layer.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    activation : function or str or None, default nn.ReLU()
        Activation function or name of activation function.
    """
    return ConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        stride=stride,
        padding=padding,
        group=group,
        has_bias=has_bias,
        use_bn=use_bn,
        bn_eps=bn_eps,
        activation=activation)


def conv3x3_block(in_channels,
                  out_channels,
                  stride=1,
                  padding=1,
                  dilation=1,
                  group=1,
                  has_bias=False,
                  use_bn=True,
                  bn_eps=1e-5,
                  activation=nn.ReLU()):
    """
    3x3 version of the standard convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    padding : int, or tuple/list of 2 int, or tuple/list of 4 int, default 1
        Padding value for convolution layer.
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    group : int, default 1
        Number of group.
    has_bias : bool, default False
        Whether the layer uses a has_bias vector.
    use_bn : bool, default True
        Whether to use BatchNorm layer.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    activation : function or str or None, default nn.ReLU()
        Activation function or name of activation function.
    """
    return ConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        dilation=dilation,
        group=group,
        has_bias=has_bias,
        use_bn=use_bn,
        bn_eps=bn_eps,
        activation=activation)



def dwconv3x3_block(in_channels,
                    out_channels,
                    stride=1,
                    padding=1,
                    dilation=1,
                    has_bias=False,
                    bn_eps=1e-5,
                    activation=nn.ReLU()):
    """
    3x3 depthwise version of the standard convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    padding : int, or tuple/list of 2 int, or tuple/list of 4 int, default 1
        Padding value for convolution layer.
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    has_bias : bool, default False
        Whether the layer uses a has_bias vector.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    activation : function or str or None, default nn.ReLU()
        Activation function or name of activation function.
    """
    return dwconv_block(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        dilation=dilation,
        has_bias=has_bias,
        bn_eps=bn_eps,
        activation=activation)



def dwconv5x5_block(in_channels,
                    out_channels,
                    stride=1,
                    padding=2,
                    dilation=1,
                    has_bias=False,
                    bn_eps=1e-5,
                    activation=nn.ReLU()):
    """
    5x5 depthwise version of the standard convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    padding : int, or tuple/list of 2 int, or tuple/list of 4 int, default 2
        Padding value for convolution layer.
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    has_bias : bool, default False
        Whether the layer uses a has_bias vector.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    activation : function or str or None, default nn.ReLU()
        Activation function or name of activation function.
    """
    return dwconv_block(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=5,
        stride=stride,
        padding=padding,
        dilation=dilation,
        has_bias=has_bias,
        bn_eps=bn_eps,
        activation=activation)


def dwconv_block(in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=1,
                 dilation=1,
                 has_bias=False,
                 use_bn=True,
                 bn_eps=1e-5,
                 activation=nn.ReLU()):
    """
    Depthwise version of the standard convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple/list of 2 int
        Convolution window size.
    stride : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    padding : int, or tuple/list of 2 int, or tuple/list of 4 int, default 1
        Padding value for convolution layer.
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    has_bias : bool, default False
        Whether the layer uses a has_bias vector.
    use_bn : bool, default True
        Whether to use BatchNorm layer.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    activation : function or str or None, default nn.ReLU()
        Activation function or name of activation function.
    """
    return ConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        group=out_channels,
        has_bias=has_bias,
        use_bn=use_bn,
        bn_eps=bn_eps,
        activation=activation)


class Swish(nn.Cell):
    """
    Swish activation function from 'Searching for Activation Functions,' https://arxiv.org/abs/1710.05941.
    """
    def __init__(self):
        super(Swish, self).__init__()
        self.sigmoid = nn.Sigmoid()
    def construct(self, x):
        return x * self.sigmoid(x)


class Identity(nn.Cell):
    """
    Identity block.
    """

    def constructor(self, x):
        return x


def get_activation_layer(activation):
    """
    Create activation layer from string/function.

    Parameters:
    ----------
    activation : function, or str, or nn.Cell
        Activation function or name of activation function.

    Returns:
    -------
    nn.Cell
        Activation layer.
    """

    if isfunction(activation):
        active = activation()
    elif isinstance(activation, str):
        if activation == "relu":
            active = nn.ReLU()
        elif activation == "relu6":
            active = nn.ReLU6()
        elif activation == "swish":
            active = Swish()
        elif activation == "hswish":
            active = nn.HSwish()
        elif activation == "sigmoid":
            active = nn.Sigmoid()
        elif activation == "hsigmoid":
            active = nn.HSigmoid()
        elif activation == "identity":
            active = Identity()
        else:
            raise NotImplementedError()
    elif isinstance(activation, nn.Cell):
        active = activation
    else:
        return ValueError()
    return active
