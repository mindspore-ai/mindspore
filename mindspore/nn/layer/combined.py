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
"""Use combination of Conv, Dense, Relu, Batchnorm."""

from .normalization import BatchNorm2d
from .activation import get_activation
from ..cell import Cell
from . import conv, basic
from ..._checkparam import ParamValidator as validator


__all__ = ['Conv2d', 'Dense']

class Conv2d(Cell):
    r"""
    A combination of convolution, Batchnorm, activation layer.

    For a more Detailed overview of Conv2d op.

    Args:
        in_channels (int): The number of input channel :math:`C_{in}`.
        out_channels (int): The number of output channel :math:`C_{out}`.
        kernel_size (Union[int, tuple]): The data type is int or tuple with 2 integers. Specifies the height
            and width of the 2D convolution window. Single int means the value if for both height and width of
            the kernel. A tuple of 2 ints means the first value is for the height and the other is for the
            width of the kernel.
        stride (int): Specifies stride for all spatial dimensions with the same value. Value of stride should be
            greater or equal to 1 but bounded by the height and width of the input. Default: 1.
        pad_mode (str): Specifies padding mode. The optional values are "same", "valid", "pad". Default: "same".
        padding (int): Implicit paddings on both sides of the input. Default: 0.
        dilation (int): Specifying the dilation rate to use for dilated convolution. If set to be :math:`k > 1`,
            there will be :math:`k - 1` pixels skipped for each sampling location. Its value should be greater
            or equal to 1 and bounded by the height and width of the input. Default: 1.
        group (int): Split filter into groups, `in_ channels` and `out_channels` should be
            divisible by the number of groups. Default: 1.
        has_bias (bool): Specifies whether the layer uses a bias vector. Default: False.
        weight_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the convolution kernel.
            It can be a Tensor, a string, an Initializer or a numbers.Number. When a string is specified,
            values from 'TruncatedNormal', 'Normal', 'Uniform', 'HeUniform' and 'XavierUniform' distributions as well
            as constant 'One' and 'Zero' distributions are possible. Alias 'xavier_uniform', 'he_uniform', 'ones'
            and 'zeros' are acceptable. Uppercase and lowercase are both acceptable. Refer to the values of
            Initializer for more details. Default: 'normal'.
        bias_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the bias vector. Possible
            Initializer and string are the same as 'weight_init'. Refer to the values of
            Initializer for more details. Default: 'zeros'.
        batchnorm (bool): Specifies to used batchnorm or not. Default: None.
        activation (string): Specifies activation type. The optional values are as following:
            'softmax', 'logsoftmax', 'relu', 'relu6', 'tanh', 'gelu', 'sigmoid',
            'prelu', 'leakyrelu', 'hswish', 'hsigmoid'. Default: None.

    Inputs:
        - **input** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor of shape :math:`(N, C_{out}, H_{out}, W_{out})`.

    Examples:
        >>> net = combined.Conv2d(120, 240, 4, batchnorm=True, activation='ReLU')
        >>> input = Tensor(np.ones([1, 120, 1024, 640]), mindspore.float32)
        >>> net(input).shape()
        (1, 240, 1024, 640)
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 pad_mode='same',
                 padding=0,
                 dilation=1,
                 group=1,
                 has_bias=False,
                 weight_init='normal',
                 bias_init='zeros',
                 batchnorm=None,
                 activation=None):
        super(Conv2d, self).__init__()
        self.conv = conv.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            pad_mode,
            padding,
            dilation,
            group,
            has_bias,
            weight_init,
            bias_init)
        self.has_bn = batchnorm is not None
        self.has_act = activation is not None
        self.batchnorm = batchnorm
        if batchnorm is True:
            self.batchnorm = BatchNorm2d(out_channels)
        elif batchnorm is not None:
            validator.check_isinstance('batchnorm', batchnorm, (BatchNorm2d,))
        self.activation = get_activation(activation)

    def construct(self, x):
        x = self.conv(x)
        if self.has_bn:
            x = self.batchnorm(x)
        if self.has_act:
            x = self.activation(x)
        return x


class Dense(Cell):
    r"""
    A combination of Dense, Batchnorm, activation layer.

    For a more Detailed overview of Dense op.

    Args:
        in_channels (int): The number of channels in the input space.
        out_channels (int): The number of channels in the output space.
        weight_init (Union[Tensor, str, Initializer, numbers.Number]): The trainable weight_init parameter. The dtype
            is same as input x. The values of str refer to the function `initializer`. Default: 'normal'.
        bias_init (Union[Tensor, str, Initializer, numbers.Number]): The trainable bias_init parameter. The dtype is
            same as input x. The values of str refer to the function `initializer`. Default: 'zeros'.
        has_bias (bool): Specifies whether the layer uses a bias vector. Default: True.
        activation (str): Regularizer function applied to the output of the layer, eg. 'relu'. Default: None.
        batchnorm (bool): Specifies to used batchnorm or not. Default: None.
        activation (string): Specifies activation type. The optional values are as following:
            'softmax', 'logsoftmax', 'relu', 'relu6', 'tanh', 'gelu', 'sigmoid',
            'prelu', 'leakyrelu', 'hswish', 'hsigmoid'. Default: None.

    Inputs:
        - **input** (Tensor) - Tensor of shape :math:`(N, in\_channels)`.

    Outputs:
        Tensor of shape :math:`(N, out\_channels)`.

    Examples:
        >>> net = nn.Dense(3, 4)
        >>> input = Tensor(np.random.randint(0, 255, [2, 3]), mindspore.float32)
        >>> net(input)
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 weight_init='normal',
                 bias_init='zeros',
                 has_bias=True,
                 batchnorm=None,
                 activation=None):
        super(Dense, self).__init__()
        self.dense = basic.Dense(
            in_channels,
            out_channels,
            weight_init,
            bias_init,
            has_bias)
        self.has_bn = batchnorm is not None
        self.has_act = activation is not None
        if batchnorm is True:
            self.batchnorm = BatchNorm2d(out_channels)
        elif batchnorm is not None:
            validator.check_isinstance('batchnorm', batchnorm, (BatchNorm2d,))
        self.activation = get_activation(activation)

    def construct(self, x):
        x = self.dense(x)
        if self.has_bn:
            x = self.batchnorm(x)
        if self.has_act:
            x = self.activation(x)
        return x
