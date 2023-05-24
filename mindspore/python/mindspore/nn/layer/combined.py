# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
"""Combined cells."""
from __future__ import absolute_import

from mindspore import nn
from mindspore.ops.primitive import Primitive
import mindspore._checkparam as Validator
from mindspore.nn.layer.normalization import BatchNorm2d, BatchNorm1d
from mindspore.nn.layer.activation import get_activation, LeakyReLU
from mindspore.nn.cell import Cell


__all__ = [
    'Conv2dBnAct',
    'DenseBnAct'
]


class Conv2dBnAct(Cell):
    r"""
    A combination of convolution, Batchnorm, and activation layer.

    This part is a more detailed overview of Conv2d operation.

    Args:
        in_channels (int): The number of input channel :math:`C_{in}`.
        out_channels (int): The number of output channel :math:`C_{out}`.
        kernel_size (Union[int, tuple]): The data type is int or a tuple of 2 integers. Specifies the height
            and width of the 2D convolution window. Single int means the value is for both height and width of
            the kernel. A tuple of 2 ints means the first value is for the height and the other is for the
            width of the kernel.
        stride (int): Specifies stride for all spatial dimensions with the same value. The value of stride must be
            greater than or equal to 1 and lower than any one of the height and width of the `x`. Default: ``1`` .
        pad_mode (str): Specifies padding mode. The optional values are ``"same"`` , ``"valid"`` , ``"pad"`` .
            Default: ``"same"`` .
        padding (int): Implicit paddings on both sides of the `x`. Default: ``0`` .
        dilation (int): Specifies the dilation rate to use for dilated convolution. If set to be :math:`k > 1`,
            there will be :math:`k - 1` pixels skipped for each sampling location. Its value must be greater than
            or equal to 1 and lower than any one of the height and width of the `x`. Default: ``1`` .
        group (int): Splits filter into groups, `in_channels` and `out_channels` must be
            divisible by the number of groups. Default: ``1`` .
        has_bias (bool): Specifies whether the layer uses a bias vector. Default: ``False`` .
        weight_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the convolution kernel.
            It can be a Tensor, a string, an Initializer or a number. When a string is specified,
            values from ``'TruncatedNormal'`` , ``'Normal'`` , ``'Uniform'`` , ``'HeUniform'`` and ``'XavierUniform'``
            distributions as well as constant ``'One'`` and ``'Zero'`` distributions are possible. Alias
            ``'xavier_uniform'`` , ``'he_uniform'`` , ``'ones'`` and ``'zeros'`` are acceptable. Uppercase and
            lowercase are both acceptable. Refer to the values of Initializer for more details. Default: ``'normal'`` .
        bias_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the bias vector. Possible
            Initializer and string are the same as 'weight_init'. Refer to the values of
            Initializer for more details. Default: ``'zeros'`` .
        has_bn (bool): Specifies to used batchnorm or not. Default: ``False`` .
        momentum (float): Momentum for moving average for batchnorm, must be [0, 1]. Default: ``0.997`` .
        eps (float): Term added to the denominator to improve numerical stability for batchnorm, should be greater
            than 0. Default: ``1e-5`` .
        activation (Union[str, Cell, Primitive]): Specifies activation type. The optional values are as following:
            'softmax', 'logsoftmax', 'relu', 'relu6', 'tanh', 'gelu', 'sigmoid',
            'prelu', 'leakyrelu', 'hswish', 'hsigmoid'. Default: ``None`` .
        alpha (float): Slope of the activation function at x < 0 for LeakyReLU. Default: ``0.2`` .
        after_fake(bool): Determine whether there must be a fake quantization operation after Cond2dBnAct.
            Default: ``True`` .

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`. The data type is float32.

    Outputs:
        Tensor of shape :math:`(N, C_{out}, H_{out}, W_{out})`. The data type is float32.

    Raises:
        TypeError: If `in_channels`, `out_channels`, `stride`, `padding` or `dilation` is not an int.
        TypeError: If `has_bias` is not a bool.
        ValueError: If `in_channels` or `out_channels` `stride`, `padding` or `dilation` is less than 1.
        ValueError: If `pad_mode` is not one of 'same', 'valid', 'pad'.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> from mindspore import Tensor, nn
        >>> import numpy as np
        >>> net = nn.Conv2dBnAct(120, 240, 4, has_bn=True, activation='relu')
        >>> x = Tensor(np.ones([1, 120, 1024, 640]), mindspore.float32)
        >>> result = net(x)
        >>> output = result.shape
        >>> print(output)
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
                 has_bn=False,
                 momentum=0.997,
                 eps=1e-5,
                 activation=None,
                 alpha=0.2,
                 after_fake=True):
        """Initialize Conv2dBnAct."""
        super(Conv2dBnAct, self).__init__()

        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              pad_mode=pad_mode,
                              padding=padding,
                              dilation=dilation,
                              group=group,
                              has_bias=has_bias,
                              weight_init=weight_init,
                              bias_init=bias_init)
        self.has_bn = Validator.check_bool(has_bn, "has_bn", self.cls_name)
        self.has_act = activation is not None
        self.after_fake = Validator.check_bool(after_fake, "after_fake", self.cls_name)
        if has_bn:
            self.batchnorm = BatchNorm2d(out_channels, eps, momentum)
        if activation == "leakyrelu":
            self.activation = LeakyReLU(alpha)
        else:
            self.activation = get_activation(activation) if isinstance(activation, str) else activation
            if activation is not None and not isinstance(self.activation, (Cell, Primitive)):
                raise TypeError(f"For '{self.cls_name}', the 'activation' must be str or Cell or Primitive, "
                                f"but got {type(activation).__name__}.")

    def construct(self, x):
        x = self.conv(x)
        if self.has_bn:
            x = self.batchnorm(x)
        if self.has_act:
            x = self.activation(x)
        return x


class DenseBnAct(Cell):
    r"""
    A combination of Dense, Batchnorm, and the activation layer.

    This part is a more detailed overview of Dense op.

    Args:
        in_channels (int): The number of channels in the input space.
        out_channels (int): The number of channels in the output space.
        weight_init (Union[Tensor, str, Initializer, numbers.Number]): The trainable weight_init parameter. The dtype
            is same as `x`. The values of str refer to the function `initializer`. Default: ``'normal'`` .
        bias_init (Union[Tensor, str, Initializer, numbers.Number]): The trainable bias_init parameter. The dtype is
            same as `x`. The values of str refer to the function `initializer`. Default: ``'zeros'`` .
        has_bias (bool): Specifies whether the layer uses a bias vector. Default: ``True`` .
        has_bn (bool): Specifies to use batchnorm or not. Default: ``False`` .
        momentum (float): Momentum for moving average for batchnorm, must be [0, 1]. Default: ``0.9`` .
        eps (float): Term added to the denominator to improve numerical stability for batchnorm, should be greater
            than 0. Default: ``1e-5`` .
        activation (Union[str, Cell, Primitive]): Specifies activation type. The optional values are as following:
            'softmax', 'logsoftmax', 'relu', 'relu6', 'tanh', 'gelu', 'sigmoid',
            'prelu', 'leakyrelu', 'hswish', 'hsigmoid'. Default:  ``None`` .
        alpha (float): Slope of the activation function at x < 0 for LeakyReLU. Default: ``0.2`` .
        after_fake(bool): Determine whether there must be a fake quantization operation after DenseBnAct.
            Default: ``True`` .

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, in\_channels)`. The data type is float32.

    Outputs:
        Tensor of shape :math:`(N, out\_channels)`. The data type is float32.

    Raises:
        TypeError: If `in_channels` or `out_channels` is not an int.
        TypeError: If `has_bias`, `has_bn` or `after_fake` is not a bool.
        TypeError: If `momentum` or `eps` is not a float.
        ValueError: If `momentum` is not in range [0, 1.0].

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> from mindspore import Tensor, nn
        >>> import numpy as np
        >>> net = nn.DenseBnAct(3, 4)
        >>> x = Tensor(np.random.randint(0, 255, [2, 3]), mindspore.float32)
        >>> result = net(x)
        >>> output = result.shape
        >>> print(output)
        (2, 4)
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 weight_init='normal',
                 bias_init='zeros',
                 has_bias=True,
                 has_bn=False,
                 momentum=0.9,
                 eps=1e-5,
                 activation=None,
                 alpha=0.2,
                 after_fake=True):
        """Initialize DenseBnAct."""
        super(DenseBnAct, self).__init__()
        self.dense = nn.Dense(
            in_channels,
            out_channels,
            weight_init,
            bias_init,
            has_bias)
        self.has_bn = Validator.check_bool(has_bn, "has_bn", self.cls_name)
        self.has_act = activation is not None
        self.after_fake = Validator.check_bool(after_fake, "after_fake", self.cls_name)
        if has_bn:
            self.batchnorm = BatchNorm1d(out_channels, eps, momentum)
        if activation == "leakyrelu":
            self.activation = LeakyReLU(alpha)
        else:
            self.activation = get_activation(activation) if isinstance(activation, str) else activation
            if activation is not None and not isinstance(self.activation, (Cell, Primitive)):
                raise TypeError(f"For '{self.cls_name}', the 'activation' must be str or Cell or Primitive, "
                                f"but got {type(activation).__name__}.")

    def construct(self, x):
        x = self.dense(x)
        if self.has_bn:
            x = self.batchnorm(x)
        if self.has_act:
            x = self.activation(x)
        return x
