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
"""Aware quantization."""

import numpy as np
import mindspore.nn as nn
import mindspore.common.dtype as mstype
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.common.parameter import Parameter
from mindspore.common.initializer import initializer
from mindspore.common.tensor import Tensor
from mindspore._checkparam import check_int_positive, check_bool, twice
from mindspore.nn.cell import Cell
from mindspore.nn.layer.conv import _Conv
from mindspore.nn.layer.activation import get_activation

__all__ = [
    'FakeQuantWithMinMax',
    'Conv2dBatchNormQuant',
    'Conv2dQuant',
    'DenseQuant',
    'ReLUQuant',
    'ReLU6Quant',
    'HSwishQuant',
    'HSigmoidQuant',
    'TensorAddQuant',
]


class FakeQuantWithMinMax(Cell):
    r"""
    Aware Quantization training op. This OP provide Fake quantization observer function on data with min and max.

    Args:
        min_init (int, list): The dimension of channel or 1(layer). Default: -6.
        max_init (int, list): The dimension of channel or 1(layer). Default: 6.
        num_bits (int): Quantization number bit, support 4 and 8bit. Default: 8.
        ema (bool): Exponential Moving Average algorithm update min and max. Default: False.
        ema_decay (float): Exponential Moving Average algorithm parameter. Default: 0.9999.
        per_channel (bool): Quantization by layer or channel. Default: False.
        channel_size (int): declarate the min and max channel size, Default: 1.
        quant_delay (int): Quantization delay parameters according by global step. Default: 0.
        symmetric (bool): Quantization algorithm use symmetric or not. Default: False.
        narrow_range (bool): Quantization algorithm use narrow range or not. Default: False.

    Inputs:
        - **x** (Tensor) - The input of FakeQuantWithMinMax.

    Outputs:
        Tensor, with the same type and shape as the `x`.

    """

    def __init__(self,
                 min_init=-6,
                 max_init=6,
                 num_bits=8,
                 ema=False,
                 ema_decay=0.999,
                 per_channel=False,
                 channel_size=1,
                 quant_delay=0,
                 symmetric=False,
                 narrow_range=False):
        super(FakeQuantWithMinMax, self).__init__()

        self.min_init = min_init
        self.num_bits = num_bits
        self.max_init = max_init
        self.ema = ema
        self.ema_decay = ema_decay
        self.per_channel = per_channel
        self.channel_size = channel_size
        self.quant_delay = quant_delay
        self.symmetric = symmetric
        self.narrow_range = narrow_range

        if per_channel:
            min_array = np.array([self.min_init for i in range(
                0, self.channel_size)]).astype(np.float32)
            max_array = np.array([self.max_init for i in range(
                0, self.channel_size)]).astype(np.float32)
            self.fake_quant_train = P.FakeQuantWithMinMaxPerChannel(num_bits=self.num_bits,
                                                                    ema=self.ema,
                                                                    ema_decay=self.ema_decay,
                                                                    quant_delay=self.quant_delay,
                                                                    symmetric=self.symmetric,
                                                                    narrow_range=self.narrow_range,
                                                                    training=True)
            self.fake_quant_infer = P.FakeQuantWithMinMaxPerChannel(num_bits=self.num_bits,
                                                                    ema=self.ema,
                                                                    ema_decay=ema_decay,
                                                                    quant_delay=quant_delay,
                                                                    symmetric=self.symmetric,
                                                                    narrow_range=self.narrow_range,
                                                                    training=False)
        else:
            min_array = np.array([min_init]).reshape(1).astype(np.float32)
            max_array = np.array([max_init]).reshape(1).astype(np.float32)
            self.fake_quant_train = P.FakeQuantWithMinMax(num_bits=self.num_bits,
                                                          ema=self.ema,
                                                          ema_decay=self.ema_decay,
                                                          quant_delay=self.quant_delay,
                                                          symmetric=self.symmetric,
                                                          narrow_range=self.narrow_range,
                                                          training=True)
            self.fake_quant_infer = P.FakeQuantWithMinMax(num_bits=self.num_bits,
                                                          ema=self.ema,
                                                          ema_decay=ema_decay,
                                                          quant_delay=quant_delay,
                                                          symmetric=self.symmetric,
                                                          narrow_range=self.narrow_range,
                                                          training=False)

        self.min = Parameter(
            Tensor(min_array), name='quant_min', requires_grad=False)
        self.max = Parameter(
            Tensor(max_array), name='quant_max', requires_grad=False)

    def extend_repr(self):
        s = 'min_init={}, max_init={}, ema={}, ema_decay={},  per_channel={}, channel_size={}, quant_delay={}'.format(
            self.min_init, self.max_init, self.ema, self.ema_decay, self.per_channel, self.channel_size,
            self.quant_delay)
        return s

    def construct(self, x):
        if self.training:
            out = self.fake_quant_train(x, self.min, self.max)
        else:
            out = self.fake_quant_infer(x, self.min, self.max)
        return out


class Conv2dBatchNormQuant(Cell):
    r"""
    2D convolution with BatchNormal op folded layer.

    For a more Detailed overview of Conv2d op.

    Args:
        in_channels (int): The number of input channel :math:`C_{in}`.
        out_channels (int): The number of output channel :math:`C_{out}`.
        kernel_size (Union[int, tuple]): Specifies the height and width of the 2D convolution window.
        stride (int): Specifies stride for all spatial dimensions with the same value.
        pad_mode: (str): Specifies padding mode. The optional values are "same", "valid", "pad". Default: "same".
        padding: (int): Implicit paddings on both sides of the input. Default: 0.
        eps (int): Parameters for BatchNormal. Default: 1e-5.
        momentum (int): Parameters for BatchNormal op. Default: 0.9.
        weight_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the
            convolution kernel. Default: 'None'.
        beta_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the
            beta vector. Default: 'None'.
        gamma_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the
            gamma vector. Default: 'None'.
        mean_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the
            mean vector. Default: 'None'.
        var_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the
            variance vector. Default: 'None'.
        quant_delay (int): Quantization delay parameters according by global step. Default: 0.
        freeze_bn (int): Quantization freeze BatchNormal op according by global step. Default: 100000.
        fake (bool): Conv2dBatchNormQuant Cell add FakeQuantWithMinMax op or not. Default: True.
        num_bits (int): Quantization number bit, support 4 and 8bit. Default: 8.
        per_channel (bool): FakeQuantWithMinMax Parameters. Default: False.
        symmetric (bool): Quantization algorithm use symmetric or not. Default: False.
        narrow_range (bool): Quantization algorithm use narrow range or not. Default: False.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor of shape :math:`(N, C_{out}, H_{out}, W_{out})`.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 pad_mode,
                 padding=0,
                 dilation=1,
                 group=1,
                 eps=1e-5,
                 momentum=0.9,
                 weight_init=None,
                 beta_init=None,
                 gamma_init=None,
                 mean_init=None,
                 var_init=None,
                 quant_delay=0,
                 freeze_bn=100000,
                 fake=True,
                 num_bits=8,
                 per_channel=False,
                 symmetric=False,
                 narrow_range=False):
        super(Conv2dBatchNormQuant, self).__init__()
        self.stride = stride
        self.conv = P.Conv2D(out_channel=out_channels,
                             kernel_size=kernel_size,
                             mode=1,
                             pad_mode=pad_mode,
                             pad=padding,
                             stride=stride,
                             dilation=1,
                             group=group)
        self.fake = fake
        self.freeze_bn = freeze_bn
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        if weight_init is None:
            weight_init = initializer(
                'normal', [out_channels, in_channels // group, *kernel_size])
        self.weight = Parameter(weight_init, name='weight')
        if gamma_init is None:
            gamma_init = initializer('ones', [out_channels])
        self.gamma = Parameter(gamma_init, name='gamma')
        if beta_init is None:
            beta_init = initializer('zeros', [out_channels])
        self.beta = Parameter(beta_init, name='beta')
        if mean_init is None:
            mean_init = initializer('zeros', [out_channels])
        self.moving_mean = Parameter(
            mean_init, name='moving_mean', requires_grad=False)
        if var_init is None:
            var_init = initializer('ones', [out_channels])
        self.moving_variance = Parameter(
            var_init, name='moving_variance', requires_grad=False)

        self.step = Parameter(initializer(
            'normal', [1], dtype=mstype.int32), name='step', requires_grad=False)

        self.fake_quant_weight = nn.FakeQuantWithMinMax(min_init=-6,
                                                        max_init=6,
                                                        ema=False,
                                                        num_bits=num_bits,
                                                        quant_delay=quant_delay,
                                                        per_channel=per_channel,
                                                        channel_size=out_channels,
                                                        symmetric=symmetric,
                                                        narrow_range=narrow_range)

        self.batchnorm_fold_train = P.BatchNormFold(epsilon=eps,
                                                    momentum=momentum,
                                                    is_training=True,
                                                    freeze_bn=freeze_bn)
        self.batchnorm_fold_infer = P.BatchNormFold(epsilon=eps,
                                                    momentum=momentum,
                                                    is_training=False,
                                                    freeze_bn=freeze_bn)
        self.correct_mul = P.CorrectionMul()
        self.relu = P.ReLU()
        self.batchnorm_fold2 = P.BatchNormFold2(freeze_bn=freeze_bn)
        self.batchnorm_fold2_infer = P.BatchNormFold2(freeze_bn=0)
        self.one = Tensor(1, mstype.int32)
        self.assignadd = P.AssignAdd()

    def extend_repr(self):
        s = 'fake={}, freeze_bn={}'.format(self.fake, self.freeze_bn)
        return s

    def construct(self, x):
        if self.training:
            beta = self.beta
            gamma = self.gamma
            gmean = self.moving_mean
            gvar = self.moving_variance
            step = self.step
            out_conv = self.conv(x, self.weight)
            batch_mean, batch_std, running_mean, running_std = self.batchnorm_fold_train(
                out_conv, gmean, gvar, step)
            # BN fold1
            weight = self.correct_mul(self.weight, gamma, running_std)
            if self.fake:
                weight = self.fake_quant_weight(weight)
            out = self.conv(x, weight)
            # BN fold2
            out = self.batchnorm_fold2(
                out, beta, gamma, batch_std, batch_mean, running_std, running_mean, step)
            F.control_depend(out, self.assignadd(self.step, self.one))
        else:
            step = self.step
            out_conv = self.conv(x, self.weight)
            batch_mean, batch_std, running_mean, running_std = self.batchnorm_fold_infer(
                out_conv, self.moving_mean, self.moving_variance, step)
            weight = self.correct_mul(self.weight, self.gamma, running_std)
            if self.fake:
                weight = self.fake_quant_weight(weight)
            out = self.conv(x, weight)
            out = self.batchnorm_fold2_infer(out, self.beta, self.gamma, batch_std, batch_mean,
                                             running_std, running_mean, step)
        return out


class Conv2dQuant(_Conv):
    r"""
    2D convolution with fake quant op layer.

    For a more Detailed overview of Conv2d op.

    Args:
        in_channels (int): The number of input channel :math:`C_{in}`.
        out_channels (int): The number of output channel :math:`C_{out}`.
        kernel_size (Union[int, tuple]): Specifies the height and width of the 2D convolution window.
        stride (int): Specifies stride for all spatial dimensions with the same value. Default: 1.
        pad_mode: (str): Specifies padding mode. The optional values are "same", "valid", "pad". Default: "same".
        padding: (int): Implicit paddings on both sides of the input. Default: 0.
        dilation (int): Specifying the dilation rate to use for dilated convolution. Default: 1.
        group (int): Split filter into groups, `in_ channels` and `out_channels` should be
            divisible by the number of groups. Default: 1.
        has_bias (bool): Specifies whether the layer uses a bias vector. Default: False.
        weight_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the convolution kernel.
            Default: 'normal'.
        bias_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the bias vector. Default: 'zeros'.
        quant_delay (int): Quantization delay parameters according by global step. Default: 0.
        num_bits (int): Quantization number bit, support 4 and 8bit. Default: 8.
        per_channel (bool): FakeQuantWithMinMax Parameters. Default: False.
        symmetric (bool): Quantization algorithm use symmetric or not. Default: False.
        narrow_range (bool): Quantization algorithm use narrow range or not. Default: False.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor of shape :math:`(N, C_{out}, H_{out}, W_{out})`.

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
                 quant_delay=0,
                 num_bits=8,
                 per_channel=False,
                 symmetric=False,
                 narrow_range=False):
        kernel_size = twice(kernel_size)
        super(Conv2dQuant, self).__init__(in_channels, out_channels, kernel_size, stride, pad_mode, padding, dilation,
                                          group, has_bias, weight_init, bias_init)
        self.conv2d = P.Conv2D(out_channel=self.out_channels, kernel_size=self.kernel_size, mode=1,
                               pad_mode=self.pad_mode, pad=self.padding, stride=self.stride, dilation=self.dilation,
                               group=self.group)
        self.bias_add = P.BiasAdd()
        if pad_mode not in ('valid', 'same', 'pad'):
            raise ValueError('Attr \'pad_mode\' of \'Conv2d\' Op passed '
                             + str(pad_mode) + ', should be one of values in \'valid\', \'same\', \'pad\'.')
        self.fake_quant_weight = nn.FakeQuantWithMinMax(min_init=-6,
                                                        max_init=6,
                                                        ema=False,
                                                        num_bits=num_bits,
                                                        quant_delay=quant_delay,
                                                        per_channel=per_channel,
                                                        channel_size=out_channels,
                                                        symmetric=symmetric,
                                                        narrow_range=narrow_range)

    def construct(self, x):
        weight_q = self.fake_quant_weight(self.weight)
        out = self.conv2d(x, weight_q)
        if self.has_bias:
            return self.bias_add(out, self.bias)
        return out


class DenseQuant(Cell):
    r"""
    The fully connected layer with fake quant op.

    For a more Detailed overview of Dense op.

    Args:
        in_channels (int): The dimension of the input space.
        out_channels (int): The dimension of the output space.
        weight_init (Union[Tensor, str, Initializer, numbers.Number]): The trainable weight_init parameter. The dtype
            is same as input x. The values of str refer to the function `initializer`. Default: 'normal'.
        bias_init (Union[Tensor, str, Initializer, numbers.Number]): The trainable bias_init parameter. The dtype is
            same as input x. The values of str refer to the function `initializer`. Default: 'zeros'.
        has_bias (bool): Specifies whether the layer uses a bias vector. Default: True.
        activation (str): Regularizer function applied to the output of the layer, eg. 'relu'. Default: None.
        num_bits (int): Quantization number bit, support 4 and 8bit. Default: 8.
        quant_delay (int): Quantization delay parameters according by global step. Default: 0.
        per_channel (bool): FakeQuantWithMinMax Parameters. Default: False.
        symmetric (bool): Quantization algorithm use symmetric or not. Default: False.
        narrow_range (bool): Quantization algorithm use narrow range or not. Default: False.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor of shape :math:`(N, C_{out}, H_{out}, W_{out})`.
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            weight_init='normal',
            bias_init='zeros',
            has_bias=True,
            activation=None,
            num_bits=8,
            quant_delay=0,
            per_channel=False,
            symmetric=False,
            narrow_range=False):
        super(DenseQuant, self).__init__()
        self.in_channels = check_int_positive(in_channels)
        self.out_channels = check_int_positive(out_channels)
        self.has_bias = check_bool(has_bias)

        if isinstance(weight_init, Tensor):
            if weight_init.dim() != 2 or weight_init.shape()[0] != out_channels or \
                    weight_init.shape()[1] != in_channels:
                raise ValueError("weight_init shape error")

        self.weight = Parameter(initializer(
            weight_init, [out_channels, in_channels]), name="weight")

        if self.has_bias:
            if isinstance(bias_init, Tensor):
                if bias_init.dim() != 1 or bias_init.shape()[0] != out_channels:
                    raise ValueError("bias_init shape error")

            self.bias = Parameter(initializer(
                bias_init, [out_channels]), name="bias")

        self.matmul = P.MatMul(transpose_b=True)
        self.bias_add = P.BiasAdd()

        self.activation = get_activation(activation)
        self.activation_flag = self.activation is not None
        self.fake_quant_weight = nn.FakeQuantWithMinMax(min_init=-6,
                                                        max_init=6,
                                                        ema=False,
                                                        num_bits=num_bits,
                                                        quant_delay=quant_delay,
                                                        per_channel=per_channel,
                                                        channel_size=out_channels,
                                                        symmetric=symmetric,
                                                        narrow_range=narrow_range)

    def construct(self, x):
        """Use operators to construct to Dense layer."""
        output = self.fake_quant_weight(self.weight)
        output = self.matmul(x, output)
        if self.has_bias:
            output = self.bias_add(output, self.bias)
        if self.activation_flag:
            return self.activation(output)
        return output

    def extend_repr(self):
        """A pretty print for Dense layer."""
        str_info = 'in_channels={}, out_channels={}, weight={}, has_bias={}'.format(
            self.in_channels, self.out_channels, self.weight, self.has_bias)
        if self.has_bias:
            str_info = str_info + ', bias={}'.format(self.bias)
        if self.activation_flag:
            str_info = str_info + ', activation={}'.format(self.activation)

        return str_info


class ReLUQuant(Cell):
    r"""
    ReLUQuant activation function. Add Fake Quant OP after Relu OP.

    For a more Detailed overview of ReLU op.

    Args:
        num_bits (int): Quantization number bit, support 4 and 8bit. Default: 8.
        quant_delay (int): Quantization delay parameters according by global step. Default: 0.
        symmetric (bool): Quantization algorithm use symmetric or not. Default: False.
        narrow_range (bool): Quantization algorithm use narrow range or not. Default: False.

    Inputs:
        - **x** (Tensor) - The input of ReLUQuant.

    Outputs:
        Tensor, with the same type and shape as the `x`.

    """

    def __init__(self,
                 num_bits=8,
                 quant_delay=0,
                 symmetric=False,
                 narrow_range=False):
        super(ReLUQuant, self).__init__()
        self.fake_quant_act = nn.FakeQuantWithMinMax(min_init=0,
                                                     max_init=6,
                                                     num_bits=num_bits,
                                                     quant_delay=quant_delay,
                                                     ema=True,
                                                     symmetric=symmetric,
                                                     narrow_range=narrow_range)
        self.relu = P.ReLU()

    def construct(self, x):
        x = self.relu(x)
        x = self.fake_quant_act(x)
        return x


class ReLU6Quant(Cell):
    r"""
    ReLU6Quant activation function.

    Add Fake Quant OP after Relu6. Not Recommand to used these cell for Fake Quant Op
    Will climp the max range of the activation and the relu6 do the same operation.
    For a more Detailed overview of ReLU6 op.

    Args:
        num_bits (int): Quantization number bit, support 4 and 8bit. Default: 8.
        quant_delay (int): Quantization delay parameters according by global step. Default: 0.
        symmetric (bool): Quantization algorithm use symmetric or not. Default: False.
        narrow_range (bool): Quantization algorithm use narrow range or not. Default: False.

    Inputs:
        - **x** (Tensor) - The input of ReLU6Quant.

    Outputs:
        Tensor, with the same type and shape as the `x`.

    """

    def __init__(self, num_bits=8, quant_delay=0, symmetric=False,
                 narrow_range=False):
        super(ReLU6Quant, self).__init__()
        self.fake_quant_act = nn.FakeQuantWithMinMax(min_init=0,
                                                     max_init=6,
                                                     num_bits=num_bits,
                                                     quant_delay=quant_delay,
                                                     ema=True,
                                                     symmetric=symmetric,
                                                     narrow_range=narrow_range)
        self.relu6 = P.ReLU6()

    def construct(self, x):
        x = self.relu6(x)
        x = self.fake_quant_act(x)
        return x


class HSwishQuant(Cell):
    r"""
    HSwishQuant activation function. Add Fake Quant OP after HSwish OP.

    For a more Detailed overview of HSwish op.

    Args:
        num_bits (int): Quantization number bit, support 4 and 8bit. Default: 8.
        quant_delay (int): Quantization delay parameters according by global step. Default: 0.
        symmetric (bool): Quantization algorithm use symmetric or not. Default: False.
        narrow_range (bool): Quantization algorithm use narrow range or not. Default: False.

    Inputs:
        - **x** (Tensor) - The input of HSwishQuant.

    Outputs:
        Tensor, with the same type and shape as the `x`.

    """

    def __init__(self,
                 num_bits=8,
                 quant_delay=0,
                 symmetric=False,
                 narrow_range=False):
        super(HSwishQuant, self).__init__()
        self.fake_quant_act_before = nn.FakeQuantWithMinMax(min_init=0,
                                                            max_init=6,
                                                            num_bits=num_bits,
                                                            quant_delay=quant_delay,
                                                            ema=True,
                                                            symmetric=symmetric,
                                                            narrow_range=narrow_range)
        self.fake_quant_act_after = nn.FakeQuantWithMinMax(min_init=0,
                                                           max_init=6,
                                                           num_bits=num_bits,
                                                           quant_delay=quant_delay,
                                                           ema=True,
                                                           symmetric=symmetric,
                                                           narrow_range=narrow_range)
        self.act = P.HSwish()

    def construct(self, x):
        x = self.fake_quant_act_before(x)
        x = self.act(x)
        x = self.fake_quant_act_after(x)
        return x


class HSigmoidQuant(Cell):
    r"""
    HSigmoidQuant activation function. Add Fake Quant OP before and after HSigmoid OP.

    For a more Detailed overview of HSigmoid op.

    Args:
        num_bits (int): Quantization number bit, support 4 and 8bit. Default: 8.
        quant_delay (int): Quantization delay parameters according by global step. Default: 0.
        symmetric (bool): Quantization algorithm use symmetric or not. Default: False.
        narrow_range (bool): Quantization algorithm use narrow range or not. Default: False.

    Inputs:
        - **x** (Tensor) - The input of HSigmoidQuant.

    Outputs:
        Tensor, with the same type and shape as the `x`.

    """

    def __init__(self,
                 num_bits=8,
                 quant_delay=0,
                 symmetric=False,
                 narrow_range=False):
        super(HSigmoidQuant, self).__init__()
        self.fake_quant_act_before = nn.FakeQuantWithMinMax(min_init=0,
                                                            max_init=6,
                                                            num_bits=num_bits,
                                                            quant_delay=quant_delay,
                                                            ema=True,
                                                            symmetric=symmetric,
                                                            narrow_range=narrow_range)
        self.fake_quant_act_after = nn.FakeQuantWithMinMax(min_init=0,
                                                           max_init=6,
                                                           num_bits=num_bits,
                                                           quant_delay=quant_delay,
                                                           ema=True,
                                                           symmetric=symmetric,
                                                           narrow_range=narrow_range)
        self.act = P.HSigmoid()

    def construct(self, x):
        x = self.fake_quant_act_before(x)
        x = self.act(x)
        x = self.fake_quant_act_after(x)
        return x


class TensorAddQuant(Cell):
    r"""
    Add Fake Quant OP after TensorAdd OP.

    For a more Detailed overview of TensorAdd op.

    Args:
        num_bits (int): Quantization number bit, support 4 and 8bit. Default: 8.
        quant_delay (int): Quantization delay parameters according by global step. Default: 0.
        symmetric (bool): Quantization algorithm use symmetric or not. Default: False.
        narrow_range (bool): Quantization algorithm use narrow range or not. Default: False.

    Inputs:
        - **x** (Tensor) - The input of TensorAddQuant.

    Outputs:
        Tensor, with the same type and shape as the `x`.

    """

    def __init__(self,
                 num_bits=8,
                 quant_delay=0,
                 symmetric=False,
                 narrow_range=False):
        super(TensorAddQuant, self).__init__()
        self.fake_quant_act = nn.FakeQuantWithMinMax(min_init=-6,
                                                     max_init=6,
                                                     num_bits=num_bits,
                                                     quant_delay=quant_delay,
                                                     ema=True,
                                                     symmetric=symmetric,
                                                     narrow_range=narrow_range)
        self.add = P.TensorAdd()

    def construct(self, x1, x2):
        x = self.add(x1, x2)
        x = self.fake_quant_act(x)
        return x
