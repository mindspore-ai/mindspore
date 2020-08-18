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
"""Quantization aware training."""

from functools import partial
import numpy as np

from mindspore import nn
import mindspore.common.dtype as mstype
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.common.parameter import Parameter
from mindspore.common.initializer import initializer
from mindspore.common.tensor import Tensor
from mindspore._checkparam import check_int_positive, check_bool, twice
from mindspore._checkparam import Rel
import mindspore.context as context

from .normalization import BatchNorm2d, BatchNorm1d
from .activation import get_activation, ReLU, LeakyReLU
from ..cell import Cell
from . import conv, basic
from ..._checkparam import ParamValidator as validator
from ...ops.operations import _quant_ops as Q

__all__ = [
    'Conv2dBnAct',
    'DenseBnAct',
    'FakeQuantWithMinMax',
    'Conv2dBnFoldQuant',
    'Conv2dBnWithoutFoldQuant',
    'Conv2dQuant',
    'DenseQuant',
    'ActQuant',
    'LeakyReLUQuant',
    'HSwishQuant',
    'HSigmoidQuant',
    'TensorAddQuant',
    'MulQuant',
]


class Conv2dBnAct(Cell):
    r"""
    A combination of convolution, Batchnorm, activation layer.

    This part is a more detailed overview of Conv2d op.

    Args:
        in_channels (int): The number of input channel :math:`C_{in}`.
        out_channels (int): The number of output channel :math:`C_{out}`.
        kernel_size (Union[int, tuple]): The data type is int or a tuple of 2 integers. Specifies the height
            and width of the 2D convolution window. Single int means the value is for both height and width of
            the kernel. A tuple of 2 ints means the first value is for the height and the other is for the
            width of the kernel.
        stride (int): Specifies stride for all spatial dimensions with the same value. The value of stride should be
            greater than or equal to 1 and lower than any one of the height and width of the input. Default: 1.
        pad_mode (str): Specifies padding mode. The optional values are "same", "valid", "pad". Default: "same".
        padding (int): Implicit paddings on both sides of the input. Default: 0.
        dilation (int): Specifying the dilation rate to use for dilated convolution. If set to be :math:`k > 1`,
            there will be :math:`k - 1` pixels skipped for each sampling location. Its value should be greater than
            or equal to 1 and lower than any one of the height and width of the input. Default: 1.
        group (int): Split filter into groups, `in_ channels` and `out_channels` should be
            divisible by the number of groups. Default: 1.
        has_bias (bool): Specifies whether the layer uses a bias vector. Default: False.
        weight_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the convolution kernel.
            It can be a Tensor, a string, an Initializer or a number. When a string is specified,
            values from 'TruncatedNormal', 'Normal', 'Uniform', 'HeUniform' and 'XavierUniform' distributions as well
            as constant 'One' and 'Zero' distributions are possible. Alias 'xavier_uniform', 'he_uniform', 'ones'
            and 'zeros' are acceptable. Uppercase and lowercase are both acceptable. Refer to the values of
            Initializer for more details. Default: 'normal'.
        bias_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the bias vector. Possible
            Initializer and string are the same as 'weight_init'. Refer to the values of
            Initializer for more details. Default: 'zeros'.
        has_bn (bool): Specifies to used batchnorm or not. Default: False.
        activation (Cell): Specifies activation type. The optional values are as following:
            'softmax', 'logsoftmax', 'relu', 'relu6', 'tanh', 'gelu', 'sigmoid',
            'prelu', 'leakyrelu', 'hswish', 'hsigmoid'. Default: None.

    Inputs:
        - **input** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor of shape :math:`(N, C_{out}, H_{out}, W_{out})`.

    Examples:
        >>> net = Conv2dBnAct(120, 240, 4, has_bn=True, activation='ReLU')
        >>> input = Tensor(np.ones([1, 120, 1024, 640]), mindspore.float32)
        >>> net(input).shape
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
                 momentum=0.9,
                 eps=1e-5,
                 activation=None,
                 alpha=0.2,
                 after_fake=True):
        super(Conv2dBnAct, self).__init__()

        if context.get_context('device_target') == "Ascend" and group > 1:
            self.conv = conv.DepthwiseConv2d(in_channels,
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
        else:
            self.conv = conv.Conv2d(in_channels,
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

        self.has_bn = validator.check_bool("has_bn", has_bn)
        self.has_act = activation is not None
        self.after_fake = after_fake
        if has_bn:
            self.batchnorm = BatchNorm2d(out_channels, eps, momentum)
        if activation == "leakyrelu":
            self.activation = LeakyReLU(alpha)
        else:
            self.activation = get_activation(activation)

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
            is same as input x. The values of str refer to the function `initializer`. Default: 'normal'.
        bias_init (Union[Tensor, str, Initializer, numbers.Number]): The trainable bias_init parameter. The dtype is
            same as input x. The values of str refer to the function `initializer`. Default: 'zeros'.
        has_bias (bool): Specifies whether the layer uses a bias vector. Default: True.
        activation (Cell): The regularization function applied to the output of the layer, eg. 'ReLU'. Default: None.
        has_bn (bool): Specifies to use batchnorm or not. Default: False.
        activation (string): Specifies activation type. The optional values are as following:
            'Softmax', 'LogSoftmax', 'ReLU', 'ReLU6', 'Tanh', 'GELU', 'Sigmoid',
            'PReLU', 'LeakyReLU', 'h-Swish', and 'h-Sigmoid'. Default: None.

    Inputs:
        - **input** (Tensor) - Tensor of shape :math:`(N, in\_channels)`.

    Outputs:
        Tensor of shape :math:`(N, out\_channels)`.

    Examples:
        >>> net = nn.DenseBnAct(3, 4)
        >>> input = Tensor(np.random.randint(0, 255, [2, 3]), mindspore.float32)
        >>> net(input)
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 weight_init='normal',
                 bias_init='zeros',
                 has_bias=True,
                 has_bn=False,
                 activation=None,
                 after_fake=True):
        super(DenseBnAct, self).__init__()
        self.dense = basic.Dense(
            in_channels,
            out_channels,
            weight_init,
            bias_init,
            has_bias)
        self.has_bn = validator.check_bool("has_bn", has_bn)
        self.has_act = activation is not None
        self.after_fake = after_fake
        if has_bn:
            self.batchnorm = BatchNorm1d(out_channels)
        self.activation = get_activation(activation)

    def construct(self, x):
        x = self.dense(x)
        if self.has_bn:
            x = self.batchnorm(x)
        if self.has_act:
            x = self.activation(x)
        return x


class BatchNormFoldCell(Cell):
    """
    Batch normalization folded.

    Args:
        momentum (float): Momentum value should be [0, 1]. Default: 0.9.
        epsilon (float): A small float number to avoid dividing by 0. 1e-5 if dtype in
            float32 else 1e-3. Default: 1e-5.
        freeze_bn (int): Delay in steps at which computation switches from regular batch
            norm to frozen mean and std. Default: 0.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C, H, W)`.
        - **mean** (Tensor) - Tensor of shape :math:`(C,)`.
        - **variance** (Tensor) - Tensor of shape :math:`(C,)`.
        - **global_step** (Tensor) - Tensor to record current global step.

    Outputs:
        Tuple of 4 Tensor, the normalized input and the updated parameters.

        - **batch_mean** (Tensor) - Tensor of shape :math:`(C,)`.
        - **batch_std** (Tensor) - Tensor of shape :math:`(C,)`.
        - **running_mean** (Tensor) - Tensor of shape :math:`(C,)`.
        - **running_std** (Tensor) - Tensor of shape :math:`(C,)`.

    """

    def __init__(self, momentum=0.9, epsilon=1e-5, freeze_bn=0):
        """init batch norm fold layer"""
        super(BatchNormFoldCell, self).__init__()
        self.epsilon = epsilon
        self.is_gpu = context.get_context('device_target') == "GPU"
        if self.is_gpu:
            self.bn_train = Q.BatchNormFold(momentum, epsilon, is_training=True, freeze_bn=freeze_bn)
            self.bn_infer = Q.BatchNormFold(momentum, epsilon, is_training=False, freeze_bn=freeze_bn)
        else:
            self.bn_reduce = P.BNTrainingReduce()
            self.bn_update = Q.BatchNormFoldD(momentum, epsilon, is_training=True, freeze_bn=freeze_bn)

    def construct(self, x, mean, variance, global_step):
        if self.is_gpu:
            if self.training:
                batch_mean, batch_std, running_mean, running_std = self.bn_train(x, mean, variance, global_step)
            else:
                batch_mean, batch_std, running_mean, running_std = self.bn_infer(x, mean, variance, global_step)
        else:
            if self.training:
                x_sum, x_square_sum = self.bn_reduce(x)
                _, batch_mean, batch_std, running_mean, running_std, mean_updated, variance_updated = \
                    self.bn_update(x, x_sum, x_square_sum, mean, variance)
                P.Assign()(mean, mean_updated)
                P.Assign()(variance, variance_updated)
            else:
                batch_mean = P.ZerosLike()(variance)
                batch_std = P.OnesLike()(variance)
                running_mean = P.TensorAdd()(mean, 0.)
                running_std = P.Sqrt()(P.TensorAdd()(variance, self.epsilon))
        return batch_mean, batch_std, running_mean, running_std


class FakeQuantWithMinMax(Cell):
    r"""
    Quantization aware op. This OP provides the fake quantization observer function on data with min and max.

    Args:
        min_init (int, float): The dimension of channel or 1(layer). Default: -6.
        max_init (int, float): The dimension of channel or 1(layer). Default: 6.
        ema (bool): The exponential Moving Average algorithm updates min and max. Default: False.
        ema_decay (float): Exponential Moving Average algorithm parameter. Default: 0.999.
        per_channel (bool):  Quantization granularity based on layer or on channel. Default: False.
        channel_axis (int): Quantization by channel axis. Default: 1.
        num_channels (int): declarate the min and max channel size, Default: 1.
        num_bits (int): The bit number of quantization, supporting 4 and 8bits. Default: 8.
        symmetric (bool): Whether the quantization algorithm is symmetric or not. Default: False.
        narrow_range (bool): Whether the quantization algorithm uses narrow range or not. Default: False.
        quant_delay (int): Quantization delay parameters according to the global step. Default: 0.

    Inputs:
        - **x** (Tensor) - The input of FakeQuantWithMinMax.

    Outputs:
        Tensor, with the same type and shape as the `x`.

    Examples:
        >>> fake_quant = FakeQuantWithMinMax()
        >>> input_x = Tensor(np.array([[1, 2, 1], [-2, 0, -1]]), mindspore.float32)
        >>> result = fake_quant(input_x)
    """

    def __init__(self,
                 min_init=-6,
                 max_init=6,
                 ema=False,
                 ema_decay=0.999,
                 per_channel=False,
                 channel_axis=1,
                 num_channels=1,
                 num_bits=8,
                 symmetric=False,
                 narrow_range=False,
                 quant_delay=0):
        """init FakeQuantWithMinMax layer"""
        super(FakeQuantWithMinMax, self).__init__()
        validator.check_type("min_init", min_init, [int, float])
        validator.check_type("max_init", max_init, [int, float])
        validator.check("min_init", min_init, "max_init", max_init, rel=Rel.LT)
        validator.check_integer('quant_delay', quant_delay, 0, Rel.GE)
        self.min_init = min_init
        self.max_init = max_init
        self.num_bits = num_bits
        self.ema = ema
        self.ema_decay = ema_decay
        self.per_channel = per_channel
        self.num_channels = num_channels
        self.channel_axis = channel_axis
        self.quant_delay = quant_delay
        self.symmetric = symmetric
        self.narrow_range = narrow_range
        self.is_ascend = context.get_context('device_target') == "Ascend"

        # init tensor min and max for fake quant op
        if self.per_channel:
            min_array = np.array([self.min_init] * self.num_channels).astype(np.float32)
            max_array = np.array([self.max_init] * self.num_channels).astype(np.float32)
        else:
            min_array = np.array([self.min_init]).astype(np.float32)
            max_array = np.array([self.max_init]).astype(np.float32)
        self.minq = Parameter(Tensor(min_array), name='quant_min', requires_grad=False)
        self.maxq = Parameter(Tensor(max_array), name='quant_max', requires_grad=False)

        # init fake quant relative op
        if self.per_channel:
            quant_fun = partial(Q.FakeQuantPerChannel, channel_axis=self.channel_axis)
            ema_fun = partial(Q.MinMaxUpdatePerChannel, channel_axis=self.channel_axis)
        else:
            quant_fun = Q.FakeQuantPerLayer
            ema_fun = Q.MinMaxUpdatePerLayer

        self.ema_update = ema_fun(ema=self.ema, ema_decay=self.ema_decay)
        if self.is_ascend:
            self.fake_quant_train = quant_fun(num_bits=self.num_bits,
                                              symmetric=self.symmetric,
                                              narrow_range=self.narrow_range)
            self.fake_quant_infer = self.fake_quant_train
        else:
            quant_fun = partial(quant_fun,
                                ema=self.ema,
                                ema_decay=ema_decay,
                                num_bits=self.num_bits,
                                symmetric=self.symmetric,
                                narrow_range=self.narrow_range,
                                quant_delay=self.quant_delay)
            self.fake_quant_train = quant_fun(training=True)
            self.fake_quant_infer = quant_fun(training=False)

    def extend_repr(self):
        s = 'num_bits={}, symmetric={}, narrow_range={}, ema={}({}), per_channel={}({}, {}), ' \
            'quant_delay={}, min_init={}, max_init={}'.format(self.num_bits, self.symmetric, self.narrow_range,
                                                              self.ema, self.ema_decay, self.per_channel,
                                                              self.channel_axis, self.num_channels, self.quant_delay,
                                                              self.min_init, self.max_init)
        return s

    def construct(self, x):
        if self.training:
            min_up, max_up = self.ema_update(x, self.minq, self.maxq)
            P.Assign()(self.minq, min_up)
            P.Assign()(self.maxq, max_up)
            out = self.fake_quant_train(x, self.minq, self.maxq)
        else:
            out = self.fake_quant_infer(x, self.minq, self.maxq)
        return out


class Conv2dBnFoldQuant(Cell):
    r"""
    2D convolution with BatchNormal op folded construct.

    This part is a more detailed overview of Conv2d op.

    Args:
        in_channels (int): The number of input channel :math:`C_{in}`.
        out_channels (int): The number of output channel :math:`C_{out}`.
        kernel_size (Union[int, tuple]): Specifies the height and width of the 2D convolution window.
        stride (int): Specifies stride for all spatial dimensions with the same value.
        pad_mode (str): Specifies padding mode. The optional values are "same", "valid", "pad". Default: "same".
        padding (int): Implicit paddings on both sides of the input. Default: 0.
        eps (float): Parameters for BatchNormal. Default: 1e-5.
        momentum (float): Parameters for BatchNormal op. Default: 0.997.
        weight_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the
            convolution kernel. Default: 'normal'.
        beta_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the
            beta vector. Default: 'zeros'.
        gamma_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the
            gamma vector. Default: 'ones'.
        mean_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the
            mean vector. Default: 'zeros'.
        var_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the
            variance vector. Default: 'ones'.
        fake (bool): Whether Conv2dBnFoldQuant Cell adds FakeQuantWithMinMax op. Default: True.
        per_channel (bool): FakeQuantWithMinMax Parameters. Default: False.
        num_bits (int): The bit number of quantization, supporting 4 and 8bits. Default: 8.
        symmetric (bool): The quantization algorithm is symmetric or not. Default: False.
        narrow_range (bool): The quantization algorithm uses narrow range or not. Default: False.
        quant_delay (int): The Quantization delay parameters according to the global step. Default: 0.
        freeze_bn (int): The quantization freeze BatchNormal op is according to the global step. Default: 100000.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor of shape :math:`(N, C_{out}, H_{out}, W_{out})`.

    Examples:
        >>> conv2d_bn = nn.Conv2dBnFoldQuant(1, 6, kernel_size=(2, 2), stride=(1, 1), pad_mode="valid")
        >>> x = Tensor(np.random.randint(-2, 2, (2, 1, 1, 3)), mindspore.float32)
        >>> y = conv2d_bn(x)
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
                 eps=1e-5,
                 momentum=0.997,
                 weight_init='normal',
                 beta_init='zeros',
                 gamma_init='ones',
                 mean_init='zeros',
                 var_init='ones',
                 fake=True,
                 per_channel=False,
                 num_bits=8,
                 symmetric=False,
                 narrow_range=False,
                 quant_delay=0,
                 freeze_bn=100000):
        """init Conv2dBnFoldQuant layer"""
        super(Conv2dBnFoldQuant, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = twice(kernel_size)
        self.stride = twice(stride)
        self.pad_mode = pad_mode
        self.padding = padding
        self.dilation = twice(dilation)
        self.group = group
        self.eps = eps
        self.momentum = momentum
        self.quant_delay = quant_delay
        self.freeze_bn = freeze_bn
        self.fake = fake
        self.num_bits = num_bits
        self.per_channel = per_channel
        self.symmetric = symmetric
        self.narrow_range = narrow_range
        self.is_gpu = context.get_context('device_target') == "GPU"

        # initialize convolution op and Parameter
        if context.get_context('device_target') == "Ascend" and group > 1:
            validator.check_integer('group', group, in_channels, Rel.EQ)
            validator.check_integer('group', group, out_channels, Rel.EQ)
            self.conv = P.DepthwiseConv2dNative(channel_multiplier=1,
                                                kernel_size=self.kernel_size,
                                                pad_mode=pad_mode,
                                                pad=padding,
                                                stride=self.stride,
                                                dilation=self.dilation)
            weight_shape = [1, in_channels, *self.kernel_size]
            channel_axis = 1
        else:
            self.conv = P.Conv2D(out_channel=out_channels,
                                 kernel_size=self.kernel_size,
                                 pad_mode=pad_mode,
                                 pad=padding,
                                 stride=self.stride,
                                 dilation=self.dilation,
                                 group=group)
            weight_shape = [out_channels, in_channels // group, *self.kernel_size]
            channel_axis = 0
        self.weight = Parameter(initializer(weight_init, weight_shape), name='weight')

        # initialize BatchNorm Parameter
        self.gamma = Parameter(initializer(gamma_init, [out_channels]), name='gamma')
        self.beta = Parameter(initializer(beta_init, [out_channels]), name='beta')
        self.moving_mean = Parameter(initializer(mean_init, [out_channels]), name='moving_mean', requires_grad=False)
        self.moving_variance = Parameter(initializer(var_init, [out_channels]), name='moving_variance',
                                         requires_grad=False)

        # initialize fake ops
        self.fake_quant_weight = FakeQuantWithMinMax(min_init=-6,
                                                     max_init=6,
                                                     ema=False,
                                                     per_channel=per_channel,
                                                     channel_axis=channel_axis,
                                                     num_channels=out_channels,
                                                     num_bits=num_bits,
                                                     symmetric=symmetric,
                                                     narrow_range=narrow_range,
                                                     quant_delay=quant_delay)
        self.batchnorm_fold = BatchNormFoldCell(epsilon=eps, momentum=momentum, freeze_bn=freeze_bn)
        self.correct_mul = Q.CorrectionMul(channel_axis)
        if context.get_context('device_target') == "Ascend":
            self.batchnorm_fold2_train = Q.BatchNormFold2_D(freeze_bn=freeze_bn)
            self.batchnorm_fold2_infer = Q.BatchNormFold2_D(freeze_bn=0)
        elif context.get_context('device_target') == "GPU":
            self.batchnorm_fold2_train = Q.BatchNormFold2(freeze_bn=freeze_bn)
            self.batchnorm_fold2_infer = Q.BatchNormFold2(freeze_bn=0)
        else:
            raise ValueError("Unsupported platform: {}".format(context.get_context('device_target')))
        self.step = Parameter(initializer('normal', [1], dtype=mstype.int32), name='step', requires_grad=False)
        self.one = Tensor(1, mstype.int32)
        self.assignadd = P.AssignAdd()

    def extend_repr(self):
        s = 'in_channels={}, out_channels={}, kernel_size={}, stride={}, ' \
            'pad_mode={}, padding={}, dilation={}, group={}, ' \
            'fake={}, freeze_bn={}, momentum={}, quant_delay={}'.format(self.in_channels, self.out_channels,
                                                                        self.kernel_size, self.stride,
                                                                        self.pad_mode, self.padding, self.dilation,
                                                                        self.group,
                                                                        self.fake, self.freeze_bn, self.momentum,
                                                                        self.quant_delay)
        return s

    def construct(self, x):
        out_conv = self.conv(x, self.weight)
        # BN fold1
        batch_mean, batch_std, running_mean, running_std = self.batchnorm_fold(out_conv,
                                                                               self.moving_mean,
                                                                               self.moving_variance,
                                                                               self.step)
        # fake weight
        weight = self.correct_mul(self.weight, self.gamma, running_std)
        if self.fake:
            weight = self.fake_quant_weight(weight)
        out = self.conv(x, weight)
        # BN fold2
        if self.is_gpu:
            if self.training:
                out = self.batchnorm_fold2_train(out, self.beta, self.gamma,
                                                 batch_std, batch_mean, running_std, running_mean, self.step)
                F.control_depend(out, self.assignadd(self.step, self.one))
            else:
                out = self.batchnorm_fold2_infer(out, self.beta, self.gamma,
                                                 batch_std, batch_mean, running_std, running_mean, self.step)
        else:
            if self.training:
                out = self.batchnorm_fold2_train(out, self.beta, self.gamma, batch_std, batch_mean, running_std)
                F.control_depend(out, self.assignadd(self.step, self.one))
            else:
                out = self.batchnorm_fold2_infer(out, self.beta, self.gamma, running_std, running_mean, running_std)
        return out


class Conv2dBnWithoutFoldQuant(Cell):
    r"""
    2D convolution + batchnorm without fold with fake quant construct.

    This part is a more detailed overview of Conv2d op.

    Args:
        in_channels (int): The number of input channel :math:`C_{in}`.
        out_channels (int): The number of output channel :math:`C_{out}`.
        kernel_size (Union[int, tuple]): Specifies the height and width of the 2D convolution window.
        stride (int): Specifies stride for all spatial dimensions with the same value. Default: 1.
        pad_mode (str): Specifies padding mode. The optional values are "same", "valid", "pad". Default: "same".
        padding (int): Implicit paddings on both sides of the input. Default: 0.
        dilation (int): Specifying the dilation rate to use for dilated convolution. Default: 1.
        group (int): Split filter into groups, `in_ channels` and `out_channels` should be
            divisible by the number of groups. Default: 1.
        has_bias (bool): Specifies whether the layer uses a bias vector. Default: False.
        eps (float): Parameters for BatchNormal. Default: 1e-5.
        momentum (float): Parameters for BatchNormal op. Default: 0.997.
        weight_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the convolution kernel.
            Default: 'normal'.
        bias_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the bias vector. Default: 'zeros'.
        per_channel (bool): FakeQuantWithMinMax Parameters. Default: False.
        num_bits (int): The bit number of quantization, supporting 4 and 8bits. Default: 8.
        symmetric (bool): The quantization algorithm is symmetric or not. Default: False.
        narrow_range (bool): The quantization algorithm uses narrow range or not. Default: False.
        quant_delay (int): Quantization delay parameters according to the global step. Default: 0.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor of shape :math:`(N, C_{out}, H_{out}, W_{out})`.

    Examples:
        >>> conv2d_quant = nn.Conv2dBnWithoutFoldQuant(1, 6, kernel_size=(2, 2), stride=(1, 1), pad_mode="valid")
        >>> x = Tensor(np.random.randint(-2, 2, (2, 1, 1, 3)), mstype.float32)
        >>> y = conv2d_quant(x)
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
                 eps=1e-5,
                 momentum=0.997,
                 weight_init='normal',
                 bias_init='zeros',
                 per_channel=False,
                 num_bits=8,
                 symmetric=False,
                 narrow_range=False,
                 quant_delay=0):
        super(Conv2dBnWithoutFoldQuant, self).__init__()
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size
        self.in_channels = check_int_positive(in_channels)
        self.out_channels = check_int_positive(out_channels)
        self.has_bias = has_bias
        self.stride = twice(stride)
        self.dilation = twice(dilation)
        self.pad_mode = pad_mode
        self.padding = padding
        self.group = group
        self.quant_delay = quant_delay

        weight_shape = [out_channels, in_channels // group, *self.kernel_size]
        self.weight = Parameter(initializer(weight_init, weight_shape), name='weight')

        self.bias_add = P.BiasAdd()
        if check_bool(has_bias):
            self.bias = Parameter(initializer(bias_init, [out_channels]), name='bias')
        else:
            self.bias = None

        self.conv = P.Conv2D(out_channel=self.out_channels,
                             kernel_size=self.kernel_size,
                             mode=1,
                             pad_mode=self.pad_mode,
                             pad=self.padding,
                             stride=self.stride,
                             dilation=self.dilation,
                             group=self.group)
        self.fake_quant_weight = FakeQuantWithMinMax(min_init=-6,
                                                     max_init=6,
                                                     ema=False,
                                                     per_channel=per_channel,
                                                     channel_axis=0,
                                                     num_channels=out_channels,
                                                     num_bits=num_bits,
                                                     symmetric=symmetric,
                                                     narrow_range=narrow_range,
                                                     quant_delay=quant_delay)
        self.batchnorm = BatchNorm2d(out_channels, eps=eps, momentum=momentum)

    def construct(self, x):
        weight = self.fake_quant_weight(self.weight)
        out = self.conv(x, weight)
        if self.has_bias:
            out = self.bias_add(out, self.bias)
        out = self.batchnorm(out)
        return out

    def extend_repr(self):
        s = 'in_channels={}, out_channels={}, kernel_size={}, stride={}, ' \
            'pad_mode={}, padding={}, dilation={}, group={}, ' \
            'has_bias={}, quant_delay={}'.format(self.in_channels, self.out_channels, self.kernel_size, self.stride,
                                                 self.pad_mode, self.padding, self.dilation, self.group,
                                                 self.has_bias, self.quant_delay)
        return s


class Conv2dQuant(Cell):
    r"""
    2D convolution with fake quant op layer.

    This part is a more detailed overview of Conv2d op.

    Args:
        in_channels (int): The number of input channel :math:`C_{in}`.
        out_channels (int): The number of output channel :math:`C_{out}`.
        kernel_size (Union[int, tuple]): Specifies the height and width of the 2D convolution window.
        stride (int): Specifies stride for all spatial dimensions with the same value. Default: 1.
        pad_mode (str): Specifies padding mode. The optional values are "same", "valid", "pad". Default: "same".
        padding (int): Implicit paddings on both sides of the input. Default: 0.
        dilation (int): Specifying the dilation rate to use for dilated convolution. Default: 1.
        group (int): Split filter into groups, `in_ channels` and `out_channels` should be
            divisible by the number of groups. Default: 1.
        has_bias (bool): Specifies whether the layer uses a bias vector. Default: False.
        weight_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the convolution kernel.
            Default: 'normal'.
        bias_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the bias vector. Default: 'zeros'.
        per_channel (bool): FakeQuantWithMinMax Parameters. Default: False.
        num_bits (int): The bit number of quantization, supporting 4 and 8bits. Default: 8.
        symmetric (bool): The quantization algorithm is symmetric or not. Default: False.
        narrow_range (bool): The quantization algorithm uses narrow range or not. Default: False.
        quant_delay (int): Quantization delay parameters according to the global step. Default: 0.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor of shape :math:`(N, C_{out}, H_{out}, W_{out})`.

    Examples:
        >>> conv2d_quant = nn.Conv2dQuant(1, 6, kernel_size= (2, 2), stride=(1, 1), pad_mode="valid")
        >>> x = Tensor(np.random.randint(-2, 2, (2, 1, 1, 3)), mindspore.float32)
        >>> y = conv2d_quant(x)
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
                 per_channel=False,
                 num_bits=8,
                 symmetric=False,
                 narrow_range=False,
                 quant_delay=0):
        super(Conv2dQuant, self).__init__()
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size
        self.in_channels = check_int_positive(in_channels)
        self.out_channels = check_int_positive(out_channels)
        self.has_bias = has_bias
        self.stride = twice(stride)
        self.dilation = twice(dilation)
        self.pad_mode = pad_mode
        self.padding = padding
        self.group = group
        self.quant_delay = quant_delay

        weight_shape = [out_channels, in_channels // group, *self.kernel_size]
        self.weight = Parameter(initializer(weight_init, weight_shape), name='weight')

        self.bias_add = P.BiasAdd()
        if check_bool(has_bias):
            self.bias = Parameter(initializer(bias_init, [out_channels]), name='bias')
        else:
            self.bias = None

        self.conv = P.Conv2D(out_channel=self.out_channels,
                             kernel_size=self.kernel_size,
                             mode=1,
                             pad_mode=self.pad_mode,
                             pad=self.padding,
                             stride=self.stride,
                             dilation=self.dilation,
                             group=self.group)
        self.fake_quant_weight = FakeQuantWithMinMax(min_init=-6,
                                                     max_init=6,
                                                     ema=False,
                                                     per_channel=per_channel,
                                                     channel_axis=0,
                                                     num_channels=out_channels,
                                                     num_bits=num_bits,
                                                     symmetric=symmetric,
                                                     narrow_range=narrow_range,
                                                     quant_delay=quant_delay)

    def construct(self, x):
        weight = self.fake_quant_weight(self.weight)
        out = self.conv(x, weight)
        if self.has_bias:
            return self.bias_add(out, self.bias)
        return out

    def extend_repr(self):
        s = 'in_channels={}, out_channels={}, kernel_size={}, stride={}, ' \
            'pad_mode={}, padding={}, dilation={}, group={}, ' \
            'has_bias={}, quant_delay={}'.format(self.in_channels, self.out_channels, self.kernel_size, self.stride,
                                                 self.pad_mode, self.padding, self.dilation, self.group,
                                                 self.has_bias, self.quant_delay)
        return s


class DenseQuant(Cell):
    r"""
    The fully connected layer with fake quant op.

    This part is a more detailed overview of Dense op.

    Args:
        in_channels (int): The dimension of the input space.
        out_channels (int): The dimension of the output space.
        weight_init (Union[Tensor, str, Initializer, numbers.Number]): The trainable weight_init parameter. The dtype
            is same as input x. The values of str refer to the function `initializer`. Default: 'normal'.
        bias_init (Union[Tensor, str, Initializer, numbers.Number]): The trainable bias_init parameter. The dtype is
            same as input x. The values of str refer to the function `initializer`. Default: 'zeros'.
        has_bias (bool): Specifies whether the layer uses a bias vector. Default: True.
        activation (str): The regularization function applied to the output of the layer, eg. 'relu'. Default: None.
        per_channel (bool): FakeQuantWithMinMax Parameters. Default: False.
        num_bits (int): The bit number of quantization, supporting 4 and 8bits. Default: 8.
        symmetric (bool): The quantization algorithm is symmetric or not. Default: False.
        narrow_range (bool): The quantization algorithm uses narrow range or not. Default: False.
        quant_delay (int): Quantization delay parameters according to the global step. Default: 0.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor of shape :math:`(N, C_{out}, H_{out}, W_{out})`.

    Examples:
        >>> dense_quant = nn.DenseQuant(3, 6)
        >>> input_x = Tensor(np.random.randint(-2, 2, (2, 3)), mindspore.float32)
        >>> result = dense_quant(input_x)
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            weight_init='normal',
            bias_init='zeros',
            has_bias=True,
            activation=None,
            per_channel=False,
            num_bits=8,
            symmetric=False,
            narrow_range=False,
            quant_delay=0):
        super(DenseQuant, self).__init__()
        self.in_channels = check_int_positive(in_channels)
        self.out_channels = check_int_positive(out_channels)
        self.has_bias = check_bool(has_bias)

        if isinstance(weight_init, Tensor):
            if weight_init.dim() != 2 or weight_init.shape[0] != out_channels or \
                    weight_init.shape[1] != in_channels:
                raise ValueError("weight_init shape error")

        self.weight = Parameter(initializer(
            weight_init, [out_channels, in_channels]), name="weight")

        if self.has_bias:
            if isinstance(bias_init, Tensor):
                if bias_init.dim() != 1 or bias_init.shape[0] != out_channels:
                    raise ValueError("bias_init shape error")

            self.bias = Parameter(initializer(
                bias_init, [out_channels]), name="bias")

        self.matmul = P.MatMul(transpose_b=True)
        self.bias_add = P.BiasAdd()

        self.activation = get_activation(activation)
        self.activation_flag = self.activation is not None
        self.fake_quant_weight = FakeQuantWithMinMax(min_init=-6,
                                                     max_init=6,
                                                     ema=False,
                                                     per_channel=per_channel,
                                                     channel_axis=0,
                                                     num_channels=out_channels,
                                                     num_bits=num_bits,
                                                     symmetric=symmetric,
                                                     narrow_range=narrow_range,
                                                     quant_delay=quant_delay)

    def construct(self, x):
        """Use operators to construct the Dense layer."""
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


class _QuantActivation(Cell):
    r"""
    Base class for quantization aware training activation function. Add Fake Quant OP after activation OP.
    """

    def get_origin(self):
        raise NotImplementedError


class ActQuant(_QuantActivation):
    r"""
    Quantization aware training activation function.

    Add the fake quant op to the end of activation op, by which the output of activation op will be truncated.
    Please check `FakeQuantWithMinMax` for more details.

    Args:
        activation (Cell): Activation cell class.
        ema_decay (float): Exponential Moving Average algorithm parameter. Default: 0.999.
        per_channel (bool):  Quantization granularity based on layer or on channel. Default: False.
        num_bits (int): The bit number of quantization, supporting 4 and 8bits. Default: 8.
        symmetric (bool): The quantization algorithm is symmetric or not. Default: False.
        narrow_range (bool): The quantization algorithm uses narrow range or not. Default: False.
        quant_delay (int): Quantization delay parameters according to the global steps. Default: 0.

    Inputs:
        - **x** (Tensor) - The input of ReLU6Quant.

    Outputs:
        Tensor, with the same type and shape as the `x`.

    Examples:
        >>> act_quant = nn.ActQuant(nn.ReLU())
        >>> input_x = Tensor(np.array([[1, 2, -1], [-2, 0, -1]]), mindspore.float32)
        >>> result = act_quant(input_x)
    """

    def __init__(self,
                 activation,
                 ema_decay=0.999,
                 per_channel=False,
                 num_bits=8,
                 symmetric=False,
                 narrow_range=False,
                 quant_delay=0):
        super(ActQuant, self).__init__()
        self.fake_quant_act = FakeQuantWithMinMax(min_init=0,
                                                  max_init=6,
                                                  ema=True,
                                                  ema_decay=ema_decay,
                                                  per_channel=per_channel,
                                                  num_bits=num_bits,
                                                  symmetric=symmetric,
                                                  narrow_range=narrow_range,
                                                  quant_delay=quant_delay)
        self.act = activation

    def construct(self, x):
        x = self.act(x)
        x = self.fake_quant_act(x)
        return x

    def get_origin(self):
        return self.act

class LeakyReLUQuant(_QuantActivation):
    r"""
    LeakyReLUQuant activation function. Add Fake Quant OP after HSwish OP.

    This part is a more detailed overview of HSwish op.

    Args:
        activation (Cell): Activation cell class.
        ema_decay (float): Exponential Moving Average algorithm parameter. Default: 0.999.
        per_channel (bool):  Quantization granularity based on layer or on channel. Default: False.
        num_bits (int): The bit number of quantization, supporting 4 and 8bits. Default: 8.
        symmetric (bool): The quantization algorithm is symmetric or not. Default: False.
        narrow_range (bool): The quantization algorithm uses narrow range or not. Default: False.
        quant_delay (int): Quantization delay parameters according to the global step. Default: 0.

    Inputs:
        - **x** (Tensor) - The input of LeakyReLUQuant.

    Outputs:
        Tensor, with the same type and shape as the `x`.

    Examples:
        >>> activation = nn.LeakyReLUQuant(nn.LeakyReLU())
        >>> input = Tensor(np.array([[1, 2, 1], [-2, 0, -1]]), mindspore.float32)
        >>> result = activation(input)
    """

    def __init__(self,
                 activation,
                 ema_decay=0.999,
                 per_channel=False,
                 num_bits=8,
                 symmetric=False,
                 narrow_range=False,
                 quant_delay=0):
        super(LeakyReLUQuant, self).__init__()
        self.fake_quant_act_before = FakeQuantWithMinMax(min_init=-6,
                                                         max_init=6,
                                                         ema=True,
                                                         ema_decay=ema_decay,
                                                         per_channel=per_channel,
                                                         num_bits=num_bits,
                                                         symmetric=symmetric,
                                                         narrow_range=narrow_range,
                                                         quant_delay=quant_delay)
        self.fake_quant_act_after = FakeQuantWithMinMax(min_init=-6,
                                                        max_init=6,
                                                        ema=True,
                                                        ema_decay=ema_decay,
                                                        per_channel=per_channel,
                                                        num_bits=num_bits,
                                                        symmetric=symmetric,
                                                        narrow_range=narrow_range,
                                                        quant_delay=quant_delay)
        if issubclass(activation.__class__, nn.LeakyReLU):
            self.act = activation
        else:
            raise ValueError("Activation should be `nn.LeakyReLU`")

    def construct(self, x):
        x = self.fake_quant_act_before(x)
        x = self.act(x)
        x = self.fake_quant_act_after(x)
        return x

    def get_origin(self):
        return self.act



class HSwishQuant(_QuantActivation):
    r"""
    HSwishQuant activation function. Add Fake Quant OP after HSwish OP.

    This part is a more detailed overview of HSwish op.

    Args:
        activation (Cell): Activation cell class.
        ema_decay (float): Exponential Moving Average algorithm parameter. Default: 0.999.
        per_channel (bool):  Quantization granularity based on layer or on channel. Default: False.
        num_bits (int): The bit number of quantization, supporting 4 and 8bits. Default: 8.
        symmetric (bool): Whether the quantization algorithm is symmetric or not. Default: False.
        narrow_range (bool): Whether the quantization algorithm uses narrow range or not. Default: False.
        quant_delay (int): Quantization delay parameters according to the global step. Default: 0.

    Inputs:
        - **x** (Tensor) - The input of HSwishQuant.

    Outputs:
        Tensor, with the same type and shape as the `x`.

    Examples:
        >>> activation = nn.HSwishQuant(nn.HSwish())
        >>> input = Tensor(np.array([[1, 2, 1], [-2, 0, -1]]), mindspore.float32)
        >>> result = activation(input)
    """

    def __init__(self,
                 activation,
                 ema_decay=0.999,
                 per_channel=False,
                 num_bits=8,
                 symmetric=False,
                 narrow_range=False,
                 quant_delay=0):
        super(HSwishQuant, self).__init__()
        self.fake_quant_act_before = FakeQuantWithMinMax(min_init=-6,
                                                         max_init=6,
                                                         ema=True,
                                                         ema_decay=ema_decay,
                                                         per_channel=per_channel,
                                                         num_bits=num_bits,
                                                         symmetric=symmetric,
                                                         narrow_range=narrow_range,
                                                         quant_delay=quant_delay)
        self.fake_quant_act_after = FakeQuantWithMinMax(min_init=-6,
                                                        max_init=6,
                                                        ema=True,
                                                        ema_decay=ema_decay,
                                                        per_channel=per_channel,
                                                        num_bits=num_bits,
                                                        symmetric=symmetric,
                                                        narrow_range=narrow_range,
                                                        quant_delay=quant_delay)
        if issubclass(activation.__class__, nn.HSwish):
            self.act = activation
        else:
            raise ValueError("Activation should be `nn.HSwish`")

    def construct(self, x):
        x = self.fake_quant_act_before(x)
        x = self.act(x)
        x = self.fake_quant_act_after(x)
        return x

    def get_origin(self):
        return self.act


class HSigmoidQuant(_QuantActivation):
    r"""
    HSigmoidQuant activation function. Add Fake Quant OP before and after HSigmoid OP.

    This part is a more detailed overview of HSigmoid op.

    Args:
        activation (Cell): Activation cell class.
        ema_decay (float): Exponential Moving Average algorithm parameter. Default: 0.999.
        per_channel (bool):  Quantization granularity based on layer or on channel. Default: False.
        num_bits (int): The bit number of quantization, supporting 4 and 8bits. Default: 8.
        symmetric (bool): Whether the quantization algorithm is symmetric or not. Default: False.
        narrow_range (bool): Whether the quantization algorithm uses narrow range or not. Default: False.
        quant_delay (int): Quantization delay parameters according to the global step. Default: 0.

    Inputs:
        - **x** (Tensor) - The input of HSigmoidQuant.

    Outputs:
        Tensor, with the same type and shape as the `x`.

    Examples:
        >>> activation = nn.HSigmoidQuant(nn.HSigmoid())
        >>> input = Tensor(np.array([[1, 2, 1], [-2, 0, -1]]), mindspore.float32)
        >>> result = activation(input)
    """

    def __init__(self,
                 activation,
                 ema_decay=0.999,
                 per_channel=False,
                 num_bits=8,
                 symmetric=False,
                 narrow_range=False,
                 quant_delay=0):
        super(HSigmoidQuant, self).__init__()
        self.fake_quant_act_before = FakeQuantWithMinMax(min_init=-6,
                                                         max_init=6,
                                                         ema=True,
                                                         ema_decay=ema_decay,
                                                         per_channel=per_channel,
                                                         num_bits=num_bits,
                                                         symmetric=symmetric,
                                                         narrow_range=narrow_range,
                                                         quant_delay=quant_delay)
        self.fake_quant_act_after = FakeQuantWithMinMax(min_init=-6,
                                                        max_init=6,
                                                        ema=True,
                                                        ema_decay=ema_decay,
                                                        per_channel=per_channel,
                                                        num_bits=num_bits,
                                                        symmetric=symmetric,
                                                        narrow_range=narrow_range,
                                                        quant_delay=quant_delay)
        if issubclass(activation.__class__, nn.HSigmoid):
            self.act = activation
        else:
            raise ValueError("Activation should be `nn.HSigmoid`")

    def construct(self, x):
        x = self.fake_quant_act_before(x)
        x = self.act(x)
        x = self.fake_quant_act_after(x)
        return x

    def get_origin(self):
        return self.act


class TensorAddQuant(Cell):
    r"""
    Add Fake Quant OP after TensorAdd OP.

    This part is a more detailed overview of TensorAdd op.

    Args:
        ema_decay (float): Exponential Moving Average algorithm parameter. Default: 0.999.
        per_channel (bool):  Quantization granularity based on layer or on channel. Default: False.
        num_bits (int): The bit number of quantization, supporting 4 and 8bits. Default: 8.
        symmetric (bool): The quantization algorithm is symmetric or not. Default: False.
        narrow_range (bool): The quantization algorithm uses narrow range or not. Default: False.
        quant_delay (int): Quantization delay parameters according to the global step. Default: 0.

    Inputs:
        - **x** (Tensor) - The input of TensorAddQuant.

    Outputs:
        Tensor, with the same type and shape as the `x`.

    Examples:
        >>> add_quant = nn.TensorAddQuant()
        >>> input_x = Tensor(np.array([[1, 2, 1], [-2, 0, -1]]), mindspore.float32)
        >>> input_y = Tensor(np.random.randint(-2, 2, (2, 3)), mindspore.float32)
        >>> result = add_quant(input_x, input_y)
    """

    def __init__(self,
                 ema_decay=0.999,
                 per_channel=False,
                 num_bits=8,
                 symmetric=False,
                 narrow_range=False,
                 quant_delay=0):
        super(TensorAddQuant, self).__init__()
        self.fake_quant_act = FakeQuantWithMinMax(min_init=-6,
                                                  max_init=6,
                                                  ema=True,
                                                  ema_decay=ema_decay,
                                                  per_channel=per_channel,
                                                  num_bits=num_bits,
                                                  symmetric=symmetric,
                                                  narrow_range=narrow_range,
                                                  quant_delay=quant_delay)
        self.add = P.TensorAdd()

    def construct(self, x1, x2):
        x = self.add(x1, x2)
        x = self.fake_quant_act(x)
        return x


class MulQuant(Cell):
    r"""
    Add Fake Quant OP after Mul OP.

    This part is a more detailed overview of Mul op.

    Args:
        ema_decay (float): Exponential Moving Average algorithm parameter. Default: 0.999.
        per_channel (bool):  Quantization granularity based on layer or on channel. Default: False.
        num_bits (int): The bit number of quantization, supporting 4 and 8bits. Default: 8.
        symmetric (bool): The quantization algorithm is symmetric or not. Default: False.
        narrow_range (bool): The quantization algorithm uses narrow range or not. Default: False.
        quant_delay (int): Quantization delay parameters according to the global step. Default: 0.

    Inputs:
        - **x** (Tensor) - The input of MulQuant.

    Outputs:
        Tensor, with the same type and shape as the `x`.

    """

    def __init__(self,
                 ema_decay=0.999,
                 per_channel=False,
                 num_bits=8,
                 symmetric=False,
                 narrow_range=False,
                 quant_delay=0):
        super(MulQuant, self).__init__()
        self.fake_quant_act = FakeQuantWithMinMax(min_init=-6,
                                                  max_init=6,
                                                  ema=True,
                                                  ema_decay=ema_decay,
                                                  per_channel=per_channel,
                                                  num_bits=num_bits,
                                                  symmetric=symmetric,
                                                  narrow_range=narrow_range,
                                                  quant_delay=quant_delay)
        self.mul = P.Mul()

    def construct(self, x1, x2):
        x = self.mul(x1, x2)
        x = self.fake_quant_act(x)
        return x


class QuantBlock(Cell):
    r"""
    A quant block of Conv/Dense, activation layer for Ascend deploy.

    Calculate Conv or Dense in Int8, with Quant and DeQuant.

    Notes:
        This block is only for deploy, and not trainable.

    Args:
        in_channels (int): The number of channels in the input space.
        out_channels (int): The number of channels in the output space.
        weight_init (Union[Tensor, str, Initializer, numbers.Number]): The trainable weight_init parameter. The dtype
            is same as input x. The values of str refer to the function `initializer`. Default: 'normal'.
        bias_init (Union[Tensor, str, Initializer, numbers.Number]): The trainable bias_init parameter. The dtype is
            same as input x. The values of str refer to the function `initializer`. Default: 'zeros'.
        has_bias (bool): Specifies whether the layer uses a bias vector. Default: True.
        activation (str): The regularization function applied to the output of the layer, eg. 'relu'. Default: None.
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
                 core_op,
                 weight,
                 quant_op,
                 dequant_op,
                 dequant_scale,
                 bias=None,
                 activation=None):
        super(QuantBlock, self).__init__()
        self.core_op = core_op
        self.weight = weight
        self.quant = quant_op
        self.dequant = dequant_op
        self.dequant_scale = dequant_scale
        self.bias = bias
        self.has_bias = bias is not None
        self.activation = activation
        self.has_act = activation is not None
        if isinstance(activation, ReLU):
            self.activation = None
            self.has_act = False
            self.dequant.add_prim_attr("relu_flag", True)
        self.bias_add = P.BiasAdd()

    def construct(self, x):
        x = self.quant(x)
        if self.has_bias:
            x = self.core_op(x, self.weight, self.bias)
        else:
            x = self.core_op(x, self.weight)
        if self.has_act:
            x = self.activation(x)
        x = self.dequant(x, self.dequant_scale)
        return x

    def extend_repr(self):
        str_info = f'quant={self.quant}, core_op={type(self.core_op)}, weight=shape[{self.weight.shape}]'
        if self.has_bias:
            str_info = str_info + f', bias=shape[{self.bias.shape}]'
        if self.has_act:
            str_info = str_info + f', activation={self.activation}'
        str_info = str_info + f', dequant={self.dequant}'
        return str_info
