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

from functools import partial
import numpy as np
import mindspore.common.dtype as mstype
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.common.parameter import Parameter
from mindspore.common.initializer import initializer
from mindspore.common.tensor import Tensor
from mindspore._checkparam import check_int_positive, check_bool, twice
from mindspore._checkparam import Validator as validator, Rel
from mindspore.nn.cell import Cell
from mindspore.nn.layer.activation import get_activation
import mindspore.context as context

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
    'MulQuant',
]


class BatchNormFoldCell(Cell):
    """
    Batch normalization folded.

    Args:
        momentum (float): Momentum value should be [0, 1]. Default: 0.1.
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
            self.bn_train = P.BatchNormFold(momentum, epsilon, is_training=True, freeze_bn=freeze_bn)
            self.bn_infer = P.BatchNormFold(momentum, epsilon, is_training=False, freeze_bn=freeze_bn)
        else:
            self.bn_reduce = P.BNTrainingReduce()
            self.bn_update = P.BatchNormFoldD(momentum, epsilon, is_training=True, freeze_bn=freeze_bn)

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
    Aware Quantization op. This OP provide Fake quantization observer function on data with min and max.

    Args:
        min_init (int, float): The dimension of channel or 1(layer). Default: -6.
        max_init (int, float): The dimension of channel or 1(layer). Default: 6.
        num_bits (int): Quantization number bit, support 4 and 8bit. Default: 8.
        ema (bool): Exponential Moving Average algorithm update min and max. Default: False.
        ema_decay (float): Exponential Moving Average algorithm parameter. Default: 0.999.
        per_channel (bool): Quantization by layer or channel. Default: False.
        out_channels (int): declarate the min and max channel size, Default: 1.
        quant_delay (int): Quantization delay parameters according by global step. Default: 0.
        symmetric (bool): Quantization algorithm use symmetric or not. Default: False.
        narrow_range (bool): Quantization algorithm use narrow range or not. Default: False.

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
                 num_bits=8,
                 ema=False,
                 ema_decay=0.999,
                 per_channel=False,
                 channel_axis=1,
                 out_channels=1,
                 quant_delay=0,
                 symmetric=False,
                 narrow_range=False,
                 training=True):
        """init FakeQuantWithMinMax layer"""
        super(FakeQuantWithMinMax, self).__init__()
        self.min_init = min_init
        self.max_init = max_init
        self.num_bits = num_bits
        self.ema = ema
        self.ema_decay = ema_decay
        self.per_channel = per_channel
        self.out_channels = out_channels
        self.channel_axis = channel_axis
        self.quant_delay = quant_delay
        self.symmetric = symmetric
        self.narrow_range = narrow_range
        self.training = training
        self.is_ascend = context.get_context('device_target') == "Ascend"

        # init tensor min and max for fake quant op
        if self.per_channel:
            min_array = np.array([self.min_init for i in range(0, self.out_channels)]).astype(np.float32)
            max_array = np.array([self.max_init for i in range(0, self.out_channels)]).astype(np.float32)
        else:
            min_array = np.array([self.min_init]).reshape(1).astype(np.float32)
            max_array = np.array([self.max_init]).reshape(1).astype(np.float32)
        self.minq = Parameter(Tensor(min_array), name='quant_min', requires_grad=False)
        self.maxq = Parameter(Tensor(max_array), name='quant_max', requires_grad=False)

        # init fake quant relative op
        if per_channel:
            quant_fun = partial(P.FakeQuantPerChannel, channel_axis=self.channel_axis)
            ema_fun = partial(P.FakeQuantMinMaxPerChannelUpdate, channel_axis=self.channel_axis)
        else:
            quant_fun = P.FakeQuantPerLayer
            ema_fun = P.FakeQuantMinMaxPerLayerUpdate

        if self.is_ascend:
            self.fake_quant = quant_fun(num_bits=self.num_bits,
                                        symmetric=self.symmetric,
                                        narrow_range=self.narrow_range,
                                        training=self.training)
        else:
            self.fake_quant = quant_fun(num_bits=self.num_bits,
                                        ema=self.ema,
                                        ema_decay=ema_decay,
                                        quant_delay=quant_delay,
                                        symmetric=self.symmetric,
                                        narrow_range=self.narrow_range,
                                        training=self.training)
        if self.ema:
            self.ema_update = ema_fun(num_bits=self.num_bits,
                                      ema=self.ema,
                                      ema_decay=self.ema_decay,
                                      symmetric=self.symmetric,
                                      narrow_range=self.narrow_range,
                                      training=self.training)

    def extend_repr(self):
        s = 'num_bits={}, symmetric={}, narrow_range={}, ema={}({}), per_channel={}({}, {}), ' \
            'quant_delay={}, min_init={}, max_init={}'.format(
                self.num_bits, self.symmetric, self.narrow_range, self.ema, self.ema_decay, self.per_channel,
                self.channel_axis, self.out_channels, self.quant_delay, self.min_init, self.max_init)
        return s

    def construct(self, x):
        if self.ema and self.is_ascend:
            min_up, max_up = self.ema_update(x, self.minq, self.maxq)
            out = self.fake_quant(x, min_up, max_up)
            P.Assign()(self.minq, min_up)
            P.Assign()(self.maxq, max_up)
        else:
            out = self.fake_quant(x, self.minq, self.maxq)
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

    Examples:
        >>> batchnorm_quant = nn.Conv2dBatchNormQuant(1, 6, kernel_size= (2, 2), stride=(1, 1), pad_mode="valid",
        >>>                                           dilation=(1, 1))
        >>> input_x = Tensor(np.random.randint(-2, 2, (2, 1, 1, 3)), mindspore.float32)
        >>> result = batchnorm_quant(input_x)
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
        """init Conv2dBatchNormQuant layer"""
        super(Conv2dBatchNormQuant, self).__init__()
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
            validator.check_integer('group', group, in_channels, Rel.EQ, 'Conv2dBatchNormQuant')
            validator.check_integer('group', group, out_channels, Rel.EQ, 'Conv2dBatchNormQuant')
            self.conv = P.DepthwiseConv2dNative(channel_multiplier=1,
                                                kernel_size=self.kernel_size,
                                                pad_mode=pad_mode,
                                                pad=padding,
                                                stride=self.stride,
                                                dilation=self.dilation)
            if weight_init is None:
                weight_init = initializer('normal', [1, in_channels, *self.kernel_size])
            channel_axis = 1
        else:
            self.conv = P.Conv2D(out_channel=out_channels,
                                 kernel_size=self.kernel_size,
                                 pad_mode=pad_mode,
                                 pad=padding,
                                 stride=self.stride,
                                 dilation=self.dilation,
                                 group=group)
            if weight_init is None:
                weight_init = initializer('normal', [out_channels, in_channels // group, *self.kernel_size])
            channel_axis = 0
        self.weight = Parameter(weight_init, name='weight')

        # initialize batchnorm Parameter
        if gamma_init is None:
            gamma_init = initializer('ones', [out_channels])
        self.gamma = Parameter(gamma_init, name='gamma')
        if beta_init is None:
            beta_init = initializer('zeros', [out_channels])
        self.beta = Parameter(beta_init, name='beta')
        if mean_init is None:
            mean_init = initializer('zeros', [out_channels])
        self.moving_mean = Parameter(mean_init, name='moving_mean', requires_grad=False)
        if var_init is None:
            var_init = initializer('ones', [out_channels])
        self.moving_variance = Parameter(var_init, name='moving_variance', requires_grad=False)

        # initialize fake ops
        self.fake_quant_weight = FakeQuantWithMinMax(min_init=-6,
                                                     max_init=6,
                                                     ema=False,
                                                     num_bits=num_bits,
                                                     quant_delay=quant_delay,
                                                     per_channel=per_channel,
                                                     out_channels=out_channels,
                                                     symmetric=symmetric,
                                                     narrow_range=narrow_range)
        self.batchnorm_fold = BatchNormFoldCell(epsilon=eps, momentum=momentum, freeze_bn=freeze_bn)
        self.correct_mul = P.CorrectionMul(channel_axis)
        if context.get_context('device_target') == "Ascend":
            self.batchnorm_fold2_train = P.BatchNormFold2_D(freeze_bn=freeze_bn)
            self.batchnorm_fold2_infer = P.BatchNormFold2_D(freeze_bn=0)
        elif context.get_context('device_target') == "GPU":
            self.batchnorm_fold2_train = P.BatchNormFold2(freeze_bn=freeze_bn)
            self.batchnorm_fold2_infer = P.BatchNormFold2(freeze_bn=0)
        else:
            raise ValueError("Unsupported platform: {}".format(context.get_context('device_target')))
        self.step = Parameter(initializer('normal', [1], dtype=mstype.int32), name='step', requires_grad=False)
        self.one = Tensor(1, mstype.int32)
        self.assignadd = P.AssignAdd()

    def extend_repr(self):
        s = 'in_channels={}, out_channels={}, kernel_size={}, stride={}, ' \
            'pad_mode={}, padding={}, dilation={}, group={}, ' \
            'fake={}, freeze_bn={}, momentum={}, quant_delay={}'.format(
                self.in_channels, self.out_channels, self.kernel_size, self.stride,
                self.pad_mode, self.padding, self.dilation, self.group,
                self.fake, self.freeze_bn, self.momentum, self.quant_delay)
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


class Conv2dQuant(Cell):
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
            Default: None.
        bias_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the bias vector. Default: None.
        quant_delay (int): Quantization delay parameters according by global step. Default: 0.
        num_bits (int): Quantization number bit, support 4 and 8bit. Default: 8.
        per_channel (bool): FakeQuantWithMinMax Parameters. Default: False.
        symmetric (bool): Quantization algorithm use symmetric or not. Default: False.
        narrow_range (bool): Quantization algorithm use narrow range or not. Default: False.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor of shape :math:`(N, C_{out}, H_{out}, W_{out})`.

    Examples:
        >>> conv2d_quant = nn.Conv2dQuant(1, 6, kernel_size= (2, 2), stride=(1, 1), pad_mode="valid",
        >>>                               dilation=(1, 1))
        >>> input_x = Tensor(np.random.randint(-2, 2, (2, 1, 1, 3)), mindspore.float32)
        >>> result = conv2d_quant(input_x)
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
                 weight_init=None,
                 bias_init=None,
                 quant_delay=0,
                 num_bits=8,
                 per_channel=False,
                 symmetric=False,
                 narrow_range=False):
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

        if weight_init is None:
            weight_init = initializer(
                'normal', [out_channels, in_channels // group, *self.kernel_size])
        self.weight = Parameter(weight_init, name='weight')
        if bias_init is None:
            bias_init = initializer('zeros', [out_channels])
        if has_bias:
            self.bias = Parameter(bias_init, name='bias')
            self.bias_add = P.BiasAdd()

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
                                                     num_bits=num_bits,
                                                     quant_delay=quant_delay,
                                                     per_channel=per_channel,
                                                     out_channels=out_channels,
                                                     symmetric=symmetric,
                                                     narrow_range=narrow_range)

    def construct(self, x):
        weight = self.fake_quant_weight(self.weight)
        out = self.conv(x, weight)
        if self.has_bias:
            return self.bias_add(out, self.bias)
        return out

    def extend_repr(self):
        s = 'in_channels={}, out_channels={}, kernel_size={}, stride={}, ' \
            'pad_mode={}, padding={}, dilation={}, group={}, ' \
            'has_bias={}, quant_delay={}'.format(
                self.in_channels, self.out_channels, self.kernel_size, self.stride,
                self.pad_mode, self.padding, self.dilation, self.group,
                self.has_bias, self.quant_delay)
        return s


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
        self.fake_quant_weight = FakeQuantWithMinMax(min_init=-6,
                                                     max_init=6,
                                                     ema=False,
                                                     num_bits=num_bits,
                                                     quant_delay=quant_delay,
                                                     per_channel=per_channel,
                                                     out_channels=out_channels,
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
        ema_decay (float): Exponential Moving Average algorithm parameter. Default: 0.999.
        symmetric (bool): Quantization algorithm use symmetric or not. Default: False.
        narrow_range (bool): Quantization algorithm use narrow range or not. Default: False.

    Inputs:
        - **x** (Tensor) - The input of ReLUQuant.

    Outputs:
        Tensor, with the same type and shape as the `x`.

    Examples:
        >>> relu_quant = nn.ReLUQuant()
        >>> input_x = Tensor(np.array([[1, 2, 0], [-1, -2, 1]]), mindspore.float32)
        >>> result = relu_quant(input_x)
    """

    def __init__(self,
                 num_bits=8,
                 quant_delay=0,
                 ema_decay=0.999,
                 symmetric=False,
                 narrow_range=False):
        super(ReLUQuant, self).__init__()
        self.fake_quant_act = FakeQuantWithMinMax(min_init=0,
                                                  max_init=6,
                                                  num_bits=num_bits,
                                                  quant_delay=quant_delay,
                                                  ema=True,
                                                  ema_decay=ema_decay,
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
        ema_decay (float): Exponential Moving Average algorithm parameter. Default: 0.999.
        symmetric (bool): Quantization algorithm use symmetric or not. Default: False.
        narrow_range (bool): Quantization algorithm use narrow range or not. Default: False.

    Inputs:
        - **x** (Tensor) - The input of ReLU6Quant.

    Outputs:
        Tensor, with the same type and shape as the `x`.

    Examples:
        >>> relu6_quant = nn.ReLU6Quant(4, 1)
        >>> input_x = Tensor(np.array([[1, 2, -1], [-2, 0, -1]]), mindspore.float32)
        >>> result = relu6_quant(input_x)
    """

    def __init__(self,
                 num_bits=8,
                 quant_delay=0,
                 ema_decay=0.999,
                 symmetric=False,
                 narrow_range=False):
        super(ReLU6Quant, self).__init__()
        self.fake_quant_act = FakeQuantWithMinMax(min_init=0,
                                                  max_init=6,
                                                  num_bits=num_bits,
                                                  quant_delay=quant_delay,
                                                  ema=True,
                                                  ema_decay=ema_decay,
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
        ema_decay (float): Exponential Moving Average algorithm parameter. Default: 0.999.
        symmetric (bool): Quantization algorithm use symmetric or not. Default: False.
        narrow_range (bool): Quantization algorithm use narrow range or not. Default: False.

    Inputs:
        - **x** (Tensor) - The input of HSwishQuant.

    Outputs:
        Tensor, with the same type and shape as the `x`.

    Examples:
        >>> hswish_quant = nn.HSwishQuant(4, 1)
        >>> input_x = Tensor(np.array([[1, 2, 1], [-2, 0, -1]]), mindspore.float32)
        >>> result = hswish_quant(input_x)
    """

    def __init__(self,
                 num_bits=8,
                 quant_delay=0,
                 ema_decay=0.999,
                 symmetric=False,
                 narrow_range=False):
        super(HSwishQuant, self).__init__()
        self.fake_quant_act_before = FakeQuantWithMinMax(min_init=-6,
                                                         max_init=6,
                                                         num_bits=num_bits,
                                                         quant_delay=quant_delay,
                                                         ema=True,
                                                         ema_decay=ema_decay,
                                                         symmetric=symmetric,
                                                         narrow_range=narrow_range)
        self.fake_quant_act_after = FakeQuantWithMinMax(min_init=-6,
                                                        max_init=6,
                                                        num_bits=num_bits,
                                                        quant_delay=quant_delay,
                                                        ema=True,
                                                        ema_decay=ema_decay,
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
        ema_decay (float): Exponential Moving Average algorithm parameter. Default: 0.999.
        symmetric (bool): Quantization algorithm use symmetric or not. Default: False.
        narrow_range (bool): Quantization algorithm use narrow range or not. Default: False.

    Inputs:
        - **x** (Tensor) - The input of HSigmoidQuant.

    Outputs:
        Tensor, with the same type and shape as the `x`.

    Examples:
        >>> hsigmoid_quant = nn.HSigmoidQuant(4, 1)
        >>> input_x = Tensor(np.array([[1, 2, 1], [-2, 0, -1]]), mindspore.float32)
        >>> result = hsigmoid_quant(input_x)
    """

    def __init__(self,
                 num_bits=8,
                 quant_delay=0,
                 ema_decay=0.999,
                 symmetric=False,
                 narrow_range=False):
        super(HSigmoidQuant, self).__init__()
        self.fake_quant_act_before = FakeQuantWithMinMax(min_init=-6,
                                                         max_init=6,
                                                         num_bits=num_bits,
                                                         quant_delay=quant_delay,
                                                         ema=True,
                                                         symmetric=symmetric,
                                                         narrow_range=narrow_range)
        self.fake_quant_act_after = FakeQuantWithMinMax(min_init=-6,
                                                        max_init=6,
                                                        num_bits=num_bits,
                                                        quant_delay=quant_delay,
                                                        ema=True,
                                                        ema_decay=ema_decay,
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
        ema_decay (float): Exponential Moving Average algorithm parameter. Default: 0.999.
        symmetric (bool): Quantization algorithm use symmetric or not. Default: False.
        narrow_range (bool): Quantization algorithm use narrow range or not. Default: False.

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
                 num_bits=8,
                 quant_delay=0,
                 ema_decay=0.999,
                 symmetric=False,
                 narrow_range=False):
        super(TensorAddQuant, self).__init__()
        self.fake_quant_act = FakeQuantWithMinMax(min_init=-6,
                                                  max_init=6,
                                                  num_bits=num_bits,
                                                  quant_delay=quant_delay,
                                                  ema=True,
                                                  ema_decay=ema_decay,
                                                  symmetric=symmetric,
                                                  narrow_range=narrow_range)
        self.add = P.TensorAdd()

    def construct(self, x1, x2):
        x = self.add(x1, x2)
        x = self.fake_quant_act(x)
        return x


class MulQuant(Cell):
    r"""
    Add Fake Quant OP after Mul OP.

    For a more Detailed overview of Mul op.

    Args:
        num_bits (int): Quantization number bit, support 4 and 8bit. Default: 8.
        quant_delay (int): Quantization delay parameters according by global step. Default: 0.
        ema_decay (float): Exponential Moving Average algorithm parameter. Default: 0.999.
        symmetric (bool): Quantization algorithm use symmetric or not. Default: False.
        narrow_range (bool): Quantization algorithm use narrow range or not. Default: False.

    Inputs:
        - **x** (Tensor) - The input of MulQuant.

    Outputs:
        Tensor, with the same type and shape as the `x`.

    """

    def __init__(self,
                 num_bits=8,
                 quant_delay=0,
                 ema_decay=0.999,
                 symmetric=False,
                 narrow_range=False):
        super(MulQuant, self).__init__()
        self.fake_quant_act = FakeQuantWithMinMax(min_init=-6,
                                                  max_init=6,
                                                  num_bits=num_bits,
                                                  quant_delay=quant_delay,
                                                  ema=True,
                                                  ema_decay=ema_decay,
                                                  symmetric=symmetric,
                                                  narrow_range=narrow_range)
        self.mul = P.Mul()

    def construct(self, x1, x2):
        x = self.mul(x1, x2)
        x = self.fake_quant_act(x)
        return x
