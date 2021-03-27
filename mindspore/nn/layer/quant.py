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
"""Quantization aware training."""

from functools import partial
from collections import namedtuple
import numpy as np
import mindspore.common.dtype as mstype
from mindspore.ops.primitive import Primitive
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.common.parameter import Parameter
from mindspore.common.initializer import initializer
from mindspore.common.tensor import Tensor
from mindspore._checkparam import Validator, Rel, twice
from mindspore.compression.common import QuantDtype
import mindspore.context as context
from .normalization import BatchNorm2d
from .activation import get_activation, ReLU
from ..cell import Cell
from ...ops.operations import _quant_ops as Q

__all__ = [
    'FakeQuantWithMinMaxObserver',
    'Conv2dBnFoldQuantOneConv',
    'Conv2dBnFoldQuant',
    'Conv2dBnWithoutFoldQuant',
    'Conv2dQuant',
    'DenseQuant',
    'ActQuant',
    'TensorAddQuant',
    'MulQuant',
]


class BatchNormFoldCell(Cell):
    """
    Batch normalization folded.

    Args:
        momentum (float): Momentum value must be [0, 1]. Default: 0.9.
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
        """Initialize batch norm fold layer"""
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
                running_mean = P.Add()(mean, 0.)
                running_std = P.Sqrt()(P.Add()(variance, self.epsilon))
        return batch_mean, batch_std, running_mean, running_std


def _partial_init(cls_or_self, **kwargs):
    """
    Wrapper that allows creation of class factories.

    This can be useful when there is a need to create classes with the same
    constructor arguments, but different instances.

    Examples:
        >>> class Foo:
        ...     def __init__(self, a, b, answer):
        ...         pass
        >>> Foo.partial_init = classmethod(_partial_init)
        >>> foo_builder = Foo.partial_init(a=3, b=4).partial_init(answer=42)
        >>> foo_instance1 = foo_builder()
        >>> foo_instance2 = foo_builder()
        >>> result = (id(foo_instance1) == id(foo_instance2))
        >>> print(result)
        False
    """

    class _PartialWrapper:
        r"""
        class of wrapper that allows creation of class factories.
        """

        def __init__(self, p):
            self.p = p

        def __call__(self, *args, **keywords):
            return self.p(*args, **keywords)

        def __repr__(self):
            return self.p.__repr__()

        partial_init = _partial_init

    r = _PartialWrapper(partial(cls_or_self, **kwargs))
    return r


class _Observer(Cell):
    """
    Base class of Observer. Observer is used to calculate the statistics of specific layer.

    Notes:
        This class is an abstract class.

    Args:
        quant_dtype (QuantDtype): The type of FakeQuant data.
    """

    def __init__(self, quant_dtype):
        super(_Observer, self).__init__()
        self.quant_dtype = quant_dtype

    def extend_repr(self):
        s = f"quant_dtype={self.quant_dtype}"
        return s

    def construct(self):
        pass

    partial_init = classmethod(_partial_init)


class UniformQuantObserver(_Observer):
    """
    The base class of Uniform Quantization Observer.

    Args:
        quant_dtype (QuantDtype): The type of FakeQuant data. Default: QuantDtype.INT8.
        per_channel (bool):  Quantization granularity based on layer or on channel. Default: False.
        symmetric (bool): Whether the quantization algorithm is symmetric or not. Default: False.
        narrow_range (bool): Whether the quantization algorithm uses narrow range or not. Default: False.
        num_channels (int): declarate the min and max channel size, Default: 1.

    Returns:
        Tensor.
    """

    min_max_map = {
        QuantDtype.INT2: (-2, 1),
        QuantDtype.INT3: (-4, 3),
        QuantDtype.INT4: (-8, 7),
        QuantDtype.INT5: (-16, 15),
        QuantDtype.INT6: (-32, 31),
        QuantDtype.INT7: (-64, 63),
        QuantDtype.INT8: (-128, 127),

        QuantDtype.UINT2: (0, 3),
        QuantDtype.UINT3: (0, 7),
        QuantDtype.UINT4: (0, 15),
        QuantDtype.UINT5: (0, 31),
        QuantDtype.UINT6: (0, 63),
        QuantDtype.UINT7: (0, 127),
        QuantDtype.UINT8: (0, 255)
    }

    def __init__(self, quant_dtype=QuantDtype.INT8, per_channel=False, symmetric=False, narrow_range=False,
                 num_channels=1):
        super(UniformQuantObserver, self).__init__(quant_dtype)
        self.per_channel = per_channel
        self.symmetric = symmetric
        self.narrow_range = narrow_range
        self.num_channels = num_channels


class FakeQuantWithMinMaxObserver(UniformQuantObserver):
    r"""
    Quantization aware operation which provides the fake quantization observer function on data with min and max.

    The running min/max :math:`x_{min}` and :math:`x_{max}` are computed as:

    .. math::

        \begin{array}{ll} \\
            x_{min} =
            \begin{cases}
                \min(\min(X), 0)
                  & \text{ if } ema = \text{False} \\
                \min((1 - c) \min(X) + \text{c } x_{min}, 0)
                  & \text{ if } \text{otherwise}
            \end{cases}\\
            x_{max} =
            \begin{cases}
                \max(\max(X), 0)
                  & \text{ if } ema = \text{False} \\
                \max((1 - c) \max(X) + \text{c } x_{max}, 0)
                  & \text{ if } \text{otherwise}
            \end{cases}
        \end{array}

    where X is the input tensor, and :math:`c` is the `ema_decay`.

    The scale and zero point zp is computed as:

    .. math::

        \begin{array}{ll} \\
            scale =
            \begin{cases}
                \frac{x_{max} - x_{min}}{Q_{max} - Q_{min}}
                  & \text{ if } symmetric = \text{False} \\
                \frac{2\max(x_{max}, \left | x_{min} \right |) }{Q_{max} - Q_{min}}
                  & \text{ if } \text{otherwise}
            \end{cases}\\
            zp\_min = Q_{min} - \frac{x_{min}}{scale} \\
            zp = \left \lfloor \min(Q_{max}, \max(Q_{min}, zp\_min)) + 0.5 \right \rfloor
        \end{array}

    where :math:`Q_{max}` and :math:`Q_{min}` is decided by quant_dtype, for example, if quant_dtype=INT8,
    then :math:`Q_{max} = 127` and :math:`Q_{min} = -128`.

    The fake quant output is computed as:

    .. math::

        \begin{array}{ll} \\
            u_{min} = (Q_{min} - zp) * scale \\
            u_{max} = (Q_{max} - zp) * scale \\
            u_X = \left \lfloor \frac{\min(u_{max}, \max(u_{min}, X)) - u_{min}}{scale}
            + 0.5 \right \rfloor \\
            output = u_X * scale + u_{min}
        \end{array}


    Args:
        min_init (int, float): The initialized min value. Default: -6.
        max_init (int, float): The initialized max value. Default: 6.
        ema (bool): The exponential Moving Average algorithm updates min and max. Default: False.
        ema_decay (float): Exponential Moving Average algorithm parameter. Default: 0.999.
        per_channel (bool):  Quantization granularity based on layer or on channel. Default: False.
        channel_axis (int): Quantization by channel axis. Default: 1.
        num_channels (int): declarate the min and max channel size, Default: 1.
        quant_dtype (QuantDtype): The datatype of quantization, supporting 4 and 8bits. Default: QuantDtype.INT8.
        symmetric (bool): Whether the quantization algorithm is symmetric or not. Default: False.
        narrow_range (bool): Whether the quantization algorithm uses narrow range or not. Default: False.
        quant_delay (int): Quantization delay parameters according to the global step. Default: 0.

    Inputs:
        - **input** (Tensor) - The input of FakeQuantWithMinMaxObserver.

    Outputs:
        Tensor, with the same type and shape as the `input`.

    Raises:
        TypeError: If `min_init` or `max_init` is neither int nor float.
        TypeError: If `quant_delay` is not an int.
        TypeError: If `min_init` is not less than `max_init`.
        TypeError: If `quant_delay` is not greater than or equal to 0.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> fake_quant = nn.FakeQuantWithMinMaxObserver()
        >>> input = Tensor(np.array([[1, 2, 1], [-2, 0, -1]]), mindspore.float32)
        >>> output = fake_quant(input)
        >>> print(output)
        [[ 0.9882355  1.9764705  0.9882355]
         [-1.9764705  0.        -0.9882355]]
    """

    def __init__(self,
                 min_init=-6,
                 max_init=6,
                 ema=False,
                 ema_decay=0.999,
                 per_channel=False,
                 channel_axis=1,
                 num_channels=1,
                 quant_dtype=QuantDtype.INT8,
                 symmetric=False,
                 narrow_range=False,
                 quant_delay=0):
        """Initialize FakeQuantWithMinMaxObserver"""
        super(FakeQuantWithMinMaxObserver, self).__init__(quant_dtype=quant_dtype, per_channel=per_channel,
                                                          symmetric=symmetric, narrow_range=narrow_range,
                                                          num_channels=num_channels)
        Validator.check_value_type("min_init", min_init, [int, float], type(self).__name__)
        Validator.check_value_type("max_init", max_init, [int, float], type(self).__name__)
        Validator.check("min_init", min_init, "max_init", max_init, rel=Rel.LT)
        Validator.check_non_negative_int(quant_delay, 'quant_delay')
        self.min_init = min_init
        self.max_init = max_init
        self.quant_dtype = quant_dtype
        self.ema = ema
        self.ema_decay = ema_decay
        self.per_channel = per_channel
        self.num_channels = num_channels
        self.channel_axis = channel_axis
        self.quant_delay = quant_delay
        self.symmetric = symmetric
        self.narrow_range = narrow_range
        self.is_ascend = context.get_context('device_target') == "Ascend"

        # init tensor min and max for fake quantized operation
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
            self.fake_quant_train = quant_fun(num_bits=self.quant_dtype.num_bits,
                                              symmetric=self.symmetric,
                                              narrow_range=self.narrow_range,
                                              quant_delay=self.quant_delay)
            self.fake_quant_infer = self.fake_quant_train
        else:
            quant_fun = partial(quant_fun,
                                ema=self.ema,
                                ema_decay=ema_decay,
                                num_bits=self.quant_dtype.num_bits,
                                symmetric=self.symmetric,
                                narrow_range=self.narrow_range,
                                quant_delay=self.quant_delay)
            self.fake_quant_train = quant_fun(training=True)
            self.fake_quant_infer = quant_fun(training=False)

    def extend_repr(self):
        s = 'quant_dtype={}, symmetric={}, narrow_range={}, ema={}({}), per_channel={}({}, {}), ' \
            'quant_delay={}, min_init={}, max_init={}'.format(self.quant_dtype, self.symmetric, self.narrow_range,
                                                              self.ema, self.ema_decay, self.per_channel,
                                                              self.channel_axis, self.num_channels, self.quant_delay,
                                                              self.min_init, self.max_init)
        return s

    def construct(self, x):
        if self.training:
            min_up, max_up = self.ema_update(x, self.minq, self.maxq)
            self.minq = min_up
            self.maxq = max_up
            out = self.fake_quant_train(x, self.minq, self.maxq)
        else:
            out = self.fake_quant_infer(x, self.minq, self.maxq)
        return out


QuantConfig = namedtuple("QuantConfig", ['weight', 'activation'])

quant_config_default = QuantConfig(weight=FakeQuantWithMinMaxObserver, activation=FakeQuantWithMinMaxObserver)


class Conv2dBnFoldQuantOneConv(Cell):
    r"""
    2D convolution which use the convolution layer statistics once to calculate BatchNormal operation folded construct.

    This part is a more detailed overview of Conv2d operation. For more detials about Quantilization,
    please refer to :class:`mindspore.nn.FakeQuantWithMinMaxObserver`.

    Args:
        in_channels (int): The number of input channel :math:`C_{in}`.
        out_channels (int): The number of output channel :math:`C_{out}`.
        kernel_size (Union[int, tuple]): Specifies the height and width of the 2D convolution window.
        stride (int): Specifies stride for all spatial dimensions with the same value.
        pad_mode (str): Specifies padding mode. The optional values are "same", "valid", "pad". Default: "same".
        padding (int): Implicit paddings on both sides of the input. Default: 0.
        eps (float): Parameters for BatchNormal. Default: 1e-5.
        momentum (float): Parameters for BatchNormal op. Default: 0.997.
        dilation (int): Specifies the dilation rate to use for dilated convolution. Default: 1.
        group (int): Splits filter into groups, `in_ channels` and `out_channels` must be
            divisible by the number of groups. Default: 1.
        has_bias (bool): Specifies whether the layer uses a bias vector. Default: False.
        weight_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the
            convolution kernel. Default: 'normal'.
        bias_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the
            bias vector. Default: 'zeros'.
        beta_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the
            beta vector. Default: 'zeros'.
        gamma_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the
            gamma vector. Default: 'ones'.
        mean_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the
            mean vector. Default: 'zeros'.
        var_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the
            variance vector. Default: 'ones'.
        fake (bool): Whether Conv2dBnFoldQuant Cell adds FakeQuantWithMinMaxObserver. Default: True.
        quant_config (QuantConfig): Configures the oberser types and quant settings of weight and activation. Can be
            generated by compression.quant.create_quant_config method.
            Default: both set to default FakeQuantWithMinMaxObserver.
        quant_dtype (QuantDtype): Specifies the FakeQuant datatype. Default: QuantDtype.INT8.

    Inputs:
        - **input** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor of shape :math:`(N, C_{out}, H_{out}, W_{out})`.

    Raises:
        TypeError: If `in_channels`, `out_channels`, `stride`, `padding` or `dilation` is not an int.
        TypeError: If `has_bias` is not a bool.
        ValueError: If `in_channels` or `out_channels` `stride`, `padding` or `dilation` is less than 1.
        ValueError: If `pad_mode` is not one of 'same', 'valid', 'pad'.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> qconfig = compression.quant.create_quant_config()
        >>> conv2d_bnfold = nn.Conv2dBnFoldQuantOneConv(1, 6, kernel_size=(2, 2), stride=(1, 1), pad_mode="valid",
        ...                                             quant_config=qconfig)
        >>> input = Tensor(np.random.randint(-2, 2, (2, 1, 3, 3)), mindspore.float32)
        >>> result = conv2d_bnfold(input)
        >>> output = result.shape
        >>> print(output)
        (2, 6, 2, 2)
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
                 has_bias=False,
                 weight_init='normal',
                 bias_init='zeros',
                 beta_init='zeros',
                 gamma_init='ones',
                 mean_init='zeros',
                 var_init='ones',
                 fake=True,
                 quant_config=quant_config_default,
                 quant_dtype=QuantDtype.INT8):
        """Initialize Conv2dBnFoldQuant layer"""
        super(Conv2dBnFoldQuantOneConv, self).__init__()
        self.in_channels = Validator.check_positive_int(in_channels)
        self.out_channels = Validator.check_positive_int(out_channels)
        self.kernel_size = twice(kernel_size)
        self.stride = twice(stride)
        self.pad_mode = pad_mode
        self.padding = padding
        self.dilation = twice(dilation)
        self.group = group
        self.eps = eps
        self.momentum = momentum
        self.has_bias = has_bias
        self.fake = fake
        self.quant_config = quant_config
        self.quant_dtype = quant_dtype
        data_format = 'NCHW'
        self.format = Validator.check_string(data_format, ['NCHW', 'NHWC'], 'format', self.cls_name)
        self._target = context.get_context("device_target")
        self.is_graph_mode = context.get_context("mode") == context.GRAPH_MODE
        if context.get_context("enable_ge"):
            self.is_ge_backend = True
        else:
            self.is_ge_backend = False
        self.enable_default_train = self.is_graph_mode and \
                                    (self.is_ge_backend or self._target == "Ascend")

        # initialize convolution op and Parameter
        self.conv = P.Conv2D(out_channel=out_channels,
                             kernel_size=self.kernel_size,
                             pad_mode=pad_mode,
                             pad=padding,
                             stride=self.stride,
                             dilation=self.dilation,
                             group=group)
        weight_shape = [out_channels, in_channels // group, *self.kernel_size]
        channel_axis = 0
        self.channel_axis = channel_axis
        self.weight = Parameter(initializer(weight_init, weight_shape), name='weight')
        self.bias_add = P.BiasAdd()
        if Validator.check_bool(has_bias):
            self.bias = Parameter(initializer(bias_init, [out_channels]), name='bias')
        else:
            self.bias = None

        # initialize BatchNorm Parameter
        self.gamma = Parameter(initializer(gamma_init, [out_channels]), name='gamma')
        self.beta = Parameter(initializer(beta_init, [out_channels]), name='beta')
        self.moving_mean = Parameter(initializer(mean_init, [out_channels]), name='moving_mean', requires_grad=False)
        self.moving_variance = Parameter(initializer(var_init, [out_channels]), name='moving_variance',
                                         requires_grad=False)

        # initialize fake ops
        self.fake_quant_weight = quant_config.weight(min_init=-6,
                                                     max_init=6,
                                                     ema=False,
                                                     channel_axis=channel_axis,
                                                     num_channels=out_channels,
                                                     quant_dtype=quant_dtype)
        self.bn_train = P.BatchNorm(is_training=True, epsilon=self.eps,
                                    momentum=self.momentum, data_format=self.format)

        self.bn_infer = P.BatchNorm(is_training=False, epsilon=self.eps, data_format=self.format)
        data_parallel_strategy = ((1,), (1,))
        data_parallel_strategy_one = ((1,), ())
        self.sub_mean = P.Sub().shard(data_parallel_strategy)
        self.sub_var = P.Sub().shard(data_parallel_strategy)
        self.mul_mean = P.Mul().shard(data_parallel_strategy_one)
        self.mul_var = P.Mul().shard(data_parallel_strategy_one)
        self.assign_sub_mean = P.AssignSub().shard(data_parallel_strategy)
        self.assign_sub_var = P.AssignSub().shard(data_parallel_strategy)
        self.reshape = P.Reshape()

    def extend_repr(self):
        s = 'in_channels={}, out_channels={}, kernel_size={}, stride={}, ' \
            'pad_mode={}, padding={}, dilation={}, group={}, ' \
            'fake={}, momentum={}, quant_delay={}'.format(self.in_channels, self.out_channels,
                                                          self.kernel_size, self.stride,
                                                          self.pad_mode, self.padding, self.dilation,
                                                          self.group,
                                                          self.fake, self.momentum,
                                                          self.fake_quant_weight.quant_delay)
        return s

    def construct(self, x):
        running_std = P.Sqrt()(P.Add()(self.moving_variance, self.eps))
        scale_factor = self.gamma / running_std
        if self.channel_axis:
            scale_factor = self.reshape(scale_factor, (1, -1, 1, 1))
        else:
            scale_factor = self.reshape(scale_factor, (-1, 1, 1, 1))
        weight = self.weight * scale_factor
        if self.fake:
            weight = self.fake_quant_weight(weight)
        conv = self.conv(x, weight)
        scale_factor = self.reshape(scale_factor, (1, -1, 1, 1))
        if self.enable_default_train:
            scale_factor = P.Reciprocal()(scale_factor)
            conv_orig = conv * scale_factor
        else:
            conv_orig = conv / scale_factor
        if self.training:
            return self.bn_train(conv_orig,
                                 self.gamma,
                                 self.beta,
                                 self.moving_mean,
                                 self.moving_variance)[0]

        return self.bn_infer(conv_orig,
                             self.gamma,
                             self.beta,
                             self.moving_mean,
                             self.moving_variance)[0]


class Conv2dBnFoldQuant(Cell):
    r"""
    2D convolution with BatchNormal operation folded construct.

    This part is a more detailed overview of Conv2d operation. For more detials about Quantilization,
    please refer to :class:`mindspore.nn.FakeQuantWithMinMaxObserver`.

    Args:
        in_channels (int): The number of input channel :math:`C_{in}`.
        out_channels (int): The number of output channel :math:`C_{out}`.
        kernel_size (Union[int, tuple]): Specifies the height and width of the 2D convolution window.
        stride (int): Specifies stride for all spatial dimensions with the same value.
        pad_mode (str): Specifies padding mode. The optional values are "same", "valid", "pad". Default: "same".
        padding (int): Implicit paddings on both sides of the input. Default: 0.
        eps (float): Parameters for BatchNormal. Default: 1e-5.
        momentum (float): Parameters for BatchNormal op. Default: 0.997.
        dilation (int): Specifies the dilation rate to use for dilated convolution. Default: 1.
        group (int): Splits filter into groups, `in_ channels` and `out_channels` must be
            divisible by the number of groups. Default: 1.
        has_bias (bool): Specifies whether the layer uses a bias vector. Default: False.
        weight_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the
            convolution kernel. Default: 'normal'.
        bias_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the
            bias vector. Default: 'zeros'.
        beta_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the
            beta vector. Default: 'zeros'.
        gamma_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the
            gamma vector. Default: 'ones'.
        mean_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the
            mean vector. Default: 'zeros'.
        var_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the
            variance vector. Default: 'ones'.
        fake (bool): Whether Conv2dBnFoldQuant Cell adds FakeQuantWithMinMaxObserver. Default: True.
        quant_config (QuantConfig): Configures the oberser types and quant settings of weight and activation. Can be
            generated by compression.quant.create_quant_config method.
            Default: both set to default FakeQuantWithMinMaxObserver.
        quant_dtype (QuantDtype): Specifies the FakeQuant datatype. Default: QuantDtype.INT8.
        freeze_bn (int): The quantization freeze BatchNormal op is according to the global step. Default: 100000.

    Inputs:
        - **input** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor of shape :math:`(N, C_{out}, H_{out}, W_{out})`.

    Raises:
        TypeError: If `in_channels`, `out_channels`, `stride`, `padding` or `dilation` is not an int.
        TypeError: If `has_bias` is not a bool.
        ValueError: If `in_channels` or `out_channels` `stride`, `padding` or `dilation` is less than 1.
        ValueError: If `pad_mode` is not one of 'same', 'valid', 'pad'.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> qconfig = compression.quant.create_quant_config()
        >>> conv2d_bnfold = nn.Conv2dBnFoldQuant(1, 6, kernel_size=(2, 2), stride=(1, 1), pad_mode="valid",
        ...                                      quant_config=qconfig)
        >>> input = Tensor(np.random.randint(-2, 2, (2, 1, 3, 3)), mindspore.float32)
        >>> output = conv2d_bnfold(input)
        >>> print(output.shape)
        (2, 6, 2, 2)
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
                 has_bias=False,
                 weight_init='normal',
                 bias_init='zeros',
                 beta_init='zeros',
                 gamma_init='ones',
                 mean_init='zeros',
                 var_init='ones',
                 fake=True,
                 quant_config=quant_config_default,
                 quant_dtype=QuantDtype.INT8,
                 freeze_bn=100000):
        """Initialize Conv2dBnFoldQuant layer"""
        super(Conv2dBnFoldQuant, self).__init__()
        self.in_channels = Validator.check_positive_int(in_channels)
        self.out_channels = Validator.check_positive_int(out_channels)
        self.kernel_size = twice(kernel_size)
        self.stride = twice(stride)
        self.pad_mode = pad_mode
        self.padding = padding
        self.dilation = twice(dilation)
        self.group = group
        self.eps = eps
        self.momentum = momentum
        self.has_bias = has_bias
        self.freeze_bn = freeze_bn
        self.fake = fake
        self.quant_config = quant_config
        self.quant_dtype = quant_dtype
        self.is_gpu = context.get_context('device_target') == "GPU"

        # initialize convolution op and Parameter
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
        self.bias_add = P.BiasAdd()
        if Validator.check_bool(has_bias):
            self.bias = Parameter(initializer(bias_init, [out_channels]), name='bias')
        else:
            self.bias = None

        # initialize BatchNorm Parameter
        self.gamma = Parameter(initializer(gamma_init, [out_channels]), name='gamma')
        self.beta = Parameter(initializer(beta_init, [out_channels]), name='beta')
        self.moving_mean = Parameter(initializer(mean_init, [out_channels]), name='moving_mean', requires_grad=False)
        self.moving_variance = Parameter(initializer(var_init, [out_channels]), name='moving_variance',
                                         requires_grad=False)

        # initialize fake ops
        self.fake_quant_weight = quant_config.weight(min_init=-6,
                                                     max_init=6,
                                                     ema=False,
                                                     channel_axis=channel_axis,
                                                     num_channels=out_channels,
                                                     quant_dtype=quant_dtype)
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
                                                                        self.fake_quant_weight.quant_delay)
        return s

    def construct(self, x):
        out_conv = self.conv(x, self.weight)
        if self.has_bias:
            out_conv = self.bias_add(out_conv, self.bias)
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
        if self.has_bias:
            out = self.bias_add(out, self.bias)
        # BN fold2
        if self.is_gpu:
            if self.training:
                out = self.batchnorm_fold2_train(out, self.beta, self.gamma,
                                                 batch_std, batch_mean, running_std, running_mean, self.step)
                self.assignadd(self.step, self.one)
            else:
                out = self.batchnorm_fold2_infer(out, self.beta, self.gamma,
                                                 batch_std, batch_mean, running_std, running_mean, self.step)
        else:
            if self.training:
                out = self.batchnorm_fold2_train(out, self.beta, self.gamma, batch_std, batch_mean, running_std)
                self.assignadd(self.step, self.one)
            else:
                out = self.batchnorm_fold2_infer(out, self.beta, self.gamma, running_std, running_mean, running_std)
        return out


class Conv2dBnWithoutFoldQuant(Cell):
    r"""
    2D convolution and batchnorm without fold with fake quantized construct.

    This part is a more detailed overview of Conv2d operation. For more detials about Quantilization,
    please refer to :class:`mindspore.nn.FakeQuantWithMinMaxObserver`.

    Args:
        in_channels (int): The number of input channel :math:`C_{in}`.
        out_channels (int): The number of output channel :math:`C_{out}`.
        kernel_size (Union[int, tuple]): Specifies the height and width of the 2D convolution window.
        stride (int): Specifies stride for all spatial dimensions with the same value. Default: 1.
        pad_mode (str): Specifies padding mode. The optional values are "same", "valid", "pad". Default: "same".
        padding (int): Implicit paddings on both sides of the input. Default: 0.
        dilation (int): Specifies the dilation rate to use for dilated convolution. Default: 1.
        group (int): Splits filter into groups, `in_ channels` and `out_channels` must be
            divisible by the number of groups. Default: 1.
        has_bias (bool): Specifies whether the layer uses a bias vector. Default: False.
        eps (float): Parameters for BatchNormal. Default: 1e-5.
        momentum (float): Parameters for BatchNormal op. Default: 0.997.
        weight_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the convolution kernel.
            Default: 'normal'.
        bias_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the bias vector. Default: 'zeros'.
        quant_config (QuantConfig): Configures the oberser types and quant settings of weight and activation. Can be
            generated by compression.quant.create_quant_config method.
            Default: both set to default FakeQuantWithMinMaxObserver.
        quant_dtype (QuantDtype): Specifies the FakeQuant datatype. Default: QuantDtype.INT8.

    Inputs:
        - **input** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor of shape :math:`(N, C_{out}, H_{out}, W_{out})`.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> qconfig = compression.quant.create_quant_config()
        >>> conv2d_no_bnfold = nn.Conv2dBnWithoutFoldQuant(1, 6, kernel_size=(2, 2), stride=(1, 1), pad_mode="valid",
        ...                                                quant_config=qconfig)
        >>> input = Tensor(np.random.randint(-2, 2, (2, 1, 3, 3)), mstype.float32)
        >>> output = conv2d_no_bnfold(input)
        >>> print(output.shape)
        (2, 6, 2, 2)
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
                 quant_config=quant_config_default,
                 quant_dtype=QuantDtype.INT8):
        super(Conv2dBnWithoutFoldQuant, self).__init__()
        self.in_channels = Validator.check_positive_int(in_channels)
        self.out_channels = Validator.check_positive_int(out_channels)
        self.has_bias = has_bias
        self.kernel_size = twice(kernel_size)
        self.stride = twice(stride)
        self.dilation = twice(dilation)
        self.pad_mode = pad_mode
        self.padding = padding
        self.group = group

        self.bias_add = P.BiasAdd()
        if Validator.check_bool(has_bias):
            self.bias = Parameter(initializer(bias_init, [out_channels]), name='bias')
        else:
            self.bias = None
        # initialize convolution op and Parameter
        self.conv = P.Conv2D(out_channel=self.out_channels,
                             kernel_size=self.kernel_size,
                             mode=1,
                             pad_mode=self.pad_mode,
                             pad=self.padding,
                             stride=self.stride,
                             dilation=self.dilation,
                             group=self.group)
        weight_shape = [out_channels, in_channels // group, *self.kernel_size]
        channel_axis = 0
        self.weight = Parameter(initializer(weight_init, weight_shape), name='weight')
        self.fake_quant_weight = quant_config.weight(min_init=-6,
                                                     max_init=6,
                                                     ema=False,
                                                     channel_axis=channel_axis,
                                                     num_channels=out_channels,
                                                     quant_dtype=quant_dtype)
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
                                                 self.has_bias, self.fake_quant_weight.quant_delay)
        return s


class Conv2dQuant(Cell):
    r"""
    2D convolution with fake quantized operation layer.

    This part is a more detailed overview of Conv2d operation. For more detials about Quantilization,
    please refer to :class:`mindspore.nn.FakeQuantWithMinMaxObserver`.

    Args:
        in_channels (int): The number of input channel :math:`C_{in}`.
        out_channels (int): The number of output channel :math:`C_{out}`.
        kernel_size (Union[int, tuple]): Specifies the height and width of the 2D convolution window.
        stride (int): Specifies stride for all spatial dimensions with the same value. Default: 1.
        pad_mode (str): Specifies padding mode. The optional values are "same", "valid", "pad". Default: "same".
        padding (int): Implicit paddings on both sides of the input. Default: 0.
        dilation (int): Specifies the dilation rate to use for dilated convolution. Default: 1.
        group (int): Splits filter into groups, `in_ channels` and `out_channels` must be
            divisible by the number of groups. Default: 1.
        has_bias (bool): Specifies whether the layer uses a bias vector. Default: False.
        weight_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the convolution kernel.
            Default: 'normal'.
        bias_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the bias vector. Default: 'zeros'.
        quant_config (QuantConfig): Configures the oberser types and quant settings of weight and activation. Can be
            generated by compression.quant.create_quant_config method.
            Default: both set to default FakeQuantWithMinMaxObserver.
        quant_dtype (QuantDtype): Specifies the FakeQuant datatype. Default: QuantDtype.INT8.

    Inputs:
        - **input** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor of shape :math:`(N, C_{out}, H_{out}, W_{out})`.

    Raises:
        TypeError: If `in_channels`, `out_channels`, `stride`, `padding` or `dilation` is not an int.
        TypeError: If `has_bias` is not a bool.
        ValueError: If `in_channels` or `out_channels` `stride`, `padding` or `dilation` is less than 1.
        ValueError: If `pad_mode` is not one of 'same', 'valid', 'pad'.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> qconfig = compression.quant.create_quant_config()
        >>> conv2d_quant = nn.Conv2dQuant(1, 6, kernel_size= (2, 2), stride=(1, 1), pad_mode="valid",
        ...                               quant_config=qconfig)
        >>> input = Tensor(np.random.randint(-2, 2, (2, 1, 3, 3)), mindspore.float32)
        >>> output = conv2d_quant(input)
        >>> print(output.shape)
        (2, 6, 2, 2)
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
                 quant_config=quant_config_default,
                 quant_dtype=QuantDtype.INT8):
        super(Conv2dQuant, self).__init__()
        self.in_channels = Validator.check_positive_int(in_channels)
        self.out_channels = Validator.check_positive_int(out_channels)
        self.has_bias = has_bias
        self.kernel_size = twice(kernel_size)
        self.stride = twice(stride)
        self.dilation = twice(dilation)
        self.pad_mode = pad_mode
        self.padding = padding
        self.group = group

        weight_shape = [out_channels, in_channels // group, *self.kernel_size]
        self.weight = Parameter(initializer(weight_init, weight_shape), name='weight')

        self.bias_add = P.BiasAdd()
        if Validator.check_bool(has_bias):
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
        channel_axis = 0
        self.fake_quant_weight = quant_config.weight(min_init=-6,
                                                     max_init=6,
                                                     ema=False,
                                                     channel_axis=channel_axis,
                                                     num_channels=out_channels,
                                                     quant_dtype=quant_dtype)

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
                                                 self.has_bias, self.fake_quant_weight.quant_delay)
        return s


class DenseQuant(Cell):
    r"""
    The fully connected layer with fake quantized operation.

    This part is a more detailed overview of Dense operation. For more detials about Quantilization,
    please refer to :class:`mindspore.nn.FakeQuantWithMinMaxObserver`.

    Args:
        in_channels (int): The dimension of the input space.
        out_channels (int): The dimension of the output space.
        weight_init (Union[Tensor, str, Initializer, numbers.Number]): The trainable weight_init parameter. The dtype
            is same as input. The values of str refer to the function `initializer`. Default: 'normal'.
        bias_init (Union[Tensor, str, Initializer, numbers.Number]): The trainable bias_init parameter. The dtype is
            same as input. The values of str refer to the function `initializer`. Default: 'zeros'.
        has_bias (bool): Specifies whether the layer uses a bias vector. Default: True.
        activation (Union[str, Cell, Primitive]): The regularization function applied to the output of the layer,
            eg. 'relu'. Default: None.
        quant_config (QuantConfig): Configures the oberser types and quant settings of weight and activation. Can be
            generated by compression.quant.create_quant_config method.
            Default: both set to default FakeQuantWithMinMaxObserver.
        quant_dtype (QuantDtype): Specifies the FakeQuant datatype. Default: QuantDtype.INT8.

    Inputs:
        - **input** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor of shape :math:`(N, C_{out}, H_{out}, W_{out})`.

    Raises:
        TypeError: If `in_channels`, `out_channels` is not an int.
        TypeError: If `has_bias` is not a bool.
        ValueError: If `in_channels` or `out_channels` is less than 1.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> qconfig = compression.quant.create_quant_config()
        >>> dense_quant = nn.DenseQuant(3, 6, quant_config=qconfig)
        >>> input = Tensor(np.random.randint(-2, 2, (2, 3)), mindspore.float32)
        >>> result = dense_quant(input)
        >>> output = result.shape
        >>> print(output)
        (2, 6)
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 weight_init='normal',
                 bias_init='zeros',
                 has_bias=True,
                 activation=None,
                 quant_config=quant_config_default,
                 quant_dtype=QuantDtype.INT8):
        super(DenseQuant, self).__init__()
        self.in_channels = Validator.check_positive_int(in_channels)
        self.out_channels = Validator.check_positive_int(out_channels)
        self.has_bias = Validator.check_bool(has_bias)

        if isinstance(weight_init, Tensor):
            if weight_init.ndim != 2 or weight_init.shape[0] != out_channels or \
                    weight_init.shape[1] != in_channels:
                raise ValueError("weight_init shape error")

        self.weight = Parameter(initializer(
            weight_init, [out_channels, in_channels]), name="weight")

        if self.has_bias:
            if isinstance(bias_init, Tensor):
                if bias_init.ndim != 1 or bias_init.shape[0] != out_channels:
                    raise ValueError("bias_init shape error")

            self.bias = Parameter(initializer(
                bias_init, [out_channels]), name="bias")

        self.matmul = P.MatMul(transpose_b=True)
        self.bias_add = P.BiasAdd()

        self.activation = get_activation(activation) if isinstance(activation, str) else activation
        if activation is not None and not isinstance(self.activation, (Cell, Primitive)):
            raise TypeError("The activation must be str or Cell or Primitive,"" but got {}.".format(activation))
        self.activation_flag = self.activation is not None
        self.fake_quant_weight = quant_config.weight(min_init=-6,
                                                     max_init=6,
                                                     ema=False,
                                                     channel_axis=0,
                                                     num_channels=out_channels,
                                                     quant_dtype=quant_dtype)

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
        s = 'in_channels={}, out_channels={}, weight={}, has_bias={}'.format(
            self.in_channels, self.out_channels, self.weight, self.has_bias)
        if self.has_bias:
            s += ', bias={}'.format(self.bias)
        if self.activation_flag:
            s += ', activation={}'.format(self.activation)
        return s


class _QuantActivation(Cell):
    r"""
    Base class for quantization aware training activation function. Adds fake quantized operation
    after activation operation.
    """

    def get_origin(self):
        raise NotImplementedError


class ActQuant(_QuantActivation):
    r"""
    Quantization aware training activation function.

    Add the fake quantized operation to the end of activation operation, by which the output of activation operation
    will be truncated. For more detials about Quantilization,
    please refer to :class:`mindspore.nn.FakeQuantWithMinMaxObserver`.

    Args:
        activation (Cell): Activation cell.
        ema (bool): The exponential Moving Average algorithm updates min and max. Default: False.
        ema_decay (float): Exponential Moving Average algorithm parameter. Default: 0.999.
        fake_before (bool): Whether add fake quantized operation before activation. Default: False.
        quant_config (QuantConfig): Configures the oberser types and quant settings of weight and activation. Can be
            generated by compression.quant.create_quant_config method.
            Default: both set to default FakeQuantWithMinMaxObserver.
        quant_dtype (QuantDtype): Specifies the FakeQuant datatype. Default: QuantDtype.INT8.

    Inputs:
        - **input** (Tensor) - The input of ActQuant.

    Outputs:
        Tensor, with the same type and shape as the `input`.

    Raises:
        TypeError: If `activation` is not an instance of Cell.
        TypeError: If `fake_before` is not a bool.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> qconfig = compression.quant.create_quant_config()
        >>> act_quant = nn.ActQuant(nn.ReLU(), quant_config=qconfig)
        >>> input = Tensor(np.array([[1, 2, -1], [-2, 0, -1]]), mindspore.float32)
        >>> output = act_quant(input)
        >>> print(output)
        [[0.9882355 1.9764705 0.       ]
         [0.        0.        0.       ]]
    """

    def __init__(self,
                 activation,
                 ema=False,
                 ema_decay=0.999,
                 fake_before=False,
                 quant_config=quant_config_default,
                 quant_dtype=QuantDtype.INT8):
        super(ActQuant, self).__init__()
        self.act = Validator.check_isinstance("activation", activation, Cell)
        self.fake_before = Validator.check_bool(fake_before, "fake_before")
        if self.fake_before:
            self.fake_quant_act_before = quant_config.activation(min_init=-6,
                                                                 max_init=6,
                                                                 ema=ema,
                                                                 ema_decay=ema_decay,
                                                                 quant_dtype=quant_dtype)
        self.fake_quant_act = quant_config.activation(min_init=-6,
                                                      max_init=6,
                                                      ema=ema,
                                                      ema_decay=ema_decay,
                                                      quant_dtype=quant_dtype)

    def construct(self, x):
        if self.fake_before:
            x = self.fake_quant_act_before(x)
        x = self.act(x)
        x = self.fake_quant_act(x)
        return x

    def get_origin(self):
        return self.act


class TensorAddQuant(Cell):
    r"""
    Adds fake quantized operation after TensorAdd operation.

    This part is a more detailed overview of TensorAdd operation. For more detials about Quantilization,
    please refer to :class:`mindspore.nn.FakeQuantWithMinMaxObserver`.

    Args:
        ema_decay (float): Exponential Moving Average algorithm parameter. Default: 0.999.
        quant_config (QuantConfig): Configures the oberser types and quant settings of weight and activation. Can be
            generated by compression.quant.create_quant_config method.
            Default: both set to default FakeQuantWithMinMaxObserver.
        quant_dtype (QuantDtype): Specifies the FakeQuant datatype. Default: QuantDtype.INT8.

    Inputs:
        - **input_x1** (Tensor) - The first tensor of TensorAddQuant.
        - **input_x2** (Tensor) - The second tensor of TensorAddQuant.

    Outputs:
        Tensor, with the same type and shape as the `input_x1`.

    Raises:
        TypeError: If `ema_decay` is not a float.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> qconfig = compression.quant.create_quant_config()
        >>> add_quant = nn.TensorAddQuant(quant_config=qconfig)
        >>> input_x1 = Tensor(np.array([[1, 2, 1], [-2, 0, -1]]), mindspore.float32)
        >>> input_x2 = Tensor(np.ones((2, 3)), mindspore.float32)
        >>> output = add_quant(input_x1, input_x2)
        >>> print(output)
        [[ 1.9764705  3.011765   1.9764705]
         [-0.9882355  0.9882355  0.       ]]
    """

    def __init__(self,
                 ema_decay=0.999,
                 quant_config=quant_config_default,
                 quant_dtype=QuantDtype.INT8):
        super(TensorAddQuant, self).__init__()
        self.fake_quant_act = quant_config.activation(min_init=-6,
                                                      max_init=6,
                                                      ema=True,
                                                      ema_decay=ema_decay,
                                                      quant_dtype=quant_dtype)
        self.add = P.Add()

    def construct(self, x1, x2):
        x = self.add(x1, x2)
        x = self.fake_quant_act(x)
        return x


class MulQuant(Cell):
    r"""
    Adds fake quantized operation after `Mul` operation.

    This part is a more detailed overview of `Mul` operation. For more detials about Quantilization,
    please refer to :class:`mindspore.nn.FakeQuantWithMinMaxObserver`.

    Args:
        ema_decay (float): Exponential Moving Average algorithm parameter. Default: 0.999.
        quant_config (QuantConfig): Configures the oberser types and quant settings of weight and activation. Can be
            generated by compression.quant.create_quant_config method.
            Default: both set to default FakeQuantWithMinMaxObserver.
        quant_dtype (QuantDtype): Specifies the FakeQuant datatype. Default: QuantDtype.INT8.

    Inputs:
        - **input_x1** (Tensor) - The first tensor of MulQuant.
        - **input_x2** (Tensor) - The second tensor of MulQuant.

    Outputs:
        Tensor, with the same type and shape as the `input_x1`.

    Raises:
        TypeError: If `ema_decay` is not a float.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> qconfig = compression.quant.create_quant_config()
        >>> mul_quant = nn.MulQuant(quant_config=qconfig)
        >>> input_x1 = Tensor(np.array([[1, 2, 1], [-2, 0, -1]]), mindspore.float32)
        >>> input_x2 = Tensor(np.ones((2, 3)) * 2, mindspore.float32)
        >>> output = mul_quant(input_x1, input_x2)
        >>> print(output)
        [[ 1.9764705  4.0000005  1.9764705]
         [-4.         0.        -1.9764705]]
    """

    def __init__(self,
                 ema_decay=0.999,
                 quant_config=quant_config_default,
                 quant_dtype=QuantDtype.INT8):
        super(MulQuant, self).__init__()
        self.fake_quant_act = quant_config.activation(min_init=-6,
                                                      max_init=6,
                                                      ema=True,
                                                      ema_decay=ema_decay,
                                                      quant_dtype=quant_dtype)
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
        self.bias_add = P.BiasAdd()
        self.sub = P.Sub()
        self.weight_offset = Parameter(np.zeros(shape=weight.shape, dtype=np.int8), name='weight_offset')

    def construct(self, x):
        x = self.quant(x)
        if self.has_bias:
            weight = self.sub(self.weight, self.weight_offset)
            x = self.core_op(x, weight)
            x = self.bias_add(x, self.bias)
        else:
            x = self.core_op(x, self.weight)
        x = self.dequant(x, self.dequant_scale)
        x = F.cast(x, mstype.float32)
        if self.has_act:
            x = self.activation(x)
        return x

    def extend_repr(self):
        s = f'quant={self.quant}, core_op={type(self.core_op)}, weight=shape[{self.weight.shape}]'
        if self.has_bias:
            s += f', bias=shape[{self.bias.shape}]'
        if self.has_act:
            s += f', activation={self.activation}'
        s += f', dequant={self.dequant}'
        return s


class QuantMindirBlock(Cell):
    """A quant binary block of Conv/Dense, activation layer for export MINDIR model.

       Args:
        core_op (Cell): The operation cell.
        weight (Tensor): The weight of the cell.
        bias (Tensor): The bias of the cell. Default: None.
        activation (str): The regularization function applied to the output of the layer, eg. 'relu'. Default: None.
        param_dict (dict): The information of the cell.
    """

    def __init__(self,
                 core_op,
                 weight,
                 bias=None,
                 activation=None,
                 param_dict=None):

        super(QuantMindirBlock, self).__init__()
        self.core_op = core_op
        if activation is not None:
            self.core_op.add_prim_attr("activation_name", activation.__class__.__name__)
        self.core_op.add_prim_attr("filter_maxq", Tensor(param_dict["filter_maxq"]))
        self.core_op.add_prim_attr("filter_minq", Tensor(param_dict["filter_minq"]))
        if param_dict["output_maxq"] is not None:
            self.core_op.add_prim_attr("output_maxq", Tensor(param_dict["output_maxq"]))
            self.core_op.add_prim_attr("output_minq", Tensor(param_dict["output_minq"]))
        self.core_op.add_prim_attr("symmetric", Tensor(param_dict["symmetric"]))
        if hasattr(core_op, 'pad_mode'):
            self.core_op.add_prim_attr("pad_mode", core_op.pad_mode)
        self.core_op.add_prim_attr("num_bits", Tensor(8))
        self.core_op.add_prim_attr("narrow_range", Tensor(False))
        if param_dict["input_maxq"] == 'None':
            self.core_op.add_prim_attr("mean", Tensor(param_dict["mean"]))
            self.core_op.add_prim_attr("std_dev", Tensor(param_dict["std_dev"]))
        elif param_dict["input_maxq"] is not None:
            self.core_op.add_prim_attr("input_maxq", Tensor(param_dict["input_maxq"]))
            self.core_op.add_prim_attr("input_minq", Tensor(param_dict["input_minq"]))

        self.weight = weight
        self.bias = bias
        self.has_bias = bias is not None
        self.activation = activation
        self.has_act = activation is not None
        self.bias_add = P.BiasAdd()
        if isinstance(activation, ReLU):
            self.activation = None
            self.has_act = False

    def construct(self, x):
        if self.has_bias:
            x = self.core_op(x, self.weight)
            x = self.bias_add(x, self.bias)
        else:
            x = self.core_op(x, self.weight)
        return x

    def extend_repr(self):
        s = f'core_op={type(self.core_op)}, weight=shape[{self.weight.shape}]'
        if self.has_bias:
            s += f', bias=shape[{self.bias.shape}]'
        if self.has_act:
            s += f', activation={self.activation}'
        return s
