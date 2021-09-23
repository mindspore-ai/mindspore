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
from mindspore.common.parameter import Parameter
from mindspore.common.initializer import initializer
from mindspore.common.tensor import Tensor
from mindspore._checkparam import Validator, twice
from mindspore.compression.common import QuantDtype
import mindspore.context as context
from .normalization import BatchNorm2d
from .activation import get_activation
from ..cell import Cell
from ... import nn
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
    Batch Normalization folded.

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
        """Initialize _Observer."""
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
        """Initialize UniformQuantObserver."""
        super(UniformQuantObserver, self).__init__(quant_dtype)
        self.per_channel = per_channel
        self.symmetric = symmetric
        self.narrow_range = narrow_range
        self.num_channels = num_channels


class FakeQuantWithMinMaxObserver(UniformQuantObserver):
    r"""
    Quantization aware operation which provides the fake quantization observer function on data with min and max.

    The detail of the quantization mode `DEFAULT` is described as below:

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

    The detail of the quantization mode `LEARNED_SCALE` is described as below:

    The fake quant output is computed as:

    .. math::

        \bar{X}=\left\{\begin{matrix}
        clip\left ( \frac{X}{maxq},0,1\right ) \qquad \quad if\quad neg\_trunc\\
        clip\left ( \frac{X}{maxq},-1,1\right )\qquad \ if\quad otherwise
        \end{matrix}\right. \\

        output=\frac{floor\left ( \bar{X}\ast  Q_{max}+0.5  \right ) \ast scale }{Q_{max}}

    where X is the input tensor.
    where :math:`Q_{max}` (quant_max) is decided by quant_dtype and neg_trunc, for example, if quant_dtype=INT8
    and neg_trunc works, :math:`Q_{max} = 256` , otherwise math:`Q_{max} = 127`.

    The maxq is updated by training, and its gradient is calculated as follows:

    .. math::

        \frac{\partial \ output}{\partial \ maxq} = \left\{\begin{matrix}
        -\frac{X}{maxq}+\left \lfloor \frac{X}{maxq} \right \rceil \qquad if\quad bound_{lower}< \frac{X}{maxq}< 1\\
        -1 \qquad \quad \qquad \quad if\quad \frac{X}{maxq}\le bound_{lower}\\
         1  \qquad \quad \qquad \quad if\quad \frac{X}{maxq}\ge  1 \qquad \quad
        \end{matrix}\right. \\

        bound_{lower}=
        \left\{\begin{matrix}
         0\qquad \quad if\quad neg\_trunc\\
        -1\qquad if\quad otherwise
        \end{matrix}\right.

    Then minq is computed as:

    .. math::

        minq=\left\{\begin{matrix}
        0  \qquad \qquad \quad if\quad neg\_trunc\\
        -maxq\qquad if\quad otherwise
        \end{matrix}\right.

    When exporting, the scale and zero point zp is computed as:

    .. math::

        scale=\frac{maxq}{quant\_max} ,\quad zp=0 \\

    zp is equal to 0 consistently, due to the LEARNED_SCALE`s symmetric nature.

    Args:
        min_init (int, float, list): The initialized min value. Default: -6.
        max_init (int, float, list): The initialized max value. Default: 6.
        ema (bool): The exponential Moving Average algorithm updates min and max. Default: False.
        ema_decay (float): Exponential Moving Average algorithm parameter. Default: 0.999.
        per_channel (bool):  Quantization granularity based on layer or on channel. Default: False.
        channel_axis (int): Quantization by channel axis. Default: 1.
        num_channels (int): declarate the min and max channel size, Default: 1.
        quant_dtype (QuantDtype): The datatype of quantization, supporting 4 and 8bits. Default: QuantDtype.INT8.
        symmetric (bool): Whether the quantization algorithm is symmetric or not. Default: False.
        narrow_range (bool): Whether the quantization algorithm uses narrow range or not. Default: False.
        quant_delay (int): Quantization delay parameters according to the global step. Default: 0.
        neg_trunc (bool): Whether the quantization algorithm uses nagetive truncation or not. Default: False.
        mode (str): Optional quantization mode, currently only `DEFAULT`(QAT) and `LEARNED_SCALE` are supported.
            Default: ("DEFAULT")
    Inputs:
        - **x** (Tensor) - The input of FakeQuantWithMinMaxObserver. The input dimension is preferably 2D or 4D.

    Outputs:
        Tensor, with the same type and shape as the `x`.

    Raises:
        TypeError: If `min_init` or `max_init` is not int, float or list.
        TypeError: If `quant_delay` is not an int.
        ValueError: If `quant_delay` is less than 0.
        ValueError: If `min_init` is not less than `max_init`.
        ValueError: If `mode` is neither `DEFAULT` nor `LEARNED_SCALE`.
        ValueError: If `mode` is `LEARNED_SCALE` and `symmetric` is not `True`.
        ValueError: If `mode` is `LEARNED_SCALE`, and `narrow_range` is not `True` unless when `neg_trunc` is `True`.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import mindspore
        >>> from mindspore import Tensor
        >>> fake_quant = nn.FakeQuantWithMinMaxObserver()
        >>> x = Tensor(np.array([[1, 2, 1], [-2, 0, -1]]), mindspore.float32)
        >>> result = fake_quant(x)
        >>> print(result)
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
                 quant_delay=0,
                 neg_trunc=False,
                 mode="DEFAULT"):
        """Initialize FakeQuantWithMinMaxObserver"""
        super(FakeQuantWithMinMaxObserver, self).__init__(quant_dtype=quant_dtype, per_channel=per_channel,
                                                          symmetric=symmetric, narrow_range=narrow_range,
                                                          num_channels=num_channels)
        Validator.check_value_type("min_init", min_init, [int, float, list], type(self).__name__)
        Validator.check_value_type("max_init", max_init, [int, float, list], type(self).__name__)
        Validator.check_non_negative_int(quant_delay, 'quant_delay', self.cls_name)
        self.min_init = min_init
        self.max_init = max_init
        self.quant_dtype = quant_dtype
        self.num_bits = quant_dtype.num_bits
        self.ema = ema
        self.ema_decay = ema_decay
        self.per_channel = per_channel
        self.num_channels = num_channels
        self.channel_axis = channel_axis
        self.quant_delay = quant_delay
        self.symmetric = symmetric
        self.narrow_range = narrow_range
        self.neg_trunc = neg_trunc
        self.mode = mode
        self.is_ascend = context.get_context('device_target') == "Ascend"
        self.Neg = P.Neg()

        min_array = self._get_init_array(self.min_init)
        max_array = self._get_init_array(self.max_init)
        if not np.greater(max_array, min_array).all():
            raise ValueError(f"For '{self.cls_name}', the 'max_init' should be greater than 'min_init', "
                             f"but got 'max_init': {max_init}, 'min_init': {min_init}.")
        if self.mode == "DEFAULT":
            self._default_init(min_array, max_array)
        elif self.mode == "LEARNED_SCALE":
            self._learned_scale_init(min_array, max_array)
        else:
            raise ValueError(f"For '{self.cls_name}', only `DEFAULT` and `LEARNED_SCALE` mode are valid, but got "
                             f"'mode': {self.mode}.")

    def reset(self, quant_dtype=QuantDtype.INT8, min_init=-6, max_init=6):
        r"""
        Reset the quant max parameter (eg. 256) and the initial value of the minq parameter and maxq parameter,
        this function is currently only valid for `LEARNED_SCALE` mode.
        """
        if self.mode == "LEARNED_SCALE":
            self.quant_dtype = quant_dtype
            self.num_bits = quant_dtype.num_bits
            self._calculate_quant_max()
            if self.neg_trunc:
                min_init = 0

            self.min_init = min_init
            self.max_init = max_init
            min_array = self._get_init_array(self.min_init)
            max_array = self._get_init_array(self.max_init)
            if not np.greater(max_array, min_array).all():
                raise ValueError(f"For '{self.cls_name}', the 'max_init' should be greater than 'min_init', "
                                 f"but got 'max_array': {max_array}, 'min_init': {min_init}.")

            self.minq.set_data(Tensor(min_array))
            self.maxq.set_data(Tensor(max_array))
            self.quant_max.set_data(Tensor(np.array([self._quant_max]).astype(np.float32)))
        else:
            raise ValueError(f"For '{self.cls_name}', only `LEARNED_SCALE` mode is valid, but got 'mode': {self.mode}.")

    def _default_init(self, min_array, max_array):
        """
        Initialization of `DEFAULT`(QAT) mode.
        """
        # init tensor min and max for fake quantized operation
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
                                ema_decay=self.ema_decay,
                                num_bits=self.quant_dtype.num_bits,
                                symmetric=self.symmetric,
                                narrow_range=self.narrow_range,
                                quant_delay=self.quant_delay)
            self.fake_quant_train = quant_fun(training=True)
            self.fake_quant_infer = quant_fun(training=False)

    def _learned_scale_init(self, min_array, max_array):
        """
        Initialization of `LEARNED_SCALE` mode.
        """
        if not self.symmetric:
            raise ValueError(f"For '{self.cls_name}', the 'LEARNED_SCALE' mode only support 'symmetric' quant, "
                             f"but got 'symmetric': {self.symmetric}. Please set 'symmetric' to True.")
        if self.neg_trunc:
            min_array = self._get_init_array(0)
            if self.narrow_range:
                raise ValueError(f"For '{self.cls_name}', the 'LEARNED_SCALE' mode only support the combination of "
                                 f"'neg_trunc=True and narrow_range=False' config scenario, but got 'narrow_range': "
                                 f"{self.narrow_range}.")
        elif not self.narrow_range:
            raise ValueError(f"For '{self.cls_name}', the 'LEARNED_SCALE' mode only support 'narrow_range=True' "
                             f"config, except for 'neg_trunc=True' scenario. But got 'narrow_range': "
                             f"{self.narrow_range}.")

        self._calculate_quant_max()

        self.minq = Parameter(Tensor(min_array), name='minq')
        self.maxq = Parameter(Tensor(max_array), name='maxq')
        self.quant_max = Parameter(Tensor(np.array([self._quant_max]).astype(np.float32)),
                                   name="quant_max", requires_grad=False)

        # init fake quant relative op
        if self.per_channel:
            quant_fun = partial(Q.FakeLearnedScaleQuantPerChannel, channel_axis=self.channel_axis)
        else:
            quant_fun = Q.FakeLearnedScaleQuantPerLayer

        quant_fun = partial(quant_fun,
                            quant_delay=self.quant_delay,
                            neg_trunc=self.neg_trunc)
        self.fake_quant_train = quant_fun(training=True)
        self.fake_quant_infer = quant_fun(training=False)

    def _get_init_array(self, init_date):
        """
        Convert the initial value to array.
        """
        if isinstance(init_date, list) and self.per_channel and len(init_date) != self.num_channels:
            raise ValueError(f"For '{self.cls_name}', the length of 'min_init/max_init' list should be equal to "
                             f"'num_channels' for perchannel quant scenario, but got {len(init_date)}.")
        if isinstance(init_date, list) and not self.per_channel and len(init_date) != 1:
            raise ValueError(f"For '{self.cls_name}', the length of the 'min_init/max_init' list should be 1 for "
                             f"perlayer quant scenario, but got {len(init_date)}.")

        if isinstance(init_date, list):
            min_max_array = np.array(init_date).astype(np.float32)
        elif self.per_channel and not isinstance(init_date, list):
            min_max_array = np.array([init_date] * self.num_channels).astype(np.float32)
        else:
            min_max_array = np.array([init_date]).astype(np.float32)
        return min_max_array

    def _calculate_quant_max(self):
        """
        The quantization range is calculated according to num_bits.
        """
        if not self.neg_trunc:
            self._quant_max = (1 << (self.num_bits - 1)) - 1
        else:
            self._quant_max = (1 << self.num_bits) - 1

    def extend_repr(self):
        """Display instance object as string."""
        s = 'quant_dtype={}, symmetric={}, narrow_range={}, ema={}({}), per_channel={}({}, {}), ' \
            'quant_delay={}, min_init={}, max_init={}'.format(self.quant_dtype, self.symmetric, self.narrow_range,
                                                              self.ema, self.ema_decay, self.per_channel,
                                                              self.channel_axis, self.num_channels, self.quant_delay,
                                                              self.min_init, self.max_init)
        return s

    def construct(self, x):
        if self.mode == "LEARNED_SCALE":
            if self.training:
                out = self.fake_quant_train(x, self.maxq, self.quant_max)
                if not self.neg_trunc:
                    self.minq = self.Neg(self.maxq)
            else:
                out = self.fake_quant_infer(x, self.maxq, self.quant_max)
        else:
            if self.training:
                min_up, max_up = self.ema_update(x, self.minq, self.maxq)
                self.minq = min_up
                self.maxq = max_up
                out = self.fake_quant_train(x, self.minq, self.maxq)
            else:
                out = self.fake_quant_infer(x, self.minq, self.maxq)
        return out


QuantConfig = namedtuple("QuantConfig", ['weight', 'activation'])

quant_config_default = QuantConfig(weight=FakeQuantWithMinMaxObserver.partial_init(),
                                   activation=FakeQuantWithMinMaxObserver.partial_init())


class Conv2dBnFoldQuantOneConv(Cell):
    r"""
    2D convolution which use the convolution layer statistics once to calculate Batch Normalization
    operation folded construct.

    This part is a more detailed overview of Conv2d operation. For more details about Quantization,
    please refer to the implementation of subclass of class:`_Observer`, for example,
    :class:`FakeQuantWithMinMaxObserver`.

    .. math::
        w_{q}=quant(\frac{w}{\sqrt{var_{G}+\epsilon}}*\gamma )

        b=\frac{-\mu _{G} }{\sqrt{var_{G}+\epsilon }}*\gamma +\beta

        y=w_{q}\times x+b

    where :math:`quant` is the continuous execution of quant and dequant, you can refer to the implementation of
    subclass of class:`_Observer`, for example, class:`mindspore.nn.FakeQuantWithMinMaxObserver`.
    `mu _{G}` and `var_{G}` represent the global mean and variance respectively.

    Args:
        in_channels (int): The number of input channel :math:`C_{in}`.
        out_channels (int): The number of output channel :math:`C_{out}`.
        kernel_size (Union[int, tuple[int]]): Specifies the height and width of the 2D convolution window.
        stride (Union[int, tuple[int]]): Specifies stride for all spatial dimensions with the same value. Default: 1.
        pad_mode (str): Specifies padding mode. The optional values are "same", "valid", "pad". Default: "same".
        padding (Union[int, tuple[int]]): Implicit paddings on both sides of the `x`. Default: 0.
        dilation (Union[int, tuple[int]]): Specifies the dilation rate to use for dilated convolution. Default: 1.
        group (int): Splits filter into groups, `in_ channels` and `out_channels` must be
            divisible by the number of groups. Default: 1.
        eps (float): Parameters for Batch Normalization. Default: 1e-5.
        momentum (float): Parameters for Batch Normalization op. Default: 0.997.
        has_bias (bool): Specifies whether the layer uses a bias vector, which is temporarily invalid. Default: False.
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
        quant_config (QuantConfig): Configures the types of quant observer and quant settings of weight and
            activation. Note that, QuantConfig is a special namedtuple, which is designed for quantization
            and can be generated by :func:`mindspore.compression.quant.create_quant_config` method.
            Default: QuantConfig with both items set to default :class:`FakeQuantWithMinMaxObserver`.
        quant_dtype (QuantDtype): Specifies the FakeQuant datatype. Default: QuantDtype.INT8.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor of shape :math:`(N, C_{out}, H_{out}, W_{out})`.

    Raises:
        TypeError: If `in_channels`, `out_channels` or `group` is not an int.
        TypeError: If `kernel_size`, `stride`, `padding` or `dilation` is neither an int nor a tuple.
        TypeError: If `has_bias` or `fake` is not a bool.
        TypeError: If `data_format` is not a string.
        ValueError: If `in_channels`, `out_channels`, `kernel_size`, `stride` or `dilation` is less than 1.
        ValueError: If `padding` is less than 0.
        ValueError: If `pad_mode` is not one of 'same', 'valid', 'pad'.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import mindspore
        >>> from mindspore.compression import quant
        >>> from mindspore import Tensor
        >>> qconfig = quant.create_quant_config()
        >>> conv2d_bnfold = nn.Conv2dBnFoldQuantOneConv(1, 1, kernel_size=(2, 2), stride=(1, 1), pad_mode="valid",
        ...                                             weight_init="ones", quant_config=qconfig)
        >>> x = Tensor(np.array([[[[1, 0, 3], [1, 4, 7], [2, 5, 2]]]]), mindspore.float32)
        >>> result = conv2d_bnfold(x)
        >>> print(result)
        [[[[5.9296875 13.8359375]
           [11.859375 17.78125]]]]
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
        self.in_channels = Validator.check_positive_int(in_channels, "in_channels", self.cls_name)
        self.out_channels = Validator.check_positive_int(out_channels, "out_channels", self.cls_name)
        self.kernel_size = twice(kernel_size)
        self.stride = twice(stride)
        self.dilation = twice(dilation)
        for kernel_size_elem in self.kernel_size:
            Validator.check_positive_int(kernel_size_elem, 'kernel_size item', self.cls_name)
        for stride_elem in self.stride:
            Validator.check_positive_int(stride_elem, 'stride item', self.cls_name)
        for dilation_elem in self.dilation:
            Validator.check_positive_int(dilation_elem, 'dilation item', self.cls_name)
        if pad_mode not in ('valid', 'same', 'pad'):
            raise ValueError(f"For '{self.cls_name}', the 'pad_mode' should be one of values "
                             f"in ('valid', 'same', 'pad'), but got {pad_mode}.")
        self.pad_mode = pad_mode
        if isinstance(padding, int):
            Validator.check_non_negative_int(padding, 'padding', self.cls_name)
            self.padding = padding
        elif isinstance(padding, tuple):
            for pad in padding:
                Validator.check_non_negative_int(pad, 'padding item', self.cls_name)
            self.padding = padding
        else:
            raise TypeError(f"For '{self.cls_name}', the type of 'padding' must be int/tuple(int), but got "
                            f"{type(padding).__name__}!")
        self.group = Validator.check_positive_int(group, "group", self.cls_name)
        self.eps = eps
        self.momentum = 1 - momentum
        self.has_bias = has_bias
        self.fake = Validator.check_bool(fake, "fake", self.cls_name)
        self.quant_config = quant_config
        self.quant_dtype = quant_dtype
        data_format = 'NCHW'
        self.format = Validator.check_string(data_format, ['NCHW', 'NHWC'], 'format', self.cls_name)
        self._target = context.get_context("device_target")
        self.is_graph_mode = context.get_context("mode") == context.GRAPH_MODE
        self.is_ge_backend = False
        if context.get_context("enable_ge"):
            self.is_ge_backend = True
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
        self.bias = None
        if Validator.check_bool(has_bias, "has_bias", self.cls_name):
            self.bias = Parameter(initializer(bias_init, [out_channels]), name='bias')

        # initialize BatchNorm Parameter
        self.gamma = Parameter(initializer(gamma_init, [out_channels]), name='gamma')
        self.beta = Parameter(initializer(beta_init, [out_channels]), name='beta')
        self.moving_mean = Parameter(initializer(mean_init, [out_channels]), name='moving_mean', requires_grad=False)
        self.moving_variance = Parameter(initializer(var_init, [out_channels]), name='moving_variance',
                                         requires_grad=False)

        # initialize fake ops
        self.fake_quant_weight = quant_config.weight(ema=False,
                                                     channel_axis=channel_axis,
                                                     num_channels=out_channels,
                                                     quant_dtype=quant_dtype)
        self.freeze_bn = False
        if self.fake_quant_weight.mode == "LEARNED_SCALE":
            self.freeze_bn = True
        self.bn_train = P.BatchNorm(is_training=True, epsilon=self.eps,
                                    momentum=self.momentum, data_format=self.format)

        self.bn_infer = P.BatchNorm(is_training=False, epsilon=self.eps, data_format=self.format)
        self.sub_mean = P.Sub()
        self.sub_var = P.Sub()
        self.mul_mean = P.Mul()
        self.mul_var = P.Mul()
        self.assign_sub_mean = P.AssignSub()
        self.assign_sub_var = P.AssignSub()
        self.reshape = P.Reshape()

    def extend_repr(self):
        """Display instance object as string."""
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

        if self.freeze_bn:
            return conv + self.reshape((self.beta - self.gamma * self.moving_mean / running_std), (1, -1, 1, 1))
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
    2D convolution with Batch Normalization operation folded construct.

    This part is a more detailed overview of Conv2d operation. For more details about Quantization,
    please refer to the implementation of subclass of class:`_Observer`, for example,
    :class:`FakeQuantWithMinMaxObserver`.

    .. math::
        y = x\times w+  b

        w_{q}=quant(\frac{w}{\sqrt{Var[y]+\epsilon}}*\gamma )

        y_{out}= w_{q}\times x+\frac{b-E[y]}{\sqrt{Var[y]+\epsilon}}*\gamma +\beta

    where :math:`quant` is the continuous execution of quant and dequant, you can refer to the implementation of
    subclass of class:`_Observer`, for example, class:`mindspore.nn.FakeQuantWithMinMaxObserver`. Two convolution
    and Batch Normalization operation are used here, the purpose of the first convolution and Batch Normalization
    is to count the mean `E[y]` and variance `Var[y]` of current batch output for quantization.

    Args:
        in_channels (int): The number of input channel :math:`C_{in}`.
        out_channels (int): The number of output channel :math:`C_{out}`.
        kernel_size (Union[int, tuple[int]]): Specifies the height and width of the 2D convolution window.
        stride (Union[int, tuple[int]]): Specifies stride for all spatial dimensions with the same value. Default: 1.
        pad_mode (str): Specifies padding mode. The optional values are "same", "valid", "pad". Default: "same".
        padding (Union[int, tuple[int]]): Implicit paddings on both sides of the `x`. Default: 0.
        dilation (Union[int, tuple[int]]): Specifies the dilation rate to use for dilated convolution. Default: 1.
        group (int): Splits filter into groups, `in_ channels` and `out_channels` must be
            divisible by the number of groups. Default: 1.
        eps (float): Parameters for Batch Normalization. Default: 1e-5.
        momentum (float): Parameters for Batch Normalization op. Default: 0.997.
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
        quant_config (QuantConfig): Configures the types of quant observer and quant settings of weight and
            activation. Note that, QuantConfig is a special namedtuple, which is designed for quantization
            and can be generated by :func:`mindspore.compression.quant.create_quant_config` method.
            Default: QuantConfig with both items set to default :class:`FakeQuantWithMinMaxObserver`.
        quant_dtype (QuantDtype): Specifies the FakeQuant datatype. Default: QuantDtype.INT8.
        freeze_bn (int): The quantization freeze Batch Normalization op is according to the global step.
            Default: 100000.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor of shape :math:`(N, C_{out}, H_{out}, W_{out})`.

    Raises:
        TypeError: If `in_channels`, `out_channels` or `group` is not an int.
        TypeError: If `kernel_size`, `stride`, `padding` or `dilation` is neither an int nor a tuple.
        TypeError: If `has_bias` or `fake` is not a bool.
        ValueError: If `in_channels`, `out_channels`, `kernel_size`, `stride` or `dilation` is less than 1.
        ValueError: If `padding` is less than 0.
        ValueError: If `pad_mode` is not one of 'same', 'valid', 'pad'.
        ValueError: If `device_target` in context is neither `Ascend` nor `GPU`.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import mindspore
        >>> from mindspore.compression import quant
        >>> from mindspore import Tensor
        >>> qconfig = quant.create_quant_config()
        >>> conv2d_bnfold = nn.Conv2dBnFoldQuant(1, 1, kernel_size=(2, 2), stride=(1, 1), pad_mode="valid",
        ...                                      weight_init="ones", quant_config=qconfig)
        >>> x = Tensor(np.array([[[[1, 0, 3], [1, 4, 7], [2, 5, 2]]]]), mindspore.float32)
        >>> result = conv2d_bnfold(x)
        >>> print(result)
        [[[[5.9296875 13.8359375]
           [11.859375 17.78125]]]]
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
        self.in_channels = Validator.check_positive_int(in_channels, "in_channels", self.cls_name)
        self.out_channels = Validator.check_positive_int(out_channels, "out_channels", self.cls_name)
        self.kernel_size = twice(kernel_size)
        self.stride = twice(stride)
        self.dilation = twice(dilation)
        for kernel_size_elem in self.kernel_size:
            Validator.check_positive_int(kernel_size_elem, 'kernel_size item', self.cls_name)
        for stride_elem in self.stride:
            Validator.check_positive_int(stride_elem, 'stride item', self.cls_name)
        for dilation_elem in self.dilation:
            Validator.check_positive_int(dilation_elem, 'dilation item', self.cls_name)
        if pad_mode not in ('valid', 'same', 'pad'):
            raise ValueError(f"For '{self.cls_name}', the 'pad_mode' should be one of values in "
                             f"('valid', 'same', 'pad'), but got {pad_mode}.")
        self.pad_mode = pad_mode
        if isinstance(padding, int):
            Validator.check_non_negative_int(padding, 'padding', self.cls_name)
            self.padding = padding
        elif isinstance(padding, tuple):
            for pad in padding:
                Validator.check_non_negative_int(pad, 'padding item', self.cls_name)
            self.padding = padding
        else:
            raise TypeError(f"For '{self.cls_name}', the type of 'padding' must be int/tuple(int), "
                            f"but got {type(padding).__name__}!")
        self.group = Validator.check_positive_int(group, "group", self.cls_name)
        self.eps = eps
        self.momentum = momentum
        self.has_bias = has_bias
        self.freeze_bn = freeze_bn
        self.fake = Validator.check_bool(fake, "fake", self.cls_name)
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
        self.bias = None
        if Validator.check_bool(has_bias, "has_bias", self.cls_name):
            self.bias = Parameter(initializer(bias_init, [out_channels]), name='bias')

        # initialize BatchNorm Parameter
        self.gamma = Parameter(initializer(gamma_init, [out_channels]), name='gamma')
        self.beta = Parameter(initializer(beta_init, [out_channels]), name='beta')
        self.moving_mean = Parameter(initializer(mean_init, [out_channels]), name='moving_mean', requires_grad=False)
        self.moving_variance = Parameter(initializer(var_init, [out_channels]), name='moving_variance',
                                         requires_grad=False)

        # initialize fake ops
        self.fake_quant_weight = quant_config.weight(ema=False,
                                                     channel_axis=channel_axis,
                                                     num_channels=out_channels,
                                                     quant_dtype=quant_dtype)
        self.batchnorm_fold = BatchNormFoldCell(epsilon=eps, momentum=momentum, freeze_bn=freeze_bn)
        self.correct_mul = Q.CorrectionMul(channel_axis)
        if context.get_context('device_target') == "Ascend":
            self.batchnorm_fold2_train = Q.BatchNormFold2D(freeze_bn=freeze_bn)
            self.batchnorm_fold2_infer = Q.BatchNormFold2D(freeze_bn=0)
        elif context.get_context('device_target') == "GPU":
            self.batchnorm_fold2_train = Q.BatchNormFold2(freeze_bn=freeze_bn)
            self.batchnorm_fold2_infer = Q.BatchNormFold2(freeze_bn=0)
        else:
            raise ValueError(f"For '{self.cls_name}', only the 'Ascend' and 'GPU' platforms"
                             f" are supported, but got {context.get_context('device_target')}.")
        self.step = Parameter(initializer('normal', [1], dtype=mstype.int32), name='step', requires_grad=False)
        self.one = Tensor(1, mstype.int32)
        self.assignadd = P.AssignAdd()

    def extend_repr(self):
        """Display instance object as string."""
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

    This part is a more detailed overview of Conv2d operation. For more details about Quantization,
    please refer to the implementation of subclass of class:`_Observer`, for example,
    class:`mindspore.nn.FakeQuantWithMinMaxObserver`.

    .. math::
        y =x\times quant(w)+  b

        y_{bn} =\frac{y-E[y] }{\sqrt{Var[y]+  \epsilon  } } *\gamma +  \beta

    where :math:`quant` is the continuous execution of quant and dequant, you can refer to the implementation of
    subclass of class:`_Observer`, for example, class:`mindspore.nn.FakeQuantWithMinMaxObserver`.

    Args:
        in_channels (int): The number of input channel :math:`C_{in}`.
        out_channels (int): The number of output channel :math:`C_{out}`.
        kernel_size (Union[int, tuple[int]]): Specifies the height and width of the 2D convolution window.
        stride (Union[int, tuple[int]]): Specifies stride for all spatial dimensions with the same value. Default: 1.
        pad_mode (str): Specifies padding mode. The optional values are "same", "valid", "pad". Default: "same".
        padding (Union[int, tuple[int]]): Implicit paddings on both sides of the `x`. Default: 0.
        dilation (Union[int, tuple[int]]): Specifies the dilation rate to use for dilated convolution. Default: 1.
        group (int): Splits filter into groups, `in_ channels` and `out_channels` must be
            divisible by the number of groups. Default: 1.
        has_bias (bool): Specifies whether the layer uses a bias vector. Default: False.
        eps (float): Parameters for Batch Normalization. Default: 1e-5.
        momentum (float): Parameters for Batch Normalization op. Default: 0.997.
        weight_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the convolution kernel.
            Default: 'normal'.
        bias_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the bias vector. Default: 'zeros'.
        quant_config (QuantConfig): Configures the types of quant observer and quant settings of weight and
            activation. Note that, QuantConfig is a special namedtuple, which is designed for quantization
            and can be generated by :func:`mindspore.compression.quant.create_quant_config` method.
            Default: QuantConfig with both items set to default :class:`FakeQuantWithMinMaxObserver`.
        quant_dtype (QuantDtype): Specifies the FakeQuant datatype. Default: QuantDtype.INT8.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor of shape :math:`(N, C_{out}, H_{out}, W_{out})`.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Raises:
        TypeError: If `in_channels`, `out_channels` or `group` is not an int.
        TypeError: If `kernel_size`, `stride`, `padding` or `dilation` is neither an int nor a tuple.
        TypeError: If `has_bias` is not a bool.
        ValueError: If `in_channels`, `out_channels`, `kernel_size`, `stride` or `dilation` is less than 1.
        ValueError: If `padding` is less than 0.
        ValueError: If `pad_mode` is not one of 'same', 'valid', 'pad'.

    Examples:
        >>> import mindspore
        >>> from mindspore.compression import quant
        >>> from mindspore import Tensor
        >>> qconfig = quant.create_quant_config()
        >>> conv2d_no_bnfold = nn.Conv2dBnWithoutFoldQuant(1, 1, kernel_size=(2, 2), stride=(1, 1), pad_mode="valid",
        ...                                                weight_init='ones', quant_config=qconfig)
        >>> x = Tensor(np.array([[[[1, 0, 3], [1, 4, 7], [2, 5, 2]]]]), mindspore.float32)
        >>> result = conv2d_no_bnfold(x)
        >>> print(result)
        [[[[5.929658  13.835868]
           [11.859316  17.78116]]]]
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
        """Initialize Conv2dBnWithoutFoldQuant."""
        super(Conv2dBnWithoutFoldQuant, self).__init__()
        self.in_channels = Validator.check_positive_int(in_channels, "in_channels", self.cls_name)
        self.out_channels = Validator.check_positive_int(out_channels, "out_channels", self.cls_name)
        self.has_bias = has_bias
        self.kernel_size = twice(kernel_size)
        self.stride = twice(stride)
        self.dilation = twice(dilation)
        for kernel_size_elem in self.kernel_size:
            Validator.check_positive_int(kernel_size_elem, 'kernel_size item', self.cls_name)
        for stride_elem in self.stride:
            Validator.check_positive_int(stride_elem, 'stride item', self.cls_name)
        for dilation_elem in self.dilation:
            Validator.check_positive_int(dilation_elem, 'dilation item', self.cls_name)
        if pad_mode not in ('valid', 'same', 'pad'):
            raise ValueError(f"For '{self.cls_name}', the 'pad_mode' should be one of values in "
                             f"('valid', 'same', 'pad'), but got {pad_mode}.")
        self.pad_mode = pad_mode
        if isinstance(padding, int):
            Validator.check_non_negative_int(padding, 'padding', self.cls_name)
            self.padding = padding
        elif isinstance(padding, tuple):
            for pad in padding:
                Validator.check_non_negative_int(pad, 'padding item', self.cls_name)
            self.padding = padding
        else:
            raise TypeError(f"For '{self.cls_name}', the type of 'padding' must be int/tuple(int), "
                            f"but got {type(padding).__name__}!")
        self.group = Validator.check_positive_int(group, "group", self.cls_name)
        self.bias_add = P.BiasAdd()
        if Validator.check_bool(has_bias, "has_bias", self.cls_name):
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
        self.fake_quant_weight = quant_config.weight(ema=False,
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
        """Display instance object as string."""
        s = 'in_channels={}, out_channels={}, kernel_size={}, stride={}, ' \
            'pad_mode={}, padding={}, dilation={}, group={}, ' \
            'has_bias={}, quant_delay={}'.format(self.in_channels, self.out_channels, self.kernel_size, self.stride,
                                                 self.pad_mode, self.padding, self.dilation, self.group,
                                                 self.has_bias, self.fake_quant_weight.quant_delay)
        return s


class Conv2dQuant(Cell):
    r"""
    2D convolution with fake quantized operation layer.

    This part is a more detailed overview of Conv2d operation. For more details about Quantization,
    please refer to the implementation of subclass of class:`_Observer`, for example,
    class:`mindspore.nn.FakeQuantWithMinMaxObserver`.

    Args:
        in_channels (int): The number of input channel :math:`C_{in}`.
        out_channels (int): The number of output channel :math:`C_{out}`.
        kernel_size (Union[int, tuple[int]]): Specifies the height and width of the 2D convolution window.
        stride (Union[int, tuple[int]]): Specifies stride for all spatial dimensions with the same value. Default: 1.
        pad_mode (str): Specifies padding mode. The optional values are "same", "valid", "pad". Default: "same".
        padding (Union[int, tuple[int]]): Implicit paddings on both sides of the `x`. Default: 0.
        dilation (Union[int, tuple[int]]): Specifies the dilation rate to use for dilated convolution. Default: 1.
        group (int): Splits filter into groups, `in_ channels` and `out_channels` must be
            divisible by the number of groups. Default: 1.
        has_bias (bool): Specifies whether the layer uses a bias vector. Default: False.
        weight_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the convolution kernel.
            Default: 'normal'.
        bias_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the bias vector. Default: 'zeros'.
        quant_config (QuantConfig): Configures the types of quant observer and quant settings of weight and
            activation. Note that, QuantConfig is a special namedtuple, which is designed for quantization
            and can be generated by :func:`mindspore.compression.quant.create_quant_config` method.
            Default: QuantConfig with both items set to default :class:`FakeQuantWithMinMaxObserver`.
        quant_dtype (QuantDtype): Specifies the FakeQuant datatype. Default: QuantDtype.INT8.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.
          The input dimension is preferably 2D or 4D.

    Outputs:
        Tensor of shape :math:`(N, C_{out}, H_{out}, W_{out})`.

    Raises:
        TypeError: If `in_channels`, `out_channels` or `group` is not an int.
        TypeError: If `kernel_size`, `stride`, `padding` or `dilation` is neither an int nor a tuple.
        TypeError: If `has_bias` is not a bool.
        ValueError: If `in_channels`, `out_channels`, `kernel_size`, `stride` or `dilation` is less than 1.
        ValueError: If `padding` is less than 0.
        ValueError: If `pad_mode` is not one of 'same', 'valid', 'pad'.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import mindspore
        >>> from mindspore.compression import quant
        >>> from mindspore import Tensor
        >>> qconfig = quant.create_quant_config()
        >>> conv2d_quant = nn.Conv2dQuant(1, 1, kernel_size=(2, 2), stride=(1, 1), pad_mode="valid",
        ...                               weight_init='ones', quant_config=qconfig)
        >>> x = Tensor(np.array([[[[1, 0, 3], [1, 4, 7], [2, 5, 2]]]]), mindspore.float32)
        >>> result = conv2d_quant(x)
        >>> print(result)
        [[[[5.9296875  13.8359375]
           [11.859375  17.78125]]]]
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
        """Initialize Conv2dQuant."""
        super(Conv2dQuant, self).__init__()
        self.in_channels = Validator.check_positive_int(in_channels, "in_channels", self.cls_name)
        self.out_channels = Validator.check_positive_int(out_channels, "out_channels", self.cls_name)
        self.has_bias = has_bias
        self.kernel_size = twice(kernel_size)
        self.stride = twice(stride)
        self.dilation = twice(dilation)
        for kernel_size_elem in self.kernel_size:
            Validator.check_positive_int(kernel_size_elem, 'kernel_size item', self.cls_name)
        for stride_elem in self.stride:
            Validator.check_positive_int(stride_elem, 'stride item', self.cls_name)
        for dilation_elem in self.dilation:
            Validator.check_positive_int(dilation_elem, 'dilation item', self.cls_name)
        if pad_mode not in ('valid', 'same', 'pad'):
            raise ValueError(f"For '{self.cls_name}', the 'pad_mode' should be one of values "
                             f"in ('valid', 'same', 'pad'), but got {pad_mode}.")
        self.pad_mode = pad_mode
        if isinstance(padding, int):
            Validator.check_non_negative_int(padding, 'padding', self.cls_name)
            self.padding = padding
        elif isinstance(padding, tuple):
            for pad in padding:
                Validator.check_non_negative_int(pad, 'padding item', self.cls_name)
            self.padding = padding
        else:
            raise TypeError(f"For '{self.cls_name}', the type of 'padding' must be int/tuple(int), "
                            f"but got {type(padding).__name__}!")
        self.group = Validator.check_positive_int(group, "group", self.cls_name)

        weight_shape = [out_channels, in_channels // group, *self.kernel_size]
        self.weight = Parameter(initializer(weight_init, weight_shape), name='weight')

        self.bias_add = P.BiasAdd()
        if Validator.check_bool(has_bias, "has_bias", self.cls_name):
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
        self.fake_quant_weight = quant_config.weight(ema=False,
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
        """Display instance object as string."""
        s = 'in_channels={}, out_channels={}, kernel_size={}, stride={}, ' \
            'pad_mode={}, padding={}, dilation={}, group={}, ' \
            'has_bias={}, quant_delay={}'.format(self.in_channels, self.out_channels, self.kernel_size, self.stride,
                                                 self.pad_mode, self.padding, self.dilation, self.group,
                                                 self.has_bias, self.fake_quant_weight.quant_delay)
        return s


class DenseQuant(Cell):
    r"""
    The fully connected layer with fake quantized operation.

    This part is a more detailed overview of Dense operation. For more details about Quantization,
    please refer to the implementation of subclass of class:`_Observer`, for example,
    class:`mindspore.nn.FakeQuantWithMinMaxObserver`.

    Args:
        in_channels (int): The dimension of the input space.
        out_channels (int): The dimension of the output space.
        weight_init (Union[Tensor, str, Initializer, numbers.Number]): The trainable weight_init parameter. The dtype
            is same as `x`. The values of str refer to the function `initializer`. Default: 'normal'.
        bias_init (Union[Tensor, str, Initializer, numbers.Number]): The trainable bias_init parameter. The dtype is
            same as `x`. The values of str refer to the function `initializer`. Default: 'zeros'.
        has_bias (bool): Specifies whether the layer uses a bias vector. Default: True.
        activation (Union[str, Cell, Primitive]): The regularization function applied to the output of the layer,
            eg. 'relu'. Default: None.
        quant_config (QuantConfig): Configures the types of quant observer and quant settings of weight and
            activation. Note that, QuantConfig is a special namedtuple, which is designed for quantization
            and can be generated by :func:`mindspore.compression.quant.create_quant_config` method.
            Default: QuantConfig with both items set to default :class:`FakeQuantWithMinMaxObserver`.
        quant_dtype (QuantDtype): Specifies the FakeQuant datatype. Default: QuantDtype.INT8.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.
          The input dimension is preferably 2D or 4D.

    Outputs:
        Tensor of shape :math:`(N, C_{out}, H_{out}, W_{out})`.

    Raises:
        TypeError: If `in_channels`, `out_channels` is not an int.
        TypeError: If `has_bias` is not a bool.
        TypeError: If `activation` is not str, Cell and Primitive.
        ValueError: If `in_channels` or `out_channels` is less than 1.
        ValueError: If the dims of `weight_init` is not equal to 2 or the first element of `weight_init` is not equal
            to `out_channels` or the second element of `weight_init` is not equal to `in_channels`.
        ValueError: If the dims of `bias_init` is not equal to 1 or the element of `bias_init` is not equal
            to `out_channels`.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import mindspore
        >>> from mindspore.compression import quant
        >>> from mindspore import Tensor
        >>> qconfig = quant.create_quant_config()
        >>> dense_quant = nn.DenseQuant(2, 1, weight_init='ones', quant_config=qconfig)
        >>> x = Tensor(np.array([[1, 5], [3, 4]]), mindspore.float32)
        >>> result = dense_quant(x)
        >>> print(result)
        [[5.929413]
         [6.9176483]]
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
        """Initialize DenseQuant."""
        super(DenseQuant, self).__init__()
        self.in_channels = Validator.check_positive_int(in_channels, "in_channels", self.cls_name)
        self.out_channels = Validator.check_positive_int(out_channels, "out_channels", self.cls_name)
        self.has_bias = Validator.check_bool(has_bias, "has_bias", self.cls_name)

        if isinstance(weight_init, Tensor):
            if weight_init.ndim != 2 or weight_init.shape[0] != out_channels or \
                    weight_init.shape[1] != in_channels:
                raise ValueError(f"For '{self.cls_name}', weight init shape error. The ndim of 'weight_init' should "
                                 f"be equal to 2, and the first dim should be equal to 'out_channels', and the "
                                 f"second dim should be equal to 'in_channels'. But got 'weight_init': {weight_init}, "
                                 f"'out_channels': {out_channels}, 'in_channels': {in_channels}.")

        self.weight = Parameter(initializer(
            weight_init, [out_channels, in_channels]), name="weight")

        if self.has_bias:
            if isinstance(bias_init, Tensor):
                if bias_init.ndim != 1 or bias_init.shape[0] != out_channels:
                    raise ValueError(f"For '{self.cls_name}', bias init shape error. The ndim of 'bias_init' should "
                                     f"be equal to 1, and the first dim should be equal to 'out_channels'. But got "
                                     f"'bias_init': {bias_init}, 'out_channels': {out_channels}.")

            self.bias = Parameter(initializer(
                bias_init, [out_channels]), name="bias")

        self.matmul = P.MatMul(transpose_b=True)
        self.bias_add = P.BiasAdd()

        self.activation = get_activation(activation) if isinstance(activation, str) else activation
        if activation is not None and not isinstance(self.activation, (Cell, Primitive)):
            raise TypeError(f"For '{self.cls_name}', the 'activation' must be str or Cell or Primitive, "
                            f"but got {activation}.")

        self.activation_flag = self.activation is not None
        self.fake_quant_weight = quant_config.weight(ema=False,
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

    Add the fake quantized operation to the end of activation operation, by which the output of activation
    operation will be truncated. For more details about Quantization, please refer to the implementation
    of subclass of class:`_Observer`, for example, class:`mindspore.nn.FakeQuantWithMinMaxObserver`.

    Args:
        activation (Cell): Activation cell.
        ema (bool): The exponential Moving Average algorithm updates min and max. Default: False.
        ema_decay (float): Exponential Moving Average algorithm parameter. Default: 0.999.
        fake_before (bool): Whether add fake quantized operation before activation. Default: False.
        quant_config (QuantConfig): Configures the types of quant observer and quant settings of weight and
            activation. Note that, QuantConfig is a special namedtuple, which is designed for quantization
            and can be generated by :func:`mindspore.compression.quant.create_quant_config` method.
            Default: QuantConfig with both items set to default :class:`FakeQuantWithMinMaxObserver`.
        quant_dtype (QuantDtype): Specifies the FakeQuant datatype. Default: QuantDtype.INT8.

    Inputs:
        - **x** (Tensor) - The input of ActQuant. The input dimension is preferably 2D or 4D.

    Outputs:
        Tensor, with the same type and shape as the `x`.

    Raises:
        TypeError: If `activation` is not an instance of Cell.
        TypeError: If `fake_before` is not a bool.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import mindspore
        >>> from mindspore.compression import quant
        >>> from mindspore import Tensor
        >>> qconfig = quant.create_quant_config()
        >>> act_quant = nn.ActQuant(nn.ReLU(), quant_config=qconfig)
        >>> x = Tensor(np.array([[1, 2, -1], [-2, 0, -1]]), mindspore.float32)
        >>> result = act_quant(x)
        >>> print(result)
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
        """Initialize ActQuant."""
        super(ActQuant, self).__init__()
        act_class = activation.__class__
        act_list = [nn.ReLU, nn.ReLU6]
        self.act = Validator.check_isinstance("activation", activation, Cell)
        self.fake_before = Validator.check_bool(fake_before, "fake_before", self.cls_name)
        if self.fake_before:
            self.fake_quant_act_before = quant_config.activation(min_init=-6,
                                                                 max_init=6,
                                                                 ema=ema,
                                                                 ema_decay=ema_decay,
                                                                 quant_dtype=quant_dtype)
        self.neg_trunc = False
        self.narrow_range = False
        preset_dict = quant_config.activation.p.keywords
        if 'mode' in preset_dict and preset_dict['mode'] == "LEARNED_SCALE" and act_class in act_list:
            self.neg_trunc = True
        elif 'narrow_range' in preset_dict:
            self.narrow_range = preset_dict['narrow_range']

        self.fake_quant_act = quant_config.activation(min_init=-6,
                                                      max_init=6,
                                                      ema=ema,
                                                      ema_decay=ema_decay,
                                                      quant_dtype=quant_dtype,
                                                      neg_trunc=self.neg_trunc,
                                                      narrow_range=self.narrow_range)

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

    This part is a more detailed overview of TensorAdd operation. For more details about Quantization,
    please refer to the implementation of subclass of class:`_Observer`, for example,
    class:`mindspore.nn.FakeQuantWithMinMaxObserver`.

    Args:
        ema_decay (float): Exponential Moving Average algorithm parameter. Default: 0.999.
        quant_config (QuantConfig): Configures the types of quant observer and quant settings of weight and
            activation. Note that, QuantConfig is a special namedtuple, which is designed for quantization
            and can be generated by :func:`mindspore.compression.quant.create_quant_config` method.
            Default: QuantConfig with both items set to default :class:`FakeQuantWithMinMaxObserver`.
        quant_dtype (QuantDtype): Specifies the FakeQuant datatype. Default: QuantDtype.INT8.

    Inputs:
        - **x1** (Tensor) - The first tensor of TensorAddQuant. The input dimension is preferably 2D or 4D.
        - **x2** (Tensor) - The second tensor of TensorAddQuant. Has the same shape with `x1`.

    Outputs:
        Tensor, with the same type and shape as the `x1`.

    Raises:
        TypeError: If `ema_decay` is not a float.
        ValueError: If the shape of `x2` is different with `x1`.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import mindspore
        >>> from mindspore.compression import quant
        >>> from mindspore import Tensor
        >>> qconfig = quant.create_quant_config()
        >>> add_quant = nn.TensorAddQuant(quant_config=qconfig)
        >>> x1 = Tensor(np.array([[1, 2, 1], [-2, 0, -1]]), mindspore.float32)
        >>> x2 = Tensor(np.ones((2, 3)), mindspore.float32)
        >>> output = add_quant(x1, x2)
        >>> print(output)
        [[ 1.9764705  3.011765   1.9764705]
         [-0.9882355  0.9882355  0.       ]]
    """

    def __init__(self,
                 ema_decay=0.999,
                 quant_config=quant_config_default,
                 quant_dtype=QuantDtype.INT8):
        """Initialize TensorAddQuant."""
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

    This part is a more detailed overview of `Mul` operation. For more details about Quantization,
    please refer to the implementation of subclass of class:`_Observer`, for example,
    class:`mindspore.nn.FakeQuantWithMinMaxObserver`.

    Args:
        ema_decay (float): Exponential Moving Average algorithm parameter. Default: 0.999.
        quant_config (QuantConfig): Configures the types of quant observer and quant settings of weight and
            activation. Note that, QuantConfig is a special namedtuple, which is designed for quantization
            and can be generated by :func:`mindspore.compression.quant.create_quant_config` method.
            Default: QuantConfig with both items set to default :class:`FakeQuantWithMinMaxObserver`.
        quant_dtype (QuantDtype): Specifies the FakeQuant datatype. Default: QuantDtype.INT8.

    Inputs:
        - **x1** (Tensor) - The first tensor of MulQuant. The input dimension is preferably 2D or 4D.
        - **x2** (Tensor) - The second tensor of MulQuant. Has the same shape with `x1`.

    Outputs:
        Tensor, with the same type and shape as the `x1`.

    Raises:
        TypeError: If `ema_decay` is not a float.
        ValueError: If the shape of `x2` is different with `x1`.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import mindspore
        >>> from mindspore.compression import quant
        >>> from mindspore import Tensor
        >>> qconfig = quant.create_quant_config()
        >>> mul_quant = nn.MulQuant(quant_config=qconfig)
        >>> x1 = Tensor(np.array([[1, 2, 1], [-2, 0, -1]]), mindspore.float32)
        >>> x2 = Tensor(np.ones((2, 3)) * 2, mindspore.float32)
        >>> output = mul_quant(x1, x2)
        >>> print(output)
        [[ 1.9764705  4.0000005  1.9764705]
         [-4.         0.        -1.9764705]]
    """

    def __init__(self,
                 ema_decay=0.999,
                 quant_config=quant_config_default,
                 quant_dtype=QuantDtype.INT8):
        """Initialize MulQuant."""
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
