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
"""conv"""
from __future__ import absolute_import

import math
import numpy as np

from mindspore import context
from mindspore.ops import operations as P
import mindspore.common.dtype as mstype
from mindspore.ops.primitive import _primexpr
from mindspore.common.parameter import Parameter
from mindspore.common.initializer import initializer, HeUniform, Uniform, _calculate_fan_in_and_fan_out
from mindspore.common.tensor import Tensor
from mindspore import _checkparam as Validator
from mindspore._checkparam import twice, _check_3d_int_or_tuple
from mindspore._extends import cell_attr_register
from mindspore.nn.cell import Cell

__all__ = ['Conv2d', 'Conv2dTranspose', 'Conv1d', 'Conv1dTranspose', 'Conv3d', 'Conv3dTranspose']


class _Conv(Cell):
    """
    Applies a N-D convolution over an input signal composed of several input planes.
    """

    def __init__(self,
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
                 bias_init,
                 data_format='NCHW',
                 transposed=False,
                 dtype=mstype.float32):
        """Initialize _Conv."""
        super(_Conv, self).__init__()
        self.in_channels = Validator.check_positive_int(in_channels, 'in_channels', self.cls_name)
        self.out_channels = Validator.check_positive_int(out_channels, 'out_channels', self.cls_name)
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad_mode = pad_mode
        self.data_format = Validator.check_string(data_format, ['NCHW', 'NHWC', 'NCDHW'], 'format', self.cls_name)
        if context.get_context("device_target") != "GPU" and self.data_format == "NHWC":
            raise ValueError(f"For '{self.cls_name}', the \"NHWC\" format only support in GPU target, "
                             f"but got the 'format' is {self.data_format} and "
                             f"the platform is {context.get_context('device_target')}.")
        if isinstance(padding, int):
            Validator.check_non_negative_int(padding, 'padding', self.cls_name)
            self.padding = padding
        elif isinstance(padding, tuple):
            for pad in padding:
                Validator.check_non_negative_int(pad, 'padding item', self.cls_name)
            self.padding = padding
        else:
            raise TypeError(f"For '{self.cls_name}', the type of 'padding' must be int or tuple(int), "
                            f"but got {type(padding).__name__}.")

        self.dilation = dilation
        self.group = Validator.check_positive_int(group)
        self.has_bias = has_bias
        for kernel_size_elem in kernel_size:
            Validator.check_positive_int(kernel_size_elem, 'kernel_size item', self.cls_name)
        for stride_elem in stride:
            Validator.check_positive_int(stride_elem, 'stride item', self.cls_name)
        for dilation_elem in dilation:
            Validator.check_positive_int(dilation_elem, 'dilation item', self.cls_name)
        if in_channels % group != 0:
            raise ValueError(f"For '{self.cls_name}', the attr 'in_channels' must be divisible by attr 'group', "
                             f"but got 'in_channels': {in_channels} and 'group': {group}.")
        if out_channels % group != 0:
            raise ValueError(f"For '{self.cls_name}', the 'out_channels' must be divisible by attr 'group', "
                             f"but got 'out_channels': {out_channels} and 'group': {group}.")
        if transposed:
            shape = [in_channels, out_channels // group, *kernel_size]
        else:
            shape = [out_channels, *kernel_size, in_channels // group] if self.data_format == "NHWC" else \
                [out_channels, in_channels // group, *kernel_size]
        if weight_init is None:
            weight_init = HeUniform(math.sqrt(5))
        self.weight_init = weight_init
        self.weight = Parameter(initializer(self.weight_init, shape, dtype=dtype), name='weight')

        self.bias_init = bias_init
        if Validator.check_bool(has_bias, "has_bias", self.cls_name):
            if bias_init is None:
                fan_in, _ = _calculate_fan_in_and_fan_out(shape)
                if fan_in != 0:
                    bound = 1 / math.sqrt(fan_in)
                    bias_init = Uniform(bound)
                else:
                    bias_init = 'zeros'
                self.bias_init = bias_init
            self.bias = Parameter(initializer(self.bias_init, [out_channels], dtype=dtype), name='bias')
        else:
            self.bias = None

    def construct(self, *inputs):
        """Must be overridden by all subclasses."""
        raise NotImplementedError

    def extend_repr(self):
        s = 'input_channels={}, output_channels={}, kernel_size={}, ' \
            'stride={}, pad_mode={}, padding={}, dilation={}, ' \
            'group={}, has_bias={}, ' \
            'weight_init={}, bias_init={}, format={}'.format(
                self.in_channels,
                self.out_channels,
                self.kernel_size,
                self.stride,
                self.pad_mode,
                self.padding,
                self.dilation,
                self.group,
                self.has_bias,
                self.weight_init,
                self.bias_init,
                self.data_format)
        return s


class Conv2d(_Conv):
    r"""
    2D convolution layer.

    Applies a 2D convolution over an input tensor which is typically of shape :math:`(N, C_{in}, H_{in}, W_{in})`,
    where :math:`N` is batch size, :math:`C` is channel number, :math:`H` is feature height, :math:`W` is feature width.

    The output is calculated based on formula:

    .. math::

        \text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) +
        \sum_{k = 0}^{C_{in} - 1} \text{ccor}({\text{weight}(C_{\text{out}_j}, k), \text{X}(N_i, k)})

    where :math:`bias` is the output channel bias, :math:`ccor` is
    the `cross-correlation <https://en.wikipedia.org/wiki/Cross-correlation>`_,
    , :math:`weight` is the convolution kernel value and :math:`X` represents the input feature map.

    Here are the indices' meanings:
    - :math:`i` corresponds to the batch number, ranging from 0 to N-1, where N is the batch size of the input.

    - :math:`j` corresponds to the output channel, ranging from 0 to C_{out}-1, where C_{out} is the number of
      output channels, which is also equal to the number of kernels.

    - :math:`k` corresponds to the input channel, ranging from 0 to C_{in}-1, where C_{in} is the number of
      input channels, which is also equal to the number of channels in the convolutional kernels.

    Therefore, in the above formula, :math:`{bias}(C_{out_j})` represents the bias of the :math:`j`-th
    output channel, :math:`{weight}(C_{out_j}, k)` represents the slice of the :math:`j`-th convolutional
    kernel in the :math:`k`-th channel, and :math:`{X}(N_i, k)` represents the slice of the :math:`k`-th input
    channel in the :math:`i`-th batch of the input feature map.

    The shape of the convolutional kernel is given by :math:`(kernel\_size[0], kernel\_size[1])`,
    where :math:`kernel\_size[0]` and :math:`kernel\_size[1]` are the height and width of the kernel, respectively.
    If we consider the input and output channels as well as the `group` parameter, the complete kernel shape
    will be :math:`(C_{out}, C_{in} / \text{group}, \text{kernel_size[0]}, \text{kernel_size[1]})`,
    where `group` is the number of groups dividing `x`'s input channel when applying group convolution.

    For more details about convolution layer, please refer to `Gradient Based Learning Applied to Document Recognition
    <http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf>`_.

    Note:
        On Ascend platform, only group convolution in depthwise convolution scenarios is supported.
        That is, when `group>1`, condition `in\_channels` = `out\_channels` = `group` must be satisfied.

    Args:
        in_channels (int): The channel number of the input tensor of the Conv2d layer.
        out_channels (int): The channel number of the output tensor of the Conv2d layer.
        kernel_size (Union[int, tuple[int]]): Specifies the height and width of the 2D convolution kernel.
            The data type is an integer or a tuple of two integers. An integer represents the height
            and width of the convolution kernel. A tuple of two integers represents the height
            and width of the convolution kernel respectively.
        stride (Union[int, tuple[int]], optional): The movement stride of the 2D convolution kernel.
            The data type is an integer or a tuple of two or four integers. An integer represents the movement step size
            in both height and width directions. A tuple of two integers represents the movement step size in the height
            and width directions respectively. Default: ``1`` .
        pad_mode (str, optional): Specifies the padding mode with a padding value of 0. It can be set to:
            ``"same"`` , ``"valid"`` or ``"pad"`` . Default: ``"same"`` .

            - ``"same"``: Pad the input around its edges so that the shape of input and output
              are the same when `stride` is set to ``1``.
              The amount of padding to is calculated by the operator internally, If the amount is even, it is
              uniformly distributed around the input, if it is odd, the excess amount goes to the right/bottom side.
              If this mode is set, `padding` must be 0.
            - ``"valid"``: No padding is applied to the input, and the output returns the maximum
              possible height and width. Extra pixels that could not complete a full stride will
              be discarded. If this mode is set, `padding` must be 0.
            - ``"pad"``: Pad the input with a specified amount. In this mode, the amount of padding
              in the height and width directions is determined by the `padding` parameter.
              If this mode is set, `padding` must be greater than or equal to 0.

        padding (Union[int, tuple[int]], optional): The number of padding
            on the height and width directions of the input.
            The data type is an integer or a tuple of four integers. If `padding` is an integer,
            then the top, bottom, left, and right padding are all equal to `padding`.
            If `padding` is a tuple of 4 integers, then the top, bottom, left, and right padding
            is equal to `padding[0]`, `padding[1]`, `padding[2]`, and `padding[3]` respectively.
            The value should be greater than or equal to 0. Default: ``0`` .
        dilation (Union(int, tuple[int]), optional): Specifies the dilation rate to use for dilated convolution.
            It can be a single int or a tuple of 2 or 4 integers. A single int means the dilation size is the same
            in both the height and width directions. A tuple of two ints represents the dilation size in
            the height and width directions, respectively. For a tuple of four ints, the two ints correspond
            to (N, C) dimension are treated as 1, and the two correspond to (H, W) dimensions is the
            dilation size in the height and width directions respectively.
            Assuming :math:`dilation=(d0, d1)`, the convolutional kernel samples the input with a
            spacing of :math:`d0-1` elements in the height direction and :math:`d1-1` elements in the width direction.
            The values in the height and width dimensions are in the ranges [1, H] and [1, W], respectively.
            Default: ``1`` .
        group (int, optional): Splits filter into groups, `in_channels` and `out_channels` must be
            divisible by `group`. If the group is equal to `in_channels` and `out_channels`,
            this 2D convolution layer also can be called 2D depthwise convolution layer. Default: ``1`` .
        has_bias (bool, optional): Whether the Conv2d layer has a bias parameter. Default: ``False`` .
        weight_init (Union[Tensor, str, Initializer, numbers.Number], optional): Initialization method of
            weight parameter.
            It can be a Tensor, a string, an Initializer or a numbers.Number. When a string is specified,
            values from ``'TruncatedNormal'`` , ``'Normal'`` , ``'Uniform'`` , ``'HeUniform'`` and ``'XavierUniform'``
            distributions as well as constant ``'One'`` and ``'Zero'`` distributions are possible. Alias
            ``'xavier_uniform'`` , ``'he_uniform'`` , ``'ones'`` and ``'zeros'`` are acceptable. Uppercase and
            lowercase are both acceptable. Refer to the values of
            `Initializer <https://www.mindspore.cn/docs/en/master/api_python/mindspore.common.initializer.html>`_,
            for more details. Default: ``None`` , weight will be initialized using ``'HeUniform'``.
        bias_init (Union[Tensor, str, Initializer, numbers.Number], optional): Initialization method of bias parameter.
            Available initialization methods are the same as 'weight_init'. Refer to the values of
            `Initializer <https://www.mindspore.cn/docs/en/master/api_python/mindspore.common.initializer.html>`_,
            for more details. Default: ``None`` , bias will be initialized using ``'Uniform'`` .
        data_format (str, optional): The optional value for data format, is ``'NHWC'`` or ``'NCHW'`` .
            Default: ``'NCHW'`` .
        dtype (:class:`mindspore.dtype`): Dtype of Parameters. Default: ``mstype.float32`` .

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})` \
          or :math:`(N, H_{in}, W_{in}, C_{in})`.

    Outputs:
        Tensor of shape :math:`(N, C_{out}, H_{out}, W_{out})` or :math:`(N, H_{out}, W_{out}, C_{out})`.

        pad_mode is ``'same'``:

        .. math::
            \begin{array}{ll} \\
                H_{out} = \left \lceil{\frac{H_{in}}{\text{stride[0]}}} \right \rceil \\
                W_{out} = \left \lceil{\frac{W_{in}}{\text{stride[1]}}} \right \rceil \\
            \end{array}

        pad_mode is ``'valid'``:

        .. math::
            \begin{array}{ll} \\
                H_{out} = \left \lceil{\frac{H_{in} - \text{dilation[0]} \times (\text{kernel_size[0]} - 1) }
                {\text{stride[0]}}} \right \rceil \\
                W_{out} = \left \lceil{\frac{W_{in} - \text{dilation[1]} \times (\text{kernel_size[1]} - 1) }
                {\text{stride[1]}}} \right \rceil \\
            \end{array}

        pad_mode is ``'pad'``:

        .. math::
            \begin{array}{ll} \\
                H_{out} = \left \lfloor{\frac{H_{in} + padding[0] + padding[1] - (\text{kernel_size[0]} - 1) \times
                \text{dilation[0]} - 1 }{\text{stride[0]}} + 1} \right \rfloor \\
                W_{out} = \left \lfloor{\frac{W_{in} + padding[2] + padding[3] - (\text{kernel_size[1]} - 1) \times
                \text{dilation[1]} - 1 }{\text{stride[1]}} + 1} \right \rfloor \\
            \end{array}

    Raises:
        TypeError: If `in_channels`, `out_channels` or `group` is not an int.
        TypeError: If `kernel_size`, `stride`, `padding` or `dilation` is neither an int not a tuple.
        ValueError: If `in_channels`, `out_channels`, `kernel_size`, `stride` or `dilation` is less than 1.
        ValueError: If `padding` is less than 0.
        ValueError: If `pad_mode` is not one of 'same', 'valid', 'pad'.
        ValueError: If `padding` is a tuple whose length is not equal to 4.
        ValueError: If `pad_mode` is not equal to 'pad' and `padding` is not equal to (0, 0, 0, 0).
        ValueError: If `data_format` is neither 'NCHW' not 'NHWC'.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> from mindspore import Tensor, nn
        >>> import numpy as np
        >>> net = nn.Conv2d(120, 240, 4, has_bias=False, weight_init='normal')
        >>> x = Tensor(np.ones([1, 120, 1024, 640]), mindspore.float32)
        >>> output = net(x).shape
        >>> print(output)
        (1, 240, 1024, 640)
    """

    @cell_attr_register
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
                 data_format='NCHW',
                 dtype=mstype.float32):
        """Initialize Conv2d."""
        kernel_size = twice(kernel_size)
        stride = twice(stride)
        self._dilation = dilation
        dilation = twice(dilation)
        Validator.check_positive_int(group, 'group', self.cls_name)
        if not (in_channels % group == 0 and out_channels % group == 0):
            raise ValueError(f"The argument 'group' should be divisible by 'in_channels' " \
                             f"and 'out_channels', but got group:{group}, in_channels:{in_channels}, " \
                             f"out_channels:{out_channels}.")
        super(Conv2d, self).__init__(
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
            bias_init,
            data_format,
            dtype=dtype)
        self.conv2d = P.Conv2D(out_channel=self.out_channels,
                               kernel_size=self.kernel_size,
                               mode=1,
                               pad_mode=self.pad_mode,
                               pad=self.padding,
                               stride=self.stride,
                               dilation=self.dilation,
                               group=self.group,
                               data_format=self.data_format)
        self.bias_add = P.BiasAdd(data_format=self.data_format)

    def construct(self, x):
        output = self.conv2d(x, self.weight)
        if self.has_bias:
            output = self.bias_add(output, self.bias)
        return output


@_primexpr
def _check_input_3d(input_shape, op_name):
    if len(input_shape) != 3:
        raise ValueError(f"For '{op_name}', the dimension of input must be 3d, but got {len(input_shape)}.")


class Conv1d(_Conv):
    r"""
    1D convolution layer.

    Applies a 1D convolution over an input tensor which is typically of shape :math:`(N, C_{in}, L_{in})`,
    where :math:`N` is batch size, :math:`C` is channel number, :math:`L` is input sequence width.

    The output is calculated based on formula:

    .. math::

        \text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) +
        \sum_{k = 0}^{C_{in} - 1} \text{ccor}({\text{weight}(C_{\text{out}_j}, k), \text{X}(N_i, k)})

    where :math:`bias` is the output channel bias, :math:`ccor` is
    the `cross-correlation <https://en.wikipedia.org/wiki/Cross-correlation>`_,
    , :math:`weight` is the convolution kernel value and :math:`X` represents the input feature map.

    Here are the indices' meanings:
    - :math:`i` corresponds to the batch number, ranging from 0 to N-1, where N is the batch size of the input.

    - :math:`j` corresponds to the output channel, ranging from 0 to C_{out}-1, where C_{out} is the number of
      output channels, which is also equal to the number of kernels.

    - :math:`k` corresponds to the input channel, ranging from 0 to C_{in}-1, where C_{in} is the number of
      input channels, which is also equal to the number of channels in the convolutional kernels.

    Therefore, in the above formula, :math:`{bias}(C_{out_j})` represents the bias of the :math:`j`-th
    output channel, :math:`{weight}(C_{out_j}, k)` represents the slice of the :math:`j`-th convolutional
    kernel in the :math:`k`-th channel, and :math:`{X}(N_i, k)` represents the slice of the :math:`k`-th input
    channel in the :math:`i`-th batch of the input feature map.

    The shape of the convolutional kernel is given by :math:`(kernel\_size)`,
    where :math:`kernel\_size` is the width of the kernel.
    If we consider the input and output channels as well as the `group` parameter, the complete kernel shape
    will be :math:`(C_{out}, C_{in} / \text{group}, \text{kernel_size})`,
    where `group` is the number of groups dividing `x`'s input channel when applying group convolution.

    For more details about convolution layer, please refer to `Gradient Based Learning Applied to Document Recognition
    <http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf>`_
    and `ConvNets <http://cs231n.github.io/convolutional-networks/>`_ .

    Note:
        On Ascend platform, only group convolution in depthwise convolution scenarios is supported.
        That is, when `group>1`, condition `in\_channels` = `out\_channels` = `group` must be satisfied.

    Args:
        in_channels (int): The channel number of the input tensor of the Conv1d layer.
        out_channels (int): The channel number of the output tensor of the Conv1d layer.
        kernel_size (int): Specifies the width of the 1D convolution kernel.
        stride (int, optional): The movement stride of the 1D convolution kernel. Default: ``1`` .
        pad_mode (str, optional): Specifies the padding mode with a padding value of 0. It can be set to:
            ``"same"`` , ``"valid"`` or ``"pad"`` . Default: ``"same"`` .

            - ``"same"``: Pad the input at the begin and end so that the shape of input and output
              are the same when `stride` is set to ``1``.
              The amount of padding to is calculated by the operator internally. If the amount is even, it is
              uniformly distributed around the input, if it is odd, the excess padding is goes to the right side.
              If this mode is set, `padding` must be 0.
            - ``"valid"``: No padding is applied to the input, and the output returns the maximum
              possible length. Extra pixels that could not complete a full stride will
              be discarded. If this mode is set, `padding` must be 0.
            - ``"pad"``: Pad the input with a specified amount. In this mode, the amount of padding
              at the begin and end is determined by the `padding` parameter.
              If this mode is set, `padding` must be greater than or equal to 0.

        padding (Union(int, tuple[int], list[int]), optional):  Specifies the amount of padding to apply on
            both side of `input` when `pad_mode` is set to ``"pad"``. The
            paddings of left and right are the same, equal to padding or padding[0] when padding is a tuple of
            1 integer. Default: ``0`` .
        dilation (Union(int, tuple[int]), optional): Specifies the dilation rate to use for dilated convolution.
            It can be a single int or a tuple of 1 integer.
            Assuming :math:`dilation=(d0,)`, the convolutional kernel samples the input with a
            spacing of :math:`d0-1` elements in the width direction.
            The value should be in the ranges [1, L].
            Default: ``1`` .
        group (int, optional): Splits filter into groups, `in_channels` and `out_channels` must be
            divisible by `group`. Default: ``1`` .
        has_bias (bool, optional): Whether the Conv1d layer has a bias parameter. Default: ``False`` .
        weight_init (Union[Tensor, str, Initializer, numbers.Number], optional):
            Initialization method of weight parameter.
            It can be a Tensor, a string, an Initializer or a numbers.Number. When a string is specified,
            values from ``'TruncatedNormal'`` , ``'Normal'`` , ``'Uniform'`` , ``'HeUniform'`` and ``'XavierUniform'``
            distributions as well as constant 'One' and 'Zero' distributions are possible. Alias ``'xavier_uniform'`` ,
            ``'he_uniform'`` , ``'ones'`` and ``'zeros'`` are acceptable. Uppercase and lowercase are both acceptable.
            Refer to the values of
            `Initializer <https://www.mindspore.cn/docs/en/master/api_python/mindspore.common.initializer.html>`_,
            for more details. Default: ``None`` , weight will be initialized using ``'HeUniform'``.
        bias_init (Union[Tensor, str, Initializer, numbers.Number], optional): Initialization method of bias parameter.
            Available initialization methods are the same as 'weight_init'. Refer to the values of
            `Initializer <https://www.mindspore.cn/docs/en/master/api_python/mindspore.common.initializer.html>`_,
            for more details. Default: ``None`` , bias will be initialized using ``'Uniform'``.
        dtype (:class:`mindspore.dtype`): Dtype of Parameters. Default: ``mstype.float32`` .

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, L_{in})` .

    Outputs:
        Tensor of shape :math:`(N, C_{out}, L_{out})`.

        pad_mode is ``'same'``:

        .. math::
            L_{out} = \left \lceil{\frac{L_{in}}{\text{stride}}} \right \rceil

        pad_mode is ``'valid'``:

        .. math::
            L_{out} = \left \lceil{\frac{L_{in} - \text{dilation} \times (\text{kernel_size} - 1) }
            {\text{stride}}} \right \rceil

        pad_mode is ``'pad'``:

        .. math::
            L_{out} = \left \lfloor{\frac{L_{in} + 2 \times padding - (\text{kernel_size} - 1) \times
            \text{dilation} - 1 }{\text{stride}} + 1} \right \rfloor

    Raises:
        TypeError: If `in_channels`, `out_channels`, `kernel_size`, `stride`, `padding` or `dilation` is not an int.
        ValueError: If `in_channels`, `out_channels`, `kernel_size`, `stride` or `dilation` is less than 1.
        ValueError: If `padding` is less than 0.
        ValueError: If `pad_mode` is not one of 'same', 'valid', 'pad'.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> from mindspore import Tensor, nn
        >>> import numpy as np
        >>> net = nn.Conv1d(120, 240, 4, has_bias=False, weight_init='normal')
        >>> x = Tensor(np.ones([1, 120, 640]), mindspore.float32)
        >>> output = net(x).shape
        >>> print(output)
        (1, 240, 640)
    """

    @cell_attr_register
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
                 dtype=mstype.float32):
        """Initialize Conv1d."""
        Validator.check_value_type("kernel_size", kernel_size, [int], self.cls_name)
        Validator.check_value_type("stride", stride, [int], self.cls_name)
        Validator.check_value_type("padding", padding, [int], self.cls_name)
        Validator.check_value_type("dilation", dilation, [int], self.cls_name)
        Validator.check_int(kernel_size, 1, Validator.GE, 'kernel_size', self.cls_name)
        Validator.check_int(stride, 1, Validator.GE, 'stride', self.cls_name)
        Validator.check_non_negative_int(padding, 'padding', self.cls_name)
        Validator.check_int(dilation, 1, Validator.GE, 'dilation', self.cls_name)
        Validator.check_positive_int(group, 'group', self.cls_name)
        if not (in_channels % group == 0 and out_channels % group == 0):
            raise ValueError(f"The argument 'group' should be divisible by 'in_channels' " \
                             f"and 'out_channels', but got group:{group}, in_channels:{in_channels}, " \
                             f"out_channels:{out_channels}.")
        kernel_size = (1, kernel_size)
        stride = (1, stride)
        dilation = (1, dilation)
        get_shape = P.Shape()
        get_dtype = P.DType()
        if isinstance(weight_init, Tensor):
            weight_init_shape = get_shape(weight_init)
            Validator.check_equal_int(len(weight_init_shape), 3, 'weight_init_shape', self.cls_name)
            weight_init_dtype = get_dtype(weight_init)
            weight_init_value = weight_init.asnumpy()
            weight_init_value = np.expand_dims(weight_init_value, 2)
            weight_init = Tensor(weight_init_value, weight_init_dtype)

        super(Conv1d, self).__init__(
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
            bias_init,
            dtype=dtype)
        self.padding = (0, 0, padding, padding)
        Validator.check_string(pad_mode, ['valid', 'same', 'pad'], 'pad_mode', self.cls_name)
        self.conv2d = P.Conv2D(out_channel=self.out_channels,
                               kernel_size=self.kernel_size,
                               mode=1,
                               pad_mode=self.pad_mode,
                               pad=self.padding,
                               stride=self.stride,
                               dilation=self.dilation,
                               group=self.group)
        self.bias_add = P.BiasAdd()
        self.expand_dims = P.ExpandDims()
        self.squeeze = P.Squeeze(2)
        self.shape = P.Shape()

    def construct(self, x):
        x_shape = self.shape(x)
        _check_input_3d(x_shape, self.cls_name)
        x = self.expand_dims(x, 2)
        output = self.conv2d(x, self.weight)
        if self.has_bias:
            output = self.bias_add(output, self.bias)

        output = self.squeeze(output)
        return output


@_primexpr
def _check_input_5dims(input_shape, op_name):
    if len(input_shape) != 5:
        raise ValueError(f"For '{op_name}', the dimension of input must be 5d, but got {len(input_shape)}.")


class Conv3d(_Conv):
    r"""
    3D convolution layer.

    Applies a 3D convolution over an input tensor which is typically of shape
    :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})`, where :math:`N` is batch size, :math:`C` is channel number,
    :math:`D` is feature depth, :math:`H` is feature height, :math:`W` is feature width.

    The output is calculated based on formula:

    .. math::

        \text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) +
        \sum_{k = 0}^{C_{in} - 1} \text{ccor}({\text{weight}(C_{\text{out}_j}, k), \text{X}(N_i, k)})

    where :math:`bias` is the output channel bias, :math:`ccor` is
    the `cross-correlation <https://en.wikipedia.org/wiki/Cross-correlation>`_,
    , :math:`weight` is the convolution kernel value and :math:`X` represents the input feature map.

    Here are the indices' meanings:
    - :math:`i` corresponds to the batch number, ranging from 0 to N-1, where N is the batch size of the input.

    - :math:`j` corresponds to the output channel, ranging from 0 to C_{out}-1, where C_{out} is the number of
      output channels, which is also equal to the number of kernels.

    - :math:`k` corresponds to the input channel, ranging from 0 to C_{in}-1, where C_{in} is the number of
      input channels, which is also equal to the number of channels in the convolutional kernels.

    Therefore, in the above formula, :math:`{bias}(C_{out_j})` represents the bias of the :math:`j`-th
    output channel, :math:`{weight}(C_{out_j}, k)` represents the slice of the :math:`j`-th convolutional
    kernel in the :math:`k`-th channel, and :math:`{X}(N_i, k)` represents the slice of the :math:`k`-th input
    channel in the :math:`i`-th batch of the input feature map.

    The shape of the convolutional kernel is given by
    :math:`(\text{kernel_size[0]}, \text{kernel_size[1]}, \text{kernel_size[2]})`
    where :math:`kernel\_size[0]` , :math:`kernel\_size[1]` and :math:`kernel\_size[2]` are the depth,
    height and width of the kernel, respectively.
    If we consider the input and output channels as well as the `group` parameter, the complete kernel shape
    will be :math:`(C_{out}, C_{in} / \text{group}, \text{kernel_size[0]},
    \text{kernel_size[1]}, \text{kernel_size[2]})`,
    where `group` is the number of groups dividing `x`'s input channel when applying group convolution.

    For more details about convolution layer, please refer to `Gradient Based Learning Applied to Document Recognition
    <http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf>`_.

    Note:
        On Ascend platform, only group convolution in depthwise convolution scenarios is supported.
        That is, when `group>1`, condition `in\_channels` = `out\_channels` = `group` must be satisfied.

    Args:
        in_channels (int): The channel number of the input tensor of the Conv3d layer.
        out_channels (int): The channel number of the output tensor of the Conv3d layer.
        kernel_size (Union[int, tuple[int]]): Specifies the depth, height and width of the 3D convolution kernel.
            It can be a single int or a tuple of 3 integers. A single int means the value is for depth, height
            and the width. A tuple of 3 ints means the first value is
            for depth and the rest is for the height and width.
        stride (Union[int, tuple[int]], optional): The movement stride of the 3D convolution kernel.
            The data type is an integer or a tuple of three integers. An integer represents the movement step size
            in depth, height and width directions. A tuple of three integers represents the movement step size
            in the depth, height and width directions respectively. Default: ``1`` .
        pad_mode (str, optional): Specifies the padding mode with a padding value of 0. It can be set to:
            ``"same"`` , ``"valid"`` or ``"pad"`` . Default: ``"same"`` .

            - ``"same"``: Pad the input around its depth/height/width dimension so that the shape of input and output
              are the same when `stride` is set to ``1``.
              The amount of padding to is calculated by the operator internally.  If the amount is even,
              it isuniformly distributed around the input, if it is odd, the excess amount goes
              to the front/right/bottom side.
              If this mode is set, `padding` must be 0.
            - ``"valid"``: No padding is applied to the input, and the output returns the maximum
              possible depth, height and width. Extra pixels that could not complete a full stride will
              be discarded. If this mode is set, `padding` must be 0.
            - ``"pad"``: Pad the input with a specified amount. In this mode, the amount of padding
              in the depth, height and width dimension is determined by the `padding` parameter.
              If this mode is set, `padding` must be greater than or equal to 0.

        padding (Union(int, tuple[int]), optional): The number of padding on the depth,
            height and width directions of the input.
            The data type is an integer or a tuple of six integers. If `padding` is an integer,
            then the head, tail, top, bottom, left, and right padding are all equal to `padding`.
            If `padding` is a tuple of six integers, then the head, tail, top, bottom, left, and right padding
            is equal to `padding[0]`, `padding[1]`, `padding[2]`, `padding[3]`, `padding[4]` and `padding[5]`
            respectively. The value should be greater than or equal to 0. Default: ``0`` .
        dilation (Union[int, tuple[int]], optional): Specifies the dilation rate to use for dilated convolution.
            It can be a single int or a tuple of 3 integers. A single int means the dilation size is the same
            in the depth, height and width directions. A tuple of 3 ints represents the dilation size in
            the depth, height and width directions, respectively.
            Assuming :math:`dilation=(d0, d1, d2)`, the convolutional kernel samples the input with a
            spacing of :math:`d0-1` elements in the depth direction, :math:`d1-1` elements in the height direction,
            :math:`d2-1` elements in the width direction respectively.
            The values in the depth, height and width dimensions are in
            the ranges [1, D], [1, H] and [1, W], respectively.
            Default: ``1`` .
        group (int, optional): Splits filter into groups, `in_channels` and `out_channels` must be
            divisible by `group`. Default: ``1`` .
        has_bias (bool, optional): Whether the Conv3d layer has a bias parameter. Default: ``False`` .
        weight_init (Union[Tensor, str, Initializer, numbers.Number], optional):
            Initialization method of weight parameter.
            It can be a Tensor, a string, an Initializer or a numbers.Number. When a string is specified,
            values from ``'TruncatedNormal'`` , ``'Normal'`` , ``'Uniform'`` , ``'HeUniform'`` and ``'XavierUniform'``
            distributions as well as constant ``'One'`` and ``'Zero'`` distributions are possible. Alias
            ``'xavier_uniform'`` , ``'he_uniform'`` , ``'ones'`` and ``'zeros'`` are acceptable. Uppercase and
            lowercase are both acceptable. Refer to the values of
            `Initializer <https://www.mindspore.cn/docs/en/master/api_python/mindspore.common.initializer.html>`_,
            for more details. Default: ``None`` , weight will be initialized using ``'HeUniform'``.
        bias_init (Union[Tensor, str, Initializer, numbers.Number], optional): Initialization method of bias parameter.
            Available initialization methods are the same as 'weight_init'. Refer to the values of
            `Initializer <https://www.mindspore.cn/docs/en/master/api_python/mindspore.common.initializer.html>`_,
            for more details. Default: ``None`` , bias will be initialized using ``'Uniform'`` .
        data_format (str, optional): The optional value for data format. Currently only support ``'NCDHW'`` .
        dtype (:class:`mindspore.dtype`): Dtype of Parameters. Default: ``mstype.float32`` .


    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})`.
          Currently input data type only support float16 and float32.

    Outputs:
        Tensor of shape is :math:`(N, C_{out}, D_{out}, H_{out}, W_{out})`.

        pad_mode is ``'same'`` :

        .. math::
            \begin{array}{ll} \\
                D_{out} = \left \lceil{\frac{D_{in}}{\text{stride[0]}}} \right \rceil \\
                H_{out} = \left \lceil{\frac{H_{in}}{\text{stride[1]}}} \right \rceil \\
                W_{out} = \left \lceil{\frac{W_{in}}{\text{stride[2]}}} \right \rceil \\
            \end{array}


        pad_mode is ``'valid'`` :

        .. math::
            \begin{array}{ll} \\
                D_{out} = \left \lfloor{\frac{D_{in} - \text{dilation[0]} \times (\text{kernel_size[0]} - 1) }
                {\text{stride[0]}} + 1} \right \rfloor \\
                H_{out} = \left \lfloor{\frac{H_{in} - \text{dilation[1]} \times (\text{kernel_size[1]} - 1) }
                {\text{stride[1]}} + 1} \right \rfloor \\
                W_{out} = \left \lfloor{\frac{W_{in} - \text{dilation[2]} \times (\text{kernel_size[2]} - 1) }
                {\text{stride[2]}} + 1} \right \rfloor \\
            \end{array}

        pad_mode is ``'pad'`` :

        .. math::
            \begin{array}{ll} \\
                D_{out} = \left \lfloor{\frac{D_{in} + padding[0] + padding[1] - (\text{dilation[0]} - 1) \times
                \text{kernel_size[0]} - 1 }{\text{stride[0]}} + 1} \right \rfloor \\
                H_{out} = \left \lfloor{\frac{H_{in} + padding[2] + padding[3] - (\text{dilation[1]} - 1) \times
                \text{kernel_size[1]} - 1 }{\text{stride[1]}} + 1} \right \rfloor \\
                W_{out} = \left \lfloor{\frac{W_{in} + padding[4] + padding[5] - (\text{dilation[2]} - 1) \times
                \text{kernel_size[2]} - 1 }{\text{stride[2]}} + 1} \right \rfloor \\
            \end{array}

    Raises:
        TypeError: If `in_channels`, `out_channels` or `group` is not an int.
        TypeError: If `kernel_size`, `stride`, `padding` or `dilation` is neither an int nor a tuple.
        ValueError: If `out_channels`, `kernel_size`, `stride` or `dilation` is less than 1.
        ValueError: If `padding` is less than 0.
        ValueError: If `pad_mode` is not one of 'same', 'valid', 'pad'.
        ValueError: If `padding` is a tuple whose length is not equal to 6.
        ValueError: If `pad_mode` is not equal to 'pad' and `padding` is not equal to (0, 0, 0, 0, 0, 0).
        ValueError: If `data_format` is not 'NCDHW'.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> from mindspore import Tensor, nn
        >>> import numpy as np
        >>> x = Tensor(np.ones([16, 3, 10, 32, 32]), mindspore.float32)
        >>> conv3d = nn.Conv3d(in_channels=3, out_channels=32, kernel_size=(4, 3, 3))
        >>> output = conv3d(x)
        >>> print(output.shape)
        (16, 32, 10, 32, 32)
    """

    @cell_attr_register
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
                 data_format='NCDHW',
                 dtype=mstype.float32):
        """Initialize Conv3d."""
        if not (in_channels % group == 0 and out_channels % group == 0):
            raise ValueError("The argument 'group' should be divisible by 'in_channels' " \
                             "and 'out_channels'")

        kernel_size = _check_3d_int_or_tuple("kernel_size", kernel_size, self.cls_name)
        stride = _check_3d_int_or_tuple("stride", stride, self.cls_name)
        dilation = _check_3d_int_or_tuple("dilation", dilation, self.cls_name)
        Validator.check_value_type('padding', padding, (int, tuple), self.cls_name)
        if isinstance(padding, tuple):
            Validator.check_equal_int(len(padding), 6, 'padding size', self.cls_name)
        super(Conv3d, self).__init__(
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
            bias_init,
            data_format,
            dtype=dtype)
        out_channels = self.out_channels
        self.conv3d = P.Conv3D(out_channel=out_channels,
                               kernel_size=self.kernel_size,
                               mode=1,
                               pad_mode=self.pad_mode,
                               pad=self.padding,
                               stride=self.stride,
                               dilation=self.dilation,
                               group=group,
                               data_format=self.data_format)
        self.bias_add = P.BiasAdd(data_format=self.data_format)
        self.shape = P.Shape()

    def construct(self, x):
        x_shape = self.shape(x)
        _check_input_5dims(x_shape, self.cls_name)
        out = self.conv3d(x, self.weight)
        if self.has_bias:
            out = self.bias_add(out, self.bias)
        return out


class Conv3dTranspose(_Conv):
    r"""
    Calculates a 3D transposed convolution, which can be regarded as Conv3d for the gradient of the input.
    It also called deconvolution (although it is not an actual deconvolution).

    The input is typically of shape :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})`,
    where :math:`N` is batch size, :math:`C_{in}` is a number of
    channels, :math:`D_{in}, H_{in}, W_{in}` are the depth, height and width of the feature layer respectively.

    When Conv3d and Conv3dTranspose are initialized with the same parameters, and `pad_mode` is set to 'pad',
    :math:`dilation * (kernel\_size - 1) - padding` amount of zero will be paded to the depth, height and width
    directions of the input, they are inverses of each other in regard to the input and output shapes in this case.
    However, when `stride` > 1, Conv2d maps multiple input shapes to the same output shape. Deconvolutional network
    can refer to `Deconvolutional Networks <https://www.matthewzeiler.com/mattzeiler/deconvolutionalnetworks.pdf>`_.

    Args:
        in_channels (int): The channel number of the input tensor of the Conv3dTranspose layer.
        out_channels (int): The channel number of the output tensor of the Conv3dTranspose layer.
        kernel_size (Union[int, tuple[int]]): Specifies the depth, height and width of the 3D convolution kernel.
            The data type is an integer or a tuple of three integers. An integer represents the depth, height
            and width of the convolution kernel. A tuple of three integers represents the depth, height
            and width of the convolution kernel respectively.
        stride (Union[int, tuple[int]]): The movement stride of the 3D convolution kernel.
            The data type is an integer or a tuple of three integers. An integer represents the movement step size
            in depth, height and width directions. A tuple of three integers represents the movement step size
            in the depth, height and width directions respectively. Default: ``1`` .
        pad_mode (str, optional): Specifies the padding mode with a padding value of 0. It can be set to:
            ``"same"`` , ``"valid"`` or ``"pad"`` . Default: ``"same"`` .

            - ``"same"``: Pad the input around its depth/height/width dimension so that the shape of input and output
              are the same when `stride` is set to ``1``.
              The amount of padding to is calculated by the operator internally.  If the amount is even,
              it isuniformly distributed around the input, if it is odd, the excess amount goes
              to the front/right/bottom side.
              If this mode is set, `padding` must be 0.
            - ``"valid"``: No padding is applied to the input, and the output returns the maximum
              possible depth, height and width. Extra pixels that could not complete a full stride will
              be discarded. If this mode is set, `padding` must be 0.
            - ``"pad"``: Pad the input with a specified amount. In this mode, the amount of padding
              in the depth, height and width dimension is determined by the `padding` parameter.
              If this mode is set, `padding` must be greater than or equal to 0.

        padding (Union(int, tuple[int])): The number of padding on the depth, height and width directions of the input.
            The data type is an integer or a tuple of six integers. If `padding` is an integer,
            then the head, tail, top, bottom, left, and right padding are all equal to `padding`.
            If `padding` is a tuple of six integers, then the head, tail, top, bottom, left, and right padding
            is equal to `padding[0]`, `padding[1]`, `padding[2]`, `padding[3]`, `padding[4]` and `padding[5]`
            respectively. The value should be greater than or equal to 0. Default: ``0`` .
        dilation (Union[int, tuple[int]]): Dilation size of 3D convolution kernel.
            The data type is an integer or a tuple of three integers. If :math:`k > 1`, the kernel is sampled
            every `k` elements. The value of `k` on the depth, height and width directions is in range of
            [1, D], [1, H] and [1, W] respectively. Default: ``1`` .
        group (int): Splits filter into groups, `in_channels` and `out_channels` must be
            divisible by `group`. Default: ``1`` .
        output_padding (Union(int, tuple[int])): The number of padding on the depth, height and width directions of
            the output. The data type is an integer or a tuple of six integers. If `output_padding` is an integer,
            then the head, tail, top, bottom, left, and right padding are all equal to `output_padding`.
            If `output_padding` is a tuple of six integers, then the head, tail, top, bottom, left, and right padding
            is equal to `output_padding[0]`, `output_padding[1]`, `output_padding[2]`, `output_padding[3]`,
            `output_padding[4]` and `output_padding[5]` respectively. The value should be greater than or equal to 0.
            Default: ``0`` .
        has_bias (bool): Whether the Conv3dTranspose layer has a bias parameter. Default: ``False`` .
        weight_init (Union[Tensor, str, Initializer, numbers.Number]): Initialization method of weight parameter.
            It can be a Tensor, a string, an Initializer or a numbers.Number. When a string is specified,
            values from ``'TruncatedNormal'`` , ``'Normal'`` , ``'Uniform'`` , ``'HeUniform'`` and ``'XavierUniform'``
            distributions as well as constant ``'One'`` and ``'Zero'`` distributions are possible. Alias
            ``'xavier_uniform'`` , ``'he_uniform'`` , ``'ones'`` and ``'zeros'`` are acceptable. Uppercase and
            lowercase are both acceptable. Refer to the values of Initializer for more details. Default: ``None`` ,
            weight will be initialized using HeUniform.
        bias_init (Union[Tensor, str, Initializer, numbers.Number]): Initialization method of bias parameter.
            Available initialization methods are the same as 'weight_init'. Refer to the values of
            Initializer for more details. Default: ``None`` , bias will be initialized using Uniform.
        data_format (str): The optional value for data format. Currently only support ``'NCDHW'`` .
            Default: ``'NCDHW'`` .
        dtype (:class:`mindspore.dtype`): Dtype of Parameters. Default: ``mstype.float32`` .

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})`.
          Currently input data dtype only support float16 and float32.

    Outputs:
        Tensor, the shape is :math:`(N, C_{out}, D_{out}, H_{out}, W_{out})`.

        pad_mode is ``'same'`` :

        .. math::
            \begin{array}{ll} \\
                D_{out} = \left \lfloor{\frac{D_{in}}{\text{stride[0]}} + 1} \right \rfloor \\
                H_{out} = \left \lfloor{\frac{H_{in}}{\text{stride[1]}} + 1} \right \rfloor \\
                W_{out} = \left \lfloor{\frac{W_{in}}{\text{stride[2]}} + 1} \right \rfloor \\
            \end{array}


        pad_mode is ``'valid'`` :

        .. math::
            \begin{array}{ll} \\
                D_{out} = \left \lfloor{\frac{D_{in} - \text{dilation[0]} \times (\text{kernel_size[0]} - 1) }
                {\text{stride[0]}} + 1} \right \rfloor \\
                H_{out} = \left \lfloor{\frac{H_{in} - \text{dilation[1]} \times (\text{kernel_size[1]} - 1) }
                {\text{stride[1]}} + 1} \right \rfloor \\
                W_{out} = \left \lfloor{\frac{W_{in} - \text{dilation[2]} \times (\text{kernel_size[2]} - 1) }
                {\text{stride[2]}} + 1} \right \rfloor \\
            \end{array}

        pad_mode is ``'pad'`` :

        .. math::
            \begin{array}{ll} \\
                D_{out} = \left \lfloor{\frac{D_{in} + padding[0] + padding[1] - (\text{dilation[0]} - 1) \times
                \text{kernel_size[0]} - 1 }{\text{stride[0]}} + 1} \right \rfloor \\
                H_{out} = \left \lfloor{\frac{H_{in} + padding[2] + padding[3] - (\text{dilation[1]} - 1) \times
                \text{kernel_size[1]} - 1 }{\text{stride[1]}} + 1} \right \rfloor \\
                W_{out} = \left \lfloor{\frac{W_{in} + padding[4] + padding[5] - (\text{dilation[2]} - 1) \times
                \text{kernel_size[2]} - 1 }{\text{stride[2]}} + 1} \right \rfloor \\
            \end{array}

    Raises:
        TypeError: If `in_channels`, `out_channels` or `group` is not an int.
        TypeError: If `kernel_size`, `stride`, `padding` , `dilation` or `output_padding`
                   is neither an int not a tuple of three.
        TypeError: If input data type is not float16 or float32.
        ValueError: If `in_channels`, `out_channels`, `kernel_size`, `stride` or `dilation` is less than 1.
        ValueError: If `padding` is less than 0.
        ValueError: If `pad_mode` is not one of 'same', 'valid', 'pad'.
        ValueError: If `padding` is a tuple whose length is not equal to 6.
        ValueError: If `pad_mode` is not equal to 'pad' and `padding` is not equal to (0, 0, 0, 0, 0, 0).
        ValueError: If `data_format` is not 'NCDHW'.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> from mindspore import Tensor, nn
        >>> import numpy as np
        >>> x = Tensor(np.ones([32, 16, 10, 32, 32]), mindspore.float32)
        >>> conv3d_transpose = nn.Conv3dTranspose(in_channels=16, out_channels=3, kernel_size=(4, 6, 2),
        ...                                       pad_mode='pad')
        >>> output = conv3d_transpose(x)
        >>> print(output.shape)
        (32, 3, 13, 37, 33)
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 pad_mode="same",
                 padding=0,
                 dilation=1,
                 group=1,
                 output_padding=0,
                 has_bias=False,
                 weight_init=None,
                 bias_init=None,
                 data_format='NCDHW',
                 dtype=mstype.float32):
        """Initialize Conv3dTranspose."""
        if not (in_channels % group == 0 and out_channels % group == 0):
            raise ValueError("The argument 'group' should be divisible by 'in_channels' " \
                             "and 'out_channels'")

        kernel_size = _check_3d_int_or_tuple("kernel_size", kernel_size, self.cls_name)
        stride = _check_3d_int_or_tuple("stride", stride, self.cls_name)
        dilation = _check_3d_int_or_tuple("dilation", dilation, self.cls_name)
        Validator.check_value_type('padding', padding, (int, tuple), self.cls_name)
        if isinstance(padding, tuple):
            Validator.check_equal_int(len(padding), 6, 'padding size', self.cls_name)
        self.output_padding = _check_3d_int_or_tuple("output_padding", output_padding, self.cls_name,
                                                     greater_zero=False)
        super(Conv3dTranspose, self).__init__(
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
            bias_init,
            data_format,
            transposed=True,
            dtype=dtype)
        self.conv3d_transpose = P.Conv3DTranspose(in_channel=self.in_channels,
                                                  out_channel=self.out_channels,
                                                  kernel_size=self.kernel_size,
                                                  mode=1,
                                                  pad_mode=self.pad_mode,
                                                  pad=self.padding,
                                                  stride=self.stride,
                                                  dilation=self.dilation,
                                                  group=self.group,
                                                  output_padding=self.output_padding,
                                                  data_format=self.data_format)
        self.bias_add = P.BiasAdd(data_format=self.data_format)
        self.shape = P.Shape()

    def construct(self, x):
        x_shape = self.shape(x)
        _check_input_5dims(x_shape, self.cls_name)
        output = self.conv3d_transpose(x, self.weight)
        if self.has_bias:
            output = self.bias_add(output, self.bias)
        return output


def _deconv_output_length(is_valid, is_same, is_pad, input_length, filter_size, stride_size, dilation_size, padding):
    """Calculate the width and height of output."""
    length = 0
    filter_size = filter_size + (filter_size - 1) * (dilation_size - 1)
    if is_valid:
        if filter_size - stride_size > 0:
            length = input_length * stride_size + filter_size - stride_size
        else:
            length = input_length * stride_size
    elif is_same:
        length = input_length * stride_size
    elif is_pad:
        length = input_length * stride_size - padding + filter_size - stride_size

    return length


class Conv2dTranspose(_Conv):
    r"""
    Calculates a 2D transposed convolution, which can be regarded as Conv2d for the gradient of the input,
    also called deconvolution (although it is not an actual deconvolution).

    The input is typically of shape :math:`(N, C_{in}, H_{in}, W_{in})`,
    where :math:`N` is batch size, :math:`C_{in}` is space dimension,
    :math:`H_{in}, W_{in}` are the height and width of the feature layer respectively.

    When Conv2d and Conv2dTranspose are initialized with the same parameters, and `pad_mode` is set to 'pad',
    :math:`dilation * (kernel\_size - 1) - padding` amount of zero will be paded to the height and width
    directions of the input, they are inverses of each other in regard to the input and output shapes in this case.
    However, when `stride` > 1, Conv2d maps multiple input shapes to the same output shape. Deconvolutional network
    can refer to `Deconvolutional Networks <https://www.matthewzeiler.com/mattzeiler/deconvolutionalnetworks.pdf>`_.

    Args:
        in_channels (int): The channel number of the input tensor of the Conv2dTranspose layer.
        out_channels (int): The channel number of the output tensor of the Conv2dTranspose layer.
        kernel_size (Union[int, tuple[int]]): Specifies the height and width of the 2D convolution kernel.
            The data type is an integer or a tuple of two integers. An integer represents the height
            and width of the convolution kernel. A tuple of two integers represents the height
            and width of the convolution kernel respectively.
        stride (Union[int, tuple[int]]): The movement stride of the 2D convolution kernel.
            The data type is an integer or a tuple of two integers. An integer represents the movement step size
            in both height and width directions. A tuple of two integers represents the movement step size in the height
            and width directions respectively. Default: ``1`` .
        pad_mode (str, optional): Specifies the padding mode with a padding value of 0. It can be set to:
            ``"same"`` , ``"valid"`` or ``"pad"`` . Default: ``"same"`` .

            - ``"same"``: Pad the input around its edges so that the shape of input and output
              are the same when `stride` is set to ``1``.
              The amount of padding to is calculated by the operator internally, If the amount is even, it is
              uniformly distributed around the input, if it is odd, the excess amount goes to the right/bottom side.
              If this mode is set, `padding` must be 0.
            - ``"valid"``: No padding is applied to the input, and the output returns the maximum
              possible height and width. Extra pixels that could not complete a full stride will
              be discarded. If this mode is set, `padding` must be 0.
            - ``"pad"``: Pad the input with a specified amount. In this mode, the amount of padding
              in the height and width directions is determined by the `padding` parameter.
              If this mode is set, `padding` must be greater than or equal to 0.

        padding (Union[int, tuple[int]]): The number of padding on the height and width directions of the input.
            The data type is an integer or a tuple of four integers. If `padding` is an integer,
            then the top, bottom, left, and right padding are all equal to `padding`.
            If `padding` is a tuple of 4 integers, then the top, bottom, left, and right padding
            is equal to `padding[0]`, `padding[1]`, `padding[2]`, and `padding[3]` respectively.
            The value should be greater than or equal to 0. Default: ``0`` .
        output_padding (Union[int, tuple[int]]): The number of padding on the height and width directions of the output.
            The data type is an integer or a tuple of two integers. If `output_padding` is an integer,
            then the bottom and right padding are all equal to `output_padding`. If `output_padding` is a tuple of
            2 integers, then the bottom and right padding is equal to `output_padding[0]`, `output_padding[1]`
            respectively. If `output_padding` is not equal to 0, `pad_mode` must be `pad`.
            The value should be in range of `[0, max(stride, dilation))` . Default: ``0`` .
        dilation (Union[int, tuple[int]]): Dilation size of 2D convolution kernel.
            The data type is an integer or a tuple of two integers. If :math:`k > 1`, the kernel is sampled
            every `k` elements. The value of `k` on the height and width directions is in range of [1, H]
            and [1, W] respectively. Default: ``1`` .
        group (int): Splits filter into groups, `in_channels` and `out_channels` must be divisible by `group`.
            Default: ``1`` .
        has_bias (bool): Whether the Conv2dTranspose layer has a bias parameter. Default: ``False`` .
        weight_init (Union[Tensor, str, Initializer, numbers.Number]): Initialization method of weight parameter.
            It can be a Tensor, a string, an Initializer or a numbers.Number. When a string is specified,
            values from ``'TruncatedNormal'`` , ``'Normal'`` , ``'Uniform'`` , ``'HeUniform'`` and ``'XavierUniform'``
            distributions as well as constant ``'One'`` and ``'Zero'`` distributions are possible. Alias
            ``'xavier_uniform'`` , ``'he_uniform'`` , ``'ones'`` and ``'zeros'`` are acceptable. Uppercase and
            lowercase are both acceptable. Refer to the values of Initializer for more details. Default: ``None`` ,
            weight will be initialized using HeUniform.
        bias_init (Union[Tensor, str, Initializer, numbers.Number]): Initialization method of bias parameter.
            Available initialization methods are the same as 'weight_init'. Refer to the values of
            Initializer for more details. Default: ``None`` , bias will be initialized using Uniform.
        dtype (:class:`mindspore.dtype`): Dtype of Parameters. Default: ``mstype.float32`` .

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor of shape :math:`(N, C_{out}, H_{out}, W_{out})`.

        pad_mode is ``'same'``:

        .. math::
            \begin{array}{ll} \\
                H_{out} = \text H_{in}\times \text {stride[0]} \\
                W_{out} = \text W_{in}\times \text {stride[1]} \\
            \end{array}

        pad_mode is ``'valid'``:

        .. math::
            \begin{array}{ll} \\
                H_{out} = \text H_{in}\times \text {stride[0]} + \max\{(\text{dilation[0]} - 1) \times
                (\text{kernel_size[0]} - 1) - \text {stride[0]}, 0 \} \\
                W_{out} = \text W_{in}\times \text {stride[1]} + \max\{(\text{dilation[1]} - 1) \times
                (\text{kernel_size[1]} - 1) - \text {stride[1]}, 0 \} \\
            \end{array}

        pad_mode is ``'pad'``:

        .. math::
            \begin{array}{ll} \\
                H_{out} = \text H_{in}\times \text {stride[0]} - (padding[0] + padding[1])
                + \text{kernel_size[0]} + (\text{dilation[0]} - 1) \times
                (\text{kernel_size[0]} - 1) - \text {stride[0]} + \text {output_padding[0]} \\
                W_{out} = \text W_{in}\times \text {stride[1]} - (padding[2] + padding[3])
                + \text{kernel_size[1]} + (\text{dilation[1]} - 1) \times
                (\text{kernel_size[1]} - 1) - \text {stride[1]} + \text {output_padding[1]} \\
            \end{array}

    Raises:
        TypeError: If `in_channels`, `out_channels` or `group` is not an int.
        TypeError: If `kernel_size`, `stride`, `padding` or `dilation` is neither an int not a tuple.
        ValueError: If `in_channels`, `out_channels`, `kernel_size`, `stride` or `dilation` is less than 1.
        ValueError: If `padding` is less than 0.
        ValueError: If `pad_mode` is not one of 'same', 'valid', 'pad'.
        ValueError: If `padding` is a tuple whose length is not equal to 4.
        ValueError: If `pad_mode` is not equal to 'pad' and `padding` is not equal to (0, 0, 0, 0).

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> from mindspore import Tensor, nn
        >>> import numpy as np
        >>> net = nn.Conv2dTranspose(3, 64, 4, has_bias=False, weight_init='normal', pad_mode='pad')
        >>> x = Tensor(np.ones([1, 3, 16, 50]), mindspore.float32)
        >>> output = net(x).shape
        >>> print(output)
        (1, 64, 19, 53)
        """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 pad_mode='same',
                 padding=0,
                 output_padding=0,
                 dilation=1,
                 group=1,
                 has_bias=False,
                 weight_init=None,
                 bias_init=None,
                 dtype=mstype.float32):
        """Initialize Conv2dTranspose."""
        kernel_size = twice(kernel_size)
        stride = twice(stride)
        dilation = twice(dilation)
        Validator.check_value_type('padding', padding, (int, tuple), self.cls_name)
        if isinstance(padding, tuple):
            Validator.check_equal_int(len(padding), 4, 'padding size', self.cls_name)
        Validator.check_value_type('output_padding', output_padding, (int, tuple), self.cls_name)
        if isinstance(output_padding, tuple):
            Validator.check_equal_int(len(output_padding), 2, 'output_padding size', self.cls_name)
        # out_channels and in_channels swap.
        # cause Conv2DBackpropInput's out_channel refers to Conv2D's out_channel,
        # then Conv2dTranspose's out_channel refers to Conv2DBackpropInput's in_channel.
        super(Conv2dTranspose, self).__init__(
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
            bias_init,
            transposed=True,
            dtype=dtype)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.shape = P.Shape()
        Validator.check_string(pad_mode, ['valid', 'same', 'pad'], 'pad_mode', self.cls_name)
        self.is_valid = self.pad_mode == 'valid'
        self.is_same = self.pad_mode == 'same'
        self.is_pad = self.pad_mode == 'pad'
        self.output_padding = output_padding

        # cause Conv2DTranspose's out_channel refers to Conv2D's out_channel.
        self.conv2d_transpose = P.Conv2DTranspose(out_channel=in_channels,
                                                  kernel_size=kernel_size,
                                                  mode=1,
                                                  pad_mode=pad_mode,
                                                  pad=padding,
                                                  stride=stride,
                                                  dilation=dilation,
                                                  group=group)
        self.bias_add = P.BiasAdd()
        if isinstance(self.padding, int):
            self.padding_top, self.padding_bottom, self.padding_left, self.padding_right = (self.padding,) * 4
        else:
            self.padding_top, self.padding_bottom, self.padding_left, self.padding_right = self.padding

    def shard(self, strategy):
        self.conv2d_transpose.shard(strategy)
        return self

    def construct(self, x):
        n, _, h, w = self.shape(x)
        h_out = _deconv_output_length(self.is_valid, self.is_same, self.is_pad, h, self.kernel_size[0],
                                      self.stride[0], self.dilation[0], self.padding_top + self.padding_bottom)
        w_out = _deconv_output_length(self.is_valid, self.is_same, self.is_pad, w, self.kernel_size[1],
                                      self.stride[1], self.dilation[1], self.padding_left + self.padding_right)
        conv2d_trans_ret = self.conv2d_transpose(x, self.weight, (n, self.out_channels, h_out, w_out))
        if self.has_bias:
            conv2d_trans_ret = self.bias_add(conv2d_trans_ret, self.bias)
        if isinstance(self.output_padding, tuple):
            if self.output_padding[0] < 0 or self.output_padding[0] >= max(self.dilation[0], self.stride[0]):
                raise ValueError("output_padding[0] must be in range of [0, max(stride_h, dilation_h)).")
            if self.output_padding[1] < 0 or self.output_padding[1] >= max(self.dilation[1], self.stride[1]):
                raise ValueError("output_padding[1] must be in range of [0, max(stride_w, dilation_w)).")
            if not self.is_pad and (self.output_padding[0] > 0 or self.output_padding[1] > 0):
                raise ValueError("when output_padding is not zero, pad_mode must be 'pad'")

            pad = P.Pad(paddings=((0, 0), (0, 0), (0, self.output_padding[0]), (0, self.output_padding[1])))
            return pad(conv2d_trans_ret)

        if self.output_padding == 0:
            return conv2d_trans_ret

        if self.output_padding < 0 or self.output_padding >= max(self.dilation[0], self.stride[0]):
            raise ValueError("output_padding must be in range of [0, max(stride_h, dilation_h)).")
        if self.output_padding < 0 or self.output_padding >= max(self.dilation[1], self.stride[1]):
            raise ValueError("output_padding must be in range of [0, max(stride_w, dilation_w)).")
        if not self.is_pad and self.output_padding > 0:
            raise ValueError("when output_padding is not zero, pad_mode must be 'pad'")
        pad = P.Pad(paddings=((0, 0), (0, 0), (0, self.output_padding), (0, self.output_padding)))
        return pad(conv2d_trans_ret)


class Conv1dTranspose(_Conv):
    r"""
    Calculates a 1D transposed convolution, which can be regarded as Conv1d for the gradient of the input,
    also called deconvolution (although it is not an actual deconvolution).

    The input is typically of shape :math:`(N, C_{in}, L_{in})`, where :math:`N` is batch size,
    :math:`C_{in}` is a number of channels
    and :math:`L_{in}` is a length of sequence.

    When Conv1d and ConvTranspose1d are initialized with the same parameters, and `pad_mode` is set to 'pad',
    :math:`dilation * (kernel\_size - 1) - padding` amount of zero will be paded to both sizes of input,
    they are inverses of each other in regard to the input and output shapes in this case.
    However, when `stride` > 1, Conv1d maps multiple input shapes to the same output shape. Deconvolutional network
    can refer to `Deconvolutional Networks <https://www.matthewzeiler.com/mattzeiler/deconvolutionalnetworks.pdf>`_.

    Args:
        in_channels (int): The channel number of the input tensor of the Conv1dTranspose layer.
        out_channels (int): The channel number of the output tensor of the Conv1dTranspose layer.
        kernel_size (int): Specifies the width of the 1D convolution kernel.
        stride (int): The movement stride of the 1D convolution kernel. Default: ``1`` .
        pad_mode (str, optional): Specifies the padding mode with a padding value of 0. It can be set to:
            ``"same"`` , ``"valid"`` or ``"pad"`` . Default: ``"same"`` .

            - ``"same"``: Pad the input at the begin and end so that the shape of input and output
              are the same when `stride` is set to ``1``.
              The amount of padding to is calculated by the operator internally. If the amount is even, it is
              uniformly distributed around the input, if it is odd, the excess padding is goes to the right side.
              If this mode is set, `padding` must be 0.
            - ``"valid"``: No padding is applied to the input, and the output returns the maximum
              possible length. Extra pixels that could not complete a full stride will
              be discarded. If this mode is set, `padding` must be 0.
            - ``"pad"``: Pad the input with a specified amount. In this mode, the amount of padding
              at the begin and end is determined by the `padding` parameter.
              If this mode is set, `padding` must be greater than or equal to 0.

        padding (int): The number of padding on both sides of input.
            The value should be greater than or equal to 0. Default: ``0`` .
        dilation (int): Dilation size of 1D convolution kernel. If :math:`k > 1`, the kernel is sampled
            every `k` elements. The value of `k` is in range of [1, L]. Default: ``1`` .
        group (int): Splits filter into groups, `in_channels` and `out_channels` must be
            divisible by `group`. When `group` > 1, the Ascend platform is not supported yet. Default: ``1`` .
        has_bias (bool): Whether the Conv1dTranspose layer has a bias parameter. Default: ``False``.
        weight_init (Union[Tensor, str, Initializer, numbers.Number]): Initialization method of weight parameter.
            It can be a Tensor, a string, an Initializer or a numbers.Number. When a string is specified,
            values from ``'TruncatedNormal'`` , ``'Normal'`` , ``'Uniform'`` , ``'HeUniform'`` and ``'XavierUniform'``
            distributions as well as constant ``'One'`` and ``'Zero'`` distributions are possible. Alias
            ``'xavier_uniform'`` , ``'he_uniform'``, ``'ones'`` and ``'zeros'`` are acceptable. Uppercase and lowercase
            are both acceptable. Refer to the values of Initializer for more details. Default: ``None`` ,
            weight will be initialized using HeUniform.
        bias_init (Union[Tensor, str, Initializer, numbers.Number]): Initialization method of bias parameter.
            Available initialization methods are the same as 'weight_init'. Refer to the values of
            Initializer for more details. Default: ``None`` , bias will be initialized using Uniform.
        dtype (:class:`mindspore.dtype`): Dtype of Parameters. Default: ``mstype.float32`` .

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, L_{in})`.

    Outputs:
        Tensor of shape :math:`(N, C_{out}, L_{out})`.

        pad_mode is ``'same'``: :math:`L_{out} = \frac{ L_{in} + \text{stride} - 1 }{ \text{stride} }`

        pad_mode is ``'valid'``:
        :math:`L_{out} = (L_{in} - 1) \times \text{stride} + \text{dilation} \times (\text{kernel_size} - 1) + 1`

        pad_mode is ``'pad'``:
        :math:`L_{out} = (L_{in} - 1) \times \text{stride} - 2 \times \text{padding}
        + \text{dilation} \times (\text{kernel_size} - 1) + 1`

    Raises:
        TypeError: If `in_channels`, `out_channels`, `kernel_size`, `stride`, `padding` or `dilation` is not an int.
        ValueError: If `in_channels`, `out_channels`, `kernel_size`, `stride` or `dilation` is less than 1.
        ValueError: If `padding` is less than 0.
        ValueError: If `pad_mode` is not one of 'same', 'valid', 'pad'.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> from mindspore import Tensor, nn
        >>> import numpy as np
        >>> net = nn.Conv1dTranspose(3, 64, 4, has_bias=False, weight_init='normal', pad_mode='pad')
        >>> x = Tensor(np.ones([1, 3, 50]), mindspore.float32)
        >>> output = net(x).shape
        >>> print(output)
        (1, 64, 53)
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
                 dtype=mstype.float32):
        """Initialize Conv1dTranspose."""
        Validator.check_value_type("kernel_size", kernel_size, [int], self.cls_name)
        Validator.check_value_type("stride", stride, [int], self.cls_name)
        Validator.check_value_type("padding", padding, [int], self.cls_name)
        Validator.check_value_type("dilation", dilation, [int], self.cls_name)
        Validator.check_int(kernel_size, 1, Validator.GE, 'kernel_size', self.cls_name)
        Validator.check_int(stride, 1, Validator.GE, 'stride', self.cls_name)
        Validator.check_non_negative_int(padding, 'padding', self.cls_name)
        Validator.check_int(dilation, 1, Validator.GE, 'dilation', self.cls_name)
        kernel_size = (1, kernel_size)
        stride = (1, stride)
        dilation = (1, dilation)
        get_shape = P.Shape()
        get_dtype = P.DType()
        if isinstance(weight_init, Tensor):
            weight_init_shape = get_shape(weight_init)
            Validator.check_equal_int(len(weight_init_shape), 3, 'weight_init_shape', self.cls_name)
            weight_init_dtype = get_dtype(weight_init)
            weight_init_value = weight_init.asnumpy()
            weight_init_value = np.expand_dims(weight_init_value, 2)
            weight_init = Tensor(weight_init_value, weight_init_dtype)
        # out_channels and in_channels swap.
        # cause Conv2DBackpropInput's out_channel refers to Conv2D's out_channel,
        # then Conv1dTranspose's out_channel refers to Conv2DBackpropInput's in_channel.
        super(Conv1dTranspose, self).__init__(
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
            bias_init,
            transposed=True,
            dtype=dtype)
        self.padding = (0, 0, padding, padding)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.shape = P.Shape()
        Validator.check_string(pad_mode, ['valid', 'same', 'pad'], 'pad_mode', self.cls_name)
        self.is_valid = self.pad_mode == 'valid'
        self.is_same = self.pad_mode == 'same'
        self.is_pad = self.pad_mode == 'pad'

        # cause Conv2DBackpropInput's out_channel refers to Conv2D's out_channel.
        self.conv2d_transpose = P.Conv2DBackpropInput(out_channel=in_channels,
                                                      kernel_size=kernel_size,
                                                      mode=1,
                                                      pad_mode=pad_mode,
                                                      pad=self.padding,
                                                      stride=stride,
                                                      dilation=dilation,
                                                      group=group)
        self.bias_add = P.BiasAdd()
        self.expand_dims = P.ExpandDims()
        self.squeeze = P.Squeeze(2)

    def shard(self, strategy):
        self.conv2d_transpose.shard(strategy)
        return self

    def construct(self, x):
        x_shape = self.shape(x)
        _check_input_3d(x_shape, self.cls_name)
        x = self.expand_dims(x, 2)

        n, _, h, w = self.shape(x)

        h_out = _deconv_output_length(self.is_valid, self.is_same, self.is_pad, h, self.kernel_size[0],
                                      self.stride[0], self.dilation[0], self.padding[0] + self.padding[1])
        w_out = _deconv_output_length(self.is_valid, self.is_same, self.is_pad, w, self.kernel_size[1],
                                      self.stride[1], self.dilation[1], self.padding[2] + self.padding[3])
        output = self.conv2d_transpose(x, self.weight, (n, self.out_channels, h_out, w_out))
        if self.has_bias:
            output = self.bias_add(output, self.bias)

        output = self.squeeze(output)
        return output
