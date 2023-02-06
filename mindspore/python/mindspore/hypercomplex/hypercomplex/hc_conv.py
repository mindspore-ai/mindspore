# Copyright 2023 Huawei Technologies Co., Ltd
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
"""Hypercomplex Convolution"""
import numbers
from typing import Type, TypeVar, Tuple, Union
from abc import abstractmethod

import numpy as np
from mindspore._checkparam import Validator, Rel, twice, _check_3d_int_or_tuple
from mindspore import context
from mindspore import log as logger
from mindspore.common.initializer import initializer, Initializer
from mindspore.common.parameter import Parameter
from mindspore.common.tensor import Tensor
from mindspore.nn.cell import Cell
from mindspore.ops import operations as P
from mindspore.hypercomplex.hypercomplex._hc_conv_impl import _ConvImpl as ConvImpl
from mindspore.hypercomplex.utils import get_x_and_y, to_2channel, \
                                _size_1_t, _size_2_t, _size_3_t


TConvImpl = TypeVar('TConvImpl', bound=ConvImpl)


class _ConvNd(Cell):
    r"""
    The base class of the abstraction part of Convolution layer of the second-order hypercomplex input.

    Calculates the convolution on the input tensor which is typically of shape :math:`(2, N, C_{in}, *, ..., *)`,
    where :math:`N` is batch size, :math:`C_{in}` is a number of input channels, and the count of spatial
    dimensions denoted by '*' is defined by the specific subclass. The formula is defined as:

    .. math::

        \text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) +
        \sum_{k = 0}^{C_{in} - 1} \text{ccor}({\text{weight}(C_{\text{out}_j}, k), \text{inp}(N_i, k)})

    where :math:`ccor` is the `cross-correlation <https://en.wikipedia.org/wiki/Cross-correlation>`_, the exact
    implementation of which is defined and provided by the implementor part of the convolution layer,
    :math:`C_{in}` is the channel number of the input, :math:`out_{j}` corresponds to the jth channel of
    the output and :math:`j` is in the range of :math:`[0, C_{out}-1]`. :math:`\text{weight}(C_{\text{out}_j}, k)`
    is a convolution kernel slice with shape :math:`(\text{kernel_size[0]}, \text{kernel_size[1]})`,
    where :math:`\text{kernel_size[0]}` and :math:`\text{kernel_size[1]}` are the height and width of the convolution
    kernel respectively. :math:`\text{bias}` is the bias parameter and :math:`\text{inp}` is the input tensor.
    In this case, `data_format` of the input tensor is 'NCHW' and the shape of full convolution kernel is
    :math:`(2, C_{out}, C_{in} / \text{group}, *, ..., *)`, where `group` is the number of groups to split
    the input `inp` in the channel dimension, and the '*' symbols denote the corresponding kernel dimensions.
    If `data_format` of the input tensor is 'NHWC', the shape of full convolution kernel will be
    :math:`(2, C_{out}, *, ..., *, C_{in} / \text{group}`.

    For more details, please refers to the paper `Gradient Based Learning Applied to Document
    Recognition <http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf>`_.

    Note:
        On Ascend platform, only group convolution in depthwise convolution scenarios is supported.
        That is, when `group > 1`, condition `in\_channels` = `out\_channels` = `group` must be satisfied.

        This is not a self-sufficient class. In order to construct a convolution layer, one should instantiate this
        class and an implementor class, which acts like a bridge pattern and determine the exact set of hypercomplex
        numbers. That implies the rules of multiplication and therefore affects how a convolution works.

        'NCHW' format is supported only with GPU target device as of now.

    Args:
        conv_impl (TConvImpl): The implementor object of the convolution layer. Essentially, the concrete class name
            of this argument defines the algebra that the convolution layer will operate on.
        in_channels (int): The channel number of the input tensor of the convolution layer.
        out_channels (int): The channel number of the output tensor of the convolution layer.
        kernel_size (Union[int, tuple[int]]): Specifies the spatial dimensions of the convolution kernel.
            The data type is an integer or a tuple of integers. An integer represents the size of all the
            spatial dimensions of the convolution kernel at once. A tuple of integers represents the spatial
            dimensions of the convolution kernel individually.
        stride (Union[int, tuple[int]]): The movement stride of the convolution kernel.
            The data type is an integer or a tuple of integers. An integer represents the movement step size
            in all directions at once. A tuple of integers represents the movement step size in every direction
            individually.
        pad_mode (str): Specifies padding mode. The optional values are "same", "valid", "pad".

            - same: The width of the output is the same as the value of the input divided by `stride`.
              If this mode is set, the value of `padding` must be 0.

            - valid: Returns a valid calculated output without padding. Excess pixels that do not satisfy the
              calculation will be discarded. If this mode is set, the value of `padding` must be 0.

            - pad: Pads the input. Padding `padding` size of zero on both sides of the input.
              If this mode is set, the value of `padding` must be greater than or equal to 0.

        padding (Union[int, tuple[int]]): The number of padding on the spatial dimensions of the input.
            The data type is an integer or a tuple of integers, which then must be twice as long as the number
            of spatial dimensions. If `padding` is an integer, then all the leading and trailing paddings in
            all dimensions are equal to `padding`. The value should be greater than or equal to 0.
            If `padding` is a tuple of integers, then the paddings are enumerated pair-wise from the first to
            the last spatial dimension, the first element of  the pair being equal to the leading padding,
            and the second element of the pair being equal to the trailing padding of the corresponding
            spatial dimension.
        dilation (Union[int, tuple[int]]): Dilation size of convolution kernel.
            The data type is an integer or a tuple of integers. If :math:`k > 1`, the kernel is sampled
            every `k` elements.
        group (int): Splits filter into groups, `in_channels` and `out_channels` must be
            divisible by `group`. If the group is equal to `in_channels` and `out_channels`,
            this convolution layer also can be called depthwise convolution layer.
        has_bias (bool): Whether the convolution layer has a bias parameter.
        weight_init (Union[Tensor, str, Initializer, numbers.Number]): Initialization method of weight parameter.
            It can be a Tensor, a string, an Initializer or a numbers.Number. When a string is specified,
            values from 'TruncatedNormal', 'Normal', 'Uniform', 'HeUniform' and 'XavierUniform' distributions as well
            as constant 'One' and 'Zero' distributions are possible. Alias 'xavier_uniform', 'he_uniform', 'ones'
            and 'zeros' are acceptable. Uppercase and lowercase are both acceptable. Refer to the values of
            Initializer for more details.
        bias_init (Union[Tensor, str, Initializer, numbers.Number]): Initialization method of bias parameter.
            Available initialization methods are the same as 'weight_init'. Refer to the values of
            Initializer for more details.
        data_format (str): The optional value for data format, is 'NHWC' or 'NCHW' or 'NCDHW'.

    Inputs:
        - **inp** (Tensor) - Tensor of shape :math:`(2, N, C_{in}, *, ..., *)` or :math:`(2, N, *, ..., *, C_{in})`,
            with float16 or float32 data type, or :math:`(N, C_{in}, *, ..., *)` or :math:`(N, *, ..., *, C_{in})`
            with complex64 data type.

    Outputs:
        Tensor of the same data type as `inp` and of shape :math:`(2, N, C_{out}, *, ..., *)` or
        :math:`(2, N, *, ..., *, C_{out})`, with float16 or float32 data type, or :math:`(N, C_{out}, *, ..., *)` or
        :math:`(N, *, ..., *, C_{out})`, with complex64 data type.

    Raises:
        TypeError: If `in_channels`, `out_channels` or `group` is not an int.
        TypeError: If `kernel_size`, `stride`, `padding` or `dilation` is neither an int not a tuple.
        TypeError: If any two of `inp`, `weight_init` and `bias_init` are Tensors of different data type.
        ValueError: If `in_channels`, `out_channels`, `kernel_size`, `stride` or `dilation` is less than 1.
        ValueError: If `padding` is less than 0.
        ValueError: If `pad_mode` is not one of 'same', 'valid', 'pad'.
        ValueError: If `padding` is a tuple whose length is not twice as big as the count of spatial dimensions.
        ValueError: If `pad_mode` is not equal to 'pad' and `padding` is a tuple which contains non-zero elements.
        ValueError: If `data_format` is neither 'NCHW', 'NHWC', nor 'NCDHW', or it is 'NCHW' and the target
            device is not GPU.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    def __init__(self,
                 conv_impl: Type[TConvImpl],
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Tuple[int, ...],
                 stride: Tuple[int, ...],
                 pad_mode: str,
                 padding: Tuple[int, ...],
                 dilation: Tuple[int, ...],
                 group: int,
                 has_bias: bool,
                 weight_init: Union[Tensor, str, Initializer, numbers.Number],
                 bias_init: Union[Tensor, str, Initializer, numbers.Number],
                 data_format: str = 'NCHW',
                 transposed: bool = False) -> None:
        """Initialize _ConvNd."""
        super(_ConvNd, self).__init__()

        self.in_channels = Validator.check_positive_int(in_channels, 'in_channels', self.cls_name)
        self.out_channels = Validator.check_positive_int(out_channels, 'out_channels', self.cls_name)
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad_mode = pad_mode
        self.weight_init = weight_init
        self.bias_init = bias_init
        self.data_format = Validator.check_string(data_format,
                                                  ['NCHW', 'NHWC', 'NCDHW'],
                                                  'format',
                                                  self.cls_name)
        if context.get_context("device_target") != "GPU" and self.data_format == "NHWC":
            raise ValueError("NHWC format only support in GPU target.")
        if isinstance(padding, int):
            Validator.check_non_negative_int(padding, 'padding', self.cls_name)
            self.padding = padding
        elif isinstance(padding, tuple):
            for pad in padding:
                Validator.check_non_negative_int(pad, 'padding item', self.cls_name)
            self.padding = padding
        else:
            raise TypeError("padding type must be int/tuple(int) cannot be {}!".format(type(padding)))

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
            raise ValueError(f"Attr 'in_channels' of {self.cls_name} Op must be divisible by "
                             f"attr 'group' of {self.cls_name} Op.")
        if out_channels % group != 0:
            raise ValueError(f"Attr 'out_channels' {self.cls_name} Op must be divisible by "
                             f"attr 'group' of {self.cls_name} Op.")
        if transposed:
            shape = [in_channels, out_channels // group, *kernel_size]
        else:
            shape = [out_channels, *kernel_size, in_channels // group] if self.data_format == "NHWC" else \
                [out_channels, in_channels // group, *kernel_size]
        self.dtype = self.weight_init.dtype if isinstance(self.weight_init, Tensor) else None

        # Weight initialization
        self.conv_impl = conv_impl(self.weight_init, shape, data_format=data_format)

        # Bias initialization
        if Validator.check_bool(has_bias, "has_bias", self.cls_name):
            if isinstance(bias_init, Tensor):
                if self.dtype is None:
                    self.dtype = bias_init.dtype
                elif self.dtype != bias_init.dtype:
                    raise TypeError("Data type of the weight_init tensor and the bias init tensor must be equal, "
                                    f"but got weight_init.dtype={self.dtype} and bias_init.dtype={bias_init.dtype}")
                bias_init_x, bias_init_y = get_x_and_y(bias_init)
            else:
                bias_init_x = bias_init_y = bias_init
            self.bias_x = Parameter(initializer(bias_init_x, [out_channels]), name='bias_x')
            self.bias_y = Parameter(initializer(bias_init_y, [out_channels]), name='bias_y')
            self.bias_add = P.BiasAdd()
        else:
            if self.bias_init != 'zeros':
                logger.warning("Value of 'has_bias' is False, value of 'bias_init' will be ignored.")
            self.bias_x = None
            self.bias_y = None
            self.bias_add = None

    def construct(self, u: Tensor) -> Tensor:
        if self.dtype is not None and self.dtype != u.dtype:
            raise TypeError("dtype must be equal to the data type of the inputs tensor, but got: "
                            f"dtype={self.dtype} and inputs.dtype={u.dtype}")
        x, y = get_x_and_y(u)
        out_x, out_y = self._construct(x, y)
        out = to_2channel(out_x, out_y, u.dtype)
        return out

    def extend_repr(self):
        """extend representation"""
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

    @abstractmethod
    def _construct(self,
                   x: Tensor,
                   y: Tensor) -> Tuple[Tensor, Tensor]:
        pass

    def _check_input_5dims(self, input_shape: tuple) -> None:
        if len(input_shape) != 5:
            raise ValueError(f"For {self.cls_name}, input should be 5 dims, but got shape {input_shape}.")


class Conv2d(_ConvNd):
    r"""
    2D convolution layer on the second-order hypercomplex input.

    Calculates the 2D convolution on the input tensor which is typically of shape
    :math:`(2, N, C_{in}, H_{in}, W_{in})`, where :math:`N` is batch size, :math:`C_{in}` is a number of channels,
    :math:`H_{in}, W_{in}` are the height and width of the feature layer respectively. The formula is defined as:

    .. math::

        \text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) +
        \sum_{k = 0}^{C_{in} - 1} \text{ccor}({\text{weight}(C_{\text{out}_j}, k), \text{inp}(N_i, k)})

    where :math:`ccor` is the `cross-correlation <https://en.wikipedia.org/wiki/Cross-correlation>`_, the exact
    implementation of which is defined and provided by the implementor part of the convolution layer,
    :math:`C_{in}` is the channel number of the input, :math:`out_{j}` corresponds to the jth channel of
    the output and :math:`j` is in the range of :math:`[0, C_{out}-1]`. :math:`\text{weight}(C_{\text{out}_j}, k)`
    is a convolution kernel slice with shape :math:`(\text{kernel_size[0]}, \text{kernel_size[1]})`,
    where :math:`\text{kernel_size[0]}` and :math:`\text{kernel_size[1]}` are the height and width of the convolution
    kernel respectively. :math:`\text{bias}` is the bias parameter and :math:`\text{inp}` is the input tensor.
    In this case, `data_format` of the input tensor is 'NCHW' and the shape of full convolution kernel is
    :math:`(C_{out}, C_{in} / \text{group}, \text{kernel_size[0]}, \text{kernel_size[1]})`,
    where `group` is the number of groups to split the input `inp` in the channel dimension. If `data_format` of the
    input tensor is 'NHWC', the shape of full convolution kernel will be
    :math:`(C_{out}, \text{kernel_size[0]}, \text{kernel_size[1]}), C_{in} / \text{group}`.

    For more details, please refers to the paper `Gradient Based Learning Applied to Document
    Recognition <http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf>`_.

    Note:
        On Ascend platform, only group convolution in depthwise convolution scenarios is supported.
        That is, when `group > 1`, condition `in\_channels` = `out\_channels` = `group` must be satisfied.

        'NCHW' format is supported only with GPU target device as of now.

    Args:
        conv_impl (TConvImpl): The implementor object of the convolution layer. Essentially, the concrete class name
            of this argument defines the algebra that the convolution layer will operate on.
        in_channels (int): The channel number of the input tensor of the Conv2d layer.
        out_channels (int): The channel number of the output tensor of the Conv2d layer.
        kernel_size (Union[int, tuple[int]]): Specifies the height and width of the 2D convolution kernel.
            The data type is an integer or a tuple of two integers. An integer represents the height
            and width of the convolution kernel. A tuple of two integers represents the height
            and width of the convolution kernel respectively.
        stride (Union[int, tuple[int]]): The movement stride of the 2D convolution kernel.
            The data type is an integer or a tuple of two integers. An integer represents the movement step size
            in both height and width directions. A tuple of two integers represents the movement step size in the
            height and width directions respectively. Default: 1.
        pad_mode (str): Specifies padding mode. The optional values are
            "same", "valid", "pad". Default: "same".

            - same: The width of the output is the same as the value of the input divided by `stride`.
              If this mode is set, the value of `padding` must be 0.

            - valid: Returns a valid calculated output without padding. Excess pixels that do not satisfy the
              calculation will be discarded. If this mode is set, the value of `padding` must be 0.

            - pad: Pads the input. Padding `padding` size of zero on both sides of the input.
              If this mode is set, the value of `padding` must be greater than or equal to 0.

        padding (Union[int, tuple[int]]): The number of padding on the height and width directions of the input.
            The data type is an integer or a tuple of four integers. If `padding` is an integer,
            then the top, bottom, left, and right padding are all equal to `padding`.
            If `padding` is a tuple of 4 integers, then the top, bottom, left, and right padding
            is equal to `padding[0]`, `padding[1]`, `padding[2]`, and `padding[3]` respectively.
            The value should be greater than or equal to 0. Default: 0.
        dilation (Union[int, tuple[int]]): Dilation size of 2D convolution kernel.
            The data type is an integer or a tuple of two integers. If :math:`k > 1`, the kernel is sampled
            every `k` elements. The value of `k` on the height and width directions is in range of [1, H]
            and [1, W] respectively. Default: 1.
        group (int): Splits filter into groups, `in_channels` and `out_channels` must be
            divisible by `group`. If the group is equal to `in_channels` and `out_channels`,
            this 2D convolution layer also can be called 2D depthwise convolution layer. Default: 1.
        has_bias (bool): Whether the Conv2d layer has a bias parameter. Default: False.
        weight_init (Union[Tensor, str, Initializer, numbers.Number]): Initialization method of weight parameter.
            It can be a Tensor, a string, an Initializer or a numbers.Number. When a string is specified,
            values from 'TruncatedNormal', 'Normal', 'Uniform', 'HeUniform' and 'XavierUniform' distributions as well
            as constant 'One' and 'Zero' distributions are possible. Alias 'xavier_uniform', 'he_uniform', 'ones'
            and 'zeros' are acceptable. Uppercase and lowercase are both acceptable. Refer to the values of
            Initializer for more details. Default: 'normal'.
        bias_init (Union[Tensor, str, Initializer, numbers.Number]): Initialization method of bias parameter.
            Available initialization methods are the same as 'weight_init'. Refer to the values of
            Initializer for more details. Default: 'zeros'.
        data_format (str): The optional value for data format, is 'NHWC' or 'NCHW'.
            Default: 'NCHW'.

    Inputs:
        - **inp** (Tensor) - Tensor of shape :math:`(2, N, C_{in}, H_{in}, W_{in})` or
          :math:`(2, N, H_{in}, W_{in}, C_{in})`, with float16 or float32 data type, or
          :math:`(N, C_{in}, H_{in}, W_{in})` or :math:`(N, H_{in}, W_{in}, C_{in})`, with complex64 data type.

    Outputs:
        Tensor of the same data type as `inp` and of shape :math:`(2, N, C_{out}, H_{out}, W_{out})` or
        :math:`(2, N, H_{out}, W_{out}, C_{out})`, with float16 or float32 data type, or
        :math:`(N, C_{out}, H_{out}, W_{out})` or :math:`(N, H_{out}, W_{out}, C_{out})`, with complex64 data type.

        pad_mode is 'same':

        .. math::
            \begin{array}{ll} \\
                H_{out} ＝ \left \lceil{\frac{H_{in}}{\text{stride[0]}}} \right \rceil \\
                W_{out} ＝ \left \lceil{\frac{W_{in}}{\text{stride[1]}}} \right \rceil \\
            \end{array}

        pad_mode is 'valid':

        .. math::
            \begin{array}{ll} \\
                H_{out} ＝ \left \lceil{\frac{H_{in} - \text{dilation[0]} \times (\text{kernel_size[0]} - 1) }
                {\text{stride[0]}}} \right \rceil \\
                W_{out} ＝ \left \lceil{\frac{W_{in} - \text{dilation[1]} \times (\text{kernel_size[1]} - 1) }
                {\text{stride[1]}}} \right \rceil \\
            \end{array}

        pad_mode is 'pad':

        .. math::
            \begin{array}{ll} \\
                H_{out} ＝ \left \lfloor{\frac{H_{in} + padding[0] + padding[1] - (\text{kernel_size[0]} - 1) \times
                \text{dilation[0]} - 1 }{\text{stride[0]}} + 1} \right \rfloor \\
                W_{out} ＝ \left \lfloor{\frac{W_{in} + padding[2] + padding[3] - (\text{kernel_size[1]} - 1) \times
                \text{dilation[1]} - 1 }{\text{stride[1]}} + 1} \right \rfloor \\
            \end{array}

    Raises:
        TypeError: If `in_channels`, `out_channels` or `group` is not an int.
        TypeError: If `kernel_size`, `stride`, `padding` or `dilation` is neither an int not a tuple.
        TypeError: If any two of `inp`, `weight_init` and `bias_init` are Tensors of different data type.
        ValueError: If `in_channels`, `out_channels`, `kernel_size`, `stride` or `dilation` is less than 1.
        ValueError: If `padding` is less than 0.
        ValueError: If `pad_mode` is not one of 'same', 'valid', 'pad'.
        ValueError: If `padding` is a tuple whose length is not equal to 4.
        ValueError: If `pad_mode` is not equal to 'pad' and `padding` is not equal to (0, 0, 0, 0).
        ValueError: If `data_format` is neither 'NCHW' not 'NHWC'.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    def __init__(self,
                 conv_impl: Type[TConvImpl],
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _size_2_t,
                 stride: _size_2_t = 1,
                 pad_mode: str = 'same',
                 padding: _size_2_t = 0,
                 dilation: _size_2_t = 1,
                 group: int = 1,
                 has_bias: bool = False,
                 weight_init: Union[Tensor, str, Initializer, numbers.Number] = 'normal',
                 bias_init: Union[Tensor, str, Initializer, numbers.Number] = 'zeros',
                 data_format: str = 'NCHW') -> None:
        """Initialize Conv2d."""
        kernel_size = twice(kernel_size)
        stride = twice(stride)
        self._dilation = dilation
        dilation = twice(dilation)
        super(Conv2d, self).__init__(conv_impl,
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
                                     data_format)
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

    def _construct(self,
                   x: Tensor,
                   y: Tensor) -> Tuple[Tensor, Tensor]:
        out_x, out_y = self.conv_impl(self.conv2d, x, y)
        if self.has_bias:
            out_x = self.bias_add(out_x, self.bias_x)
            out_y = self.bias_add(out_y, self.bias_y)
        return out_x, out_y


class Conv1d(_ConvNd):
    r"""
    1D convolution layer on the second-order hypercomplex input.

    Calculates the 1D convolution on the input tensor which is typically of shape :math:`(2, N, C_{in}, L_{in})`,
    where :math:`N` is batch size, :math:`C_{in}` is a number of channels and :math:`L_{in}` is a length of sequence.
    The formula is defined as:

    .. math::

        \text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) +
        \sum_{k = 0}^{C_{in} - 1} \text{ccor}({\text{weight}(C_{\text{out}_j}, k), \text{inp}(N_i, k)})

    where :math:`ccor` is the `cross-correlation <https://en.wikipedia.org/wiki/Cross-correlation>`_, the exact
    implementation of which is defined and provided by the implementor part of the convolution layer,
    :math:`C_{in}` is the channel number of the input, :math:`out_{j}` corresponds to the jth channel of
    the output and :math:`j` is in the range of :math:`[0, C_{out}-1]`. :math:`\text{weight}(C_{\text{out}_j}, k)`
    is a convolution kernel slice with shape :math:`\text{kernel_size}`, where :math:`\text{kernel_size}`
    is the width of the convolution kernel. :math:`\text{bias}` is the bias parameter,
    and :math:`\text{inp}` is the input tensor. The shape of full convolution kernel is
    :math:`(C_{out}, C_{in} / \text{group}, \text{kernel_size})`,
    where `group` is the number of groups to split the input `inp` in the channel dimension.

    For more details, please refers to the paper `Gradient Based Learning Applied to Document
    Recognition <http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf>`_.

    Note:
        On Ascend platform, only group convolution in depthwise convolution scenarios is supported.
        That is, when `group > 1`, condition `in\_channels` = `out\_channels` = `group` must be satisfied.

    Args:
        conv_impl (TConvImpl): The implementor object of the convolution layer. Essentially, the concrete class name
            of this argument defines the algebra that the convolution layer will operate on.
        in_channels (int): The channel number of the input tensor of the Conv1d layer.
        out_channels (int): The channel number of the output tensor of the Conv1d layer.
        kernel_size (int): Specifies the width of the 1D convolution kernel.
        stride (int): The movement stride of the 1D convolution kernel. Default: 1.
        pad_mode (str): Specifies padding mode. The optional values are
            "same", "valid", "pad". Default: "same".

            - same: The width of the output is the same as the value of the input divided by `stride`.
              If this mode is set, the value of `padding` must be 0.

            - valid: Returns a valid calculated output without padding. Excess pixels that do not satisfy the
              calculation will be discarded. If this mode is set, the value of `padding` must be 0.

            - pad: Pads the input. Padding `padding` size of zero on both sides of the input.
              If this mode is set, the value of `padding` must be greater than or equal to 0.

        padding (int): The number of padding on both sides of input.
            The value should be greater than or equal to 0. Default: 0.
        dilation (int): Dilation size of 1D convolution kernel. If :math:`k > 1`, the kernel is sampled
            every `k` elements. The value of `k` is in range of [1, L]. Default: 1.
        group (int): Splits filter into groups, `in_channels` and `out_channels` must be
            divisible by `group`. Default: 1.
        has_bias (bool): Whether the Conv1d layer has a bias parameter. Default: False.
        weight_init (Union[Tensor, str, Initializer, numbers.Number]): Initialization method of weight parameter.
            It can be a Tensor, a string, an Initializer or a numbers.Number. When a string is specified,
            values from 'TruncatedNormal', 'Normal', 'Uniform', 'HeUniform' and 'XavierUniform' distributions as well
            as constant 'One' and 'Zero' distributions are possible. Alias 'xavier_uniform', 'he_uniform', 'ones'
            and 'zeros' are acceptable. Uppercase and lowercase are both acceptable. Refer to the values of
            Initializer for more details. Default: 'normal'.
        bias_init (Union[Tensor, str, Initializer, numbers.Number]): Initialization method of bias parameter.
            Available initialization methods are the same as 'weight_init'. Refer to the values of
            Initializer for more details. Default: 'zeros'.

    Inputs:
        - **inp** (Tensor) - Tensor of shape :math:`(2, N, C_{in}, L_{in})`, with float16 or float32 data type,
        or :math:`(N, C_{in}, L_{in})`, with complex64 data type.

    Outputs:
        Tensor of the same data type as `inp` and of shape :math:`(2, N, C_{out}, L_{out})`, with float16 or float32
        data type, or :math:`(N, C_{out}, L_{out})`, with complex64 data type.

        pad_mode is 'same':

        .. math::
            L_{out} = \left \lceil{\frac{L_{in}}{\text{stride}}} \right \rceil

        pad_mode is 'valid':

        .. math::
            L_{out} = \left \lceil{\frac{L_{in} - \text{dilation} \times (\text{kernel_size} - 1) }
            {\text{stride}}} \right \rceil

        pad_mode is 'pad':

        .. math::
            L_{out} = \left \lfloor{\frac{L_{in} + 2 \times padding - (\text{kernel_size} - 1) \times
            \text{dilation} - 1 }{\text{stride}} + 1} \right \rfloor

    Raises:
        TypeError: If `in_channels`, `out_channels`, `kernel_size`, `stride`, `padding` or `dilation` is not an int.
        TypeError: If any two of `inp`, `weight_init` and `bias_init` are Tensors of different data type.
        ValueError: If `in_channels`, `out_channels`, `kernel_size`, `stride` or `dilation` is less than 1.
        ValueError: If `padding` is less than 0.
        ValueError: If `pad_mode` is not one of 'same', 'valid', 'pad'.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    def __init__(self,
                 conv_impl: Type[TConvImpl],
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _size_1_t,
                 stride: _size_1_t = 1,
                 pad_mode: str = 'same',
                 padding: _size_1_t = 0,
                 dilation: _size_1_t = 1,
                 group: int = 1,
                 has_bias: bool = False,
                 weight_init: Union[Tensor, str, Initializer, numbers.Number] = 'normal',
                 bias_init: Union[Tensor, str, Initializer, numbers.Number] = 'zeros') -> None:
        """Initialize Conv1d."""
        Validator.check_value_type("kernel_size", kernel_size, [int], self.cls_name)
        Validator.check_value_type("stride", stride, [int], self.cls_name)
        Validator.check_value_type("padding", padding, [int], self.cls_name)
        Validator.check_value_type("dilation", dilation, [int], self.cls_name)
        Validator.check_int(kernel_size, 1, Rel.GE, 'kernel_size', self.cls_name)
        Validator.check_int(stride, 1, Rel.GE, 'stride', self.cls_name)
        Validator.check_non_negative_int(padding, 'padding', self.cls_name)
        Validator.check_int(dilation, 1, Rel.GE, 'dilation', self.cls_name)
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

        super(Conv1d, self).__init__(conv_impl,
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
        if pad_mode not in ('valid', 'same', 'pad'):
            raise ValueError('Attr \'pad_mode\' of \'Conv1d\' Op passed '
                             + str(pad_mode) + ', should be one of values in \'valid\', \'same\', \'pad\'.')
        self.expand_dims = P.ExpandDims()
        self.squeeze = P.Squeeze(2)
        self.shape = P.Shape()

    def _check_input_3d(self, input_shape: tuple):
        if len(input_shape) != 3:
            raise ValueError(f"For '{self.cls_name}', the dimension of input must be 3d, but got {len(input_shape)}.")
        return None

    def _construct(self,
                   x: Tensor,
                   y: Tensor) -> Tuple[Tensor, Tensor]:
        x_shape = self.shape(x)
        self._check_input_3d(x_shape)
        x = self.expand_dims(x, 2)
        y = self.expand_dims(y, 2)
        out_x, out_y = self.conv_impl(self.conv2d, x, y)
        if self.has_bias:
            out_x = self.bias_add(out_x, self.bias_x)
            out_y = self.bias_add(out_y, self.bias_y)
        out_x = self.squeeze(out_x)
        out_y = self.squeeze(out_y)
        return out_x, out_y


class Conv3d(_ConvNd):
    r"""
    3D convolution layer on the second-order hypercomplex input.

    Calculates the 3D convolution on the input tensor which is typically of shape
    :math:`(2, N, C_{in}, D_{in}, H_{in}, W_{in})`,
    where :math:`N` is batch size, :math:`C_{in}` is a number of channels,
    :math:`D_{in}, H_{in}, W_{in}` are the depth, height and width of the feature layer respectively.
    The formula is defined as:

    .. math::

        \text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) +
        \sum_{k = 0}^{C_{in} - 1} \text{ccor}({\text{weight}(C_{\text{out}_j}, k), \text{inp}(N_i, k)})

    where :math:`ccor` is the `cross-correlation <https://en.wikipedia.org/wiki/Cross-correlation>`_, the exact
    implementation of which is defined and provided by the implementor part of the convolution layer,
    :math:`C_{in}` is the channel number of the input, :math:`out_{j}` corresponds to the jth channel of
    the output and :math:`j` is in the range of :math:`[0，C_{out}-1]`. :math:`\text{weight}(C_{\text{out}_j}, k)`
    is a convolution kernel slice with shape
    :math:`(\text{kernel_size[0]}, \text{kernel_size[1]}, \text{kernel_size[2]})`,
    where :math:`\text{kernel_size[0]}`, :math:`\text{kernel_size[1]}` and :math:`\text{kernel_size[2]}` are
    the depth, height and width of the convolution kernel respectively. :math:`\text{bias}` is the bias parameter
    and :math:`\text{inp}` is the input tensor.
    The shape of full convolution kernel is
    :math:`(C_{out}, C_{in} / \text{group}, \text{kernel_size[0]}, \text{kernel_size[1]}, \text{kernel_size[2]})`,
    where `group` is the number of groups to split the input `x` in the channel dimension.

    For more details, please refers to the paper `Gradient Based Learning Applied to Document
    Recognition <http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf>`_.

    Note:
        On Ascend platform, only group convolution in depthwise convolution scenarios is supported.
        That is, when `group>1`, condition `in\_channels` = `out\_channels` = `group` must be satisfied.

    Args:
        conv_impl (TConvImpl): The implementor object of the convolution layer. Essentially, the concrete class name
            of this argument defines the algebra that the convolution layer will operate on.
        in_channels (int): The channel number of the input tensor of the Conv3d layer.
        out_channels (int): The channel number of the output tensor of the Conv3d layer.
        kernel_size (Union[int, tuple[int]]): Specifies the depth, height and width of the 3D convolution kernel.
            The data type is an integer or a tuple of three integers. An integer represents the depth, height
            and width of the convolution kernel. A tuple of three integers represents the depth, height
            and width of the convolution kernel respectively.
        stride (Union[int, tuple[int]]): The movement stride of the 3D convolution kernel.
            The data type is an integer or a tuple of three integers. An integer represents the movement step size
            in depth, height and width directions. A tuple of three integers represents the movement step size
            in the depth, height and width directions respectively. Default: 1.
        pad_mode (str): Specifies padding mode. The optional values are
            "same", "valid", "pad". Default: "same".

            - same: The width of the output is the same as the value of the input divided by `stride`.
              If this mode is set, the value of `padding` must be 0.

            - valid: Returns a valid calculated output without padding. Excess pixels that do not satisfy the
              calculation will be discarded. If this mode is set, the value of `padding` must be 0.

            - pad: Pads the input. Padding `padding` size of zero on both sides of the input.
              If this mode is set, the value of `padding` must be greater than or equal to 0.

        padding (Union(int, tuple[int])): The number of padding on the depth, height and width directions of the input.
            The data type is an integer or a tuple of six integers. If `padding` is an integer,
            then the head, tail, top, bottom, left, and right padding are all equal to `padding`.
            If `padding` is a tuple of six integers, then the head, tail, top, bottom, left, and right padding
            is equal to `padding[0]`, `padding[1]`, `padding[2]`, `padding[3]`, `padding[4]` and `padding[5]`
            respectively. The value should be greater than or equal to 0. Default: 0.
        dilation (Union[int, tuple[int]]): Dilation size of 3D convolution kernel.
            The data type is an integer or a tuple of three integers. If :math:`k > 1`, the kernel is sampled
            every `k` elements. The value of `k` on the depth, height and width directions is in range of
            [1, D], [1, H] and [1, W] respectively. Default: 1.
        group (int): Splits filter into groups, `in_channels` and `out_channels` must be
            divisible by `group`. Default: 1. Only 1 is currently supported.
        has_bias (bool): Whether the Conv3d layer has a bias parameter. Default: False.
        weight_init (Union[Tensor, str, Initializer, numbers.Number]): Initialization method of weight parameter.
            It can be a Tensor, a string, an Initializer or a numbers.Number. When a string is specified,
            values from 'TruncatedNormal', 'Normal', 'Uniform', 'HeUniform' and 'XavierUniform' distributions as well
            as constant 'One' and 'Zero' distributions are possible. Alias 'xavier_uniform', 'he_uniform', 'ones'
            and 'zeros' are acceptable. Uppercase and lowercase are both acceptable. Refer to the values of
            Initializer for more details. Default: 'normal'.
        bias_init (Union[Tensor, str, Initializer, numbers.Number]): Initialization method of bias parameter.
            Available initialization methods are the same as 'weight_init'. Refer to the values of
            Initializer for more details. Default: 'zeros'.
        data_format (str): The optional value for data format. Currently only support "NCDHW".


    Inputs:
        - **inp** (Tensor) - Tensor of shape :math:`(2, N, C_{in}, D_{in}, H_{in}, W_{in})`, with float16 or float32
          data type, or :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})`, with complex64 data type.

    Outputs:
        Tensor of the same data type as `inp` and of shape :math:`(2, N, C_{out}, D_{out}, H_{out}, W_{out})`, with
        float16 or float32 data type, or :math:`(N, C_{out}, D_{out}, H_{out}, W_{out})`, with complex64 data type.

        pad_mode is 'same':

        .. math::
            \begin{array}{ll} \\
                D_{out} ＝ \left \lceil{\frac{D_{in}}{\text{stride[0]}}} \right \rceil \\
                H_{out} ＝ \left \lceil{\frac{H_{in}}{\text{stride[1]}}} \right \rceil \\
                W_{out} ＝ \left \lceil{\frac{W_{in}}{\text{stride[2]}}} \right \rceil \\
            \end{array}


        pad_mode is 'valid':

        .. math::
            \begin{array}{ll} \\
                D_{out} ＝ \left \lfloor{\frac{D_{in} - \text{dilation[0]} \times (\text{kernel_size[0]} - 1) }
                {\text{stride[0]}} + 1} \right \rfloor \\
                H_{out} ＝ \left \lfloor{\frac{H_{in} - \text{dilation[1]} \times (\text{kernel_size[1]} - 1) }
                {\text{stride[1]}} + 1} \right \rfloor \\
                W_{out} ＝ \left \lfloor{\frac{W_{in} - \text{dilation[2]} \times (\text{kernel_size[2]} - 1) }
                {\text{stride[2]}} + 1} \right \rfloor \\
            \end{array}

        pad_mode is 'pad':

        .. math::
            \begin{array}{ll} \\
                D_{out} ＝ \left \lfloor{\frac{D_{in} + padding[0] + padding[1] - (\text{dilation[0]} - 1) \times
                \text{kernel_size[0]} - 1 }{\text{stride[0]}} + 1} \right \rfloor \\
                H_{out} ＝ \left \lfloor{\frac{H_{in} + padding[2] + padding[3] - (\text{dilation[1]} - 1) \times
                \text{kernel_size[1]} - 1 }{\text{stride[1]}} + 1} \right \rfloor \\
                W_{out} ＝ \left \lfloor{\frac{W_{in} + padding[4] + padding[5] - (\text{dilation[2]} - 1) \times
                \text{kernel_size[2]} - 1 }{\text{stride[2]}} + 1} \right \rfloor \\
            \end{array}

    Raises:
        TypeError: If `in_channels`, `out_channels` or `group` is not an int.
        TypeError: If `kernel_size`, `stride`, `padding` or `dilation` is neither an int nor a tuple.
        TypeError: If any two of `inp`, `weight_init` and `bias_init` are Tensors of different data type.
        ValueError: If `out_channels`, `kernel_size`, `stride` or `dilation` is less than 1.
        ValueError: If `padding` is less than 0.
        ValueError: If `pad_mode` is not one of 'same', 'valid', 'pad'.
        ValueError: If `padding` is a tuple whose length is not equal to 6.
        ValueError: If `pad_mode` is not equal to 'pad' and `padding` is not equal to (0, 0, 0, 0, 0, 0).
        ValueError: If `data_format` is not 'NCDHW'.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    def __init__(self,
                 conv_impl: Type[TConvImpl],
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _size_3_t,
                 stride: _size_3_t = (1, 1, 1),
                 pad_mode: str = 'same',
                 padding: _size_3_t = 0,
                 dilation: _size_3_t = (1, 1, 1),
                 group: int = 1,
                 has_bias: bool = False,
                 weight_init: Union[Tensor, str, Initializer, numbers.Number] = 'normal',
                 bias_init: Union[Tensor, str, Initializer, numbers.Number] = 'zeros',
                 data_format: str = 'NCDHW') -> None:
        """Initialize Conv3d."""
        self.conv_impl = conv_impl
        kernel_size = _check_3d_int_or_tuple("kernel_size", kernel_size, self.cls_name)
        stride = _check_3d_int_or_tuple("stride", stride, self.cls_name)
        dilation = _check_3d_int_or_tuple("dilation", dilation, self.cls_name)
        Validator.check_value_type('padding', padding, (int, tuple), self.cls_name)
        if isinstance(padding, tuple):
            Validator.check_equal_int(len(padding), 6, 'padding size', self.cls_name)
        super(Conv3d, self).__init__(conv_impl,
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
                                     data_format)
        self.conv3d = P.Conv3D(out_channel=self.out_channels,
                               kernel_size=self.kernel_size,
                               mode=1,
                               pad_mode=self.pad_mode,
                               pad=self.padding,
                               stride=self.stride,
                               dilation=self.dilation,
                               group=self.group,
                               data_format=self.data_format)
        self.bias_add = P.BiasAdd(data_format=self.data_format)
        self.shape = P.Shape()

    def _construct(self,
                   x: Tensor,
                   y: Tensor) -> Tuple[Tensor, Tensor]:
        x_shape = self.shape(x)
        self._check_input_5dims(x_shape)
        out_x, out_y = self.conv_impl(self.conv3d, x, y)
        if self.has_bias:
            out_x = self.bias_add(out_x, self.bias_x)
            out_y = self.bias_add(out_y, self.bias_y)
        return out_x, out_y


class Conv3dTranspose(_ConvNd):
    r"""
    3D transposed convolution layer on the second-order hypercomplex input.

    Calculates a 3D transposed convolution, which can be regarded as Conv3d for the gradient of the input.
    It also called deconvolution (although it is not an actual deconvolution).

    The input is typically of shape :math:`(2, N, C, D_{in}, H_{in}, W_{in})`, where :math:`N` is batch size,
    :math:`C` is a number of channels, :math:`D_{in}, H_{in}, W_{in}` are the depth, height and width of
    the feature layer respectively.

    When Conv3d and Conv3dTranspose are initialized with the same parameters, and `pad_mode` is set to 'pad',
    :math:`dilation * (kernel\_size - 1) - padding` amount of zero will be padded to the depth, height and width
    directions of the input, they are inverses of each other in regard to the input and output shapes in this case.
    However, when `stride` > 1, Conv3d maps multiple input shapes to the same output shape. Deconvolutional network
    can refer to `Deconvolutional Networks <https://www.matthewzeiler.com/mattzeiler/deconvolutionalnetworks.pdf>`_.

    Args:
        conv_impl (TConvImpl): The implementor object of the convolution layer. Essentially, the concrete class name
            of this argument defines the algebra that the convolution layer will operate on.
        in_channels (int): The channel number of the input tensor of the Conv3dTranspose layer.
        out_channels (int): The channel number of the output tensor of the Conv3dTranspose layer.
        kernel_size (Union[int, tuple[int]]): Specifies the depth, height and width of the 3D convolution kernel.
            The data type is an integer or a tuple of three integers. An integer represents the depth, height
            and width of the convolution kernel. A tuple of three integers represents the depth, height
            and width of the convolution kernel respectively.
        stride (Union[int, tuple[int]]): The movement stride of the 3D convolution kernel.
            The data type is an integer or a tuple of three integers. An integer represents the movement step size
            in depth, height and width directions. A tuple of three integers represents the movement step size
            in the depth, height and width directions respectively. Default: 1.
        pad_mode (str): Specifies padding mode. The optional values are
            "same", "valid", "pad". Default: "same".

            - same: The width of the output is the same as the value of the input divided by `stride`.
              If this mode is set, the value of `padding` must be 0.

            - valid: Returns a valid calculated output without padding. Excess pixels that do not satisfy the
              calculation will be discarded. If this mode is set, the value of `padding` must be 0.

            - pad: Pads the input. Padding `padding` size of zero on both sides of the input.
              If this mode is set, the value of `padding` must be greater than or equal to 0.

        padding (Union(int, tuple[int])): The number of padding on the depth, height and width directions of the input.
            The data type is an integer or a tuple of six integers. If `padding` is an integer,
            then the head, tail, top, bottom, left, and right padding are all equal to `padding`.
            If `padding` is a tuple of six integers, then the head, tail, top, bottom, left, and right padding
            is equal to `padding[0]`, `padding[1]`, `padding[2]`, `padding[3]`, `padding[4]` and `padding[5]`
            respectively. The value should be greater than or equal to 0. Default: 0.
        dilation (Union[int, tuple[int]]): Dilation size of 3D convolution kernel.
            The data type is an integer or a tuple of three integers. If :math:`k > 1`, the kernel is sampled
            every `k` elements. The value of `k` on the depth, height and width directions is in range of
            [1, D], [1, H] and [1, W] respectively. Default: 1.
        group (int): Splits filter into groups, `in_channels` and `out_channels` must be
            divisible by `group`. Default: 1. Only 1 is currently supported.
        output_padding (Union(int, tuple[int])): The number of padding on the depth, height and width directions of
            the output. The data type is an integer or a tuple of six integers. If `output_padding` is an integer,
            then the head, tail, top, bottom, left, and right padding are all equal to `output_padding`.
            If `output_padding` is a tuple of six integers, then the head, tail, top, bottom, left, and right padding
            is equal to `output_padding[0]`, `output_padding[1]`, `output_padding[2]`, `output_padding[3]`,
            `output_padding[4]` and `output_padding[5]` respectively. The value should be greater than or equal to 0.
            Default: 0.
        has_bias (bool): Whether the Conv3dTranspose layer has a bias parameter. Default: False.
        weight_init (Union[Tensor, str, Initializer, numbers.Number]): Initialization method of weight parameter.
            It can be a Tensor, a string, an Initializer or a numbers.Number. When a string is specified,
            values from 'TruncatedNormal', 'Normal', 'Uniform', 'HeUniform' and 'XavierUniform' distributions as well
            as constant 'One' and 'Zero' distributions are possible. Alias 'xavier_uniform', 'he_uniform', 'ones'
            and 'zeros' are acceptable. Uppercase and lowercase are both acceptable. Refer to the values of
            Initializer for more details. Default: 'normal'.
        bias_init (Union[Tensor, str, Initializer, numbers.Number]): Initialization method of bias parameter.
            Available initialization methods are the same as 'weight_init'. Refer to the values of
            Initializer for more details. Default: 'zeros'.
        data_format (str): The optional value for data format. Currently only support 'NCDHW'.


    Inputs:
        - **inp** (Tensor) - Tensor of shape :math:`(2, N, C_{in}, D_{in}, H_{in}, W_{in})`, with float16 and float32
        data type, or :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})`, with complex64 data type.

    Outputs:
        Tensor of the same data type as `inp` and of shape :math:`(2, N, C_{out}, D_{out}, H_{out}, W_{out})`, with
        float16 or float32 data type, or :math:`(N, C_{out}, D_{out}, H_{out}, W_{out})`, with complex64 data type.

        pad_mode is 'same':

        .. math::
            \begin{array}{ll} \\
                D_{out} ＝ \left \lfloor{\frac{D_{in}}{\text{stride[0]}} + 1} \right \rfloor \\
                H_{out} ＝ \left \lfloor{\frac{H_{in}}{\text{stride[1]}} + 1} \right \rfloor \\
                W_{out} ＝ \left \lfloor{\frac{W_{in}}{\text{stride[2]}} + 1} \right \rfloor \\
            \end{array}


        pad_mode is 'valid':

        .. math::
            \begin{array}{ll} \\
                D_{out} ＝ \left \lfloor{\frac{D_{in} - \text{dilation[0]} \times (\text{kernel_size[0]} - 1) }
                {\text{stride[0]}} + 1} \right \rfloor \\
                H_{out} ＝ \left \lfloor{\frac{H_{in} - \text{dilation[1]} \times (\text{kernel_size[1]} - 1) }
                {\text{stride[1]}} + 1} \right \rfloor \\
                W_{out} ＝ \left \lfloor{\frac{W_{in} - \text{dilation[2]} \times (\text{kernel_size[2]} - 1) }
                {\text{stride[2]}} + 1} \right \rfloor \\
            \end{array}

        pad_mode is 'pad':

        .. math::
            \begin{array}{ll} \\
                D_{out} ＝ \left \lfloor{\frac{D_{in} + padding[0] + padding[1] - (\text{dilation[0]} - 1) \times
                \text{kernel_size[0]} - 1 }{\text{stride[0]}} + 1} \right \rfloor \\
                H_{out} ＝ \left \lfloor{\frac{H_{in} + padding[2] + padding[3] - (\text{dilation[1]} - 1) \times
                \text{kernel_size[1]} - 1 }{\text{stride[1]}} + 1} \right \rfloor \\
                W_{out} ＝ \left \lfloor{\frac{W_{in} + padding[4] + padding[5] - (\text{dilation[2]} - 1) \times
                \text{kernel_size[2]} - 1 }{\text{stride[2]}} + 1} \right \rfloor \\
            \end{array}

    Raises:
        TypeError: If `in_channels`, `out_channels` or `group` is not an int.
        TypeError: If `kernel_size`, `stride`, `padding` , `dilation` or `output_padding`
                   is neither an int not a tuple of three.
        TypeError: If any two of `inp`, `weight_init` and `bias_init` are Tensors of different data type.
        TypeError: If input data type is not float16 or float32.
        ValueError: If `in_channels`, `out_channels`, `kernel_size`, `stride` or `dilation` is less than 1.
        ValueError: If `padding` is less than 0.
        ValueError: If `pad_mode` is not one of 'same', 'valid', 'pad'.
        ValueError: If `padding` is a tuple whose length is not equal to 6.
        ValueError: If `pad_mode` is not equal to 'pad' and `padding` is not equal to (0, 0, 0, 0, 0, 0).
        ValueError: If `data_format` is not 'NCDHW'.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    def __init__(self,
                 conv_impl: Type[TConvImpl],
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _size_3_t,
                 stride: _size_3_t = (1, 1, 1),
                 pad_mode: str = 'same',
                 padding: _size_3_t = 0,
                 dilation: _size_3_t = (1, 1, 1),
                 group: int = 1,
                 output_padding: int = 0,
                 has_bias: bool = False,
                 weight_init: Union[Tensor, str, Initializer, numbers.Number] = 'normal',
                 bias_init: Union[Tensor, str, Initializer, numbers.Number] = 'zeros',
                 data_format: str = 'NCDHW') -> None:
        """Initialize Conv3dTranspose."""
        self.conv_impl = conv_impl
        kernel_size = _check_3d_int_or_tuple("kernel_size", kernel_size, self.cls_name)
        stride = _check_3d_int_or_tuple("stride", stride, self.cls_name)
        dilation = _check_3d_int_or_tuple("dilation", dilation, self.cls_name)
        Validator.check_value_type('padding', padding, (int, tuple), self.cls_name)
        if isinstance(padding, tuple):
            Validator.check_equal_int(len(padding), 6, 'padding size', self.cls_name)
        self.output_padding = _check_3d_int_or_tuple("output_padding", output_padding, self.cls_name,
                                                     greater_zero=False)
        super(Conv3dTranspose, self).__init__(conv_impl,
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
                                              transposed=True)
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

    def _construct(self,
                   x: Tensor,
                   y: Tensor) -> Tuple[Tensor, Tensor]:
        x_shape = self.shape(x)
        self._check_input_5dims(x_shape)
        out_x, out_y = self.conv_impl(self.conv3d_transpose, x, y)
        if self.has_bias:
            out_x = self.bias_add(out_x, self.bias_x)
            out_y = self.bias_add(out_y, self.bias_y)
        return out_x, out_y
