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
"""Dual Operators"""
import numbers
from typing import Union
from mindspore.common.initializer import Initializer
from mindspore.common.tensor import Tensor
# Batch Normalization
from mindspore.hypercomplex.hypercomplex.hc_bn import BatchNorm1d as HBatchNorm1d, \
    BatchNorm2d as HBatchNorm2d, BatchNorm3d as HBatchNorm3d
from mindspore.hypercomplex.dual._dual_bn_impl import _BatchNormImpl as BatchNormImpl
# Convolution
from mindspore.hypercomplex.hypercomplex.hc_conv import Conv1d as HConv1d, \
    Conv2d as HConv2d, Conv3d as HConv3d
from mindspore.hypercomplex.dual._dual_conv_impl import _ReDuConvImpl as ConvImpl
# Dense
from mindspore.hypercomplex.hypercomplex.hc_dense import Dense as HDense
from mindspore.hypercomplex.dual._dual_dense_impl import _DenseImpl as DenseImpl
from mindspore.hypercomplex.hypercomplex.uniform_operator import _UniformOperator

from mindspore.hypercomplex.utils import _size_1_t, _size_2_t, _size_3_t


class Conv2d(_UniformOperator):
    r"""
    2D convolution layer on the dual-valued input.

    Calculates the 2D convolution on the input tensor which is typically of shape
    :math:`(2, N, C_{in}, H_{in}, W_{in})`, where :math:`N` is batch size, :math:`C_{in}` is a number of channels,
    :math:`H_{in}, W_{in}` are the height and width of the feature layer respectively. The formula is defined as:

    .. math::

        \text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) +
        \sum_{k = 0}^{C_{in} - 1} \text{hccor}({\text{weight}(C_{\text{out}_j}, k), \text{inp}(N_i, k)})

    where :math:`C_{in}` is the channel number of the input, :math:`out_{j}` corresponds to the jth channel of
    the output and :math:`j` is in the range of :math:`[0, C_{out}-1]`. :math:`\text{weight}(C_{\text{out}_j}, k)`
    is a convolution kernel slice with shape :math:`(\text{kernel_size[0]}, \text{kernel_size[1]})`,
    where :math:`\text{kernel_size[0]}` and :math:`\text{kernel_size[1]}` are the height and width of the convolution
    kernel respectively. :math:`\text{bias}` is the bias parameter and :math:`\text{inp}` is the input tensor.
    In this case, `data_format` of the input tensor is 'NCHW' and the shape of full convolution kernel is
    :math:`(C_{out}, C_{in} / \text{group}, \text{kernel_size[0]}, \text{kernel_size[1]})`,
    where `group` is the number of groups to split the input `inp` in the channel dDuension. If `data_format` of the
    input tensor is 'NHWC', the shape of full convolution kernel will be
    :math:`(C_{out}, \text{kernel_size[0]}, \text{kernel_size[1]}), C_{in} / \text{group}`.
    :math:`hccor` is the dual-valued `cross-correlation <https://en.wikipedia.org/wiki/Cross-correlation>`_.
    This implies the operation as:

    .. math::
        \text{Re(ccor)} = \text{ccor}(\text{Re(kernel)}, \text{Re(inp)}) + \text{Re(bias)}\\
        \text{Du(ccor)} = \text{ccor}(\text{Du(kernel)}, \text{Re(inp)})
        + \text{ccor}(\text{Re(kernel)}, \text{Du(inp)}) + \text{Du(bias)}

    where and :math:`cccor` is the real-valued `cross-correlation <https://en.wikipedia.org/wiki/Cross-correlation>`_,
    :math:`inp` is the dual input tensors, :math:`\text{kernel}` is a dual weight matrix with the same
    data type as the :math:`inp` created by the layer, and :math:`\text{bias}` is a dual bias vector with the same
    data type as the :math:`inp` created by the layer (only if has_bias is True). :math:`\text{Re(...)}`
    and :math:`\text{Du(...)}` are respectively real and dual parts of the dual-valued expression inside
    the parentheses.

    For more details, please refers to the paper `Gradient Based Learning Applied to Document
    Recognition <http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf>`_.

    Note:
        On Ascend platform, only group convolution in depthwise convolution scenarios is supported.
        That is, when `group > 1`, condition `in\_channels` = `out\_channels` = `group` must be satisfied.

        'NCHW' format is supported only with GPU target device as of now.

    Args:
        in_channels (int): The channel number of the input tensor of the Conv2d layer.
        out_channels (int): The channel number of the output tensor of the Conv2d layer.
        kernel_size (Union[int, tuple[int]]): Specifies the height and width of the 2D convolution kernel.
            The data type is an integer or a tuple of two integers. An integer represents the height
            and width of the convolution kernel. A tuple of two integers represents the height
            and width of the convolution kernel respectively.
        stride (Union[int, tuple[int]]): The movement stride of the 2D convolution kernel.
            The data type is an integer or a tuple of two integers. An integer represents the movement step size
            in both height and width directions. A tuple of two integers represents the movement step size in
            the height and width directions respectively. Default: 1.
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
        - **inp** (Tensor) - Tensor of shape :math:`(2, N, C_{in}, H_{in}, W_{in})` \
          or :math:`(2, N, H_{in}, W_{in}, C_{in})`.

    Outputs:
        Tensor of shape :math:`(2, N, C_{out}, H_{out}, W_{out})` or :math:`(2, N, H_{out}, W_{out}, C_{out})`.

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
        ValueError: If `in_channels`, `out_channels`, `kernel_size`, `stride` or `dilation` is less than 1.
        ValueError: If `padding` is less than 0.
        ValueError: If `pad_mode` is not one of 'same', 'valid', 'pad'.
        ValueError: If `padding` is a tuple whose length is not equal to 4.
        ValueError: If `pad_mode` is not equal to 'pad' and `padding` is not equal to (0, 0, 0, 0).
        ValueError: If `data_format` is neither 'NCHW' not 'NHWC'.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore.hypercomplex.dual import Conv2d
        >>> from mindspore import Tensor
        >>> w = Tensor(np.random.random((2, 128, 3, 7, 7)).astype(np.float32))
        >>> b = Tensor(np.random.random((2, 128)).astype(np.float32))
        >>> net = Conv2d(
        >>>     in_channels=3, out_channels=128, kernel_size=7, stride=2, padding=3,
        >>>     pad_mode='pad', weight_init=w, bias_init=b, has_bias=True
        >>> )
        >>> inp = Tensor(np.random.random((2, 16, 3, 224, 224)).astype(np.float32))
        >>> out = net(inp)
        >>> print(out.shape)
        (2, 16, 128, 112, 112)
    """

    def __init__(self,
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
        super(Conv2d, self).__init__(HConv2d,
                                     ConvImpl,
                                     in_channels=in_channels,
                                     out_channels=out_channels,
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     pad_mode=pad_mode,
                                     padding=padding,
                                     dilation=dilation,
                                     group=group,
                                     has_bias=has_bias,
                                     weight_init=weight_init,
                                     bias_init=bias_init,
                                     data_format=data_format)


class Conv1d(_UniformOperator):
    r"""
    1D convolution layer on the dual-valued input.

    Calculates the 1D convolution on the input tensor which is typically of shape :math:`(2, N, C_{in}, L_{in})`,
    where :math:`N` is batch size, :math:`C_{in}` is a number of channels and :math:`L_{in}` is a length of sequence.
    The formula is defined as:

    .. math::

        \text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) +
        \sum_{k = 0}^{C_{in} - 1} \text{ccor}({\text{weight}(C_{\text{out}_j}, k), \text{inp}(N_i, k)})

    where :math:`C_{in}` is the channel number of the input, :math:`out_{j}` corresponds to the jth channel of
    the output and :math:`j` is in the range of :math:`[0, C_{out}-1]`. :math:`\text{weight}(C_{\text{out}_j}, k)`
    is a convolution kernel slice with shape :math:`\text{kernel_size}`, where :math:`\text{kernel_size}`
    is the width of the convolution kernel. :math:`\text{bias}` is the bias parameter,
    and :math:`\text{inp}` is the input tensor. The shape of full convolution kernel is
    :math:`(C_{out}, C_{in} / \text{group}, \text{kernel_size})`,
    where `group` is the number of groups to split the input `inp` in the channel dimension.
    :math:`ccor` is the dual-valued `cross-correlation <https://en.wikipedia.org/wiki/Cross-correlation>`_.
    This implies the operation as:

    .. math::
        \text{Re(ccor)} = \text{ccor}(\text{Re(kernel)}, \text{Re(inp)}) + \text{Re(bias)}\\
        \text{Du(ccor)} = \text{ccor}(\text{Du(kernel)}, \text{Re(inp)})
        + \text{ccor}(\text{Re(kernel)}, \text{Du(inp)}) + \text{Du(bias)}

    where and :math:`ccor` is the real-valued `cross-correlation <https://en.wikipedia.org/wiki/Cross-correlation>`_,
    :math:`inp` is the dual input tensors, :math:`\text{kernel}` is a dual weight matrix with the same
    data type as the :math:`inp` created by the layer, and :math:`\text{bias}` is a dual bias vector with the same
    data type as the :math:`inp` created by the layer (only if has_bias is True). :math:`\text{Re(...)}` and
    :math:`\text{Du(...)}` are respectively real and dual parts of the dual-valued expression inside the parentheses.

    For more details, please refers to the paper `Gradient Based Learning Applied to Document
    Recognition <http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf>`_.

    Note:
        On Ascend platform, only group convolution in depthwise convolution scenarios is supported.
        That is, when `group > 1`, condition `in\_channels` = `out\_channels` = `group` must be satisfied.

    Args:
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
        - **inp** (Tensor) - Tensor of shape :math:`(2, N, C_{in}, L_{in})`.

    Outputs:
        Tensor of shape :math:`(2, N, C_{out}, L_{out})`.

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
        ValueError: If `in_channels`, `out_channels`, `kernel_size`, `stride` or `dilation` is less than 1.
        ValueError: If `padding` is less than 0.
        ValueError: If `pad_mode` is not one of 'same', 'valid', 'pad'.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore.hypercomplex.dual import Conv1d
        >>> from mindspore import Tensor
        >>> w = Tensor(np.random.random((2, 16, 1, 6)).astype(np.float32))
        >>> b = Tensor(np.random.random((2, 16)).astype(np.float32))
        >>> net = Conv1d(
        >>>     in_channels=1, out_channels=16, kernel_size=6, stride=2, padding=2,
        >>>     pad_mode='pad', weight_init=w, bias_init=b, has_bias=True
        >>> )
        >>> u = Tensor(np.random.random((2, 8, 1, 4096)).astype(np.float32))
        >>> out = net(u)
        >>> print(out.shape)
        (2, 8, 16, 2048)
    """

    def __init__(self,
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
        super(Conv1d, self).__init__(HConv1d,
                                     ConvImpl,
                                     in_channels=in_channels,
                                     out_channels=out_channels,
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     pad_mode=pad_mode,
                                     padding=padding,
                                     dilation=dilation,
                                     group=group,
                                     has_bias=has_bias,
                                     weight_init=weight_init,
                                     bias_init=bias_init)


class Conv3d(_UniformOperator):
    r"""
    3D convolution layer on the dual-valued input.

    Calculates the 3D convolution on the input tensor which is typically of shape
    :math:`(2, N, C_{in}, D_{in}, H_{in}, W_{in})`,
    where :math:`N` is batch size, :math:`C_{in}` is a number of channels,
    :math:`D_{in}, H_{in}, W_{in}` are the depth, height and width of the feature layer respectively.
    The formula is defined as:

    .. math::

        \text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) +
        \sum_{k = 0}^{C_{in} - 1} \text{ccor}({\text{weight}(C_{\text{out}_j}, k), \text{inp}(N_i, k)})

    where :math:`C_{in}` is the channel number of the input, :math:`out_{j}` corresponds to the jth channel of
    the output and :math:`j` is in the range of :math:`[0，C_{out}-1]`. :math:`\text{weight}(C_{\text{out}_j}, k)`
    is a convolution kernel slice with shape
    :math:`(\text{kernel_size[0]}, \text{kernel_size[1]}, \text{kernel_size[2]})`,
    where :math:`\text{kernel_size[0]}`, :math:`\text{kernel_size[1]}` and :math:`\text{kernel_size[2]}` are
    the depth, height and width of the convolution kernel respectively. :math:`\text{bias}` is the bias parameter
    and :math:`\text{inp}` is the input tensor.
    The shape of full convolution kernel is
    :math:`(C_{out}, C_{in} / \text{group}, \text{kernel_size[0]}, \text{kernel_size[1]}, \text{kernel_size[2]})`,
    where `group` is the number of groups to split the input `x` in the channel dimension.
    :math:`hccor` is the dual-valued `cross-correlation <https://en.wikipedia.org/wiki/Cross-correlation>`_.
    This implies the operation as:

    .. math::
        \text{Re(ccor)} = \text{ccor}(\text{Re(kernel)}, \text{Re(inp)}) + \text{Re(bias)}\\
        \text{Du(ccor)} = \text{ccor}(\text{Du(kernel)}, \text{Re(inp)})
        + \text{ccor}(\text{Re(kernel)}, \text{Du(inp)}) + \text{Du(bias)}

    where and :math:`ccor` is the real-valued `cross-correlation <https://en.wikipedia.org/wiki/Cross-correlation>`_,
    :math:`inp` is the dual input tensors, :math:`\text{kernel}` is a dual weight matrix with the same
    data type as the :math:`inp` created by the layer, and :math:`\text{bias}` is a dual bias vector with the same
    data type as the :math:`inp` created by the layer (only if has_bias is True). :math:`\text{Re(...)}`
    and :math:`\text{Du(...)}` are respectively real and dual parts of the dual-valued expression inside
    the parentheses.

    For more details, please refers to the paper `Gradient Based Learning Applied to Document
    Recognition <http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf>`_.

    Note:
        On Ascend platform, only group convolution in depthwise convolution scenarios is supported.
        That is, when `group>1`, condition `in\_channels` = `out\_channels` = `group` must be satisfied.

    Args:
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
        - **inp** (Tensor) - Tensor of shape :math:`(2, N, C_{in}, D_{in}, H_{in}, W_{in})`.
          Currently input data type only support float16 and float32.

    Outputs:
        Tensor of shape is :math:`(2, N, C_{out}, D_{out}, H_{out}, W_{out})`.

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
        ValueError: If `out_channels`, `kernel_size`, `stride` or `dilation` is less than 1.
        ValueError: If `padding` is less than 0.
        ValueError: If `pad_mode` is not one of 'same', 'valid', 'pad'.
        ValueError: If `padding` is a tuple whose length is not equal to 6.
        ValueError: If `pad_mode` is not equal to 'pad' and `padding` is not equal to (0, 0, 0, 0, 0, 0).
        ValueError: If `data_format` is not 'NCDHW'.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore.hypercomplex.dual import Conv3d
        >>> from mindspore import Tensor
        >>> w = Tensor(np.random.random((2, 128, 3, 3, 3, 3)).astype(np.float32))
        >>> b = Tensor(np.random.random((2, 128)).astype(np.float32))
        >>> net = Conv3d(
        >>>     in_channels=3, out_channels=128, kernel_size=3, stride=1, padding=1,
        >>>     pad_mode='pad', weight_init=w, bias_init=b, has_bias=True
        >>> )
        >>> u = Tensor(np.random.random((2, 64, 3, 32, 32, 32)).astype(np.float32))
        >>> out = net(u)
        >>> print(out.shape)
        (2, 64, 128, 32, 32, 32)
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _size_3_t,
                 stride: _size_3_t = 1,
                 pad_mode: str = 'same',
                 padding: _size_3_t = 0,
                 dilation: _size_3_t = 1,
                 group: int = 1,
                 has_bias: bool = False,
                 weight_init: Union[Tensor, str, Initializer, numbers.Number] = 'normal',
                 bias_init: Union[Tensor, str, Initializer, numbers.Number] = 'zeros',
                 data_format: str = 'NCDHW') -> None:
        super(Conv3d, self).__init__(HConv3d,
                                     ConvImpl,
                                     in_channels=in_channels,
                                     out_channels=out_channels,
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     pad_mode=pad_mode,
                                     padding=padding,
                                     dilation=dilation,
                                     group=group,
                                     has_bias=has_bias,
                                     weight_init=weight_init,
                                     bias_init=bias_init,
                                     data_format=data_format)


class BatchNorm1d(_UniformOperator):
    r"""
    The dual-valued Batch Normalization layer over a second-order dual input of four dimensions, including
    one spatial dimension, or three dimensions.

    This layer applies Batch Normalization over a dual input to reduce internal covariate shift.
    Batch Normalization is widely used in convolutional networks. It rescales and recenters the feature using
    a mini-batch of data and the learned parameters which can be described by the following formula:

    .. math::
        \begin{align}
        \mathrm{Var}[inp] = \mathrm{E}[\| inp_i - \mathrm{E}[inp] \|^2]\\
        out = \frac{inp - \mathrm{E}[inp]}{\sqrt{\mathrm{Var}[inp] + \delta}}\\
        \text{Re(\hat{out})} = \text{Re(out)} * \text{Re(\gamma)} + \text{Re(\beta)}\\
        \text{Du(\hat{out})} = \text{Re(out)} * \text{Du(\gamma)}
        + \text{Du(out)} * \text{Re(\gamma)} + \text{Du(\beta)},
        \end{align}

    where :math:`inp` is the dual input tensors, :math:`\mathrm{E}[inp]` is the arithmetic mean of the input tensor
    over the batch dimension, :math:`\mathrm{Var}[inp]` is the statistical variance of the input tensor over spatial
    dimensions, based on the dual norm :math:`\|x+\epsilon y\|=\left|\frac{y}{2}\right|+\sqrt{x^2+\frac{y^2}{4}}`,
    :math:`\gamma` and :math:`\beta` are dual learnable parameters representing the scale and shift coefficients
    respectively, and :math:`\delta` is a small positive constant, which is needed to avoid division by zero in case
    statistical variance is close to zero. :math:`\text{Re(...)}` and :math:`\text{Du(...)}` are respectively real
    and dual parts of the dual-valued expression inside the parentheses.

    Args:
        num_features (int): The number of features in the input space.
        eps (float): A small positive threshold, which is needed to avoid division by zero. Default: :math:`10^{-5}`
        momentum (float): A floating hyperparameter of the momentum for the running_mean and running_var computation.
            Default: 0.9.
        affine (bool): A bool value. When set to True, gamma and beta can be learned. Default: True.
        gamma_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the gamma weight.
            The values of str refer to the function `initializer` including 'zeros', 'ones', etc. Default: 'ones'.
        beta_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the beta weight.
            The values of str refer to the function `initializer` including 'zeros', 'ones', etc. Default: 'zeros'.
        moving_mean_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the moving mean.
            The values of str refer to the function `initializer` including 'zeros', 'ones', etc. Default: 'zeros'.
        moving_var_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the moving variance.
            The values of str refer to the function `initializer` including 'zeros', 'ones', etc. Default: 'ones'.
        use_batch_statistics (bool):

            - If True, use the mean value and variance value of current batch data and track running mean
              and running variance.
            - If False, use the mean value and variance value of specified value, and not track statistical value.
            - If None, the use_batch_statistics is automatically set to True or False according to the training
              and evaluation mode. During training, the parameter is set to True, and during evaluation, the
              parameter is set to False. Default: None.

    Inputs:
        - **inp** (Tensor) - Tensor of shape :math:`(2, N, C, W)` or :math:`(2, N, C)`.
          '2' denotes that the input tensor belongs to the dual domain and has got a real and
          a dual parts. The `num_features` in `Args` has to be equal to :math:`C` in `Inputs`.

    Outputs:
        Tensor, the normalized, scaled, offset tensor of the same shape as :math:`u`:
        :math:`(2, N, C, W)` or :math:`(2, N, C)`

    Raises:
        TypeError: If `num_features` is not an int.
        TypeError: If `eps` is not a float.
        ValueError: If `num_features` is less than 1.
        ValueError: If `momentum` is not in range [0, 1].
        ValueError: if 'inp' is not a Tensor of 3 or 4 dimensions.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore.hypercomplex.dual import BatchNorm1d
        >>> from mindspore import Tensor
        >>> u = Tensor(np.random.random((2, 8, 64, 32)).astype(np.float32))
        >>> bn = BatchNorm1d(64)
        >>> y = bn(u)
        >>> print(y.shape)
        (2, 8, 64, 32)
    """

    def __init__(self,
                 num_features: int,
                 eps: float = 1e-5,
                 momentum: float = 0.9,
                 affine: bool = True,
                 gamma_init: Union[Tensor, str, Initializer, numbers.Number] = 'ones',
                 beta_init: Union[Tensor, str, Initializer, numbers.Number] = 'zeros',
                 moving_mean_init: Union[Tensor, str, Initializer, numbers.Number] = 'zeros',
                 moving_var_init: Union[Tensor, str, Initializer, numbers.Number] = 'ones',
                 use_batch_statistics: bool = True) -> None:
        super(BatchNorm1d, self).__init__(HBatchNorm1d,
                                          BatchNormImpl,
                                          num_features=num_features,
                                          eps=eps,
                                          momentum=momentum,
                                          affine=affine,
                                          gamma_init=gamma_init,
                                          beta_init=beta_init,
                                          moving_mean_init=moving_mean_init,
                                          moving_var_init=moving_var_init,
                                          use_batch_statistics=use_batch_statistics)


class BatchNorm2d(_UniformOperator):
    r"""
    The dual-valued Batch Normalization layer  over a second-order dual input of five dimensions, including
    two spatial dimensions.

    This layer applies Batch Normalization over a dual input to reduce internal covariate shift.
    Batch Normalization is widely used in convolutional networks. It rescales and recenters the feature using
    a mini-batch of data and the learned parameters which can be described by the following formula:

    .. math::
        \begin{align}
        \mathrm{Var}[inp] = \mathrm{E}[\| inp_i - \mathrm{E}[inp] \|^2]\\
        out = \frac{inp - \mathrm{E}[inp]}{\sqrt{\mathrm{Var}[inp] + \delta}}\\
        \text{Re(\hat{out})} = \text{Re(out)} * \text{Re(\gamma)} + \text{Re(\beta)}\\
        \text{Du(\hat{out})} = \text{Re(out)} * \text{Du(\gamma)}\
        + \text{Du(out)} * \text{Re(\gamma)} + \text{Du(\beta)},
        \end{align}

    where :math:`inp` is the dual input tensors, :math:`\mathrm{E}[inp]` is the arithmetic mean of the input tensor
    over the batch dimension, :math:`\mathrm{Var}[inp]` is the statistical variance of the input tensor over spatial
    dimensions, based on the dual norm :math:`\|x+\epsilon y\|=\left|\frac{y}{2}\right|+\sqrt{x^2+\frac{y^2}{4}}`,
    :math:`\gamma` and :math:`\beta` are dual learnable parameters representing the scale and shift coefficients
    respectively, and :math:`\delta` is a small positive constant, which is needed to avoid division by zero in case
    statistical variance is close to zero. :math:`\text{Re(...)}` and :math:`\text{Du(...)}` are respectively real
    and dual parts of the dual-valued expression inside the parentheses.

    Args:
        num_features (int): The number of features in the input space.
        eps (float): A small positive threshold, which is needed to avoid division by zero. Default: :math:`10^{-5}`
        momentum (float): A floating hyperparameter of the momentum for the running_mean and running_var computation.
            Default: 0.9.
        affine (bool): A bool value. When set to True, gamma and beta can be learned. Default: True.
        gamma_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the gamma weight.
            The values of str refer to the function `initializer` including 'zeros', 'ones', etc. Default: 'ones'.
        beta_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the beta weight.
            The values of str refer to the function `initializer` including 'zeros', 'ones', etc. Default: 'zeros'.
        moving_mean_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the moving mean.
            The values of str refer to the function `initializer` including 'zeros', 'ones', etc. Default: 'zeros'.
        moving_var_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the moving variance.
            The values of str refer to the function `initializer` including 'zeros', 'ones', etc. Default: 'ones'.
        use_batch_statistics (bool):

            - If True, use the mean value and variance value of current batch data and track running mean
              and running variance.
            - If False, use the mean value and variance value of specified value, and not track statistical value.
            - If None, the use_batch_statistics is automatically set to True or False according to the training
              and evaluation mode. During training, the parameter is set to True, and during evaluation, the
              parameter is set to False. Default: None.
        data_format (str): The optional value for data format, is 'NHWC' or 'NCHW'. Default: 'NCHW'.

    Inputs:
        - **inp** (Tensor) - Tensor of shape :math:`(2, N, C, H, W)` if data_format is 'NCHW', or
          :math:`(2, N, H, W, C)` if data_format is 'NHWC'. '2' denotes that the input tensor belongs to the dual
          domain and has got a real and a dual parts. The `num_features` in `Args` has to be equal to
          :math:`C` in `Inputs`.

    Outputs:
        Tensor, the normalized, scaled, offset tensor of the same shape as :math:`u`:
        :math:`(2, N, C, H, W)` if data_format is 'NCHW', or :math:`(2, N, H, W, C)` if data_format is 'NHWC'.

    Raises:
        TypeError: If `num_features` is not an int.
        TypeError: If `eps` is not a float.
        ValueError: If `num_features` is less than 1.
        ValueError: If `momentum` is not in range [0, 1].
        ValueError: If `data_format` is neither 'NHWC' not 'NCHW'.
        ValueError: if 'inp' is not a Tensor of 5 dimensions.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore.hypercomplex.dual import BatchNorm2d
        >>> from mindspore import Tensor
        >>> u = Tensor(np.random.random((2, 8, 64, 32, 32)).astype(np.float32))
        >>> bn = BatchNorm2d(64)
        >>> y = bn(u)
        >>> print(y.shape)
        (2, 8, 64, 32, 32)
    """

    def __init__(self,
                 num_features: int,
                 eps: float = 1e-5,
                 momentum: float = 0.9,
                 affine: bool = True,
                 gamma_init: Union[Tensor, str, Initializer, numbers.Number] = 'ones',
                 beta_init: Union[Tensor, str, Initializer, numbers.Number] = 'zeros',
                 moving_mean_init: Union[Tensor, str, Initializer, numbers.Number] = 'zeros',
                 moving_var_init: Union[Tensor, str, Initializer, numbers.Number] = 'ones',
                 use_batch_statistics: bool = True,
                 data_format='NCHW') -> None:
        super(BatchNorm2d, self).__init__(HBatchNorm2d,
                                          BatchNormImpl,
                                          num_features=num_features,
                                          eps=eps,
                                          momentum=momentum,
                                          affine=affine,
                                          gamma_init=gamma_init,
                                          beta_init=beta_init,
                                          moving_mean_init=moving_mean_init,
                                          moving_var_init=moving_var_init,
                                          use_batch_statistics=use_batch_statistics,
                                          data_format=data_format)


class BatchNorm3d(_UniformOperator):
    r"""
    The dual-valued Batch Normalization layer over a second-order dual input of six dimensions, including
    three spatial dimensions.

    This layer applies Batch Normalization over a dual input to reduce internal covariate shift.
    Batch Normalization is widely used in convolutional networks. It rescales and recenters the feature using
    a mini-batch of data and the learned parameters which can be described by the following formula:

    .. math::
        \begin{align}
        \mathrm{Var}[inp] = \mathrm{E}[\| inp_i - \mathrm{E}[inp] \|^2]\\
        out = \frac{inp - \mathrm{E}[inp]}{\sqrt{\mathrm{Var}[inp] + \delta}}\\
        \text{Re(\hat{out})} = \text{Re(out)} * \text{Re(\gamma)} + \text{Re(\beta)}\\
        \text{Du(\hat{out})} = \text{Re(out)} * \text{Du(\gamma)}
        + \text{Du(out)} * \text{Re(\gamma)} + \text{Du(\beta)},
        \end{align}

    where :math:`inp` is the dual input tensors, :math:`\mathrm{E}[inp]` is the arithmetic mean of the input tensor
    over the batch dimension, :math:`\mathrm{Var}[inp]` is the statistical variance of the input tensor over spatial
    dimensions, based on the dual norm :math:`\|x+\epsilon y\|=\left|\frac{y}{2}\right|+\sqrt{x^2+\frac{y^2}{4}}`,
    :math:`\gamma` and :math:`\beta` are dual learnable parameters representing the scale and shift coefficients
    respectively, and :math:`\delta` is a small positive constant, which is needed to avoid division by zero in case
    statistical variance is close to zero. :math:`\text{Re(...)}` and :math:`\text{Du(...)}` are respectively real
    and dual parts of the dual-valued expression inside the parentheses.

    Args:
        num_features (int): The number of features in the input space.
        eps (float): A small positive threshold, which is needed to avoid division by zero. Default: :math:`10^{-5}`
        momentum (float): A floating hyperparameter of the momentum for the running_mean and running_var computation.
            Default: 0.9.
        affine (bool): A bool value. When set to True, gamma and beta can be learned. Default: True.
        gamma_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the gamma weight.
            The values of str refer to the function `initializer` including 'zeros', 'ones', etc. Default: 'ones'.
        beta_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the beta weight.
            The values of str refer to the function `initializer` including 'zeros', 'ones', etc. Default: 'zeros'.
        moving_mean_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the moving mean.
            The values of str refer to the function `initializer` including 'zeros', 'ones', etc. Default: 'zeros'.
        moving_var_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the moving variance.
            The values of str refer to the function `initializer` including 'zeros', 'ones', etc. Default: 'ones'.
        use_batch_statistics (bool):

            - If True, use the mean value and variance value of current batch data and track running mean
              and running variance.
            - If False, use the mean value and variance value of specified value, and not track statistical value.
            - If None, the use_batch_statistics is automatically set to True or False according to the training
              and evaluation mode. During training, the parameter is set to True, and during evaluation, the
              parameter is set to False. Default: None.
        data_format (str): The optional value for data format. Only 'NCDHW' format is supported as of now.
            Default: 'NCDHW'.

    Inputs:
        - **inp** (Tensor) - Tensor of shape :math:`(2, N, C, D, H, W)`. '2' denotes that the input tensor belongs
          to the dual domain and has got a real and a dual parts. The `num_features` in `Args` has to be equal
          to :math:`C` in `Inputs`.

    Outputs:
        Tensor, the normalized, scaled, offset tensor of the same shape :math:`(2, N, C, D, H, W)` as :math:`u`.

    Raises:
        TypeError: If `num_features` is not an int.
        TypeError: If `eps` is not a float.
        ValueError: If `num_features` is less than 1.
        ValueError: If `momentum` is not in range [0, 1].
        ValueError: If `data_format` is not 'NCDHW'.
        ValueError: if 'inp' is not a Tensor of 6 dimensions.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore.hypercomplex.dual import BatchNorm3d
        >>> from mindspore import Tensor
        >>> u = Tensor(np.random.random((2, 8, 64, 32, 32, 32)).astype(np.float32))
        >>> bn = BatchNorm3d(64)
        >>> y = bn(u)
        >>> print(y.shape)
        (2, 8, 64, 32, 32, 32)
    """

    def __init__(self,
                 num_features: int,
                 eps: float = 1e-5,
                 momentum: float = 0.9,
                 affine: bool = True,
                 gamma_init: Union[Tensor, str, Initializer, numbers.Number] = 'ones',
                 beta_init: Union[Tensor, str, Initializer, numbers.Number] = 'zeros',
                 moving_mean_init: Union[Tensor, str, Initializer, numbers.Number] = 'zeros',
                 moving_var_init: Union[Tensor, str, Initializer, numbers.Number] = 'ones',
                 use_batch_statistics: bool = True,
                 data_format='NCDHW') -> None:
        super(BatchNorm3d, self).__init__(HBatchNorm3d,
                                          BatchNormImpl,
                                          num_features=num_features,
                                          eps=eps,
                                          momentum=momentum,
                                          affine=affine,
                                          gamma_init=gamma_init,
                                          beta_init=beta_init,
                                          moving_mean_init=moving_mean_init,
                                          moving_var_init=moving_var_init,
                                          use_batch_statistics=use_batch_statistics,
                                          data_format=data_format)


class Dense(_UniformOperator):
    r"""
    The dual-valued dense connected layer.

    Applies dense connected layer for the dual-valued input. This layer implements the operation as:

    .. math::
        \begin{align}
        \text{Re(out)} = \text{Re(inp)} * \text{Re(kernel)} + \text{Re(bias)}\\
        \text{Du(out)} = \text{Re(inp)} * \text{Du(kernel)} + \text{Du(inp)} * \text{Re(kernel)} + \text{Du(bias)},
        \end{align}

    where :math:`inp` is the dual input tensors, :math:`\text{kernel}` is a dual weight matrix with the same
    data type as the :math:`inp` created by the layer, and :math:`\text{bias}` is a dual bias vector with the same
    data type as the :math:`inp` created by the layer (only if has_bias is True). :math:`\text{Re(...)}` and
    :math:`\text{Du(...)}` are respectively real and dual parts of the dual-valued expression inside the parentheses.

    Args:
        in_channels (int): The number of channels in the input space.
        out_channels (int): The number of channels in the output space.
        weight_init (Union[Tensor, str, Initializer, numbers.Number]): The trainable weight_init parameter. The dtype
            is same as `inp`. The values of str refer to the function `initializer`. Default: 'normal'.
        bias_init (Union[Tensor, str, Initializer, numbers.Number]): The trainable bias_init parameter. The dtype is
            same as `inp`. The values of str refer to the function `initializer`. Default: 'zeros'.
        has_bias (bool): Specifies whether the layer uses a bias vector. Default: True.

    Inputs:
        - **inp** (Tensor) - Tensor of shape :math:`(2, *, ..., *, in\_channels)`. '2' denotes that the input tensor
          belongs to the dual domain and has got a real and a dual parts. The `in_channels` in `Args`
          has to be equal to :math:`in\_channels` in `Inputs`. The count of mediator dimensions denoted by '*' is
          arbitrary but must be at least one.

    Outputs:
        Tensor of shape :math:`(2, *, ..., *, out\_channels)`. The count of mediator dimensions is the same as one
        in 'Inputs'.

    Raises:
        TypeError: If `in_channels` or `out_channels` is not an int.
        TypeError: If `has_bias` is not a bool.
        ValueError: If length of shape of `weight_init` is not equal to 3,
                    or shape[0] of 'weight_init' is not equal to 2,
                    or shape[1] of `weight_init` is not equal to `out_channels`,
                    or shape[2] of `weight_init` is not equal to `in_channels`.
        ValueError: If length of shape of `bias_init` is not equal to 2,
                    or shape[0] of 'bias_init' is not equal to 2,
                    or shape[1] of `bias_init` is not equal to `out_channels`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore.hypercomplex.dual import Dense
        >>> from mindspore import Tensor
        >>> w = Tensor(np.random.random((2, 7, 5)).astype(np.float32))
        >>> b = Tensor(np.random.random((2, 7)).astype(np.float32))
        >>> net = Dense(in_channels=5, out_channels=7, weight_init=w, bias_init=b, has_bias=True)
        >>> u = Tensor(np.random.random((2, 34, 1, 5)).astype(np.float32))
        >>> out = net(u)
        >>> print(out.shape)
        (2, 34, 1, 7)
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 weight_init: Union[Tensor, str, Initializer, numbers.Number] = 'normal',
                 bias_init: Union[Tensor, str, Initializer, numbers.Number] = 'zeros',
                 has_bias: bool = True) -> None:
        super(Dense, self).__init__(HDense,
                                    DenseImpl,
                                    in_channels=in_channels,
                                    out_channels=out_channels,
                                    weight_init=weight_init,
                                    bias_init=bias_init,
                                    has_bias=has_bias)
