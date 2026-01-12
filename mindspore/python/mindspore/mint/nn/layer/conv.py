# Copyright 2024 Huawei Technologies Co., Ltd
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

from mindspore.ops.auto_generate.gen_ops_prim import (conv1d_ext_op, conv1d_padding_op, conv2d_ext_op,
                                                      conv2d_padding_op, conv3d_ext_op, conv3d_padding_op)
from mindspore.ops.function.nn_func import pad_ext, conv_transpose2d
from mindspore.ops.function.array_func import rank
import mindspore.common.dtype as mstype
from mindspore.common.parameter import Parameter
from mindspore.common.initializer import initializer, HeUniform, Uniform, _calculate_fan_in_and_fan_out
from mindspore import _checkparam as Validator
from mindspore._checkparam import once_sequence, twice_sequence, triple_sequence
from mindspore._extends import cell_attr_register
from mindspore.nn.cell import Cell
from mindspore.ops.functional import isconstant

__all__ = ['Conv2d', 'ConvTranspose2d', 'Conv3d', 'Conv1d']


class _Conv(Cell):
    """
    Applies a N-D convolution over an input signal composed of several input planes.
    """
    @staticmethod
    def _check_channels(in_channels, out_channels, groups):
        in_channels = Validator.check_non_negative_int(in_channels)
        out_channels = Validator.check_non_negative_int(out_channels)
        groups = Validator.check_positive_int(groups)
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups.')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups.')
        return in_channels, out_channels, groups

    @staticmethod
    def _check_padding(padding, stride):
        valid_padding_strings = {'same', 'valid'}
        if isinstance(padding, str):
            if padding not in valid_padding_strings:
                raise ValueError(f"The value of 'padding' must be one of '{valid_padding_strings}', "
                                 f"but got {padding}.")
            if padding == 'same' and any(s != 1 for s in stride):
                raise ValueError("padding='same' is not supported for strided convolutions")

    @staticmethod
    def _check_padding_mode(padding_mode):
        valid_padding_modes = {'zeros', 'reflect', 'replicate', 'circular'}
        if padding_mode not in valid_padding_modes:
            raise ValueError(f"The value of 'padding_mode' must be one of '{valid_padding_modes}', "
                             f"but got {padding_mode}.")

    @staticmethod
    def _check_positive_sequence(name, seq, cls_name):
        for elem in seq:
            Validator.check_positive_int(elem, f'{name} item', cls_name)

    @staticmethod
    def _calc_reversed_padding(padding, dilation, kernel_size):
        if isinstance(padding, str):
            reversed_padding = [0, 0] * len(kernel_size)
            if padding == 'same':
                for d, k, i in zip(dilation, kernel_size, range(len(kernel_size) - 1, -1, -1)):
                    total_padding = d * (k - 1)
                    left_pad = total_padding // 2
                    reversed_padding[2 * i] = left_pad
                    reversed_padding[2 * i + 1] = total_padding - left_pad
            return reversed_padding
        return tuple(x for x in reversed(padding) for _ in range(2))

    @staticmethod
    def _get_weight_shape(in_channels, out_channels, groups, kernel_size, transposed):
        if transposed:
            return [in_channels, out_channels // groups, *kernel_size]
        return [out_channels, in_channels // groups, *kernel_size]

    @staticmethod
    def _create_bias(bias, shape, out_channels, dtype, cls_name):
        if not Validator.check_bool(bias, "bias", cls_name):
            return None
        fan_in, _ = _calculate_fan_in_and_fan_out(shape)
        if fan_in != 0:
            bound = 1 / math.sqrt(fan_in)
            bias_init = Uniform(bound)
        else:
            bias_init = 'zeros'
        return Parameter(initializer(bias_init, [out_channels], dtype=dtype), name='bias')

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 dilation,
                 transposed,
                 output_padding,
                 groups,
                 bias,
                 padding_mode,
                 dtype=mstype.float32):
        """Initialize _Conv."""
        super(_Conv, self).__init__()
        self.padding = padding
        self.stride = stride
        self.transposed = transposed
        self.output_padding = output_padding
        self.padding_mode = padding_mode
        self.kernel_size = kernel_size
        self.dilation = dilation

        self.in_channels, self.out_channels, self.groups = self._check_channels(in_channels, out_channels, groups)
        self._check_padding(self.padding, self.stride)
        self._check_padding_mode(self.padding_mode)
        self._check_positive_sequence('kernel_size', self.kernel_size, self.cls_name)
        self._check_positive_sequence('stride', self.stride, self.cls_name)
        self._check_positive_sequence('dilation', self.dilation, self.cls_name)

        self._reversed_padding = self._calc_reversed_padding(self.padding, self.dilation, self.kernel_size)
        shape = self._get_weight_shape(self.in_channels, self.out_channels, self.groups, self.kernel_size,
                                       self.transposed)
        weight_init = HeUniform(math.sqrt(5))
        self.weight = Parameter(initializer(weight_init, shape, dtype=dtype), name='weight')
        self.bias = self._create_bias(bias, shape, self.out_channels, dtype, self.cls_name)

    def construct(self, *inputs):
        """Must be overridden by all subclasses."""
        raise NotImplementedError

    def extend_repr(self):
        bias = self.bias is not None
        s = 'input_channels={}, output_channels={}, kernel_size={}, ' \
            'stride={}, padding={}, dilation={}, ' \
            'groups={}, bias={}'.format(
                self.in_channels,
                self.out_channels,
                self.kernel_size,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
                bias)
        return s


class Conv1d(_Conv):
    r"""
    1D convolution layer.

    Applies a 1D convolution over an input tensor which is typically of shape :math:`(N, C_{in}, L_{in})`,
    where :math:`N` is batch size, :math:`C` is channel number, :math:`L` is sequence length.

    The output is calculated based on formula:

    .. math::

        \text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) +
        \sum_{k = 0}^{C_{in} - 1} \text{ccor}({\text{weight}(C_{\text{out}_j}, k), \text{input}(N_i, k)})


    where :math:`bias` is the output channel bias, :math:`ccor` is
    the `cross-correlation <https://en.wikipedia.org/wiki/Cross-correlation>`_,
    :math:`weight` is the convolution kernel value and :math:`input` represents the input feature map.

    - :math:`i` corresponds to the batch number, the range is :math:`[0, N-1]`,
      where :math:`N` is the batch size of the input.

    - :math:`j` corresponds to the output channel, the range is :math:`[0, C_{out}-1]`,
      where :math:`C_{out}` is the number of
      output channels, which is also equal to the number of kernels.

    - :math:`k` corresponds to the input channel, the range is :math:`[0, C_{in}-1]`,
      where :math:`C_{in}` is the number of
      input channels, which is also equal to the number of channels in the convolutional kernels.

    Therefore, in the above formula, :math:`{bias}(C_{\text{out}_j})` represents the bias of the :math:`j`-th
    output channel, :math:`{weight}(C_{\text{out}_j}, k)` represents the slice of the :math:`j`-th convolutional
    kernel in the :math:`k`-th channel, and :math:`{input}(N_i, k)` represents the slice of the :math:`k`-th input
    channel in the :math:`i`-th batch of the input feature map.

    The shape of the convolutional kernel is given by :math:`(\text{kernel_size})`,
    where :math:`\text{kernel_size}` is the length of the kernel.
    If we consider the input and output channels as well as the `groups` parameter, the complete kernel shape
    will be :math:`(C_{out}, C_{in} / \text{groups}, \text{kernel_size})`,
    where `groups` is the number of groups dividing `input`'s input channel when applying groups convolution.

    For more details about convolution layer, please refer to `Gradient Based Learning Applied to Document Recognition
    <http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf>`_.

    Args:
        in_channels (int): The channel number of the input tensor of the Conv1d layer.
        out_channels (int): The channel number of the output tensor of the Conv1d layer.
        kernel_size (Union[int, tuple[int], list[int]]): Specifies the length of the 1D convolution kernel.
            The data type is an integer or a tuple/list of one integer.
        stride (Union[int, tuple[int], list[int]], optional): The movement stride of the 1D convolution kernel.
            The data type is an integer or a tuple/list of one integer. Default: ``1`` .
        padding (Union[int, tuple[int], list[int], str], optional): The number of padding
            on the input.
            The data type is an integer or a tuple/list of one integer or string {``"valid"``, ``"same"``}.
            The value should be greater than or equal to 0. Default: ``0`` .

            - ``"same"``: Pad the input around its edges so that the shape of input and output
              are the same when `stride` is set to ``1``.
              The amount of padding to is calculated by the operator internally, If the amount is even, it is
              uniformly distributed around the input, if it is odd, the excess amount goes to the right side.
              If this mode is set, `stride` must be 1.

            - ``"valid"``: No padding is applied to the input, and the output returns the maximum
              possible length. Extra sequence that could not complete a full stride will
              be discarded.

        dilation (Union[int, tuple[int], list[int]], optional): Specifies the dilation
            rate to use for dilated convolution.
            It can be a single int or a tuple/list of 1 integer. 
            Assuming :math:`dilation=(d)`, the convolutional kernel samples the input with a
            spacing of :math:`d-1` elements in the length direction.
            Default: ``1`` .
        groups (int, optional): Splits filter into groups, `in_channels` and `out_channels` must be
            divisible by `groups`. If the groups is equal to `in_channels` and `out_channels`,
            this 1D convolution layer also can be called 1D depthwise convolution layer. Default: ``1`` .
            The following restraints must be met:

            - :math:`(C_{in} \text{ % } \text{groups} == 0)`
            - :math:`(C_{out} \text{ % } \text{groups} == 0)`
            - :math:`(C_{out} >= \text{groups})`
            - :math:`(\text{weight[1]} = C_{in} / \text{groups})`

        bias (bool, optional): Whether the Conv1d layer has a bias parameter. Default: ``True`` .
        padding_mode (str, optional): Specifies the padding mode with a padding value of 0. It can be set to:
            ``"zeros"`` , ``"reflect"`` or ``"replicate"`` . Default: ``"zeros"`` .
        dtype (:class:`mindspore.dtype`, optional): Dtype of Parameters. Default: ``None``, using ``mstype.float32``.

    Variables:
        - **weight** (Tensor) - The weight of the convolution layer, with shape
          :math:`(C_{out}, C_{in} / \text{groups}, \text{kernel_size[0]})`.
        - **bias** (Tensor) - The bias of the convolution layer, with shape
          :math:`(C_{out})`. If bias is False, this will be None.

    Inputs:
        - **input** (Tensor) - Tensor of shape :math:`(N, C_{in}, L_{in})` \
          or :math:`(C_{in}, L_{in})`.

    Outputs:
        Tensor of shape :math:`(N, C_{out}, L_{out})` or :math:`(C_{out}, L_{out})`.

        padding is ``'same'``:

        .. math::
            \begin{array}{ll} \\
                L_{out} = \left \lceil{\frac{L_{in}}{\text{stride}}} \right \rceil \\
            \end{array}

        padding is ``'valid'``:

        .. math::
            \begin{array}{ll} \\
                L_{out} = \left \lfloor{\frac{L_{in} - \text{dilation} \times (\text{kernel_size} - 1) - 1}
                {\text{stride}}} \right \rfloor + 1 \\
            \end{array}

        padding is int or tuple/list:

        .. math::
            \begin{array}{ll} \\
                L_{out} = \left \lfloor{\frac{L_{in} + 2 \times {padding} - \text{dilation} \times
                (\text{kernel_size} - 1) - 1}{\text{stride}}} \right \rfloor + 1 \\
            \end{array}

    Raises:
        ValueError: Args and size of the input feature map should satisfy the output formula to ensure that the size of
            the output feature map is positive; otherwise, an error will be reported.
        RuntimeError: On Ascend, due to the limitation of the L1 cache size of different NPU chip, if input size or
            kernel size is too large, it may trigger an error.
        TypeError: If `in_channels`, `out_channels` or `groups` is not an int.
        TypeError: If `kernel_size`, `stride` or `dilation` is neither an int nor a tuple/list.
        ValueError: If `in_channels`, `out_channels`, `kernel_size`, `stride` or `dilation` is less than 1.
        ValueError: If `padding` is less than 0.
        ValueError: If `padding` is `same` , `stride` is not equal to 1.
        ValueError: The input parameters do not satisfy the convolution output formula.
        ValueError: The `kernel_size` cannot exceed the size of the input feature map.
        ValueError: The value of padding cannot cause the calculation area to exceed the input size.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore
        >>> from mindspore import Tensor, mint
        >>> import numpy as np
        >>> net = mint.nn.Conv1d(120, 240, 4, bias=False)
        >>> x = Tensor(np.ones([1, 120, 1024]), mindspore.float32)
        >>> output = net(x).shape
        >>> print(output)
        (1, 240, 1021)
    """
    @cell_attr_register
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros',
                 dtype=None):
        """Initialize Conv1d."""
        kernel_size_ = once_sequence(kernel_size)
        stride_ = once_sequence(stride)
        padding_ = padding if isinstance(padding, str) else once_sequence(padding)
        dilation_ = once_sequence(dilation)
        if not dtype:
            dtype = mstype.float32
        super(Conv1d, self).__init__(in_channels, out_channels, kernel_size_, stride_, padding_, dilation_, False,
                                     once_sequence(0), groups, bias, padding_mode, dtype)
        if isinstance(padding, str) and padding_mode == "zeros":
            self.conv1d = conv1d_padding_op
        else:
            self.conv1d = conv1d_ext_op


    def construct(self, input):
        if self.padding_mode != "zeros":
            output = self.conv1d(pad_ext(input, self._reversed_padding, mode=self.padding_mode), self.weight,
                                 self.bias, self.stride, (0,), self.dilation, self.groups)
        else:
            output = self.conv1d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return output


class Conv2d(_Conv):
    r"""
    2D convolution layer.

    Applies a 2D convolution over an input signal composed of several input planes which is typically of shape
    :math:`(N, C_{in}, H_{in}, W_{in})`,
    where :math:`N` is batch size, :math:`C` is channel number, :math:`H` is feature height, :math:`W` is feature width.

    Calculate according to the following formula:

    .. math::

        \text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) +
        \sum_{k = 0}^{C_{in} - 1} \text{ccor}({\text{weight}(C_{\text{out}_j}, k), \text{Input}(N_i, k)})

    where :math:`bias` is the output channel bias, :math:`ccor` is
    the `cross-correlation <https://en.wikipedia.org/wiki/Cross-correlation>`_,
    :math:`weight` is the convolution kernel value and :math:`Input` represents the input feature map.

    - :math:`i` corresponds to the batch number, the range is :math:`[0, N-1]`.

    - :math:`j` corresponds to the output channel, the range is :math:`[0, C_{out}-1]`,
      where :math:`C_{out}` is equal to the number of kernels.

    - :math:`k` corresponds to the input channel, the range is :math:`[0, C_{in}-1]`,
      where :math:`C_{in}` is equal to the number of channels in the convolutional kernels.

    Therefore, in the above formula, :math:`{bias}(C_{\text{out}_j})` represents the bias of the :math:`j`-th
    output channel, :math:`{weight}(C_{\text{out}_j}, k)` represents the slice of the :math:`j`-th convolutional
    kernel in the :math:`k`-th channel, and :math:`{Input}(N_i, k)` represents the slice of the :math:`k`-th input
    channel in the :math:`i`-th batch of the input feature map.

    The shape of the convolutional kernel is given by :math:`(\text{kernel_size[0]},\text{kernel_size[1]})`,
    where :math:`\text{kernel_size[0]}`
    and :math:`\text{kernel_size[1]}` are the height and width of the kernel, respectively.
    If we consider the input and output channels as well as the `groups` parameter, the complete kernel shape
    will be :math:`(C_{out}, C_{in} / \text{groups}, \text{kernel_size[0]}, \text{kernel_size[1]})`,
    where `groups` is the number of groups dividing `Input`'s input channel when applying groups convolution.

    For more details about convolution layer, please refer to `Gradient Based Learning Applied to Document Recognition
    <http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf>`_.

    Args:
        in_channels (int): Number of channels in the input image.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (Union[int, tuple[int], list[int]]): Specifies the height and width of the 2D convolution kernel.
            The data type is an integer or a tuple/list of two integers. An integer represents a square convolution 
            kernel. A tuple/list of two integers represents the height
            and width of the convolution kernel respectively.
        stride (Union[int, tuple[int], list[int]], optional): The movement stride of the 2D convolution kernel.
            The data type is an integer or a tuple/list of two integers. An integer represents the movement step size
            in any directions. A tuple/list of two integers represents the movement step size in the height
            and width directions respectively. Default: ``1`` .
        padding (Union[int, tuple[int], list[int], str], optional): The number of padding
            on the height and width directions of the input.
            The data type is an integer or a tuple/list of two integers or string {``"valid"``, ``"same"``}.
            If `padding` is an integer, then `padding_{H}` and `padding_{W}` are all equal to `padding`.
            If `padding` is a tuple/list of 2 integers, then `padding_{H}` and `padding_{W}`
            is equal to `padding[0]` and `padding[1]` respectively.
            The value should be greater than or equal to 0. Default: ``0`` .

            - ``"same"``: Pad the input around its edges so that the shape of input and output
              are the same when `stride` is set to ``1``.
              The amount of padding to is calculated by the operator internally, If the amount is even, it is
              uniformly distributed around the input, if it is odd, the excess amount goes to the right/bottom side.
              If this mode is set, `stride` must be 1.

            - ``"valid"``: No padding is applied to the input, and the output returns the maximum
              possible height and width. Extra pixels that could not complete a full stride will
              be discarded.

        dilation (Union[int, tuple[int], list[int]], optional): Specifies the dilation rate to use
            for dilated convolution.
            It can be a single int or a tuple/list of 2 integers. A single int means the dilation size is the same
            in both the height and width directions. A tuple/list of two ints represents the dilation size in
            the height and width directions, respectively.
            Assuming :math:`dilation=(d0, d1)`, the convolutional kernel samples the input with a
            spacing of :math:`d0-1` elements in the height direction and :math:`d1-1` elements in the width direction.
            The values in the height and width dimensions are in the ranges [1, H] and [1, W], respectively.
            Default: ``1`` .
        groups (int, optional): Splits filter into groups, `in_channels` and `out_channels` must be
            divisible by `groups`. If the groups is equal to `in_channels` and `out_channels`,
            this 2D convolution layer also can be called 2D depthwise convolution layer. Default: ``1`` .
            The following restraints must be met:

            - :math:`(C_{in} \text{ % } \text{groups} == 0)`
            - :math:`(C_{out} \text{ % } \text{groups} == 0)`
            - :math:`(C_{out} >= \text{groups})`
            - :math:`(\text{weight[1]} = C_{in} / \text{groups})`

        bias (bool, optional): Whether the Conv2d layer has a bias parameter. Default: ``True`` .
        padding_mode (str, optional): Specifies the padding mode with a padding value of 0. It can be set to:
            ``"zeros"`` , ``"reflect"`` or ``"replicate"`` . Default: ``"zeros"`` .
        dtype (:class:`mindspore.dtype`, optional): Dtype of Parameters. Default: ``None``, using ``mstype.float32``.

    Variables:
        - **weight** (Tensor) - The weight of the convolution layer, with shape
          :math:`(C_{out}, C_{in} / \text{groups}, \text{kernel_size[0]}, \text{kernel_size[1]})`.
        - **bias** (Tensor) - The bias of the convolution layer, with shape
          :math:`(C_{out})`. If bias is False, this will be None.

    Inputs:
        - **Input** (Tensor) - :math:`(N, C_{in}, H_{in}, W_{in})` or :math:`(C_{in}, H_{in}, W_{in})`.
          When it's an empty Tesnor, backpropagation is currently not supported.

    Outputs:
        - **Output** (Tensor) - :math:`(N, C_{out}, H_{out}, W_{out})` or :math:`(C_{out}, H_{out}, W_{out})`.
  
          - padding is ``'same'``:
    
            .. math::
                \begin{array}{ll} \\
                    H_{out} = \left \lceil{\frac{H_{in}}{\text{stride[0]}}} \right \rceil \\
                    W_{out} = \left \lceil{\frac{W_{in}}{\text{stride[1]}}} \right \rceil \\
                \end{array}
    
          - padding is ``'valid'``:
    
            .. math::
                \begin{array}{ll} \\
                    H_{out} = \left \lfloor{\frac{H_{in} - \text{dilation[0]} \times (\text{kernel_size[0]} - 1) - 1}
                    {\text{stride[0]}}} \right \rfloor + 1 \\
                    W_{out} = \left \lfloor{\frac{W_{in} - \text{dilation[1]} \times (\text{kernel_size[1]} - 1) - 1}
                    {\text{stride[1]}}} \right \rfloor + 1 \\
                \end{array}
    
          - padding is int or tuple/list:
    
            .. math::
                \begin{array}{ll} \\
                    H_{out} = \left \lfloor{\frac{H_{in} + 2 \times padding[0] - \text{dilation[0]} \times
                    (\text{kernel_size[0]} - 1) - 1}{\text{stride[0]}}} \right \rfloor + 1 \\
                    W_{out} = \left \lfloor{\frac{W_{in} + 2 \times padding[1] - \text{dilation[1]} \times
                    (\text{kernel_size[1]} - 1) - 1}{\text{stride[1]}}} \right \rfloor + 1 \\
                \end{array}

    Raises:
        ValueError: Args and size of the input feature map should satisfy the output formula to ensure that the size of
            the output feature map is positive; otherwise, an error will be reported.
        RuntimeError: On Ascend, due to the limitation of the L1 cache size of different NPU chip, if input size or
            kernel size is too large, it may trigger an error.
        ValueError: If `in_channels`, `out_channels`, `kernel_size`, `stride` or `dilation` is less than 1.
        ValueError: If `padding` is less than 0.
        ValueError: If `padding` is `same` , `stride` is not equal to 1.
        ValueError: The input parameters do not satisfy the convolution output formula.
        ValueError: The `kernel_size` cannot exceed the size of the input feature map.
        ValueError: The value of padding cannot cause the calculation area to exceed the input size.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> net = mindspore.mint.nn.Conv2d(120, 240, 4, bias=False)
        >>> x = mindspore.tensor(np.ones([1, 120, 1024, 640]), mindspore.float32)
        >>> output = net(x).shape
        >>> print(output)
        (1, 240, 1021, 637)
    """
    @cell_attr_register
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros',
                 dtype=None):
        """Initialize Conv2d."""
        kernel_size_ = twice_sequence(kernel_size)
        stride_ = twice_sequence(stride)
        padding_ = padding if isinstance(padding, str) else twice_sequence(padding)
        dilation_ = twice_sequence(dilation)
        if not dtype:
            dtype = mstype.float32
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size_, stride_, padding_, dilation_, False,
                                     twice_sequence(0), groups, bias, padding_mode, dtype)
        if isinstance(padding, str) and padding_mode == "zeros":
            self.conv2d = conv2d_padding_op
        else:
            self.conv2d = conv2d_ext_op


    def construct(self, input):
        if self.padding_mode != "zeros":
            output = self.conv2d(pad_ext(input, self._reversed_padding, mode=self.padding_mode), self.weight,
                                 self.bias, self.stride, (0, 0), self.dilation, self.groups)
        else:
            output = self.conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return output


class Conv3d(_Conv):
    r"""
    3D convolution layer.

    Applies a 3D convolution over an input tensor. The input tensor is typically of
    shape :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})`, where :math:`N` is batch size, :math:`C`
    is channel number, :math:`D, H, W` are the depth, height and width of the feature graph, respectively.

    The output is calculated based on formula:

    .. math::

        \text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) +
        \sum_{k = 0}^{C_{in} - 1} \text{ccor}({\text{weight}(C_{\text{out}_j}, k), \text{input}(N_i, k)})

    where :math:`bias` is the output channel bias, :math:`ccor` is
    the `cross-correlation <https://en.wikipedia.org/wiki/Cross-correlation>`_,
    :math:`weight` is the convolution kernel value and :math:`input` represents the input feature map.

    Here are the indices' meanings:

    - :math:`i` corresponds to the batch number, the range is :math:`[0, N-1]`,
      where :math:`N` is the batch size of the input.

    - :math:`j` corresponds to the output channel, the range is :math:`[0, C_{out}-1]`,
      where :math:`C_{out}` is the number of
      output channels, which is also equal to the number of kernels.

    - :math:`k` corresponds to the input channel, the range is :math:`[0, C_{in}-1]`,
      where :math:`C_{in}` is the number of
      input channels, which is also equal to the number of channels in the convolutional kernels.

    Therefore, in the above formula, :math:`{bias}(C_{\text{out}_j})` represents the bias of the :math:`j`-th
    output channel, :math:`{weight}(C_{\text{out}_j}, k)` represents the slice of the :math:`j`-th convolutional
    kernel in the :math:`k`-th channel, and :math:`{input}(N_i, k)` represents the slice of the :math:`k`-th input
    channel in the :math:`i`-th batch of the input feature map.

    The shape of the convolutional kernel is given by
    :math:`(\text{kernel_size[0]},\text{kernel_size[1]},\text{kernel_size[2]})`,
    where :math:`\text{kernel_size[0]}`, :math:`\text{kernel_size[1]}`
    and :math:`\text{kernel_size[2]}` are the depth, height and width of the kernel, respectively.
    If we consider the input and output channels as well as the `groups` parameter, the complete kernel shape
    will be
    :math:`(C_{out}, C_{in} / \text{groups}, \text{kernel_size[0]}, \text{kernel_size[1]}, \text{kernel_size[2]})`,
    where `groups` is the number of groups dividing `input`'s input channel when applying groups convolution.

    For more details about convolution layer, please refer to `Gradient Based Learning Applied to Document Recognition
    <http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf>`_.

    For the detail of limitations of the parameters, please refer to :func:`mindspore.mint.nn.functional.conv3d`.

    .. warning::
        It is only supported on Atlas A2 Training Series Products.

    Args:
        in_channels (int): The channel number of the input tensor of the Conv3d layer.
        out_channels (int): The channel number of the output tensor of the Conv3d layer.
        kernel_size (Union[int, tuple[int], list[int]]): Specifies the depth, height and width of the 3D convolution
            kernel. The data type is an integer or a tuple/list of three integers. An integer represents the depth,
            height and width of the convolution kernel. A tuple/list of three integers represents the depth, height
            and width of the convolution kernel respectively.
        stride (Union[int, tuple[int], list[int]], optional): The movement stride of the 3D convolution kernel.
            The data type is an integer or a tuple/list of three integers. An integer represents the movement step size
            in depth, height and width directions. A tuple/list of three integers represents the movement step size
            in the depth, height and width directions respectively. Default: ``1`` .
        padding (Union[int, tuple[int], list[int], str], optional): The number of padding
            on the depth, height and width directions of the input.
            The data type is an integer or string {``"valid"``, ``"same"``} or a tuple/list of three integers.
            The value should be greater than or equal to 0. Default: ``0`` .

            - ``"same"``: Pad the input around its edges so that the shape of input and output
              are the same when `stride` is set to ``1``.
              The amount of padding is calculated by the operator internally. If the amount is even, it is
              uniformly distributed around the input, if it is odd, the excess amount goes to the right/bottom side.

            - ``"valid"``: No padding is applied to the input, and the output returns the maximum
              possible depth, height and width. Extra pixels that could not complete a full stride will
              be discarded.

        dilation (Union[int, tuple[int], list[int]], optional): Controlling the spacing between the kernel elements.
            Default: ``1`` .
        groups (int, optional): Splits filter into groups, `in_channels` and `out_channels` must be
            divisible by `groups`. If the groups is equal to `in_channels` and `out_channels`,
            this 3D convolution layer also can be called 3D depthwise convolution layer. Default: ``1`` .
            The following restraints must be met:

            - :math:`(C_{in} \text{ % } \text{groups} == 0)`
            - :math:`(C_{out} \text{ % } \text{groups} == 0)`
            - :math:`(C_{out} >= \text{groups})`
            - :math:`(\text{weight[1]} = C_{in} / \text{groups})`

        bias (bool, optional): Whether the Conv3d layer has a bias parameter. Default: ``True`` .
        padding_mode (str, optional): Specifies the padding mode, the padding value is 0. It can be set to:
            ``"zeros"`` , ``"reflect"`` or ``"replicate"`` . Default: ``"zeros"`` .
        dtype (:class:`mindspore.dtype`, optional): Dtype of Parameters. Default: ``None``, using ``mstype.float32``.

    Variables:
        - **weight** (Tensor) - The weight of the convolution layer, with shape
          :math:`(C_{out}, C_{in} / \text{groups}, \text{kernel_size[0]},
          \text{kernel_size[1]}, \text{kernel_size[2]})`.
        - **bias** (Tensor) - The bias of the convolution layer, with shape
          :math:`(C_{out})`. If `bias` is ``False``, this will be ``None``.

    Inputs:
        - **input** (Tensor) - Tensor of shape :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})` \
          or :math:`(C_{in}, D_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor of shape :math:`(N, C_{out}, D_{out}, H_{out}, W_{out})`
        or :math:`(C_{out}, D_{out}, H_{out}, W_{out})`.

        padding is ``"same"``:

        .. math::
            \begin{array}{ll} \\
                D_{out} = \left \lceil{\frac{D_{in}}{\text{stride[0]}}} \right \rceil \\
                H_{out} = \left \lceil{\frac{H_{in}}{\text{stride[1]}}} \right \rceil \\
                W_{out} = \left \lceil{\frac{W_{in}}{\text{stride[2]}}} \right \rceil \\
            \end{array}

        padding is ``"valid"``:

        .. math::
            \begin{array}{ll} \\
                D_{out} = \left \lfloor{\frac{D_{in} - \text{dilation[0]} \times (\text{kernel_size[0]} - 1) - 1}
                {\text{stride[0]}}} \right \rfloor + 1 \\
                H_{out} = \left \lfloor{\frac{H_{in} - \text{dilation[1]} \times (\text{kernel_size[1]} - 1) - 1}
                {\text{stride[1]}}} \right \rfloor + 1 \\
                W_{out} = \left \lfloor{\frac{W_{in} - \text{dilation[2]} \times (\text{kernel_size[2]} - 1) - 1}
                {\text{stride[2]}}} \right \rfloor + 1 \\
            \end{array}

        padding is int or tuple/list:

        .. math::
            \begin{array}{ll} \\
                D_{out} = \left \lfloor{\frac{D_{in} + 2 \times padding[0] - \text{dilation[0]} \times
                (\text{kernel_size[0]} - 1) - 1}{\text{stride[0]}} + 1} \right \rfloor \\
                H_{out} = \left \lfloor{\frac{H_{in} + 2 \times padding[1] - \text{dilation[1]} \times
                (\text{kernel_size[1]} - 1) - 1}{\text{stride[1]}} + 1} \right \rfloor \\
                W_{out} = \left \lfloor{\frac{W_{in} + 2 \times padding[2] - \text{dilation[2]} \times
                (\text{kernel_size[2]} - 1) - 1}{\text{stride[2]}} + 1} \right \rfloor \\
            \end{array}

    Raises:
        ValueError: If `in_channels`, `out_channels`, `kernel_size`, `stride` or `dilation` is less than 1.
        ValueError: If `padding` is less than 0.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> net = mindspore.mint.nn.Conv3d(120, 10, 4)
        >>> x = mindspore.Tensor(np.ones([1, 120, 10, 23, 34]), mindspore.float32)
        >>> output = net(x).shape
        >>> print(output)
        (1, 10, 7, 20, 31)
    """
    @cell_attr_register
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros',
                 dtype=None):
        """Initialize Conv3d."""
        kernel_size_ = triple_sequence(kernel_size)
        stride_ = triple_sequence(stride)
        padding_ = padding if isinstance(padding, str) else triple_sequence(padding)
        dilation_ = triple_sequence(dilation)
        if not dtype:
            dtype = mstype.float32
        super(Conv3d, self).__init__(in_channels, out_channels, kernel_size_, stride_, padding_, dilation_, False,
                                     triple_sequence(0), groups, bias, padding_mode, dtype)
        if isinstance(padding, str) and padding_mode == "zeros":
            self.conv3d = conv3d_padding_op
        else:
            self.conv3d = conv3d_ext_op


    def construct(self, input):
        if self.padding_mode != "zeros":
            output = self.conv3d(pad_ext(input, self._reversed_padding, mode=self.padding_mode), self.weight,
                                 self.bias, self.stride, (0, 0, 0), self.dilation, self.groups)
        else:
            output = self.conv3d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return output


def batchify(input, num_spatial_dims, ops_name):
    """Conv input batchify"""
    dim_count_no_batch = num_spatial_dims + 1
    dim_count_batch = dim_count_no_batch + 1
    is_batched = (input.ndim == dim_count_batch)
    if not (input.ndim == dim_count_no_batch or is_batched):
        raise TypeError(f"For {ops_name}, Expected {dim_count_no_batch}D (unbatched) or {dim_count_batch}D (batched)," \
                        f"but got input of ndim: {input.ndim}D")
    if is_batched:
        return input, is_batched
    return input.unsqueeze(0), is_batched


class _ConvTranspose(_Conv):
    """
    Applies a N-D convolution over an input signal composed of several input planes.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding, groups,
                 bias, padding_mode, dtype=None):
        if padding_mode != "zeros":
            raise ValueError(
                f'Only "zeros" padding mode is supported for {self.__class__.__name__}'
            )
        super(_ConvTranspose, self).__init__(in_channels, out_channels, kernel_size,
                                             stride, padding, dilation, transposed,
                                             output_padding, groups, bias, padding_mode, dtype)

    def _check_output_size(self, output_size, min_sizes, max_sizes, input_shape):
        if isconstant(output_size) and isconstant(min_sizes)\
            and isconstant(max_sizes) and isconstant(input_shape):
            for i in range(len(output_size)):
                size = output_size[i]
                min_size = min_sizes[i]
                max_size = max_sizes[i]
                if size < min_size or size > max_size:
                    raise ValueError(
                        f"requested an output size of {output_size}, but valid sizes range "
                        f"from {min_sizes} to {max_sizes} (for an input of {input_shape})"
                    )

    # dilation being an optional parameter is for backwards
    # compatibility
    def _output_padding(self, input, output_size, stride, padding, kernel_size,
                        num_spatial_dims, dilation):
        "the computation of output padding"
        if output_size is None:
            ret = tuple(self.output_padding)  # converting to list if was not already
        else:
            input_rank = rank(input)
            has_batch_dim = input_rank == (num_spatial_dims + 2)
            num_non_spatial_dims = 2 if has_batch_dim else 1
            if isconstant(output_size) and isconstant(input_rank) and\
                len(output_size) != num_spatial_dims and len(output_size) != (num_non_spatial_dims + num_spatial_dims):
                raise ValueError(
                    f"ConvTranspose{num_spatial_dims}D: for {input_rank}D input, ",
                    f"output_size must have {num_spatial_dims} ",
                    f"or {num_non_spatial_dims + num_spatial_dims} elements (got {len(output_size)})"
                )
            output_size = output_size[-num_spatial_dims:]

            min_sizes = []
            max_sizes = []
            for d in range(num_spatial_dims):
                dim_size = (
                    (input.shape[d + num_non_spatial_dims] - 1) * stride[d]
                    - 2 * padding[d]
                    + (dilation[d] if dilation is not None else 1)
                    * (kernel_size[d] - 1)
                    + 1
                )
                min_sizes.append(dim_size)
                max_sizes.append(min_sizes[d] + stride[d] - 1)
            self._check_output_size(output_size, min_sizes, max_sizes, input.shape)

            res = []
            for d in range(num_spatial_dims):
                res.append(output_size[d] - min_sizes[d])
            ret = res
        return ret

    def construct(self, *inputs):
        """Must be overridden by all subclasses."""
        raise NotImplementedError


def _pair(x, arg_name, class_name):
    if isinstance(x, int):
        return (x, x)
    if isinstance(x, (tuple, list)):
        if len(x) == 1:
            return (x[0], x[-1])
        return x
    raise ValueError(f"For '{class_name}', '{arg_name}'",
                     f" should be int, tuple or list, but got {x}")


class ConvTranspose2d(_ConvTranspose):
    r"""
    Applies a 2D transposed convolution operator over an input image
    composed of several input planes.

    This module can be seen as the gradient of Conv2d with respect to its input.
    It is also known as a fractionally-strided convolution or
    a deconvolution (although it is not an actual deconvolution operation as it does
    not compute a true inverse of convolution).

    The parameters `kernel_size`, `stride`, `padding`, `output_padding` can either be:

    - a single ``int`` -- in which case the same value is used for the height and width dimensions
    - a ``tuple`` of two ints -- in which case, the first `int` is used for the height dimension,
      and the second `int` for the width dimension

    .. warning::
        - This is an experimental API that is subject to change or deletion.
        - In the scenario where inputs are non-contiguous, `output_padding` must be less than `stride` .
        - For Atlas training products, when the dtype of input is float32, the `groups` only supports 1.

    Args:
        in_channels (int): Number of channels in the input image.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (Union[int, tuple(int)]): Size of the convolving kernel.
        stride (Union[int, tuple(int)], optional): Stride of the convolution. Default ``1``.
        padding (Union[int, tuple(int)], optional): :math:`dilation * (kernel\_size - 1) - padding` zero-padding
            will be added to both sides of each dimension in the input. Default ``0``.
        output_padding (Union[int, tuple(int)], optional): Additional size added to one side of each dimension
            in the output shape. The value of `output_padding` must be less than `stride` or `dilation`.
            Default ``0``.
        groups (int, optional): Number of blocked connections from input channels to output channels. Default ``1``
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default ``True``.
        dilation (Union[int, tuple(int)], optional): Spacing between kernel elements. Default ``1``.
        padding_mode (str, optional): Specifies the padding mode. For now, it can only be set to: ``"zeros"``. 
            Default ``"zeros"``.
        dtype (mindspore.dtype, optional): Dtype of Parameters. Default ``None``. when it's ``None``,
            the dtype of Parameters would be mstype.float32.

    Variables:
        - **weight** (Parameter) - the learnable weights of the module of shape
          :math:`(\text{in_channels}, \frac{\text{out_channels}}{\text{groups}},
          \text{kernel_size[0]}, \text{kernel_size[1]})`. The values of these weights are sampled from
          :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
          :math:`k = \frac{groups}{C_\text{out} * \prod_{i=0}^{1}\text{kernel_size}[i]}` .
        - **bias** (Parameter) - the learnable bias of the module of shape :math:`(\text{out_channels},)`.
          If :attr:`bias` is ``True``, then the values of these weights are sampled from
          :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
          :math:`k = \frac{groups}{C_\text{out} * \prod_{i=0}^{1}\text{kernel_size}[i]}`.

    Inputs:
        - **input** (Tensor) - The input tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`
          or :math:`(C_{in}, H_{in}, W_{in})` .

    Outputs:
        - **output** (Tensor) - The output tensor of shape :math:`(N, C_{out}, H_{out}, W_{out})`
          or :math:`(C_{out}, H_{out}, W_{out})`, where

          .. math::
                H_{out} = (H_{in} - 1) \times \text{stride}[0] - 2 \times \text{padding}[0] + \text{dilation}[0]
                          \times (\text{kernel_size}[0] - 1) + \text{output_padding}[0] + 1
          .. math::
                W_{out} = (W_{in} - 1) \times \text{stride}[1] - 2 \times \text{padding}[1] + \text{dilation}[1]
                          \times (\text{kernel_size}[1] - 1) + \text{output_padding}[1] + 1

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore
        >>> # With square kernels and equal stride
        >>> m = mindspore.mint.nn.ConvTranspose2d(16, 33, 3, stride=2)
        >>> # non-square kernels and unequal stride and with padding
        >>> m = mindspore.mint.nn.ConvTranspose2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
        >>> input = mindspore.mint.randn(20, 16, 50, 100)
        >>> output = m(input)
        >>> # exact output size can be also specified as an argument
        >>> input = mindspore.mint.randn(1, 16, 12, 12)
        >>> downsample = mindspore.mint.nn.Conv2d(16, 16, 3, stride=2, padding=1)
        >>> upsample = mindspore.mint.nn.ConvTranspose2d(16, 16, 3, stride=2, padding=1)
        >>> h = downsample(input)
        >>> h.shape
        (1, 16, 6, 6)
        >>> output = upsample(h, output_size=input.shape)
        >>> output.shape
        (1, 16, 12, 12)

    .. _`here`:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md

    .. _`Deconvolutional Networks`:
        https://www.matthewzeiler.com/mattzeiler/deconvolutionalnetworks.pdf
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0,
                 groups=1, bias=True, dilation=1, padding_mode="zeros", dtype=None):
        dtype = mstype.float32 if dtype is None else dtype
        kernel_size = _pair(kernel_size, "kernel_size", "ConvTranspose2d")
        stride = _pair(stride, "kernel_size", "ConvTranspose2d")
        padding = _pair(padding, "kernel_size", "ConvTranspose2d")
        dilation = _pair(dilation, "kernel_size", "ConvTranspose2d")
        output_padding = _pair(output_padding, "kernel_size", "ConvTranspose2d")
        super(ConvTranspose2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            True,
            output_padding,
            groups,
            bias,
            padding_mode,
            dtype
        )

    def construct(self, input, output_size=None):
        num_spatial_dims = 2
        output_padding = self._output_padding(
            input,
            output_size,
            self.stride,  # type: ignore[arg-type]
            self.padding,  # type: ignore[arg-type]
            self.kernel_size,  # type: ignore[arg-type]
            num_spatial_dims,
            self.dilation,  # type: ignore[arg-type]
        )

        return conv_transpose2d(
            input,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            output_padding,
            self.groups,
            self.dilation,
        )
