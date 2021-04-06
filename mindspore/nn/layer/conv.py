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
import numpy as np
from mindspore import log as logger
from mindspore import context
from mindspore.ops import operations as P
from mindspore.ops.primitive import constexpr
from mindspore.common.parameter import Parameter
from mindspore.common.initializer import initializer
from mindspore.common.tensor import Tensor
from mindspore._checkparam import Validator, Rel, twice, _check_3d_int_or_tuple
from mindspore._extends import cell_attr_register
from ..cell import Cell

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
                 transposed=False):
        super(_Conv, self).__init__()
        self.in_channels = Validator.check_positive_int(in_channels)
        self.out_channels = Validator.check_positive_int(out_channels)
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad_mode = pad_mode
        self.weight_init = weight_init
        self.bias_init = bias_init
        self.format = Validator.check_string(data_format, ['NCHW', 'NHWC', 'NCDHW'], 'format', self.cls_name)
        if context.get_context("device_target") != "GPU" and self.format == "NHWC":
            raise ValueError("NHWC format only support in GPU target.")
        if context.get_context("device_target") != "Ascend" and self.format == "NCDHW":
            raise ValueError("NCDHW format only support in Ascend target.")
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
            shape = [out_channels, *kernel_size, in_channels // group] if self.format == "NHWC" else \
                [out_channels, in_channels // group, *kernel_size]
        self.weight = Parameter(initializer(self.weight_init, shape), name='weight')

        if Validator.check_bool(has_bias):
            self.bias = Parameter(initializer(self.bias_init, [out_channels]), name='bias')
        else:
            if self.bias_init != 'zeros':
                logger.warning("Value of 'has_bias' is False, value of 'bias_init' will be ignored.")
            self.bias = None

    def construct(self, *inputs):
        """Must be overridden by all subclasses."""
        raise NotImplementedError


class Conv2d(_Conv):
    r"""
    2D convolution layer.

    Applies a 2D convolution over an input tensor which is typically of shape :math:`(N, C_{in}, H_{in}, W_{in})`,
    where :math:`N` is batch size, :math:`C_{in}` is channel number, and :math:`H_{in}, W_{in})` are height and width.
    For each batch of shape :math:`(C_{in}, H_{in}, W_{in})`, the formula is defined as:

    .. math::

        out_j = \sum_{i=0}^{C_{in} - 1} ccor(W_{ij}, X_i) + b_j,

    where :math:`ccor` is the cross-correlation operator, :math:`C_{in}` is the input channel number, :math:`j` ranges
    from :math:`0` to :math:`C_{out} - 1`, :math:`W_{ij}` corresponds to the :math:`i`-th channel of the :math:`j`-th
    filter and :math:`out_{j}` corresponds to the :math:`j`-th channel of the output. :math:`W_{ij}` is a slice
    of kernel and it has shape :math:`(\text{ks_h}, \text{ks_w})`, where :math:`\text{ks_h}` and
    :math:`\text{ks_w}` are the height and width of the convolution kernel. The full kernel has shape
    :math:`(C_{out}, C_{in} // \text{group}, \text{ks_h}, \text{ks_w})`, where group is the group number
    to split the input in the channel dimension.

    If the 'pad_mode' is set to be "valid", the output height and width will be
    :math:`\left \lfloor{1 + \frac{H_{in} + 2 \times \text{padding} - \text{ks_h} -
    (\text{ks_h} - 1) \times (\text{dilation} - 1) }{\text{stride}}} \right \rfloor`    and
    :math:`\left \lfloor{1 + \frac{W_{in} + 2 \times \text{padding} - \text{ks_w} -
    (\text{ks_w} - 1) \times (\text{dilation} - 1) }{\text{stride}}} \right \rfloor`    respectively.

    The first introduction can be found in paper `Gradient Based Learning Applied to Document Recognition
    <http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf>`_.

    Args:
        in_channels (int): The number of input channel :math:`C_{in}`.
        out_channels (int): The number of output channel :math:`C_{out}`.
        kernel_size (Union[int, tuple[int]]): The data type is int or a tuple of 2 integers. Specifies the height
            and width of the 2D convolution window. Single int means the value is for both the height and the width of
            the kernel. A tuple of 2 ints means the first value is for the height and the other is for the
            width of the kernel.
        stride (Union[int, tuple[int]]): The distance of kernel moving, an int number that represents
            the height and width of movement are both strides, or a tuple of two int numbers that
            represent height and width of movement respectively. Default: 1.
        pad_mode (str): Specifies padding mode. The optional values are
            "same", "valid", "pad". Default: "same".

            - same: Adopts the way of completion. The height and width of the output will be the same as
              the input. The total number of padding will be calculated in horizontal and vertical
              directions and evenly distributed to top and bottom, left and right if possible. Otherwise, the
              last extra padding will be done from the bottom and the right side. If this mode is set, `padding`
              must be 0.

            - valid: Adopts the way of discarding. The possible largest height and width of output will be returned
              without padding. Extra pixels will be discarded. If this mode is set, `padding`
              must be 0.

            - pad: Implicit paddings on both sides of the input. The number of `padding` will be padded to the input
              Tensor borders. `padding` must be greater than or equal to 0.

        padding (Union[int, tuple[int]]): Implicit paddings on both sides of the input. If `padding` is one integer,
                    the paddings of top, bottom, left and right are the same, equal to padding. If `padding` is a tuple
                    with four integers, the paddings of top, bottom, left and right will be equal to padding[0],
                    padding[1], padding[2], and padding[3] accordingly. Default: 0.
        dilation (Union[int, tuple[int]]): The data type is int or a tuple of 2 integers. Specifies the dilation rate
                                      to use for dilated convolution. If set to be :math:`k > 1`, there will
                                      be :math:`k - 1` pixels skipped for each sampling location. Its value must
                                      be greater or equal to 1 and bounded by the height and width of the
                                      input. Default: 1.
        group (int): Splits filter into groups, `in_ channels` and `out_channels` must be
            divisible by the number of groups. If the group is equal to `in_channels` and `out_channels`,
            this 2D convolution layer also can be called 2D depthwise convolution layer. Default: 1.
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
        data_format (str): The optional value for data format, is 'NHWC' or 'NCHW'.
            Default: 'NCHW'.

    Inputs:
        - **input** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})` \
          or :math:`(N, H_{in}, W_{in}, C_{in})`.

    Outputs:
        Tensor of shape :math:`(N, C_{out}, H_{out}, W_{out})` or :math:`(N, H_{out}, W_{out}, C_{out})`.

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
        >>> net = nn.Conv2d(120, 240, 4, has_bias=False, weight_init='normal')
        >>> input = Tensor(np.ones([1, 120, 1024, 640]), mindspore.float32)
        >>> output = net(input).shape
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
                 weight_init='normal',
                 bias_init='zeros',
                 data_format='NCHW'):
        kernel_size = twice(kernel_size)
        stride = twice(stride)
        self._dilation = dilation
        dilation = twice(dilation)
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
            data_format)
        self.conv2d = P.Conv2D(out_channel=self.out_channels,
                               kernel_size=self.kernel_size,
                               mode=1,
                               pad_mode=self.pad_mode,
                               pad=self.padding,
                               stride=self.stride,
                               dilation=self.dilation,
                               group=self.group,
                               data_format=self.format)
        self.bias_add = P.BiasAdd()

    def construct(self, x):
        output = self.conv2d(x, self.weight)
        if self.has_bias:
            output = self.bias_add(output, self.bias)
        return output

    def extend_repr(self):
        s = 'input_channels={}, output_channels={}, kernel_size={},' \
            'stride={},  pad_mode={}, padding={}, dilation={}, ' \
            'group={}, has_bias={}' \
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
                self.format)
        return s


@constexpr
def _check_input_3d(input_shape):
    if len(input_shape) != 3:
        raise ValueError(f"Input should be 3d, but got shape {input_shape}")


class Conv1d(_Conv):
    r"""
    1D convolution layer.

    Applies a 1D convolution over an input tensor which is typically of shape :math:`(N, C_{in}, W_{in})`,
    where :math:`N` is batch size and :math:`C_{in}` is channel number. For each batch of shape
    :math:`(C_{in}, W_{in})`, the formula is defined as:

    .. math::

        out_j = \sum_{i=0}^{C_{in} - 1} ccor(W_{ij}, X_i) + b_j,

    where :math:`ccor` is the cross correlation operator, :math:`C_{in}` is the input channel number, :math:`j` ranges
    from :math:`0` to :math:`C_{out} - 1`, :math:`W_{ij}` corresponds to the :math:`i`-th channel of the :math:`j`-th
    filter and :math:`out_{j}` corresponds to the :math:`j`-th channel of the output. :math:`W_{ij}` is a slice
    of kernel and it has shape :math:`(\text{ks_w})`, where :math:`\text{ks_w}` is the width of the convolution kernel.
    The full kernel has shape :math:`(C_{out}, C_{in} // \text{group}, \text{ks_w})`, where group is the group number
    to split the input in the channel dimension.

    If the 'pad_mode' is set to be "valid", the output width will be
    :math:`\left \lfloor{1 + \frac{W_{in} + 2 \times \text{padding} - \text{ks_w} -
    (\text{ks_w} - 1) \times (\text{dilation} - 1) }{\text{stride}}} \right \rfloor`    respectively.

    The first introduction of convolution layer can be found in paper `Gradient Based Learning Applied to Document
    Recognition <http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf>`_.

    Args:
        in_channels (int): The number of input channel :math:`C_{in}`.
        out_channels (int): The number of output channel :math:`C_{out}`.
        kernel_size (int): The data type is int. Specifies the
            width of the 1D convolution window.
        stride (int): The distance of kernel moving, an int number that represents
            the width of movement. Default: 1.
        pad_mode (str): Specifies padding mode. The optional values are
            "same", "valid", "pad". Default: "same".

            - same: Adopts the way of completion. The output width will be the same as the input.
              The total number of padding will be calculated in the horizontal
              direction and evenly distributed to left and right if possible. Otherwise, the
              last extra padding will be done from the bottom and the right side. If this mode is set, `padding`
              must be 0.

            - valid: Adopts the way of discarding. The possible largest width of the output will be returned
              without padding. Extra pixels will be discarded. If this mode is set, `padding`
              must be 0.

            - pad: Implicit paddings on both sides of the input. The number of `padding` will be padded to the input
              Tensor borders. `padding` must be greater than or equal to 0.

        padding (int): Implicit paddings on both sides of the input. Default: 0.
        dilation (int): The data type is int. Specifies the dilation rate
                                      to use for dilated convolution. If set to be :math:`k > 1`, there will
                                      be :math:`k - 1` pixels skipped for each sampling location. Its value must
                                      be greater or equal to 1 and bounded by the height and width of the
                                      input. Default: 1.
        group (int): Splits filter into groups, `in_ channels` and `out_channels` must be
            divisible by the number of groups. Default: 1.
        has_bias (bool): Specifies whether the layer uses a bias vector. Default: False.
        weight_init (Union[Tensor, str, Initializer, numbers.Number]): An initializer for the convolution kernel.
            It can be a Tensor, a string, an Initializer or a number. When a string is specified,
            values from 'TruncatedNormal', 'Normal', 'Uniform', 'HeUniform' and 'XavierUniform' distributions as well
            as constant 'One' and 'Zero' distributions are possible. Alias 'xavier_uniform', 'he_uniform', 'ones'
            and 'zeros' are acceptable. Uppercase and lowercase are both acceptable. Refer to the values of
            Initializer for more details. Default: 'normal'.
        bias_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the bias vector. Possible
            Initializer and string are the same as 'weight_init'. Refer to the values of
            Initializer for more details. Default: 'zeros'.

    Inputs:
        - **input** (Tensor) - Tensor of shape :math:`(N, C_{in}, W_{in})`.

    Outputs:
        Tensor of shape :math:`(N, C_{out}, W_{out})`.

    Raises:
        TypeError: If `in_channels`, `out_channels`, `kernel_size`, `stride`, `padding` or `dilation` is not an int.
        ValueError: If `in_channels`, `out_channels`, `kernel_size`, `stride` or `dilation` is less than 1.
        ValueError: If `padding` is less than 0.
        ValueError: If `pad_mode` is not one of 'same', 'valid', 'pad'.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> net = nn.Conv1d(120, 240, 4, has_bias=False, weight_init='normal')
        >>> input = Tensor(np.ones([1, 120, 640]), mindspore.float32)
        >>> output = net(input).shape
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
                 weight_init='normal',
                 bias_init='zeros'):

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
            bias_init)
        self.padding = (0, 0, padding, padding)
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

    def construct(self, x):
        x_shape = self.shape(x)
        _check_input_3d(x_shape)
        x = self.expand_dims(x, 2)
        output = self.conv2d(x, self.weight)
        if self.has_bias:
            output = self.bias_add(output, self.bias)

        output = self.squeeze(output)
        return output

    def extend_repr(self):
        s = 'input_channels={}, output_channels={}, kernel_size={},' \
            'stride={},  pad_mode={}, padding={}, dilation={}, ' \
            'group={}, has_bias={},' \
            'weight_init={}, bias_init={}'.format(
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
                self.bias_init)
        return s


@constexpr
def _check_input_5dims(input_shape, op_name):
    if len(input_shape) != 5:
        raise ValueError(f"For {op_name}, input should be 5 dims, but got shape {input_shape}.")


class Conv3d(_Conv):
    r"""
    3D convolution layer.

    Applies a 3D convolution over an input tensor which is typically of shape
    :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})` and output shape
    :math:`(N, C_{out}, D_{out}, H_{out}, W_{out})`. where :math:`N` is batch size. :math:`C` is channel number.
    the formula is defined as:

    .. math::

        \operatorname{out}\left(N_{i}, C_{\text {out}_j}\right)=\operatorname{bias}\left(C_{\text {out}_j}\right)+
        \sum_{k=0}^{C_{in}-1} ccor(\text {weight}\left(C_{\text {out}_j}, k\right),
        \operatorname{input}\left(N_{i}, k\right))

    where :math:`ccor` is the cross-correlation operator.

    If the 'pad_mode' is set to be "valid", the output height and width will be
    :math:`\left \lfloor{1 + \frac{D_{in} + 2 \times \text{padding} - \text{ks_d} -
    (\text{ks_d} - 1) \times (\text{dilation} - 1) }{\text{stride}}} \right \rfloor` and
    :math:`\left \lfloor{1 + \frac{H_{in} + 2 \times \text{padding} - \text{ks_h} -
    (\text{ks_h} - 1) \times (\text{dilation} - 1) }{\text{stride}}} \right \rfloor` and
    :math:`\left \lfloor{1 + \frac{W_{in} + 2 \times \text{padding} - \text{ks_w} -
    (\text{ks_w} - 1) \times (\text{dilation} - 1) }{\text{stride}}} \right \rfloor` respectively.

    Args:
        in_channels (int): The number of input channel :math:`C_{in}`.
        out_channels (int): The number of output channel :math:`C_{out}`.
        kernel_size (Union[int, tuple[int]]): The data type is int or a tuple of 3 integers. Specifies the depth, height
            and width of the 3D convolution window. Single int means the value is for the depth, height and the width of
            the kernel. A tuple of 3 ints means the first value is for the depth, second value is for height and the
            other is for the width of the kernel.
        stride (Union[int, tuple[int]]): The distance of kernel moving, an int number that represents
            the depth, height and width of movement are both strides, or a tuple of three int numbers that
            represent depth, height and width of movement respectively. Default: 1.
        pad_mode (str): Specifies padding mode. The optional values are
            "same", "valid", "pad". Default: "same".

            - same: Adopts the way of completion. The depth, height and width of the output will be the same as
              the input. The total number of padding will be calculated in depth, horizontal and vertical
              directions and evenly distributed to head and tail, top and bottom, left and right if possible.
              Otherwise, the last extra padding will be done from the tail, bottom and the right side.
              If this mode is set, `padding` must be 0.

            - valid: Adopts the way of discarding. The possible largest depth, height and width of output
              will be returned without padding. Extra pixels will be discarded. If this mode is set, `padding`
              must be 0.

            - pad: Implicit paddings on both sides of the input in depth, height, width. The number of `padding` will
              be padded to the input Tensor borders. `padding` must be greater than or equal to 0.

        padding (Union(int, tuple[int])): Implicit paddings on both sides of the input.
            The data type is int or a tuple of 6 integers. Default: 0. If `padding` is an integer,
            the paddings of head, tail, top, bottom, left and right are the same, equal to padding.
            If `paddings` is a tuple of six integers, the padding of head, tail, top, bottom, left and right equal to
            padding[0], padding[1], padding[2], padding[3], padding[4] and padding[5] correspondingly.
        dilation (Union[int, tuple[int]]): The data type is int or a tuple of 3 integers
            : math:`(dilation_d, dilation_h, dilation_w)`. Currently, dilation on depth only supports the case of 1.
            Specifies the dilation rate to use for dilated convolution. If set to be :math:`k > 1`,
            there will be :math:`k - 1` pixels skipped for each sampling location.
            Its value must be greater or equal to 1 and bounded by the height and width of the input. Default: 1.
        group (int): Splits filter into groups, `in_ channels` and `out_channels` must be
            divisible by the number of groups. Default: 1. Only 1 is currently supported.
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
        data_format (str): The optional value for data format. Currently only support "NCDHW".

    Inputs:
        - **input** (Tensor) - Tensor of shape :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})`.
          Currently input data type only support float16 and float32.

    Outputs:
        Tensor, the value that applied 3D convolution. The shape is :math:`(N, C_{out}, D_{out}, H_{out}, W_{out})`.

    Raises:
        TypeError: If `in_channels`, `out_channels` or `group` is not an int.
        TypeError: If `kernel_size`, `stride`, `padding` or `dilation` is neither an int nor a tuple of six.
        ValueError: If `out_channels`, `kernel_size`, `stride` or `dilation` is less than 1.
        ValueError: If `padding` is less than 0.
        ValueError: If `pad_mode` is not one of 'same', 'valid', 'pad'.
        ValueError: If `padding` is a tuple whose length is not equal to 6.
        ValueError: If `pad_mode` is not equal to 'pad' and `padding` is not equal to (0, 0, 0, 0, 0, 0).
        ValueError: If `data_format` is not 'NCDHW'.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> input = Tensor(np.ones([16, 3, 10, 32, 32]), mindspore.float32)
        >>> conv3d = nn.Conv3d(in_channels=3, out_channels=32, kernel_size=(4, 3, 3))
        >>> output = conv3d(input)
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
                 weight_init='normal',
                 bias_init='zeros',
                 data_format='NCDHW'):
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
            data_format)
        self.conv3d = P.Conv3D(out_channel=self.out_channels,
                               kernel_size=self.kernel_size,
                               mode=1,
                               pad_mode=self.pad_mode,
                               pad=self.padding,
                               stride=self.stride,
                               dilation=self.dilation,
                               group=self.group,
                               data_format=self.format)
        self.bias_add = P.BiasAdd(data_format=self.format)
        self.shape = P.Shape()

    def construct(self, x):
        x_shape = self.shape(x)
        _check_input_5dims(x_shape, self.cls_name)
        output = self.conv3d(x, self.weight)
        if self.has_bias:
            output = self.bias_add(output, self.bias)
        return output

    def extend_repr(self):
        s = 'input_channels={}, output_channels={}, kernel_size={},' \
            'stride={},  pad_mode={}, padding={}, dilation={}, ' \
            'group={}, has_bias={}' \
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
                self.format)
        return s


class Conv3dTranspose(_Conv):
    r"""
    Compute a 3D transposed convolution, which is also known as a deconvolution
    (although it is not an actual deconvolution).

    Input is typically of shape :math:`(N, C, D, H, W)`, where :math:`N` is batch size and :math:`C` is channel number.

    If the 'pad_mode' is set to be "pad", the height and width of output are defined as:

    .. math::
        D_{out} = (D_{in} - 1) \times \text{stride_d} - 2 \times \text{padding_d} + \text{dilation_d} \times
        (\text{kernel_size_d} - 1) + \text{output_padding_d} + 1

        H_{out} = (H_{in} - 1) \times \text{stride_h} - 2 \times \text{padding_h} + \text{dilation_h} \times
        (\text{kernel_size_h} - 1) + \text{output_padding_h} + 1

        W_{out} = (W_{in} - 1) \times \text{stride_w} - 2 \times \text{padding_w} + \text{dilation_w} \times
        (\text{kernel_size_w} - 1) + \text{output_padding_w} + 1

    Args:
        in_channels (int): The number of input channel :math:`C_{in}`.
        out_channels (int): The number of output channel :math:`C_{out}`.
        kernel_size (Union[int, tuple[int]]): The kernel size of the 3D convolution.
        stride (Union[int, tuple[int]]): The distance of kernel moving, an int number that represents
            the depth, height and width of movement are both strides, or a tuple of three int numbers that
            represent depth, height and width of movement respectively. Its value must be equal to or greater than 1.
            Default: 1.
        pad_mode (str): Select the mode of the pad. The optional values are
            "pad", "same", "valid". Default: "same".

            - same: Adopts the way of completion. The depth, height and width of the output will be the same as
              the input. The total number of padding will be calculated in depth, horizontal and vertical
              directions and evenly distributed to head and tail, top and bottom, left and right if possible.
              Otherwise, the last extra padding will be done from the tail, bottom and the right side.
              If this mode is set, `padding` and `output_padding` must be 0.

            - valid: Adopts the way of discarding. The possible largest depth, height and width of output
              will be returned without padding. Extra pixels will be discarded. If this mode is set, `padding`
              and `output_padding` must be 0.

            - pad: Implicit paddings on both sides of the input in depth, height, width. The number of `pad` will
              be padded to the input Tensor borders. `padding` must be greater than or equal to 0.

        padding (Union(int, tuple[int])): The pad value to be filled. Default: 0. If `padding` is an integer,
            the paddings of head, tail, top, bottom, left and right are the same, equal to padding.
            If `padding` is a tuple of six integers, the padding of head, tail, top, bottom, left and right equal to
            padding[0], padding[1], padding[2], padding[3], padding[4] and padding[5] correspondingly.
        dilation (Union(int, tuple[int])): The data type is int or a tuple of 3 integers
            : math:`(dilation_d, dilation_h, dilation_w)`. Currently, dilation on depth only supports the case of 1.
            Specifies the dilation rate to use for dilated convolution. If set to be :math:`k > 1`,
            there will be :math:`k - 1` pixels skipped for each sampling location.
            Its value must be greater or equal to 1 and bounded by the height and width of the input. Default: 1.
        group (int): Splits filter into groups, `in_ channels` and `out_channels` must be
            divisible by the number of groups. Default: 1. Only 1 is currently supported.
        output_padding (Union(int, tuple[int])): Add extra size to each dimension of the output. Default: 0.
            Must be greater than or equal to 0.
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
        data_format (str): The optional value for data format. Currently only support 'NCDHW'.

    Inputs:
        - **input** (Tensor) - Tensor of shape :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})`.
          Currently input data type only support float16 and float32.

    Outputs:
        Tensor, the shape is :math:`(N, C_{out}, D_{out}, H_{out}, W_{out})`.

    Supported Platforms:
        ``Ascend``

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

    Examples:
        >>> input = Tensor(np.ones([32, 16, 10, 32, 32]), mindspore.float32)
        >>> conv3d_transpose = nn.Conv3dTranspose(in_channels=16, out_channels=3, kernel_size=(4, 6, 2), pad_mode='pad')
        >>> output = conv3d_transpose(input)
        >>> print(output.shape)
        (32, 3, 13, 37, 33)
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
                 output_padding=0,
                 has_bias=False,
                 weight_init='normal',
                 bias_init='zeros',
                 data_format='NCDHW'):
        kernel_size = _check_3d_int_or_tuple("kernel_size", kernel_size, self.cls_name)
        stride = _check_3d_int_or_tuple("stride", stride, self.cls_name)
        dilation = _check_3d_int_or_tuple("dilation", dilation, self.cls_name)
        Validator.check_value_type('padding', padding, (int, tuple), self.cls_name)
        if isinstance(padding, tuple):
            Validator.check_equal_int(len(padding), 6, 'padding size', self.cls_name)
        output_padding = _check_3d_int_or_tuple("output_padding", output_padding, self.cls_name, greater_zero=False)
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
                                                  output_padding=output_padding,
                                                  data_format=self.format)
        self.bias_add = P.BiasAdd(data_format=self.format)
        self.shape = P.Shape()

    def construct(self, x):
        x_shape = self.shape(x)
        _check_input_5dims(x_shape, self.cls_name)
        output = self.conv3d_transpose(x, self.weight)
        if self.has_bias:
            output = self.bias_add(output, self.bias)
        return output

    def extend_repr(self):
        s = 'input_channels={}, output_channels={}, kernel_size={},' \
            'stride={},  pad_mode={}, padding={}, dilation={}, ' \
            'group={}, has_bias={},' \
            'weight_init={}, bias_init={}'.format(self.in_channels,
                                                  self.out_channels,
                                                  self.kernel_size,
                                                  self.stride,
                                                  self.pad_mode,
                                                  self.padding,
                                                  self.dilation,
                                                  self.group,
                                                  self.has_bias,
                                                  self.weight_init,
                                                  self.bias_init)
        return s


class Conv2dTranspose(_Conv):
    r"""
    2D transposed convolution layer.

    Compute a 2D transposed convolution, which is also known as a deconvolution
    (although it is not an actual deconvolution).

    Input is typically of shape :math:`(N, C, H, W)`, where :math:`N` is batch size and :math:`C` is channel number.

    If the 'pad_mode' is set to be "pad", the height and width of output are defined as:

    .. math::

        H_{out} = (H_{in} - 1) \times \text{stride} - 2 \times \text{padding} + \text{dilation} \times
        (\text{ks_h} - 1) + 1

        W_{out} = (W_{in} - 1) \times \text{stride} - 2 \times \text{padding} + \text{dilation} \times
        (\text{ks_w} - 1) + 1

    where :math:`\text{ks_h}` is the height of the convolution kernel and :math:`\text{ks_w}` is the width
    of the convolution kernel.

    Args:
        in_channels (int): The number of channels in the input space.
        out_channels (int): The number of channels in the output space.
        kernel_size (Union[int, tuple]): int or a tuple of 2 integers, which specifies the height
            and width of the 2D convolution window. Single int means the value is for both the height and the width of
            the kernel. A tuple of 2 ints means the first value is for the height and the other is for the
            width of the kernel.
        stride (Union[int, tuple[int]]): The distance of kernel moving, an int number that represents
            the height and width of movement are both strides, or a tuple of two int numbers that
            represent height and width of movement respectively. Its value must be equal to or greater than 1.
            Default: 1.
        pad_mode (str): Select the mode of the pad. The optional values are
            "pad", "same", "valid". Default: "same".

            - pad: Implicit paddings on both sides of the input.

            - same: Adopted the way of completion.

            - valid: Adopted the way of discarding.
        padding (Union[int, tuple[int]]): Implicit paddings on both sides of the input. If `padding` is one integer,
                    the paddings of top, bottom, left and right are the same, equal to padding. If `padding` is a tuple
                    with four integers, the paddings of top, bottom, left and right will be equal to padding[0],
                    padding[1], padding[2], and padding[3] accordingly. Default: 0.
        dilation (Union[int, tuple[int]]): The data type is int or a tuple of 2 integers. Specifies the dilation rate
                                      to use for dilated convolution. If set to be :math:`k > 1`, there will
                                      be :math:`k - 1` pixels skipped for each sampling location. Its value must
                                      be greater than or equal to 1 and bounded by the height and width of the
                                      input. Default: 1.
        group (int): Splits filter into groups, `in_channels` and `out_channels` must be
            divisible by the number of groups. This does not support for Davinci devices when group > 1. Default: 1.
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

    Inputs:
        - **input** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor of shape :math:`(N, C_{out}, H_{out}, W_{out})`.

    Raises:
        TypeError: If `in_channels`, `out_channels` or `group` is not an int.
        TypeError: If `kernel_size`, `stride`, `padding` or `dilation` is neither an int not a tuple.
        ValueError: If `in_channels`, `out_channels`, `kernel_size`, `stride` or `dilation` is less than 1.
        ValueError: If `padding` is less than 0.
        ValueError: If `pad_mode` is not one of 'same', 'valid', 'pad'.
        ValueError: If `padding` is a tuple whose length is not equal to 4.
        ValueError: If `pad_mode` is not equal to 'pad' and `padding` is not equal to (0, 0, 0, 0).

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> net = nn.Conv2dTranspose(3, 64, 4, has_bias=False, weight_init='normal', pad_mode='pad')
        >>> input = Tensor(np.ones([1, 3, 16, 50]), mindspore.float32)
        >>> output = net(input).shape
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
                 dilation=1,
                 group=1,
                 has_bias=False,
                 weight_init='normal',
                 bias_init='zeros'):
        kernel_size = twice(kernel_size)
        stride = twice(stride)
        dilation = twice(dilation)
        Validator.check_value_type('padding', padding, (int, tuple), self.cls_name)
        if isinstance(padding, tuple):
            Validator.check_equal_int(len(padding), 4, 'padding size', self.cls_name)
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
            transposed=True)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.shape = P.Shape()
        if pad_mode not in ('valid', 'same', 'pad'):
            raise ValueError('Attr \'pad_mode\' of \'Conv2dTranspose\' Op passed '
                             + str(pad_mode) + ', should be one of values in \'valid\', \'same\', \'pad\'.')
        self.is_valid = self.pad_mode == 'valid'
        self.is_same = self.pad_mode == 'same'
        self.is_pad = self.pad_mode == 'pad'
        if Validator.check_bool(has_bias):
            self.bias = Parameter(initializer(bias_init, [out_channels]), name='bias')

        # cause Conv2DBackpropInput's out_channel refers to Conv2D's out_channel.
        self.conv2d_transpose = P.Conv2DBackpropInput(out_channel=in_channels,
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

    def _deconv_output_length(self, input_length, filter_size, stride_size, dilation_size, padding):
        """Calculate the width and height of output."""
        length = 0
        filter_size = filter_size + (filter_size - 1) * (dilation_size - 1)
        if self.is_valid:
            if filter_size - stride_size > 0:
                length = input_length * stride_size + filter_size - stride_size
            else:
                length = input_length * stride_size
        elif self.is_same:
            length = input_length * stride_size
        elif self.is_pad:
            length = input_length * stride_size - padding + filter_size - stride_size

        return length

    def construct(self, x):
        n, _, h, w = self.shape(x)
        h_out = self._deconv_output_length(h, self.kernel_size[0], self.stride[0], self.dilation[0],
                                           self.padding_top + self.padding_bottom)
        w_out = self._deconv_output_length(w, self.kernel_size[1], self.stride[1], self.dilation[1],
                                           self.padding_left + self.padding_right)
        if self.has_bias:
            return self.bias_add(self.conv2d_transpose(x, self.weight, (n, self.out_channels, h_out, w_out)),
                                 self.bias)
        return self.conv2d_transpose(x, self.weight, (n, self.out_channels, h_out, w_out))

    def extend_repr(self):
        s = 'input_channels={}, output_channels={}, kernel_size={},' \
            'stride={},  pad_mode={}, padding={}, dilation={}, ' \
            'group={}, has_bias={},' \
            'weight_init={}, bias_init={}'.format(self.in_channels,
                                                  self.out_channels,
                                                  self.kernel_size,
                                                  self.stride,
                                                  self.pad_mode,
                                                  self.padding,
                                                  self.dilation,
                                                  self.group,
                                                  self.has_bias,
                                                  self.weight_init,
                                                  self.bias_init)
        return s


class Conv1dTranspose(_Conv):
    r"""
    1D transposed convolution layer.

    Compute a 1D transposed convolution, which is also known as a deconvolution
    (although it is not an actual deconvolution).

    Input is typically of shape :math:`(N, C, W)`, where :math:`N` is batch size and :math:`C` is channel number.

    If the 'pad_mode' is set to be "pad", the width of output is defined as:

    .. math::

        W_{out} = (W_{in} - 1) \times \text{stride} - 2 \times \text{padding} + \text{dilation} \times
        (\text{ks_w} - 1) + 1

    where :math:`\text{ks_w}` is the width of the convolution kernel.

    Args:
        in_channels (int): The number of channels in the input space.
        out_channels (int): The number of channels in the output space.
        kernel_size (int): int, which specifies the width of the 1D convolution window.
        stride (int): The distance of kernel moving, an int number that represents
            the width of movement. Default: 1.
        pad_mode (str): Select the mode of the pad. The optional values are
            "pad", "same", "valid". Default: "same".

            - pad: Implicit paddings on both sides of the input.

            - same: Adopted the way of completion.

            - valid: Adopted the way of discarding.
        padding (int): Implicit paddings on both sides of the input. Default: 0.
        dilation (int): The data type is int. Specifies the dilation rate
                                      to use for dilated convolution. If set to be :math:`k > 1`, there will
                                      be :math:`k - 1` pixels skipped for each sampling location. Its value must
                                      be greater or equal to 1 and bounded by the width of the
                                      input. Default: 1.
        group (int): Splits filter into groups, `in_channels` and `out_channels` must be
            divisible by the number of groups. This is not support for Davinci devices when group > 1. Default: 1.
        has_bias (bool): Specifies whether the layer uses a bias vector. Default: False.
        weight_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the convolution kernel.
            It can be a Tensor, a string, an Initializer or a numbers.Number. When a string is specified,
            values from 'TruncatedNormal', 'Normal', 'Uniform', 'HeUniform' and 'XavierUniform' distributions as well
            as constant 'One' and 'Zero' distributions are possible. Alias 'xavier_uniform', 'he_uniform', 'ones'
            and 'zeros' are acceptable. Uppercase and lowercase are both acceptable. Refer to the values of
            Initializer for more details. Default: 'normal'.
        bias_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the bias vector. Possible
            Initializer and string are the same as 'weight_init'. Refer to the values of
            Initializer for more details. Default: 'zeros'.

    Inputs:
        - **input** (Tensor) - Tensor of shape :math:`(N, C_{in}, W_{in})`.

    Outputs:
        Tensor of shape :math:`(N, C_{out}, W_{out})`.

    Raises:
        TypeError: If `in_channels`, `out_channels`, `kernel_size`, `stride`, `padding` or `dilation` is not an int.
        ValueError: If `in_channels`, `out_channels`, `kernel_size`, `stride` or `dilation` is less than 1.
        ValueError: If `padding` is less than 0.
        ValueError: If `pad_mode` is not one of 'same', 'valid', 'pad'.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> net = nn.Conv1dTranspose(3, 64, 4, has_bias=False, weight_init='normal', pad_mode='pad')
        >>> input = Tensor(np.ones([1, 3, 50]), mindspore.float32)
        >>> output = net(input).shape
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
                 weight_init='normal',
                 bias_init='zeros'):
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
            transposed=True)
        self.padding = (0, 0, padding, padding)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.shape = P.Shape()
        if pad_mode not in ('valid', 'same', 'pad'):
            raise ValueError('Attr \'pad_mode\' of \'Conv1dTranspose\' Op passed '
                             + str(pad_mode) + ', should be one of values in \'valid\', \'same\', \'pad\'.')
        self.is_valid = self.pad_mode == 'valid'
        self.is_same = self.pad_mode == 'same'
        self.is_pad = self.pad_mode == 'pad'
        if Validator.check_bool(has_bias):
            self.bias = Parameter(initializer(bias_init, [out_channels]), name='bias')

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

    def _deconv_output_length(self, input_length, filter_size, stride_size, dilation_size, padding):
        """Calculate the width and height of output."""
        length = 0
        filter_size = filter_size + (filter_size - 1) * (dilation_size - 1)
        if self.is_valid:
            if filter_size - stride_size > 0:
                length = input_length * stride_size + filter_size - stride_size
            else:
                length = input_length * stride_size
        elif self.is_same:
            length = input_length * stride_size
        elif self.is_pad:
            length = input_length * stride_size - padding + filter_size - stride_size

        return length

    def construct(self, x):
        x_shape = self.shape(x)
        _check_input_3d(x_shape)
        x = self.expand_dims(x, 2)

        n, _, h, w = self.shape(x)

        h_out = self._deconv_output_length(h, self.kernel_size[0], self.stride[0], self.dilation[0],
                                           self.padding[0] + self.padding[1])
        w_out = self._deconv_output_length(w, self.kernel_size[1], self.stride[1], self.dilation[1],
                                           self.padding[2] + self.padding[3])
        output = self.conv2d_transpose(x, self.weight, (n, self.out_channels, h_out, w_out))
        if self.has_bias:
            output = self.bias_add(output, self.bias)

        output = self.squeeze(output)
        return output

    def extend_repr(self):
        s = 'input_channels={}, output_channels={}, kernel_size={},' \
            'stride={},  pad_mode={}, padding={}, dilation={}, ' \
            'group={}, has_bias={},' \
            'weight_init={}, bias_init={}'.format(self.in_channels,
                                                  self.out_channels,
                                                  self.kernel_size,
                                                  self.stride,
                                                  self.pad_mode,
                                                  self.padding,
                                                  self.dilation,
                                                  self.group,
                                                  self.has_bias,
                                                  self.weight_init,
                                                  self.bias_init)
        return s
