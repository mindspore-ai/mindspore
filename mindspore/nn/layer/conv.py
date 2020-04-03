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
"""conv"""
from mindspore import log as logger
from mindspore.ops import operations as P
from mindspore.common.parameter import Parameter
from mindspore.common.initializer import initializer
from mindspore._checkparam import check_bool, twice, check_int_positive, check_int_non_negative, check_int
from mindspore._extends import cell_attr_register
from ..cell import Cell


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
                 bias_init):
        super(_Conv, self).__init__()
        self.in_channels = check_int_positive(in_channels)
        self.out_channels = check_int_positive(out_channels)
        self.kernel_size = kernel_size
        self.stride = check_int_positive(stride)
        self.pad_mode = pad_mode
        self.padding = check_int_non_negative(padding)
        self.dilation = check_int(dilation)
        self.group = check_int_positive(group)
        self.has_bias = has_bias
        if (not isinstance(kernel_size, tuple)) or len(kernel_size) != 2 or \
            (not isinstance(kernel_size[0], int)) or (not isinstance(kernel_size[1], int)) or \
                kernel_size[0] < 1 or kernel_size[1] < 1:
            raise ValueError("Attr 'kernel_size' of 'Conv2D' Op passed "
                             + str(self.kernel_size) + ", should be a int or tuple and equal to or greater than 1.")
        if in_channels % group != 0:
            raise ValueError("Attr 'in_channels' of 'Conv2D' Op must be divisible by "
                             "attr 'group' of 'Conv2D' Op.")
        if out_channels % group != 0:
            raise ValueError("Attr 'out_channels' of 'Conv2D' Op must be divisible by "
                             "attr 'group' of 'Conv2D' Op.")

        self.weight = Parameter(initializer(weight_init, [out_channels, in_channels // group, *kernel_size]),
                                name='weight')

        if check_bool(has_bias):
            self.bias = Parameter(initializer(bias_init, [out_channels]), name='bias')
        else:
            if bias_init != 'zeros':
                logger.warning("Value of 'has_bias' is False, value of 'bias_init' will be ignored.")
            self.bias = None

    def construct(self, *inputs):
        """Must be overridden by all subclasses."""
        raise NotImplementedError


class Conv2d(_Conv):
    r"""
    2D convolution layer.

    Applies a 2D convolution over an input tensor which is typically of shape :math:`(N, C_{in}, H_{in}, W_{in})`,
    where :math:`N` is batch size and :math:`C_{in}` is channel number. For each batch of shape
    :math:`(C_{in}, H_{in}, W_{in})`, the formula is defined as:

    .. math::

        out_j = \sum_{i=0}^{C_{in} - 1} ccor(W_{ij}, X_i) + b_j,

    where :math:`ccor` is cross correlation operator, :math:`C_{in}` is the input channel number, :math:`j` ranges
    from :math:`0` to :math:`C_{out} - 1`, :math:`W_{ij}` corresponds to :math:`i`-th channel of the :math:`j`-th
    filter and :math:`out_{j}` corresponds to the :math:`j`-th channel of the output. :math:`W_{ij}` is a slice
    of kernel and it has shape :math:`(\text{ks_h}, \text{ks_w})`, where :math:`\text{ks_h}` and
    :math:`\text{ks_w}` are height and width of the convolution kernel. The full kernel has shape
    :math:`(C_{out}, C_{in} // \text{group}, \text{ks_h}, \text{ks_w})`, where group is the group number
    to split the input in the channel dimension.

    If the 'pad_mode' is set to be "valid", the output height and width will be
    :math:`\left \lfloor{1 + \frac{H_{in} + 2 \times \text{padding} - \text{ks_h} -
    (\text{ks_h} - 1) \times (\text{dilation} - 1) }{\text{stride}}} \right \rfloor` and
    :math:`\left \lfloor{1 + \frac{W_{in} + 2 \times \text{padding} - \text{ks_w} -
    (\text{ks_w} - 1) \times (\text{dilation} - 1) }{\text{stride}}} \right \rfloor` respectively.

    The first introduction can be found in paper `Gradient Based Learning Applied to Document Recognition
    <http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf>`_.

    Args:
        in_channels (int): The number of input channel :math:`C_{in}`.
        out_channels (int): The number of output channel :math:`C_{out}`.
        kernel_size (Union[int, tuple]): The data type is int or tuple with 2 integers. Specifies the height
            and width of the 2D convolution window. Single int means the value if for both height and width of
            the kernel. A tuple of 2 ints means the first value is for the height and the other is for the
            width of the kernel.
        stride (int): Specifies stride for all spatial dimensions with the same value. Value of stride should be
            greater or equal to 1 but bounded by the height and width of the input. Default: 1.
        pad_mode (str): Specifies padding mode. The optional values are
            "same", "valid", "pad". Default: "same".

            - same: Adopts the way of completion. Output height and width will be the same as the input.
              Total number of padding will be calculated for horizontal and vertical
              direction and evenly distributed to top and bottom, left and right if possible. Otherwise, the
              last extra padding will be done from the bottom and the right side. If this mode is set, `padding`
              must be 0.

            - valid: Adopts the way of discarding. The possibly largest height and width of output will be return
              without padding. Extra pixels will be discarded. If this mode is set, `padding`
              must be 0.

            - pad: Implicit paddings on both sides of the input. The number of `padding` will be padded to the input
              Tensor borders. `padding` should be greater than or equal to 0.

        padding (int): Implicit paddings on both sides of the input. Default: 0.
        dilation (int): Specifying the dilation rate to use for dilated convolution. If set to be :math:`k > 1`,
            there will be :math:`k - 1` pixels skipped for each sampling location. Its value should be greater
            or equal to 1 and bounded by the height and width of the input. Default: 1.
        group (int): Split filter into groups, `in_ channels` and `out_channels` should be
            divisible by the number of groups. Default: 1.
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

    Returns:
        Tensor, output tensor.

    Inputs:
        - **input** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor of shape :math:`(N, C_{out}, H_{out}, W_{out})`.

    Examples:
        >>> net = nn.Conv2d(120, 240, 4, has_bias=False, weight_init='normal')
        >>> input = mindspore.Tensor(np.ones([1, 120, 1024, 640]), mindspore.float32)
        >>> net(input).shape()
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
                 bias_init='zeros'):
        kernel_size = twice(kernel_size)
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
            bias_init)
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
            raise ValueError('Attr \'pad_mode\' of \'Conv2d\' Op passed '
                             + str(pad_mode) + ', should be one of values in \'valid\', \'same\', \'pad\'.')

    def construct(self, x):
        output = self.conv2d(x, self.weight)
        if self.has_bias:
            output = self.bias_add(output, self.bias)
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
                self.weight,
                self.bias)

        if self.has_bias:
            s += ', bias={}'.format(self.bias)
        return s


class Conv2dTranspose(_Conv):
    r"""
    2D transposed convolution layer.

    Compute a 2D transposed convolution, which is also know as a deconvolution
    (although it is not actual deconvolution).

    Input is typically of shape :math:`(N, C, H, W)`, where :math:`N` is batch size and :math:`C` is channel number.

    Args:
        in_channels (int): The number of channels in the input space.
        out_channels (int): The number of channels in the output space.
        kernel_size (Union[int, tuple]): int or tuple with 2 integers, which specifies the  height
            and width of the 2D convolution window. Single int means the value is for both height and width of
            the kernel. A tuple of 2 ints means the first value is for the height and the other is for the
            width of the kernel.
        stride (int): Specifies the same value for all spatial dimensions. Default: 1.
        pad_mode (str): Select the mode of the pad. The optional values are
            "pad", "same", "valid". Default: "same".

            - pad: Implicit paddings on both sides of the input.

            - same: Adopted the way of completion.

            - valid: Adopted the way of discarding.
        padding (int): Implicit paddings on both sides of the input. Default: 0.
        dilation (int): Specifies the dilation rate to use for dilated
            convolution. Default: 1.
        group (int): Split filter into groups, `in_channels` and `out_channels` should be
            divisible by the number of groups. Default: 1.
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
        - **input** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor of shape :math:`(N, C_{out}, H_{out}, W_{out})`.

    Examples:
        >>> net = nn.Conv2dTranspose(3, 64, 4, has_bias=False, weight_init='normal')
        >>> input = Tensor(np.ones([1, 3, 16, 50]), mstype.float32)
        >>> net(input)
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
        # out_channels and in_channels swap.
        # cause Conv2DBackpropInput's out_channel refers to Conv2D's out_channel,
        # then Conv2dTranspose's out_channel refers to Conv2DBackpropInput's in_channel.
        super(Conv2dTranspose, self).__init__(
            out_channels,
            in_channels,
            kernel_size,
            stride,
            pad_mode,
            padding,
            dilation,
            group,
            has_bias,
            weight_init,
            bias_init)

        self.out_channels = out_channels
        self.in_channels = in_channels
        self.shape = P.Shape()
        if pad_mode not in ('valid', 'same', 'pad'):
            raise ValueError('Attr \'pad_mode\' of \'Conv2dTranspose\' Op passed '
                             + str(pad_mode) + ', should be one of values in \'valid\', \'same\', \'pad\'.')
        self.is_valid = self.pad_mode == 'valid'
        self.is_same = self.pad_mode == 'same'
        self.is_pad = self.pad_mode == 'pad'
        if check_bool(has_bias):
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

    def set_strategy(self, strategy):
        self.conv2d_transpose.set_strategy(strategy)
        return self

    def _deconv_output_length(self, input_length, filter_size):
        """Calculate the width and height of output."""
        length = 0
        if self.is_valid:
            if filter_size - self.stride > 0:
                length = input_length * self.stride + filter_size - self.stride
            else:
                length = input_length * self.stride
        elif self.is_same:
            length = input_length * self.stride
        elif self.is_pad:
            length = input_length * self.stride - 2 * self.padding + filter_size + \
                     (filter_size - 1) * (self.dilation - 1) - self.stride

        return length

    def construct(self, x):
        n, _, h, w = self.shape(x)
        h_out = self._deconv_output_length(h, self.kernel_size[0])
        w_out = self._deconv_output_length(w, self.kernel_size[1])
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
                                                  self.weight,
                                                  self.bias)
        return s
