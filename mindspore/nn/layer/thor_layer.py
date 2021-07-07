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

"""layers for second order optimization"""
import numpy as np
import mindspore.common.dtype as mstype
from mindspore.common.tensor import Tensor
from mindspore.common.initializer import initializer, Initializer
from mindspore.ops import operations as P
from mindspore.common.parameter import Parameter
from mindspore._checkparam import Validator, Rel, twice
from mindspore import context
from mindspore.nn.cell import Cell
from mindspore.nn.layer.activation import get_activation


__all__ = ['DenseThor', 'Conv2dThor', 'EmbeddingThor']


class DenseThor(Cell):
    r"""
    The dense connected layer and saving the information needed for THOR.

    Applies dense connected layer for the input and saves the information A and G in the dense connected layer
    needed for THOR, the detail can be seen in paper: https://www.aaai.org/AAAI21Papers/AAAI-6611.ChenM.pdf
    This layer implements the operation as:

    .. math::
        \text{outputs} = \text{activation}(\text{inputs} * \text{kernel} + \text{bias}),

    where :math:`\text{activation}` is the activation function , :math:`\text{kernel}` is a weight matrix with the same
    data type as the inputs created by the layer, and :math:`\text{bias}` is a bias vector
    with the same data type as the inputs created by the layer (only if has_bias is True).

    Args:
        in_channels (int): The number of the input channels.
        out_channels (int): The number of the output channels.
        weight_init (Union[Tensor, str, Initializer, numbers.Number]): The trainable weight_init parameter. The dtype
            is same as `x`. The values of str refer to the function `initializer`. Default: 'normal'.
        bias_init (Union[Tensor, str, Initializer, numbers.Number]): The trainable bias_init parameter. The dtype is
            same as `x`. The values of str refer to the function `initializer`. Default: 'zeros'.
        has_bias (bool): Specifies whether the layer uses a bias vector. Default: True.
        activation (str): activate function applied to the output of the fully connected layer, eg. 'ReLU'.
            Default: None.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, in\_channels)`.

    Outputs:
        Tensor of shape :math:`(N, out\_channels)`.

    Raises:
        ValueError: If the shape of `weight_init` or `bias_init` is incorrect.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> x = Tensor(np.array([[1, 2, 3], [3, 4, 5]]), mindspore.float32)
        >>> net = nn.DenseThor(3, 4, weight_init="ones")
        >>> output = net(x)
        >>> print(output)
        [[  6.  6.  6.  6.]
         [ 12. 12. 12. 12. ]]
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 weight_init='normal',
                 bias_init='zeros',
                 has_bias=True,
                 activation=None):
        """Initialize DenseThor."""
        super(DenseThor, self).__init__()
        self.thor = True
        self.in_channels = Validator.check_positive_int(in_channels)
        self.out_channels = Validator.check_positive_int(out_channels)
        self.has_bias = Validator.check_bool(has_bias)
        if isinstance(weight_init, Tensor):
            if weight_init.dim() != 2 or weight_init.shape[0] != out_channels or \
                    weight_init.shape[1] != in_channels:
                raise ValueError("Weight init shape error.")
        self.weight = Parameter(initializer(weight_init, [out_channels, in_channels]), name="weight")

        self.bias = None
        if self.has_bias:
            if isinstance(bias_init, Tensor):
                if bias_init.dim() != 1 or bias_init.shape[0] != out_channels:
                    raise ValueError("Bias init shape error.")
            self.bias = Parameter(initializer(bias_init, [out_channels]), name="bias")
            self.bias_add = P.BiasAdd()

        self.matmul = P.MatMul(transpose_b=True)
        self.activation = get_activation(activation)
        self.activation_flag = self.activation is not None

        self.matrix_a = Parameter(Tensor(np.zeros([in_channels, in_channels]).astype(np.float32)),
                                  name='matrix_a', requires_grad=False)
        self.shape = P.Shape()
        self.reshape = P.Reshape()
        self.transpose = P.Transpose()
        self.mul = P.Mul()
        self.is_Ascend = True
        if context.get_context("device_target") == "Ascend":
            self._process_ascend_dense_thor(out_channels)
        else:
            self.is_Ascend = False
            self.matrix_g = Parameter(Tensor(np.eye(out_channels).astype(np.float32)),
                                      name="matrix_g", requires_grad=False)
            self.cube_matmul = P.MatMul(transpose_a=True)
        self.getG = P.InsertGradientOf(self.save_gradient)

    def _process_ascend_dense_thor(self, out_channels):
        """process ascend dense thor"""
        if out_channels == 1001:
            self.matrix_g = Parameter(Tensor(np.zeros([1024, 1024]).astype(np.float32)),
                                      name='matrix_g', requires_grad=False)
            self.pad = P.Pad(((0, 23), (0, 23)))
            self.pad1 = P.Pad(((0, 7), (0, 7)))
            self.slice = P.Slice()
            self.add = P.TensorAdd()
        else:
            self.matrix_g = Parameter(Tensor(np.eye(out_channels).astype(np.float32)),
                                      name="matrix_g", requires_grad=False)
            self.abs = P.Abs()
            self.reduce_max = P.ReduceMax(keep_dims=False)
            self.neg = P.Neg()
            self.reduce_sum = P.ReduceSum()
        self.matmul = P.MatMul(transpose_b=True)
        self.cube_matmul = P.CusMatMulCube(transpose_a=True)
        self.cast = P.Cast()
        self.is_nsp_layer = (out_channels == 2)

    def save_gradient(self, dout):
        """
           this function only for thor optimizer
           save_gradient
        """
        out = dout
        if self.is_Ascend:
            if not self.is_nsp_layer:
                shape = self.shape(dout)
                normalizer = self.cast(shape[0], mstype.float32)
                matrix_g = self.cube_matmul(dout, dout)
                matrix_g = self.mul(matrix_g, 1.0 / normalizer)
                if self.out_channels == 1001:
                    matrix_g = P.Pad(((0, 23), (0, 23)))(matrix_g)
                self.matrix_g = matrix_g
        else:
            dout_shape = self.shape(dout)
            normalizer = dout_shape[0]
            matrix_g = self.cube_matmul(dout, dout)
            matrix_g = self.mul(matrix_g, 1.0 / normalizer)
            self.matrix_g = matrix_g
        return out

    def construct(self, x):
        if self.thor:
            if self.is_Ascend:
                inputs = self.cube_matmul(x, x)
                shape = self.shape(x)
                normalizer = self.cast(shape[0], mstype.float32)
                matrix_a = self.mul(inputs, 1.0 / normalizer)
                self.matrix_a = matrix_a
            else:
                inputs = self.cube_matmul(x, x)
                inputs_shape = self.shape(inputs)
                normalizer = inputs_shape[0]
                matrix_a = self.mul(inputs, 1.0 / normalizer)
                self.matrix_a = matrix_a
            x = self.matmul(x, self.weight)
            x = self.getG(x)
        else:
            x = self.matmul(x, self.weight)
        if self.has_bias:
            x = self.bias_add(x, self.bias)
        if self.activation_flag:
            x = self.activation(x)
        return x

    def extend_repr(self):
        s = 'input_channels={}, output_channels={}'.format(self.in_channels, self.out_channels)
        if self.has_bias:
            s += ', has_bias={}'.format(self.has_bias)
        return s


class _ConvThor(Cell):
    """
    Applies a N-D convolution over an input signal composed of multiple input planes.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, pad_mode,
                 padding, dilation, group, has_bias, weight_init, bias_init, transposed=False):
        """Initialize _ConvThor."""
        super(_ConvThor, self).__init__()
        self.in_channels = Validator.check_positive_int(in_channels)
        self.out_channels = Validator.check_positive_int(out_channels)
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad_mode = pad_mode
        self.bias_init = bias_init
        if isinstance(padding, tuple):
            for pad in padding:
                Validator.check_non_negative_int(pad, 'padding item', self.cls_name)
            self.padding = padding
        elif isinstance(padding, int):
            Validator.check_non_negative_int(padding, 'padding', self.cls_name)
            self.padding = padding
        else:
            raise TypeError("padding type must be int or tuple(int) cannot be {}!".format(type(padding)))

        self.dilation = dilation
        self.group = Validator.check_positive_int(group)
        self.has_bias = has_bias
        self.__validate_kernel_size(kernel_size)
        self.__validate_stride(stride)
        self.__validate_dilation(dilation)
        if in_channels % group != 0:
            raise ValueError("Attr 'in_channels' of 'Conv2DThor' Op must be divisible by "
                             "attr 'group' of 'Conv2DThor' Op.")
        if out_channels % group != 0:
            raise ValueError("Attr 'out_channels' of 'Conv2DThor' Op must be divisible by "
                             "attr 'group' of 'Conv2DThor' Op.")
        if not transposed:
            shape = [out_channels, in_channels // group, *kernel_size]
        else:
            shape = [in_channels, out_channels // group, *kernel_size]
        self.weight = Parameter(initializer(weight_init, shape), name='weight')

        if Validator.check_bool(has_bias):
            self.bias = Parameter(initializer(self.bias_init, [out_channels]), name='bias')
        else:
            if self.bias_init != 'zeros':
                logger.warning("Value of 'has_bias' is False, value of 'bias_init' will be ignored.")
            self.bias = None

    def __validate_kernel_size(self, kernel_size):
        """validate kernel size."""
        if (not isinstance(kernel_size[0], int)) or (not isinstance(kernel_size[1], int)) or \
                isinstance(kernel_size[0], bool) or isinstance(kernel_size[1], bool) or \
                kernel_size[0] < 1 or kernel_size[1] < 1:
            raise ValueError("Attr 'kernel_size' of 'Conv2D' Op passed "
                             + str(self.kernel_size) + ", should be a int or tuple and equal to or greater than 1.")

    def __validate_stride(self, stride):
        """validate stride."""
        if (not isinstance(stride[0], int)) or (not isinstance(stride[1], int)) or \
                isinstance(stride[0], bool) or isinstance(stride[1], bool) or stride[0] < 1 or stride[1] < 1:
            raise ValueError("Attr 'stride' of 'Conv2D' Op passed "
                             + str(self.stride) + ", should be a int or tuple and equal to or greater than 1.")

    def __validate_dilation(self, dilation):
        """validate dilation."""
        if (not isinstance(dilation[0], int)) or (not isinstance(dilation[1], int)) or \
                isinstance(dilation[0], bool) or isinstance(dilation[1], bool) or dilation[0] < 1 or dilation[1] < 1:
            raise ValueError("Attr 'dilation' of 'Conv2D' Op passed "
                             + str(self.dilation) + ", should be a int or tuple and equal to or greater than 1.")


class Conv2dThor(_ConvThor):
    r"""
    2D convolution layer and saving the information needed for THOR.


    Applies a 2D convolution over an input tensor which is typically of shape :math:`(N, C_{in}, H_{in}, W_{in})`,
    where :math:`N` is batch size, :math:`C_{in}` is channel number, and :math:`H_{in}, W_{in})` are height and width.
    And saves the information A and G in the 2D convolution layer needed for THOR.
    The detail can be seen in paper: https://www.aaai.org/AAAI21Papers/AAAI-6611.ChenM.pdf

    For each batch of shape :math:`(C_{in}, H_{in}, W_{in})`, the formula is defined as:


    .. math::

        out_j = \sum_{i=0}^{C_{in} - 1} ccor(W_{ij}, X_i) + b_j,

    where :math:`ccor` is the cross-correlation operator, :math:`C_{in}` is the input channel number, :math:`j` ranges
    from :math:`0` to :math:`C_{out} - 1`, :math:`W_{ij}` corresponds to the :math:`i`-th channel of the :math:`j`-th
    filter and :math:`out_{j}` corresponds to the :math:`j`-th channel of the output. :math:`W_{ij}` is a slice
    of kernel and it has shape :math:`(\text{ks_h}, \text{ks_w})`, where :math:`\text{ks_h}` and
    :math:`\text{ks_w}` are the height and width of the convolution kernel. The full kernel has shape
    :math:`(C_{out}, C_{in} // \text{group}, \text{ks_h}, \text{ks_w})`, where group is the group number
    to split the input `x` in the channel dimension.

    If the 'pad_mode' is set to be "valid", the output height and width will be
    :math:`\left \lfloor{1 + \frac{H_{in} + 2 \times \text{padding} - \text{ks_h} -
    (\text{ks_h} - 1) \times (\text{dilation} - 1) }{\text{stride}}} \right \rfloor` and
    :math:`\left \lfloor{1 + \frac{W_{in} + 2 \times \text{padding} - \text{ks_w} -
    (\text{ks_w} - 1) \times (\text{dilation} - 1) }{\text{stride}}} \right \rfloor` respectively.


    Args:
        in_channels (int): The number of the input channel :math:`C_{in}`.
        out_channels (int): The number of the output channel :math:`C_{out}`.
        kernel_size (Union[int, tuple[int]]): The data type is int or a tuple of 2 integers. Specifies the height
            and width of the 2D convolution window. Single int means that the value is not only the height, but also
            the width of the kernel. A tuple of 2 integers means the height and the width of the kernel respectively.
        stride (Union[int, tuple[int]]): The distance of kernel moving, an int number represents the height and width
             of movement, or a tuple of two int numbers that represent height and width of movement, respectively.
             Default: 1.
        pad_mode (str): Specifies padding mode. The optional values are
            "same", "valid", "pad". Default: "same".

            - same: Adopts the way of completion. The shape of the output will be the same as
              the `x`. The total number of padding will be calculated in horizontal and vertical
              directions and evenly distributed to top and bottom, left and right if possible. Otherwise, the
              last extra padding will be done from the bottom and the right side. If this mode is set, `padding`
              must be 0.

            - valid: Adopts the way of discarding. The possible largest height and width of output will be returned
              without padding. Extra pixels will be discarded. If this mode is set, `padding` must be 0.

            - pad: Implicit paddings on both sides of the input `x`. The number of `padding` will be padded to the input
              Tensor borders. `padding` must be greater than or equal to 0.

        padding (Union[int, tuple[int]]): Implicit paddings on both sides of the input `x`. If `padding` is an integer,
                    the paddings of top, bottom, left and right are the same, equal to padding. If `padding` is a tuple
                    with four integers, the paddings of top, bottom, left and right will be equal to padding[0],
                    padding[1], padding[2], and padding[3] accordingly. Default: 0.
        dilation (Union[int, tuple[int]]): The data type is int or a tuple of 2 integers. Specifies the dilation rate
                                      to use for dilated convolution. If set to be :math:`k > 1`, there will
                                      be :math:`k - 1` pixels skipped for each sampling location. Its value must
                                      be greater or equal to 1 and bounded by the height and width of the  input `x`.
                                      Default: 1.
        group (int): Splits filter into groups, `in_ channels` and `out_channels` must be
            divisible by the number of groups. If the group is equal to `in_channels` and `out_channels`,
            this 2D convolution layer also can be called 2D depthwise convolution layer. Default: 1.
        has_bias (bool): Specifies whether the layer uses a bias vector. Default: False.
        weight_init (Union[Tensor, str, Initializer, numbers.Number]): Initializes the convolution kernel.
            It can be a Tensor, a string, an Initializer or a number. When a string is specified,
            values from 'TruncatedNormal', 'Normal', 'Uniform', 'HeUniform' and 'XavierUniform' distributions as well
            as constant 'One' and 'Zero' distributions are possible. Alias 'xavier_uniform', 'he_uniform', 'ones'
            and 'zeros' are acceptable. Uppercase and lowercase are both acceptable. Refer to the values of
            Initializer for more details. Default: 'normal'.
        bias_init (Union[Tensor, str, Initializer, numbers.Number]): Initializes the bias vector. Possible
            Initializer and string are the same as 'weight_init'. Refer to the values of
            Initializer for more details. Default: 'zeros'.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor of shape :math:`(N, C_{out}, H_{out}, W_{out})`.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> net = nn.Conv2dThor(120, 240, 4, has_bias=False, weight_init='normal')
        >>> x = Tensor(np.ones([1, 120, 1024, 640]), mindspore.float32)
        >>> print(net(x).shape)
        (1, 240, 1024, 640)
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 pad_mode='same', padding=0, dilation=1, group=1, has_bias=False,
                 weight_init='normal', bias_init='zeros'):
        """Initialize Conv2dThor."""
        kernel_size = twice(kernel_size)
        stride = twice(stride)
        self._dilation = dilation
        dilation = twice(dilation)
        super(Conv2dThor, self).__init__(in_channels, out_channels, kernel_size,
                                         stride, pad_mode, padding, dilation, group, has_bias, weight_init, bias_init)
        self.conv2d = P.Conv2D(out_channel=self.out_channels, kernel_size=self.kernel_size,
                               mode=1, pad_mode=self.pad_mode, pad=self.padding,
                               stride=self.stride, dilation=self.dilation, group=self.group)
        self._init_depthwise_conv2d(weight_init)
        self.bias_add = P.BiasAdd()

        self.thor = True
        self.hw = kernel_size[0] * kernel_size[1]
        self.matrix_a_dim = self.in_channels * self.kernel_size[0] * self.kernel_size[1]
        self.matrix_g_dim = self.out_channels
        self.shape = P.Shape()
        self.reshape = P.Reshape()
        self.mul = P.Mul()
        self.cast = P.Cast()
        self.a_normalizer = Parameter(initializer(0, [1], mstype.float32), name="a_normalizer", requires_grad=False)
        self.g_normalizer = Parameter(initializer(0, [1], mstype.float32), name="g_normalizer", requires_grad=False)
        self.is_Ascend = True
        if context.get_context("device_target") == "Ascend":
            self._process_ascend_conv2d_thor(kernel_size, stride)
        else:
            self.is_Ascend = False
            self.img2col = P.Im2Col(kernel_size=kernel_size, stride=stride, pad_mode="same")
            self.matmul = P.MatMul(transpose_b=True)
            self.reduce_mean = P.ReduceMean(keep_dims=False)
            self.matrix_a_cov = Parameter(Tensor(np.zeros([self.matrix_a_dim, self.matrix_a_dim]).astype(np.float32)),
                                          name='matrix_a', requires_grad=False)
            self.matrix_g_cov = Parameter(Tensor(np.zeros([self.matrix_g_dim, self.matrix_g_dim]).astype(np.float32)),
                                          name='matrix_g', requires_grad=False)
        self.getG = P.InsertGradientOf(self.save_gradient)

    def _process_ascend_conv2d_thor(self, kernel_size, stride):
        """process ascend conv2d thor"""
        ksizes = (1, kernel_size[0], kernel_size[1], 1)
        strides = (1, stride[0], stride[1], 1)
        self.img2col = P.CusImg2Col(ksizes=ksizes, strides=strides)
        self.cube_matmul = P.CusMatMulCube(transpose_a=True)
        self.transpose02314 = P.CusTranspose02314()
        dampinga_dim = self.matrix_a_dim
        self.diag_block_dim = 128
        if (self.matrix_a_dim % self.diag_block_dim) != 0 and self.matrix_a_dim > self.diag_block_dim:
            dampinga_dim = (self.matrix_a_dim // self.diag_block_dim + 1) * self.diag_block_dim
        dampingg_dim = self.matrix_g_dim
        if (self.matrix_g_dim % self.diag_block_dim) != 0 and self.matrix_g_dim > self.diag_block_dim:
            dampingg_dim = (self.matrix_g_dim // self.diag_block_dim + 1) * self.diag_block_dim
        self.matrix_a_cov = Parameter(Tensor(np.zeros([dampinga_dim, dampinga_dim]).astype(np.float32)),
                                      name='matrix_a', requires_grad=False)
        self.matrix_g_cov = Parameter(Tensor(np.zeros([dampingg_dim, dampingg_dim]).astype(np.float32)),
                                      name='matrix_g', requires_grad=False)

        self.channels_slice_flag = False
        self.C0 = 16
        if self.in_channels % self.C0 != 0:
            self.channels_slice_flag = True
        self.pada_flag = False
        if (self.matrix_a_dim // self.diag_block_dim) * self.diag_block_dim != self.matrix_a_dim \
                and self.matrix_a_dim > self.diag_block_dim:
            self.pada_flag = True
            pad_dim = self.diag_block_dim - self.matrix_a_dim % self.diag_block_dim
            self.pada = P.Pad(((0, pad_dim), (0, pad_dim)))
        self.slice = P.Slice()

    def _init_depthwise_conv2d(self, weight_init):
        """Initialize depthwise conv2d op"""
        if context.get_context("device_target") == "Ascend" and self.group > 1:
            self.dilation = self._dilation
            Validator.check_integer('group', self.group, self.in_channels, Rel.EQ)
            Validator.check_integer('group', self.group, self.out_channels, Rel.EQ)
            self.conv2d = P.DepthwiseConv2dNative(channel_multiplier=1,
                                                  kernel_size=self.kernel_size,
                                                  pad_mode=self.pad_mode,
                                                  pad=self.padding,
                                                  stride=self.stride,
                                                  dilation=self.dilation)
            weight_shape = [1, self.in_channels, *self.kernel_size]
            self.weight_init = weight_init
            if isinstance(weight_init, Tensor):
                self.weight_init = Tensor(weight_init.asnumpy().swapaxes(0, 1), weight_init.dtype)
            if isinstance(weight_init, Initializer):
                self.weight_init.shape = weight_shape
            self.weight = Parameter(initializer(self.weight_init, weight_shape), name='weight')

    def save_gradient(self, dout):
        """save_gradient"""
        out = dout
        if self.is_Ascend:
            dout = self.transpose02314(dout)
            dout_shape = self.shape(dout)
            normalizer = dout_shape[0]
            matrix_g = self.cube_matmul(dout, dout)
            normalizer = self.cast(normalizer, mstype.float32)
            matrix_g = self.mul(matrix_g, 1.0 / normalizer)
            self.g_normalizer = normalizer
            self.matrix_g_cov = matrix_g
        else:
            dout = self.reduce_mean(dout, 0)
            dout_shape = self.shape(dout)
            dout = self.reshape(dout, (dout_shape[0], -1))
            dout_shape = self.shape(dout)
            normalizer = dout_shape[1]
            dout = self.cast(dout, mstype.float32)
            matrix_g = self.matmul(dout, dout)
            matrix_g = self.mul(matrix_g, 1.0 / normalizer)
            self.g_normalizer = normalizer
            self.matrix_g_cov = matrix_g
        return out

    def construct(self, x):
        if self.thor:
            matrix_a = self.img2col(x)
            matrix_a_shape = self.shape(matrix_a)
            if self.is_Ascend:
                normalizer = matrix_a_shape[0]
                matrix_a = self.cube_matmul(matrix_a, matrix_a)
                if self.channels_slice_flag:
                    matrix_a = self.reshape(matrix_a, (self.hw, self.C0, self.hw, self.C0))
                    matrix_a = self.slice(matrix_a, (0, 0, 0, 0),
                                          (self.hw, self.in_channels, self.hw, self.in_channels))
                    matrix_a = self.reshape(matrix_a, (self.matrix_a_dim, self.matrix_a_dim))
                normalizer = self.cast(normalizer, mstype.float32)
                matrix_a = self.mul(matrix_a, 1.0 / normalizer)
                if self.pada_flag:
                    matrix_a = self.pada(matrix_a)
                self.a_normalizer = normalizer
                self.matrix_a_cov = matrix_a
            else:
                matrix_a = self.reshape(matrix_a, (matrix_a_shape[0] * matrix_a_shape[1] * matrix_a_shape[2],
                                                   matrix_a_shape[3], -1))
                matrix_a = self.reduce_mean(matrix_a, 1)
                matrix_a_shape = self.shape(matrix_a)
                normalizer = matrix_a_shape[1]
                matrix_a = self.cast(matrix_a, mstype.float32)
                matrix_a = self.matmul(matrix_a, matrix_a)
                matrix_a = self.mul(matrix_a, 1.0 / normalizer)
                self.a_normalizer = normalizer
                self.matrix_a_cov = matrix_a
            output = self.conv2d(x, self.weight)
            output = self.getG(output)
        else:
            output = self.conv2d(x, self.weight)
            if self.has_bias:
                output = self.bias_add(output, self.bias)
        return output

    def extend_repr(self):
        s = 'input_channels={}, output_channels={}, kernel_size={}, stride={}, ' \
            'pad_mode={}, padding={}, dilation={}, group={}, has_bias={}, ' \
            'weight_init={}, bias_init={}'.format(self.in_channels, self.out_channels, self.kernel_size,
                                                  self.stride, self.pad_mode, self.padding, self.dilation,
                                                  self.group, self.has_bias, self.weight_init, self.bias_init)
        return s


class EmbeddingThor(Cell):
    r"""
    A simple lookup table that stores embeddings of a fixed dictionary and size
    and saving the information needed for THOR.

    This module is often used to store word embeddings and retrieve them using
    indices. The input to the module is a list of indices, and the output is
    the corresponding word embeddings. And saves the information A and G in the dense connected layer
    needed for THOR, the detail can be seen in paper: https://www.aaai.org/AAAI21Papers/AAAI-6611.ChenM.pdf

    Note:
        When 'use_one_hot' is set to True, the type of the input `x` must be mindspore.int32.

    Args:
        vocab_size (int): The size of the dictionary of embeddings.
        embedding_size (int): The size of each embedding vector.
        use_one_hot (bool): Specifies whether to apply one_hot encoding form. Default: False.
        embedding_table (Union[Tensor, str, Initializer, numbers.Number]): Initializes the embedding_table.
            Refer to class `initializer` for the values of string when a string is specified. Default: 'normal'.
        dtype (:class:`mindspore.dtype`): Data type of input `x`. Default: mindspore.float32.
        padding_idx (int, None): When the padding_idx encounters index, the output embedding vector of this index
                                 will be initialized to zero. Default: None. The feature is inactivated.
    Inputs:
        - **x** (Tensor) - Tensor of input shape :math:`(\text{batch_size}, \text{x_length})`. The elements of
          the Tensor must be integer and not larger than vocab_size. Otherwise the corresponding embedding vector will
          be zero.

    Outputs:
        Tensor of output shape :math:`(\text{batch_size}, \text{x_length}, \text{embedding_size})`.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> net = nn.EmbeddingThor(20000, 768,  True)
        >>> x = Tensor(np.ones([8, 128]), mindspore.int32)
        >>>
        >>> # Maps the input word IDs to word embedding.
        >>> output = net(x)
        >>> output.shape
        (8, 128, 768)
    """

    def __init__(self, vocab_size, embedding_size, use_one_hot=False, embedding_table='normal',
                 dtype=mstype.float32, padding_idx=None):
        """Initialize EmbeddingThor."""
        super(EmbeddingThor, self).__init__()
        self.vocab_size = Validator.check_value_type('vocab_size', vocab_size, [int], self.cls_name)
        self.embedding_size = Validator.check_value_type('embedding_size', embedding_size, [int], self.cls_name)
        Validator.check_value_type('use_one_hot', use_one_hot, [bool], self.cls_name)
        Validator.check_subclass("dtype", dtype, mstype.number_type, self.cls_name)
        self.use_one_hot = use_one_hot
        self.dtype = dtype
        self.init_tensor = initializer(embedding_table, [vocab_size, embedding_size])
        self.padding_idx = padding_idx
        if padding_idx is not None:
            self.padding_idx = Validator.check_int_range(padding_idx, 0, vocab_size, Rel.INC_BOTH,
                                                         "padding_idx", self.cls_name)
            self.init_tensor = self.init_tensor.to_tensor().asnumpy()
            self.init_tensor[self.padding_idx] = 0
        self.embedding_table = Parameter(self.init_tensor, name='embedding_table')
        self.expand = P.ExpandDims()
        self.reshape_flat = P.Reshape()
        self.shp_flat = (-1,)
        self.gather = P.GatherV2()
        self.one_hot = P.OneHot()
        self.on_value = Tensor(1.0, self.dtype)
        self.off_value = Tensor(0.0, self.dtype)
        self.array_mul = P.MatMul()
        self.reshape = P.Reshape()
        self.get_shp = P.Shape()
        self.thor = True
        self.matrix_a = Parameter(Tensor(np.zeros([vocab_size]).astype(np.float32)),
                                  name='matrix_a', requires_grad=False)
        self.matrix_g = Parameter(Tensor(np.zeros([embedding_size, embedding_size]).astype(np.float32)),
                                  name="matrix_g", requires_grad=False)
        self.reduce_sum = P.ReduceSum(keep_dims=False)
        self.getG = P.InsertGradientOf(self.save_gradient)
        self.cast = P.Cast()
        if context.get_context("device_target") == "Ascend":
            self.cube_matmul = P.CusMatMulCube(transpose_a=True)
        else:
            self.cube_matmul = P.MatMul(transpose_a=True)
        self.mul = P.Mul()

    def save_gradient(self, dout):
        """
           this function only for thor optimizer
           save_gradient
        """
        out = dout
        shape = self.get_shp(dout)
        normalizer = self.cast(shape[0], mstype.float32)
        matrix_g = self.cube_matmul(dout, dout)
        matrix_g = self.mul(matrix_g, 1.0 / normalizer)
        self.matrix_g = matrix_g
        return out

    def construct(self, ids):
        extended_ids = self.expand(ids, -1)
        out_shape = self.get_shp(ids) + (self.embedding_size,)
        flat_ids = self.reshape_flat(extended_ids, self.shp_flat)

        if self.use_one_hot:
            one_hot_ids = self.one_hot(flat_ids, self.vocab_size, self.on_value, self.off_value)
            output_for_reshape = self.array_mul(one_hot_ids, self.embedding_table)
        else:
            if self.thor:
                one_hot_ids = self.one_hot(flat_ids, self.vocab_size, self.on_value, self.off_value)
                matrix_a = self.reduce_sum(one_hot_ids, 0)
                self.matrix_a = matrix_a
                output_for_reshape = self.gather(self.embedding_table, flat_ids, 0)
                output_for_reshape = self.getG(output_for_reshape)
            else:
                output_for_reshape = self.gather(self.embedding_table, flat_ids, 0)

        output = self.reshape(output_for_reshape, out_shape)
        return output

    def extend_repr(self):
        s = 'vocab_size={}, embedding_size={}, use_one_hot={}, embedding_table={}, dtype={}, padding_idx={}'.format(
            self.vocab_size, self.embedding_size, self.use_one_hot, self.embedding_table, self.dtype, self.padding_idx)
        return s
