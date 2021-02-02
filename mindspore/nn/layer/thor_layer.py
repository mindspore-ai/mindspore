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


__all__ = ['Dense_Thor', 'Conv2d_Thor', 'Embedding_Thor']

class Dense_Thor(Cell):
    r"""
    The dense connected layer.

    Applies dense connected layer for the input. This layer implements the operation as:

    .. math::
        \text{outputs} = \text{activation}(\text{inputs} * \text{kernel} + \text{bias}),

    where :math:`\text{activation}` is the activation function passed as the activation
    argument (if passed in), :math:`\text{kernel}` is a weight matrix with the same
    data type as the inputs created by the layer, and :math:`\text{bias}` is a bias vector
    with the same data type as the inputs created by the layer (only if has_bias is True).

    Args:
        in_channels (int): The number of channels in the input space.
        out_channels (int): The number of channels in the output space.
        weight_init (Union[Tensor, str, Initializer, numbers.Number]): The trainable weight_init parameter. The dtype
            is same as input x. The values of str refer to the function `initializer`. Default: 'normal'.
        bias_init (Union[Tensor, str, Initializer, numbers.Number]): The trainable bias_init parameter. The dtype is
            same as input x. The values of str refer to the function `initializer`. Default: 'zeros'.
        has_bias (bool): Specifies whether the layer uses a bias vector. Default: True.
        activation (str): activate function applied to the output of the fully connected layer, eg. 'ReLU'.
            Default: None.

    Raises:
        ValueError: If weight_init or bias_init shape is incorrect.

    Inputs:
        - **input** (Tensor) - Tensor of shape :math:`(N, in\_channels)`.

    Outputs:
        Tensor of shape :math:`(N, out\_channels)`.

    Examples:
        >>> input = Tensor(np.random.randint(0, 255, [2, 3]), mindspore.float32)
        >>> net = nn.Dense(3, 4)
        >>> net(input)
        [[ 2.5246444   2.2738023   0.5711005  -3.9399147 ]
         [ 1.0739875   4.0155234   0.94188046 -5.459526  ]]
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 weight_init='normal',
                 bias_init='zeros',
                 has_bias=True,
                 activation=None):
        super(Dense_Thor, self).__init__()
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

        self.matrix_A = Parameter(Tensor(np.zeros([in_channels, in_channels]).astype(np.float32)),
                                  name='matrix_A', requires_grad=False)
        self.shape = P.Shape()
        self.reshape = P.Reshape()
        self.transpose = P.Transpose()
        self.mul = P.Mul()
        self.is_Ascend = True
        if context.get_context("device_target") == "Ascend":
            if out_channels == 1001:
                self.matrix_G = Parameter(Tensor(np.zeros([1024, 1024]).astype(np.float32)),
                                          name='matrix_G', requires_grad=False)
                self.pad = P.Pad(((0, 23), (0, 23)))
                self.pad1 = P.Pad(((0, 7), (0, 7)))
                self.slice = P.Slice()
                self.add = P.TensorAdd()
            else:
                self.matrix_G = Parameter(Tensor(np.eye(out_channels).astype(np.float32)),
                                          name="matrix_G", requires_grad=False)
                self.abs = P.Abs()
                self.reduce_max = P.ReduceMax(keep_dims=False)
                self.neg = P.Neg()
                self.reduce_sum = P.ReduceSum()
            self.matmul = P.MatMul(transpose_b=True)
            self.cube_matmul = P.CusMatMulCube(transpose_a=True)
            self.cast = P.Cast()
            self.is_nsp_layer = (out_channels == 2)
        else:
            self.is_Ascend = False
            self.matrix_G = Parameter(Tensor(np.eye(out_channels).astype(np.float32)),
                                      name="matrix_G", requires_grad=False)
            self.cube_matmul = P.MatMul(transpose_a=True)
        self.getG = P.InsertGradientOf(self.save_gradient)


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
                matrix_G = self.cube_matmul(dout, dout)
                matrix_G = self.mul(matrix_G, 1.0 / normalizer)
                if self.out_channels == 1001:
                    matrix_G = P.Pad(((0, 23), (0, 23)))(matrix_G)
                self.matrix_G = matrix_G
        else:
            dout_shape = self.shape(dout)
            normalizer = dout_shape[0]
            matrix_G = self.cube_matmul(dout, dout)
            matrix_G = self.mul(matrix_G, 1.0 / normalizer)
            self.matrix_G = matrix_G
        return out

    def construct(self, x):
        if self.thor:
            if self.is_Ascend:
                inputs = self.cube_matmul(x, x)
                shape = self.shape(x)
                normalizer = self.cast(shape[0], mstype.float32)
                matrix_A = self.mul(inputs, 1.0 / normalizer)
                self.matrix_A = matrix_A
            else:
                inputs = self.cube_matmul(x, x)
                inputs_shape = self.shape(inputs)
                normalizer = inputs_shape[0]
                matrix_A = self.mul(inputs, 1.0 / normalizer)
                self.matrix_A = matrix_A
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
        # if self.activation_flag:
        #     s += ', activation={}'.format(self.activation)
        return s

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
                 transposed=False):
        super(_Conv, self).__init__()
        self.in_channels = Validator.check_positive_int(in_channels)
        self.out_channels = Validator.check_positive_int(out_channels)
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad_mode = pad_mode
        # self.weight_init = weight_init
        self.bias_init = bias_init
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
        if (not isinstance(kernel_size[0], int)) or (not isinstance(kernel_size[1], int)) or \
                isinstance(kernel_size[0], bool) or isinstance(kernel_size[1], bool) or \
                kernel_size[0] < 1 or kernel_size[1] < 1:
            raise ValueError("Attr 'kernel_size' of 'Conv2D' Op passed "
                             + str(self.kernel_size) + ", should be a int or tuple and equal to or greater than 1.")
        if (not isinstance(stride[0], int)) or (not isinstance(stride[1], int)) or \
                isinstance(stride[0], bool) or isinstance(stride[1], bool) or stride[0] < 1 or stride[1] < 1:
            raise ValueError("Attr 'stride' of 'Conv2D' Op passed "
                             + str(self.stride) + ", should be a int or tuple and equal to or greater than 1.")
        if (not isinstance(dilation[0], int)) or (not isinstance(dilation[1], int)) or \
                isinstance(dilation[0], bool) or isinstance(dilation[1], bool) or dilation[0] < 1 or dilation[1] < 1:
            raise ValueError("Attr 'dilation' of 'Conv2D' Op passed "
                             + str(self.dilation) + ", should be a int or tuple and equal to or greater than 1.")
        if in_channels % group != 0:
            raise ValueError("Attr 'in_channels' of 'Conv2D' Op must be divisible by "
                             "attr 'group' of 'Conv2D' Op.")
        if out_channels % group != 0:
            raise ValueError("Attr 'out_channels' of 'Conv2D' Op must be divisible by "
                             "attr 'group' of 'Conv2D' Op.")
        if transposed:
            shape = [in_channels, out_channels // group, *kernel_size]
        else:
            shape = [out_channels, in_channels // group, *kernel_size]
        self.weight = Parameter(initializer(weight_init, shape), name='weight')

        if Validator.check_bool(has_bias):
            self.bias = Parameter(initializer(self.bias_init, [out_channels]), name='bias')
        else:
            if self.bias_init != 'zeros':
                logger.warning("Value of 'has_bias' is False, value of 'bias_init' will be ignored.")
            self.bias = None

    def construct(self, *inputs):
        """Must be overridden by all subclasses."""
        raise NotImplementedError


class Conv2d_Thor(_Conv):
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
    (\text{ks_h} - 1) \times (\text{dilation} - 1) }{\text{stride}}} \right \rfloor` and
    :math:`\left \lfloor{1 + \frac{W_{in} + 2 \times \text{padding} - \text{ks_w} -
    (\text{ks_w} - 1) \times (\text{dilation} - 1) }{\text{stride}}} \right \rfloor` respectively.

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

    Inputs:
        - **input** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor of shape :math:`(N, C_{out}, H_{out}, W_{out})`.

    Examples:
        >>> net = nn.Conv2d(120, 240, 4, has_bias=False, weight_init='normal')
        >>> input = Tensor(np.ones([1, 120, 1024, 640]), mindspore.float32)
        >>> net(input).shape
        (1, 240, 1024, 640)
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
        self._dilation = dilation
        dilation = twice(dilation)
        super(Conv2d_Thor, self).__init__(
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
        self._init_depthwise_conv2d(weight_init)
        self.bias_add = P.BiasAdd()

        self.thor = True
        self.hw = kernel_size[0] * kernel_size[1]
        self.matrix_A_dim = self.in_channels * self.kernel_size[0] * self.kernel_size[1]
        self.matrix_G_dim = self.out_channels
        self.shape = P.Shape()
        self.reshape = P.Reshape()
        self.mul = P.Mul()
        self.cast = P.Cast()
        self.A_normalizer = Parameter(initializer(0, [1], mstype.float32), name="A_normalizer", requires_grad=False)
        self.G_normalizer = Parameter(initializer(0, [1], mstype.float32), name="G_normalizer", requires_grad=False)
        self.is_Ascend = True
        if context.get_context("device_target") == "Ascend":
            ksizes = (1, kernel_size[0], kernel_size[1], 1)
            strides = (1, stride[0], stride[1], 1)
            self.img2col = P.CusImg2Col(ksizes=ksizes, strides=strides)
            self.cube_matmul = P.CusMatMulCube(transpose_a=True)
            self.transpose02314 = P.CusTranspose02314()
            dampingA_dim = self.matrix_A_dim
            self.diag_block_dim = 128
            if (self.matrix_A_dim % self.diag_block_dim) != 0 and self.matrix_A_dim > self.diag_block_dim:
                dampingA_dim = (self.matrix_A_dim // self.diag_block_dim + 1) * self.diag_block_dim
            dampingG_dim = self.matrix_G_dim
            if (self.matrix_G_dim % self.diag_block_dim) != 0 and self.matrix_G_dim > self.diag_block_dim:
                dampingG_dim = (self.matrix_G_dim // self.diag_block_dim + 1) * self.diag_block_dim
            self.matrix_A_cov = Parameter(Tensor(np.zeros([dampingA_dim, dampingA_dim]).astype(np.float32)),
                                          name='matrix_A', requires_grad=False)
            self.matrix_G_cov = Parameter(Tensor(np.zeros([dampingG_dim, dampingG_dim]).astype(np.float32)),
                                          name='matrix_G', requires_grad=False)

            self.channels_slice_flag = False
            self.C0 = 16
            if self.in_channels % self.C0 != 0:
                self.channels_slice_flag = True
            self.padA_flag = False
            if (self.matrix_A_dim // self.diag_block_dim) * self.diag_block_dim != self.matrix_A_dim \
                    and self.matrix_A_dim > self.diag_block_dim:
                self.padA_flag = True
                pad_dim = self.diag_block_dim - self.matrix_A_dim % self.diag_block_dim
                self.padA = P.Pad(((0, pad_dim), (0, pad_dim)))
            self.slice = P.Slice()
        else:
            self.is_Ascend = False
            self.img2col = P.Im2Col(kernel_size=kernel_size, stride=stride, pad_mode="same")
            self.matmul = P.MatMul(transpose_b=True)
            self.reduce_mean = P.ReduceMean(keep_dims=False)
            self.matrix_A_cov = Parameter(Tensor(np.zeros([self.matrix_A_dim, self.matrix_A_dim]).astype(np.float32)),
                                          name='matrix_A', requires_grad=False)
            self.matrix_G_cov = Parameter(Tensor(np.zeros([self.matrix_G_dim, self.matrix_G_dim]).astype(np.float32)),
                                          name='matrix_G', requires_grad=False)
        self.getG = P.InsertGradientOf(self.save_gradient)


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
            matrix_G = self.cube_matmul(dout, dout)
            normalizer = self.cast(normalizer, mstype.float32)
            matrix_G = self.mul(matrix_G, 1.0 / normalizer)
            self.G_normalizer = normalizer
            self.matrix_G_cov = matrix_G
        else:
            dout = self.reduce_mean(dout, 0)
            dout_shape = self.shape(dout)
            dout = self.reshape(dout, (dout_shape[0], -1))
            dout_shape = self.shape(dout)
            normalizer = dout_shape[1]
            dout = self.cast(dout, mstype.float32)
            matrix_G = self.matmul(dout, dout)
            matrix_G = self.mul(matrix_G, 1.0 / normalizer)
            self.G_normalizer = normalizer
            self.matrix_G_cov = matrix_G
        return out



    def construct(self, x):
        if self.thor:
            matrix_A = self.img2col(x)
            matrix_A_shape = self.shape(matrix_A)
            if self.is_Ascend:
                normalizer = matrix_A_shape[0]
                matrix_A = self.cube_matmul(matrix_A, matrix_A)
                if self.channels_slice_flag:
                    matrix_A = self.reshape(matrix_A, (self.hw, self.C0, self.hw, self.C0))
                    matrix_A = self.slice(matrix_A, (0, 0, 0, 0),
                                          (self.hw, self.in_channels, self.hw, self.in_channels))
                    matrix_A = self.reshape(matrix_A, (self.matrix_A_dim, self.matrix_A_dim))
                normalizer = self.cast(normalizer, mstype.float32)
                matrix_A = self.mul(matrix_A, 1.0 / normalizer)
                if self.padA_flag:
                    matrix_A = self.padA(matrix_A)
                self.A_normalizer = normalizer
                self.matrix_A_cov = matrix_A
            else:
                matrix_A = self.reshape(matrix_A, (matrix_A_shape[0] * matrix_A_shape[1] * matrix_A_shape[2],
                                                   matrix_A_shape[3], -1))
                matrix_A = self.reduce_mean(matrix_A, 1)
                matrix_A_shape = self.shape(matrix_A)
                normalizer = matrix_A_shape[1]
                matrix_A = self.cast(matrix_A, mstype.float32)
                matrix_A = self.matmul(matrix_A, matrix_A)
                matrix_A = self.mul(matrix_A, 1.0 / normalizer)
                self.A_normalizer = normalizer
                self.matrix_A_cov = matrix_A
            output = self.conv2d(x, self.weight)
            output = self.getG(output)
        else:
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
                self.weight_init,
                self.bias_init)
        return s

class Embedding_Thor(Cell):
    r"""
    A simple lookup table that stores embeddings of a fixed dictionary and size.

    This module is often used to store word embeddings and retrieve them using
    indices. The input to the module is a list of indices, and the output is
    the corresponding word embeddings.

    Note:
        When 'use_one_hot' is set to True, the type of the input must be mindspore.int32.

    Args:
        vocab_size (int): Size of the dictionary of embeddings.
        embedding_size (int): The size of each embedding vector.
        use_one_hot (bool): Specifies whether to apply one_hot encoding form. Default: False.
        embedding_table (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the embedding_table.
            Refer to class `initializer` for the values of string when a string
            is specified. Default: 'normal'.
        dtype (:class:`mindspore.dtype`): Data type of input. Default: mindspore.float32.
        padding_idx (int, None): When the padding_idx encounters index, the output embedding vector of this index
                                 will be initialized to zero. Default: None. The feature is inactivated.
    Inputs:
        - **input** (Tensor) - Tensor of shape :math:`(\text{batch_size}, \text{input_length})`. The elements of
          the Tensor must be integer and not larger than vocab_size. Otherwise the corresponding embedding vector will
          be zero.

    Outputs:
        Tensor of shape :math:`(\text{batch_size}, \text{input_length}, \text{embedding_size})`.

    Examples:
        >>> net = nn.Embedding(20000, 768,  True)
        >>> input_data = Tensor(np.ones([8, 128]), mindspore.int32)
        >>>
        >>> # Maps the input word IDs to word embedding.
        >>> output = net(input_data)
        >>> output.shape
        (8, 128, 768)
    """

    def __init__(self, vocab_size, embedding_size, use_one_hot=False, embedding_table='normal',
                 dtype=mstype.float32, padding_idx=None):
        super(Embedding_Thor, self).__init__()
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
        self.matrix_A = Parameter(Tensor(np.zeros([vocab_size]).astype(np.float32)),
                                  name='matrix_A', requires_grad=False)
        self.matrix_G = Parameter(Tensor(np.zeros([embedding_size, embedding_size]).astype(np.float32)),
                                  name="matrix_G", requires_grad=False)
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
        matrix_G = self.cube_matmul(dout, dout)
        matrix_G = self.mul(matrix_G, 1.0 / normalizer)
        self.matrix_G = matrix_G
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
                matrix_A = self.reduce_sum(one_hot_ids, 0)
                self.matrix_A = matrix_A
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
