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
"""thor_layer"""
import numpy as np

import mindspore.common.dtype as mstype
from mindspore._checkparam import Validator, twice
from mindspore._extends import cell_attr_register
from mindspore.common.initializer import initializer
from mindspore.common.parameter import Parameter
from mindspore.common.tensor import Tensor
from mindspore.nn.cell import Cell
from mindspore.nn.layer.activation import get_activation
from mindspore.ops import operations as P

C0 = 16


def caculate_device_shape(matrix_dim, channel, is_A):
    ll = (0)
    if is_A:
        if channel // C0 == 0:
            matrix_dim = (matrix_dim / channel) * C0
        ll = (int(matrix_dim // C0), int(matrix_dim // C0), C0, C0), int(matrix_dim)
    else:
        ll = (int(matrix_dim // C0), int(matrix_dim // C0), C0, C0), int(matrix_dim)
    return ll


def caculate_matmul_shape(matrix_A_dim, matrix_G_dim, split_dim):
    split_dimA = split_dim
    split_dimG = split_dim
    if matrix_A_dim % split_dim == 0:
        batch_w = matrix_A_dim // split_dim
    else:
        if matrix_A_dim < split_dim:
            batch_w = 1
            split_dimA = matrix_A_dim
        else:
            batch_w = matrix_A_dim // split_dim + 1

    if matrix_G_dim % split_dim == 0:
        batch_h = matrix_G_dim // split_dim
    else:
        if matrix_G_dim < split_dim:
            batch_h = 1
            split_dimG = matrix_G_dim
        else:
            batch_h = matrix_G_dim // split_dim + 1
    matrix_A_shape = (batch_h, batch_w, split_dimA, split_dimA)
    matrix_G_shape = (batch_h, split_dimG, split_dimG)
    return matrix_A_shape, matrix_G_shape

class _Conv(Cell):
    r"""Applies a N-D convolution over an input signal composed of several input
       planes.
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
                 data_format,
                 has_bias,
                 weight_init,
                 bias_init,
                 ):
        super(_Conv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad_mode = pad_mode
        self.padding = padding
        self.dilation = dilation
        self.group = group
        self.data_format = data_format
        self.has_bias = has_bias
        if not (isinstance(in_channels, int) and in_channels > 0):
            raise ValueError('Attr \'in_channels\' of \'Conv2D\' Op passed '
                             + str(in_channels) + ', should be a int and greater than 0.')
        if (not isinstance(kernel_size, tuple)) or len(kernel_size) != 2 or \
                (not isinstance(kernel_size[0], int)) or (not isinstance(kernel_size[1], int)) or \
                kernel_size[0] < 1 or kernel_size[1] < 1:
            raise ValueError('Attr \'kernel_size\' of \'Conv2D\' Op passed '
                             + str(self.kernel_size) + ', should be a int or tuple and equal to or greater than 1.')
        if in_channels % group != 0:
            raise ValueError('Attr \'in_channels\' of \'Conv2D\' Op must be divisible by '
                             'attr \'group\' of \'Conv2D\' Op.')
        if out_channels % group != 0:
            raise ValueError('Attr \'out_channels\' of \'Conv2D\' Op must be divisible by '
                             'attr \'group\' of \'Conv2D\' Op.')

        self.weight = Parameter(initializer(
            weight_init, [out_channels, in_channels // group, *kernel_size]))

        if Validator.check_bool(has_bias):
            self.bias = Parameter(initializer(bias_init, [out_channels]))
        else:
            if bias_init != 'zeros':
                logger.warning("Value of 'has_bias' is False, value of 'bias_init' will be ignored.")
            self.bias = None

    def construct(self, *inputs):
        raise NotImplementedError


class Conv2d_Thor_GPU(_Conv):
    """Conv2d_Thor"""
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 pad_mode='same',
                 padding=0,
                 dilation=1,
                 group=1,
                 data_format='NCHW',
                 has_bias=False,
                 weight_init='normal',
                 damping=0.03,
                 loss_scale=1,
                 frequency=278,
                 batch_size=32,
                 bias_init='zeros'):
        self.thor = True
        self.hw = kernel_size * kernel_size
        kernel_size = twice(kernel_size)
        super(Conv2d_Thor_GPU, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            pad_mode,
            padding,
            dilation,
            group,
            data_format,
            has_bias,
            weight_init,
            bias_init,
        )
        self.conv2d = P.Conv2D(out_channel=self.out_channels,
                               kernel_size=self.kernel_size,
                               mode=1,
                               pad_mode=self.pad_mode,
                               pad=self.padding,
                               stride=self.stride,
                               dilation=self.dilation,
                               group=self.group
                               )

        self.matrix_A_dim = self.in_channels * self.kernel_size[0] * self.kernel_size[1]
        self.matrix_G_dim = self.out_channels

        split_dim = 128
        matrix_A_shape, matrix_G_shape = caculate_matmul_shape(self.matrix_A_dim, self.matrix_G_dim, split_dim)
        self.matrix_A_inv = Parameter(np.zeros(matrix_A_shape).astype(np.float32), requires_grad=False)
        self.matrix_G_inv = Parameter(np.zeros(matrix_G_shape).astype(np.float32), requires_grad=False)
        self.broadcast_to = P.BroadcastTo(matrix_A_shape)
        self.cov_step = Parameter(initializer(0, [1], mstype.int32), requires_grad=False)
        self.img2col = P.Im2Col(kernel_size=kernel_size, stride=stride, pad_mode="same")
        self.matmul = P.MatMul(transpose_b=True)
        self.shape = P.Shape()
        self.reshape = P.Reshape()
        self.mul = P.Mul()
        self.getG = P.InsertGradientOf(self.save_gradient)
        self.loss_scale = Tensor(1 / loss_scale, mstype.float16)
        self.batch_size = Tensor(batch_size, mstype.float16)
        self.transpose = P.Transpose()
        self.cast = P.Cast()
        self.gather = P.Gather()
        self.freq = Tensor(frequency, mstype.int32)
        self.axis = 0
        self.sqrt = P.Sqrt()
        self.reduce_mean = P.ReduceMean(keep_dims=False)
        self.damping = Parameter(Tensor(damping), requires_grad=False)
        self.dampingA = Tensor(np.identity(self.matrix_A_dim), mstype.float32)
        self.dampingG = Tensor(np.identity(self.matrix_G_dim), mstype.float32)
        self.cholesky = P.CholeskyTrsm(split_dim=split_dim)
        self.vector_matmul = P.BatchMatMul(transpose_a=True)

    def save_gradient(self, dout):
        """save_gradient"""
        out = dout
        dout = self.mul(dout, self.loss_scale)
        dout = self.mul(dout, self.batch_size)
        dout = self.reduce_mean(dout, 0)
        dout_shape = self.shape(dout)
        dout = self.reshape(dout, (dout_shape[0], -1))
        dout_shape = self.shape(dout)
        normalizer = dout_shape[1]
        dout = self.cast(dout, mstype.float32)
        matrix_G = self.matmul(dout, dout)
        matrix_G = self.mul(matrix_G, 1.0 / normalizer)
        damping_step = self.gather(self.damping, self.cov_step, 0)
        damping_step = self.cast(damping_step, mstype.float32)
        self.cov_step = self.cov_step + self.freq
        damping = self.mul(damping_step, 1.0 / normalizer)
        damping = self.sqrt(damping)
        matrix_G = matrix_G + damping * self.dampingG
        matrix_G = self.cholesky(matrix_G)
        matrix_G = self.vector_matmul(matrix_G, matrix_G)
        self.matrix_G_inv = matrix_G
        return out

    def construct(self, x):
        if self.thor:
            matrix_A = self.img2col(x)
            matrix_A_shape = self.shape(matrix_A)
            matrix_A = self.reshape(matrix_A, (matrix_A_shape[0]*matrix_A_shape[1]*matrix_A_shape[2],
                                               matrix_A_shape[3], -1))
            matrix_A = self.reduce_mean(matrix_A, 1)
            matrix_A_shape = self.shape(matrix_A)
            normalizer = matrix_A_shape[1]
            matrix_A = self.cast(matrix_A, mstype.float32)
            matrix_A = self.matmul(matrix_A, matrix_A)
            matrix_A = self.mul(matrix_A, 1.0 / normalizer)
            damping_step = self.gather(self.damping, self.cov_step, self.axis)
            damping_step = self.cast(damping_step, mstype.float32)
            damping = self.mul(damping_step, 1.0 / normalizer)
            damping = self.sqrt(damping)
            matrix_A = matrix_A + damping * self.dampingA
            matrix_A = self.cholesky(matrix_A)
            matrix_A = self.vector_matmul(matrix_A, matrix_A)
            matrix_A = self.broadcast_to(matrix_A)
            self.matrix_A_inv = matrix_A
            out = self.conv2d(x, self.weight)
            out = self.getG(out)
        else:
            out = self.conv2d(x, self.weight)

        return out

    def extra_repr(self):
        """extra_repr"""
        s = 'input_channels={}, output_channels={}, kernel_size={},' \
            'stride={},  pad_mode={}, padding={}, dilation={}, ' \
            'group={}, data_format={}, has_bias={},' \
            'weight_init={}, bias_init={}'.format(
                self.in_channels,
                self.out_channels,
                self.kernel_size,
                self.stride,
                self.pad_mode,
                self.padding,
                self.dilation,
                self.group,
                self.data_format,
                self.has_bias,
                self.weight,
                self.bias)

        if self.has_bias:
            s += ', bias={}'.format(self.bias)
        return s


class Dense_Thor_GPU(Cell):
    """Dense_Thor"""
    @cell_attr_register(attrs=['has_bias', 'activation'])
    def __init__(self,
                 in_channels,
                 out_channels,
                 weight_init='normal',
                 bias_init='zeros',
                 damping=0.03,
                 loss_scale=1,
                 frequency=278,
                 batch_size=32,
                 has_bias=True,
                 activation=None):
        super(Dense_Thor_GPU, self).__init__()
        self.in_channels = Validator.check_positive_int(in_channels)
        self.out_channels = Validator.check_positive_int(out_channels)
        self.has_bias = Validator.check_bool(has_bias)
        self.thor = True
        if isinstance(weight_init, Tensor):
            if weight_init.ndim != 2 or weight_init.shape[0] != out_channels or \
                    weight_init.shape[1] != in_channels:
                raise ValueError("weight_init shape error")

        self.weight = Parameter(initializer(weight_init, [out_channels, in_channels]))

        if self.has_bias:
            if isinstance(bias_init, Tensor):
                if bias_init.ndim != 1 or bias_init.shape[0] != out_channels:
                    raise ValueError("bias_init shape error")

            self.bias = Parameter(initializer(bias_init, [out_channels]))

        self.matmul = P.MatMul(transpose_b=True)
        self.bias_add = P.BiasAdd()

        self.activation = get_activation(activation)
        self.activation_flag = self.activation is not None
        split_dim = 128
        matrix_A_shape, matrix_G_shape = caculate_matmul_shape(self.in_channels, self.out_channels, split_dim)
        self.matrix_A_inv = Parameter(Tensor(np.zeros(matrix_A_shape).astype(np.float32)), requires_grad=False)
        self.matrix_G_inv = Parameter(Tensor(np.zeros(matrix_G_shape).astype(np.float32)), requires_grad=False)
        self.broadcast_to = P.BroadcastTo(matrix_A_shape)
        self.cov_step = Parameter(initializer(0, [1], mstype.int32), requires_grad=False)
        self.shape = P.Shape()
        self.reshape = P.Reshape()
        self.transpose = P.Transpose()
        self.mul = P.Mul()
        self.cube_matmul = P.MatMul(transpose_a=True)
        self.loss_scale = Tensor(1 / loss_scale, mstype.float16)
        self.batch_size = Tensor(batch_size, mstype.float16)
        self.getG = P.InsertGradientOf(self.save_gradient)
        self.damping = Parameter(Tensor(damping), requires_grad=False)
        self.dampingA = Tensor(np.identity(in_channels), mstype.float32)
        self.dampingG = Tensor(np.identity(out_channels), mstype.float32)
        self.cast = P.Cast()
        self.gather = P.Gather()
        self.freq = Tensor(frequency, mstype.int32)
        self.axis = 0
        self.add = P.Add()
        self.sqrt = P.Sqrt()
        self.cholesky = P.CholeskyTrsm(split_dim=split_dim)
        self.vector_matmul = P.BatchMatMul(transpose_a=True)

    def save_gradient(self, dout):
        """save_gradient"""
        out = dout
        dout = self.mul(dout, self.loss_scale)
        dout = self.mul(dout, self.batch_size)
        dout_shape = self.shape(dout)
        normalizer = dout_shape[0]
        dout = self.cast(dout, mstype.float32)
        matrix_G = self.cube_matmul(dout, dout)
        matrix_G = self.mul(matrix_G, 1.0 / normalizer)
        damping_step = self.gather(self.damping, self.cov_step, 0)
        damping_step = self.cast(damping_step, mstype.float32)
        self.cov_step = self.cov_step + self.freq
        damping = self.sqrt(damping_step)
        matrix_G = matrix_G + damping * self.dampingG
        matrix_G = self.cholesky(matrix_G)
        matrix_G = self.vector_matmul(matrix_G, matrix_G)
        self.matrix_G_inv = matrix_G
        return out

    def construct(self, x):
        """construct"""
        if self.thor:
            inputs = self.cast(x, mstype.float32)
            inputs = self.cube_matmul(inputs, inputs)
            inputs_shape = self.shape(inputs)
            normalizer = inputs_shape[0]
            matrix_A = self.mul(inputs, 1.0 / normalizer)
            damping_step = self.gather(self.damping, self.cov_step, self.axis)
            damping_step = self.cast(damping_step, mstype.float32)
            damping = self.sqrt(damping_step)
            matrix_A = matrix_A + damping * self.dampingA
            matrix_A = self.cholesky(matrix_A)
            matrix_A = self.vector_matmul(matrix_A, matrix_A)
            matrix_A = self.broadcast_to(matrix_A)
            self.matrix_A_inv = matrix_A
            output = self.matmul(x, self.weight)
            output = self.getG(output)
        else:
            output = self.matmul(x, self.weight)

        if self.has_bias:
            output = self.bias_add(output, self.bias)
        if self.activation_flag:
            return self.activation(output)
        return output

    def extend_repr(self):
        """extend_repr"""
        s = 'in_channels={}, out_channels={}'.format(self.in_channels, self.out_channels)
        if self.has_bias:
            s += ', has_bias={}'.format(self.has_bias)
        if self.activation_flag:
            s += ', activation={}'.format(self.activation)
        return s


class Conv2d_Thor(_Conv):
    """Conv2d_Thor"""
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 pad_mode='same',
                 padding=0,
                 dilation=1,
                 group=1,
                 data_format='NCHW',
                 has_bias=False,
                 weight_init='normal',
                 damping=0.03,
                 loss_scale=1,
                 frequency=278,
                 batch_size=32,
                 bias_init='zeros'):
        self.thor = True
        ksizes = (1, kernel_size, kernel_size, 1)
        self.hw = kernel_size * kernel_size
        strides = (1, stride, stride, 1)
        kernel_size = twice(kernel_size)
        super(Conv2d_Thor, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            pad_mode,
            padding,
            dilation,
            group,
            data_format,
            has_bias,
            weight_init,
            bias_init,
        )
        self.conv2d = P.Conv2D(out_channel=self.out_channels,
                               kernel_size=self.kernel_size,
                               mode=1,
                               pad_mode=self.pad_mode,
                               pad=self.padding,
                               stride=self.stride,
                               dilation=self.dilation,
                               group=self.group
                               )
        self.batch_size = batch_size
        self.img2col = P.CusImg2Col(ksizes=ksizes, strides=strides)
        self.cube_matmul = P.CusMatMulCube(transpose_a=True)
        self.matrix_combine = P.CusMatrixCombine()
        self.cholesky = P.CusCholeskyTrsm()
        self.transpose02314 = P.CusTranspose02314()
        self.matrix_A_dim = self.in_channels * self.kernel_size[0] * self.kernel_size[1]
        self.matrix_G_dim = self.out_channels
        self.matrix_A_device_shape, self.matrix_A_device_dim = caculate_device_shape(self.matrix_A_dim,
                                                                                     self.in_channels, True)
        self.matrix_G_device_shape, self.matrix_G_device_dim = caculate_device_shape(self.matrix_G_dim,
                                                                                     self.in_channels, False)
        self.matrix_A_device_temp_shape = (
            self.matrix_A_device_shape[0], self.matrix_A_device_shape[2], self.matrix_A_device_shape[1],
            self.matrix_A_device_shape[3])
        self.matrix_G_device_temp_shape = (
            self.matrix_G_device_shape[0], self.matrix_G_device_shape[2], self.matrix_G_device_shape[1],
            self.matrix_G_device_shape[3])
        self.matrix_A_inv = Parameter(
            Tensor(np.reshape(np.identity(self.matrix_A_device_dim).astype(np.float16), self.matrix_A_device_shape)),
            requires_grad=False)
        self.A_inv_max = Parameter(initializer(0, [1], mstype.float32), requires_grad=False)
        self.matrix_G_inv = Parameter(
            Tensor(np.reshape(np.identity(self.matrix_G_device_dim).astype(np.float16), self.matrix_G_device_shape)),
            requires_grad=False)

        self.G_inv_max = Parameter(initializer(0, [1], mstype.float32), requires_grad=False)
        self.fake_G = Tensor(
            np.reshape(np.identity(self.matrix_G_device_dim).astype(np.float16), self.matrix_G_device_shape))

        self.shape = P.Shape()
        self.reshape = P.Reshape()
        self.transpose = P.Transpose()
        self.cov_step = Parameter(initializer(0, [1], mstype.int32), requires_grad=False)
        self.mul = P.Mul()
        self.cast = P.Cast()
        self.damping = Tensor(damping)
        self.vector_matmul = P.CusBatchMatMul()
        self.diag_block_dim = 128
        self.channels_slice_flag = False
        if self.in_channels % C0 != 0:
            self.channels_slice_flag = True

        self.padA_flag = False
        if (self.matrix_A_dim // self.diag_block_dim) * self.diag_block_dim != self.matrix_A_dim \
            and self.matrix_A_dim > self.diag_block_dim:
            self.padA_flag = True
            pad_dim = self.diag_block_dim - self.matrix_A_dim % self.diag_block_dim
            self.padA = P.Pad(((0, pad_dim), (0, pad_dim)))
        self.device_shape_pad_flag = False
        if self.matrix_A_dim != self.matrix_A_device_dim:
            self.device_shape_pad_flag = True
            self.device_shape_pad = P.Pad(((0, 0), (0, C0 - self.in_channels), (0, 0), (0, C0 - self.in_channels)))
        self.slice = P.Slice()
        self.gather = P.Gather()
        self.freq = Tensor(frequency, mstype.int32)
        self.loss_scale = Tensor(1 / loss_scale, mstype.float16)
        self.axis = 0

        dampingA_dim = self.matrix_A_dim
        if (self.matrix_A_dim % self.diag_block_dim) != 0 and self.matrix_A_dim > self.diag_block_dim:
            dampingA_dim = (self.matrix_A_dim // self.diag_block_dim + 1) * self.diag_block_dim
        dampingG_dim = self.matrix_G_dim
        if (self.matrix_G_dim % self.diag_block_dim) != 0 and self.matrix_G_dim > self.diag_block_dim:
            dampingG_dim = (self.matrix_G_dim // self.diag_block_dim + 1) * self.diag_block_dim

        self.dampingA = Tensor(np.identity(dampingA_dim), mstype.float32)
        self.dampingG = Tensor(np.identity(dampingG_dim), mstype.float32)
        self.fused_abs_max1 = P.CusFusedAbsMax1([self.matrix_A_dim, self.matrix_A_dim])
        self.fused_abs_max2 = P.CusFusedAbsMax1()
        self.log = P.Log()
        self.exp = P.Exp()
        self.sqrt = P.Sqrt()
        self.getG = P.InsertGradientOf(self.save_gradient)

    def save_gradient(self, dout):
        """save_gradient"""
        out = dout
        dout = self.mul(dout, self.loss_scale)
        dout = self.mul(dout, 32.0)
        dout = self.transpose02314(dout)
        dout_shape = self.shape(dout)
        normalizer = dout_shape[0]

        matrix_G = self.cube_matmul(dout, dout)
        normalizer = self.cast(normalizer, mstype.float32)
        matrix_G = self.mul(matrix_G, 1.0 / normalizer)
        damping_step = self.gather(self.damping, self.cov_step, 0)
        self.cov_step = self.cov_step + self.freq
        damping_step = self.cast(damping_step, mstype.float32)
        damping = self.mul(damping_step, 32.0 / normalizer)
        damping = self.sqrt(damping)
        dampingG = self.cast(self.dampingG, mstype.float32)
        matrix_G = matrix_G + damping * dampingG

        matrix_G_inv = self.cholesky(matrix_G)
        matrix_G_inv = self.vector_matmul(matrix_G_inv, matrix_G_inv)
        matrix_G_inv_max = self.fused_abs_max2(matrix_G_inv)
        matrix_G_inv_max = self.fused_abs_max2(matrix_G_inv_max)
        self.G_inv_max = matrix_G_inv_max
        matrix_G_inv = self.matrix_combine(matrix_G_inv)
        matrix_G_inv = self.reshape(matrix_G_inv, self.matrix_G_device_temp_shape)
        matrix_G_inv = self.transpose(matrix_G_inv, (2, 0, 1, 3))
        matrix_G = self.cast(matrix_G_inv, mstype.float16)
        self.matrix_G_inv = matrix_G
        return out

    def construct(self, x):
        if self.thor:
            matrix_A = self.img2col(x)
            matrix_A_shape = self.shape(matrix_A)
            normalizer = matrix_A_shape[0]
            matrix_A = self.cube_matmul(matrix_A, matrix_A)

            if self.channels_slice_flag:
                matrix_A = self.reshape(matrix_A, (self.hw, C0, self.hw, C0))
                matrix_A = self.slice(matrix_A, (0, 0, 0, 0), (self.hw, self.in_channels, self.hw, self.in_channels))
                matrix_A = self.reshape(matrix_A, (self.matrix_A_dim, self.matrix_A_dim))
            normalizer = self.cast(normalizer, mstype.float32)
            matrix_A = self.mul(matrix_A, 1.0 / normalizer)
            if self.padA_flag:
                matrix_A = self.padA(matrix_A)
            damping_step = self.gather(self.damping, self.cov_step, self.axis)
            damping_step = self.cast(damping_step, mstype.float32)
            damping = self.mul(damping_step, 32.0 / normalizer)
            damping = self.sqrt(damping)
            damping_A = self.cast(self.dampingA, mstype.float32)
            matrix_A = matrix_A + damping * damping_A
            matrix_A_inv = self.cholesky(matrix_A)
            matrix_A_inv = self.vector_matmul(matrix_A_inv, matrix_A_inv)
            matrix_A_inv_max = self.fused_abs_max1(matrix_A_inv)
            matrix_A_inv_max = self.fused_abs_max2(matrix_A_inv_max)
            self.A_inv_max = matrix_A_inv_max
            matrix_A_inv = self.matrix_combine(matrix_A_inv)
            matrix_A_inv = self.cast(matrix_A_inv, mstype.float16)
            if self.padA_flag:
                matrix_A_inv = self.slice(matrix_A_inv, (0, 0), (self.matrix_A_dim, self.matrix_A_dim))

            if self.device_shape_pad_flag:
                matrix_A_inv = self.reshape(matrix_A_inv, (self.hw, self.in_channels, self.hw, self.in_channels))
                matrix_A_inv = self.device_shape_pad(matrix_A_inv)
            matrix_A_inv = self.reshape(matrix_A_inv, self.matrix_A_device_temp_shape)
            matrix_A_inv = self.transpose(matrix_A_inv, (2, 0, 1, 3))
            self.matrix_A_inv = matrix_A_inv
            out = self.conv2d(x, self.weight)
            out = self.getG(out)
        else:
            out = self.conv2d(x, self.weight)

        return out

    def extra_repr(self):
        """extra_repr"""
        s = 'input_channels={}, output_channels={}, kernel_size={},' \
            'stride={},  pad_mode={}, padding={}, dilation={}, ' \
            'group={}, data_format={}, has_bias={},' \
            'weight_init={}, bias_init={}'.format(
                self.in_channels,
                self.out_channels,
                self.kernel_size,
                self.stride,
                self.pad_mode,
                self.padding,
                self.dilation,
                self.group,
                self.data_format,
                self.has_bias,
                self.weight,
                self.bias)

        if self.has_bias:
            s += ', bias={}'.format(self.bias)
        return s


class Dense_Thor(Cell):
    """Dense_Thor"""
    @cell_attr_register(attrs=['has_bias', 'activation'])
    def __init__(self,
                 in_channels,
                 out_channels,
                 weight_init='normal',
                 bias_init='zeros',
                 damping=0.03,
                 loss_scale=1,
                 frequency=278,
                 batch_size=32,
                 has_bias=True,
                 activation=None):
        super(Dense_Thor, self).__init__()
        self.in_channels = Validator.check_positive_int(in_channels)
        self.out_channels = Validator.check_positive_int(out_channels)
        self.has_bias = Validator.check_bool(has_bias)
        self.thor = True
        self.batch_size = batch_size
        if isinstance(weight_init, Tensor):
            if weight_init.ndim != 2 or weight_init.shape[0] != out_channels or \
                    weight_init.shape[1] != in_channels:
                raise ValueError("weight_init shape error")

        self.weight = Parameter(initializer(weight_init, [out_channels, in_channels]))

        if self.has_bias:
            if isinstance(bias_init, Tensor):
                if bias_init.ndim != 1 or bias_init.shape[0] != out_channels:
                    raise ValueError("bias_init shape error")

            self.bias = Parameter(initializer(bias_init, [out_channels]))

        self.matmul = P.MatMul(transpose_b=True)
        self.bias_add = P.BiasAdd()

        self.activation = get_activation(activation)
        self.activation_flag = self.activation is not None

        self.matrix_A_inv = Parameter(Tensor(np.zeros([128, 128, 16, 16]).astype(np.float16)), requires_grad=False)
        self.matrix_G_inv = Parameter(Tensor(np.zeros([63, 63, 16, 16]).astype(np.float16)), requires_grad=False)
        self.fake_G = Tensor(np.zeros([63, 63, 16, 16]).astype(np.float16))

        self.matmul = P.MatMul(transpose_b=True)
        self.cube_matmul = P.CusMatMulCube(transpose_a=True)
        self.matrix_combine = P.CusMatrixCombine()
        self.cholesky = P.CusCholeskyTrsm()
        self.shape = P.Shape()
        self.reshape = P.Reshape()
        self.transpose = P.Transpose()
        self.cov_step = Parameter(initializer(0, [1], mstype.int32), requires_grad=False)
        self.mul = P.Mul()
        self.cast = P.Cast()
        self.damping = Tensor(damping)
        self.loss_scale = Tensor(1 / loss_scale, mstype.float16)
        self.vector_matmul = P.CusBatchMatMul()
        self.pad = P.Pad(((0, 23), (0, 23)))
        self.pad1 = P.Pad(((0, 7), (0, 7)))
        self.slice = P.Slice()
        self.gather = P.Gather()
        self.assignadd = P.AssignAdd()
        self.freq = Tensor(frequency, mstype.int32)
        self.axis = 0
        self.A_inv_max = Parameter(initializer(0, [1], mstype.float32), requires_grad=False)
        self.G_inv_max = Parameter(initializer(0, [1], mstype.float32), requires_grad=False)
        self.fused_abs_max1 = P.CusFusedAbsMax1([1001, 1001])
        self.fused_abs_max2 = P.CusFusedAbsMax1()
        self.log = P.Log()
        self.exp = P.Exp()
        self.dampingA = Tensor(np.identity(2048), mstype.float32)
        self.dampingG = Tensor(np.identity(1024), mstype.float32)
        self.add = P.Add()
        self.sqrt = P.Sqrt()
        self.getG = P.InsertGradientOf(self.save_gradient)

    def save_gradient(self, dout):
        """save_gradient"""
        out = dout
        dout = self.mul(dout, self.loss_scale)
        dout = self.mul(dout, 32.0)
        normalizer = 32
        matrix_G = self.cube_matmul(dout, dout)
        normalizer = self.cast(normalizer, mstype.float32)
        matrix_G = self.mul(matrix_G, 1.0 / normalizer)
        matrix_G = self.pad(matrix_G)
        damping_step = self.gather(self.damping, self.cov_step, 0)
        damping_step = self.cast(damping_step, mstype.float32)
        self.cov_step = self.cov_step + self.freq
        damping = self.sqrt(damping_step)
        dampingG = self.cast(self.dampingG, mstype.float32)
        matrix_G = matrix_G + damping * dampingG
        matrix_G_inv = self.cholesky(matrix_G)
        matrix_G_inv = self.vector_matmul(matrix_G_inv, matrix_G_inv)
        matrix_G_inv_max = self.fused_abs_max1(matrix_G_inv)
        matrix_G_inv_max = self.fused_abs_max2(matrix_G_inv_max)
        self.G_inv_max = matrix_G_inv_max
        matrix_G_inv = self.matrix_combine(matrix_G_inv)
        matrix_G_inv = self.slice(matrix_G_inv, (0, 0), (1001, 1001))
        matrix_G_inv = self.pad1(matrix_G_inv)
        matrix_G_inv_shape = self.shape(matrix_G_inv)
        matrix_G_inv = self.reshape(matrix_G_inv, (matrix_G_inv_shape[0] / 16, 16, matrix_G_inv_shape[0] / 16, 16))
        matrix_G_inv = self.transpose(matrix_G_inv, (2, 0, 1, 3))
        matrix_G_inv = self.cast(matrix_G_inv, mstype.float16)
        self.matrix_G_inv = matrix_G_inv
        return out

    def construct(self, x):
        """construct"""
        if self.thor:
            inputs = self.cube_matmul(x, x)
            normalizer = 32
            normalizer = self.cast(normalizer, mstype.float32)
            matrix_A = self.mul(inputs, 1.0 / normalizer)

            damping_step = self.gather(self.damping, self.cov_step, self.axis)
            damping_step = self.cast(damping_step, mstype.float32)
            damping = self.sqrt(damping_step)
            dampingA = self.cast(self.dampingA, mstype.float32)
            matrix_A = matrix_A + damping * dampingA
            matrix_A_inv = self.cholesky(matrix_A)
            matrix_A_inv = self.vector_matmul(matrix_A_inv, matrix_A_inv)

            matrix_A_inv_max = self.fused_abs_max2(matrix_A_inv)
            matrix_A_inv_max = self.fused_abs_max2(matrix_A_inv_max)
            self.A_inv_max = matrix_A_inv_max

            matrix_A_inv = self.matrix_combine(matrix_A_inv)
            matrix_A_inv_shape = self.shape(matrix_A_inv)
            matrix_A_inv = self.reshape(matrix_A_inv, (matrix_A_inv_shape[0] / 16, 16, matrix_A_inv_shape[0] / 16, 16))
            matrix_A_inv = self.transpose(matrix_A_inv, (2, 0, 1, 3))
            matrix_A_inv = self.cast(matrix_A_inv, mstype.float16)
            self.matrix_A_inv = matrix_A_inv
            output = self.matmul(x, self.weight)
            output = self.getG(output)
        else:
            output = self.matmul(x, self.weight)

        if self.has_bias:
            output = self.bias_add(output, self.bias)
        if self.activation_flag:
            return self.activation(output)
        return output

    def extend_repr(self):
        """extend_repr"""
        s = 'in_channels={}, out_channels={}'.format(self.in_channels, self.out_channels)
        if self.has_bias:
            s += ', has_bias={}'.format(self.has_bias)
        if self.activation_flag:
            s += ', activation={}'.format(self.activation)
        return s
