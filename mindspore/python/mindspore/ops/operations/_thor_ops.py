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
"""thor_ops"""
import math

from mindspore.ops.primitive import prim_attr_register, PrimitiveWithInfer
from mindspore.common import dtype as mstype
from mindspore import _checkparam as validator
from mindspore.ops.operations.nn_ops import _check_positive_int_or_tuple

__all__ = ["CusBatchMatMul",
           "CusCholeskyTrsm",
           "CusFusedAbsMax1",
           "CusImg2Col",
           "CusMatMulCubeDenseLeft",
           "CusMatMulCubeFraczRightMul",
           "CusMatMulCube",
           "CusMatrixCombine",
           "CusTranspose02314",
           "CusMatMulCubeDenseRight",
           "CusMatMulCubeFraczLeftCast",
           "LoadIm2Col"
           ]


class CusBatchMatMul(PrimitiveWithInfer):
    """
    Multiplies matrix `a` by matrix `b` in batch.

    The rank of input tensors must be `3`.

    Inputs:
        - **input_x** (Tensor) - The first tensor to be multiplied. The shape of the tensor is :math:`(N, D, D)`.
        - **input_y** (Tensor) - The second tensor to be multiplied. The shape of the tensor is :math:`(N, D, D)`. If
          `transpose_b` is True.

    Outputs:
        Tensor, the shape of the output tensor is :math:`(N, D, D)`.

    Examples:
        >>> input_x = Tensor(np.ones(shape=[2, 128, 128]), mindspore.float32)
        >>> input_y = Tensor(np.ones(shape=[2, 128, 128]), mindspore.float32)
        >>> cus_batch_matmul = ops.CusBatchMatMul()
        >>> output = cus_batch_matmul(input_x, input_y)
    """

    @prim_attr_register
    def __init__(self):
        """Initialize CusBatchMatMul"""
        self.init_prim_io_names(inputs=['x1', 'x2'], outputs=['y'])
        from mindspore.ops._op_impl._custom_op.batch_matmul_impl import cus_batch_matmul

    def infer_shape(self, data1_shape, data2_shape):
        return data1_shape

    def infer_dtype(self, data1_dtype, data2_dtype):
        return data1_dtype


class CusCholeskyTrsm(PrimitiveWithInfer):
    r"""
    L * LT = A.
    LT * (LT)^-1 = I.
    return (LT)^-1.
    Only compute the res of the diag part of input matrix with dim 128.
    The rank of input tensors must be `2`.

    Inputs:
        - **input_x** (Tensor) - The first tensor to be multiplied. The shape of the tensor is :math:`(N, N)`.

    Outputs:
        Tensor, the shape of the output tensor is :math:`(N // Split\_dim, Split\_dim, Split\_dim)`.

    Examples:
        >>> input_x = Tensor(np.ones(shape=[256, 256]), mindspore.float32)
        >>> cus_choleskytrsm = ops.CusCholeskyTrsm()
        >>> output = cus_choleskytrsm(input_x)
    """

    @prim_attr_register
    def __init__(self):
        """Initialize CusCholeskyTrsm"""
        self.init_prim_io_names(inputs=['x1'], outputs=['y'])
        from mindspore.ops._op_impl._custom_op.cholesky_trsm_impl import cus_cholesky_trsm

    def infer_shape(self, data1_shape):
        ll = []
        m, _ = data1_shape
        if m >= 128:
            ll = [m // 128, 128, 128]
        else:
            ll = [1, 64, 64]
        return ll

    def infer_dtype(self, data1_dtype):
        return data1_dtype


class CusFusedAbsMax1(PrimitiveWithInfer):
    """
    Computes the abs max of Tensor input.

    The rank of input tensors must be `4` or `2`.
    Inputs:
        - **input_x** (Tensor) - The first tensor to be multiplied. The shape of the tensor is :math:`(N0, M0, N1, M1)`
          or math:`(32, 64)`.
    Outputs:
        Tensor, the shape of the output tensor is :math:`(32, 64)` or math:`(1, )`.

    Examples:
        >>> input_x = Tensor(np.ones(shape=[1, 3]), mindspore.float32)
        >>> cus_fused_abs_max1 = ops.CusFusedAbsMax1()
        >>> output = cus_fused_abs_max1(input_x)
    """

    @prim_attr_register
    def __init__(self, origin_shape=(-1, -1)):
        """Initialize CusFusedAbsMax1"""
        self.init_prim_io_names(inputs=['x1'], outputs=['y'])
        self.origin_shape = origin_shape
        from mindspore.ops._op_impl._custom_op.fused_abs_max1_impl import cus_fused_abs_max1

    def infer_shape(self, data1_shape):
        ll = []
        if len(data1_shape) == 2:
            ll = [1]
        else:
            ll = [32, 64]
        return ll

    def infer_dtype(self, data1_dtype):
        return data1_dtype


class CusImg2Col(PrimitiveWithInfer):
    """
    Img2cols the feature map and the result in reorganized in NC1HWC0.

    Args:
        - **strides** (listInt) - the stride of the ops.
        - **ksizes** (listInt) - the kernel size of the ops.
    Inputs:
        - **input_x** (Tensor) - The shape of the tensor is :math:`(N, C, H, W)`.
    Outputs:
        Tensor, the shape of the output tensor is :math:`(N * H_O * W_O, C1 * K_W * K_H * C0)`.
    Examples:
        >>> input_x = Tensor(np.ones(shape=[32, 3, 224, 224]), mindspore.float16)
        >>> cusimg2col = ops.CusImg2Col()
        >>> output = cusimg2col(input_x)
    """

    @prim_attr_register
    def __init__(self, ksizes, strides, dilates=(1, 1, 1, 1), mode="NC1HWC0"):
        """Initialize CusImg2Col"""
        self.init_prim_io_names(inputs=['x1'], outputs=['y'])
        self.ksizes = ksizes
        self.strides = strides
        self.dilates = dilates
        self.mode = mode
        from mindspore.ops._op_impl._custom_op.img2col_impl import cus_img2col

    def infer_shape(self, data1_shape):
        bs, c, h, w = data1_shape
        _, stride_h, stride_w, _ = self.strides
        _, k_w, k_h, _ = self.ksizes
        c0 = 16
        c1 = c // 16
        if c1 == 0:
            c1 = 1
        shape = [bs * int(h // stride_h) * int(w // stride_w), k_w * k_h * c1 * c0]
        return shape

    def infer_dtype(self, data1_dtype):
        return data1_dtype


class CusMatMulCubeDenseLeft(PrimitiveWithInfer):
    """
    Multiplies matrix `a` by matrix `b`.

    The rank of input_x1 must be `4`, the fractal format of the normal matrix.
    The rank of input_x2 must be `2`.

    Inputs:
        - **input_x1** (Tensor) - The first tensor to be multiplied.
          The shape of the tensor is :math:`(N0, M0, N1, M1)`.
        - **input_x2** (Tensor) - The second tensor to be multiplied. The shape of the tensor is :math:`(M, C)`.
    Outputs:
        Tensor, the shape of the output tensor is :math:`(N, C)`.
    Examples:
        >>> input_x = Tensor(np.ones(shape=[16, 16, 16, 16]), mindspore.float16)
        >>> input_y = Tensor(np.ones(shape=[256, 256]), mindspore.float16)
        >>> matmulcubedenseleft = ops.CusMatMulCubeDenseLeft()
        >>> output = matmulcubedenseleft(input_x, input_y)
    """

    @prim_attr_register
    def __init__(self):
        """Initialize CusMatMulCubeDenseLeft"""
        self.init_prim_io_names(inputs=['x1', 'x2'], outputs=['y'])
        from mindspore.ops._op_impl._custom_op.matmul_cube_dense_left_impl import cus_matmul_cube_dense_left

    def infer_shape(self, data1_shape, data2_shape):
        return data2_shape

    def infer_dtype(self, data1_dtype, data2_dtype):
        return mstype.float16


class CusMatMulCubeFraczRightMul(PrimitiveWithInfer):
    """
    Multiplies matrix `a` by matrix `b` and muls the result by scalar `c`.

    The rank of input_x1 tensors must be `2`.
    The rank of input_x2 tensors must be `4`.

    Inputs:
        - **input_x1** (Tensor) - The first tensor to be multiplied. The shape of the tensor is :math:`(N, C)`.
        - **input_x2** (Tensor) - The second tensor to be multiplied.
          The shape of the tensor is :math:`(C1, M1, C0, M0)`.
        - **input_x3** (Tensor) - The third tensor to be multiplied. The shape of the tensor if :math`(1, )`.
    Outputs:
        Tensor, the shape of the output tensor is :math:`(N, M)`.
    Examples:
        >>> input_x1 = Tensor(np.ones(shape=[256, 256]), mindspore.float16)
        >>> input_x2 = Tensor(np.ones(shape=[16, 16, 16, 16]), mindspore.float16)
        >>> input_x3 = Tensor(np.ones(shape=[1, ]), mindspore.float16)
        >>> cusmatmulfraczrightmul = ops.CusMatMulCubeFraczRightMul()
        >>> output = cusmatmulfraczrightmul(input_x1, input_x2, input_x3)
    """

    @prim_attr_register
    def __init__(self):
        """Initialize CusMatMulCubeFraczRightMul"""
        self.init_prim_io_names(inputs=['x1', 'x2', 'x3'], outputs=['y'])
        from mindspore.ops._op_impl._custom_op.matmul_cube_fracz_right_mul_impl import cus_matmul_cube_fraczrightmul

    def infer_shape(self, data1_shape, data2_shape, data3_shape):
        return data1_shape

    def infer_dtype(self, data1_dtype, data2_dtype, data3_dtype):
        return mstype.float32


class CusMatMulCube(PrimitiveWithInfer):
    """
    Multiplies matrix `a` by matrix `b`.

    The rank of input tensors must be `2`.

    Args:
        transpose_a (bool): If true, `a` is transposed before multiplication. Default: ``False``.
        transpose_b (bool): If true, `b` is transposed before multiplication. Default: ``False``.

    Inputs:
        - **input_x** (Tensor) - The first tensor to be multiplied. The shape of the tensor is :math:`(N, C)`. If
          `transpose_a` is True, its shape must be :math:`(N, C)` after transposing.
        - **input_y** (Tensor) - The second tensor to be multiplied. The shape of the tensor is :math:`(C, M)`. If
          `transpose_b` is True, its shape must be :math:`(C, M)` after transpose.

    Outputs:
        Tensor, the shape of the output tensor is :math:`(N, M)`.

    Examples:
        >>> input_x = Tensor(np.ones(shape=[256, 256]), mindspore.float16)
        >>> input_y = Tensor(np.ones(shape=[256, 256]), mindspore.float16)
        >>> cusmatmulcube = ops.CusMatMulCube()
        >>> output = matmul(input_x, input_y)
    """

    @prim_attr_register
    def __init__(self, transpose_a=False, transpose_b=False):
        """Initialize CusMatMulCube"""
        self.init_prim_io_names(inputs=['x1', 'x2'], outputs=['y'])
        self.transpose_a = transpose_a
        self.transpose_b = transpose_b
        from mindspore.ops._op_impl._custom_op.matmul_cube_impl import cus_matmul_cube

    def infer_shape(self, data1_shape, data2_shape):
        if self.transpose_a:
            _, m = data1_shape
        else:
            m, _ = data1_shape
        if self.transpose_b:
            n, _ = data2_shape
        else:
            _, n = data2_shape
        shape = [m, n]
        return shape

    def infer_dtype(self, data1_dtype, data2_dtype):
        return mstype.float32


class CusMatrixCombine(PrimitiveWithInfer):
    """
    move the batch matrix to result matrix diag part.
    The rank of input tensors must be `3`.

    Inputs:
        - **input_x** (Tensor) - The shape of the tensor is :math:`(N, D, D)`.

    Outputs:
        Tensor, the shape of the output tensor is :math:`(N * D, N * D)`.

    Examples:
        >>> input_x = Tensor(np.ones(shape=[2, 128, 128]), mindspore.float32)
        >>> cusmatrixcombine = ops.CusMatrixCombine()
        >>> output = cusmatrixcombine(input_x)
    """

    @prim_attr_register
    def __init__(self):
        """Initialize CusMatrixCombine"""
        self.init_prim_io_names(inputs=['x'], outputs=['y'])
        from mindspore.ops._op_impl._custom_op.matrix_combine_impl import cus_matrix_combine

    def infer_shape(self, data_shape):
        a, b, c = data_shape
        shape = [a * b, a * c]

        return shape

    def infer_dtype(self, data_dtype):
        return data_dtype


class CusTranspose02314(PrimitiveWithInfer):
    """
    Permute input tensor with perm (0, 2, 3, 1, 4)

    The rank of input tensors must be `5` with format NC1HWC0.

    Inputs:
        - **input_x** (Tensor) - The shape of the tensor is :math:`(N, C1, H, W, C0)`.

    Outputs:
        Tensor, the shape of the output tensor is :math:`(N, H, W, C1, C0)`.

    Examples:
        >>> input_x = Tensor(np.ones(shape=[32, 1, 224, 224, 16]), mindspore.float16)
        >>> custranspose02314 = ops.CusTranspose02314()
        >>> output = custranspose02314(input_x)
    """

    @prim_attr_register
    def __init__(self):
        """Initialize CusTranspose02314"""
        self.init_prim_io_names(inputs=['x1'], outputs=['y'])
        from mindspore.ops._op_impl._custom_op.transpose02314_impl import cus_transpose02314

    def get_bprop(self):
        """Get backprop for CusTranspose02314."""

        def bprop(x, out, dout):
            return (C.zeros_like(x),)

        return bprop

    def infer_shape(self, data1_shape):
        n, c, h, w = data1_shape
        c0 = 16
        c1 = c // 16
        shape = (n * h * w, c1 * c0)
        return shape

    def infer_dtype(self, data1_dtype):
        return data1_dtype


class CusMatMulCubeDenseRight(PrimitiveWithInfer):
    """
    Multiplies matrix `a` by matrix `b`.

    The rank of input_x1 tensor must be `2`.
    The rank of input_x2 tensor must be `4`.

    Inputs:
        - **input_x** (Tensor) - The first tensor to be multiplied. The shape of the tensor is :math:`(N, C)`.
        - **input_y** (Tensor) - The second tensor to be multiplied.
          The shape of the tensor is :math:`(C1, M1, M0, C0)`.

    Outputs:
        Tensor, the shape of the output tensor is :math:`(N, M)`.

    Examples:
        >>> input_x = Tensor(np.ones(shape=[256, 256]), mindspore.float16)
        >>> input_y = Tensor(np.ones(shape=[16, 16, 16, 16]), mindspore.float16)
        >>> cusmatmulcubedenseright = ops.CusMatMulCubeDenseRight()
        >>> output = cusmatmulcubedenseright(input_x, input_y)
    """

    @prim_attr_register
    def __init__(self):
        """Initialize CusMatMulCubeDenseRight"""
        self.init_prim_io_names(inputs=['x1', 'x2', 'x3'], outputs=['y'])
        from mindspore.ops._op_impl._custom_op.matmul_cube_dense_right_impl import cus_matmul_cube_dense_right

    def infer_shape(self, data1_shape, data2_shape, data3_shape):
        return data1_shape

    def infer_dtype(self, data1_dtype, data2_dtype, data3_dtype):
        return mstype.float32


class CusMatMulCubeFraczLeftCast(PrimitiveWithInfer):
    """
    Multiplies matrix `a` by matrix `b`.

    The rank of input_x1 tensor must be `4`.
    The rank of input_x2 tensors must be `2`.

    Inputs:
        - **input_x1** (Tensor) - The first tensor to be multiplied.
          The shape of the tensor is :math:`(C1, N1, N0, C0)`.
        - **input_x2** (Tensor) - The second tensor to be multiplied. The shape of the tensor is :math:`(C, M)`.

    Outputs:
        Tensor, the shape of the output tensor is :math:`(N, M)`.

    Examples:
        >>> input_x = Tensor(np.ones(shape=[16, 16, 16, 16]), mindspore.float16)
        >>> input_y = Tensor(np.ones(shape=[256, 256]), mindspore.float16)
        >>> cusmatmulcubefraczleftcast = ops.CusMatMulCubeFraczLeftCast()
        >>> output = cusmatmulcubefraczleftcast(input_x, input_y)
    """

    @prim_attr_register
    def __init__(self):
        """Initialize CusMatMulCubeFraczLeftCast"""
        self.init_prim_io_names(inputs=['x1', 'x2'], outputs=['y'])
        from mindspore.ops._op_impl._custom_op.matmul_cube_fracz_left_cast_impl import cus_matmul_cube_fraczleftcast

    def infer_shape(self, data1_shape, data2_shape):
        return data2_shape

    def infer_dtype(self, data1_dtype, data2_dtype):
        return mstype.float16


class ThorIm2Col(PrimitiveWithInfer):
    """
    extracts image paths from image.

    The rank of input_x1 must be `4`, data_format is "NCHW".

    Inputs:
        - **input_x1** (Tensor) - The feature map.
          The shape of the tensor is :math:`(N, C, H, W)`.
    Outputs:
        Tensor.
    Examples:
        >>> input_x = Tensor(np.random.rand(32, 3, 224, 224).astype(np.float16))
        >>> img2col = ops.CusMatMulCubeDenseLeft(kernel_size=7, pad=3, stride=2)
        >>> output = img2col(input_x)
    """

    @prim_attr_register
    def __init__(self,
                 kernel_size,
                 pad_mode="valid",
                 pad=0,
                 stride=1,
                 dilation=1):
        """Initialize ThorIm2Col"""
        self.init_prim_io_names(inputs=['x'], outputs=['output'])
        self.kernel_size = _check_positive_int_or_tuple('kernel_size', kernel_size, self.name)
        self.add_prim_attr('kernel_size', self.kernel_size)
        self.stride = _check_positive_int_or_tuple('stride', stride, self.name, allow_four=True, ret_four=True)
        self.add_prim_attr('stride', self.stride)
        self.dilation = _check_positive_int_or_tuple('dilation', dilation, self.name, allow_four=True, ret_four=True)
        self.add_prim_attr('dilation', self.dilation)
        validator.check_value_type('pad', pad, (int,), self.name)
        self.pad_mode = validator.check_string(pad_mode, ['valid', 'same', 'pad'], 'pad_mode', self.name)
        self.pad = validator.check_pad_value_by_mode(pad_mode, pad, self.name)
        if self.pad_mode == 'pad':
            validator.check_non_negative_int(self.pad, 'pad', self.name)
        self.add_prim_attr('data_format', "NCHW")

    def infer_shape(self, x_shape):
        validator.check_equal_int(len(x_shape), 4, "x rank", self.name)
        kernel_size_h = self.kernel_size[0]
        kernel_size_w = self.kernel_size[1]
        stride_h = self.stride[2]
        stride_w = self.stride[3]
        dilation_h = self.dilation[2]
        dilation_w = self.dilation[3]
        if self.pad_mode == "valid":
            h_out = math.ceil((x_shape[2] - dilation_h * (kernel_size_h - 1)) / stride_h)
            w_out = math.ceil((x_shape[3] - dilation_w * (kernel_size_w - 1)) / stride_w)
            pad_top, pad_bottom, pad_left, pad_right = 0, 0, 0, 0
        elif self.pad_mode == "same":
            h_out = math.ceil(x_shape[2] / stride_h)
            w_out = math.ceil(x_shape[3] / stride_w)
            pad_needed_h = max(0, (h_out - 1) * stride_h + dilation_h * (kernel_size_h - 1) + 1 - x_shape[2])
            pad_top = math.floor(pad_needed_h / 2)
            pad_bottom = pad_needed_h - pad_top
            pad_needed_w = max(0, (w_out - 1) * stride_w + dilation_w * (kernel_size_w - 1) + 1 - x_shape[3])
            pad_left = math.floor(pad_needed_w / 2)
            pad_right = pad_needed_w - pad_left
        elif self.pad_mode == 'pad':
            pad_top, pad_bottom, pad_left, pad_right = self.pad, self.pad, self.pad, self.pad
            h_out = 1 + (x_shape[2] + 2 * self.pad - kernel_size_h - (kernel_size_h - 1) * (dilation_h - 1)) / stride_h
            w_out = 1 + (x_shape[3] + 2 * self.pad - kernel_size_w - (kernel_size_w - 1) * (dilation_w - 1)) / stride_w
            h_out = math.floor(h_out)
            w_out = math.floor(w_out)
        self.pad_list = [pad_top, pad_bottom, pad_left, pad_right]
        self.add_prim_attr('pad_list', (pad_top, pad_bottom, pad_left, pad_right))
        batch_size = x_shape[0]
        channel = x_shape[1]
        k_h = kernel_size_h
        k_w = kernel_size_w
        out_shape = [channel, k_h, k_w, batch_size, h_out, w_out]
        return out_shape

    def infer_dtype(self, x_dtype):
        valid_dtypes = [mstype.float16, mstype.float32]
        validator.check_tensor_dtype_valid('x', x_dtype, valid_dtypes, self.name)
        return x_dtype


class NewIm2Col(PrimitiveWithInfer):
    """
    extracts image paths from image by using TBE.

    The rank of input_x1 must be `4`, data_format is "NCHW".

    Inputs:
        - **input_x1** (Tensor) - The feature map.
          The shape of the tensor is :math:`(N, C, H, W)`.
    Outputs:
        Tensor. The shape of the tensor is :math:`(N, H, W, C)`.

    Examples:
        >>> input_x = Tensor(np.random.rand(32, 3, 224, 224).astype(np.float16))
        >>> im2col = ops.NewIm2Col(ksizes=(7,7), strides=2)
        >>> output = im2col(input_x)
    """

    @prim_attr_register
    def __init__(self,
                 ksizes,
                 padding_mode="SAME",
                 strides=1,
                 dilations=1,
                 pads=0):
        """Initialize NewIm2Col"""
        self.init_prim_io_names(inputs=['x'], outputs=['output'])
        self.ksizes = ksizes
        self.strides = strides
        self.add_prim_attr('ksizes', self.ksizes)
        self.add_prim_attr('strides', self.strides)
        self.dilations = dilations
        self.add_prim_attr('dilations', self.dilations)
        self.padding_mode = validator.check_string(padding_mode, ['VALID', 'SAME'], 'padding_mode', self.name)
        self.add_prim_attr('data_format', "NCHW")
        self.pads = pads

    def infer_shape(self, x_shape):
        "infer shape"
        validator.check_equal_int(len(x_shape), 4, "x rank", self.name)
        kernel_size_h = self.ksizes[0]
        kernel_size_w = self.ksizes[1]
        stride_h = self.strides
        stride_w = self.strides
        dilation_h = self.dilations
        dilation_w = self.dilations
        if self.padding_mode == "VALID":
            h_out = math.ceil((x_shape[2] - dilation_h * (kernel_size_h - 1)) / stride_h)
            w_out = math.ceil((x_shape[3] - dilation_w * (kernel_size_w - 1)) / stride_w)
            pad_top, pad_bottom, pad_left, pad_right = 0, 0, 0, 0
        elif self.padding_mode == "SAME":
            h_out = math.ceil(x_shape[2] / stride_h)
            w_out = math.ceil(x_shape[3] / stride_w)
            pad_needed_h = max(0, (h_out - 1) * stride_h + dilation_h * (kernel_size_h - 1) + 1 - x_shape[2])
            pad_top = math.floor(pad_needed_h / 2)
            pad_bottom = pad_needed_h - pad_top
            pad_needed_w = max(0, (w_out - 1) * stride_w + dilation_w * (kernel_size_w - 1) + 1 - x_shape[3])
            pad_left = math.floor(pad_needed_w / 2)
            pad_right = pad_needed_w - pad_left
        self.pad_list = [pad_top, pad_bottom, pad_left, pad_right]
        self.add_prim_attr('pad_list', (pad_top, pad_bottom, pad_left, pad_right))
        batch_size = x_shape[0]
        channel = x_shape[1]
        k_h = kernel_size_h
        k_w = kernel_size_w
        out_shape = [batch_size, h_out, w_out, channel * k_h * k_w]
        return out_shape

    def infer_dtype(self, x_dtype):
        "infer dtype"
        valid_dtypes = [mstype.float16, mstype.int8]
        validator.check_tensor_dtype_valid('x', x_dtype, valid_dtypes, self.name)
        return x_dtype


class LoadIm2Col(PrimitiveWithInfer):
    """
    extracts image patches from image.

    The rank of input_x1 must be `4`, data_format is "NCHW".
    Only supports when C is divisible by 16.

    Inputs:
        - **input_x1** (Tensor) - The feature map.
          The shape of the tensor is :math:`(N, C, H, W)`.
    Outputs:
        Tensor.
    Examples:
        >>> input_x = Tensor(np.random.rand(32, 16, 224, 224).astype(np.float16))
        >>> img2col = ops.LoadIm2Col(kernel_size=(7,7), stride=(2,2))
        >>> output = img2col(input_x)
    """

    @prim_attr_register
    def __init__(self,
                 ksizes,
                 strides,
                 pad_mode="same",
                 dilates=(1, 1, 1, 1)):
        """Initialize LoadIm2Col"""

        self.init_prim_io_names(inputs=['x1'], outputs=['y'])
        self.ksizes = ksizes
        self.strides = strides
        self.pad_mode = validator.check_string(pad_mode, ['same'], 'pad_mode', self.name)
        self.dilation = dilates

    def infer_shape(self, data1_shape):
        bs, c, h, w = data1_shape
        stride_h, stride_w = self.strides
        k_w, k_h = self.ksizes
        h_out = math.ceil(h / stride_h)
        w_out = math.ceil(w / stride_w)
        m = h_out * w_out
        if m % 16 != 0:
            shape = [(bs * m) // 16, (c * k_h * k_w) // 16, 16, 16]
        else:
            shape = [bs, m // 16, (c * k_h * k_w) // 16, 16, 16]
        return shape

    def infer_dtype(self, data1_dtype):
        return data1_dtype


class UpdateThorGradient(PrimitiveWithInfer):
    """
    Updates Thor Gradient with Approximate Fisher info matrix(for GPU backend).

    The rank of input_x1 must be `3`, which indicates the A matrix.
    The rank of input_x2 must be `2`, which indicates the 1st-order gradient.
    The rank of input_x3 must be `4`, which indicates the G matrix.

    Inputs:
        - **input_x1** (Tensor) - The first input is the diag part of the cov matrix of feature map.
          Supported dtype [float32].
        - **input_x2** (Tensor) - The second input is the corresponding 1st-order grad. Supported dtype [float32].
        - **input_x3** (Tensor) - The third input is the diag part of the cov matrix of dout.
          Supported dtype [float32].

    Outputs:
        Tensor, the shape is the same as the shape of input_x2, it will be used to update the weights.

    Examples:
        >>> input_x1 = Tensor(np.random.rand(16, 128, 128).astype(np.float32))
        >>> input_x2 = Tensor(np.random.rand(2048, 1024).astype(np.float32))
        >>> temp_x3 = np.random.rand(8, 128, 128).astype(np.float32)
        >>> input_x3 = np.zeros(16,8,128,128).astype(np.float32)
        >>> for i in range(16):
        ...     input_x3[i,:,:,:] = temp_x3
        >>> input_x3 = Tensor(input_x3)
        >>> update_thor_gradient = ops.UpdateThorGradient(split_dim=128)
        >>> output = update_thor_gradient(input_x1, input_x2, input_x3)
    """

    @prim_attr_register
    def __init__(self, split_dim=1):
        """Initialize UpdateThorGradient"""
        self.init_prim_io_names(inputs=['x1', 'x2', 'x3'], outputs=['y'])
        self.split_dim = split_dim
        self.add_prim_attr('split_dim', self.split_dim)

    def infer_shape(self, x1_shape, x2_shape, x3_shape):
        return x2_shape

    def infer_dtype(self, x1_dtype, x2_dtype, x3_dtype):
        validator.check_tensors_dtypes_same_and_valid(
            {'x1_dtype': x1_dtype, 'x2_dtype': x2_dtype, 'x3_dtype': x3_dtype},
            [mstype.float32], self.name)
        return x2_dtype


class _Cholesky(PrimitiveWithInfer):
    """
    Inner API for _Cholesky base class.
    """

    @prim_attr_register
    def __init__(self, lower=False, clean=True, split_dim=0):
        self.init_prim_io_names(inputs=['x1'], outputs=['y'])
        self.lower = validator.check_value_type("lower", lower, [bool], self.lower)
        self.clean = validator.check_value_type("clean", clean, [bool], self.clean)
        self.lower = lower
        self.add_prim_attr('lower', self.lower)
        self.clean = clean
        self.add_prim_attr('clean', self.clean)
        self.split_dim = split_dim
        self.add_prim_attr('split_dim', self.split_dim)

    def infer_shape(self, x1_shape):
        if self.split_dim != 0:
            height = x1_shape[0]
            width = x1_shape[1]
            if height <= self.split_dim:
                out_shape = [1, height, width]
            else:
                batch = height // self.split_dim
                if height != batch * self.split_dim:
                    batch += 1
                out_shape = [batch, self.split_dim, self.split_dim]
        else:
            out_shape = x1_shape
        return out_shape

    def infer_dtype(self, x1_dtype):
        validator.check_tensor_dtype_valid('x1', x1_dtype, [mstype.float32, mstype.float64], self.name)
        return x1_dtype


class Cholesky(_Cholesky):
    """
    Inner API for positive-definite matrix Cholesky decomposition GPU backend.
    """


class CholeskyTrsm(_Cholesky):
    """
    Inner API for resnet50 THOR GPU backend.
    """


class DetTriangle(PrimitiveWithInfer):
    """
    Calculate the determinant of triangle matrices.

    Args:
        fill_mode (tuple): The target shape to broadcast.

    Inputs:
        - **input_x** (Tensor) - The input tensor.

    Outputs:
        Tensor, with the given `shape` and the same data type as `input_x`.

    Examples:
        >>> shape = (2, 3)
        >>> input_x = Tensor(np.array([1, 2, 3]).astype(np.float32))
        >>> broadcast_to = P.BroadcastTo(shape)
        >>> broadcast_to(input_x)
        [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]
    """

    @prim_attr_register
    def __init__(self, fill_mode=0):
        self.init_prim_io_names(inputs=['x1'], outputs=['y'])
        self.fill_mode = fill_mode
        self.add_prim_attr('fill_mode', self.fill_mode)

    def infer_shape(self, x1_shape):
        out_shape = x1_shape
        del out_shape[-2:]
        return out_shape

    def infer_dtype(self, x1_dtype):
        validator.check_tensor_dtype_valid('x1', x1_dtype, [mstype.float32], self.name)
        return x1_dtype


class ProdForceSeA(PrimitiveWithInfer):
    """
    ProdForceSeA.
    """

    @prim_attr_register
    def __init__(self, natoms=192):
        self.init_prim_io_names(inputs=['net_deriv_tensor', "in_deriv_tensor", "nlist_tensor"], outputs=['y'])
        self.natoms = natoms
        self.add_prim_attr('natoms', self.natoms)

    def infer_shape(self, x1_shape, x2_shape, x3_shape):
        out_shape = [x3_shape[0], x3_shape[1], 3]
        return out_shape

    def infer_dtype(self, x1_dtype, x2_dtype, x3_dtype):
        return x1_dtype
