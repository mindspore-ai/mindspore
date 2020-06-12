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
"""thor_ops"""
from ..primitive import prim_attr_register, PrimitiveWithInfer
from ...common import dtype as mstype


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
        >>> cus_batch_matmul = P.CusBatchMatMul()
        >>> output = cus_batch_matmul(input_x, input_y)
    """

    @prim_attr_register
    def __init__(self):
        """init CusBatchMatMul"""
        self.init_prim_io_names(inputs=['x1', 'x2'], outputs=['y'])
        from mindspore.ops._op_impl._custom_op.batch_matmul_impl import CusBatchMatMul

    def infer_shape(self, data1_shape, data2_shape):
        return data1_shape

    def infer_dtype(self, data1_dtype, data2_dtype):
        return data1_dtype


class CusCholeskyTrsm(PrimitiveWithInfer):
    """
    L * LT = A.
    LT * (LT)^-1 = I.
    return (LT)^-1.
    Only compute the res of the diag part of input matrix with dim 128.
    The rank of input tensors must be `2`.

    Inputs:
        - **input_x** (Tensor) - The first tensor to be multiplied. The shape of the tensor is :math:`(N, N)`.

    Outputs:
        Tensor, the shape of the output tensor is :math:`(N // Split_dim, Split_dim, Split_dim)`.

    Examples:
        >>> input_x = Tensor(np.ones(shape=[256, 256]), mindspore.float32)
        >>> cus_choleskytrsm = P.CusCholeskyTrsm()
        >>> output = matmul(input_x)
    """

    @prim_attr_register
    def __init__(self):
        """init CusCholeskyTrsm"""
        self.init_prim_io_names(inputs=['x1'], outputs=['y'])
        from mindspore.ops._op_impl._custom_op.cholesky_trsm_impl import CusCholeskyTrsm

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
    Compute the abs max of Tensor input.

    The rank of input tensors must be `4` or `2`.
    Inputs:
        - **input_x** (Tensor) - The first tensor to be multiplied. The shape of the tensor is :math:`(N0, M0, N1, M1)`
          or math:`(32, 64)`.
    Outputs:
        Tensor, the shape of the output tensor is :math:`(32, 64)` or math:`(1, )`.

    Examples:
        >>> input_x = Tensor(np.ones(shape=[1, 3]), mindspore.float32)
        >>> cus_fused_abs_max1 = P.CusFusedAbsMax1()
        >>> output = cus_fused_abs_max1(input_x)
    """

    @prim_attr_register
    def __init__(self, origin_shape=[-1, -1]):
        """init CusFusedAbsMax1"""
        self.init_prim_io_names(inputs=['x1'], outputs=['y'])
        self.origin_shape = origin_shape
        from mindspore.ops._op_impl._custom_op.fused_abs_max1_impl import CusFusedAbsMax1

    def infer_shape(self, data1_shape):
        ll = []
        if len(data1_shape) == 2:
            ll = [1,]
        else:
            ll = [32, 64]
        return ll

    def infer_dtype(self, data1_dtype):
        return data1_dtype


class CusImg2Col(PrimitiveWithInfer):
    """
    Img2col the feature map and the result in reorganized in NC1HWC0.

    Args:
        - **strides** (listInt) - the stride of the ops.
        - **ksizes** (listInt) - the kernel size of the ops.
    Inputs:
        - **input_x** (Tensor) - The shape of the tensor is :math:`(N, C, H, W)`.
    Outputs:
        Tensor, the shape of the output tensor is :math:`(N * H_O * W_O, C1 * K_W * K_H * C0)`.
    Examples:
        >>> input_x = Tensor(np.ones(shape=[32, 3, 224, 224]), mindspore.float16)
        >>> cusimg2col = P.CusImg2Col()
        >>> output = cusimg2col(input_x)
    """

    @prim_attr_register
    def __init__(self, ksizes, strides, dilates=(1, 1, 1, 1), mode="NC1HWC0"):
        """init CusImg2Col"""
        self.init_prim_io_names(inputs=['x1'], outputs=['y'])
        self.ksizes = ksizes
        self.strides = strides
        self.dilates = dilates
        self.mode = mode
        from mindspore.ops._op_impl._custom_op.img2col_impl import CusImg2Col

    def infer_shape(self, data1_shape):
        bs, c, h, w = data1_shape
        _, stride_h, stride_w, _ = self.strides
        _, k_w, k_h, _ = self.ksizes
        # assert m == n
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
        >>> matmulcubedenseleft = P.CusMatMulCubeDenseLeft()
        >>> output = matmulcubedenseleft(input_x, input_y)
    """

    @prim_attr_register
    def __init__(self):
        """init CusMatMulCubeDenseLeft"""
        self.init_prim_io_names(inputs=['x1', 'x2'], outputs=['y'])
        from mindspore.ops._op_impl._custom_op.matmul_cube_dense_left_impl import CusMatMulCubeDenseLeft

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
        >>> cusmatmulfraczrightmul = P.CusMatMulCubeFraczRightMul()
        >>> output = cusmatmulfraczrightmul(input_x1, input_x2, input_x3)
    """

    @prim_attr_register
    def __init__(self):
        """init CusMatMulCubeFraczRightMul"""
        self.init_prim_io_names(inputs=['x1', 'x2', 'x3'], outputs=['y'])
        from mindspore.ops._op_impl._custom_op.matmul_cube_fracz_right_mul_impl import CusMatMulCubeFraczRightMul

    def infer_shape(self, data1_shape, data2_shape, data3_shape):
        return data1_shape

    def infer_dtype(self, data1_dtype, data2_dtype, data3_dtype):
        return mstype.float32


class CusMatMulCube(PrimitiveWithInfer):
    """
    Multiplies matrix `a` by matrix `b`.

    The rank of input tensors must be `2`.

    Args:
        transpose_a (bool): If True, `a` is transposed before multiplication. Default: False.
        transpose_b (bool): If True, `b` is transposed before multiplication. Default: False.

    Inputs:
        - **input_x** (Tensor) - The first tensor to be multiplied. The shape of the tensor is :math:`(N, C)`. If
          `transpose_a` is True, its shape should be :math:`(N, C)` after transposing.
        - **input_y** (Tensor) - The second tensor to be multiplied. The shape of the tensor is :math:`(C, M)`. If
          `transpose_b` is True, its shape should be :math:`(C, M)` after transpose.

    Outputs:
        Tensor, the shape of the output tensor is :math:`(N, M)`.

    Examples:
        >>> input_x = Tensor(np.ones(shape=[256, 256]), mindspore.float16)
        >>> input_y = Tensor(np.ones(shape=[256, 256]), mindspore.float16)
        >>> cusmatmulcube = P.CusMatMulCube()
        >>> output = matmul(input_x, input_y)
    """

    @prim_attr_register
    def __init__(self, transpose_a=False, transpose_b=False):
        """init CusMatMulCube"""
        self.init_prim_io_names(inputs=['x1', 'x2'], outputs=['y'])
        self.transpose_a = transpose_a
        self.transpose_b = transpose_b
        from mindspore.ops._op_impl._custom_op.matmul_cube_impl import CusMatMulCube

    def infer_shape(self, data1_shape, data2_shape):
        # shape = [1, data1_shape[1], data2_shape[2], 16, 16]
        # return shape
        if self.transpose_a:
            k1, m = data1_shape
        else:
            m, k1 = data1_shape
        if self.transpose_b:
            n, k2 = data2_shape
        else:
            k2, n = data2_shape
        assert k1 == k2
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
        >>> cusmatrixcombine = P.CusMatrixCombine()
        >>> output = cusmatrixcombine(input_x)
    """

    @prim_attr_register
    def __init__(self):
        """init CusMatrixCombine"""
        self.init_prim_io_names(inputs=['x'], outputs=['y'])
        from mindspore.ops._op_impl._custom_op.matrix_combine_impl import CusMatrixCombine

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
        >>> custranspose02314 = P.CusTranspose02314()
        >>> output = custranspose02314(input_x)
    """

    @prim_attr_register
    def __init__(self):
        """init CusTranspose02314"""
        self.init_prim_io_names(inputs=['x1'], outputs=['y'])
        from mindspore.ops._op_impl._custom_op.transpose02314_impl import CusTranspose02314
    def get_bprop(self):
        def bprop(x, out, dout):
            return (C.zeros_like(x),)

        return bprop

    def infer_shape(self, data1_shape):
        assert len(data1_shape) == 4
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
        >>> cusmatmulcubedenseright = P.CusMatMulCubeDenseRight()
        >>> output = cusmatmulcubedenseright(input_x, input_y)
    """

    @prim_attr_register
    def __init__(self):
        """init CusMatMulCubeDenseRight"""
        self.init_prim_io_names(inputs=['x1', 'x2', 'x3'], outputs=['y'])
        from mindspore.ops._op_impl._custom_op.matmul_cube_dense_right_impl import CusMatMulCubeDenseRight

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
        >>> cusmatmulcubefraczleftcast = P.CusMatMulCubeFraczLeftCast()
        >>> output = cusmatmulcubefraczleftcast(input_x, input_y)
    """

    @prim_attr_register
    def __init__(self):
        """init CusMatMulCubeFraczLeftCast"""
        self.init_prim_io_names(inputs=['x1', 'x2'], outputs=['y'])
        from mindspore.ops._op_impl._custom_op.matmul_cube_fracz_left_cast_impl import CusMatMulCubeFraczLeftCast

    def infer_shape(self, data1_shape, data2_shape):
        return data2_shape

    def infer_dtype(self, data1_dtype, data2_dtype):
        return mstype.float16
