# Copyright 2021-2022 Huawei Technologies Co., Ltd
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

"""Define the grad rules of math related operations."""

import numpy as np
import mindspore.numpy as mnp
from mindspore.common import dtype as mstype
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore import Tensor
from mindspore.ops.operations.math_ops import Polar
from mindspore.ops.operations.math_ops import CumulativeLogsumexp
from mindspore.ops.operations.math_ops import MatrixSolve
from mindspore.ops.operations.math_ops import MatrixSolveLs
from mindspore.ops.operations.math_ops import MatrixTriangularSolve
from mindspore.ops.operations.math_ops import NanToNum
from mindspore.ops.operations.math_ops import FFTWithSize
from mindspore.ops.operations.math_ops import Cholesky
from mindspore.ops.operations.math_ops import CholeskySolve
from mindspore.ops.operations.math_ops import TridiagonalSolve
from mindspore.ops.operations.math_ops import Diagonal
from mindspore.ops.operations.math_ops import EuclideanNorm
from mindspore.ops.operations.array_ops import Transpose, MatrixSetDiagV3
from mindspore.ops.operations._inner_ops import DynamicBroadcastGradientArgs
from mindspore.ops.composite.multitype_ops.zeros_like_impl import zeros_like
from mindspore.ops.primitive import _primexpr
from mindspore.ops._grad_experimental.grad_base import bprop_getters
from mindspore.ops._grad_experimental.grad_base import sum_grad_reduce_axis
from mindspore.ops.operations.array_ops import MatrixBandPart
from mindspore.ops.operations.array_ops import ConjugateTranspose
from mindspore.ops.functional import broadcast_gradient_args


transpose = P.Transpose()
_conj = P.Conj()
shape_op = P.Shape()
reduce_sum = P.ReduceSum()
reshape = P.Reshape()


def _adjoint(a):
    return cholesky_transpose(_conj(a))


def cholesky_transpose(a):
    n = len(a.shape)
    n_range = list(range(0, n))
    n_range[-1] = n - 2
    n_range[-2] = n - 1
    return transpose(a, tuple(n_range))


@_primexpr
def renew_dim(shape, dim):
    """ Re-new dims"""
    new_dim = dim if dim >= 0 else len(shape) + dim
    tmp = [i for i in range(len(shape))]
    _ = tmp.pop(new_dim)
    return tuple(tmp)


@bprop_getters.register(EuclideanNorm)
def get_bprop_euclidean_norm(self):
    """Generate bprop for EuclideanNorm"""
    expand_dims = P.ExpandDims()
    keep_dims = self.keep_dims

    def bprop(x, axes, out, dout):
        scale_v = dout / out
        if not keep_dims and x.shape != ():
            scale_v = expand_dims(scale_v, axes)
        return (x * scale_v, zeros_like(axes))

    return bprop


@bprop_getters.register(CumulativeLogsumexp)
def get_brop_cumulative_logsumexp(self):
    """Generate bprop for CumulativeLogsumexp"""
    exp_op = P.Exp()
    greater_op = P.Greater()
    log_op = P.Log()
    cumulative_op = CumulativeLogsumexp(self.exclusive, not self.reverse)
    less_op = P.Less()
    neg_op = P.Neg()
    cast = P.Cast()

    def bprop(x, axis, out, dout):
        dtype_min = 0
        if x.dtype == mstype.float16:
            dtype_min = cast(np.finfo(np.float16).min, x.dtype)
        else:
            dtype_min = cast(np.finfo(np.float32).min, x.dtype)
        log_grad_positive = mnp.where(greater_op(dout, 0), log_op(dout), dtype_min)
        log_grad_negative = mnp.where(less_op(dout, 0), log_op(neg_op(dout)), dtype_min)
        output_pos = exp_op(cumulative_op(log_grad_positive - out, axis) + x)
        output_neg = exp_op(cumulative_op(log_grad_negative - out, axis) + x)
        return (output_pos - output_neg, zeros_like(x))

    return bprop


@bprop_getters.register(MatrixTriangularSolve)
def get_bprop_matrix_triangular_solve(self):
    """Grad definition for 'MatrixTriangularSolve' operation"""
    adjoint_a = self.adjoint
    lower_a = self.lower
    matrix_triangular_solve_op = P.MatrixTriangularSolve(lower=lower_a, adjoint=not adjoint_a)
    mat_mul_2d_op = P.MatMul()
    mat_mul_op = P.BatchMatMul()
    real_op = P.Real()
    imag_op = P.Imag()
    neg_op = P.Neg()
    complex_op = P.Complex()
    matrix_band_part_op = MatrixBandPart()

    def bprop(matrix, rhs, out, dout):
        grad_rhs = matrix_triangular_solve_op(matrix, dout)
        if matrix.dtype == mstype.complex64 or matrix.dtype == mstype.complex128:
            grad_rhs_temp = _adjoint(grad_rhs)
            out_temp = _adjoint(out)
        else:
            grad_rhs_temp = cholesky_transpose(grad_rhs)
            out_temp = cholesky_transpose(out)
        if adjoint_a:
            if len(matrix.shape) == 2:
                grad_matrix = mat_mul_2d_op(out, grad_rhs_temp)
                grad_matrix = neg_op(grad_matrix)
            else:
                grad_matrix = mat_mul_op(out, grad_rhs_temp)
                grad_matrix = neg_op(grad_matrix)
        else:
            if len(matrix.shape) == 2:
                grad_matrix = mat_mul_2d_op(grad_rhs, out_temp)
                grad_matrix = neg_op(grad_matrix)
            else:
                grad_matrix = mat_mul_op(grad_rhs, out_temp)
                grad_matrix = neg_op(grad_matrix)
        if lower_a:
            if grad_matrix.dtype == mstype.complex64 or grad_matrix.dtype == mstype.complex128:
                grad_matrix_real = matrix_band_part_op(real_op(grad_matrix), -1, 0)
                grad_matrix_imag = matrix_band_part_op(imag_op(grad_matrix), -1, 0)
                grad_matrix = complex_op(grad_matrix_real, grad_matrix_imag)
            else:
                grad_matrix = matrix_band_part_op(grad_matrix, -1, 0)
        else:
            if grad_matrix.dtype == mstype.complex64 or grad_matrix.dtype == mstype.complex128:
                grad_matrix_real = matrix_band_part_op(real_op(grad_matrix), 0, -1)
                grad_matrix_imag = matrix_band_part_op(imag_op(grad_matrix), 0, -1)
                grad_matrix = complex_op(grad_matrix_real, grad_matrix_imag)
            else:
                grad_matrix = matrix_band_part_op(grad_matrix, 0, -1)
        return (grad_matrix, grad_rhs)

    return bprop


@bprop_getters.register(MatrixSolve)
def get_bprop_matrix_solve(self):
    """Generate bprop for MatrixSolve"""
    adjoint = self.adjoint
    adjoint_a = not adjoint
    solve_op = MatrixSolve(adjoint_a)
    batchmatmul = P.BatchMatMul(transpose_b=True)
    matmul = P.MatMul(transpose_b=True)
    neg = P.Neg()
    cast = P.Cast()
    rank = P.Rank()

    def bprop(input_a, input_b, out, dout):
        out_type = F.dtype(out)
        if out_type == mstype.float64:
            out = cast(out, mstype.float32)
        grad_b = solve_op(input_a, dout)
        grad_b_type = F.dtype(grad_b)
        if grad_b_type == mstype.float64:
            grad_b = cast(grad_b, mstype.float32)

        matrix_rank = rank(input_a)

        if adjoint:
            if matrix_rank > 2:
                grad_a = batchmatmul(out, grad_b)
                grad_a = neg(grad_a)
            else:
                grad_a = matmul(out, grad_b)
                grad_a = neg(grad_a)
        else:
            if matrix_rank > 2:
                grad_a = batchmatmul(grad_b, out)
                grad_a = neg(grad_a)
            else:
                grad_a = matmul(grad_b, out)
                grad_a = neg(grad_a)
        return grad_a, grad_b

    return bprop


@_primexpr
def _generate_perm_matrix_solve_ls(x_dim):
    perm = tuple(range(x_dim - 2))
    perm = perm + (x_dim-1, x_dim-2)
    return perm


@bprop_getters.register(MatrixSolveLs)
def get_bprop_matrix_solve_ls(self):
    """Grad definition for 'MatrixSolveLs' operation"""
    fast = self.fast
    cast = P.Cast()
    neg = P.Neg()
    rank = P.Rank()
    cholesky = Cholesky()
    eye = P.Eye()
    add = P.Add()
    mul = P.Mul()
    matmul = P.MatMul()
    batch_matmul = P.BatchMatMul()
    cholesky_solve = CholeskySolve()
    _transpose = Transpose()
    conjugate_transpose = ConjugateTranspose()
    shape = P.Shape()
    _complex = P.Complex()
    scalar2tensor = P.ScalarToTensor()

    def regularized_gramian_cholesky(matrix, l2, first_kind):
        matrix_dim = rank(matrix)
        perm = _generate_perm_matrix_solve_ls(matrix_dim)
        if matrix.dtype in (mstype.complex64, mstype.complex128):
            matrix_temp = conjugate_transpose(matrix, perm)
        else:
            matrix_temp = _transpose(matrix, perm)
        if first_kind:
            if matrix_dim > 2:
                gramian = batch_matmul(matrix_temp, matrix)
            else:
                gramian = matmul(matrix_temp, matrix)
        else:
            if matrix_dim > 2:
                gramian = batch_matmul(matrix, matrix_temp)
            else:
                gramian = matmul(matrix, matrix_temp)
        if isinstance(l2, Tensor) or l2 != 0:
            matrix_shape = shape(matrix)
            if first_kind:
                small_dim = matrix_shape[-1]
            else:
                small_dim = matrix_shape[-2]
            identity = eye(small_dim, small_dim, matrix.dtype)
            gramian = add(gramian, mul(l2, identity))

        #Cholesky not support complex dtype for now
        return cholesky(gramian)

    def bprop(matrix, rhs, l2, out, dout):
        #support dtype:float32
        #support dimension: 2D,3D
        def over_determined(matrix, rhs, out, l2, dout):
            if matrix.dtype == mstype.complex64:
                l2_regularizer = _complex(cast(l2, mstype.float32), Tensor(0, mstype.float32))
            elif matrix.dtype == mstype.complex128:
                l2_regularizer = _complex(cast(l2, mstype.float64), Tensor(0, mstype.float64))
            else:
                l2_regularizer = cast(l2, matrix.dtype)
            chol = cast(regularized_gramian_cholesky(matrix, l2_regularizer, first_kind=True), matrix.dtype)
            #CholeskySolve not support complex dtype and just support 2D or 3D matrices for now
            z = cholesky_solve(dout, chol)

            matrix_dim = rank(matrix)
            perm = _generate_perm_matrix_solve_ls(matrix_dim)
            if matrix.dtype in (mstype.complex64, mstype.complex128):
                z_temp = conjugate_transpose(z, perm)
            else:
                z_temp = _transpose(z, perm)
            if matrix_dim > 2:
                xzt = batch_matmul(out, z_temp)
            else:
                xzt = matmul(out, z_temp)
            zx_sym = add(xzt, _transpose(xzt, perm))

            if matrix_dim > 2:
                grad_a = add(neg(batch_matmul(matrix, zx_sym)), batch_matmul(rhs, z_temp))
                grad_b = batch_matmul(matrix, z)
            else:
                grad_a = add(neg(matmul(matrix, zx_sym)), matmul(rhs, z_temp))
                grad_b = matmul(matrix, z)

            return (grad_a, grad_b, scalar2tensor(0, l2.dtype))

        def under_determined(matrix, rhs, l2, dout):
            if matrix.dtype == mstype.complex64:
                l2_regularizer = _complex(cast(l2, mstype.float32), Tensor(0, mstype.float32))
            elif matrix.dtype == mstype.complex128:
                l2_regularizer = _complex(cast(l2, mstype.float64), Tensor(0, mstype.float64))
            else:
                l2_regularizer = cast(l2, matrix.dtype)
            chol = cast(regularized_gramian_cholesky(matrix, l2_regularizer, first_kind=False), matrix.dtype)

            matrix_dim = rank(matrix)
            perm = _generate_perm_matrix_solve_ls(matrix_dim)
            if matrix_dim > 2:
                gramian = batch_matmul(matrix, dout)
            else:
                gramian = matmul(matrix, dout)
            #CholeskySolve not support complex dtype and just support 2D or 3D matrices for now
            grad_b = cholesky_solve(gramian, chol)
            tmp = cholesky_solve(rhs, chol)

            if matrix.dtype in (mstype.complex64, mstype.complex128):
                tmp_temp = conjugate_transpose(tmp, perm)
                matrix_temp = conjugate_transpose(matrix, perm)
            else:
                tmp_temp = _transpose(tmp, perm)
                matrix_temp = _transpose(matrix, perm)
            if matrix_dim > 2:
                a1 = batch_matmul(tmp_temp, matrix)
                a1 = neg(batch_matmul(grad_b, a1))
                a2 = dout - batch_matmul(matrix_temp, grad_b)
                if matrix.dtype in (mstype.complex64, mstype.complex128):
                    a2_temp = conjugate_transpose(a2, perm)
                else:
                    a2_temp = _transpose(a2, perm)
                a2 = batch_matmul(tmp, a2_temp)
            else:
                a1 = matmul(tmp_temp, matrix)
                a1 = neg(matmul(grad_b, a1))
                a2 = dout - matmul(matrix_temp, grad_b)
                if matrix.dtype in (mstype.complex64, mstype.complex128):
                    a2_temp = conjugate_transpose(a2, perm)
                else:
                    a2_temp = _transpose(a2, perm)
                a2 = matmul(tmp, a2_temp)

            grad_a = add(a1, a2)
            return (grad_a, grad_b, scalar2tensor(0, l2.dtype))

        if fast is False:
            raise ValueError("For MatrixSolveLs, gradient not defined for fast=False")
        matrix_shape = shape(matrix)[-2:]

        if matrix_shape[-2] >= matrix_shape[-1]:
            return over_determined(matrix, rhs, out, l2, dout)

        return under_determined(matrix, rhs, l2, dout)

    return bprop


@bprop_getters.register(NanToNum)
def get_bprop_nan_to_num(self):
    """Grad definition for `NanToNum` operation."""
    isfinite = P.IsFinite()

    def bprop(x, out, dout):
        dx = dout * isfinite(x)
        return (dx,)

    return bprop


@bprop_getters.register(Polar)
def get_bprop_polar(self):
    """Grad definition for `Polar` operation."""
    complex_op = Complex()
    conj = P.Conj()
    real = P.Real()
    sig = P.Sign()
    ones = P.Ones()
    zeros = P.Zeros()

    def bprop(input1, angle, out, dout):
        grad_conj = conj(dout)
        zero = zeros(dout.shape, input1.dtype)
        one = ones(dout.shape, input1.dtype)
        i = complex_op(zero, one)
        grad_abs = real(grad_conj * sig(out))
        result_mul_1_j = out * i
        grad_angle = real(grad_conj * result_mul_1_j)
        return (grad_abs, grad_angle)

    return bprop


@bprop_getters.register(TridiagonalSolve)
def get_bprop_tridiagonalsolve(self):
    """Grad definition for 'TridiagonalSolve' operation"""
    tridiagonalsolve = TridiagonalSolve()

    def bprop(diagonals, rhs, out, dout):
        diags = diagonals
        diag1 = diags[..., 1, :]
        zeros1 = P.Zeros()(diags.shape[:-2] + (1,), diags.dtype)
        superdiag1 = P.Concat(-1)((diags[..., 2, 1:], zeros1))
        subdiag1 = P.Concat(-1)((zeros1, diags[..., 0, :-1]))
        diags_transposed = P.Stack(-2)([superdiag1, diag1, subdiag1])
        grad_rhs = tridiagonalsolve(diags_transposed, dout)
        diag2 = P.ReduceSum()(grad_rhs * out, -1)
        zeros2 = P.Zeros()(grad_rhs.shape[:-2] + (1, grad_rhs.shape[-1]), grad_rhs.dtype)
        superdiag2 = P.ReduceSum()(grad_rhs * P.Concat(-2)((out[..., 1:, :], zeros2)), -1)
        subdiag2 = P.ReduceSum()(grad_rhs * P.Concat(-2)((zeros2, out[..., :-1, :])), -1)
        a = (P.Stack(-2)([superdiag2, diag2, subdiag2]))
        grad_diags = 0 - a
        return grad_diags, grad_rhs

    return bprop


@bprop_getters.register(CholeskySolve)
def get_bprop_cholesky_solve(self):
    """Grad definition for 'CholeskySolve' operation"""
    batchmatmul_op = P.BatchMatMul()
    matmul_op = P.MatMul()
    neg_op = P.Neg()
    upper = self.upper
    cholesky_solve = CholeskySolve(upper=self.upper)
    rank = P.Rank()

    def bprop(x1, x2, out, dout):
        flag = 0
        len_x1 = rank(x1)
        if dout.dtype == mstype.float64:
            flag = 1
            x2 = F.cast(x2, mstype.float32)
            out = F.cast(out, mstype.float32)
            dout = F.cast(dout, mstype.float32)
        dx1 = cholesky_solve(dout, x2)
        if len_x1 == 2:
            common_term = matmul_op(dx1, transpose(out, (1, 0)))
            common_term = common_term + transpose(common_term, (1, 0))
            if upper is True:
                dx2 = neg_op(matmul_op(x2, common_term))
            else:
                dx2 = neg_op(matmul_op(common_term, x2))
        else:
            x2_dim_size = len(shape_op(x2))
            x2_dim_order = list(range(x2_dim_size))
            target_order = x2_dim_order[:-2] + x2_dim_order[-2:][::-1]
            common_term = batchmatmul_op(dx1, transpose(out, tuple(target_order)))
            common_term = common_term + transpose(common_term, tuple(target_order))
            if upper is True:
                dx2 = neg_op(batchmatmul_op(x2, common_term))
            else:
                dx2 = neg_op(batchmatmul_op(common_term, x2))
        if flag == 1:
            dx1 = F.cast(dx1, mstype.float64)
            dx2 = F.cast(dx2, mstype.float64)
        return dx1, dx2

    return bprop


@bprop_getters.register(Diagonal)
def get_bprop_diagonal(self):
    """Grad definition for 'Diagonal' operation"""
    offset = self.offset
    dim1 = self.dim1
    dim2 = self.dim2
    zeros_op = P.FillV2()
    size_op = P.Size()
    transpose_op = Transpose()
    matrix_set_diag_op = MatrixSetDiagV3(align="LEFT_RIGHT")

    def bprop(x, out, dout):
        x_shape = x.shape
        x_dtype = x.dtype
        x_dim = len(x_shape)
        if dim1 < 0:
            dim1_ = dim1 + x_dim
        else:
            dim1_ = dim1
        if dim2 < 0:
            dim2_ = dim2 + x_dim
        else:
            dim2_ = dim2
        if size_op(out):
            batch_dim = out.shape[:-1]
            diag_plane = (x_shape[dim1_], x_shape[dim2_])
            dx_trans_shape = batch_dim + diag_plane
            value = Tensor(0, x_dtype)
            dx = zeros_op(dx_trans_shape, value)
            k = F.cast(offset, mstype.int32)
            dx = matrix_set_diag_op(dx, dout, k)
            dim = 0
            perm = ()
            for i in range(x_dim):
                if i == dim1_:
                    perm = perm + (x_dim - 2,)
                elif i == dim2_:
                    perm = perm + (x_dim - 1,)
                else:
                    perm = perm + (dim,)
                    dim = dim + 1
            dx = transpose_op(dx, perm)
        else:
            dx = zeros_like(x)
        return (dx,)

    return bprop


@_primexpr
def _fft_rank_offset(norm_shape, rank):
    """generate offset for fft with rank"""
    norm_shape_product = 1
    for i in norm_shape[-rank:]:
        norm_shape_product *= i
    return norm_shape_product


@_primexpr
def _fft_with_size_back_norm(norm_shape, norm, inverse, rank):
    """generate reverse term for fft_with_size"""
    if inverse is False:
        if norm == "forward":
            norm_ = 1 / _fft_rank_offset(norm_shape, rank)
        if norm == "backward":
            norm_ = 1 * _fft_rank_offset(norm_shape, rank)
        if norm == "ortho":
            norm_ = 1
    if inverse is True:
        if norm == "forward":
            norm_ = 1 * _fft_rank_offset(norm_shape, rank)
        if norm == "backward":
            norm_ = 1 / _fft_rank_offset(norm_shape, rank)
        if norm == "ortho":
            norm_ = 1
    return norm_


@_primexpr
def _rfft_norm(norm_shape, norm, rank):
    """generate norm for rfft"""
    norm_ = 1.0
    if norm == "forward":
        norm_ = 1 / _fft_rank_offset(norm_shape, rank)
    if norm == "backward":
        norm_ = 1
    if norm == "ortho":
        norm_ = 1 / np.sqrt(_fft_rank_offset(norm_shape, rank))
    return norm_


@_primexpr
def _get_last_dim_slice_shape(tensor_shape, index):
    """generate shape for slice last tensor"""
    from_shape = [0 for x in tensor_shape]
    if index < 0:
        from_shape[-1] = tensor_shape[-1] + index
    else:
        from_shape[-1] = index
    to_shape = list(tensor_shape)
    to_shape[-1] = 1
    return tuple(from_shape), tuple(to_shape)


@_primexpr
def _rfft_reshape(shape_a, shape_b):
    """generate rfft shape for reshape"""
    new_shape = list(shape_b)
    for i in range(len(shape_a) - 2):
        new_shape.insert(i, 1)
    return tuple(new_shape)


@_primexpr
def _rfft_tile_reshape(shape_a):
    """generate rfft shape for tile"""
    reshape_a = list(shape_a)
    reshape_a[-2] = 1
    reshape_a[-1] = 1
    return tuple(reshape_a)


@_primexpr
def _rfft_last_term_shape(shape_a, shape_b):
    """generate rfft shape for last term"""
    new_shape = list(shape_b)
    for i in range(len(shape_a) - 1):
        new_shape.insert(i, 1)
    return tuple(new_shape)


@_primexpr
def _batch_matmul_shape_increase(shape_before):
    """increase tensor shape for batch_matmul"""
    return (1, *shape_before)


@_primexpr
def _batch_matmul_shape_decrease(matrix_shape):
    """decrease tensor shape after batch_matmul"""
    shape_tmp = list(matrix_shape)
    shape_tmp[-1] = 1
    return tuple(shape_tmp)


@bprop_getters.register(FFTWithSize)
def get_bprop_fft_with_size(self):
    """Grad definition for `FFTWithSize` operation."""
    signal_ndim = self.signal_ndim
    inverse = self.inverse
    real = self.real
    norm = self.norm
    onesided = self.onesided
    fft_fn = FFTWithSize(signal_ndim=signal_ndim,
                         inverse=False,
                         real=False,
                         norm=norm)
    ifft_fn = FFTWithSize(signal_ndim=signal_ndim,
                          inverse=True,
                          real=False,
                          norm=norm)
    rfft_fn = FFTWithSize(signal_ndim=signal_ndim,
                          inverse=False,
                          real=True,
                          norm=norm,
                          onesided=onesided)
    irfft_fn = FFTWithSize(signal_ndim=signal_ndim,
                           inverse=True,
                           real=True,
                           norm=norm,
                           onesided=onesided)

    complex_op = P.Complex()
    to_tensor_op = P.ScalarToTensor()
    type_op = P.DType()
    concat_op = P.Concat()
    ones_op = P.Ones()
    zeros_op = P.Zeros()
    real_op = P.Real()
    imag_op = P.Imag()
    slice_op = P.Slice()
    tile_op = P.Tile()
    expand_dims = P.ExpandDims()
    transpose_op = P.Transpose()
    exp_op = P.Exp()
    reshape_op = P.Reshape()
    conj_op = P.Conj()
    batch_matmul_op = P.BatchMatMul()

    def bprop(x, out, dout):
        dx = 0
        input_type = type_op(x)
        output_type = type_op(out)
        input_shape = shape_op(x)
        offset_shape = shape_op(x)
        dout_shape = shape_op(dout)
        offset_size = to_tensor_op(_fft_with_size_back_norm(offset_shape, norm, inverse, signal_ndim), output_type)
        if real is False:
            if inverse is False:
                dx = ifft_fn(dout) * offset_size
            else:
                dx = fft_fn(dout) * offset_size
        else:
            irfft_ = FFTWithSize(signal_ndim=1, inverse=True, real=True, norm="backward", onesided=onesided,
                                 signal_sizes=offset_shape[-1:])
            irfft2d_ = FFTWithSize(signal_ndim=2, inverse=True, real=True, norm="backward", onesided=onesided,
                                   signal_sizes=offset_shape[-2:])
            irfft3d_ = FFTWithSize(signal_ndim=3, inverse=True, real=True, norm="backward", onesided=onesided,
                                   signal_sizes=offset_shape[-3:])
            if inverse is False:
                if onesided is True:
                    terms = 0
                    is_even = to_tensor_op(1 - (input_shape[-1] % 2), input_type)
                    dout_first_from, dout_first_to = _get_last_dim_slice_shape(dout_shape, 0)
                    dout_first = slice_op(dout, dout_first_from, dout_first_to)
                    rfft_offset_size = to_tensor_op(_fft_rank_offset(input_shape, signal_ndim), input_type)
                    rfft_norm_offset = to_tensor_op(_rfft_norm(input_shape, norm, signal_ndim), input_type)
                    dout_last_from, dout_last_to = _get_last_dim_slice_shape(dout_shape, -1)
                    dout_last = slice_op(dout, dout_last_from, dout_last_to)
                    if signal_ndim == 1:
                        dx = irfft_(dout)
                        vec_mask = complex_op(1 - 2 * (mnp.arange(0, input_shape[-1], 1, input_type) % 2),
                                              zeros_op(input_shape[-1], input_type))
                        terms = real_op(dout_first) + is_even * real_op(dout_last * vec_mask)
                    elif signal_ndim == 2:
                        dx = irfft2d_(dout)
                        arange_inner = mnp.arange(0, input_shape[-2], 1, input_type)
                        matrix_a = tile_op(expand_dims(arange_inner, 0), (input_shape[-2], 1))
                        matrix_b = transpose_op(matrix_a, (1, 0))
                        matrix_mul = matrix_a * matrix_b
                        imag_offset = complex_op(to_tensor_op(0, input_type), to_tensor_op(-2, input_type))
                        pi_tensor = to_tensor_op(mnp.pi, output_type)
                        matrix_mul_complex = complex_op(matrix_mul, zeros_op(shape_op(matrix_mul), input_type))
                        matrix_base_mask = exp_op(imag_offset * pi_tensor * matrix_mul_complex /
                                                  to_tensor_op(input_shape[-2], output_type))
                        expanded_matrix_mask = reshape_op(matrix_base_mask, _rfft_reshape(shape_op(dout_first),
                                                                                          shape_op(matrix_base_mask)))
                        tile_matrix_mask = complex_op(tile_op(real_op(expanded_matrix_mask), _rfft_tile_reshape(
                            shape_op(dout_first))), tile_op(imag_op(expanded_matrix_mask),
                                                            _rfft_tile_reshape(shape_op(dout_first))))
                        tile_matrix_mask_shape = shape_op(tile_matrix_mask)
                        dout_first_term = reshape_op(batch_matmul_op(reshape_op(tile_matrix_mask,
                                                                                _batch_matmul_shape_increase(
                                                                                    tile_matrix_mask_shape)),
                                                                     reshape_op(conj_op(
                                                                         dout_first), _batch_matmul_shape_increase(
                                                                             shape_op(dout_first)))),
                                                     _batch_matmul_shape_decrease(tile_matrix_mask_shape))
                        dout_last_term = reshape_op(batch_matmul_op(reshape_op(tile_matrix_mask,
                                                                               _batch_matmul_shape_increase(
                                                                                   tile_matrix_mask_shape)),
                                                                    reshape_op(conj_op(dout_last),
                                                                               _batch_matmul_shape_increase(
                                                                                   shape_op(dout_last)))),
                                                    _batch_matmul_shape_decrease(
                                                        tile_matrix_mask_shape))
                        vec_mask = complex_op(1 - 2 * (mnp.arange(0, input_shape[-1], 1, input_type) % 2), zeros_op(
                            input_shape[-1], input_type))
                        dout_last_term = complex_op(tile_op(real_op(dout_last_term), _rfft_last_term_shape(dout_shape,
                                                                                                           [input_shape[
                                                                                                               -1]])),
                                                    tile_op(imag_op(dout_last_term), _rfft_last_term_shape(
                                                        dout_shape, [input_shape[-1]])))
                        dout_last_term = dout_last_term * vec_mask
                        terms = real_op(dout_first_term) + is_even * real_op(dout_last_term)
                    elif signal_ndim == 3:
                        dx = irfft3d_(dout) * real_op(offset_size)
                    dx = to_tensor_op(0.5, input_type) * (dx * rfft_offset_size + terms) * rfft_norm_offset
                else:
                    dx = irfft_fn(dout) * real_op(offset_size)
            else:
                dx = rfft_fn(dout)
                if onesided is True:
                    if signal_ndim != 3:
                        is_odd = dout_shape[-1] % 2
                        last_shape = offset_shape[-1]
                        mask = concat_op((ones_op(1, output_type), 2.0 * ones_op(
                            (last_shape - 2 + is_odd,), output_type), ones_op((1 - is_odd,), output_type)))
                        dx = dx * complex_op(mask, zeros_op(shape_op(mask), output_type))
                        irfft_offset_size = to_tensor_op(
                            _fft_with_size_back_norm(shape_op(dout), norm, inverse, signal_ndim),
                            output_type)
                        dx = dx * complex_op(irfft_offset_size, zeros_op(1, output_type))
                    else:
                        dx = dx * complex_op(offset_size, zeros_op(1, output_type))
                else:
                    dx = dx * complex_op(offset_size, zeros_op(1, output_type))
        return (dx,)

    return bprop


def dyn_binop_grad_common(x, y, dx, dy):
    """
    Common grad definition for binary operations when the input is dynamic shape.

    The function is usually used in backprop op to reduce additional dimensions created by broadcasting.
    """
    shape_of_x = shape_op(x)
    shape_of_y = shape_op(y)
    rx, ry = DynamicBroadcastGradientArgs()(shape_of_x, shape_of_y)
    dx_origin_dtype = dx.dtype
    if dx_origin_dtype in (mstype.int16, mstype.int32, mstype.int64):
        dx = F.cast(dx, mstype.float32)
        dx = sum_grad_reduce_axis(dx, rx)
        dx = F.cast(dx, dx_origin_dtype)
    else:
        dx = sum_grad_reduce_axis(dx, rx)
    dy_origin_dtype = dy.dtype
    if dy_origin_dtype in (mstype.int16, mstype.int32, mstype.int64):
        dy = F.cast(dy, mstype.float32)
        dy = sum_grad_reduce_axis(dy, ry)
        dy = F.cast(dy, dy_origin_dtype)
    else:
        dy = sum_grad_reduce_axis(dy, ry)
    reduce_dx = reshape(dx, shape_of_x)
    reduce_dy = reshape(dy, shape_of_y)
    return reduce_dx, reduce_dy


def dyn_binop_grad_common_with_shift(x, y, dx, dy, shift):
    """
    Common grad definition for binary operations with shift when the input is dynamic shape.

    The function is usually used in backprop op to reduce additional dimensions created by broadcasting.
    """
    shape_of_x = shape_op(x)
    shape_of_y = shape_op(y)
    broadcast_shape_of_x = shape_of_x[:-shift]
    broadcast_shape_of_y = shape_of_y[:-shift]
    rx, ry = DynamicBroadcastGradientArgs()(broadcast_shape_of_x, broadcast_shape_of_y)
    dx = sum_grad_reduce_axis(dx, rx)
    dy = sum_grad_reduce_axis(dy, ry)
    reduce_dx = reshape(dx, shape_of_x)
    reduce_dy = reshape(dy, shape_of_y)
    return reduce_dx, reduce_dy


def _reduce_sum_with_cast(dx, axis):
    dx_origin_dtype = dx.dtype
    # Currently, for Ascend and GPU, the reduce_sum's input does not support int16, int32 and int64.
    if dx_origin_dtype in (mstype.int16, mstype.int32, mstype.int64):
        dx = F.cast(dx, mstype.float32)
        dx = reduce_sum(dx, axis)
        return F.cast(dx, dx_origin_dtype)
    return reduce_sum(dx, axis)


def binop_grad_common(x, y, dx, dy):
    """
    Common grad definition for binary operations.

    The function is usually used in backprop op to reduce additional dimensions created by broadcasting.
    """
    shape_of_x = shape_op(x)
    shape_of_y = shape_op(y)
    # if input shape is the same as dout shape, do not need to reduce
    reduce_dx = dx
    reduce_dy = dy
    if not (F.is_sequence_value_unknown(shape_of_x) or F.is_sequence_value_unknown(shape_of_y)):
        rx = broadcast_gradient_args(shape_of_x, shape_of_y)
        if rx[0]:
            # if dx is scalar whose shape is (), do not need reduce
            if shape_op(dx):
                dx = _reduce_sum_with_cast(dx, rx[0])
            reduce_dx = reshape(dx, shape_of_x)
        if rx[1]:
            # if dy is scalar whose shape is (), do not need reduce
            if shape_op(dy):
                dy = _reduce_sum_with_cast(dy, rx[1])
            reduce_dy = reshape(dy, shape_of_y)
        return reduce_dx, reduce_dy

    if not isinstance(shape_of_x, tuple) or not isinstance(shape_of_y, tuple):
        # x or y is scalar
        if not isinstance(shape_of_x, tuple):
            reduce_dx = _reduce_sum_with_cast(dx, ())
        if not isinstance(shape_of_y, tuple):
            reduce_dy = _reduce_sum_with_cast(dy, ())
        return reduce_dx, reduce_dy

    return dyn_binop_grad_common(x, y, dx, dy)


def binop_grad_common_with_shift(x, y, dx, dy, shift):
    """
    Common grad definition for binary operations with shift.

    The function is usually used in backprop op to reduce additional dimensions created by broadcasting.
    """
    shape_of_x = shape_op(x)
    shape_of_y = shape_op(y)
    broadcast_shape_of_x = shape_of_x[:-shift]
    broadcast_shape_of_y = shape_of_y[:-shift]
    # if input shape is the same as dout shape, do not need to reduce
    reduce_dx = dx
    reduce_dy = dy
    if not (F.is_sequence_value_unknown(broadcast_shape_of_x) or F.is_sequence_value_unknown(broadcast_shape_of_y)):
        rx = broadcast_gradient_args(broadcast_shape_of_x, broadcast_shape_of_y)
        if rx[0]:
            # if dx is scalar whose shape is (), do not need reduce
            if shape_op(dx):
                dx = _reduce_sum_with_cast(dx, rx[0])
            reduce_dx = reshape(dx, shape_of_x)
        if rx[1]:
            # if dy is scalar whose shape is (), do not need reduce
            if shape_op(dy):
                dy = _reduce_sum_with_cast(dy, rx[1])
            reduce_dy = reshape(dy, shape_of_y)
        return reduce_dx, reduce_dy

    if not isinstance(shape_of_x, tuple) or not isinstance(shape_of_y, tuple):
        # x or y is scalar
        if not isinstance(shape_of_x, tuple):
            reduce_dx = _reduce_sum_with_cast(dx, ())
        if not isinstance(shape_of_y, tuple):
            reduce_dy = _reduce_sum_with_cast(dy, ())
        return reduce_dx, reduce_dy

    return dyn_binop_grad_common_with_shift(x, y, dx, dy, shift)


@bprop_getters.register(P.TensorAdd)
def get_bprop_tensor_add(self):
    """Grad definition for `Add` operation."""

    def bprop(x, y, out, dout):
        return binop_grad_common(x, y, dout, dout)

    return bprop
