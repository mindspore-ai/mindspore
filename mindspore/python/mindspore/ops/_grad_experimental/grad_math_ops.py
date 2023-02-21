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
from mindspore import context
from mindspore.common import dtype as mstype
from mindspore.nn import LGamma
from mindspore.ops import functional as F
from mindspore.ops.functional import broadcast_gradient_args
from mindspore.ops import operations as P
from mindspore.ops.operations import _inner_ops as inner
from mindspore.ops.operations.math_ops import Trace, Bernoulli, Renorm
from mindspore import nn, ops, Tensor
from mindspore.ops.operations.math_ops import Real, Imag, Complex, Angle
from mindspore.ops.operations.math_ops import Polar
from mindspore.ops.operations.math_ops import ComplexAbs
from mindspore.ops.operations.math_ops import Sinc
from mindspore.ops.operations import _grad_ops as G
from mindspore.ops.operations.math_ops import Igamma, Igammac
from mindspore.ops.operations.math_ops import BesselI0
from mindspore.ops.operations.math_ops import BesselI1
from mindspore.ops.operations.math_ops import BesselJ0
from mindspore.ops.operations.math_ops import BesselJ1
from mindspore.ops.operations.math_ops import BesselK0
from mindspore.ops.operations.math_ops import BesselK1
from mindspore.ops.operations.math_ops import BesselK0e
from mindspore.ops.operations.math_ops import BesselK1e
from mindspore.ops.operations.math_ops import BesselY0
from mindspore.ops.operations.math_ops import BesselY1
from mindspore.ops.operations.math_ops import Lgamma
from mindspore.ops.operations.math_ops import Digamma
from mindspore.ops.operations.math_ops import Polygamma
from mindspore.ops.operations.math_ops import NextAfter
from mindspore.ops.operations.math_ops import Hypot
from mindspore.ops.operations.math_ops import ReduceStd
from mindspore.ops.operations.math_ops import LuUnpack
from mindspore.ops.operations.math_ops import MatrixExp
from mindspore.ops.operations.math_ops import CumulativeLogsumexp
from mindspore.ops.operations.math_ops import MatrixSolve
from mindspore.ops.operations.math_ops import MatrixSolveLs
from mindspore.ops.operations.math_ops import MatrixPower
from mindspore.ops.operations.math_ops import Median
from mindspore.ops.operations.math_ops import MatrixTriangularSolve
from mindspore.ops.operations.math_ops import NanToNum
from mindspore.ops.operations.math_ops import FFTWithSize
from mindspore.ops.operations.math_ops import Betainc
from mindspore.ops.operations.math_ops import Cholesky
from mindspore.ops.operations.math_ops import Fmin
from mindspore.ops.operations.math_ops import CholeskySolve
from mindspore.ops.operations.math_ops import InplaceIndexAdd
from mindspore.ops.operations.math_ops import AddV2
from mindspore.ops.operations.math_ops import TridiagonalMatMul
from mindspore.ops.operations.math_ops import TridiagonalSolve
from mindspore.ops.operations.math_ops import Logit
from mindspore.ops.operations.math_ops import Diagonal
from mindspore.ops.operations.math_ops import EuclideanNorm
from mindspore.ops.operations.array_ops import Transpose, MatrixSetDiagV3
from mindspore.ops.operations.math_ops import Fmax
from mindspore.ops.operations._inner_ops import DynamicBroadcastGradientArgs
from mindspore.ops.composite.multitype_ops.zeros_like_impl import zeros_like
from mindspore.ops.primitive import _primexpr
from mindspore.ops._grad.grad_base import bprop_getters, create_tensor_by_element, dyn_rank
from mindspore.ops._grad.grad_base import dyn_ones, dyn_fill, sum_grad_reduce_axis
from mindspore.ops._grad.grad_math_ops import binop_grad_common
from mindspore.ops.operations.array_ops import MatrixBandPart
from mindspore.ops.operations.array_ops import ConjugateTranspose

transpose = P.Transpose()
dyn_shape_op = P.TensorShape()
_conj = P.Conj()


@_primexpr
def _generate_perm(x_dim):
    perm = tuple(range(x_dim - 2))
    return perm


def _dyn_generate_perm(x_dim):
    perm = P.Range()(P.Cast()(0, x_dim.dtype), x_dim - 2, P.Cast()(1, x_dim.dtype))
    return perm


def _adjoint(a):
    return cholesky_transpose(_conj(a))


def cholesky_transpose(a):
    n = len(a.shape)
    n_range = list(range(0, n))
    n_range[-1] = n - 2
    n_range[-2] = n - 1
    return transpose(a, tuple(n_range))


@bprop_getters.register(P.ACos)
def get_bprop_acos(self):
    """Grad definition for `ACos` operation."""
    input_grad = G.ACosGrad()

    def bprop(input_x, out, dout):
        dx = input_grad(input_x, dout)
        return (dx,)

    return bprop


@bprop_getters.register(Logit)
def get_bprop_logit(self):
    """Grad definition for `Logit` operation."""
    logitgrad = G.LogitGrad(self.eps)

    def bprop(x, out, dout):
        dx = logitgrad(dout, x)
        return (dx,)

    return bprop


@bprop_getters.register(P.Roll)
def get_bprop_roll(self):
    """Generate bprop for Roll"""
    if context.get_context("device_target") == "GPU":
        shift = []
        axis = self.axis
        for tmp in enumerate(self.shift):
            shift.append(-tmp[1])
        roll_grad = P.Roll(shift, axis)
    else:
        shift = self.shift
        axis = self.axis
        roll_grad = P.Roll(-shift, axis)

    def bprop(x_input, out, dout):
        dx = roll_grad(dout)
        return (dx,)

    return bprop


@bprop_getters.register(P.Cdist)
def get_bprop_cdist(self):
    """Generate bprop for Cdist"""
    input_grad = G.CdistGrad(p=self.p)

    def bprop(input_x, input_y, out, dout):
        dout_shape = F.shape(dout)
        if F.is_sequence_value_unknown(dout_shape):
            dout_dim = dyn_rank(dout)
            dout_perm_part2 = create_tensor_by_element(
                (dout_dim - 1, dout_dim - 2))
            if dout_dim <= 2:
                dout_perm = dout_perm_part2
            else:
                dout_perm_part1 = _dyn_generate_perm(dout_dim)
                dout_perm = P.Concat(0)((dout_perm_part1, dout_perm_part2))
        else:
            dout_dim = len(dout_shape)
            dout_perm_part1 = _generate_perm(dout_dim)
            dout_perm_part2 = (dout_dim - 1, dout_dim - 2)
            dout_perm = dout_perm_part1 + dout_perm_part2
        out_perm = dout_perm
        dout_transpose = transpose(dout, dout_perm)
        out_transpose = transpose(out, out_perm)
        dx = input_grad(dout, input_x, input_y, out)
        dy = input_grad(dout_transpose, input_y, input_x, out_transpose)
        return dx, dy

    return bprop


@bprop_getters.register(P.Lerp)
def get_bprop_index_lerp(self):
    """Generate bprop for Lerp"""
    mul_op = P.Mul()
    sub_op = P.Sub()
    is_instance_op = inner.IsInstance()

    def bprop(start, end, weight, out, dout):
        dout = F.cast(dout, mstype.float32)
        dstart = mul_op(dout, 1 - weight)
        dend = mul_op(dout, weight)
        dweight = mul_op(dout, sub_op(end, start))
        dstart, dend = binop_grad_common(start, end, dstart, dend)
        if is_instance_op(weight, mstype.number):
            dweight = 0
        else:
            _, dweight = binop_grad_common(start, weight, dstart, dweight)
            dweight = F.cast(dweight, F.dtype(weight))
        dstart = F.cast(dstart, F.dtype(start))
        dend = F.cast(dend, F.dtype(end))
        return dstart, dend, dweight

    return bprop


@bprop_getters.register(LuUnpack)
def get_bprop_lu_unpack(self):
    """Grad definition for `LuUnpack` operation."""
    input_grad = G.LuUnpackGrad(L_grad_flag=True, U_grad_flag=True)

    def bprop(lu_data, lu_pivots, out, dout):
        dl, du = input_grad(dout[1], dout[2], lu_data)
        lu_data_grad = dl + du
        return (lu_data_grad, zeros_like(lu_pivots))

    return bprop


@bprop_getters.register(Sinc)
def get_bprop_sinc(self):
    """Grad definition for `Sinc` operation."""
    sin = P.Sin()
    cos = P.Cos()
    cast = P.Cast()
    conj = P.Conj()

    def bprop(x, out, dout):
        kpi = cast(np.pi, x.dtype)
        product = kpi * x
        reciprocal = (product * cos(product) - sin(product)) / (product * x)
        if reciprocal.dtype in [mstype.complex64, mstype.complex128]:
            reciprocal = conj(reciprocal)
        dx = reciprocal * dout
        return (dx,)

    return bprop


@bprop_getters.register(ReduceStd)
def get_bprop_reduce_std(self):
    """Grad definition for `ReduceStd` operation."""
    axis = list(self.axis)
    keep_dims = self.keep_dims
    unbiased = self.unbiased
    expand_dims_op = P.ExpandDims()
    size_op = P.Size()
    mul_op = P.Mul()
    sub_op = P.Sub()
    div_op = P.Div()
    add_op = P.Add()

    def bprop(x, out, dout):
        std_d = dout[0]
        std = out[0]
        mean_d = dout[1]
        mean = out[1]
        if axis == [] and x.shape != ():
            for i, _ in enumerate(x.shape):
                axis.append(i)
        for i, _ in enumerate(axis):
            if axis[i] < 0:
                axis[i] = axis[i] + len(x.shape)
        for i in range(1, len(axis)):
            for j in range(0, len(axis) - i):
                if axis[j] > axis[j + 1]:
                    axis[j], axis[j + 1] = axis[j + 1], axis[j]
        if not keep_dims and x.shape != ():
            for i in axis:
                std_d = expand_dims_op(std_d, i)
                std = expand_dims_op(std, i)
                mean_d = expand_dims_op(mean_d, i)
                mean = expand_dims_op(mean, i)
        dx = sub_op(x, mean)
        dx = mul_op(dx, std_d)
        dx = div_op(dx, std)
        num = size_op(x)
        for i, _ in enumerate(x.shape):
            if i not in axis:
                num = num / x.shape[i]
        if unbiased:
            dx = div_op(dx, num - 1)
        else:
            dx = div_op(dx, num)
        temp = div_op(mean_d, num)
        dx = add_op(dx, temp)
        return (dx,)

    return bprop


@bprop_getters.register(P.Addcdiv)
def get_bprop_index_addcdiv(self):
    """Generate bprop for Addcdiv"""
    mul_op = P.Mul()
    div_op = P.Div()
    pow_op = P.Pow()
    neg_op = P.Neg()

    def bprop(input_data, x1, x2, value, out, dout):
        dinput_data = dout
        if dout.dtype in [mstype.float16, mstype.int64, mstype.float64]:
            input_data = F.cast(input_data, mstype.float32)
            x1 = F.cast(x1, mstype.float32)
            x2 = F.cast(x2, mstype.float32)
            value = F.cast(value, mstype.float32)
            dinput_data = F.cast(dinput_data, mstype.float32)
        inner_out = mul_op(value, div_op(x1, x2)) + input_data
        dx2 = neg_op(mul_op(mul_op(mul_op(x1, value), pow_op(x2, -2)), dinput_data))
        dx1 = mul_op(dinput_data, div_op(value, x2))
        dvalue = mul_op(dinput_data, div_op(x1, x2))
        _, dinput_data = binop_grad_common(inner_out, input_data, dout, dinput_data)
        _, dx1 = binop_grad_common(inner_out, x1, dout, dx1)
        _, dx2 = binop_grad_common(inner_out, x2, dout, dx2)
        _, dvalue = binop_grad_common(inner_out, value, dout, dvalue)
        if dout.dtype in [mstype.float16, mstype.int64, mstype.float64]:
            dinput_data = F.cast(dinput_data, dout.dtype)
            dx1 = F.cast(dx1, dout.dtype)
            dx2 = F.cast(dx2, dout.dtype)
            dvalue = F.cast(dvalue, dout.dtype)
        return dinput_data, dx1, dx2, dvalue

    return bprop


@bprop_getters.register(P.Addcmul)
def get_bprop_index_addcmul(self):
    """Generate bprop for Addcmul"""
    mul_op = P.Mul()

    def bprop(input_data, x1, x2, value, out, dout):
        if dout.dtype in [mstype.float16, mstype.float64, mstype.uint8, mstype.int8, mstype.int32, mstype.int64]:
            input_data = F.cast(input_data, mstype.float32)
            x1 = F.cast(x1, mstype.float32)
            x2 = F.cast(x2, mstype.float32)
            value = F.cast(value, mstype.float32)
        dinput_data = dout
        dx1 = mul_op(dout, mul_op(value, x2))
        dx2 = mul_op(dout, mul_op(value, x1))
        inner_out = mul_op(x1, x2) * value + input_data
        dvalue = mul_op(dout, mul_op(x1, x2))
        _, dinput_data = binop_grad_common(inner_out, input_data, dout, dinput_data)
        _, dx1 = binop_grad_common(inner_out, x1, dout, dx1)
        _, dx2 = binop_grad_common(inner_out, x2, dout, dx2)
        _, dvalue = binop_grad_common(inner_out, value, dout, dvalue)
        if dout.dtype in [mstype.float16, mstype.uint8, mstype.int8, mstype.float64, mstype.int32, mstype.int64]:
            dinput_data = F.cast(dinput_data, dout.dtype)
            dx1 = F.cast(dx1, dout.dtype)
            dx2 = F.cast(dx2, dout.dtype)
            dvalue = F.cast(dvalue, dout.dtype)
        return dinput_data, dx1, dx2, dvalue

    return bprop


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


@bprop_getters.register(Renorm)
def get_bprop_renorm(self):
    """Generate bprop for Renorm """
    p = int(self.p)
    ext = 1e-7
    dim = self.dim
    max_norm = self.maxnorm
    greater_op = P.Greater()
    pow_op = P.Pow()
    abs_op = P.Abs()
    sign_op = P.Sign()
    reciprocal_op = P.Reciprocal()

    def bprop(input_x, out, dout):
        shape = F.shape(input_x)
        dims = renew_dim(shape, dim)
        norm = P.LpNorm(dims, p, keep_dims=True)(input_x)
        grad_out = (input_x * dout)
        grad_out = grad_out.sum(dims, keepdims=True)
        if p == 1:
            sig = sign_op(input_x)
            norm_bp = sig * grad_out
        elif p == 2:
            m = input_x * (grad_out / norm)
            norm_bp = F.masked_fill(m, norm == 0., 0.)
        else:
            abs_ = abs_op(input_x)
            input_scaled = input_x * pow_op(abs_, (p - 2))
            pow_ = pow_op(norm, (p - 1))
            scale_v = grad_out / pow_
            scale_v = F.masked_fill(scale_v, norm == 0., 0.)
            norm_bp = input_scaled * scale_v

        v = norm + ext
        inv_norm = reciprocal_op(v)
        grad_norm = max_norm * inv_norm * (dout - inv_norm * norm_bp)
        q = greater_op(norm, max_norm)
        return (mnp.where(q, grad_norm, dout),)

    return bprop


@bprop_getters.register(P.LpNorm)
def get_bprop_lp_norm(self):
    """Grad definition for `LpNorm` operation."""
    p = self.p
    keep_dims = self.keep_dims
    axis = self.axis
    if isinstance(axis, int):
        axis = [axis]
    sign_op = P.Sign()
    abs_op = P.Abs()
    zeros_like_op = P.ZerosLike()
    expand_dims_op = P.ExpandDims()
    pow_op = P.Pow()

    def bprop(input_x, out, dout):
        if not keep_dims and input_x.shape != ():
            for i in axis:
                dout = expand_dims_op(dout, i)
                out = expand_dims_op(out, i)

        if p == 0:
            return (zeros_like_op(input_x),)
        if p == 1:
            return (dout * sign_op(input_x),)
        if p == 2:
            input_scaled = input_x
            scale_v = dout / out
        else:
            input_scaled = pow_op(abs_op(input_x), (p - 2)) * input_x
            scale_v = dout / pow_op(out, (p - 1))
        return (input_scaled * scale_v,)

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

    def where_v2(condition, x=None, y=None):
        return_all = None
        if x is None and y is None:
            return_all = mnp.where(condition, x, y)
        elif x is not None and y is not None:
            shape_ = x.shape
            input_y = np.resize(y, shape_)
            input_y = Tensor(input_y).astype(x.dtype)
            return_all = ops.select(condition, x, input_y)
        else:
            raise ValueError("x and y must both be non-None or both be None.")
        return return_all

    def bprop(x, axis, out, dout):
        dtype_min = 0
        if x.dtype == mstype.float16:
            dtype_min = -65500e+0
        elif x.dtype == mstype.float32:
            dtype_min = -3.4028235e+38
        elif x.dtype == mstype.float64:
            dtype_min = -1.7976931348623157e+308
        log_grad_positive = where_v2(greater_op(dout, 0), log_op(dout), dtype_min)
        log_grad_negative = where_v2(less_op(dout, 0), log_op(neg_op(dout)), dtype_min)
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


@bprop_getters.register(MatrixExp)
def get_bprop_matrix_exp(self):
    """Gegerate brop for MatrixExp"""
    matrix_exp = MatrixExp()
    zeros = P.Zeros()
    concat_row = P.Concat(-1)
    concat_col = P.Concat(-2)
    cast = P.Cast()
    slice_op = P.Slice()
    range_op = P.Range()
    expand_dims = P.ExpandDims()
    dyn_shape = P.TensorShape()

    def bprop(x, out, dout):
        if F.is_sequence_value_unknown(x.shape):
            shape_x = dyn_shape(x)
            x_len = dyn_rank(x)
            input_perm = range_op(cast(0, mstype.int64), x_len, cast(1, mstype.int64))
            input_perm[-1] = input_perm[-2]
            input_perm[-2] = x_len - 1
            x_transpose = transpose(x, input_perm)
            zero_matrix = dyn_fill(mstype.float32, shape_x, 0)
        else:
            shape_x = x.shape
            x_len = len(shape_x)
            input_perm = [ele for ele in range(x_len)]
            input_perm[-1] = input_perm[-2]
            input_perm[-2] = x_len - 1
            input_perm = tuple(input_perm)
            x_transpose = P.Transpose()(x, input_perm)
            zero_matrix = zeros(shape_x, mstype.float32)

        zero_matrix = cast(zero_matrix, dout.dtype)
        meta_grad_up = concat_row((x_transpose, dout))
        meta_grad_down = concat_row((zero_matrix, x_transpose))
        meta_grad = concat_col((meta_grad_up, meta_grad_down))
        meta_grad = matrix_exp(meta_grad)

        if F.is_sequence_value_unknown(x.shape):
            begins = dyn_fill(mstype.int32, expand_dims(x_len, 0), 0)
            sizes = cast(shape_x, mstype.int32)
        else:
            begins = [0] * x_len
            sizes = [i for i in shape_x]
        n = shape_x[-1]
        begins[-1] = n
        sizes[-2] = n
        sizes[-1] = n
        return (slice_op(meta_grad, begins, sizes),)

    return bprop


@bprop_getters.register(MatrixPower)
def get_bprop_matrix_power(self):
    """Generate bprop for MatrixPower"""
    n = self.n
    batch_matmul_a = P.BatchMatMul(transpose_a=True)
    batch_matmul_b = P.BatchMatMul(transpose_b=True)
    neg = P.Neg()

    def bprop(x, out, dout):
        dout = F.cast(dout, mstype.float32)
        x = F.cast(x, mstype.float32)
        power = n
        dx = zeros_like(x)
        if power < 0:
            matrix_power = MatrixPower(n=-1)
            x_inv = matrix_power(x)
            for i in range(0, -power):
                matrix_power = MatrixPower(n=(-power - 1 - i))
                dx = dx + batch_matmul_b(dout, matrix_power(x_inv))
                dout = batch_matmul_a(x_inv, dout)
            dx = batch_matmul_b(dx, x_inv)
            dx = batch_matmul_a(x_inv, dx)
            dx = neg(dx)
        else:
            for i in range(0, power):
                matrix_power = MatrixPower(n=(power - 1 - i))
                dx = dx + batch_matmul_b(dout, matrix_power(x))
                dout = batch_matmul_a(x, dout)
        dx = F.cast(dx, F.dtype(out))
        return (dx,)

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

        a_shape = F.shape(input_a)
        if F.is_sequence_value_unknown(a_shape):
            matrix_rank = dyn_rank(input_a)
        else:
            matrix_rank = rank(input_a)

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


@bprop_getters.register(P.MatrixDeterminant)
def get_bprop_matrix_determinant(self):
    """Generate bprop for MatrixDeterminant"""
    inverse_op = P.MatrixInverse(adjoint=True)
    shape_op = P.Shape()
    reshape = P.Reshape()
    concat = P.Concat(0)

    def bprop(x, out, dout):
        if F.is_sequence_value_unknown(shape_op(x)):
            x_adj_inv = inverse_op(x)
            out_shape = dyn_shape_op(out)
            ones = create_tensor_by_element((1, 1))
            multipliers = reshape(dout * out, concat((out_shape, ones)))
            dx = multipliers * x_adj_inv
            return (dx,)
        x_adj_inv = inverse_op(x)
        multipliers = reshape(dout * out, shape_op(out) + (1, 1))
        dx = multipliers * x_adj_inv
        return (dx,)

    return bprop


@bprop_getters.register(P.LogMatrixDeterminant)
def get_bprop_log_matrix_determinant(self):
    """Generate bprop for LogMatrixDeterminant"""
    inverse_op = P.MatrixInverse(adjoint=True)
    shape_op = P.Shape()
    reshape = P.Reshape()

    def bprop(x, out, dout):
        x_adj_inv = inverse_op(x)
        if F.is_sequence_value_unknown(shape_op(out[1])):
            const_value = F.cast(1, mstype.int64)
            const_value = P.ExpandDims()(const_value, 0)
            new_shape = P.Concat()((dyn_shape_op(out[1]), const_value, const_value))
            multipliers = reshape(dout[1], new_shape)
        else:
            multipliers = reshape(dout[1], shape_op(out[1]) + (1, 1))
        dx = multipliers * x_adj_inv
        return (dx,)

    return bprop


@bprop_getters.register(Betainc)
def get_bprop_betainc(self):
    """Grad definition for 'Betainc' operation"""
    lgamma = LGamma()
    exp = P.Exp()
    log1p = P.Log1p()
    xlogy = P.Xlogy()
    dyn_shape = P.TensorShape()

    def bprop(input_a, input_b, input_x, out, dout):
        if F.is_sequence_value_unknown(F.shape(input_x)):
            sx = dyn_shape(input_x)
        else:
            sx = F.shape(input_x)
        log_beta = (lgamma(input_a) + lgamma(input_b) - lgamma(input_a + input_b))
        partial_x = exp((input_b - 1) * log1p(-input_x) + xlogy(input_a - 1, input_x) - log_beta)
        return (zeros_like(input_a), zeros_like(input_b), F.reshape(partial_x * dout, sx))

    return bprop


@bprop_getters.register(P.CholeskyInverse)
def get_bprop_cholesky_inverse(self):
    """Grad definition for `CholeskyInverse` operation."""
    matmul = P.MatMul()
    upper = self.upper
    neg = P.Neg()

    def bprop(input_x, out, dout):
        input_perm = (1, 0)
        if dout.dtype == mstype.float64:
            input_x = F.cast(input_x, mstype.float32)
            out = F.cast(out, mstype.float32)
            dout = F.cast(dout, mstype.float32)
            common_term = dout + transpose(dout, input_perm)
            common_term = F.cast(common_term, mstype.float32)
            common_term = matmul(out, matmul(common_term, out))
            if upper is True:
                dx = neg(matmul(input_x, common_term))
                dx = F.cast(dx, mstype.float64)
            else:
                dx = neg(matmul(common_term, input_x))
                dx = F.cast(dx, mstype.float64)
            return (dx,)
        common_term = dout + transpose(dout, input_perm)
        common_term = matmul(out, matmul(common_term, out))
        if upper is True:
            dx = neg(matmul(input_x, common_term))
        else:
            dx = neg(matmul(common_term, input_x))
        return (dx,)

    return bprop


@bprop_getters.register(Real)
def get_bprop_real(self):
    """Grad definition for `Real` operation."""
    complex_grad = Complex()

    def bprop(input_1, out, dout):
        zero = zeros_like(dout)
        dx = dout
        res = complex_grad(dx, zero)
        return (res,)

    return bprop


@bprop_getters.register(Imag)
def get_bprop_imag(self):
    """Grad definition for `Real` operation."""
    complex_grad = Complex()

    def bprop(input_1, out, dout):
        zero = zeros_like(dout)
        dx = dout
        res = complex_grad(zero, dx)
        return (res,)

    return bprop


@bprop_getters.register(Complex)
def get_bprop_complex(self):
    """Grad definition for `Real` operation."""
    real_grad = Real()
    imag_grad = Imag()

    def bprop(real, imag, out, dout):
        dx = real_grad(dout)
        dy = imag_grad(dout)
        return (dx, dy,)

    return bprop


@bprop_getters.register(ComplexAbs)
def get_bprop_complex_abs(self):
    """Grad definition for `Real` operation."""
    div_no_nan = P.DivNoNan()
    complex_grad = Complex()
    mul = P.Mul()

    def bprop(x, out, dout):
        return (div_no_nan(mul(complex_grad(dout, zeros_like(dout)), x), complex_grad(out, zeros_like(out))),)

    return bprop


@bprop_getters.register(NanToNum)
def get_bprop_nan_to_num(self):
    """Grad definition for `NanToNum` operation."""
    isfinite = P.IsFinite()

    def bprop(x, out, dout):
        dx = dout * isfinite(x)
        return (dx,)

    return bprop


@bprop_getters.register(Angle)
def get_bprop_angle(self):
    """Grad definition for `Angle` operation."""
    real_op = Real()
    imag_op = Imag()
    reciprocal_op = P.Reciprocal()
    complex_op = Complex()
    neg_op = P.Neg()

    def bprop(x, out, dout):
        re = real_op(x)
        im = imag_op(x)
        re = complex_op(im, re)
        z = reciprocal_op(re)
        zero = zeros_like(dout)
        complex_dout = complex_op(dout, zero)
        return (neg_op(complex_dout * z),)

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


@bprop_getters.register(P.Erfinv)
def get_bprop_erfinv(self):
    """Grad definition for `Erfinv` operation."""
    exp = P.Exp()
    square = P.Square()
    sqrt = P.Sqrt()
    cast = P.Cast()
    dtype = P.DType()

    def bprop(input_x, out, dout):
        root_pi_over_two = cast(sqrt(F.scalar_to_tensor(np.pi)) / 2, dtype(dout))
        out_square = square(out)
        dx = dout * root_pi_over_two * exp(out_square)
        return (dx,)

    return bprop


@bprop_getters.register(BesselI0)
def get_bprop_bessel_i0(self):
    """Generate bprop for BesselI0"""
    bessel_i1 = BesselI1()

    def bprop(x, out, dout):
        dx = dout * bessel_i1(x)
        return (dx,)

    return bprop


@bprop_getters.register(BesselI1)
def get_bprop_bessel_i1(self):
    """Generate bprop for BesselI1"""
    equal = P.Equal()
    div = P.Div()
    cast = P.Cast()
    dtype = P.DType()
    bessel_i0 = BesselI0()

    def bprop(x, out, dout):
        dout_dx = mnp.where(equal(x, 0.), cast(1., dtype(x)), bessel_i0(x) - div(out, x))
        dx = dout * dout_dx
        return (dx,)

    return bprop


@bprop_getters.register(BesselJ0)
def get_bprop_bessel_j0(self):
    """Generate bprop for BesselJ0"""
    bessel_j1 = BesselJ1()

    def bprop(x, out, dout):
        dx = -dout * bessel_j1(x)
        return (dx,)

    return bprop


@bprop_getters.register(BesselJ1)
def get_bprop_bessel_j1(self):
    """Generate bprop for BesselJ1"""
    equal = P.Equal()
    div = P.Div()
    cast = P.Cast()
    dtype = P.DType()
    bessel_j0 = BesselJ0()

    def bprop(x, out, dout):
        dout_dx = mnp.where(equal(x, 0.), cast(0.5, dtype(x)), bessel_j0(x) - div(out, x))
        dx = dout * dout_dx
        return (dx,)

    return bprop


@bprop_getters.register(BesselK0)
def get_bprop_bessel_k0(self):
    """Generate bprop for BesselK0"""
    bessel_k1 = BesselK1()

    def bprop(x, out, dout):
        dx = -dout * bessel_k1(x)
        return (dx,)

    return bprop


@bprop_getters.register(BesselK1)
def get_bprop_bessel_k1(self):
    """Generate bprop for BesselK1"""
    div = P.Div()
    bessel_k0 = BesselK0()

    def bprop(x, out, dout):
        dout_dx = -(bessel_k0(x) + div(out, x))
        dx = dout * dout_dx
        return (dx,)

    return bprop


@bprop_getters.register(BesselK0e)
def get_bprop_bessel_k0e(self):
    """Generate bprop for BesselK0e"""
    bessel_k1e = BesselK1e()

    def bprop(x, out, dout):
        dx = dout * (out - bessel_k1e(x))
        return (dx,)

    return bprop


@bprop_getters.register(BesselK1e)
def get_bprop_bessel_k1e(self):
    """Generate bprop for BesselK1e"""
    reciprocal = P.Reciprocal()
    bessel_k0e = BesselK0e()

    def bprop(x, out, dout):
        dout_dx = out * (1. - reciprocal(x)) - bessel_k0e(x)
        dx = dout * dout_dx
        return (dx,)

    return bprop


@bprop_getters.register(BesselY0)
def get_bprop_bessel_y0(self):
    """Generate bprop for BesselY0"""
    bessel_y1 = BesselY1()

    def bprop(x, out, dout):
        dx = -dout * bessel_y1(x)
        return (dx,)

    return bprop


@bprop_getters.register(BesselY1)
def get_bprop_bessel_y1(self):
    """Generate bprop for BesselY1"""
    div = P.Div()
    bessel_y0 = BesselY0()

    def bprop(x, out, dout):
        dout_dx = bessel_y0(x) - div(out, x)
        dx = dout * dout_dx
        return (dx,)

    return bprop


@bprop_getters.register(Hypot)
def get_bprop_hypot(self):
    """Generate bprop for Hypot"""
    mul_ = P.Mul()
    div_ = P.Div()

    def bprop(x1, x2, out, dout):
        x1_f32 = F.cast(x1, mstype.float32)
        x2_f32 = F.cast(x2, mstype.float32)
        out_f32 = F.cast(out, mstype.float32)
        dout_f32 = F.cast(dout, mstype.float32)
        dx1 = mul_(div_(x1_f32, out_f32), dout_f32)
        dx2 = mul_(div_(x2_f32, out_f32), dout_f32)
        result_dx1, result_dx2 = binop_grad_common(x1_f32, x2_f32, dx1, dx2)
        result_dx1 = F.cast(result_dx1, F.dtype(x1))
        result_dx2 = F.cast(result_dx2, F.dtype(x2))
        return (result_dx1, result_dx2)

    return bprop


@bprop_getters.register(P.Asin)
def get_bprop_asin(self):
    """Grad definition for `Asin` operation."""
    input_grad = G.AsinGrad()

    def bprop(input_x, out, dout):
        dx = input_grad(input_x, dout)
        return (dx,)

    return bprop


@bprop_getters.register(P.Trunc)
def get_bprop_trunc(self):
    """Grad definition for `Trunc` operation."""

    def bprop(input_x, output_y, dout):
        bc_x = zeros_like(input_x)
        return (bc_x,)

    return bprop


@bprop_getters.register(P.Ger)
def get_bprop_ger(self):
    """Grad definition for 'Ger' operation"""
    transpose_op = P.Transpose()
    matmul = P.MatMul()
    expand_dims = P.ExpandDims()
    squeeze = P.Squeeze(1)

    def bprop(input_x, input_y, out, dout):
        dx = squeeze(matmul(dout, expand_dims(input_y, 1)))
        dy = squeeze(matmul(transpose_op(dout, (1, 0)), expand_dims(input_x, 1)))
        return dx, dy

    return bprop


@bprop_getters.register(P.Cross)
def get_bprop_cross(self):
    """Grad definition for 'Cross' operation"""
    cross = P.Cross(dim=self.dim)

    def bprop(input1, input2, out, dout):
        return cross(input2, dout), cross(dout, input1)

    return bprop


@bprop_getters.register(Median)
def get_bprop_median(self):
    """Grad definition for 'Median' operation"""
    input_grad = G.MedianGrad(global_median=self.global_median, axis=self.axis, keep_dims=self.keep_dims)

    def bprop(x, out, dout):
        dx = F.cast(input_grad(dout[0], x, out[0], out[1]), F.dtype(x))
        return (dx,)

    return bprop


@bprop_getters.register(P.MulNoNan)
def get_bprop_mul_no_nan(self):
    """Grad definition for `MulNoNan` operation."""
    mul_func = P.Mul()

    def bprop(x, y, out, dout):
        bc_x = mul_func(dout, y)
        bc_y = mul_func(x, dout)
        return binop_grad_common(x, y, bc_x, bc_y)

    return bprop


@bprop_getters.register(Trace)
def get_bprop_trace(self):
    """Grad definition for `Trace` operation."""
    input_grad = G.TraceGrad()
    shape_op = P.Shape()
    to_array = P.TupleToArray()
    cast = P.Cast()

    def bprop(x, out, dout):
        shape = shape_op(x)
        if F.is_sequence_value_unknown(shape):
            shape = dyn_shape_op(x)
            dx = input_grad(dout, shape)
        else:
            dx = input_grad(dout, cast(to_array(shape), mstype.int64))
        return (dx,)

    return bprop


@bprop_getters.register(Fmin)
def get_bprop_fmin(self):
    """Grad definition for 'Fmin' operation"""
    shape_ = P.Shape()
    masked_fill_op = P.MaskedFill()
    logical_or_op = P.LogicalOr()
    logical_not_op = P.LogicalNot()
    logical_and_op = P.LogicalAnd()
    mul_op = P.Mul()
    is_nan_op = P.IsNan()
    reshape_ = P.Reshape()

    def bprop(x1, x2, out, dout):
        x1_dtype = F.dtype(x1)
        x2_dtype = F.dtype(x2)
        x1 = F.cast(x1, mstype.float32)
        x2 = F.cast(x2, mstype.float32)
        dout = F.cast(dout, mstype.float32)
        b1 = logical_or_op((x1 <= x2), is_nan_op(x2))
        b2 = logical_or_op((x2 < x1), logical_and_op(is_nan_op(x1), logical_not_op(is_nan_op(x2))))
        rx1 = masked_fill_op(x1, b1, 1.)
        rx1 = masked_fill_op(rx1, logical_not_op(b1), 0.)
        rx2 = masked_fill_op(x2, b2, 1.)
        rx2 = masked_fill_op(rx2, logical_not_op(b2), 0.)
        rrx1 = mul_op(rx1, dout)
        rrx2 = mul_op(rx2, dout)
        shape_of_x1 = shape_(x1)
        shape_of_x2 = shape_(x2)
        x1_dim = len(shape_of_x1)
        x2_dim = len(shape_of_x2)
        if x1_dim == 0 and x2_dim != 0:
            sum_r1 = rrx1.sum()
            sum_r2 = rrx2
        elif x1_dim == 0 and x2_dim == 0:
            sum_r1 = rrx1.sum()
            sum_r2 = rrx2.sum()
        elif x1_dim != 0 and x2_dim == 0:
            sum_r2 = rrx2.sum()
            sum_r1 = rrx1
        else:
            rx, ry = DynamicBroadcastGradientArgs()(shape_of_x1, shape_of_x2)
            sum_r1 = sum_grad_reduce_axis(rrx1, rx)
            sum_r2 = sum_grad_reduce_axis(rrx2, ry)
        brrx1 = reshape_(sum_r1, shape_of_x1)
        brrx2 = reshape_(sum_r2, shape_of_x2)
        brrx1 = F.cast(brrx1, x1_dtype)
        brrx2 = F.cast(brrx2, x2_dtype)
        return brrx1, brrx2

    return bprop


@bprop_getters.register(Fmax)
def get_bprop_fmax(self):
    """Grad definition for 'Fmax' operation"""
    shape_ = P.Shape()
    masked_fill_op = P.MaskedFill()
    logical_or_op = P.LogicalOr()
    logical_not_op = P.LogicalNot()
    logical_and_op = P.LogicalAnd()
    mul_op = P.Mul()
    is_nan_op = P.IsNan()
    reshape_ = P.Reshape()

    def bprop(x1, x2, out, dout):
        x1_dtype = F.dtype(x1)
        x2_dtype = F.dtype(x2)
        if x1_dtype != mstype.float32:
            x1 = F.cast(x1, mstype.float32)
            dout = F.cast(dout, mstype.float32)
        if x2_dtype != mstype.float32:
            x2 = F.cast(x2, mstype.float32)
            dout = F.cast(dout, mstype.float32)
        b1 = logical_or_op(logical_and_op((x1 >= x2), logical_not_op(is_nan_op(x1))), is_nan_op(x2))
        b2 = logical_or_op(logical_and_op(x2 > x1, logical_not_op(is_nan_op(x2))),
                           logical_and_op(is_nan_op(x1), logical_not_op(is_nan_op(x2))))
        rx1 = masked_fill_op(x1, b1, 1.)
        rx1 = masked_fill_op(rx1, logical_not_op(b1), 0.)
        rx2 = masked_fill_op(x2, b2, 1.)
        rx2 = masked_fill_op(rx2, logical_not_op(b2), 0.)
        rrx1 = mul_op(rx1, dout)
        rrx2 = mul_op(rx2, dout)
        shape_of_x1 = shape_(x1)
        shape_of_x2 = shape_(x2)
        x1_dim = len(shape_of_x1)
        x2_dim = len(shape_of_x2)
        if x1_dim == 0 and x2_dim != 0:
            sum_r1 = rrx1.sum()
            sum_r2 = rrx2
        elif x1_dim == 0 and x2_dim == 0:
            sum_r1 = rrx1.sum()
            sum_r2 = rrx2.sum()
        elif x1_dim != 0 and x2_dim == 0:
            sum_r2 = rrx2.sum()
            sum_r1 = rrx1
        else:
            rx, ry = DynamicBroadcastGradientArgs()(shape_of_x1, shape_of_x2)
            sum_r1 = sum_grad_reduce_axis(rrx1, rx)
            sum_r2 = sum_grad_reduce_axis(rrx2, ry)
        brrx1 = reshape_(sum_r1, shape_of_x1)
        brrx2 = reshape_(sum_r2, shape_of_x2)
        brrx1 = F.cast(brrx1, x1_dtype)
        brrx2 = F.cast(brrx2, x2_dtype)
        return brrx1, brrx2


    return bprop


@bprop_getters.register(G.MinimumGrad)
def get_bprop_minimum_grad(self):
    """Grad definition for 'MinimumGrad' operation"""
    input_grad = G.MinimumGradGrad()

    def bprop(x1, x2, grad, out, dout):
        sopd_x1, sopd_x2, sopd_grads = input_grad(x1, x2, dout[0], dout[1])
        sopd_x1 = zeros_like(x1)
        sopd_x2 = zeros_like(x2)
        return sopd_x1, sopd_x2, sopd_grads

    return bprop


@bprop_getters.register(Bernoulli)
def get_bprop_bernoulli(self):
    """"Grad definition for 'Bernoulli' operation."""

    def bprop(x, p, out, dout):
        return zeros_like(x), zeros_like(p)

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


@bprop_getters.register(Igamma)
def get_bprop_igamma(self):
    """Grad definition for `Igamma` operation."""
    shape_ = P.Shape()
    igammagrada = G.IgammaGradA()
    lgamma = nn.LGamma()
    log_ = P.Log()
    exp_ = P.Exp()
    reshape_ = P.Reshape()
    reduce_sum_ = P.ReduceSum()

    def bprop(a, x, out, dout):
        sa = shape_(a)
        sx = shape_(x)
        if F.is_sequence_value_unknown(sa) or F.is_sequence_value_unknown(sx):
            sa = dyn_shape_op(a)
            sx = dyn_shape_op(x)
            ra, rx = DynamicBroadcastGradientArgs()(sa, sx)
            partial_a = igammagrada(a, x)
            partial_x = exp_(-x + (a - 1) * log_(x) - lgamma(a))
            r1 = reshape_(sum_grad_reduce_axis(partial_a * dout, ra), sa)
            r2 = reshape_(sum_grad_reduce_axis(partial_x * dout, rx), sx)
            return r1, r2
        ra, rx = broadcast_gradient_args(sa, sx)
        partial_a = igammagrada(a, x)
        partial_x = exp_(-x + (a - 1) * log_(x) - lgamma(a))
        if ra != ():
            r1 = reshape_(reduce_sum_(partial_a * dout, ra), sa)
        else:
            r1 = reshape_(partial_a * dout, sa)
        if rx != ():
            r2 = reshape_(reduce_sum_(partial_x * dout, rx), sx)
        else:
            r2 = reshape_(partial_x * dout, sx)
        return r1, r2

    return bprop


@bprop_getters.register(Igammac)
def get_bprop_igammac(self):
    """Grad definition for `Igammac` operation."""
    shape_ = P.Shape()
    igammagrada = G.IgammaGradA()
    lgamma = nn.LGamma()
    log_ = P.Log()
    exp_ = P.Exp()
    reshape_ = P.Reshape()
    reduce_sum_ = P.ReduceSum()
    neg_ = P.Neg()

    def bprop(a, x, out, dout):
        sa = shape_(a)
        sx = shape_(x)
        if F.is_sequence_value_unknown(sa) or F.is_sequence_value_unknown(sx):
            sa = dyn_shape_op(a)
            sx = dyn_shape_op(x)
            ra, rx = DynamicBroadcastGradientArgs()(sa, sx)
            partial_a = igammagrada(a, x)
            partial_x = exp_(-x + (a - 1) * log_(x) - lgamma(a))
            r1 = neg_(reshape_(sum_grad_reduce_axis(partial_a * dout, ra), sa))
            r2 = neg_(reshape_(sum_grad_reduce_axis(partial_x * dout, rx), sx))
            return r1, r2
        ra, rx = broadcast_gradient_args(sa, sx)
        partial_a = igammagrada(a, x)
        partial_x = exp_(-x + (a - 1) * log_(x) - lgamma(a))
        if ra != ():
            r1 = neg_(reshape_(reduce_sum_(partial_a * dout, ra), sa))
        else:
            r1 = neg_(reshape_(partial_a * dout, sa))
        if rx != ():
            r2 = neg_(reshape_(reduce_sum_(partial_x * dout, rx), sx))
        else:
            r2 = neg_(reshape_(partial_x * dout, sx))
        return r1, r2

    return bprop


@bprop_getters.register(Lgamma)
def get_bprop_lgamma(self):
    """Grad definition for `Lgamma` operation."""
    digamma = Digamma()

    def bprop(x, out, dout):
        if x.dtype in (mstype.float16,):
            x = F.cast(x, mstype.float32)
            dx = dout * digamma(x)
            dx = F.cast(dx, mstype.float16)
        elif x.dtype in (mstype.int32,):
            x = F.cast(x, mstype.float32)
            dx = dout * digamma(x)
        else:
            dx = dout * digamma(x)
        return (dx,)

    return bprop


@bprop_getters.register(Digamma)
def get_bprop_digamma(self):
    """Grad definition for `Digamma` operation."""
    polygamma = Polygamma()
    a = Tensor(1)

    def bprop(x, out, dout):
        if x.dtype in (mstype.float16,):
            x = F.cast(x, mstype.float32)
            dx = dout * polygamma(a, x)
            dx = F.cast(dx, mstype.float16)
        else:
            dx = dout * polygamma(a, x)
        return (dx,)

    return bprop


@bprop_getters.register(Polygamma)
def get_bprop_polygamma(self):
    """Grad definition for `Polygamma` operation."""
    polygamma = Polygamma()

    def bprop(a, x, out, dout):
        one = Tensor(1)
        a = a + one
        if x.dtype in (mstype.float16,):
            x = F.cast(x, mstype.float64)
            dx = dout * polygamma(a, x)
            dx = F.cast(dx, mstype.float16)
        else:
            dx = dout * polygamma(a, x)
        return zeros_like(a), dx

    return bprop


@bprop_getters.register(TridiagonalMatMul)
def get_bprop_tridiagonal_matmul(self):
    """Grad definition for 'TridiagonalMatMul' operation"""

    def _leftshift(x):
        """Shifts next-to-last dimension to the left, adding zero on the right."""
        rank = P.Rank()(x)
        paddings = ((0,) * (2),) * (rank - 2) + ((0, 1), (0, 0))
        pad_op = P.Pad(paddings)
        return pad_op(x[..., 1:, :])

    def _rightshift(x):
        """Shifts next-to-last dimension to the right, adding zero on the left."""
        rank = P.Rank()(x)
        paddings = ((0,) * (2),) * (rank - 2) + ((1, 0), (0, 0))
        pad_op = P.Pad(paddings)
        return pad_op(x[..., :-1, :])

    def matrix_transpose(x):
        x_rank = P.Rank()(x)
        if x_rank > 2:
            m = x_rank - 2
            n = x_rank - 1
            x_range = range(m)
            perm = (x_range) + (n, m)
        else:
            perm = (1, 0)
        return P.Transpose()(x, perm)

    reduce_sum = P.ReduceSum()
    expand_dims = P.ExpandDims()
    conjugate = P.Conj()

    def bprop(superdiag, maindiag, subdiag, rhs, out, grad):
        superdiag_type = F.dtype(superdiag)
        superdiag_conj = matrix_transpose(superdiag)
        maindiag_conj = matrix_transpose(maindiag)
        subdiag_conj = matrix_transpose(subdiag)
        rhs_conj = rhs
        if superdiag_type in (mstype.complex64, mstype.complex128):
            superdiag_conj = conjugate(superdiag_conj)
            maindiag_conj = conjugate(maindiag_conj)
            subdiag_conj = conjugate(subdiag_conj)
            rhs_conj = conjugate(rhs)
        superdiag_grad = reduce_sum(_leftshift(rhs_conj) * grad, -1)
        maindiag_grad = reduce_sum(rhs_conj * grad, -1)
        subdiag_grad = reduce_sum(_rightshift(rhs_conj) * grad, -1)
        rhs_grad = _rightshift(superdiag_conj * grad) + maindiag_conj * grad + \
                   _leftshift(subdiag_conj * grad)
        superdiag_grad = expand_dims(superdiag_grad, -2)
        maindiag_grad = expand_dims(maindiag_grad, -2)
        subdiag_grad = expand_dims(subdiag_grad, -2)
        return superdiag_grad, maindiag_grad, subdiag_grad, rhs_grad

    return bprop


@bprop_getters.register(AddV2)
def get_bprop_add_v2(self):
    """Grad definition for `AddV2` operation."""

    def bprop(x, y, out, dout):
        return binop_grad_common(x, y, dout, dout)

    return bprop


@bprop_getters.register(CholeskySolve)
def get_bprop_cholesky_solve(self):
    """Grad definition for 'CholeskySolve' operation"""
    batchmatmul_op = P.BatchMatMul()
    matmul_op = P.MatMul()
    neg_op = P.Neg()
    shape_op = P.Shape()
    upper = self.upper
    cholesky_solve = CholeskySolve(upper=self.upper)

    def bprop(x1, x2, out, dout):
        flag = 0
        shape_x1 = shape_op(x1)
        if F.is_sequence_shape_unknown(shape_x1):
            len_x1 = dyn_rank(x1)
        else:
            len_x1 = len(shape_x1)
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


@bprop_getters.register(NextAfter)
def get_bprop_nextafter(self):
    """Grad definition for 'NextAfter' operation"""
    shape = P.Shape()
    dyn_shape = P.TensorShape()
    ones = P.Ones()
    zeros = P.Zeros()
    dtype = P.DType()
    reshape = P.Reshape()
    cast = P.Cast()

    def bprop(x1, x2, out, dout):
        dout_type = dtype(dout)
        x1_type = dtype(x1)
        x2_type = dtype(x2)
        if x1_type == mstype.float64:
            x1 = cast(x1, mstype.float32)
        if x2_type == mstype.float64:
            x2 = cast(x2, mstype.float32)
        if dout_type == mstype.float64:
            dout = cast(dout, mstype.float32)

        s_x1 = shape(x1)
        partial_x1 = ()
        if F.is_sequence_value_unknown(s_x1):
            s_x1 = dyn_shape(x1)
            partial_x1 = dyn_ones(s_x1, dtype(x1))
        else:
            partial_x1 = ones(s_x1, dtype(x1))

        s_x2 = shape(x2)
        partial_x2 = ()
        if F.is_sequence_value_unknown(s_x2):
            s_x2 = dyn_shape(x2)
            partial_x2 = dyn_fill(dtype(x2), s_x2, 0)
        else:
            partial_x2 = zeros(s_x2, dtype(x2))

        dx1 = reshape(partial_x1 * dout, s_x1)
        dx2 = reshape(partial_x2 * dout, s_x2)
        return cast(dx1, dtype(dout)), cast(dx2, dtype(dout))

    return bprop


@bprop_getters.register(Diagonal)
def get_bprop_diagonal(self):
    """Grad definition for 'Diagonal' operation"""
    offset = self.offset
    dim1 = self.dim1
    dim2 = self.dim2
    zeros = P.Zeros()
    size_op = P.Size()
    transpose_op = Transpose()
    matrix_set_diag_op = MatrixSetDiagV3(align="LEFT_RIGHT")

    def bprop(x, out, dout):
        x_shape = x.shape
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
            dx = zeros(dx_trans_shape, x.dtype)
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


@bprop_getters.register(Cholesky)
def get_bprop_cholesky(self):
    """Grad definition for `Cholesky` operation."""
    upper = self.upper
    choleskygrad = G.CholeskyGrad()

    def bprop(x, out, dout):
        out = cholesky_transpose(out) if upper else out
        dout = cholesky_transpose(dout) if upper else dout
        dx = choleskygrad(out, dout)
        return (dx,)

    return bprop


@bprop_getters.register(InplaceIndexAdd)
def get_bprop_inplace_index_add(self):
    """Generate bprop for InplaceIndexAdd"""
    gather = P.Gather()
    _axis = self.axis

    def bprop(var, indices, updates, out, dout):
        return dout, zeros_like(indices), gather(dout, indices, _axis)

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
    shape_op = P.Shape()
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
                                                                                                               -1],])),
                                                    tile_op(imag_op(dout_last_term), _rfft_last_term_shape(
                                                        dout_shape, [input_shape[-1],])))
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
