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
from mindspore.nn import LGamma
from mindspore.ops import functional as F
from mindspore.ops.functional import broadcast_gradient_args
from mindspore.ops import operations as P
from mindspore.ops.operations.math_ops import Trace, Bernoulli, Renorm
from mindspore import nn, ops, Tensor
from mindspore.ops.operations.math_ops import Real, Imag, Complex, Angle
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
from mindspore.ops.operations.math_ops import NextAfter
from mindspore.ops.operations.math_ops import Hypot
from mindspore.ops.operations.math_ops import ReduceStd
from mindspore.ops.operations.math_ops import LuUnpack
from mindspore.ops.operations.math_ops import MatrixExp
from mindspore.ops.operations.math_ops import CumulativeLogsumexp
from mindspore.ops.operations.math_ops import MatrixSolve
from mindspore.ops.operations.math_ops import MatrixPower
from mindspore.ops.operations.math_ops import Median
from mindspore.ops.operations.math_ops import Betainc
from mindspore.ops.operations.math_ops import Cholesky
from mindspore.ops.operations.math_ops import CholeskySolve
from mindspore.ops.operations.math_ops import AddV2
from mindspore.ops.operations.math_ops import TridiagonalMatMul
from mindspore.ops.operations.math_ops import Logit
from mindspore.ops.operations._inner_ops import DynamicBroadcastGradientArgs
from mindspore.ops.composite.multitype_ops.zeros_like_impl import zeros_like
from mindspore.ops.primitive import constexpr
from mindspore.ops._utils.utils import is_shape_unknown, is_dim_unknown
from mindspore.ops._grad.grad_base import bprop_getters, create_tensor_by_element, dyn_rank
from mindspore.ops._grad.grad_base import dyn_ones, dyn_fill, sum_grad_reduce_axis
from mindspore.ops._grad.grad_math_ops import binop_grad_common

transpose = P.Transpose()
dyn_shape_op = P.TensorShape()
_conj = P.Conj()


@constexpr
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


@bprop_getters.register(P.Cdist)
def get_bprop_cdist(self):
    """Generate bprop for Cdist"""
    input_grad = G.CdistGrad(p=self.p)

    def bprop(input_x, input_y, out, dout):
        dout_shape = F.shape(dout)
        if is_shape_unknown(dout_shape):
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
    is_instance_op = P.IsInstance()

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


@constexpr
def renew_dim(shape, dim):
    """ Re-new dims"""
    new_dim = dim if dim >= 0 else len(shape) + dim
    tmp = [i for i in range(len(shape))]
    _ = tmp.pop(new_dim)
    return tuple(tmp)


@bprop_getters.register(Renorm)
def get_bprop_renorm(self):
    """Generate bprop for Renorm """
    p = int(self.p)
    ext = 1e-7
    dim = self.dim
    max_norm = self.maxnorm
    greater_op = P.Greater()
    masked_fill_op = P.MaskedFill()
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
            norm_bp = masked_fill_op(m, norm == 0., 0.)
        else:
            abs_ = abs_op(input_x)
            input_scaled = input_x * pow_op(abs_, (p - 2))
            pow_ = pow_op(norm, (p - 1))
            scale_v = grad_out / pow_
            scale_v = masked_fill_op(scale_v, norm == 0., 0.)
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
        fp64_flag = False
        if x.dtype == mstype.float16:
            dtype_min = -65500e+0
        elif x.dtype == mstype.float32:
            dtype_min = -3.4028235e+38
        elif x.dtype == mstype.float64:
            dout = F.cast(dout, mstype.float32)
            x = F.cast(x, mstype.float32)
            out = F.cast(out, mstype.float32)
            dtype_min = -3.4028235e+38
            fp64_flag = True

        log_grad_positive = where_v2(greater_op(dout, 0), log_op(dout), dtype_min)
        log_grad_negative = where_v2(less_op(dout, 0), log_op(neg_op(dout)), dtype_min)
        output_pos = exp_op(cumulative_op(log_grad_positive - out, axis) + x)
        output_neg = exp_op(cumulative_op(log_grad_negative - out, axis) + x)
        if fp64_flag:
            output_pos = F.cast(output_pos, mstype.float64)
            output_neg = F.cast(output_neg, mstype.float64)
            x = F.cast(x, mstype.float64)
        return (output_pos - output_neg, zeros_like(x))

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
        if is_shape_unknown(x.shape):
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

        if is_shape_unknown(x.shape):
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


@bprop_getters.register(P.MatrixInverse)
def get_bprop_matrix_inverse(self):
    """Generate bprop for MatrixInverse"""
    matmul_x1 = nn.MatMul(transpose_x1=True)
    matmul_x2 = nn.MatMul(transpose_x2=True)
    neg = P.Neg()

    def bprop(x, out, dout):
        dx = matmul_x2(dout, out)
        dx = matmul_x1(out, dx)
        dx = neg(dx)
        return (dx,)

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
        if is_shape_unknown(a_shape):
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


@bprop_getters.register(P.MatrixDeterminant)
def get_bprop_matrix_determinant(self):
    """Generate bprop for MatrixDeterminant"""
    inverse_op = P.MatrixInverse(adjoint=True)
    shape_op = P.Shape()
    reshape = P.Reshape()
    concat = P.Concat(0)

    def bprop(x, out, dout):
        if is_shape_unknown(shape_op(x)):
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
        if is_shape_unknown(shape_op(out[1])):
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
        if is_shape_unknown(F.shape(input_x)):
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
        if is_shape_unknown(shape):
            shape = dyn_shape_op(x)
            dx = input_grad(dout, shape)
        else:
            dx = input_grad(dout, cast(to_array(shape), mstype.int64))
        return (dx,)

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
        if is_shape_unknown(sa) or is_shape_unknown(sx):
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
        if is_shape_unknown(sa) or is_shape_unknown(sx):
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
        if is_dim_unknown(shape_x1):
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
        if is_shape_unknown(s_x1):
            s_x1 = dyn_shape(x1)
            partial_x1 = dyn_ones(s_x1, dtype(x1))
        else:
            partial_x1 = ones(s_x1, dtype(x1))

        s_x2 = shape(x2)
        partial_x2 = ()
        if is_shape_unknown(s_x2):
            s_x2 = dyn_shape(x2)
            partial_x2 = dyn_fill(dtype(x2), s_x2, 0)
        else:
            partial_x2 = zeros(s_x2, dtype(x2))

        dx1 = reshape(partial_x1 * dout, s_x1)
        dx2 = reshape(partial_x2 * dout, s_x2)
        return cast(dx1, dtype(dout)), cast(dx2, dtype(dout))

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
