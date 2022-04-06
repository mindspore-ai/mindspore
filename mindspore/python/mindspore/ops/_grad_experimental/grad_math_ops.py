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

"""Define the grad rules of math related operations."""

from mindspore.common import dtype as mstype
from mindspore import nn
import mindspore.numpy as mnp
import numpy as np
from .. import functional as F
from .. import operations as P
from ..operations.math_ops import Trace
from .._grad.grad_base import bprop_getters
from .._grad.grad_math_ops import binop_grad_common
from ..composite.multitype_ops.zeros_like_impl import zeros_like
from ..operations import _grad_ops as G
from ..operations import math_ops as math
from ..primitive import constexpr
from ..operations.math_ops import ReduceStd


transpose = P.Transpose()


@constexpr
def _generate_perm(x_dim):
    perm = tuple(range(x_dim - 2))
    return perm


@bprop_getters.register(P.ACos)
def get_bprop_acos(self):
    """Grad definition for `ACos` operation."""
    input_grad = G.ACosGrad()

    def bprop(input_x, out, dout):
        dx = input_grad(input_x, dout)
        return (dx,)

    return bprop


@bprop_getters.register(P.Cdist)
def get_bprop_cdist(self):
    """Generate bprop for Cdist"""
    input_grad = G.CdistGrad(p=self.p)

    def bprop(input_x, input_y, out, dout):
        dout_shape = F.shape(dout)
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
        dx1 = mul_op(dout, div_op(value, x2))
        dx2 = neg_op(mul_op(mul_op(mul_op(x1, value), pow_op(x2, -2)), dout))
        dvalue = mul_op(dout, div_op(x1, x2))
        return dout, dx1, dx2, dvalue

    return bprop


@bprop_getters.register(P.Addcmul)
def get_bprop_index_addcmul(self):
    """Generate bprop for Addcmul"""
    mul_op = P.Mul()

    def bprop(input_data, x1, x2, value, out, dout):
        dx1 = mul_op(dout, mul_op(value, x2))
        dx2 = mul_op(dout, mul_op(value, x1))
        dvalue = mul_op(dout, mul_op(x1, x2))
        return dout, dx1, dx2, dvalue

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
            input_scaled = pow_op(abs_op(input_x), (p-2)) * input_x
            scale_v = dout / pow_op(out, (p-1))
        return (input_scaled * scale_v,)

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


@bprop_getters.register(P.MatrixDeterminant)
def get_bprop_matrix_determinant(self):
    """Generate bprop for MatrixDeterminant"""
    inverse_op = P.MatrixInverse(adjoint=True)
    shape_op = P.Shape()
    reshape = P.Reshape()

    def bprop(x, out, dout):
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
        multipliers = reshape(dout[1], shape_op(out[1]) + (1, 1))
        dx = multipliers * x_adj_inv
        return (dx,)

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
        dout_square = square(dout)
        dx = dout * root_pi_over_two * exp(dout_square)
        return (dx,)

    return bprop


@bprop_getters.register(P.BesselI0)
def get_bprop_bessel_i0(self):
    """Generate bprop for BesselI0"""
    bessel_i1 = P.BesselI1()

    def bprop(x, out, dout):
        dx = dout * bessel_i1(x)
        return (dx,)
    return bprop


@bprop_getters.register(P.BesselI1)
def get_bprop_bessel_i1(self):
    """Generate bprop for BesselI1"""
    equal = P.Equal()
    div = P.Div()
    cast = P.Cast()
    dtype = P.DType()
    bessel_i0 = P.BesselI0()

    def bprop(x, out, dout):
        dout_dx = mnp.where(equal(x, 0.), cast(1., dtype(x)), bessel_i0(x) - div(out, x))
        dx = dout * dout_dx
        return (dx,)
    return bprop


@bprop_getters.register(math.BesselK0)
def get_bprop_bessel_k0(self):
    """Generate bprop for BesselK0"""
    bessel_k1 = math.BesselK1()

    def bprop(x, out, dout):
        dx = -dout * bessel_k1(x)
        return (dx,)

    return bprop


@bprop_getters.register(math.BesselK1)
def get_bprop_bessel_k1(self):
    """Generate bprop for BesselK1"""
    div = P.Div()
    bessel_k0 = math.BesselK0()

    def bprop(x, out, dout):
        dout_dx = -(bessel_k0(x) + div(out, x))
        dx = dout * dout_dx
        return (dx,)

    return bprop


@bprop_getters.register(math.BesselK0e)
def get_bprop_bessel_k0e(self):
    """Generate bprop for BesselK0e"""
    bessel_k1e = math.BesselK1e()

    def bprop(x, out, dout):
        dx = dout * (out - bessel_k1e(x))
        return (dx,)
    return bprop


@bprop_getters.register(math.BesselK1e)
def get_bprop_bessel_k1e(self):
    """Generate bprop for BesselK1e"""
    reciprocal = P.Reciprocal()
    bessel_k0e = math.BesselK0e()

    def bprop(x, out, dout):
        dout_dx = out * (1. - reciprocal(x)) - bessel_k0e(x)
        dx = dout * dout_dx
        return (dx,)
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
        dx = input_grad(dout, cast(to_array(shape), mstype.int64))
        return (dx,)

    return bprop
