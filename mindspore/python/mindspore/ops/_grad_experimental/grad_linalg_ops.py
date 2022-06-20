# Copyright 2022 Huawei Technologies Co., Ltd
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

"""Define the grad rules of linalg related operations."""
import mindspore
import numpy as np

from .. import Tensor
from .. import functional as F
from ..composite.multitype_ops.zeros_like_impl import zeros_like
from ..operations import math_ops as math
from ..operations import linalg_ops as linalg
from ..operations import array_ops as arrays
from ..primitive import constexpr
from .._grad.grad_base import bprop_getters

_shape = arrays.Shape()
_dtype = arrays.DType()
_cast = arrays.Cast()
_transpose = arrays.Transpose()

_conj = math.Conj()
_reciprocal = math.Reciprocal()

_k_0 = Tensor(0, mindspore.int32)
_padding_0 = Tensor(0, mindspore.int32)


@constexpr
def _raise_value_error(*info):
    info_str = ""
    for obj in info:
        info_str = info_str + f"{obj}"
    raise ValueError(info_str)


def _matrix_transpose(a):
    dims = a.ndim
    if dims < 2:
        _raise_value_error(
            "To do _matrix_transpose for input a's ndim is not greater or equal to 2, which is invalid.")
    axes = F.make_range(0, dims)
    axes = axes[:-2] + (axes[-1],) + (axes[-2],)
    return _transpose(a, axes)


def _adjoint(a):
    return _matrix_transpose(_conj(a))


def _safe_reciprocal(x, epsilon=1e-20):
    # Reciprocal do not support float64, force to float32
    return x * _reciprocal(_cast(x * x + epsilon, mindspore.float32))


@constexpr
def _make_tensor(value, dtype):
    return Tensor(value, dtype)


@constexpr
def _make_zero_matrix(shape, dtype):
    return Tensor(np.zeros(shape), dtype)


def _matrix_diag(diagonal):
    diagonal_shape = _shape(diagonal)
    row = _make_tensor(diagonal_shape[-1], mindspore.int32)
    return arrays.MatrixDiagV3()(diagonal, _k_0, row, row, _make_tensor(0, _dtype(diagonal)))


def _mat_mul(x, y):
    shape = _shape(x)
    if len(shape) > 2:
        return math.BatchMatMul()(x, y)
    return math.MatMul()(x, y)


@bprop_getters.register(linalg.Svd)
def get_bprop_svd(self):
    """Generate bprop for Svd"""
    full_matrices = self.full_matrices
    compute_uv = self.compute_uv

    svd = linalg.Svd(compute_uv=True)
    square = math.Square()
    matrix_set_diag = arrays.MatrixSetDiagV3()
    expand_dims = arrays.ExpandDims()

    def bprop(a, out, dout):
        if not compute_uv:
            s, u, v = svd(a)
            da = _mat_mul(u, _mat_mul(_matrix_diag(_cast(dout[0], _dtype(a))), _adjoint(v)))
            return (da,)

        a_shape = _shape(a)
        if len(a_shape) < 2:
            _raise_value_error(
                "For input a's ndim is not greater or equal to 2, which is invalid.")
        m = a_shape[-2]
        n = a_shape[-1]

        s, u, v = out
        ds, du, dv = dout
        use_adjoint = False
        if m > n:
            use_adjoint = True
            m, n = n, m
            u, v = v, u
            du, dv = dv, du

        if full_matrices and abs(m - n) > 1:
            _raise_value_error("For 'Svd' gradient, not support for abs(m - n) > 1 with full_matrices is True.")

        s_mat = _matrix_diag(s)
        s2 = square(s)

        f = matrix_set_diag(_safe_reciprocal(expand_dims(s2, -2) - expand_dims(s2, -1)), zeros_like(s), _k_0)
        s_inv_mat = _matrix_diag(_safe_reciprocal(s))

        v1 = v[..., :, :m]
        dv1 = dv[..., :, :m]

        u_gu = _mat_mul(_adjoint(u), du)
        v_gv = _mat_mul(_adjoint(v1), dv1)

        f_u = f * u_gu
        f_v = f * v_gv
        ds_mat = _matrix_diag(_cast(ds, _dtype(a)))
        term1_nouv = (ds_mat + _mat_mul(f_u + _adjoint(f_u), s_mat) + _mat_mul(s_mat, f_v + _adjoint(f_v)))

        term1 = _mat_mul(u, _mat_mul(term1_nouv, _adjoint(v1)))

        if m == n:
            da_before_transpose = term1
        else:
            gv1t = _matrix_transpose(dv1)
            gv1t_v1 = _mat_mul(gv1t, v1)
            term2_nous = gv1t - _mat_mul(gv1t_v1, _adjoint(v1))

            if full_matrices:
                v2 = v[..., :, m:n]
                d_v2 = dv[..., :, m:n]

                v1t_gv2 = _mat_mul(_adjoint(v1), d_v2)
                term2_nous -= _mat_mul(v1t_gv2, _adjoint(v2))

            u_s_inv = _mat_mul(u, s_inv_mat)
            term2 = _mat_mul(u_s_inv, term2_nous)

            da_before_transpose = term1 + term2

        if use_adjoint:
            da = _matrix_transpose(da_before_transpose)
        else:
            da = da_before_transpose

        return (da,)

    return bprop
