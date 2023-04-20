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
"""Grad implementation of operators for scipy submodule"""
from .. import numpy as mnp
from .ops import Eig, SolveTriangular
from .utils_const import _raise_type_error
from .ops_wrapper import matrix_set_diag
from ..ops import operations as P
from ..ops import functional as F
from ..ops.operations.linalg_ops import Eigh
from ..ops._grad_experimental.grad_base import bprop_getters
from ..common import dtype as mstype

_matmul = P.MatMul(False, False)
_real = P.Real()
_conj = P.Conj()


def _compute_f(w, epsilon=1E-20):
    diff = F.expand_dims(w, -2) - F.expand_dims(w, -1)
    diff_inv = diff / (diff * diff + epsilon)
    f = matrix_set_diag(diff_inv, F.zeros_like(w))
    return f


def _adjoint(a):
    return _conj(a).T


def _diag(a):
    return F.expand_dims(a, -2) * F.eye(F.shape(a)[-1], F.shape(a)[-1], a.dtype)


def _matrix_solve(a, b):
    # work in process
    return a + b


def _batch_eyes(a):
    num_row = F.shape(a)[-1]
    batch_shape = F.shape(a)[:-2]
    shape = batch_shape + (num_row, num_row)
    eye = F.eye(num_row, num_row, a.dtype)
    return mnp.broadcast_to(eye, shape)


@bprop_getters.register(Eig)
def get_bprpo_eig(self):
    """Grad definition for `Eig` operation."""
    is_compute_v = self.compute_eigenvectors

    def bprop(a, out, dout):
        w, v, grad_w, grad_v = out[0], out[1], dout[0], dout[1]
        if not is_compute_v:
            gw_vh = F.expand_dims(grad_w, -1) * _adjoint(v)
            grad_a = _matrix_solve(_adjoint(v), gw_vh)  # not support
        else:
            vh = _adjoint(v)
            vh_gv = _matmul(vh, grad_v)
            vh_gv_diag = vh_gv.diagonal(0, -2, -1)
            vh_gv = vh_gv - _matmul(vh, v * _real(vh_gv_diag))
            f = _compute_f(w)
            grad_a = _diag(grad_w) + f * vh_gv
            grad_a = _matrix_solve(vh, _matmul(grad_a, vh))  # not support
        return (grad_a,)

    return bprop


@bprop_getters.register(Eigh)
def get_bprpo_eigh(self):
    """Grad definition for `Eigh` operation.
    Eq. 4.71 from Christoph Boeddeker, et al. 'On the Computation of Complex-valued Gradients
    with Application to Statistically Optimum Beamforming', 2017, pg. 28-29
    """
    is_compute_v = self.compute_eigenvectors
    is_lower = self.lower
    lower = int(is_lower)
    eigh = Eigh(compute_eigenvectors=True)

    def bprop(a, out, dout):
        if a.dtype in [mstype.complex64, mstype.complex128]:
            _raise_type_error(
                "For 'Eigh' operation, the data type of input 'a' don't support the complex64 or complex128.")
        if not is_compute_v:
            _, v = eigh(a)
            grad_a = _matmul(v * F.expand_dims(dout, -2), _adjoint(v))
        else:
            vh = _adjoint(out[1])
            vh_gv = _matmul(vh, dout[1])
            f = _compute_f(out[0])
            mid_part = _diag(dout[0]) + f * vh_gv
            grad_a = _matmul(out[1], _matmul(mid_part, vh))

        # The forward implementation only focus on lower part or upper part,
        # so we only retain the corresponding part.
        grad_a = grad_a + _adjoint(grad_a)
        grad_a = F.matrix_band_part(grad_a, 0 - lower, lower - 1)
        middle_diag = 0.5 * grad_a.diagonal(0, -2, -1)
        grad_a = matrix_set_diag(grad_a, middle_diag)
        return (grad_a,)

    return bprop


@bprop_getters.register(SolveTriangular)
def get_bprpo_trsm(self):
    """Grad definition for `SolveTriangular` operation.
    Appendix(see trsm) from Matthias Seeger, et al. 'Auto-Differentiating Linear Algebra', 2017, pg. 28-29
    """
    is_lower = self.lower
    is_unit_diagonal = self.unit_diagonal
    lower = int(is_lower)
    bp_trans = ("N" if self.trans in ["T", "C"] else "T")
    solve_triangular = SolveTriangular(is_lower, is_unit_diagonal, bp_trans)

    def bprop(a, b, out, dout):
        row_size = F.shape(a)[-2]
        grad_b = solve_triangular(a, dout)
        grad_b_align = F.reshape(grad_b, (row_size, -1))
        x_align = F.reshape(out, (row_size, -1))
        if bp_trans in ["T", "C"]:
            grad_a = _matmul(grad_b_align, _adjoint(x_align))
        else:
            grad_a = _matmul(x_align, _adjoint(grad_b_align))

        grad_a = -1 * F.matrix_band_part(grad_a, 0 - lower, lower - 1)
        if is_unit_diagonal:
            grad_a = matrix_set_diag(grad_a, F.fill(grad_a.dtype, (row_size,), 0))
        return grad_a, grad_b

    return bprop
