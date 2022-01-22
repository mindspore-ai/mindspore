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
from .ops import Eigh, Eig
from .. import dtype as mstype
from ..ops import operations as P
from ..ops import functional as F
from ..ops._grad.grad_base import bprop_getters

_matmul = P.MatMul(False, False)
_real = P.Real()
_conj = P.Conj()


def _compute_f(w, epsilon=1E-20):
    diff = F.expand_dims(w, -2) - F.expand_dims(w, -1)
    diff_inv = diff / (diff * diff + epsilon)
    eye_mask = F.eye(F.shape(diff)[-1], F.shape(diff)[-2], mstype.bool_)
    zeros = F.zeros_like(diff)
    return F.select(eye_mask, zeros, diff_inv)


def _adjoint(a):
    return _conj(a).T


def _diag(a):
    return F.expand_dims(a, -2) * F.eye(F.shape(a)[-1], F.shape(a)[-1], a.dtype)


def _matrix_solve(a, b):
    # work in process
    return a + b


@bprop_getters.register(Eig)
def get_bprpo_eig(self):
    """Grad definition for `Eig` operation."""
    is_compute_v = self.compute_eigenvectors

    def bprop(a, out, dout):
        w, v, grad_w, grad_v = out[0], out[1], dout[0], dout[1]
        if not is_compute_v:
            # w, _ = Eig(compute_eigenvectors=False)(a) -> a * _ = w * _
            # where a is a general matrix
            gw_vh = F.expand_dims(grad_w, -1) * _adjoint(v)
            grad_a = _matrix_solve(_adjoint(v), gw_vh)  # not support
        else:
            # w, v = Eig(compute_eigenvectors=True)(a)  -> a * v = w * v
            # where a is a general matrix
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

    def bprop(a, out, dout):
        w, v, grad_w, grad_v = out[0], out[1], dout[0], dout[1]
        if not is_compute_v:
            # w, _ = Eigh(compute_eigenvectors=False)(a) -> a * _ = w * _
            _, v = Eigh(compute_eigenvectors=True)(a)
            grad_a = _matmul(v * F.expand_dims(grad_w, -2), _adjoint(v))
        else:
            # w, v = Eigh(compute_eigenvectors=True)(a)  -> a * v = w * v
            vh_gv = _matmul(_adjoint(v), grad_v)
            f = _compute_f(w)
            mid_part = _diag(grad_w) + f * vh_gv
            grad_a = _matmul(v, _matmul(mid_part, _adjoint(v)))

        # The forward implementation only focus on lower part or upper part,
        # so we only retain the corresponding part.
        if is_lower:
            grad_a = mnp.tril(grad_a + _adjoint(grad_a))
        else:
            grad_a = mnp.triu(grad_a + _adjoint(grad_a))
        grad_a = grad_a - 0.5 * _diag(grad_a.diagonal(0, -2, -1))

        return (grad_a,)

    return bprop
