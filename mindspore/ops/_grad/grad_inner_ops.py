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

"""Generate bprop for quantization aware ops"""

from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops.primitive import constexpr
from .grad_base import bprop_getters
from ..operations import _inner_ops as inner
from ..composite.multitype_ops.zeros_like_impl import zeros_like


def _get_matrix_diag_assist(x_shape, x_dtype):
    base_eye = P.Eye()(x_shape[-1], x_shape[-1], x_dtype).flatten()
    tile = P.Tile()(base_eye, x_shape[:-1])
    assist = P.Reshape()(tile, x_shape + (x_shape[-1],))
    return assist


def _get_matrix_diag_part_assist(x_shape, x_dtype):
    base_eye = P.Eye()(x_shape[-2], x_shape[-1], x_dtype).flatten()
    tile = P.Tile()(base_eye, x_shape[:-2])
    assist = P.Reshape()(tile, x_shape)
    return assist


@constexpr
def _get_min(x):
    return min(x)


@bprop_getters.register(inner.MatrixDiag)
def get_bprop_matrix_diag(self):
    """Generate bprop for MatrixDiag"""
    get_dtype = P.DType()

    def bprop(x, y, out, dout):
        shape = F.shape(dout)
        dtype = get_dtype(dout)
        assist = _get_matrix_diag_part_assist(shape, dtype)
        dx = inner.MatrixDiagPart()(dout, assist)
        return dx, zeros_like(y)

    return bprop


@bprop_getters.register(inner.MatrixDiagPart)
def get_bprop_matrix_diag_part(self):
    """Generate bprop for MatrixDiagPart"""
    get_dtype = P.DType()

    def bprop(x, y, out, dout):
        x_shape = F.shape(x)[-2:]
        if x_shape[0] == x_shape[1]:
            shape = F.shape(dout)
            dtype = get_dtype(dout)
            assist = _get_matrix_diag_assist(shape, dtype)
            return inner.MatrixDiag()(dout, assist), zeros_like(y)
        shape = F.shape(x)
        dtype = get_dtype(x)
        assist = _get_matrix_diag_part_assist(shape, dtype)
        return inner.MatrixSetDiag()(zeros_like(x), dout, assist), zeros_like(y)

    return bprop


@bprop_getters.register(inner.MatrixSetDiag)
def get_bprop_matrix_set_diag(self):
    """Generate bprop for MatrixSetDiag"""
    get_dtype = P.DType()

    def bprop(x, y, z, out, dout):
        input_shape = F.shape(x)
        batch_shape = input_shape[:-2]
        matrix_shape = input_shape[-2:]
        diag_shape = batch_shape + (_get_min(matrix_shape),)

        grad_shape = F.shape(dout)
        grad_dtype = get_dtype(dout)
        assist = _get_matrix_diag_part_assist(grad_shape, grad_dtype)
        dx = inner.MatrixSetDiag()(dout, P.Zeros()(diag_shape, grad_dtype), assist)
        dy = inner.MatrixDiagPart()(dout, assist)
        dz = zeros_like(z)
        return dx, dy, dz

    return bprop
