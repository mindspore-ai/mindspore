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
# ===========================================================================
"""Jacobin operations."""
import numpy as np

import mindspore as ms
from mindspore.ops import functional as F
import mindspore.ops.operations as P
from mindspore import Tensor

from .listvec import list_zero

squeeze_1 = P.Squeeze(1)
indexadd_0 = P.IndexAdd(axis=0)
batchmatmul = P.BatchMatMul()


def jacobi_column_square(indices, jacobians, jacobian_scale):
    """column square"""
    for _, i in enumerate(range(len(jacobians))):
        index = indices[i]
        jacobi = jacobians[i]
        jacobi_square = F.reduce_sum(F.mul(jacobi, jacobi), axis=0)
        indexadd_0(jacobian_scale[i], squeeze_1(index), squeeze_1(jacobi_square))
    return jacobian_scale


def column_inverse_square(jacobian_scale):
    """column inverse"""
    for _, i in enumerate(range(len(jacobian_scale))):
        column_norm_shape = jacobian_scale[i].shape
        numDimV, *_ = column_norm_shape

        jacobian_scale[i][:numDimV] = 1.0 / (1.0 + F.sqrt(jacobian_scale[i][:numDimV]))


def jacobi_squared_column_norm(jacobians, indices, variables):
    """column square with column_norm"""
    column_norm = list_zero(variables)
    return jacobi_column_square(indices, jacobians, column_norm)


def jacobi_normalize(jacobi, indices, variables):
    """normalize"""
    jacobian_scale = jacobi_squared_column_norm(jacobi, indices, variables)

    for _, i in enumerate(range(len(jacobian_scale))):
        jacobian_scale[i] = 1.0 / (1.0 + F.sqrt(jacobian_scale[i]))
        column = jacobian_scale[i][indices[i]]
        shape_dim = [i for i in column.shape]
        shape_dim.insert(0, jacobi[i].shape[0])

        column = P.BroadcastTo(tuple(shape_dim))(F.expand_dims(column, 0))

        jacobi[i] = F.mul(jacobi[i], column)
    return jacobian_scale


def jacobi_block_jt(jacobians, lmDiagonal, indices, res):
    """block JtJ"""
    # t0 = time()
    for r in res:
        r = F.zeros_like(r)
    for _, varid in enumerate(range(len(jacobians))):
        jacobian = jacobians[varid]
        j_plain = F.reshape(jacobian, (jacobian.shape[0], jacobian.shape[1], -1))
        jt_js = batchmatmul(F.transpose(j_plain, (1, 2, 0)), F.transpose(j_plain, (1, 0, 2)))
        indexadd_0(res[varid], squeeze_1(indices[varid]), jt_js)

    for _, varid in enumerate(range(len(res))):
        diagonal = lmDiagonal[varid]
        diagonal = F.reshape(diagonal, (diagonal.shape[0], -1))

        diagonal_pow2 = diagonal * diagonal
        diagonal_pow2 = F.reshape(diagonal_pow2, (*diagonal_pow2.shape, 1))
        res[varid] = diagonal_pow2 * Tensor(np.eye(diagonal.shape[1]), ms.float32) + res[varid]


def jacobi_left_multiply(jacobians, residuals, variables, indices, res):
    """left multiply"""
    # t0 = time()
    for _, i in enumerate(range(len(res))):
        res[i] = F.zeros_like(res[i])

    for _, varid in enumerate(range(len(variables))):
        jacobian = jacobians[varid]
        j = F.reshape(jacobian, (jacobian.shape[0], jacobian.shape[1], -1))
        j = F.transpose(j, (1, 0, 2))
        r = F.reshape(residuals, (residuals.shape[0], residuals.shape[1], 1))

        r = P.BroadcastTo(j.shape)(r)

        jr = F.reduce_sum(F.mul(j, r), 1)
        indexadd_0(res[varid], squeeze_1(indices[varid]), jr)
    # t1 = time()
    return res


def jacobi_right_multiply(jacobians, residuals, variables, indices, res):
    """right multiply"""
    # t0 = time()
    res = F.zeros_like(res)

    for _, varid in enumerate(range(len(variables))):
        jacobian = jacobians[varid]
        residual = residuals[varid][indices[varid]]

        targetShape = [i for i in jacobian.shape]
        currentShape = [i for i in jacobian.shape]
        currentShape[0] = 1

        residual = P.BroadcastTo(tuple(targetShape))(
            F.reshape(residual, tuple(currentShape)))

        res += F.transpose(F.reduce_sum(F.reshape(F.tensor_mul(residual, jacobian),
                                                  (jacobian.shape[0], jacobian.shape[1], -1)),
                                        axis=2), (1, 0))
    # t1 = time()
    return res


def jacobi_jt_jd(jacobians, diagonal, p, variables, indices, z, res):
    """JtJD"""
    for _, i in enumerate(range(len(z))):
        z[i] = jacobi_right_multiply(jacobians, p, variables, indices, z[i])
        res = jacobi_left_multiply(jacobians, z[i], variables, indices, res)

    for _, j in enumerate(range(len(res))):
        # res[i] += p[i] * (diagonal[i] ** 2)
        res[j] += F.mul(p[j], F.mul(diagonal[j], diagonal[j]))
