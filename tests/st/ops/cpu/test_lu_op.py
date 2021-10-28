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
from typing import Generic
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
import mindspore.common.dtype as mstype
from mindspore.ops import PrimitiveWithInfer
from mindspore.ops import prim_attr_register
from mindspore._checkparam import Validator as validator
import mindspore.numpy as mnp
import scipy as scp
import numpy as np
import pytest

np.random.seed(0)

context.set_context(mode=context.GRAPH_MODE, device_target='CPU')


class LU(PrimitiveWithInfer):
    """
    LU decomposition with partial pivoting
    P.A = L.U
    """

    @prim_attr_register
    def __init__(self):
        super().__init__(name="LU")
        self.init_prim_io_names(inputs=['x'], outputs=['lu', 'pivots', 'permutation'])

    def __infer__(self, x):
        x_shape = list(x['shape'])
        x_dtype = x['dtype']
        ndim = len(x_shape)
        permutation_shape = x_shape
        if ndim == 0:
            pivots_shape = x_shape
        elif ndim == 1:
            pivots_shape = x_shape[:-1]
        else:
            pivots_shape = x_shape[-2:-1]
        output = {
            'shape': (x_shape, pivots_shape, permutation_shape),
            'dtype': (x_dtype, mstype.int32, mstype.int32),
            'value': None
        }
        return output


class LUSolver(PrimitiveWithInfer):
    """
    LUSolver for Ax = b
    """

    @prim_attr_register
    def __init__(self, trans: str):
        super().__init__(name="LUSolver")
        self.init_prim_io_names(inputs=['x', 'b'], outputs=['output'])
        self.trans = validator.check_value_type("trans", trans, [str], self.name)

    def __infer__(self, x, b):
        b_shape = list(b['shape'])
        x_dtype = x['dtype']
        output = {
            'shape': tuple(b_shape),
            'dtype': x_dtype,
            'value': None
        }
        return output


class LuNet(nn.Cell):
    def __init__(self):
        super(LuNet, self).__init__()
        self.lu = LU()

    def construct(self, a):
        return self.lu(a)


def lu_pivots_to_permutation(pivots, permutation_size: int):
    batch_dims = pivots.shape[:-1]
    k = pivots.shape[-1]
    per = mnp.arange(0, permutation_size)
    permutation = mnp.broadcast_to(per, batch_dims + (permutation_size,))
    permutation = mnp.array(permutation)
    if permutation_size == 0:
        return permutation

    for i in range(k):
        j = pivots[..., i]
        loc = mnp.ix_(*(mnp.arange(0, b) for b in batch_dims))
        x = permutation[..., i]
        y = permutation[loc + (j,)]
        permutation[..., i] = y
        permutation[loc + (j,)] = x
    return permutation


def _lu_solve_core(in_lu, permutation, b, trans):
    m = in_lu.shape[0]
    res_shape = b.shape[1:]
    prod_result = 1
    for sh in res_shape:
        prod_result *= sh
    x = mnp.reshape(b, (m, prod_result))
    if trans == 0:
        trans_str = "N"
        x = x[permutation, :]
    elif trans == 1:
        trans_str = "T"
    elif trans == 2:
        trans_str = "C"
    else:
        raise ValueError("trans error, it's value must be 0, 1, 2")
    ms_lu_solve = LUSolver(trans_str)
    output = ms_lu_solve(in_lu, x)
    return mnp.reshape(output, b.shape)


def _check_lu_shape(in_lu, b):
    if len(in_lu.shape) < 2 or in_lu.shape[-1] != in_lu.shape[-2]:
        raise ValueError("last two dimensions of LU decomposition must be equal.")

    if b.shape is None:
        raise ValueError(" LU decomposition input b's rank must >=1.")
    rhs_vector = in_lu.ndim == b.ndim + 1
    if rhs_vector:
        if b.shape[-1] != in_lu.shape[-1]:
            raise ValueError("LU decomposition: lu matrix and b must have same number of dimensions")
        mnp.expand_dims(b, axis=1)
    else:
        if b.shape[-2] != in_lu.shape[-1]:
            raise ValueError("LU decomposition: lu matrix and b must have same number of dimensions")


def lu_factor(a, overwrite_a=False, check_finite=True):
    del overwrite_a, check_finite
    mscp_lu = LuNet()
    m_lu, pivots, _ = mscp_lu(a)
    return m_lu, pivots


def lu(a, permute_l=False, overwrite_a=False, check_finite=True):
    del overwrite_a, check_finite
    mscp_lu = LuNet()
    m_lu, _, p = mscp_lu(a)
    m = a.shape[-2]
    n = a.shape[-1]
    k = min(m, n)
    a_dtype = a.dtype
    l = mnp.tril(m_lu, -1)[:, :k] + mnp.eye(m, k, dtype=a_dtype)
    u = mnp.triu(m_lu)[:k, :]
    if permute_l:
        return mnp.matmul(p, l), u
    return p, l, u


def lu_solve(lu_and_piv, b, trans=0, overwrite_b=False, check_finite=True):
    del overwrite_b, check_finite
    m_lu, pivots = lu_and_piv
    # 1. check shape
    _check_lu_shape(m_lu, b)
    # here permutation array has been calculated, just use it.
    # 2. calculate permutation
    permutation = pivots
    # 3. rhs_vector
    rhs_vector = m_lu.ndim == b.ndim + 1
    x = _lu_solve_core(m_lu, permutation, b, trans)

    return x[..., 0] if rhs_vector else x


def create_full_rank_matrix(m, n, dtype):
    a_rank = 0
    a = np.random.random((m, n)).astype(dtype)
    while a_rank != m:
        a = (a + np.eye(m, n)).astype(dtype)
        a_rank = np.linalg.matrix_rank(a)
    return a


def create_sym_pos_matrix(m, n, dtype):
    a = (np.random.random((m, n)) + np.eye(m, n)).astype(dtype)
    return np.dot(a, a.T)


@pytest.mark.platform_x86_cpu
@pytest.mark.parametrize('n', [10, 20])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_square_lu_net(n: int, dtype: Generic):
    """
    Feature: ALL To ALL
    Description: test cases for lu decomposition test cases for A[N,N]x = b[N,1]
    Expectation: the result match to scipy
    """
    a = create_full_rank_matrix(n, n, dtype)
    s_lu, _ = scp.linalg.lu_factor(a)
    mscp_lu_net = LuNet()
    tensor_a = Tensor(a)
    mscp_lu, _, _ = mscp_lu_net(tensor_a)
    assert np.allclose(mscp_lu.asnumpy(), s_lu, rtol=1.e-3, atol=1.e-3)


@pytest.mark.platform_x86_cpu
@pytest.mark.parametrize('n', [10, 20])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_lu_solver_net(n: int, dtype: Generic):
    """
    Feature: ALL To ALL
    Description: test cases for lu_solve test cases for A[N,N]x = b[N,1]
    Expectation: the result match to scipy
    """
    a = create_full_rank_matrix(n, n, dtype)
    b = np.random.random((n, 1)).astype(dtype)
    s_lu, s_piv = scp.linalg.lu_factor(a)

    tensor_a = Tensor(a)
    tensor_b = Tensor(b)
    mscp_lu_net = LuNet()
    mscp_lu, pivots, _ = mscp_lu_net(tensor_a)
    np.allclose(mscp_lu.asnumpy(), s_lu, rtol=1.e-3, atol=1.e-3)

    lu_factor_x = (s_lu, s_piv)
    msc_lu_factor = (mscp_lu, pivots)

    scp_x = scp.linalg.lu_solve(lu_factor_x, b)
    mscp_x = lu_solve(msc_lu_factor, tensor_b)

    real_b = mnp.dot(tensor_a, mscp_x)
    expected_b = np.dot(a, scp_x)

    assert np.allclose(real_b.asnumpy(), expected_b, rtol=1.e-3, atol=1.e-3)
    assert np.allclose(mscp_x.asnumpy(), scp_x, rtol=1.e-3, atol=1.e-3)
