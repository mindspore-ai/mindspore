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
"""st for scipy.linalg."""

from typing import Generic
import pytest
import numpy as onp
import scipy as osp

import mindspore.nn as nn
import mindspore.scipy as msp
from mindspore import context, Tensor
import mindspore.numpy as mnp
from tests.st.scipy_st.utils import match_array, create_full_rank_matrix, create_sym_pos_matrix, \
    create_random_rank_matrix

onp.random.seed(0)
context.set_context(mode=context.PYNATIVE_MODE)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('args', [(), (1,), (7, -1), (3, 4, 5),
                                  (onp.ones((3, 4), dtype=onp.float32), 5, onp.random.randn(5, 2).astype(onp.float32))])
def test_block_diag(args):
    """
    Feature: ALL TO ALL
    Description: test cases for block_diag
    Expectation: the result match scipy
    """
    tensor_args = tuple([Tensor(arg) for arg in args])
    ms_res = msp.linalg.block_diag(*tensor_args)

    scipy_res = osp.linalg.block_diag(*args)
    match_array(ms_res.asnumpy(), scipy_res)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('dtype', [onp.float32, onp.float64])
@pytest.mark.parametrize('shape', [(4, 4), (50, 50), (2, 5, 5)])
def test_inv(dtype, shape):
    """
    Feature: ALL TO ALL
    Description: test cases for inv
    Expectation: the result match numpy
    """
    onp.random.seed(0)
    x = create_full_rank_matrix(shape, dtype)

    ms_res = msp.linalg.inv(Tensor(x))
    scipy_res = onp.linalg.inv(x)
    match_array(ms_res.asnumpy(), scipy_res, error=3)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('n', [4, 5, 6])
@pytest.mark.parametrize('lower', [True, False])
@pytest.mark.parametrize('dtype', [onp.float64])
def test_cholesky(n: int, lower: bool, dtype: Generic):
    """
    Feature: ALL TO ALL
    Description:  test cases for cholesky [N,N]
    Expectation: the result match scipy cholesky
    """
    a = create_sym_pos_matrix((n, n), dtype)
    tensor_a = Tensor(a)
    rtol = 1.e-5
    atol = 1.e-8
    osp_c = osp.linalg.cholesky(a, lower=lower)
    msp_c = msp.linalg.cholesky(tensor_a, lower=lower)
    assert onp.allclose(osp_c, msp_c.asnumpy(), rtol=rtol, atol=atol)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('n', [4, 5, 6])
@pytest.mark.parametrize('lower', [True, False])
@pytest.mark.parametrize('dtype', [onp.float64])
def test_cho_factor(n: int, lower: bool, dtype: Generic):
    """
    Feature: ALL TO ALL
    Description:  test cases for cholesky [N,N]
    Expectation: the result match scipy cholesky
    """
    a = create_sym_pos_matrix((n, n), dtype)
    tensor_a = Tensor(a)
    msp_c, _ = msp.linalg.cho_factor(tensor_a, lower=lower)
    if lower:
        msp_reconstruct_a = mnp.dot(mnp.tril(msp_c), mnp.tril(msp_c).T)
    else:
        msp_reconstruct_a = mnp.dot(mnp.triu(msp_c).T, mnp.triu(msp_c))
    assert onp.allclose(a, msp_reconstruct_a.asnumpy())


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('n', [4, 5, 6])
@pytest.mark.parametrize('lower', [True, False])
@pytest.mark.parametrize('dtype', [onp.float64])
def test_cholesky_solver(n: int, lower: bool, dtype):
    """
    Feature: ALL TO ALL
    Description:  test cases for cholesky  solver [N,N]
    Expectation: the result match scipy cholesky_solve
    """
    a = create_sym_pos_matrix((n, n), dtype)
    b = onp.ones((n, 1), dtype=dtype)
    tensor_a = Tensor(a)
    tensor_b = Tensor(b)
    osp_c, lower = osp.linalg.cho_factor(a, lower=lower)
    msp_c, msp_lower = msp.linalg.cho_factor(tensor_a, lower=lower)
    osp_factor = (osp_c, lower)

    ms_cho_factor = (msp_c, msp_lower)
    osp_x = osp.linalg.cho_solve(osp_factor, b)
    msp_x = msp.linalg.cho_solve(ms_cho_factor, tensor_b)
    # pre tensor_a has been inplace.
    tensor_a = Tensor(a)
    assert onp.allclose(onp.dot(a, osp_x), mnp.dot(tensor_a, msp_x).asnumpy())


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('n', [4, 6, 9, 20])
@pytest.mark.parametrize('dtype',
                         [(onp.int8, "f"), (onp.int16, "f"), (onp.int32, "f"), (onp.int64, "d"), (onp.float32, "f"),
                          (onp.float64, "d")])
def test_eigh(n: int, dtype):
    """
    Feature: ALL TO ALL
    Description:  test cases for eigenvalues/eigenvector for symmetric/Hermitian matrix solver [N,N]
    Expectation: the result match scipy eigenvalues
    """
    # test for real scalar float
    tol = {"f": (1e-3, 1e-4), "d": (1e-5, 1e-8)}
    rtol = tol[dtype[1]][0]
    atol = tol[dtype[1]][1]
    A = create_sym_pos_matrix([n, n], dtype[0])
    msp_wl, msp_vl = msp.linalg.eigh(Tensor(onp.array(A).astype(dtype[0])), lower=True, eigvals_only=False)
    msp_wu, msp_vu = msp.linalg.eigh(Tensor(onp.array(A).astype(dtype[0])), lower=False, eigvals_only=False)
    assert onp.allclose(A @ msp_vl.asnumpy() - msp_vl.asnumpy() @ onp.diag(msp_wl.asnumpy()), onp.zeros((n, n)),
                        rtol,
                        atol)
    assert onp.allclose(A @ msp_vu.asnumpy() - msp_vu.asnumpy() @ onp.diag(msp_wu.asnumpy()), onp.zeros((n, n)),
                        rtol,
                        atol)
    # test for real scalar float no vector
    msp_wl0 = msp.linalg.eigh(Tensor(onp.array(A).astype(dtype[0])), lower=True, eigvals_only=True)
    msp_wu0 = msp.linalg.eigh(Tensor(onp.array(A).astype(dtype[0])), lower=False, eigvals_only=True)
    assert onp.allclose(msp_wl.asnumpy() - msp_wl0.asnumpy(), onp.zeros((n, n)), rtol, atol)
    assert onp.allclose(msp_wu.asnumpy() - msp_wu0.asnumpy(), onp.zeros((n, n)), rtol, atol)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('n', [4, 6, 9, 20])
@pytest.mark.parametrize('dtype', [(onp.complex64, "f"), (onp.complex128, "d")])
def test_eigh_complex(n: int, dtype):
    """
    Feature: ALL TO ALL
    Description:  test cases for eigenvalues/eigenvector for symmetric/Hermitian matrix solver [N,N]
    Expectation: the result match scipy eigenvalues
    """
    # test case for complex
    tol = {"f": (1e-3, 1e-4), "d": (1e-5, 1e-8)}
    rtol = tol[dtype[1]][0]
    atol = tol[dtype[1]][1]
    A = onp.array(onp.random.rand(n, n), dtype=dtype[0])
    for i in range(0, n):
        for j in range(0, n):
            if i == j:
                A[i][j] = complex(onp.random.rand(1, 1), 0)
            else:
                A[i][j] = complex(onp.random.rand(1, 1), onp.random.rand(1, 1))
    sym_al = (onp.tril((onp.tril(A) - onp.tril(A).T)) + onp.tril(A).conj().T)
    sym_au = (onp.triu((onp.triu(A) - onp.triu(A).T)) + onp.triu(A).conj().T)
    msp_wl, msp_vl = msp.linalg.eigh(Tensor(onp.array(sym_al).astype(dtype[0])), lower=True, eigvals_only=False)
    msp_wu, msp_vu = msp.linalg.eigh(Tensor(onp.array(sym_au).astype(dtype[0])), lower=False, eigvals_only=False)
    assert onp.allclose(sym_al @ msp_vl.asnumpy() - msp_vl.asnumpy() @ onp.diag(msp_wl.asnumpy()),
                        onp.zeros((n, n)), rtol, atol)
    assert onp.allclose(sym_au @ msp_vu.asnumpy() - msp_vu.asnumpy() @ onp.diag(msp_wu.asnumpy()),
                        onp.zeros((n, n)), rtol, atol)

    # test for real scalar complex no vector
    msp_wl0 = msp.linalg.eigh(Tensor(onp.array(sym_al).astype(dtype[0])), lower=True, eigvals_only=True)
    msp_wu0 = msp.linalg.eigh(Tensor(onp.array(sym_au).astype(dtype[0])), lower=False, eigvals_only=True)
    assert onp.allclose(msp_wl.asnumpy() - msp_wl0.asnumpy(), onp.zeros((n, n)), rtol, atol)
    assert onp.allclose(msp_wu.asnumpy() - msp_wu0.asnumpy(), onp.zeros((n, n)), rtol, atol)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('shape', [(4, 4), (4, 5), (5, 10), (20, 20)])
@pytest.mark.parametrize('dtype', [onp.float32, onp.float64])
def test_lu(shape: (int, int), dtype):
    """
    Feature: ALL To ALL
    Description: test cases for lu decomposition test cases for A[N,N]x = b[N,1]
    Expectation: the result match to scipy
    """
    a = create_random_rank_matrix(shape, dtype)
    s_p, s_l, s_u = osp.linalg.lu(a)
    tensor_a = Tensor(a)
    m_p, m_l, m_u = msp.linalg.lu(tensor_a)
    rtol = 1.e-5
    atol = 1.e-5
    assert onp.allclose(m_p.asnumpy(), s_p, rtol=rtol, atol=atol)
    assert onp.allclose(m_l.asnumpy(), s_l, rtol=rtol, atol=atol)
    assert onp.allclose(m_u.asnumpy(), s_u, rtol=rtol, atol=atol)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('shape', [(3, 4, 4), (3, 4, 5)])
@pytest.mark.parametrize('dtype', [onp.float32, onp.float64])
def test_batch_lu(shape: (int, int, int), dtype):
    """
    Feature: ALL To ALL
    Description: test cases for lu decomposition test cases for A[N,N]x = b[N,1]
    Expectation: the result match to scipy
    """
    b_a = create_random_rank_matrix(shape, dtype)
    b_s_p = list()
    b_s_l = list()
    b_s_u = list()
    for a in b_a:
        s_p, s_l, s_u = osp.linalg.lu(a)
        b_s_p.append(s_p)
        b_s_l.append(s_l)
        b_s_u.append(s_u)
    tensor_b_a = Tensor(onp.array(b_a))
    b_m_p, b_m_l, b_m_u = msp.linalg.lu(tensor_b_a)
    rtol = 1.e-5
    atol = 1.e-5
    assert onp.allclose(b_m_p.asnumpy(), b_s_p, rtol=rtol, atol=atol)
    assert onp.allclose(b_m_l.asnumpy(), b_s_l, rtol=rtol, atol=atol)
    assert onp.allclose(b_m_u.asnumpy(), b_s_u, rtol=rtol, atol=atol)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('n', [4, 5, 10, 20])
@pytest.mark.parametrize('dtype', [onp.float32, onp.float64])
def test_lu_factor(n: int, dtype):
    """
    Feature: ALL To ALL
    Description: test cases for lu decomposition test cases for A[N,N]x = b[N,1]
    Expectation: the result match to scipy
    """
    a = create_full_rank_matrix((n, n), dtype)
    s_lu, s_pivots = osp.linalg.lu_factor(a)
    tensor_a = Tensor(a)
    m_lu, m_pivots = msp.linalg.lu_factor(tensor_a)
    rtol = 1.e-5
    atol = 1.e-5
    assert onp.allclose(m_lu.asnumpy(), s_lu, rtol=rtol, atol=atol)
    assert onp.allclose(m_pivots.asnumpy(), s_pivots, rtol=rtol, atol=atol)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('n', [4, 5, 10, 20])
@pytest.mark.parametrize('dtype', [onp.float32, onp.float64])
def test_lu_solve(n: int, dtype):
    """
    Feature: ALL To ALL
    Description: test cases for lu_solve test cases for A[N,N]x = b[N,1]
    Expectation: the result match to scipy
    """
    a = create_full_rank_matrix((n, n), dtype)
    b = onp.random.random((n, 1)).astype(dtype)
    s_lu, s_piv = osp.linalg.lu_factor(a)

    tensor_a = Tensor(a)
    tensor_b = Tensor(b)

    m_lu, m_piv = msp.linalg.lu_factor(tensor_a)

    lu_factor_x = (s_lu, s_piv)
    msp_lu_factor = (m_lu, m_piv)

    osp_x = osp.linalg.lu_solve(lu_factor_x, b)
    msp_x = msp.linalg.lu_solve(msp_lu_factor, tensor_b)
    real_b = mnp.dot(tensor_a, msp_x)
    expected_b = onp.dot(a, osp_x)
    rtol = 1.e-3
    atol = 1.e-3
    assert onp.allclose(real_b.asnumpy(), expected_b, rtol=rtol, atol=atol)
    assert onp.allclose(msp_x.asnumpy(), osp_x, rtol=rtol, atol=atol)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('args', [(), (1,), (7, -1), (3, 4, 5),
                                  (onp.ones((3, 4), dtype=onp.float32), 5, onp.random.randn(5, 2).astype(onp.float32))])
def test_block_diag_graph(args):
    """
    Feature: ALL TO ALL
    Description: test cases for block_diag in graph mode
    Expectation: the result match scipy
    """
    context.set_context(mode=context.GRAPH_MODE)

    class TestNet(nn.Cell):
        def construct(self, inputs):
            return msp.linalg.block_diag(*inputs)

    tensor_args = tuple([Tensor(arg) for arg in args])
    ms_res = TestNet()(tensor_args)

    scipy_res = osp.linalg.block_diag(*args)
    match_array(ms_res.asnumpy(), scipy_res)
