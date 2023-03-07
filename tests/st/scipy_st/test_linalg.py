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
from mindspore.scipy.linalg import det, solve_triangular
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


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('n', [10, 20, 52])
@pytest.mark.parametrize('trans', ["N", "T", "C"])
@pytest.mark.parametrize('dtype', [onp.float32, onp.float64, onp.int32, onp.int64])
@pytest.mark.parametrize('lower', [False, True])
@pytest.mark.parametrize('unit_diagonal', [False, True])
def test_solve_triangular(n: int, dtype, lower: bool, unit_diagonal: bool, trans: str):
    """
    Feature: ALL TO ALL
    Description:  test cases for solve_triangular for batched triangular matrix solver [..., N, N]
    Expectation: the result match scipy solve_triangular result
    """
    rtol, atol = 1.e-5, 1.e-8
    if dtype == onp.float32:
        rtol, atol = 1.e-3, 1.e-3

    onp.random.seed(0)
    a = create_random_rank_matrix((n, n), dtype)
    b = create_random_rank_matrix((n,), dtype)

    output = solve_triangular(Tensor(a), Tensor(b), trans, lower, unit_diagonal).asnumpy()
    expect = osp.linalg.solve_triangular(a, b, lower=lower, unit_diagonal=unit_diagonal,
                                         trans=trans)

    assert onp.allclose(expect, output, rtol=rtol, atol=atol)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('n', [3, 4, 6])
@pytest.mark.parametrize('dtype', [onp.float32, onp.float64, onp.int32, onp.int64])
def test_solve_triangular_error_dims(n: int, dtype):
    """
    Feature: ALL TO ALL
    Description:  test cases for solve_triangular for triangular matrix solver [N,N]
    Expectation: solve_triangular raises expectated Exception
    """
    a = create_random_rank_matrix((10,) * n, dtype)
    b = create_random_rank_matrix(10, dtype)
    with pytest.raises(ValueError):
        solve_triangular(Tensor(a), Tensor(b))

    a = create_random_rank_matrix((n, n + 1), dtype)
    b = create_random_rank_matrix((10,), dtype)
    with pytest.raises(ValueError):
        solve_triangular(Tensor(a), Tensor(b))

    a = create_random_rank_matrix((10, 10), dtype)
    b = create_random_rank_matrix((11,) * n, dtype)
    with pytest.raises(ValueError):
        solve_triangular(Tensor(a), Tensor(b))

    a = create_random_rank_matrix((10, 10), dtype)
    b = create_random_rank_matrix((n,), dtype)
    with pytest.raises(ValueError):
        solve_triangular(Tensor(a), Tensor(b))


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_solve_triangular_error_tensor_dtype():
    """
    Feature: ALL TO ALL
    Description:  test cases for solve_triangular for batched triangular matrix solver [..., N, N]
    Expectation: solve_triangular raises expectated Exception
    """
    a = create_random_rank_matrix((10, 10), onp.float16)
    b = create_random_rank_matrix((10,), onp.float16)
    with pytest.raises(TypeError):
        solve_triangular(Tensor(a), Tensor(b))

    a = create_random_rank_matrix((10, 10), onp.float32)
    b = create_random_rank_matrix((10,), onp.float16)
    with pytest.raises(TypeError):
        solve_triangular(Tensor(a), Tensor(b))

    a = create_random_rank_matrix((10, 10), onp.float32)
    b = create_random_rank_matrix((10,), onp.float64)
    with pytest.raises(TypeError):
        solve_triangular(Tensor(a), Tensor(b))


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('dtype', [onp.float32, onp.float64, onp.int32, onp.int64])
@pytest.mark.parametrize('argname', ['lower', 'overwrite_b', 'check_finite'])
@pytest.mark.parametrize('wrong_argvalue', [5.0, None, 'test'])
def test_solve_triangular_error_type(dtype, argname, wrong_argvalue):
    """
    Feature: ALL TO ALL
    Description:  test cases for solve_triangular for batched triangular matrix solver [..., N, N]
    Expectation: solve_triangular raises expectated Exception
    """
    a = create_random_rank_matrix((10, 10), dtype)
    b = create_random_rank_matrix((10,), dtype)

    kwargs = {argname: wrong_argvalue}
    with pytest.raises(TypeError):
        solve_triangular(Tensor(a), Tensor(b), **kwargs)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('dtype', [onp.float32, onp.float64, onp.int32, onp.int64])
@pytest.mark.parametrize('wrong_argvalue', [5.0, None])
def test_solve_triangular_error_type_trans(dtype, wrong_argvalue):
    """
    Feature: ALL TO ALL
    Description:  test cases for solve_triangular for batched triangular matrix solver [..., N, N]
    Expectation: solve_triangular raises expectated Exception
    """
    a = create_random_rank_matrix((10, 10), dtype)
    b = create_random_rank_matrix((10,), dtype)

    with pytest.raises(TypeError):
        solve_triangular(Tensor(a), Tensor(b), trans=wrong_argvalue)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('dtype', [onp.float32, onp.float64, onp.int32, onp.int64])
@pytest.mark.parametrize('wrong_argvalue', ['D', 6])
def test_solve_triangular_error_value_trans(dtype, wrong_argvalue):
    """
    Feature: ALL TO ALL
    Description:  test cases for solve_triangular for batched triangular matrix solver [..., N, N]
    Expectation: solve_triangular raises expectated Exception
    """
    a = create_random_rank_matrix((10, 10), dtype)
    b = create_random_rank_matrix((10,), dtype)

    with pytest.raises(ValueError):
        solve_triangular(Tensor(a), Tensor(b), trans=wrong_argvalue)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_solve_triangular_error_tensor_type():
    """
    Feature: ALL TO ALL
    Description:  test cases for solve_triangular for batched triangular matrix solver [..., N, N]
    Expectation: solve_triangular raises expectated Exception
    """
    a = 'test'
    b = create_random_rank_matrix((10,), onp.float32)
    with pytest.raises(TypeError):
        solve_triangular(a, Tensor(b))

    a = [1, 2, 3]
    b = create_random_rank_matrix((10,), onp.float32)
    with pytest.raises(TypeError):
        solve_triangular(a, Tensor(b))

    a = (1, 2, 3)
    b = create_random_rank_matrix((10,), onp.float32)
    with pytest.raises(TypeError):
        solve_triangular(a, Tensor(b))


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('data_type', [onp.float32, onp.float64])
@pytest.mark.parametrize('shape', [(4, 4), (50, 50)])
def test_inv(data_type, shape):
    """
    Feature: ALL TO ALL
    Description: test cases for inv
    Expectation: the result match numpy
    """
    onp.random.seed(0)
    x = create_full_rank_matrix(shape, data_type)

    # onp.linalg.inv calls sched_yeild() but still holds GIL.
    # A deadlock can occur if executed after msp.linalg.inv.
    scipy_res = onp.linalg.inv(x)
    ms_res = msp.linalg.inv(Tensor(x))
    match_array(ms_res.asnumpy(), scipy_res, error=3)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('n', [4, 5, 6])
@pytest.mark.parametrize('lower', [True, False])
@pytest.mark.parametrize('data_type', [onp.float32, onp.float64])
def test_cholesky(n: int, lower: bool, data_type: Generic):
    """
    Feature: ALL TO ALL
    Description:  test cases for cholesky [N,N]
    Expectation: the result match scipy cholesky
    """
    a = create_sym_pos_matrix((n, n), data_type)
    tensor_a = Tensor(a)
    rtol = 1.e-3
    atol = 1.e-3
    if data_type == onp.float64:
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
@pytest.mark.parametrize('data_type', [onp.float32, onp.float64])
def test_cho_factor(n: int, lower: bool, data_type: Generic):
    """
    Feature: ALL TO ALL
    Description:  test cases for cho_factor [N,N]
    Expectation: the result match scipy cholesky
    """
    a = create_sym_pos_matrix((n, n), data_type)
    tensor_a = Tensor(a)
    msp_c, _ = msp.linalg.cho_factor(tensor_a, lower=lower)
    osp_c, _ = osp.linalg.cho_factor(a, lower=lower)
    if lower:
        msp_c = mnp.tril(msp_c)
        osp_c = onp.tril(osp_c)
    else:
        msp_c = mnp.triu(msp_c)
        osp_c = onp.triu(osp_c)

    rtol = 1.e-3
    atol = 1.e-3
    if data_type == onp.float64:
        rtol = 1.e-5
        atol = 1.e-8
    assert onp.allclose(osp_c, msp_c.asnumpy(), rtol=rtol, atol=atol)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('n', [4, 5, 6])
@pytest.mark.parametrize('lower', [True, False])
@pytest.mark.parametrize('data_type', [onp.float64])
def test_cholesky_solve(n: int, lower: bool, data_type):
    """
    Feature: ALL TO ALL
    Description:  test cases for cholesky  solver [N,N]
    Expectation: the result match scipy cholesky_solve
    """
    a = create_sym_pos_matrix((n, n), data_type)
    b = onp.ones((n, 1), dtype=data_type)
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
@pytest.mark.parametrize('lower', [True, False])
@pytest.mark.parametrize('data_type, rtol, atol',
                         [(onp.int32, 1e-5, 1e-8), (onp.int64, 1e-5, 1e-8), (onp.float32, 1e-3, 1e-4),
                          (onp.float64, 1e-5, 1e-8)])
def test_eigh(n: int, lower, data_type, rtol, atol):
    """
    Feature: ALL TO ALL
    Description:  test cases for eigenvalues/eigenvector for symmetric/Hermitian matrix solver [N,N]
    Expectation: the result match scipy eigenvalues
    """
    onp.random.seed(0)
    a = create_sym_pos_matrix([n, n], data_type)
    a_tensor = Tensor(onp.array(a))

    # test for real scalar float
    w, v = msp.linalg.eigh(a_tensor, lower=lower, eigvals_only=False)
    lhs = a @ v.asnumpy()
    rhs = v.asnumpy() @ onp.diag(w.asnumpy())
    assert onp.allclose(lhs, rhs, rtol, atol)
    # test for real scalar float no vector
    w0 = msp.linalg.eigh(a_tensor, lower=lower, eigvals_only=True)
    assert onp.allclose(w.asnumpy(), w0.asnumpy(), rtol, atol)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('n', [4, 6, 9, 20])
@pytest.mark.parametrize('data_type', [(onp.complex64, "f"), (onp.complex128, "d")])
def test_eigh_complex(n: int, data_type):
    """
    Feature: ALL TO ALL
    Description:  test cases for eigenvalues/eigenvector for symmetric/Hermitian matrix solver [N,N]
    Expectation: the result match scipy eigenvalues
    """
    # test case for complex
    tol = {"f": (1e-3, 1e-4), "d": (1e-5, 1e-8)}
    rtol = tol[data_type[1]][0]
    atol = tol[data_type[1]][1]
    A = onp.array(onp.random.rand(n, n), dtype=data_type[0])
    for i in range(0, n):
        for j in range(0, n):
            if i == j:
                A[i][j] = complex(onp.random.rand(1, 1), 0)
            else:
                A[i][j] = complex(onp.random.rand(1, 1), onp.random.rand(1, 1))
    sym_al = (onp.tril((onp.tril(A) - onp.tril(A).T)) + onp.tril(A).conj().T)
    sym_au = (onp.triu((onp.triu(A) - onp.triu(A).T)) + onp.triu(A).conj().T)
    msp_wl, msp_vl = msp.linalg.eigh(Tensor(onp.array(sym_al).astype(data_type[0])), lower=True, eigvals_only=False)
    msp_wu, msp_vu = msp.linalg.eigh(Tensor(onp.array(sym_au).astype(data_type[0])), lower=False, eigvals_only=False)
    assert onp.allclose(sym_al @ msp_vl.asnumpy() - msp_vl.asnumpy() @ onp.diag(msp_wl.asnumpy()),
                        onp.zeros((n, n)), rtol, atol)
    assert onp.allclose(sym_au @ msp_vu.asnumpy() - msp_vu.asnumpy() @ onp.diag(msp_wu.asnumpy()),
                        onp.zeros((n, n)), rtol, atol)

    # test for real scalar complex no vector
    msp_wl0 = msp.linalg.eigh(Tensor(onp.array(sym_al).astype(data_type[0])), lower=True, eigvals_only=True)
    msp_wu0 = msp.linalg.eigh(Tensor(onp.array(sym_au).astype(data_type[0])), lower=False, eigvals_only=True)
    assert onp.allclose(msp_wl.asnumpy() - msp_wl0.asnumpy(), onp.zeros((n, n)), rtol, atol)
    assert onp.allclose(msp_wu.asnumpy() - msp_wu0.asnumpy(), onp.zeros((n, n)), rtol, atol)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('dtype', [onp.float32, onp.float64, onp.int32, onp.int64])
@pytest.mark.parametrize('argname', ['lower', 'eigvals_only', 'overwrite_a', 'overwrite_b', 'turbo', 'check_finite'])
@pytest.mark.parametrize('wrong_argvalue', [5.0, None])
def test_eigh_error_type(dtype, argname, wrong_argvalue):
    """
    Feature: ALL TO ALL
    Description:  test cases for solve_triangular for batched triangular matrix solver [..., N, N]
    Expectation: eigh raises expectated Exception
    """
    a = create_random_rank_matrix((10, 10), dtype)
    b = create_random_rank_matrix((10,), dtype)

    kwargs = {argname: wrong_argvalue}
    with pytest.raises(TypeError):
        msp.linalg.eigh(Tensor(a), Tensor(b), **kwargs)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('dtype', [onp.float16, onp.int8, onp.int16])
def test_eigh_error_tensor_dtype(dtype):
    """
    Feature: ALL TO ALL
    Description:  test cases for solve_triangular for batched triangular matrix solver [..., N, N]
    Expectation: eigh raises expectated Exception
    """
    a = create_random_rank_matrix((10, 10), dtype)
    with pytest.raises(TypeError):
        msp.linalg.eigh(Tensor(a))


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('n', [1, 3, 4, 6])
@pytest.mark.parametrize('dtype', [onp.float32, onp.float64, onp.int32, onp.int64])
def test_eigh_error_dims(n: int, dtype):
    """
    Feature: ALL TO ALL
    Description:  test cases for solve_triangular for batched triangular matrix solver [..., N, N]
    Expectation: eigh raises expectated Exception
    """
    a = create_random_rank_matrix((10,) * n, dtype)
    with pytest.raises(ValueError):
        msp.linalg.eigh(Tensor(a))

    a = create_random_rank_matrix((n, n + 1), dtype)
    with pytest.raises(ValueError):
        msp.linalg.eigh(Tensor(a))


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_eigh_error_not_implemented():
    """
    Feature: ALL TO ALL
    Description:  test cases for solve_triangular for batched triangular matrix solver [..., N, N]
    Expectation: eigh raises expectated Exception
    """
    a = create_random_rank_matrix((10, 10), onp.float32)
    b = create_random_rank_matrix((10, 10), onp.float32)
    with pytest.raises(ValueError):
        msp.linalg.eigh(Tensor(a), Tensor(b))

    with pytest.raises(ValueError):
        msp.linalg.eigh(Tensor(a), 42)

    with pytest.raises(ValueError):
        msp.linalg.eigh(Tensor(a), eigvals=42)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('shape', [(4, 4), (4, 5), (5, 10), (20, 20)])
@pytest.mark.parametrize('data_type', [onp.float32, onp.float64])
def test_lu(shape: (int, int), data_type):
    """
    Feature: ALL To ALL
    Description: test cases for lu decomposition test cases for A[N,N]x = b[N,1]
    Expectation: the result match to scipy
    """
    a = create_random_rank_matrix(shape, data_type)
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
@pytest.mark.parametrize('n', [4, 5, 10, 20])
@pytest.mark.parametrize('data_type', [onp.float32, onp.float64])
def test_lu_factor(n: int, data_type):
    """
    Feature: ALL To ALL
    Description: test cases for lu decomposition test cases for A[N,N]x = b[N,1]
    Expectation: the result match to scipy
    """
    a = create_full_rank_matrix((n, n), data_type)
    s_lu, s_pivots = osp.linalg.lu_factor(a)
    tensor_a = Tensor(a)
    m_lu, m_pivots = msp.linalg.lu_factor(tensor_a)
    rtol = 1.e-5
    atol = 1.e-5
    assert onp.allclose(m_lu.asnumpy(), s_lu, rtol=rtol, atol=atol)
    assert onp.allclose(m_pivots.asnumpy(), s_pivots, rtol=rtol, atol=atol)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('n', [4, 5, 10, 20])
@pytest.mark.parametrize('data_type', [onp.float32, onp.float64])
def test_lu_solve(n: int, data_type):
    """
    Feature: ALL To ALL
    Description: test cases for lu_solve test cases for A[N,N]x = b[N,1]
    Expectation: the result match to scipy
    """
    a = create_full_rank_matrix((n, n), data_type)
    b = onp.random.random((n, 1)).astype(data_type)
    rtol = 1.e-3
    atol = 1.e-3
    if data_type == onp.float64:
        rtol = 1.e-5
        atol = 1.e-8

    s_lu, s_piv = osp.linalg.lu_factor(a)
    m_lu, m_piv = msp.linalg.lu_factor(Tensor(a))
    assert onp.allclose(m_lu.asnumpy(), s_lu, rtol=rtol, atol=atol)
    assert onp.allclose(m_piv.asnumpy(), s_piv, rtol=rtol, atol=atol)

    osp_lu_factor = (s_lu, s_piv)
    msp_lu_factor = (m_lu, m_piv)
    osp_x = osp.linalg.lu_solve(osp_lu_factor, b)
    msp_x = msp.linalg.lu_solve(msp_lu_factor, Tensor(b))
    assert onp.allclose(msp_x.asnumpy(), osp_x, rtol=rtol, atol=atol)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('shape', [(3, 3), (5, 5), (10, 10), (20, 20)])
@pytest.mark.parametrize('dtype', [onp.float32, onp.float64])
def test_det(shape, dtype):
    """
    Feature: ALL To ALL
    Description: test cases for det
    Expectation: the result match to scipy
    """
    a = onp.random.random(shape).astype(dtype)
    sp_det = osp.linalg.det(a)
    tensor_a = Tensor(a)
    ms_det = msp.linalg.det(tensor_a)
    rtol = 1.e-5
    atol = 1.e-5
    assert onp.allclose(ms_det.asnumpy(), sp_det, rtol=rtol, atol=atol)


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


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('shape', [(3, 3), (5, 5), (10, 10), (20, 20)])
@pytest.mark.parametrize('dtype', [onp.float32, onp.float64])
def test_det_graph(shape, dtype):
    """
    Feature: ALL To ALL
    Description: test cases for det in graph mode
    Expectation: the result match to scipy
    """
    context.set_context(mode=context.GRAPH_MODE)

    class TestNet(nn.Cell):
        def construct(self, a):
            return det(a)

    a = onp.random.random(shape).astype(dtype)
    sp_det = osp.linalg.det(a)
    tensor_a = Tensor(a)
    ms_det = TestNet()(tensor_a)
    rtol = 1.e-5
    atol = 1.e-5
    assert onp.allclose(ms_det.asnumpy(), sp_det, rtol=rtol, atol=atol)
