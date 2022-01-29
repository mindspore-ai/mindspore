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
"""st for scipy.ops."""
from typing import Generic
import pytest
import numpy as np
import scipy as scp
from scipy.linalg import solve_triangular

from mindspore import Tensor, context
from mindspore.scipy.ops import EighNet, EigNet, Cholesky, SolveTriangular
from tests.st.scipy_st.utils import create_sym_pos_matrix

np.random.seed(0)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('n', [3, 5, 7])
@pytest.mark.parametrize('dtype', [np.float64])
def test_cholesky(n: int, dtype: Generic):
    """
    Feature: ALL TO ALL
    Description:  test cases for cholesky [N,N]
    Expectation: the result match scipy cholesky
    """
    context.set_context(mode=context.GRAPH_MODE)
    a = create_sym_pos_matrix((n, n), dtype)
    tensor_a = Tensor(a)
    expect = scp.linalg.cholesky(a, lower=True)
    cholesky_net = Cholesky(clean=True)
    output = cholesky_net(tensor_a)
    assert np.allclose(expect, output.asnumpy())


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('n', [4, 6, 9, 10])
def test_eig_net(n: int):
    """
    Feature: ALL To ALL
    Description: test cases for eigen decomposition test cases for Ax= lambda * x /( A- lambda * E)X=0
    Expectation: the result match to numpy
    """
    context.set_context(mode=context.GRAPH_MODE)
    # test for real scalar float 32
    rtol = 1e-3
    atol = 1e-4
    msp_eig = EigNet(True)
    a = np.array(np.random.rand(n, n), dtype=np.float32)
    tensor_a = Tensor(np.array(a).astype(np.float32))
    msp_w, msp_v = msp_eig(tensor_a)
    assert np.allclose(a @ msp_v.asnumpy() - msp_v.asnumpy() @ np.diag(msp_w.asnumpy()), np.zeros((n, n)), rtol, atol)

    # test case for real scalar double 64
    a = np.array(np.random.rand(n, n), dtype=np.float64)
    rtol = 1e-5
    atol = 1e-8
    msp_eig = EigNet(True)
    msp_w, msp_v = msp_eig(Tensor(np.array(a).astype(np.float64)))

    # Compare with scipy
    assert np.allclose(a @ msp_v.asnumpy() - msp_v.asnumpy() @ np.diag(msp_w.asnumpy()), np.zeros((n, n)), rtol, atol)

    # test case for complex64
    rtol = 1e-3
    atol = 1e-4
    a = np.array(np.random.rand(n, n), dtype=np.complex64)
    for i in range(0, n):
        for j in range(0, n):
            if i == j:
                a[i][j] = complex(np.random.rand(1, 1), 0)
            else:
                a[i][j] = complex(np.random.rand(1, 1), np.random.rand(1, 1))
    msp_eig = EigNet(True)
    msp_w, msp_v = msp_eig(Tensor(np.array(a).astype(np.complex64)))
    assert np.allclose(a @ msp_v.asnumpy() - msp_v.asnumpy() @ np.diag(msp_w.asnumpy()), np.zeros((n, n)), rtol, atol)

    # test for complex128
    rtol = 1e-5
    atol = 1e-8
    a = np.array(np.random.rand(n, n), dtype=np.complex128)
    for i in range(0, n):
        for j in range(0, n):
            if i == j:
                a[i][j] = complex(np.random.rand(1, 1), 0)
            else:
                a[i][j] = complex(np.random.rand(1, 1), np.random.rand(1, 1))
    msp_eig = EigNet(True)
    msp_w, msp_v = msp_eig(Tensor(np.array(a).astype(np.complex128)))
    # Compare with scipy, scipy passed
    assert np.allclose(a @ msp_v.asnumpy() - msp_v.asnumpy() @ np.diag(msp_w.asnumpy()), np.zeros((n, n)), rtol, atol)
    msp_eig = EigNet(False)
    msp_w0 = msp_eig(Tensor(np.array(a).astype(np.complex128)))
    assert np.allclose(msp_w0.asnumpy() - msp_w.asnumpy(), np.zeros((n, n)), rtol, atol)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('n', [4, 6, 9, 10])
def test_eigh_net(n: int):
    """
    Feature: ALL To ALL
    Description: test cases for eigen decomposition test cases for Ax= lambda * x /( A- lambda * E)X=0
    Expectation: the result match to numpy
    """
    context.set_context(mode=context.GRAPH_MODE)
    rtol = 1e-3
    atol = 1e-4

    a = create_sym_pos_matrix((n, n), np.float32)
    msp_eigh = EighNet(True, True)
    msp_wl, msp_vl = msp_eigh(Tensor(np.array(a).astype(np.float32)))
    msp_eigh = EighNet(True, False)
    msp_wu, msp_vu = msp_eigh(Tensor(np.array(a).astype(np.float32)))
    sym_al = (np.tril((np.tril(a) - np.tril(a).T)) + np.tril(a).T)
    sym_au = (np.triu((np.triu(a) - np.triu(a).T)) + np.triu(a).T)
    assert np.allclose(sym_al @ msp_vl.asnumpy() - msp_vl.asnumpy() @ np.diag(msp_wl.asnumpy()), np.zeros((n, n)), rtol,
                       atol)
    assert np.allclose(sym_au @ msp_vu.asnumpy() - msp_vu.asnumpy() @ np.diag(msp_wu.asnumpy()), np.zeros((n, n)), rtol,
                       atol)

    # test case for real scalar double 64
    a = np.random.rand(n, n)
    rtol = 1e-5
    atol = 1e-8
    msp_eigh = EighNet(True, True)
    msp_wl, msp_vl = msp_eigh(Tensor(np.array(a).astype(np.float64)))
    msp_eigh = EighNet(True, False)
    msp_wu, msp_vu = msp_eigh(Tensor(np.array(a).astype(np.float64)))
    sym_al = (np.tril((np.tril(a) - np.tril(a).T)) + np.tril(a).T)
    sym_au = (np.triu((np.triu(a) - np.triu(a).T)) + np.triu(a).T)
    assert np.allclose(sym_al @ msp_vl.asnumpy() - msp_vl.asnumpy() @ np.diag(msp_wl.asnumpy()), np.zeros((n, n)), rtol,
                       atol)
    assert np.allclose(sym_au @ msp_vu.asnumpy() - msp_vu.asnumpy() @ np.diag(msp_wu.asnumpy()), np.zeros((n, n)), rtol,
                       atol)
    # test for real scalar float64 no vector
    msp_eigh = EighNet(False, True)
    msp_wl0 = msp_eigh(Tensor(np.array(a).astype(np.float64)))
    msp_eigh = EighNet(False, False)
    msp_wu0 = msp_eigh(Tensor(np.array(a).astype(np.float64)))
    assert np.allclose(msp_wl.asnumpy() - msp_wl0.asnumpy(), np.zeros((n, n)), rtol, atol)
    assert np.allclose(msp_wu.asnumpy() - msp_wu0.asnumpy(), np.zeros((n, n)), rtol, atol)

    # test case for complex64
    rtol = 1e-3
    atol = 1e-4
    a = np.array(np.random.rand(n, n), dtype=np.complex64)
    for i in range(0, n):
        for j in range(0, n):
            if i == j:
                a[i][j] = complex(np.random.rand(1, 1), 0)
            else:
                a[i][j] = complex(np.random.rand(1, 1), np.random.rand(1, 1))
    sym_al = (np.tril((np.tril(a) - np.tril(a).T)) + np.tril(a).conj().T)
    sym_au = (np.triu((np.triu(a) - np.triu(a).T)) + np.triu(a).conj().T)
    msp_eigh = EighNet(True, True)
    msp_wl, msp_vl = msp_eigh(Tensor(np.array(a).astype(np.complex64)))
    msp_eigh = EighNet(True, False)
    msp_wu, msp_vu = msp_eigh(Tensor(np.array(a).astype(np.complex64)))
    assert np.allclose(sym_al @ msp_vl.asnumpy() - msp_vl.asnumpy() @ np.diag(msp_wl.asnumpy()), np.zeros((n, n)), rtol,
                       atol)
    assert np.allclose(sym_au @ msp_vu.asnumpy() - msp_vu.asnumpy() @ np.diag(msp_wu.asnumpy()), np.zeros((n, n)), rtol,
                       atol)

    # test for complex128
    rtol = 1e-5
    atol = 1e-8
    a = np.array(np.random.rand(n, n), dtype=np.complex128)
    for i in range(0, n):
        for j in range(0, n):
            if i == j:
                a[i][j] = complex(np.random.rand(1, 1), 0)
            else:
                a[i][j] = complex(np.random.rand(1, 1), np.random.rand(1, 1))
    sym_al = (np.tril((np.tril(a) - np.tril(a).T)) + np.tril(a).conj().T)
    sym_au = (np.triu((np.triu(a) - np.triu(a).T)) + np.triu(a).conj().T)
    msp_eigh = EighNet(True, True)
    msp_wl, msp_vl = msp_eigh(Tensor(np.array(a).astype(np.complex128)))
    msp_eigh = EighNet(True, False)
    msp_wu, msp_vu = msp_eigh(Tensor(np.array(a).astype(np.complex128)))
    assert np.allclose(sym_al @ msp_vl.asnumpy() - msp_vl.asnumpy() @ np.diag(msp_wl.asnumpy()), np.zeros((n, n)), rtol,
                       atol)
    assert np.allclose(sym_au @ msp_vu.asnumpy() - msp_vu.asnumpy() @ np.diag(msp_wu.asnumpy()), np.zeros((n, n)), rtol,
                       atol)

    # test for real scalar complex128 no vector
    msp_eigh = EighNet(False, True)
    msp_wl0 = msp_eigh(Tensor(np.array(a).astype(np.complex128)))
    msp_eigh = EighNet(False, False)
    msp_wu0 = msp_eigh(Tensor(np.array(a).astype(np.complex128)))
    assert np.allclose(msp_wl.asnumpy() - msp_wl0.asnumpy(), np.zeros((n, n)), rtol, atol)
    assert np.allclose(msp_wu.asnumpy() - msp_wu0.asnumpy(), np.zeros((n, n)), rtol, atol)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('n', [10, 20])
@pytest.mark.parametrize('trans', ["N", "T", "C"])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('lower', [False, True])
@pytest.mark.parametrize('unit_diagonal', [False])
def test_solve_triangular_2d(n: int, dtype, lower: bool, unit_diagonal: bool, trans: str):
    """
    Feature: ALL TO ALL
    Description:  test cases for [N x N] X [N X 1]
    Expectation: the result match scipy
    """
    context.set_context(mode=context.GRAPH_MODE)
    a = (np.random.random((n, n)) + np.eye(n)).astype(dtype)
    b = np.random.random((n, 1)).astype(dtype)
    expect = solve_triangular(a, b, lower=lower, unit_diagonal=unit_diagonal, trans=trans)
    solve = SolveTriangular(lower, unit_diagonal, trans)
    output = solve(Tensor(a), Tensor(b)).asnumpy()
    np.testing.assert_almost_equal(expect, output, decimal=5)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('n', [10, 20])
@pytest.mark.parametrize('trans', ["N", "T", "C"])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('lower', [False, True])
@pytest.mark.parametrize('unit_diagonal', [False, True])
def test_solve_triangular_1d(n: int, dtype, lower: bool, unit_diagonal: bool, trans: str):
    """
    Feature: ALL TO ALL
    Description: test cases for [N x N] X [N]
    Expectation: the result match scipy
    """
    context.set_context(mode=context.GRAPH_MODE)
    a = (np.random.random((n, n)) + np.eye(n)).astype(dtype)
    b = np.random.random(n).astype(dtype)
    expect = solve_triangular(a, b, lower=lower, unit_diagonal=unit_diagonal, trans=trans)
    solve = SolveTriangular(lower, unit_diagonal, trans)
    output = solve(Tensor(a), Tensor(b)).asnumpy()
    np.testing.assert_almost_equal(expect, output, decimal=5)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('shape', [(4, 5), (10, 20)])
@pytest.mark.parametrize('trans', ["N", "T", "C"])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('lower', [False, True])
@pytest.mark.parametrize('unit_diagonal', [False, True])
def test_solve_triangular_matrix(shape: int, dtype, lower: bool, unit_diagonal: bool, trans: str):
    """
    Feature: ALL TO ALL
    Description: test cases for [N x N] X [N]
    Expectation: the result match scipy
    """
    if trans == 'T':
        n, m = shape
    else:
        m, n = shape
    context.set_context(mode=context.GRAPH_MODE)
    a = (np.random.random((m, m)) + np.eye(m)).astype(dtype)
    b = np.random.random((m, n)).astype(dtype)
    expect = solve_triangular(a, b, lower=lower, unit_diagonal=unit_diagonal, trans=trans)
    solve = SolveTriangular(lower, unit_diagonal, trans)
    output = solve(Tensor(a), Tensor(b)).asnumpy()
    np.testing.assert_almost_equal(expect, output, decimal=5)
