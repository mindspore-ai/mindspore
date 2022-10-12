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
from functools import reduce
import pytest
import numpy as np
import scipy as scp
from scipy.linalg import solve_triangular, eig, eigvals

from mindspore import Tensor, context, nn
from mindspore.common import dtype as mstype
from mindspore.ops.operations.math_ops import Cholesky
from mindspore.scipy.ops import Eigh, Eig, SolveTriangular
from mindspore.scipy.utils import _nd_transpose
from tests.st.scipy_st.utils import create_sym_pos_matrix, create_random_rank_matrix, compare_eigen_decomposition

np.random.seed(0)


class SolveTriangularNet(nn.Cell):
    def __init__(self, lower: bool = False, unit_diagonal: bool = False, trans: str = 'N'):
        super(SolveTriangularNet, self).__init__()
        self.solve = SolveTriangular(lower, unit_diagonal, trans)

    def construct(self, a, b):
        return self.solve(a, b)


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
    cholesky_net = Cholesky()
    output = cholesky_net(tensor_a)
    assert np.allclose(expect, output.asnumpy())


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('shape', [(3, 4, 4), (3, 5, 5), (2, 3, 5, 5)])
@pytest.mark.parametrize('lower', [True, False])
@pytest.mark.parametrize('data_type', [np.float32, np.float64])
def test_batch_cholesky(shape, lower: bool, data_type):
    """
    Feature: ALL To ALL
    Description: test cases for cholesky decomposition test cases for A[N,N]x = b[N,1]
    Expectation: the result match to scipy
    """
    b_s_l = list()
    b_s_a = list()
    tmp = np.zeros(shape[:-2])
    inner_row = shape[-2]
    inner_col = shape[-1]
    for _, _ in np.ndenumerate(tmp):
        a = create_sym_pos_matrix((inner_row, inner_col), data_type)
        s_l = scp.linalg.cholesky(a, lower)
        b_s_l.append(s_l)
        b_s_a.append(a)
    tensor_b_a = Tensor(np.array(b_s_a))
    b_m_l = Cholesky()(tensor_b_a)
    if not lower:
        b_m_l = _nd_transpose(b_m_l)
    b_s_l = np.asarray(b_s_l).reshape(b_m_l.shape)
    rtol = 1.e-3
    atol = 1.e-3
    if data_type == np.float64:
        rtol = 1.e-5
        atol = 1.e-8
    assert np.allclose(b_m_l.asnumpy(), b_s_l, rtol=rtol, atol=atol)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('shape', [(6, 6), (10, 10)])
@pytest.mark.parametrize('data_type, rtol, atol', [(np.float32, 1e-3, 1e-4), (np.float64, 1e-5, 1e-8),
                                                   (np.complex64, 1e-3, 1e-4), (np.complex128, 1e-5, 1e-8)])
def test_eig(shape, data_type, rtol, atol):
    """
    Feature: ALL To ALL
    Description: test cases for Eig operator
    Expectation: the result match eigenvalue definition and scipy eig
    """
    context.set_context(mode=context.GRAPH_MODE)
    a = create_random_rank_matrix(shape, data_type)
    tensor_a = Tensor(a)

    # Check Eig with eigenvalue definition
    msp_w, msp_v = Eig(True)(tensor_a)
    w, v = msp_w.asnumpy(), msp_v.asnumpy()
    assert np.allclose(a @ v - v @ np.diag(w), np.zeros_like(a), rtol, atol)

    # Check Eig with scipy eig
    mw, mv = w, v
    sw, sv = eig(a)
    compare_eigen_decomposition((mw, mv), (sw, sv), True, rtol, atol)

    # Eig only calculate eigenvalues when compute_v is False
    mw, _ = Eig(False)(tensor_a)
    mw = mw.asnumpy()
    sw = eigvals(a)
    compare_eigen_decomposition((mw,), (sw,), False, rtol, atol)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('shape', [(2, 4, 4)])
@pytest.mark.parametrize('data_type, rtol, atol', [(np.float32, 1e-3, 1e-4), (np.float64, 1e-5, 1e-8),
                                                   (np.complex64, 1e-3, 1e-4), (np.complex128, 1e-5, 1e-8)])
def test_batch_eig(shape, data_type, rtol, atol):
    """
    Feature: ALL To ALL
    Description: test batch cases for Eig operator
    Expectation: the result match eigenvalue definition
    """
    context.set_context(mode=context.GRAPH_MODE)
    a = create_random_rank_matrix(shape, data_type)
    tensor_a = Tensor(a)

    # Check Eig with eigenvalue definition
    msp_w, msp_v = Eig(True)(tensor_a)
    w, v = msp_w.asnumpy(), msp_v.asnumpy()
    batch_enum = np.empty(shape=shape[:-2])
    for batch_index, _ in np.ndenumerate(batch_enum):
        batch_a = a[batch_index]
        batch_w = w[batch_index]
        batch_v = v[batch_index]
        assert np.allclose(batch_a @ batch_v - batch_v @ np.diag(batch_w), np.zeros_like(batch_a), rtol, atol)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('n', [4, 6, 9, 10])
def test_eigh(n: int):
    """
    Feature: ALL To ALL
    Description: test cases for eigen decomposition test cases for Ax= lambda * x /( A- lambda * E)X=0
    Expectation: the result match to numpy
    """
    context.set_context(mode=context.GRAPH_MODE)
    rtol = 1e-3
    atol = 1e-4

    a = create_sym_pos_matrix((n, n), np.float32)
    msp_eigh = Eigh(True, True)
    msp_wl, msp_vl = msp_eigh(Tensor(np.array(a).astype(np.float32)))
    msp_eigh = Eigh(True, False)
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
    msp_eigh = Eigh(True, True)
    msp_wl, msp_vl = msp_eigh(Tensor(np.array(a).astype(np.float64)))
    msp_eigh = Eigh(True, False)
    msp_wu, msp_vu = msp_eigh(Tensor(np.array(a).astype(np.float64)))
    sym_al = (np.tril((np.tril(a) - np.tril(a).T)) + np.tril(a).T)
    sym_au = (np.triu((np.triu(a) - np.triu(a).T)) + np.triu(a).T)
    assert np.allclose(sym_al @ msp_vl.asnumpy() - msp_vl.asnumpy() @ np.diag(msp_wl.asnumpy()), np.zeros((n, n)), rtol,
                       atol)
    assert np.allclose(sym_au @ msp_vu.asnumpy() - msp_vu.asnumpy() @ np.diag(msp_wu.asnumpy()), np.zeros((n, n)), rtol,
                       atol)
    # test for real scalar float64 no vector
    msp_eigh = Eigh(False, True)
    msp_wl0 = msp_eigh(Tensor(np.array(a).astype(np.float64)))
    msp_eigh = Eigh(False, False)
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
    msp_eigh = Eigh(True, True)
    msp_wl, msp_vl = msp_eigh(Tensor(np.array(a).astype(np.complex64)))
    msp_eigh = Eigh(True, False)
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
    msp_eigh = Eigh(True, True)
    msp_wl, msp_vl = msp_eigh(Tensor(np.array(a).astype(np.complex128)))
    msp_eigh = Eigh(True, False)
    msp_wu, msp_vu = msp_eigh(Tensor(np.array(a).astype(np.complex128)))
    assert np.allclose(sym_al @ msp_vl.asnumpy() - msp_vl.asnumpy() @ np.diag(msp_wl.asnumpy()), np.zeros((n, n)), rtol,
                       atol)
    assert np.allclose(sym_au @ msp_vu.asnumpy() - msp_vu.asnumpy() @ np.diag(msp_wu.asnumpy()), np.zeros((n, n)), rtol,
                       atol)

    # test for real scalar complex128 no vector
    msp_eigh = Eigh(False, True)
    msp_wl0 = msp_eigh(Tensor(np.array(a).astype(np.complex128)))
    msp_eigh = Eigh(False, False)
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

    type_convert = {np.float32: mstype.float32, np.float64: mstype.float64}

    dynamic_net = SolveTriangularNet(lower, unit_diagonal, trans)
    place_holder = Tensor(shape=[None, None], dtype=type_convert.get(dtype))
    dynamic_net.set_inputs(place_holder, place_holder)

    output = dynamic_net(Tensor(a), Tensor(b)).asnumpy()
    np.testing.assert_almost_equal(expect, output, decimal=5)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('n', [10, 20, 15])
@pytest.mark.parametrize('batch', [(3,), (4, 5)])
@pytest.mark.parametrize('trans', ["N", "T", "C"])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('lower', [False, True])
@pytest.mark.parametrize('unit_diagonal', [False, True])
def test_solve_triangular_batched(n: int, batch, dtype, lower: bool, unit_diagonal: bool, trans: str):
    """
    Feature: ALL TO ALL
    Description:  test cases for solve_triangular for batched triangular matrix solver [..., N, N]
    Expectation: the result match scipy solve_triangular result
    """
    rtol, atol = 1.e-5, 1.e-8
    if dtype == np.float32:
        rtol, atol = 1.e-3, 1.e-3

    np.random.seed(0)
    a = create_random_rank_matrix(batch + (n, n), dtype)
    b = create_random_rank_matrix(batch + (n,), dtype)

    # mindspore
    output = SolveTriangular(lower, unit_diagonal, trans)(Tensor(a), Tensor(b)).asnumpy()

    # scipy
    batch_num = reduce(lambda x, y: x * y, batch)
    a_array = a.reshape((batch_num, n, n))
    b_array = b.reshape((batch_num, n))
    expect = np.stack([solve_triangular(a_array[i, :], b_array[i, :], lower=lower,
                                        unit_diagonal=unit_diagonal, trans=trans)
                       for i in range(batch_num)])
    expect = expect.reshape(output.shape)

    assert np.allclose(expect, output, rtol=rtol, atol=atol)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_solve_triangular_error_dims():
    """
    Feature: ALL TO ALL
    Description:  test cases for solve_triangular for batched triangular matrix solver [..., N, N]
    Expectation: solve_triangular raises expectated Exception
    """
    # matrix a is 1D
    a = create_random_rank_matrix((10,), dtype=np.float32)
    b = create_random_rank_matrix((10,), dtype=np.float32)
    with pytest.raises(ValueError):
        SolveTriangular()(Tensor(a), Tensor(b))

    # matrix a is not square matrix
    a = create_random_rank_matrix((4, 5), dtype=np.float32)
    b = create_random_rank_matrix((10,), dtype=np.float32)
    with pytest.raises(ValueError):
        SolveTriangular()(Tensor(a), Tensor(b))

    a = create_random_rank_matrix((3, 5, 4, 5), dtype=np.float32)
    b = create_random_rank_matrix((3, 5, 10,), dtype=np.float32)
    with pytest.raises(ValueError):
        SolveTriangular()(Tensor(a), Tensor(b))


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_solve_triangular_error_dims_mismatched():
    """
    Feature: ALL TO ALL
    Description:  test cases for solve_triangular for batched triangular matrix solver [..., N, N]
    Expectation: solve_triangular raises expectated Exception
    """
    # dimension of a and b is not matched
    a = create_random_rank_matrix((3, 4, 5, 5), dtype=np.float32)
    b = create_random_rank_matrix((5, 10,), dtype=np.float32)
    with pytest.raises(ValueError):
        SolveTriangular()(Tensor(a), Tensor(b))

    # last two dimensions not matched
    a = create_random_rank_matrix((3, 4, 5, 5), dtype=np.float32)
    b = create_random_rank_matrix((5, 10, 4), dtype=np.float32)
    with pytest.raises(ValueError):
        SolveTriangular()(Tensor(a), Tensor(b))

    a = create_random_rank_matrix((3, 4, 5, 5), dtype=np.float32)
    b = create_random_rank_matrix((5, 10, 4, 1), dtype=np.float32)
    with pytest.raises(ValueError):
        SolveTriangular()(Tensor(a), Tensor(b))

    # batch dimensions not matched
    a = create_random_rank_matrix((3, 4, 5, 5), dtype=np.float32)
    b = create_random_rank_matrix((5, 10, 5), dtype=np.float32)
    with pytest.raises(ValueError):
        SolveTriangular()(Tensor(a), Tensor(b))

    a = create_random_rank_matrix((3, 4, 5, 5), dtype=np.float32)
    b = create_random_rank_matrix((5, 10, 5, 1), dtype=np.float32)
    with pytest.raises(ValueError):
        SolveTriangular()(Tensor(a), Tensor(b))
