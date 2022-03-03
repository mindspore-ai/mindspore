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
"""st for scipy.sparse.linalg."""
import pytest
import numpy as onp
import scipy as osp
import scipy.sparse.linalg

import mindspore.nn as nn
import mindspore.scipy as msp
from mindspore import context
from mindspore.common import Tensor, CSRTensor
from tests.st.scipy_st.utils import create_sym_pos_matrix, create_full_rank_matrix, create_sym_pos_sparse_matrix


def _fetch_preconditioner(preconditioner, A):
    """
    Returns one of various preconditioning matrices depending on the identifier
    `preconditioner' and the input matrix A whose inverse it supposedly
    approximates.
    """
    if preconditioner == 'identity':
        M = onp.eye(A.shape[0], dtype=A.dtype)
    elif preconditioner == 'random':
        random_metrix = create_sym_pos_matrix(A.shape, A.dtype)
        M = onp.linalg.inv(random_metrix)
    elif preconditioner == 'exact':
        M = onp.linalg.inv(A)
    else:
        M = None
    return M


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('dtype, tol', [(onp.float32, 1e-5), (onp.float64, 1e-12)])
@pytest.mark.parametrize('shape', [(4, 4), (7, 7)])
@pytest.mark.parametrize('preconditioner', [None, 'identity', 'exact', 'random'])
@pytest.mark.parametrize('maxiter', [1, 3])
def test_cg_against_scipy(dtype, tol, shape, preconditioner, maxiter):
    """
    Feature: ALL TO ALL
    Description: test cases for cg
    Expectation: the result match scipy
    """
    onp.random.seed(0)
    a = create_sym_pos_matrix(shape, dtype)
    b = onp.random.random(shape[:1]).astype(dtype)
    m = _fetch_preconditioner(preconditioner, a)
    osp_res = scipy.sparse.linalg.cg(a, b, M=m, maxiter=maxiter, atol=tol, tol=tol)

    a = Tensor(a)
    b = Tensor(b)
    m = Tensor(m) if m is not None else m

    # using PYNATIVE MODE
    context.set_context(mode=context.PYNATIVE_MODE)
    msp_res_dyn = msp.sparse.linalg.cg(a, b, M=m, maxiter=maxiter, atol=tol, tol=tol)

    # using GRAPH MODE
    context.set_context(mode=context.GRAPH_MODE)
    msp_res_sta = msp.sparse.linalg.cg(a, b, M=m, maxiter=maxiter, atol=tol, tol=tol)

    kw = {"atol": tol, "rtol": tol}
    onp.testing.assert_allclose(osp_res[0], msp_res_dyn[0].asnumpy(), **kw)
    onp.testing.assert_allclose(osp_res[0], msp_res_sta[0].asnumpy(), **kw)
    assert osp_res[1] == msp_res_dyn[1].asnumpy().item()
    assert osp_res[1] == msp_res_sta[1].asnumpy().item()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('dtype', [onp.float32, onp.float64])
@pytest.mark.parametrize('shape', [(2, 2)])
def test_cg_against_numpy(dtype, shape):
    """
    Feature: ALL TO ALL
    Description: test cases for cg
    Expectation: the result match numpy
    """
    onp.random.seed(0)
    a = create_sym_pos_matrix(shape, dtype)
    b = onp.random.random(shape[:1]).astype(dtype)
    expected = onp.linalg.solve(a, b)

    # using PYNATIVE MODE
    context.set_context(mode=context.PYNATIVE_MODE)
    actual_dyn, _ = msp.sparse.linalg.cg(Tensor(a), Tensor(b))

    # using GRAPH MODE
    context.set_context(mode=context.GRAPH_MODE)
    actual_sta, _ = msp.sparse.linalg.cg(Tensor(a), Tensor(b))

    kw = {"atol": 1e-5, "rtol": 1e-5}
    onp.testing.assert_allclose(expected, actual_dyn.asnumpy(), **kw)
    onp.testing.assert_allclose(expected, actual_sta.asnumpy(), **kw)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('dtype, tol', [(onp.float32, 1e-5), (onp.float64, 1e-12)])
@pytest.mark.parametrize('shape', [(7, 7)])
@pytest.mark.parametrize('preconditioner', [None, 'identity', 'exact', 'random'])
@pytest.mark.parametrize('maxiter', [3])
def test_cg_against_scipy_graph(dtype, tol, shape, preconditioner, maxiter):
    """
    Feature: ALL TO ALL
    Description: test cases for cg within Cell object
    Expectation: the result match scipy
    """
    context.set_context(mode=context.GRAPH_MODE)

    class TestNet(nn.Cell):
        def construct(self, a, b, m, maxiter, tol):
            return msp.sparse.linalg.cg(a, b, M=m, maxiter=maxiter, atol=tol, tol=tol)

    onp.random.seed(0)
    a = create_sym_pos_matrix(shape, dtype)
    b = onp.random.random(shape[:1]).astype(dtype)
    m = _fetch_preconditioner(preconditioner, a)
    osp_res = scipy.sparse.linalg.cg(a, b, M=m, maxiter=maxiter, atol=tol, tol=tol)

    a = Tensor(a)
    b = Tensor(b)
    m = Tensor(m) if m is not None else m
    msp_res = TestNet()(a, b, m, maxiter, tol)

    kw = {"atol": tol, "rtol": tol}
    onp.testing.assert_allclose(osp_res[0], msp_res[0].asnumpy(), **kw)
    assert osp_res[1] == msp_res[1].asnumpy().item()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('dtype, tol', [(onp.float32, 1e-5)])
@pytest.mark.parametrize('shape', [(7, 7)])
@pytest.mark.parametrize('preconditioner', [None, 'identity', 'random'])
@pytest.mark.parametrize('maxiter', [3])
def test_cg_against_scipy_sparse(dtype, tol, shape, preconditioner, maxiter):
    """
    Feature: ALL TO ALL
    Description: test cases of CSRTensor for cg
    Expectation: the result match scipy.
    """
    context.set_context(mode=context.GRAPH_MODE)

    class TestNet(nn.Cell):
        def construct(self, a, b, m, maxiter, tol):
            return msp.sparse.linalg.cg(a, b, M=m, maxiter=maxiter, atol=tol, tol=tol)

    onp.random.seed(0)

    # scipy
    a = create_sym_pos_sparse_matrix(shape, dtype)
    b = onp.random.random(shape[:1]).astype(dtype)
    m = _fetch_preconditioner(preconditioner, a)
    osp_res = scipy.sparse.linalg.cg(a, b, M=m, maxiter=maxiter, atol=tol, tol=tol)

    # mindspore
    a = CSRTensor(Tensor(a.indptr), Tensor(a.indices), Tensor(a.data), shape)
    b = Tensor(b)
    m = Tensor(m) if m is not None else m
    msp_res = TestNet()(a, b, m, maxiter, tol)

    kw = {"atol": tol, "rtol": tol}
    onp.testing.assert_allclose(osp_res[0], msp_res[0].asnumpy(), **kw)
    assert osp_res[1] == msp_res[1].asnumpy().item()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('n', [3, 5, 7])
@pytest.mark.parametrize('dtype,tol', [(onp.float64, 7), (onp.float32, 3)])
@pytest.mark.parametrize('preconditioner', [None, 'identity', 'exact', 'random'])
def test_gmres_incremental_against_scipy(n, tol, dtype, preconditioner):
    """
    Feature: ALL TO ALL
    Description:  test cases for [N x N] X [N X 1]
    Expectation: the result match scipy
    """
    onp.random.seed(0)
    context.set_context(mode=context.PYNATIVE_MODE)
    A = create_full_rank_matrix((n, n), dtype)
    b = onp.random.rand(n).astype(dtype)
    x0 = onp.zeros_like(b).astype(dtype)
    M = _fetch_preconditioner(preconditioner, A)

    scipy_x, _ = osp.sparse.linalg.gmres(A, b, x0, tol=1e-07, atol=0, M=M)
    A = Tensor(A)
    b = Tensor(b)
    x0 = Tensor(x0)
    if M is not None:
        M = Tensor(M)

    gmres_x, _ = msp.sparse.linalg.gmres(A, b, x0, tol=1e-07, atol=0, solve_method='incremental', M=M)
    onp.testing.assert_almost_equal(scipy_x, gmres_x.asnumpy(), decimal=tol)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('n', [3, 5, 7])
@pytest.mark.parametrize('dtype, tol', [(onp.float64, 7), (onp.float32, 3)])
@pytest.mark.parametrize('preconditioner', [None, 'identity', 'exact', 'random'])
def test_gmres_incremental_against_scipy_graph(n, tol, dtype, preconditioner):
    """
    Feature: ALL TO ALL
    Description:  test cases for [N x N] X [N X 1]
    Expectation: the result match scipy
    """
    onp.random.seed(0)
    context.set_context(mode=context.GRAPH_MODE)
    A = create_full_rank_matrix((n, n), dtype)
    b = onp.random.rand(n).astype(dtype)
    x0 = onp.zeros_like(b).astype(dtype)
    M = _fetch_preconditioner(preconditioner, A)

    scipy_x, _ = osp.sparse.linalg.gmres(A, b, x0, tol=1e-07, atol=0, M=M)
    A = Tensor(A)
    b = Tensor(b)
    x0 = Tensor(x0)
    if M is not None:
        M = Tensor(M)

    gmres_x, _ = msp.sparse.linalg.gmres(A, b, x0, tol=1e-07, atol=0, solve_method='incremental', M=M)
    onp.testing.assert_almost_equal(scipy_x, gmres_x.asnumpy(), decimal=tol)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('n', [4, 5, 6])
@pytest.mark.parametrize('dtype, tol', [(onp.float64, 7), (onp.float32, 3)])
@pytest.mark.parametrize('preconditioner', [None, 'identity', 'exact', 'random'])
@pytest.mark.parametrize('maxiter', [1, 2])
def test_pynative_batched_gmres_against_scipy(n, dtype, tol, preconditioner, maxiter):
    """
    Feature: ALL TO ALL
    Description: test cases for gmres
    Expectation: the result match scipy
    """
    onp.random.seed(0)
    context.set_context(mode=context.PYNATIVE_MODE)
    shape = (n, n)
    a = create_full_rank_matrix(shape, dtype)
    b = onp.random.rand(n).astype(dtype=dtype)
    M = _fetch_preconditioner(preconditioner, a)
    tensor_a = Tensor(a)
    tensor_b = Tensor(b)
    M = Tensor(M) if M is not None else M

    osp_x, _ = osp.sparse.linalg.gmres(a, b, maxiter=maxiter, atol=1e-6)

    msp_x, _ = msp.sparse.linalg.gmres(tensor_a, tensor_b, maxiter=maxiter, M=M, atol=1e-6,
                                       solve_method='batched')
    onp.testing.assert_almost_equal(msp_x.asnumpy(), osp_x, decimal=tol)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('n', [5, 6])
@pytest.mark.parametrize('dtype, tol', [(onp.float64, 7), (onp.float32, 3)])
@pytest.mark.parametrize('preconditioner', [None, 'identity', 'exact', 'random'])
@pytest.mark.parametrize('maxiter', [1, 2])
def test_graph_batched_gmres_against_scipy(n, dtype, tol, preconditioner, maxiter):
    """
    Feature: ALL TO ALL
    Description: test cases for gmres
    Expectation: the result match scipy
    """
    onp.random.seed(0)
    context.set_context(mode=context.GRAPH_MODE)
    shape = (n, n)
    a = create_full_rank_matrix(shape, dtype)
    b = onp.random.rand(n).astype(dtype=dtype)
    tensor_a = Tensor(a)
    tensor_b = Tensor(b)
    M = _fetch_preconditioner(preconditioner, a)
    M = Tensor(M) if M is not None else M
    osp_x, _ = osp.sparse.linalg.gmres(a, b, maxiter=maxiter, atol=0.0)
    msp_x, _ = msp.sparse.linalg.gmres(tensor_a, tensor_b, maxiter=maxiter, M=M, atol=0.0, solve_method='batched')
    onp.testing.assert_almost_equal(msp_x.asnumpy(), osp_x, decimal=tol)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('dtype_tol', [(onp.float64, 1e-10)])
@pytest.mark.parametrize('shape', [(4, 4), (7, 7)])
@pytest.mark.parametrize('preconditioner', [None, 'identity', 'exact', 'random'])
@pytest.mark.parametrize('maxiter', [1, 3])
def test_bicgstab_against_scipy(dtype_tol, shape, preconditioner, maxiter):
    """
    Feature: ALL TO ALL
    Description: test cases for bicgstab
    Expectation: the result match scipy
    """
    onp.random.seed(0)
    dtype, tol = dtype_tol
    A = create_full_rank_matrix(shape, dtype)
    b = onp.random.random(shape[:1]).astype(dtype)
    M = _fetch_preconditioner(preconditioner, A)
    osp_res = scipy.sparse.linalg.bicgstab(A, b, M=M, maxiter=maxiter, atol=tol, tol=tol)[0]

    A = Tensor(A)
    b = Tensor(b)
    M = Tensor(M) if M is not None else M

    # using PYNATIVE MODE
    context.set_context(mode=context.PYNATIVE_MODE)
    msp_res_dyn = msp.sparse.linalg.bicgstab(A, b, M=M, maxiter=maxiter, atol=tol, tol=tol)[0]

    # using GRAPH MODE
    context.set_context(mode=context.GRAPH_MODE)
    msp_res_sta = msp.sparse.linalg.bicgstab(A, b, M=M, maxiter=maxiter, atol=tol, tol=tol)[0]

    kw = {"atol": tol, "rtol": tol}
    onp.testing.assert_allclose(osp_res, msp_res_dyn.asnumpy(), **kw)
    onp.testing.assert_allclose(osp_res, msp_res_sta.asnumpy(), **kw)
