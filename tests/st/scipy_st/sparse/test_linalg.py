import scipy.sparse.linalg
import scipy.linalg

from mindspore import Tensor, context
from mindspore.scipy.sparse import gmres
import numpy as onp
import pytest


onp.random.seed(0)


def gmres_compare_with_scipy(A, b, x):
    gmres_x, _ = gmres(Tensor(A), Tensor(b), Tensor(
        x), tol=1e-07, atol=0, solve_method='incremental')
    scipy_x, _ = scipy.sparse.linalg.gmres(A, b, x, tol=1e-07, atol=0)
    onp.testing.assert_almost_equal(scipy_x, gmres_x.asnumpy(), decimal=5)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('n', [5])
@pytest.mark.parametrize('dtype', [onp.float64])
def test_gmres_incremental_against_scipy_cpu(n, dtype):
    """
    Feature: ALL TO ALL
    Description:  test cases for [N x N] X [N X 1]
    Expectation: the result match scipy
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    # add Identity matrix to make matrix A non-singular
    A = onp.random.rand(n, n).astype(dtype)
    b = onp.random.rand(n).astype(dtype)
    gmres_compare_with_scipy(A, b, onp.zeros_like(b).astype(dtype))


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('n', [5])
@pytest.mark.parametrize('dtype', [onp.float64])
def test_gmres_incremental_against_scipy_cpu_graph(n, dtype):
    """
    Feature: ALL TO ALL
    Description:  test cases for [N x N] X [N X 1]
    Expectation: the result match scipy
    """
    context.set_context(mode=context.GRAPH_MODE)
    # add Identity matrix to make matrix A non-singular
    A = onp.random.rand(n, n).astype(dtype)
    b = onp.random.rand(n).astype(dtype)
    gmres_compare_with_scipy(A, b, onp.zeros_like(b).astype(dtype))


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('n', [5])
@pytest.mark.parametrize('dtype', [onp.float64])
def test_gmres_incremental_against_scipy_gpu(n, dtype):
    """
    Feature: ALL TO ALL
    Description:  test cases for [N x N] X [N X 1]
    Expectation: the result match scipy
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    # add Identity matrix to make matrix A non-singular
    A = onp.random.rand(n, n).astype(dtype)
    b = onp.random.rand(n).astype(dtype)
    gmres_compare_with_scipy(A, b, onp.zeros_like(b).astype(dtype))


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('n', [5])
@pytest.mark.parametrize('dtype', [onp.float64])
def test_gmres_incremental_against_scipy_gpu_graph(n, dtype):
    """
    Feature: ALL TO ALL
    Description:  test cases for [N x N] X [N X 1]
    Expectation: the result match scipy
    """
    context.set_context(mode=context.GRAPH_MODE)
    # add Identity matrix to make matrix A non-singular
    A = onp.random.rand(n, n).astype(dtype)
    b = onp.random.rand(n).astype(dtype)
    gmres_compare_with_scipy(A, b, onp.zeros_like(b).astype(dtype))
