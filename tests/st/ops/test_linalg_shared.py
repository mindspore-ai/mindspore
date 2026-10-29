# Copyright 2021-2024 Huawei Technologies Co., Ltd
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
"""Shared linear algebra primitives, independent of the removed SciPy API package."""
from typing import List
import numpy as np
import pytest
import scipy as scp
from scipy.linalg import eigvals
from mindspore import Tensor, context, nn, ops
from mindspore.ops.operations.math_ops import Cholesky
from mindspore.ops.operations.linalg_ops import Eigh
from tests.mark_utils import arg_mark

def create_random_rank_matrix(shape, dtype):
    if dtype in [np.complex64, np.complex128]:
        random_data = np.random.uniform(low=-1.0, high=1.0, size=shape).astype(dtype)
        random_data += 1j * np.random.uniform(low=-1.0, high=1.0, size=shape).astype(dtype)
    elif dtype in [np.int32, np.int64]:
        random_data = np.random.randint(10000, size=shape).astype(dtype)
    else:
        random_data = np.random.random(shape).astype(dtype)
    return random_data


def create_sym_pos_matrix(shape, dtype):
    if len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError(
            'Symmetric positive definite matrix must be a square matrix, but has shape: ', shape)

    n = shape[-1]
    count = 0
    while count < 100:
        x = np.random.random(shape).astype(dtype)
        a = (np.matmul(x, x.T) + np.eye(n)).astype(dtype)
        count += 1
        if np.min(eigvals(a)) > 0:
            return a
    raise ValueError('Symmetric positive definite matrix create failed')


def gradient_check(x, net, epsilon=1e-3, symmetric=False, enumerate_fn=np.ndenumerate):
    # Some utils
    def _tensor_to_numpy(arg: List[Tensor]) -> List[np.ndarray]:
        return [_arg.asnumpy() for _arg in arg]

    def _numpy_to_tensor(arg: List[np.ndarray]) -> List[Tensor]:
        return [Tensor(_arg) for _arg in arg]

    def _add_value(arg: List[np.ndarray], outer, inner, value):
        arg[outer][inner] += value
        return arg

    def _flatten(arg: List[np.ndarray]) -> np.ndarray:
        arg = [_arg.reshape((-1,)) for _arg in arg]
        return np.concatenate(arg)

    if isinstance(x, Tensor):
        x = [x]

    # Using automatic differentiation to calculate gradient
    grad_net = ops.GradOperation(get_all=True)(net)
    x_grad = grad_net(*x)
    x_grad = _tensor_to_numpy(x_grad)

    # Using the definition of a derivative to calculate gradient
    x = _tensor_to_numpy(x)
    x_grad_approx = [np.zeros_like(_x) for _x in x_grad]
    for outer, _x in enumerate(x):
        for inner, _ in enumerate_fn(_x):
            x = _add_value(x, outer, inner, epsilon)
            y_plus = net(*_numpy_to_tensor(x)).asnumpy()

            x = _add_value(x, outer, inner, -2 * epsilon)
            y_minus = net(*_numpy_to_tensor(x)).asnumpy()

            y_grad = (y_plus - y_minus) / (2 * epsilon)
            x = _add_value(x, outer, inner, epsilon)
            x_grad_approx = _add_value(x_grad_approx, outer, inner, y_grad)

    if symmetric:
        x_grad_approx = [0.5 * (_x_grad + _x_grad.conj().T) for _x_grad in x_grad_approx]
    x_grad = _flatten(x_grad)
    x_grad_approx = _flatten(x_grad_approx)
    numerator = np.linalg.norm(x_grad - x_grad_approx)
    denominator = np.linalg.norm(x_grad) + np.linalg.norm(x_grad_approx)
    difference = numerator / denominator
    return difference



class PrimitiveNet(nn.Cell):
    """Run a shared primitive through Cell compilation as well as eager execution."""

    def __init__(self, primitive):
        super().__init__()
        self.primitive = primitive

    def construct(self, x):
        return self.primitive(x)


@pytest.fixture(params=[context.GRAPH_MODE, context.PYNATIVE_MODE], ids=['graph', 'pynative'])
def execution_mode(request):
    previous = context.get_context('mode')
    context.set_context(mode=request.param)
    yield request.param
    context.set_context(mode=previous)


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux', 'cpu_macos'], level_mark='level1', card_mark='onecard')
@pytest.mark.usefixtures('execution_mode')
@pytest.mark.parametrize('n', [3, 4, 5, 6, 7])
@pytest.mark.parametrize('lower', [True, False])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_cholesky(n, lower, dtype):
    """
    Feature: Cholesky primitive.
    Description: Preserve single-matrix sizes, both precisions and triangular orientations.
    Expectation: The factor matches SciPy and reconstructs the input matrix.
    """
    rng = np.random.default_rng(0)
    x = rng.standard_normal((n, n)).astype(dtype)
    a = (x @ x.T + np.eye(n)).astype(dtype)
    expect = scp.linalg.cholesky(a, lower=lower)
    output = PrimitiveNet(Cholesky(upper=not lower))(Tensor(a)).asnumpy()
    rtol, atol = (1e-3, 1e-3) if dtype == np.float32 else (1e-5, 1e-8)
    np.testing.assert_allclose(output, expect, rtol=rtol, atol=atol)
    reconstructed = output @ output.T if lower else output.T @ output
    np.testing.assert_allclose(reconstructed, a, rtol=rtol, atol=atol)


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux', 'cpu_macos'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('shape', [(3, 4, 4), (3, 5, 5), (2, 3, 5, 5)])
@pytest.mark.parametrize('lower', [True, False])
@pytest.mark.parametrize('data_type', [np.float32, np.float64])
def test_batch_cholesky(shape, lower: bool, data_type):
    """
    Feature: ALL To ALL
    Description: test cases for cholesky decomposition test cases for A[N,N]x = b[N,1]
    Expectation: the result match to scipy
    """
    b_s_l = []
    b_s_a = []
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
        b_m_l = b_m_l.swapaxes(-1, -2)
    b_s_l = np.asarray(b_s_l).reshape(b_m_l.shape)
    rtol = 1.e-3
    atol = 1.e-3
    if data_type == np.float64:
        rtol = 1.e-5
        atol = 1.e-8
    assert np.allclose(b_m_l.asnumpy(), b_s_l, rtol=rtol, atol=atol)


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux', 'cpu_macos'], level_mark='level1', card_mark='onecard')
@pytest.mark.usefixtures('execution_mode')
@pytest.mark.parametrize('n', [4, 6, 9, 10, 20])
@pytest.mark.parametrize('compute_eigenvectors', [True, False])
@pytest.mark.parametrize('lower', [True, False])
@pytest.mark.parametrize('dtype, rtol, atol', [
    (np.float32, 1e-3, 1e-4), (np.float64, 1e-5, 1e-8),
    (np.complex64, 1e-3, 1e-4), (np.complex128, 1e-5, 1e-8)])
def test_eigh(n, compute_eigenvectors, lower, dtype, rtol, atol):
    """
    Feature: Eigh primitive.
    Description: Cross both triangle/vector options with real/complex dtypes and matrix sizes.
    Expectation: Eigenvalues match NumPy; eigenvectors are orthonormal and satisfy A v = v w.
    """
    rng = np.random.default_rng(0)
    a = rng.standard_normal((n, n)).astype(dtype)
    if np.issubdtype(dtype, np.complexfloating):
        a += 1j * rng.standard_normal((n, n)).astype(dtype)
        a[np.diag_indices(n)] = a.diagonal().real
    # Keep the unused triangle different to detect ignoring the lower attribute.
    triangle = np.tril(a, -1) if lower else np.triu(a, 1)
    hermitian = triangle + triangle.conj().T + np.diag(a.diagonal())
    expected = np.linalg.eigvalsh(a, UPLO='L' if lower else 'U')
    result = PrimitiveNet(Eigh(compute_eigenvectors, lower))(Tensor(a))
    if compute_eigenvectors:
        values, vectors = (item.asnumpy() for item in result)
        assert vectors.shape == (n, n)
        np.testing.assert_allclose(hermitian @ vectors, vectors * values, rtol=rtol, atol=atol)
        np.testing.assert_allclose(vectors.conj().T @ vectors, np.eye(n), rtol=rtol, atol=atol)
    else:
        values = result.asnumpy()
    assert values.shape == (n,)
    # The complex primitive does not guarantee NumPy's ascending eigenvalue order.
    # Compare the spectrum independently of order; keep eigenvector pairing above.
    np.testing.assert_allclose(np.sort(values), expected, rtol=rtol, atol=atol)


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux', 'cpu_macos'], level_mark='level1', card_mark='onecard')
@pytest.mark.usefixtures('execution_mode')
@pytest.mark.parametrize('dtype', [np.float16, np.int8, np.int16])
@pytest.mark.parametrize('compute_eigenvectors', [True, False])
def test_eigh_error_tensor_dtype(dtype, compute_eigenvectors):
    """
    Feature: Eigh primitive type inference.
    Description: Reject unsupported input dtypes without the removed SciPy wrapper.
    Expectation: Eigh raises TypeError.
    """
    net = PrimitiveNet(Eigh(compute_eigenvectors))
    with pytest.raises(TypeError, match='Eigh'):
        result = net(Tensor(np.eye(3, dtype=dtype)))
        if compute_eigenvectors:
            result = result[0]
        result.asnumpy()


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux', 'cpu_macos'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('compute_eigenvectors', [True, False])
@pytest.mark.parametrize('lower', [True, False])
@pytest.mark.parametrize('shape', [(8, 8)])
@pytest.mark.parametrize('data_type', [(np.float32, 1e-3, 1e-3), (np.float64, 1e-4, 1e-7)])
def test_eigh_grad(compute_eigenvectors, lower, shape, data_type):
    """
    Feature: ALL TO ALL
    Description: test cases for grad implementation of Eigh operator
    Expectation: the result match gradient checking.
    """
    np.random.seed(0)
    dtype, epsilon, error = data_type
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.mean = ops.ReduceMean()
            self.sum = ops.ReduceSum()
            self.compute_eigenvectors = compute_eigenvectors
            self.lower = lower
            self.eigh = Eigh(compute_eigenvectors, lower)

        def construct(self, a):
            res = None
            if self.compute_eigenvectors:
                w, v = self.eigh(a)
                res = self.sum(w) + self.mean(v)
            else:
                w = self.eigh(a)
                res = self.mean(w)
            return res

    net = Net()
    a = create_random_rank_matrix(shape, dtype)
    context.set_context(mode=context.GRAPH_MODE)
    assert gradient_check(Tensor(a), net, epsilon) < error
    context.set_context(mode=context.PYNATIVE_MODE)
    assert gradient_check(Tensor(a), net, epsilon) < error
