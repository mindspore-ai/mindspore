# Copyright 2026 Huawei Technologies Co., Ltd
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
"""Sparse storage conversions and matrix arithmetic use real CPU kernels."""
import numpy as np
import pytest

import mindspore as ms
from mindspore.ops.function import sparse_func as sparse


@pytest.fixture(scope='module', autouse=True)
def cpu_context():
    """Keep sparse kernels on CPU and restore the surrounding test context."""
    previous = {key: ms.get_context(key) for key in ('mode', 'device_target')}
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target='CPU')
    yield
    ms.set_context(**previous)


def csr(index_dtype=ms.int32, value_dtype=ms.float32, second=False):
    """Two hand-specified asymmetric matrices make transpose errors visible."""
    if second:
        return ms.CSRTensor(ms.Tensor([0, 1, 3], index_dtype), ms.Tensor([1, 0, 1], index_dtype),
                            ms.Tensor([4, 5, 6], value_dtype), (2, 2))
    return ms.CSRTensor(ms.Tensor([0, 2, 3], index_dtype), ms.Tensor([0, 1, 1], index_dtype),
                        ms.Tensor([1, 2, 3], value_dtype), (2, 2))


@pytest.fixture
def akg_cpu_backend():
    """Probe the required compiled AKG capability; unrelated runtime errors still fail."""
    try:
        sparse.csr_mv(csr(), ms.Tensor([[2.], [4.]], ms.float32)).asnumpy()
    except RuntimeError as error:
        if 'found LLVM 12.0.1 not installed' in str(error):
            pytest.skip('Installed CPU runtime was compiled without USE_LLVM; CSR AKG kernels unavailable')
        raise


@pytest.fixture
def sparse_gpu_backend():
    """SparseMatrixSparseMatMul is GPU-only, independent of available Ascend devices."""
    if not ms.hal.is_available('GPU'):
        pytest.skip('SparseMatrixSparseMatMul requires an available GPU backend')
    previous = ms.get_context('device_target')
    ms.set_context(device_target='GPU')
    yield
    ms.set_context(device_target=previous)


def dense_values(value):
    """Inspect storage independently of the conversion API under test."""
    result = np.zeros(value.shape, np.float32)
    rows, indices, values = value.indptr.asnumpy(), value.indices.asnumpy(), value.values.asnumpy()
    for row in range(value.shape[0]):
        result[row, indices[rows[row]:rows[row + 1]]] = values[rows[row]:rows[row + 1]]
    return result


@pytest.mark.parametrize('index_dtype', [ms.int32, ms.int64])
def test_csr_dense_and_coo_conversions_preserve_coordinates(index_dtype):
    """Both supported sparse index widths must retain values and coordinate ordering."""
    value = csr(index_dtype)
    np.testing.assert_array_equal(sparse.csr_to_dense(value).asnumpy(), [[1, 2], [0, 3]])
    coo = sparse.csr_to_coo(value)
    np.testing.assert_array_equal(coo.indices.asnumpy(), [[0, 0], [0, 1], [1, 1]])
    np.testing.assert_array_equal(coo.values.asnumpy(), [1, 2, 3])
    assert coo.shape == (2, 2)


@pytest.mark.parametrize('kind', ['csr', 'coo'])
def test_dense_to_sparse_preserves_only_nonzero_entries(kind):
    """Sparse conversion must retain only stored nonzero values and the dense shape."""
    result = getattr(sparse, f'dense_to_sparse_{kind}')(ms.Tensor([[1., 2.], [0., 3.]], ms.float32))
    assert result.shape == (2, 2)
    np.testing.assert_array_equal(result.values.asnumpy(), [1, 2, 3])
    if kind == 'csr':
        np.testing.assert_array_equal(result.indptr.asnumpy(), [0, 2, 3])
        np.testing.assert_array_equal(result.indices.asnumpy(), [0, 1, 1])
    else:
        np.testing.assert_array_equal(result.indices.asnumpy(), [[0, 0], [0, 1], [1, 1]])


@pytest.mark.parametrize('name', ['csr_to_dense', 'csr_to_coo', 'dense_to_sparse_csr', 'dense_to_sparse_coo'])
def test_sparse_storage_conversions_reject_wrong_container(name):
    """Conversion APIs must identify invalid storage objects before kernel dispatch."""
    with pytest.raises(TypeError, match='must be'):
        getattr(sparse, name)([1, 2])


@pytest.mark.parametrize('name', ['dense_to_sparse_csr', 'dense_to_sparse_coo'])
def test_dense_to_sparse_rejects_rank_above_two(name):
    """The two-dimensional sparse format cannot silently flatten higher ranks."""
    with pytest.raises(ValueError, match='2-D'):
        getattr(sparse, name)(ms.Tensor(np.ones((1, 2, 2), np.float32)))


@pytest.mark.parametrize('name', ['csr_to_dense', 'csr_to_coo'])
def test_csr_conversions_reject_dense_value_dimensions(name):
    """A CSR tensor with trailing dense dimensions is not a plain sparse matrix."""
    value = ms.CSRTensor(ms.Tensor([0, 1, 2], ms.int32), ms.Tensor([0, 1], ms.int32),
                         ms.Tensor([[1., 2.], [3., 4.]], ms.float32), (2, 2, 2))
    with pytest.raises(ValueError, match='2-D'):
        getattr(sparse, name)(value)


@pytest.mark.parametrize('operation,expected', [('csr_mul', [2, 4, 6]), ('csr_div', [0.5, 1, 1.5])])
@pytest.mark.parametrize('as_tensor', [False, True])
def test_csr_scalar_arithmetic_preserves_stored_value_order(operation, expected, as_tensor):
    """Scalar arithmetic returns each stored value without changing sparse coordinates."""
    operand = ms.Tensor([[2.]], ms.float32) if as_tensor else 2
    result = getattr(sparse, operation)(csr(), operand)
    np.testing.assert_array_equal(result.asnumpy(), expected)


@pytest.mark.parametrize('operation,expected', [('csr_mul', [[2, 8], [0, 12]]),
                                               ('csr_div', [[0.5, 0.5], [0, 0.75]])])
def test_csr_dense_arithmetic_uses_values_at_sparse_positions(operation, expected, akg_cpu_backend):
    """Dense broadcasting must use the correct column for every stored element."""
    result = getattr(sparse, operation)(csr(), ms.Tensor([[2., 4.]], ms.float32))
    np.testing.assert_array_equal(dense_values(result), expected)


@pytest.mark.parametrize('trans_a,trans_b', [(False, False), (True, False), (False, True)])
def test_sparse_matrix_product_transpose_flags(trans_a, trans_b, sparse_gpu_backend):
    """Sparse matrix multiplication must apply the requested transpose to each operand."""
    left, right = np.array([[1, 2], [0, 3]], np.float32), np.array([[0, 4], [5, 6]], np.float32)
    result = sparse.csr_mm(csr(), csr(second=True), trans_a=trans_a, trans_b=trans_b)
    expected = (left.T if trans_a else left) @ (right.T if trans_b else right)
    np.testing.assert_allclose(dense_values(result), expected, rtol=1e-6, atol=1e-6)


def test_sparse_matrix_vector_product(akg_cpu_backend):
    """Every row uses the matching dense vector coordinates."""
    result = sparse.csr_mv(csr(), ms.Tensor([[2.], [4.]], ms.float32))
    np.testing.assert_array_equal(result.asnumpy(), [[10], [12]])


def test_sparse_scaled_addition():
    """The two coefficients must apply to the correct sparse operand."""
    result = sparse.csr_add(csr(), csr(second=True), ms.Tensor(2., ms.float32), ms.Tensor(-1., ms.float32))
    np.testing.assert_array_equal(dense_values(result), [[2, 0], [-5, 0]])


@pytest.mark.parametrize('case', ['a', 'b', 'alpha', 'beta'])
def test_sparse_addition_requires_sparse_operands_and_tensor_coefficients(case):
    """Each sparse-add input's storage contract must be checked independently."""
    args = dict(a=csr(), b=csr(), alpha=ms.Tensor(1., ms.float32), beta=ms.Tensor(1., ms.float32))
    args[case] = 1
    with pytest.raises(TypeError, match='must be'):
        sparse.csr_add(**args)


@pytest.mark.parametrize('invalid_left', [False, True])
def test_sparse_matmul_rejects_dense_operands(invalid_left):
    """The sparse-only product must not accept a dense operand by accident."""
    args = [csr(), csr()]
    args[0 if invalid_left else 1] = ms.Tensor([[1., 0.], [0., 1.]])
    with pytest.raises(TypeError, match='must be type of CSRTensor'):
        sparse.csr_mm(*args)


def test_sparse_softmax_normalizes_each_stored_row():
    """Only stored logits participate in each sparse row's softmax denominator."""
    result = sparse.csr_softmax(csr(), ms.float32)
    probabilities = np.exp([1., 2.]) / np.exp([1., 2.]).sum()
    np.testing.assert_allclose(result.values.asnumpy(), [probabilities[0], probabilities[1], 1.], rtol=1e-6)
    np.testing.assert_array_equal(result.indptr.asnumpy(), [0, 2, 3])


def test_sparse_softmax_rejects_dense_logits():
    """Dense logits cannot be interpreted as a CSR storage object."""
    with pytest.raises(TypeError, match='CSRTensor'):
        sparse.csr_softmax(ms.Tensor([1., 2.]), ms.float32)
