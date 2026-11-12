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
"""All CSR/COO unary wrappers preserve sparse structure and transform values."""
import numpy as np
import pytest

import mindspore as ms
from mindspore.ops.function import sparse_unary_func as sparse
from tests.mark_utils import arg_mark


CASES = [
    ('cos', np.cos, [-0.25, 0.5]), ('tan', np.tan, [-0.25, 0.5]),
    ('exp', np.exp, [-0.25, 0.5]), ('inv', np.reciprocal, [-0.25, 0.5]),
    ('relu', lambda x: np.maximum(x, 0), [-2, 3]),
    ('expm1', np.expm1, [-0.25, 0.5]),
    ('isfinite', np.isfinite, [np.inf, 0.5]),
    ('asin', np.arcsin, [-0.25, 0.5]), ('sqrt', np.sqrt, [0.25, 2]),
    ('log', np.log, [0.25, 2]), ('isnan', np.isnan, [np.nan, 0.5]),
    ('acos', np.arccos, [-0.25, 0.5]), ('floor', np.floor, [-1.25, 2.5]),
    ('atan', np.arctan, [-0.25, 0.5]), ('square', np.square, [-0.25, 0.5]),
    ('relu6', lambda x: np.minimum(np.maximum(x, 0), 6), [-2, 7]),
    ('sinh', np.sinh, [-0.25, 0.5]), ('ceil', np.ceil, [-1.25, 2.5]),
    ('cosh', np.cosh, [-0.25, 0.5]),
    ('softsign', lambda x: x / (1 + np.abs(x)), [-0.25, 0.5]),
    ('log1p', np.log1p, [-0.25, 0.5]), ('round', np.round, [-1.25, 2.75]),
    ('tanh', np.tanh, [-0.25, 0.5]), ('asinh', np.arcsinh, [-0.25, 0.5]),
    ('neg', np.negative, [-0.25, 0.5]), ('acosh', np.arccosh, [1, 2]),
    ('isinf', np.isinf, [np.inf, 0.5]), ('atanh', np.arctanh, [-0.25, 0.5]),
    ('sigmoid', lambda x: 1 / (1 + np.exp(-x)), [-0.25, 0.5]),
    ('abs', np.abs, [-0.25, 0.5]), ('sin', np.sin, [-0.25, 0.5]),
]


@pytest.fixture(scope='module', autouse=True)
def cpu_context():
    """Use real CPU value kernels and restore the surrounding test context."""
    previous = {key: ms.get_context(key) for key in ('mode', 'device_target')}
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target='CPU')
    yield
    ms.set_context(**previous)


def make_sparse(kind, values):
    """Keep the structure asymmetric so swaps of rows/columns are observable."""
    tensor = ms.Tensor(np.array(values, np.float32))
    if kind == 'csr':
        return ms.CSRTensor(ms.Tensor([0, 1, 2, 2], ms.int32), ms.Tensor([3, 0], ms.int32), tensor, (3, 4))
    return ms.COOTensor(ms.Tensor([[0, 3], [1, 0]], ms.int32), tensor, (3, 4))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('kind', ['csr', 'coo'])
@pytest.mark.parametrize('operation,oracle,values', CASES, ids=[case[0] for case in CASES])
def test_sparse_unary_values_and_structure(kind, operation, oracle, values):
    """
    Feature: CSR and COO unary operations
    Description: Every unary API must retain coordinates, shape and the correct value dtype.
    Expectation: Every CSR/COO wrapper matches its NumPy value oracle while preserving exact container type,
        coordinates and shape.
    """
    value = make_sparse(kind, values)
    result = getattr(sparse, f'{kind}_{operation}')(value)
    expected = oracle(np.array(values, np.float32))
    assert type(result) is type(value)
    assert result.shape == (3, 4)
    assert result.values.dtype == (ms.bool_ if expected.dtype == np.bool_ else ms.float32)
    np.testing.assert_allclose(result.values.asnumpy(), expected, rtol=2e-6, atol=2e-6, equal_nan=True)
    np.testing.assert_array_equal(result.indices.asnumpy(), [3, 0] if kind == 'csr' else [[0, 3], [1, 0]])
    if kind == 'csr':
        np.testing.assert_array_equal(result.indptr.asnumpy(), [0, 1, 2, 2])


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('kind', ['csr', 'coo'])
@pytest.mark.parametrize('operation', [case[0] for case in CASES])
def test_sparse_unary_rejects_dense_input(kind, operation):
    """
    Feature: CSR and COO unary operations
    Description: Dense values must not be silently interpreted as sparse storage.
    Expectation: Every unary wrapper rejects a dense Tensor with the CSR/COO storage diagnostic.
    """
    with pytest.raises(TypeError, match='Expects (CSR|COO)Tensor'):
        getattr(sparse, f'{kind}_{operation}')(ms.Tensor([1, 2], ms.float32))
