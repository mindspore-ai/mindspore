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
"""smoke tests for SparseMatrixAdd"""

import pytest
import numpy as np

from mindspore import Tensor, CSRTensor, context
from mindspore.ops.function import csr_add
from mindspore.ops.operations.sparse_ops import SparseMatrixAdd
from mindspore.common import dtype as mstype
from mindspore.ops.primitive import constexpr
from mindspore.common.api import _pynative_executor


@constexpr
def _make_tensor(a, dtype=mstype.int64):
    """Converts the input to tensor."""
    if not isinstance(a, (list, tuple, int, float, bool)):
        raise TypeError("input data must be `int`, `float`, `bool`, `list` or `tuple`")
    if isinstance(a, (list, tuple)):
        a = np.asarray(a)
        if a.dtype is np.dtype('object'):
            raise ValueError('Input array must have the same size across all dimensions.')
    return Tensor(a, dtype)


def create_csr_tensor():
    a_indptr = Tensor([0, 1, 2], dtype=mstype.int32)
    a_indices = Tensor([0, 1], dtype=mstype.int32)
    a_values = Tensor([1, 2], dtype=mstype.float32)
    shape = (2, 6)
    b_indptr = Tensor([0, 1, 2], dtype=mstype.int32)
    b_indices = Tensor([0, 1], dtype=mstype.int32)
    b_values = Tensor([1, 2], dtype=mstype.float32)
    csra = CSRTensor(a_indptr, a_indices, a_values, shape)
    csrb = CSRTensor(b_indptr, b_indices, b_values, shape)
    return csra, csrb


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_function_csr_add():
    """
    Feature: Test function csr_add.
    Description: Test CSRTensor matrix add.
    Expectation: Success.
    """
    context.set_context(mode=context.GRAPH_MODE)
    alpha = Tensor(1, mstype.float32)
    beta = Tensor(1, mstype.float32)
    csra, csrb = create_csr_tensor()
    c = csr_add(csra, csrb, alpha, beta)
    c_indptr_expected = [0, 1, 2]
    c_indices_expected = [0, 1]
    c_values_excpected = [2.0, 4.0]
    assert np.allclose(c.indptr.asnumpy(), c_indptr_expected)
    assert np.allclose(c.indices.asnumpy(), c_indices_expected)
    assert np.allclose(c.values.asnumpy(), c_values_excpected)

    beta = Tensor(-1, mstype.float32)
    c = csr_add(csra, csrb, alpha, beta)
    c_indptr_expected = [0, 1, 2]
    c_indices_expected = [0, 1]
    c_values_excpected = [0.0, 0.0]
    assert np.allclose(c.indptr.asnumpy(), c_indptr_expected)
    assert np.allclose(c.indices.asnumpy(), c_indices_expected)
    assert np.allclose(c.values.asnumpy(), c_values_excpected)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_graph_csr_add():
    """
    Feature: Test ops SparseMatrixAdd.
    Description: Test CSRTensor matrix add.
    Expectation: Success.
    """
    context.set_context(mode=context.GRAPH_MODE)
    alpha = Tensor(1, mstype.float32)
    beta = Tensor(1, mstype.float32)
    csra, csrb = create_csr_tensor()
    csra_pointers = _make_tensor([0, csra.values.shape[0]], mstype.int32)
    csrb_pointers = _make_tensor([0, csrb.values.shape[0]], mstype.int32)
    csra_shape = _make_tensor(csra.shape, mstype.int32)
    csrb_shape = _make_tensor(csrb.shape, mstype.int32)
    csr_add_op = SparseMatrixAdd()
    c = csr_add_op(csra_shape, csra_pointers, csra.indptr, csra.indices, csra.values,
                   csrb_shape, csrb_pointers, csrb.indptr, csrb.indices, csrb.values, alpha, beta)
    c_indptr_expected = [0, 1, 2]
    c_indices_expected = [0, 1]
    c_values_excpected = [2.0, 4.0]
    assert np.allclose(c[2].asnumpy(), c_indptr_expected)
    assert np.allclose(c[3].asnumpy(), c_indices_expected)
    assert np.allclose(c[4].asnumpy(), c_values_excpected)

    beta = Tensor(-1, mstype.float32)
    c = csr_add_op(csra_shape, csra_pointers, csra.indptr, csra.indices, csra.values,
                   csrb_shape, csrb_pointers, csrb.indptr, csrb.indices, csrb.values, alpha, beta)
    c_indptr_expected = [0, 1, 2]
    c_indices_expected = [0, 1]
    c_values_excpected = [0.0, 0.0]
    assert np.allclose(c[2].asnumpy(), c_indptr_expected)
    assert np.allclose(c[3].asnumpy(), c_indices_expected)
    assert np.allclose(c[4].asnumpy(), c_values_excpected)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_pynative_csr_add():
    """
    Feature: Test Tensor csr_add.
    Description: Test CSRTensor matrix add.
    Expectation: Success.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    alpha = Tensor(1, mstype.float32)
    beta = Tensor(1, mstype.float32)
    csra, csrb = create_csr_tensor()
    c = csra.add(csrb, alpha, beta)
    c_indptr_expected = [0, 1, 2]
    c_indices_expected = [0, 1]
    c_values_excpected = [2.0, 4.0]
    assert np.allclose(c.indptr.asnumpy(), c_indptr_expected)
    assert np.allclose(c.indices.asnumpy(), c_indices_expected)
    assert np.allclose(c.values.asnumpy(), c_values_excpected)

    beta = Tensor(-1, mstype.float32)
    c = csra.add(csrb, alpha, beta)
    c_indptr_expected = [0, 1, 2]
    c_indices_expected = [0, 1]
    c_values_excpected = [0.0, 0.0]
    assert np.allclose(c.indptr.asnumpy(), c_indptr_expected)
    assert np.allclose(c.indices.asnumpy(), c_indices_expected)
    assert np.allclose(c.values.asnumpy(), c_values_excpected)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_tensor_csr_add():
    """
    Feature: Test Tensor csr_add.
    Description: Test CSRTensor matrix add.
    Expectation: Success.
    """
    context.set_context(mode=context.GRAPH_MODE)
    alpha = Tensor(1, mstype.float32)
    beta = Tensor(1, mstype.float32)
    csra, csrb = create_csr_tensor()
    c = csra.add(csrb, alpha, beta)
    c_indptr_expected = [0, 1, 2]
    c_indices_expected = [0, 1]
    c_values_excpected = [2.0, 4.0]
    assert np.allclose(c.indptr.asnumpy(), c_indptr_expected)
    assert np.allclose(c.indices.asnumpy(), c_indices_expected)
    assert np.allclose(c.values.asnumpy(), c_values_excpected)

    beta = Tensor(-1, mstype.float32)
    c = csra.add(csrb, alpha, beta)
    c_indptr_expected = [0, 1, 2]
    c_indices_expected = [0, 1]
    c_values_excpected = [0.0, 0.0]
    assert np.allclose(c.indptr.asnumpy(), c_indptr_expected)
    assert np.allclose(c.indices.asnumpy(), c_indices_expected)
    assert np.allclose(c.values.asnumpy(), c_values_excpected)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_csr_add_3d():
    """
    Feature: Test 3D Tensor csr_add.
    Description: Test CSRTensor matrix add.
    Expectation: Success.
    """
    alpha = Tensor(1, mstype.complex64)
    beta = Tensor(1, mstype.complex64)
    dense_shape = Tensor((2, 3, 4), mstype.int32)
    x1_batch_pointer = Tensor((0, 3, 5), mstype.int32)
    x1_row_pointer = Tensor((0, 2, 2, 3, 0, 1, 2, 2), mstype.int32)
    x1_col_indices = Tensor((0, 2, 3, 3, 0), mstype.int32)
    x1_value = Tensor((1, 2, 3, 4, 5), mstype.complex64)

    x2_batch_pointer = Tensor((0, 2, 5), mstype.int32)
    x2_row_pointer = Tensor((0, 1, 2, 2, 0, 2, 2, 3), mstype.int32)
    x2_col_indices = Tensor((3, 0, 0, 2, 3), mstype.int32)
    x2_value = Tensor((4, 5, 1, 2, 3), mstype.complex64)

    add_op = SparseMatrixAdd()
    c = add_op(dense_shape, x1_batch_pointer, x1_row_pointer, x1_col_indices, x1_value,
               dense_shape, x2_batch_pointer, x2_row_pointer, x2_col_indices, x2_value,
               alpha, beta)

    assert np.allclose(c[0].asnumpy(), (2, 3, 4))
    assert np.allclose(c[1].asnumpy(), (0, 5, 10))
    assert np.allclose(c[2].asnumpy(), (0, 3, 4, 5, 0, 3, 4, 5))
    assert np.allclose(c[3].asnumpy(), (0, 2, 3, 0, 3, 0, 2, 3, 0, 3))
    assert np.allclose(c[4].asnumpy(), (1, 2, 4, 5, 3, 1, 2, 4, 5, 3))


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_csr_add_abnormal():
    """
    Feature: Test abnormal csr_add.
    Description: Test CSRTensor matrix add.
    Expectation: Success.
    """
    alpha = Tensor(1, mstype.float32)
    beta = Tensor(1, mstype.float32)
    x1_dense_shape = Tensor((2, 3, 4), mstype.int32)
    x1_batch_pointer = Tensor((0, 3, 5), mstype.int32)
    x1_row_pointer = Tensor((0, 2, 2, 3, 0, 1, 2, 2), mstype.int32)
    x1_col_indices = Tensor((0, 2, 3, 3, 0), mstype.int32)
    x1_value = Tensor((1, 2, 3, 4, 5), mstype.float32)

    x2_dense_shape = Tensor((2, 3, 1), mstype.int32)
    x2_batch_pointer = Tensor((0, 2, 5), mstype.int32)
    x2_row_pointer = Tensor((0, 1, 2, 2, 0, 2, 2, 3), mstype.int32)
    x2_col_indices = Tensor((3, 0, 0, 2, 3), mstype.int32)
    x2_value = Tensor((4, 5, 1, 2, 3), mstype.float32)

    add_op = SparseMatrixAdd()
    with pytest.raises(RuntimeError, match="The inputs dense shape should be same"):
        add_op(x1_dense_shape, x1_batch_pointer, x1_row_pointer, x1_col_indices, x1_value,
               x2_dense_shape, x2_batch_pointer, x2_row_pointer, x2_col_indices, x2_value,
               alpha, beta)
        _pynative_executor.sync()
