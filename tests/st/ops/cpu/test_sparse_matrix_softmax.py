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
"""smoke tests for SparseMatrixSoftmax"""

import numpy as np
import pytest
import mindspore
import mindspore.nn as nn
from mindspore import Tensor, CSRTensor, context
from mindspore.common import dtype as mstype
from mindspore.ops.function import csr_softmax
from mindspore.ops.operations.sparse_ops import SparseMatrixSoftmax
from mindspore.ops.primitive import constexpr


@constexpr
def _make_tensor(a, dtype=mstype.int32):
    """Converts the input to tensor."""
    if not isinstance(a, (list, tuple, int, float, bool)):
        raise TypeError("input data must be `int`, `float`, `bool`, `list` or `tuple`")
    if isinstance(a, (list, tuple)):
        a = np.asarray(a)
        if a.dtype is np.dtype('object'):
            raise ValueError('Input array must have the same size across all dimensions.')
    return Tensor(a, dtype)


def create_csr_tensor():
    a_indptr = Tensor([0, 4, 6], dtype=mstype.int32)
    a_indices = Tensor([0, 2, 3, 4, 3, 4], dtype=mstype.int32)
    a_values = Tensor([1, 2, 3, 4, 1, 2], dtype=mstype.float32)
    shape = (2, 6)
    logits = CSRTensor(a_indptr, a_indices, a_values, shape)
    return logits


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_ops_sparse_matrix_softmax_vs_nn_softmax_int32():
    """
    Feature: Test function sparse_matrix_softmax.
    Description: Test CSRTensor matrix softmax compared with nn.Softmax.
    Expectation: Success.
    """
    context.set_context(mode=context.GRAPH_MODE)
    logits = create_csr_tensor()
    logits_pointers = _make_tensor(logits.values.shape[0], mstype.int32)
    sparse_matrix_softmax_op = SparseMatrixSoftmax(mstype.float32)
    c = sparse_matrix_softmax_op(Tensor(logits.shape, dtype=mstype.int32), logits_pointers,
                                 logits.indptr.astype(mstype.int32),
                                 logits.indices.astype(mstype.int32), logits.values)
    print(c[4])
    c_values = Tensor([1, 2, 3, 4, 1, 2], dtype=mstype.float32)
    output = nn.Softmax()
    row_index = logits.shape[0]
    start = 0
    weights = []
    for i in range(1, row_index + 1):
        single_index = logits.indptr[i] - logits.indptr[i - 1].astype(mindspore.int64)
        c_logits = c_values[start:single_index + start]
        c_weights = output(c_logits)
        start = start + single_index
        weights = np.concatenate((weights, c_weights), axis=0)
    eps = np.array([1e-6 for i in range(logits.indptr[row_index])])
    assert np.all(weights - c[4] < eps)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_ops_sparse_matrix_softmax_vs_nn_softmax_int64():
    """
    Feature: Test function sparse_matrix_softmax.
    Description: Test CSRTensor matrix softmax compared with nn.Softmax.
    Expectation: Success.
    """
    context.set_context(mode=context.GRAPH_MODE)
    logits = create_csr_tensor()
    logits_pointers = _make_tensor(logits.values.shape[0], mstype.int64)
    sparse_matrix_softmax_op = SparseMatrixSoftmax(mstype.float32)
    c = sparse_matrix_softmax_op(Tensor(logits.shape, dtype=mstype.int64), logits_pointers,
                                 logits.indptr.astype(mstype.int64),
                                 logits.indices.astype(mstype.int64), logits.values)
    c_values = Tensor([1, 2, 3, 4, 1, 2], dtype=mstype.float32)
    output = nn.Softmax()
    row_index = logits.shape[0]
    start = 0
    weights = []
    for i in range(1, row_index + 1):
        single_index = logits.indptr[i] - logits.indptr[i - 1].astype(mindspore.int64)
        c_logits = c_values[start:single_index + start]
        c_weights = output(c_logits)
        start = start + single_index
        weights = np.concatenate((weights, c_weights), axis=0)
    eps = np.array([1e-6 for i in range(logits.indptr[row_index])])
    assert np.all(weights - c[4] < eps)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_ops_sparse_matrix_softmax_vs_nn_softmax_fp32():
    """
    Feature: Test function sparse_matrix_softmax.
    Description: Test CSRTensor matrix softmax compared with nn.Softmax.
    Expectation: Success.
    """
    context.set_context(mode=context.GRAPH_MODE)
    logits = create_csr_tensor()
    logits_pointers = _make_tensor(logits.values.shape[0], mstype.int64)
    sparse_matrix_softmax_op = SparseMatrixSoftmax(mstype.float32)
    c = sparse_matrix_softmax_op(Tensor(logits.shape, dtype=mstype.int32), logits_pointers, logits.indptr,
                                 logits.indices, logits.values.astype(mstype.float32))
    c_values = Tensor([1, 2, 3, 4, 1, 2], dtype=mstype.float32)
    output = nn.Softmax()
    row_index = logits.shape[0]
    start = 0
    weights = []
    for i in range(1, row_index + 1):
        single_index = logits.indptr[i] - logits.indptr[i - 1].astype(mindspore.int64)
        c_logits = c_values[start:single_index + start]
        c_weights = output(c_logits)
        start = start + single_index
        weights = np.concatenate((weights, c_weights), axis=0)
    eps = np.array([1e-6 for i in range(logits.indptr[row_index])])
    assert np.all(weights - c[4] < eps)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_function_sparse_matrix_softmax_vs_nn_softmax():
    """
    Feature: Test function sparse_matrix_softmax.
    Description: Test CSRTensor matrix softmax compared with nn.Softmax.
    Expectation: Success.
    """
    context.set_context(mode=context.GRAPH_MODE)
    logits = create_csr_tensor()
    c = csr_softmax(logits, dtype=mstype.float32)
    c_values = Tensor([1, 2, 3, 4, 1, 2], dtype=mstype.float32)
    output = nn.Softmax()
    row_index = logits.shape[0]
    start = 0
    weights = []
    for i in range(1, row_index + 1):
        single_index = logits.indptr[i] - logits.indptr[i - 1].astype(mindspore.int64)
        c_logits = c_values[start:single_index + start]
        c_weights = output(c_logits)
        start = start + single_index
        weights = np.concatenate((weights, c_weights), axis=0)
    eps = np.array([1e-6 for i in range(logits.indptr[row_index])])
    assert np.all(weights - c.values < eps)
