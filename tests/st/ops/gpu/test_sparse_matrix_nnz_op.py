# Copyright 2022 Huawei Technologies Co., Ltd
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

import numpy as np
import pytest
import mindspore.context as context
from mindspore import Tensor
import mindspore.ops.operations.sparse_ops as P



def sparse_matrix_nnz(nptype1, nptype2):
    x_dense_shape = Tensor(np.array([2, 3]).astype(nptype1))
    x_batch_pointers = Tensor(np.array([0, 1]).astype(nptype1))
    x_row_pointers = Tensor(np.array([0, 1, 1]).astype(nptype1))
    x_col_indices = Tensor(np.array([0]).astype(nptype1))
    x_values = Tensor(np.array([99]).astype(nptype2))
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    output_ms = P.SparseMatrixNNZ()(x_dense_shape, x_batch_pointers, x_row_pointers, x_col_indices, x_values)
    output_expect = np.array([1])
    assert np.allclose(output_ms.asnumpy(), output_expect, rtol=0, atol=0)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_sparse_matrix_nnz_int32_float32():
    """
    Feature: SparseMatrixNNZ op.
    Description: Test sparse_matrix_nnz op with int32 & float32.
    Expectation: The value and shape of output are the expected values.
    """
    sparse_matrix_nnz(np.int32, np.float32)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_sparse_matrix_nnz_int64_float64():
    """
    Feature: SparseMatrixNNZ op.
    Description: Test sparse_matrix_nnz op with int64 & float64.
    Expectation: The value and shape of output are the expected values.
    """
    sparse_matrix_nnz(np.int64, np.float64)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_sparse_matrix_nnz_int64_int32():
    """
    Feature: SparseMatrixNNZ op.
    Description: Test sparse_matrix_nnz op with int64 & int32.
    Expectation: The value and shape of output are the expected values.
    """
    sparse_matrix_nnz(np.int64, np.int32)
