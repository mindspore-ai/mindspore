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
import scipy.sparse
import pytest

import mindspore.nn as nn
from mindspore import Tensor
from mindspore.common import dtype as mstype
from mindspore.ops import operations as P
from mindspore.ops.operations.sparse_ops import CSRSparseMatrixToSparseTensor


def generate_data(shape, datatype="float32", indicetype="int32", density=0.2):
    data_shape = shape[-2:]
    shape_tensor = np.array(shape, dtype=indicetype)
    is_batch_csr = len(shape) == 3
    batch_size = shape[0] if is_batch_csr else 1
    accum_nnz = 0
    x_batch_pointers = np.array(0, dtype=indicetype)
    coo_indices = []
    for i in range(batch_size):
        csr_matrix = scipy.sparse.random(data_shape[0], data_shape[1], format="csr",
                                         density=density, dtype=indicetype)
        row_pointers = np.asarray(csr_matrix.indptr, dtype=indicetype)
        col_indices = np.asarray(csr_matrix.indices, dtype=indicetype)
        values = np.asarray(csr_matrix.data, dtype=datatype)
        coo_tensor = csr_matrix.tocoo()
        indices = np.stack(
            (np.asarray(coo_tensor.row, dtype=indicetype), np.asarray(coo_tensor.col, dtype=indicetype)), axis=1)
        if is_batch_csr:
            indices = np.insert(indices, 0, i, axis=1)
        coo_indices.append(indices)
        if i == 0:
            x_row_pointers = row_pointers
            x_col_indices = col_indices
            x_values = values
        else:
            x_row_pointers = np.append(x_row_pointers, row_pointers)
            x_col_indices = np.append(x_col_indices, col_indices)
            x_values = np.append(x_values, values)
        accum_nnz += csr_matrix.nnz
        x_batch_pointers = np.append(x_batch_pointers, accum_nnz)
    output_indices = np.concatenate(coo_indices)
    x_batch_pointers = x_batch_pointers.astype(indicetype)
    return ((shape_tensor, x_batch_pointers, x_row_pointers, x_col_indices, x_values),
            (output_indices, x_values, shape_tensor))


def compare_res(res, expected):
    assert len(res) == len(expected)
    for r, e in zip(res, expected):
        assert np.allclose(r.asnumpy(), e)


class CSRToCOONet(nn.Cell):
    def __init__(self):
        super(CSRToCOONet, self).__init__()
        self.to_coo = CSRSparseMatrixToSparseTensor()

    def construct(self, shape, x_batch_pointers, x_row_pointers, x_col_indices, x_values):
        return self.to_coo(shape, x_batch_pointers, x_row_pointers, x_col_indices, x_values)


class DynamicShapeCSRToCOONet(nn.Cell):
    def __init__(self):
        super(DynamicShapeCSRToCOONet, self).__init__()
        self.unique = P.Unique()
        self.to_coo = CSRSparseMatrixToSparseTensor()

    def construct(self, shape, x_batch_pointers, x_row_pointers, x_col_indices, x_values):
        unqie_col_indices, _ = self.unique(x_col_indices)
        unique_values, _ = self.unique(x_values)
        return self.to_coo(shape, x_batch_pointers, x_row_pointers, unqie_col_indices, unique_values)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_2d_csr_to_coo():
    """
    Feature: Test 2D CSR tensor to COO tensor.
    Description: Test 2D CSR tensor(without batch dimension) to csr tensor.
    Expectation: Success.
    """
    inputs, expects = generate_data((5, 10))
    input_tensors = [Tensor(x) for x in inputs]
    net = CSRToCOONet()
    outputs = net(*input_tensors)
    compare_res(outputs, expects)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_3d_csr_to_coo():
    """
    Feature: Test 3D CSR tensor to COO tensor.
    Description: Test 3D CSR tensor(with batch dimension) to COO tensor.
    Expectation: Success.
    """
    inputs, expects = generate_data((3, 5, 10))
    input_tensors = [Tensor(x) for x in inputs]
    net = CSRToCOONet()
    outputs = net(*input_tensors)
    compare_res(outputs, expects)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_3d_csr_to_coo_fp64():
    """
    Feature: Test 3D CSR tensor to COO tensor.
    Description: Test 3D CSR tensor(with batch dimension, fp64) to COO tensor.
    Expectation: Success.
    """
    inputs, expects = generate_data((3, 5, 10), datatype="float64")
    input_tensors = [Tensor(x) for x in inputs]
    net = CSRToCOONet()
    outputs = net(*input_tensors)
    compare_res(outputs, expects)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_dynamic_shape_csr_to_coo():
    """
    Feature: Test dynamic shape.
    Description: Test CSR tensor to COO tensor.
    Expectation: Success.
    """
    shape = (3, 10)
    x_batch_pointers = Tensor([0, -1], dtype=mstype.int32)
    indptr = Tensor([0, 2, 6, 9], dtype=mstype.int32)
    indices = Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=mstype.int32)
    values = np.random.rand(9)
    values = np.sort(values)
    net = DynamicShapeCSRToCOONet()
    outputs = net(Tensor(shape, dtype=mstype.int32), x_batch_pointers, indptr, indices,
                  Tensor(values, dtype=mstype.float32))
    coo_indices = np.array([[0, 1], [0, 2], [1, 3], [1, 4], [1, 5], [1, 6], [2, 7], [2, 8], [2, 9]])
    compare_res(outputs, (coo_indices, values, np.array(shape)))
