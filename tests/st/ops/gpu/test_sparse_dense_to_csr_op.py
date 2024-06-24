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
from tests.mark_utils import arg_mark

import numpy as np
from scipy.sparse import csr_matrix
import pytest

import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops.operations.sparse_ops import DenseToCSRSparseMatrix


def generate_data(shape, datatype="float32", indicetype="int32", density=0.5):
    mask = np.random.random(shape)
    mask[mask > density] = 0
    dense_tensor = (np.random.random(shape) * 10).astype(datatype)
    dense_tensor[mask == 0] = 0
    indices = np.array(dense_tensor.nonzero()).T.astype(indicetype)
    return dense_tensor, indices


def generate_expected_res(dense_data, indices):
    """Generate the correct result via scipy"""
    if len(dense_data.shape) == 2:
        scipy_csr = csr_matrix(dense_data)
        ret = (scipy_csr.shape, (0, scipy_csr.nnz), scipy_csr.indptr, scipy_csr.indices, scipy_csr.data)
        return ret
    # 3D
    batch_size = dense_data.shape[0]
    row_length = dense_data.shape[1]
    nnz_per_batch = np.bincount(indices[:, 0])
    shape = dense_data.shape
    batch_ptr = np.zeros(batch_size + 1).astype("int32")
    batch_ptr[1:nnz_per_batch.shape[0] + 1] = nnz_per_batch
    batch_ptr = np.cumsum(batch_ptr)
    indptr = np.zeros(batch_size * (row_length + 1)).astype("int32")
    col_indices = np.zeros(indices.shape[0]).astype("int32")
    values = dense_data[dense_data != 0]
    for i in range(batch_size):
        curr_data = dense_data[i]
        scipy_csr = csr_matrix(curr_data)
        indptr[i * (row_length + 1):(i + 1) * (row_length + 1)] = scipy_csr.indptr
        col_indices[batch_ptr[i]:batch_ptr[i + 1]] = scipy_csr.indices
    ret = (shape, batch_ptr, indptr, col_indices, values)
    return ret


def compare_res(res, expected):
    for comp in zip(res, expected):
        assert (comp[0].asnumpy() == comp[1]).all()


class DenseToCSRNet(nn.Cell):
    def __init__(self):
        super(DenseToCSRNet, self).__init__()
        self.to_csr = DenseToCSRSparseMatrix()

    def construct(self, x, indices):
        return self.to_csr(x, indices)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('indicetype, datatype', [("int32", "float32"),
                                                  ("int32", "float64"),
                                                  ("int32", "complex64"),
                                                  ("int32", "complex128")])
def test_2d_dense_to_csr(indicetype, datatype):
    """
    Feature: Test 2D dense tensor to csr tensor.
    Description: Test 2D dense tensor(without batch dimension) to csr tensor.
    Expectation: Success.
    """
    data = generate_data((10, 10), datatype=datatype, indicetype=indicetype)
    net = DenseToCSRNet()
    out = net(Tensor(data[0]), Tensor(data[1]))
    expected = generate_expected_res(*data)
    compare_res(out, expected)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('indicetype, datatype', [("int32", "float32"),
                                                  ("int32", "float64"),
                                                  ("int32", "complex64"),
                                                  ("int32", "complex128")])
def test_3d_dense_to_csr(indicetype, datatype):
    """
    Feature: Test 3D dense tensor to csr tensor.
    Description: Test 3D dense tensor(with batch dimension) to csr tensor.
    Expectation: Success.
    """
    data = generate_data((3, 4, 5), datatype=datatype, indicetype=indicetype)
    net = DenseToCSRNet()
    out = net(Tensor(data[0]), Tensor(data[1]))
    expected = generate_expected_res(*data)
    compare_res(out, expected)
