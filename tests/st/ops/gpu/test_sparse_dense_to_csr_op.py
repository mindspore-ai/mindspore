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
from scipy.sparse import csr_matrix
import pytest

import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P


def generate_data(shape, datatype="float32", indicetype="int32", density=0.2):
    dense_tensor = np.random.random(shape).astype(datatype)
    dense_tensor[dense_tensor > 0.2] = 0
    indices = np.array(dense_tensor.nonzero()).T.astype(indicetype)
    return dense_tensor, indices


def generate_expected_res(dense_data, indices):
    """Generate the correct result via scipy"""
    if len(dense_data.shape) == 2:
        scipy_csr = csr_matrix(data[0])
        ret = (scipy_csr.shape, (0, scipy_csr.nnz), scipy_csr.indptr, scipy_csr.indices, scipy_csr.data)
        return ret
    # 3D
    batch_size = dense_data.shape[0]
    row_length = dense_data.shape[1]
    nnz_per_batch = np.bincount(indices[:, 0])
    shape = dense_data.shape
    batch_ptr = np.zeros(batch_size + 1).astype("int32")
    batch_ptr[1:] = nnz_per_batch
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
        self.to_csr = P.DenseToCSRSparseMatrix()

    def construct(self, x, indices):
        return self.to_csr(x, indices)


@pytest.mark.level2
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_2d_dense_to_csr():
    """
    Feature: Test 2D dense tensor to csr tensor.
    Description: Test 2D dense tensor(without batch dimension) to csr tensor.
    Expectation: Success.
    """
    data = generate_data((10, 10))
    net = DenseToCSRNet()
    out = net(Tensor(data[0]), Tensor(data[1]))
    expected = generate_expected_res(*data)
    compare_res(out, expected)


@pytest.mark.level2
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_3d_dense_to_csr():
    """
    Feature: Test 3D dense tensor to csr tensor.
    Description: Test 3D dense tensor(with batch dimension) to csr tensor.
    Expectation: Success.
    """
    data = generate_data((3, 4, 5))
    net = DenseToCSRNet()
    out = net(Tensor(data[0]), Tensor(data[1]))
    expected = generate_expected_res(*data)
    compare_res(out, expected)


@pytest.mark.level2
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_3d_dense_to_csr_fp64():
    """
    Feature: Test 3D dense tensor to csr tensor.
    Description: Test 3D dense tensor(with batch dimension, fp64) to csr tensor.
    Expectation: Success.
    """
    data = generate_data((3, 4, 5), datatype="float64")
    net = DenseToCSRNet()
    out = net(Tensor(data[0]), Tensor(data[1]))
    expected = generate_expected_res(*data)
    compare_res(out, expected)
