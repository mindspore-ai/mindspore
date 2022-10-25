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

import pytest

import mindspore as ms
import mindspore.ops.operations.sparse_ops as P
from mindspore import nn, Tensor, context


class NetSparseTensorToCSRSparseMatrix(nn.Cell):

    def __init__(self):
        super(NetSparseTensorToCSRSparseMatrix, self).__init__()
        self.op = P.SparseTensorToCSRSparseMatrix()

    def construct(self, x_indices, x_values, x_dense_shape):
        return self.op(x_indices, x_values, x_dense_shape)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_sparse_tensor_to_csr_sparse_matrix_dyn():
    """
    Feature: test SparseTensorToCSRSparseMatrix ops in cpu.
    Description: test the ops in dynamic shape.
    Expectation: expect correct shape result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')

    net = NetSparseTensorToCSRSparseMatrix()
    x_indices_dyn = Tensor(shape=[None, None], dtype=ms.int64)
    x_values_dyn = Tensor(shape=[None], dtype=ms.float32)
    x_dense_shape_dyn = Tensor(shape=[None], dtype=ms.int64)
    net.set_inputs(x_indices_dyn, x_values_dyn, x_dense_shape_dyn)

    x_indices = Tensor(
        [[0, 0, 1], [0, 1, 2], [0, 1, 3], [1, 0, 1], [1, 1, 2], [1, 1, 3]],
        dtype=ms.int64)
    x_values = Tensor([1, 4, 3, 1, 4, 3], dtype=ms.float32)
    x_dense_shape = Tensor([2, 2, 4], dtype=ms.int64)
    out = net(x_indices, x_values, x_dense_shape)

    expect_shapes = [(3,), (3,), (6,), (6,), (6,)]
    for i in range(5):
        assert expect_shapes[i] == out[i].asnumpy().shape
