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
from mindspore import Tensor, nn, context


class NetSparseMatrixNNZ(nn.Cell):

    def __init__(self):
        super(NetSparseMatrixNNZ, self).__init__()
        self.op = P.SparseMatrixNNZ()

    def construct(self, dense_shape, batch_pointers, row_pointers, col_indices,
                  values):
        return self.op(dense_shape, batch_pointers, row_pointers, col_indices,
                       values)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_enecard
def test_sparse_matrix_nnz_dyn():
    """
    Feature: test SparseMatrixNNZ ops in cpu.
    Description: test the ops in dynamic shape.
    Expectation: expect correct shape result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    net = NetSparseMatrixNNZ()

    x_dense_shape_dyn = Tensor(shape=[None], dtype=ms.int32)
    x_batch_pointers_dyn = Tensor(shape=[None], dtype=ms.int32)
    x_row_pointers_dyn = Tensor(shape=[None], dtype=ms.int32)
    x_col_indices_dyn = Tensor(shape=[None], dtype=ms.int32)
    x_values_dyn = Tensor(shape=[None], dtype=ms.float32)

    net.set_inputs(x_dense_shape_dyn, x_batch_pointers_dyn, x_row_pointers_dyn,
                   x_col_indices_dyn, x_values_dyn)

    x_dense_shape = Tensor([2, 3], dtype=ms.int32)
    x_batch_pointers = Tensor([0, 1], dtype=ms.int32)
    x_row_pointers = Tensor([0, 1, 1], dtype=ms.int32)
    x_col_indices = Tensor([0], dtype=ms.int32)
    x_values = Tensor([99], dtype=ms.float32)
    output = net(x_dense_shape, x_batch_pointers, x_row_pointers,
                 x_col_indices, x_values)

    expect_shape = (1,)
    assert output.asnumpy().shape == expect_shape
