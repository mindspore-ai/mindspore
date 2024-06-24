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


class NetSparseMatrixTranspose(nn.Cell):

    def __init__(self, conjugate=False) -> None:
        super(NetSparseMatrixTranspose, self).__init__()
        self.op = P.SparseMatrixTranspose(conjugate=conjugate)

    def construct(self, x_dense_shape, x_batch_pointers, x_row_pointers,
                  x_col_indices, x_values):
        return self.op(x_dense_shape, x_batch_pointers, x_row_pointers,
                       x_col_indices, x_values)


@pytest.mark.skip(reason="never run on ci or smoke test")
def test_sparse_matrix_transpose_dyn():
    """
    Feature: test SparseMatrixTranspose ops in cpu.
    Description: test the ops in dynamic shape.
    Expectation: expect correct shape result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    net = NetSparseMatrixTranspose()

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
    outputs = net(x_dense_shape, x_batch_pointers, x_row_pointers,
                  x_col_indices, x_values)

    expect_shapes = [(2,), (2,), (4,), (1,), (1,)]
    for i in range(5):
        assert outputs[i].asnumpy().shape == expect_shapes[i]
