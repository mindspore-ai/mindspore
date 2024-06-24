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
import pytest
import mindspore.context as context
import mindspore.nn as nn
import mindspore as ms
import mindspore.ops.operations.sparse_ops as P
from mindspore import Tensor


class Net(nn.Cell):

    def __init__(self):
        super(Net, self).__init__()
        self.op = P.SparseMatrixMatMul()

    def construct(self, x1_dense_shape, x1_batch_pointers, x1_row_pointers,
                  x1_col_indices, x1_values, x2_dense):
        return self.op(x1_dense_shape, x1_batch_pointers, x1_row_pointers,
                       x1_col_indices, x1_values, x2_dense)


def dyn_case():
    net = Net()
    x1_dense_shape_dyn = Tensor(shape=[None], dtype=ms.int32)
    x1_batch_pointers_dyn = Tensor(shape=[None], dtype=ms.int32)
    x1_row_pointers_dyn = Tensor(shape=[None], dtype=ms.int32)
    x1_col_indices_dyn = Tensor(shape=[None], dtype=ms.int32)
    x1_values_dyn = Tensor(shape=[None], dtype=ms.float32)
    x2_dense_dyn = Tensor(shape=[None, 3], dtype=ms.float32)
    net.set_inputs(x1_dense_shape_dyn, x1_batch_pointers_dyn,
                   x1_row_pointers_dyn, x1_col_indices_dyn, x1_values_dyn,
                   x2_dense_dyn)

    x1_dense_shape = Tensor([4, 5], dtype=ms.int32)
    x1_batch_pointers = Tensor([0, 4], dtype=ms.int32)
    x1_row_pointers = Tensor([0, 1, 1, 3, 4], dtype=ms.int32)
    x1_col_indices = Tensor([0, 3, 4, 0], dtype=ms.int32)
    x1_values = Tensor([1.0, 5.0, -1.0, -2.0], dtype=ms.float32)
    x2_dense = Tensor([[2.0, 0.8, 1.0], [2.9, 3.2, 0.0], [7.0, 4.6, 0.2],
                       [3.5, 4.9, 1.4], [4.0, 3.7, 6.9]],
                      dtype=ms.float32)
    out = net(x1_dense_shape, x1_batch_pointers, x1_row_pointers,
              x1_col_indices, x1_values, x2_dense)
    assert out.asnumpy().shape == (4, 3)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_sparse_segment_mat_mul_dyn():
    """
    Feature: test SparseSegmentMatMul in cpu.
    Description: test the ops in dynamic case.
    Expectation: success.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    dyn_case()
    context.set_context(mode=context.PYNATIVE_MODE, device_target='CPU')
    dyn_case()
