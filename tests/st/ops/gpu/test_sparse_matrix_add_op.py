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
        self.op = P.SparseMatrixAdd()

    def construct(self, a_shape, a_batch_pointer, a_indptr, a_indices,
                  a_values, b_shape, b_batch_pointer, b_indptr, b_indices,
                  b_values, alpha, beta):
        return self.op(a_shape, a_batch_pointer, a_indptr, a_indices, a_values,
                       b_shape, b_batch_pointer, b_indptr, b_indices, b_values,
                       alpha, beta)


def dyn_case():
    net = Net()

    a_indptr_dyn = Tensor(shape=[None], dtype=ms.int32)
    a_indices_dyn = Tensor(shape=[None], dtype=ms.int32)
    a_values_dyn = Tensor(shape=[None], dtype=ms.float32)
    a_pointers_dyn = Tensor(shape=[None], dtype=ms.int32)
    shape_dyn = Tensor(shape=[None], dtype=ms.int32)
    b_indptr_dyn = Tensor(shape=[None], dtype=ms.int32)
    b_indices_dyn = Tensor(shape=[None], dtype=ms.int32)
    b_values_dyn = Tensor(shape=[None], dtype=ms.float32)
    b_pointers_dyn = Tensor(shape=[None], dtype=ms.int32)
    alpha = Tensor(1, ms.float32)
    beta = Tensor(1, ms.float32)
    net.set_inputs(shape_dyn, a_pointers_dyn, a_indptr_dyn, a_indices_dyn,
                   a_values_dyn, shape_dyn, b_pointers_dyn, b_indptr_dyn,
                   b_indices_dyn, b_values_dyn, alpha, beta)

    a_indptr = Tensor([0, 1, 2], dtype=ms.int32)
    a_indices = Tensor([0, 1], dtype=ms.int32)
    a_values = Tensor([1, 2], dtype=ms.float32)
    a_pointers = Tensor([0, a_values.shape[0]], dtype=ms.int32)
    shape = Tensor([2, 6], dtype=ms.int32)
    b_indptr = Tensor([0, 1, 2], dtype=ms.int32)
    b_indices = Tensor([0, 1], dtype=ms.int32)
    b_values = Tensor([1, 2], dtype=ms.float32)
    b_pointers = Tensor([0, b_values.shape[0]], dtype=ms.int32)
    out = net(shape, a_pointers, a_indptr, a_indices, a_values, shape,
              b_pointers, b_indptr, b_indices, b_values, alpha, beta)

    exepct_shapes = [(2,), (2,), (3,), (2,), (2,)]
    for i in range(5):
        assert out[i].asnumpy().shape == exepct_shapes[i]


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_sparse_matrix_add_dyn():
    """
    Feature: test SparseMatrixAdd in gpu.
    Description: test the ops in dynamic case.
    Expectation: success.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    dyn_case()
    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
    dyn_case()
