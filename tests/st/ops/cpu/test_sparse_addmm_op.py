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

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
import mindspore as ms
from mindspore.ops.operations.sparse_ops import SparseAddmm


class SparseAddmmNet(nn.Cell):

    def __init__(self):
        super(SparseAddmmNet, self).__init__()
        self.op = SparseAddmm()

    def construct(self, indices, values, sparse_shape, x2_dense, x3_dense,
                  alpha, beta):
        return self.op(indices, values, sparse_shape, x2_dense, x3_dense,
                       alpha, beta)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_sparse_addmm_dyn():
    """
    Feature: test SparseAddmm ops in cpu.
    Description: test the ops in dynamic shape.
    Expectation: expect correct shape result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')

    net = SparseAddmmNet()

    indices_dyn = Tensor(shape=[None, 2], dtype=ms.int32)
    values_dyn = Tensor(shape=[None], dtype=ms.float32)
    sparse_shape = Tensor([3, 4], dtype=ms.int32)
    x2_dense_dyn = Tensor(shape=[4, None], dtype=ms.float32)
    x3_dense_dyn = Tensor(shape=[3, None], dtype=ms.float32)
    alpha = Tensor([1], dtype=ms.float32)
    beta = Tensor([1], dtype=ms.float32)

    net.set_inputs(indices_dyn, values_dyn, sparse_shape, x2_dense_dyn,
                   x3_dense_dyn, alpha, beta)

    indices = Tensor([[0, 1], [1, 2]], dtype=ms.int32)
    values = Tensor([1, 2], dtype=ms.float32)
    x2_dense = Tensor([[1, 1], [2, 2], [3, 3], [4, 4]], dtype=ms.float32)
    x3_dense = Tensor([[2, 2], [6, 6], [0, 0]], dtype=ms.float32)
    out = net(indices, values, sparse_shape, x2_dense, x3_dense, alpha, beta)

    expect_shape = (3, 2)
    assert out.asnumpy().shape == expect_shape
