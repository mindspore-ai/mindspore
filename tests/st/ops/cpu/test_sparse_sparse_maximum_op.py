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
import mindspore as ms
from mindspore.ops.operations import sparse_ops as ops
from mindspore import Tensor


class SparseSparseMaximumNet(nn.Cell):

    def __init__(self) -> None:
        super(SparseSparseMaximumNet, self).__init__()
        self.op = ops.SparseSparseMaximum()

    def construct(self, x1_indices, x1_values, x1_shape, x2_indices, x2_values,
                  x2_shape):
        return self.op(x1_indices, x1_values, x1_shape, x2_indices, x2_values,
                       x2_shape)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_sparse_sparse_maximum_dynamic_shape():
    """
    Feature: SparseSparseMaximum op in cpu.
    Description: test the op in dynamic shape.
    Expectation: expect correct shape result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    net = SparseSparseMaximumNet()

    x1_indices_dyn = Tensor(shape=[None, 2], dtype=ms.int64)
    x1_values_dyn = Tensor(shape=[None], dtype=ms.float32)
    x1_shape = Tensor([3, 3], dtype=ms.int64)
    x2_indices_dyn = Tensor(shape=[None, 2], dtype=ms.int64)
    x2_values_dyn = Tensor(shape=[None], dtype=ms.float32)
    x2_shape = Tensor([3, 3], dtype=ms.int64)

    net.set_inputs(x1_indices_dyn, x1_values_dyn, x1_shape, x2_indices_dyn,
                   x2_values_dyn, x2_shape)

    x1_indices = Tensor([[0, 1], [1, 2]], dtype=ms.int64)
    x1_values = Tensor([1, 2], dtype=ms.float32)
    x2_indices = Tensor([[0, 1], [1, 1]], dtype=ms.int64)
    x2_values = Tensor([3, 4], dtype=ms.float32)

    y_indices, y_values = net(x1_indices, x1_values, x1_shape, x2_indices,
                              x2_values, x2_shape)
    expect_y_indices_shape = (3, 2)
    expect_y_values_shape = (3,)
    assert y_indices.asnumpy().shape == expect_y_indices_shape
    assert y_values.asnumpy().shape == expect_y_values_shape
