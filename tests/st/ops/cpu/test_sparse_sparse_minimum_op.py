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
import pytest

import mindspore.context as context
import mindspore.nn as nn
import mindspore as ms
from mindspore.ops.operations import sparse_ops as ops
from mindspore import Tensor


class SparseSparseMinimumNet(nn.Cell):

    def __init__(self) -> None:
        super(SparseSparseMinimumNet, self).__init__()
        self.op = ops.SparseSparseMinimum()

    def construct(self, x1_indices, x1_values, x1_shape, x2_indices, x2_values,
                  x2_shape):
        return self.op(x1_indices, x1_values, x1_shape, x2_indices, x2_values,
                       x2_shape)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_sparse_sparse_minimum_dynamic_shape():
    """
    Feature: SparseSparseMinimum op in cpu.
    Description: test the op in dynamic shape.
    Expectation: expect correct shape result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    net = SparseSparseMinimumNet()

    x1_indices_dyn = Tensor(shape=[None, 3], dtype=ms.int64)
    x1_values_dyn = Tensor(shape=[None], dtype=ms.float32)
    x1_shape = Tensor([2, 2, 2], dtype=ms.int64)
    x2_indices_dyn = Tensor(shape=[None, 3], dtype=ms.int64)
    x2_values_dyn = Tensor(shape=[None], dtype=ms.float32)
    x2_shape = Tensor([2, 2, 2], dtype=ms.int64)

    net.set_inputs(x1_indices_dyn, x1_values_dyn, x1_shape, x2_indices_dyn,
                   x2_values_dyn, x2_shape)

    x1_indices = Tensor(np.array([[0, 0, 0], [0, 1, 0], [0, 1, 1]]).astype(np.int64))
    x1_values = Tensor([1, 2, 3], dtype=ms.float32)
    x2_indices = Tensor(np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]]).astype(np.int64))
    x2_values = Tensor([2, 4, 5], dtype=ms.float32)

    y_indices, y_values = net(x1_indices, x1_values, x1_shape, x2_indices,
                              x2_values, x2_shape)
    expect_y_indices_shape = (4, 3)
    expect_y_values_shape = (4,)
    assert y_indices.asnumpy().shape == expect_y_indices_shape
    assert y_values.asnumpy().shape == expect_y_values_shape
