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
from mindspore.ops.operations.sparse_ops import SparseReorder
from mindspore import Tensor


class Net(nn.Cell):

    def __init__(self) -> None:
        super(Net, self).__init__()
        self.op = SparseReorder()

    def construct(self, indices, values, shape):
        return self.op(indices, values, shape)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_sparse_reorder_dyn():
    """
    Feature: test SparseReorder ops in cpu.
    Description: test the ops in dynamic shape.
    Expectation: expect correct shape result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    net = Net()

    indices_dyn = Tensor(shape=[None, None], dtype=ms.int64)
    values_dyn = Tensor(shape=[None], dtype=ms.float32)
    shape_dyn = Tensor(shape=[None], dtype=ms.int64)

    net.set_inputs(indices_dyn, values_dyn, shape_dyn)

    indices = Tensor([[2, 1], [0, 1]], dtype=ms.int64)
    values = Tensor([1, 2], dtype=ms.float32)
    shape = Tensor([3, 3], dtype=ms.int64)

    y_indices, y_values = net(indices, values, shape)

    expect_y_indices_shape = (2, 2)
    expect_y_values_shape = (2,)
    assert y_indices.asnumpy().shape == expect_y_indices_shape
    assert y_values.asnumpy().shape == expect_y_values_shape
