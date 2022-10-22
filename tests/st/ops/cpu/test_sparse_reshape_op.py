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
from mindspore import Tensor, nn, context
from mindspore.ops.operations.sparse_ops import SparseReshape


class Net(nn.Cell):

    def __init__(self) -> None:
        super(Net, self).__init__()
        self.op = SparseReshape()

    def construct(self, indices, shape, new_shape):
        return self.op(indices, shape, new_shape)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_sparse_reshape_dyn():
    """
    Feature: test SparseReshape ops in cpu.
    Description: test the ops in dynamic shape.
    Expectation: expect correct shape result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    net = Net()

    indices_dyn = Tensor(shape=[None, None], dtype=ms.int64)
    shape_dyn = Tensor(shape=[None], dtype=ms.int64)
    new_shape_dyn = Tensor(shape=[None], dtype=ms.int64)

    net.set_inputs(indices_dyn, shape_dyn, new_shape_dyn)

    indices = Tensor([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 2, 3]],
                     dtype=ms.int64)
    shape = Tensor([2, 3, 6], dtype=ms.int64)
    new_shape = Tensor([9, -1], dtype=ms.int64)
    y_indices, y_shape = net(indices, shape, new_shape)

    expect_y_indices_shape = (5, 2)
    expect_y_shape_shape = (2,)
    assert y_indices.asnumpy().shape == expect_y_indices_shape
    assert y_shape.asnumpy().shape == expect_y_shape_shape
