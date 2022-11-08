# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
import mindspore.ops.operations.sparse_ops as sparse_ops
from mindspore import Tensor
from mindspore.common import dtype as mstype


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.op = sparse_ops.SparseReshape()

    def construct(self, indices, shape, new_shape):
        return self.op(indices, shape, new_shape)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_pynative():
    '''
    Feature: SparseReshape gpu TEST (PYNATIVE_MODE).
    Description: (2, 3) int64 indices, (3, ) float64 shape, (3, ) int64 new_shape
    Expectation: The result matches expected output.
    '''
    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
    indices = Tensor([[1, 1, 0], [2, 2, 1]], dtype=mstype.int64)
    shape = Tensor([3, 4, 2], dtype=mstype.int64)
    new_shape = Tensor([2, 3, -1], dtype=mstype.int64)
    y_indices_expect = np.array([[0, 2, 2], [1, 2, 1]]).astype(np.int64)
    y_shape_expect = np.array([2, 3, 4]).astype(np.int64)
    net = Net()
    y_indices, y_shape = net(indices, shape, new_shape)
    np.testing.assert_almost_equal(y_indices.asnumpy(), y_indices_expect)
    np.testing.assert_almost_equal(y_shape.asnumpy(), y_shape_expect)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_grath():
    '''
    Feature: SparseReshape gpu TEST (GRAPH_MODE).
    Description: (2, 3) int64 indices, (3, ) float64 shape, (3, ) int64 new_shape
    Expectation: The result matches expected output.
    '''
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    indices = Tensor([[1, 1, 0], [2, 2, 1]], dtype=mstype.int64)
    shape = Tensor([3, 4, 2], dtype=mstype.int64)
    new_shape = Tensor([2, -1, 4], dtype=mstype.int64)
    y_indices_expect = np.array([[0, 2, 2], [1, 2, 1]]).astype(np.int64)
    y_shape_expect = np.array([2, 3, 4]).astype(np.int64)
    net = Net()
    y_indices, y_shape = net(indices, shape, new_shape)
    np.testing.assert_almost_equal(y_indices.asnumpy(), y_indices_expect)
    np.testing.assert_almost_equal(y_shape.asnumpy(), y_shape_expect)
