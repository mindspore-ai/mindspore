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

import mindspore
import mindspore.context as context
import mindspore.nn as nn
import mindspore.ops.operations.sparse_ops as sparse_ops
from mindspore import Tensor


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.op = sparse_ops.SparseSoftmax()

    def construct(self, indices, values, shape):
        return self.op(indices, values, shape)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_net_graph():
    '''
    Feature: SparseSoftmax gpu TEST (GRAPH_MODE).
    Description: (4, 3) int64 indices, (4, ) float64 values, (3, ) int64 shape
    Expectation: The result matches expected output.
    '''
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    indices = Tensor([[0, 0, 0], [1, 1, 1], [0, 1, 1], [0, 0, 1]], dtype=mindspore.int64)
    values = Tensor([1, 2, 3, 2], dtype=mindspore.float64)
    shape = Tensor([0, 0, 0], dtype=mindspore.int64)
    expect = np.array([0.26894142, 0.73105858, 1., 1.]).astype(np.float64)
    net = Net()
    output = net(indices, values, shape)
    np.testing.assert_almost_equal(output.asnumpy(), expect, decimal=6)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_net_pynative():
    '''
    Feature: SparseSoftmax gpu TEST (PYNATIVE_MODE).
    Description: (4, 3) int64 indices, (4, ) float32 values, (3, ) int64 shape
    Expectation: The result matches expected output.
    '''
    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
    indices = Tensor([[0, 0, 0], [1, 1, 1], [0, 1, 1], [0, 0, 1]], dtype=mindspore.int64)
    values = Tensor([1, 2, 3, 2], dtype=mindspore.float32)
    shape = Tensor([0, 0, 0], dtype=mindspore.int64)
    expect = np.array([0.26894142, 0.73105858, 1., 1.]).astype(np.float32)
    net = Net()
    output = net(indices, values, shape)
    np.testing.assert_almost_equal(output.asnumpy(), expect, decimal=6)
