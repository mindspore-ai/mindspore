# Copyright 2020 Huawei Technologies Co., Ltd
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
import mindspore
from mindspore import Tensor
import mindspore.context as context
import mindspore.nn as nn
import mindspore.ops.operations.sparse_ops as P
import numpy as np


class SparseSparseMinimumNet(nn.Cell):
    def __init__(self):
        super(SparseSparseMinimumNet, self).__init__()
        self.sparsesparseminimum = P.SparseSparseMinimum()

    def construct(self, x1_indices, x1_values, x1_shape, x2_indices, x2_values, x2_shape):
        return self.sparsesparseminimum(x1_indices, x1_values, x1_shape, x2_indices, x2_values, x2_shape)


def sparse_sparse_minimum(loss):
    loss1 = loss
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    indices1 = Tensor([[0, 1], [0, 1], [2, 2], [0, 2]])
    values1 = Tensor([4, 2, 3, 4], dtype=mindspore.float32)
    shape1 = Tensor([3, 4])
    indices2 = Tensor([[0, 1], [2, 3]])
    values2 = Tensor([2, 3], dtype=mindspore.float32)
    shape2 = Tensor([3, 4])
    net = SparseSparseMinimumNet()
    m, n = net(indices1, values1, shape1, indices2, values2, shape2)
    expected_m = np.array([[0, 1], [0, 1], [2, 2], [0, 2], [2, 3]], dtype=np.int64)
    expected_n = np.array([2, 0, 0, 0, 0], dtype=np.float32)
    assert np.allclose(m.asnumpy(), expected_m, loss, loss)
    assert np.allclose(n.asnumpy(), expected_n, loss1, loss1)


def sparse_sparse_minimum_pynative(loss):
    loss1 = loss
    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
    indices1 = Tensor([[0, 1], [0, 1], [2, 2], [0, 2]])
    values1 = Tensor([4, 2, 3, 4], dtype=mindspore.float32)
    shape1 = Tensor([3, 4])
    indices2 = Tensor([[0, 1], [2, 3]])
    values2 = Tensor([2, 3], dtype=mindspore.float32)
    shape2 = Tensor([3, 4])
    net = SparseSparseMinimumNet()
    m, n = net(indices1, values1, shape1, indices2, values2, shape2)
    expected_m = np.array([[0, 1], [0, 1], [2, 2], [0, 2], [2, 3]], dtype=np.int64)
    expected_n = np.array([2, 0, 0, 0, 0], dtype=np.float32)
    assert np.allclose(m.asnumpy(), expected_m, loss, loss)
    assert np.allclose(n.asnumpy(), expected_n, loss1, loss1)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_sparse_sparse_maximum_graph_float32():
    """
    Feature: ALL To ALL
    Description: test cases for SparseSparseMinimum
    Expectation: the result match to tensorflow
    """
    sparse_sparse_minimum(loss=1.0e-4)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_sparse_sparse_minimum_pynative_float32():
    """
    Feature: ALL To ALL
    Description: test cases for SparseSparseMinimum
    Expectation: the result match to tensorflow
    """
    sparse_sparse_minimum_pynative(loss=1.0e-5)
