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
import numpy as np
from mindspore import Tensor
import mindspore.ops.operations.sparse_ops as op
from mindspore.nn import Cell
import mindspore.context as context
context.set_context(mode=context.GRAPH_MODE, device_target='GPU')


class SparseReorder(Cell):
    def __init__(self):
        super().__init__()
        self.sparse_reorder = op.SparseReorder()

    def construct(self, indices, values, shape):
        return self.sparse_reorder(indices, values, shape)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_sparse_reorder_int16():
    """
    Feature: SparseReorder
    Description: Test of input
    Expectation: The results are as expected
    """
    type_i = np.int16
    ertol_loss = 1e-05
    indices = np.array([[0, 1], [2, 1]]).astype(np.int64)
    values = np.array([1, 2]).astype(type_i)
    shape = np.array([3, 3]).astype(np.int64)
    net = SparseReorder()
    y_indices, y_values = net(Tensor(indices), Tensor(values), Tensor(shape))
    y_indices = y_indices.asnumpy()
    y_values = y_values.asnumpy()
    expect_y_indices = np.array([[0, 1], [2, 1]]).astype(np.int64)
    expect_y_values = np.array([1, 2]).astype(type_i)
    assert np.allclose(y_indices, expect_y_indices, ertol_loss)
    assert np.allclose(y_values, expect_y_values, ertol_loss)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_sparse_reorder_int32():
    """
    Feature: SparseReorder
    Description: Test of input
    Expectation: The results are as expected
    """
    type_i = np.int32
    ertol_loss = 1e-05
    indices = np.array([[0, 1], [2, 1]]).astype(np.int64)
    values = np.array([1, 2]).astype(type_i)
    shape = np.array([3, 3]).astype(np.int64)
    net = SparseReorder()
    y_indices, y_values = net(Tensor(indices), Tensor(values), Tensor(shape))
    y_indices = y_indices.asnumpy()
    y_values = y_values.asnumpy()
    expect_y_indices = np.array([[0, 1], [2, 1]]).astype(np.int64)
    expect_y_values = np.array([1, 2]).astype(type_i)
    assert np.allclose(y_indices, expect_y_indices, ertol_loss)
    assert np.allclose(y_values, expect_y_values, ertol_loss)
