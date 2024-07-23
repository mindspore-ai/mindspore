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

import numpy as np
import pytest
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore.ops.operations.sparse_ops import SparseAddmm


class SparseAddmmNet(nn.Cell):
    def __init__(self):
        super(SparseAddmmNet, self).__init__()
        self.sparse_addmm = SparseAddmm()

    def construct(self, input_indices, input_values, input_shape, x2_dense, x3_dense, alpha, beta):
        return self.sparse_addmm(input_indices, input_values, input_shape, x2_dense, x3_dense, alpha, beta)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_sparse_addmm_input_int32():
    """
    Feature: SparseAddmm gpu TEST.
    Description: 2d int32 test case for SparseAddmm
    Expectation: The value and shape of output are the expected values.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

    input_indices = Tensor(np.array([[0, 1], [1, 2]]), mstype.int32)
    input_values = Tensor(np.array([1, 2]), mstype.int32)
    input_shape = Tensor(np.array([2, 3]), mstype.int32)
    x2_dense = Tensor(np.array([[1, 1], [2, 2], [3, 3]]), mstype.int32)
    x3_dense = Tensor(np.array([[2, 2], [6, 6]]), mstype.int32)
    alpha = Tensor(np.array([1]), mstype.int32)
    beta = Tensor(np.array([1]), mstype.int32)
    net = SparseAddmmNet()

    y_dense = net(input_indices, input_values, input_shape, x2_dense, x3_dense, alpha, beta)
    y_dense_expect = np.array([[4, 4], [12, 12]], dtype=np.int64)

    assert np.allclose(y_dense.asnumpy(), y_dense_expect.astype(np.int32), 0.0001, 0.0001)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_sparse_addmm_input_int64():
    """
    Feature: SparseAddmm gpu TEST.
    Description: 2d int32 test case for SparseAddmm
    Expectation: The value and shape of output are the expected values.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

    input_indices = Tensor(np.array([[0, 1], [1, 2]]), mstype.int64)
    input_values = Tensor(np.array([1, 2]), mstype.int64)
    input_shape = Tensor(np.array([2, 3]), mstype.int64)
    x2_dense = Tensor(np.array([[1, 1], [2, 2], [3, 3]]), mstype.int64)
    x3_dense = Tensor(np.array([[2, 2], [6, 6]]), mstype.int64)
    alpha = Tensor(np.array([1]), mstype.int64)
    beta = Tensor(np.array([1]), mstype.int64)
    net = SparseAddmmNet()

    y_dense = net(input_indices, input_values, input_shape, x2_dense, x3_dense, alpha, beta)
    y_dense_expect = np.array([[4, 4], [12, 12]], dtype=np.int64)

    assert np.allclose(y_dense.asnumpy(), y_dense_expect.astype(np.int64), 0.0001, 0.0001)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_sparse_addmm_input_float32():
    """
    Feature: SparseAddmm gpu TEST.
    Description: 2d int32 test case for SparseAddmm
    Expectation: The value and shape of output are the expected values.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

    input_indices = Tensor(np.array([[0, 1], [1, 2]]), mstype.int64)
    input_values = Tensor(np.array([1.0, 2.0]), mstype.float32)
    input_shape = Tensor(np.array([2, 3]), mstype.int64)
    x2_dense = Tensor(np.array([[1, 1], [2, 2], [3, 3]]), mstype.float32)
    x3_dense = Tensor(np.array([[2, 2], [6, 6]]), mstype.float32)
    alpha = Tensor(np.array([1]), mstype.float32)
    beta = Tensor(np.array([1]), mstype.float32)
    net = SparseAddmmNet()

    y_dense = net(input_indices, input_values, input_shape, x2_dense, x3_dense, alpha, beta)
    y_dense_expect = np.array([[4, 4], [12, 12]], dtype=np.float32)

    assert np.allclose(y_dense.asnumpy(), y_dense_expect.astype(np.float32), 0.0001, 0.0001)
