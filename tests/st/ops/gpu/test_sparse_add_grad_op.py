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
from mindspore import Tensor
from mindspore.ops.operations._sparse_grad_ops import SparseAddGrad
import mindspore.nn as nn


class SparseAddGradNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.sparse_add_grad = SparseAddGrad()

    def construct(self, val_grad, x1_indices, x2_indices, sum_indices):
        return self.sparse_add_grad(val_grad, x1_indices, x2_indices, sum_indices)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_sparse_add_grad_fp32():
    """
    Feature: test SparseGrad ops
    Description: test fp32 value input
    Expectation: Output matching expected values
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    val_grad = np.array([2, 4]).astype(np.float32)
    x1_indices = np.array([[0, 1], [1, 2]]).astype(np.int64)
    x2_indices = np.array([[0, 1], [1, 2]]).astype(np.int64)
    sum_indices = np.array([[0, 1], [1, 2]]).astype(np.int64)
    dx1, dx2 = SparseAddGradNet()(Tensor(val_grad), Tensor(x1_indices), Tensor(x2_indices), Tensor(sum_indices))
    ground_truth_dx1 = np.array([2, 4], np.float32)
    ground_truth_dx2 = np.array([2, 4], np.float32)
    assert np.allclose(dx1.asnumpy(), ground_truth_dx1)
    assert np.allclose(dx2.asnumpy(), ground_truth_dx2)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_sparse_add_grad_fp64():
    """
    Feature: test SparseGrad ops
    Description: test fp64 value input
    Expectation: Output matching expected values
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    val_grad = np.array([2, 2, 2]).astype(np.float64)
    x1_indices = np.array([[0, 1], [1, 2]]).astype(np.int64)
    x2_indices = np.array([[0, 1], [1, 3]]).astype(np.int64)
    sum_indices = np.array([[0, 1], [1, 2], [1, 3]]).astype(np.int64)
    dx1, dx2 = SparseAddGradNet()(Tensor(val_grad), Tensor(x1_indices), Tensor(x2_indices), Tensor(sum_indices))
    ground_truth_dx1 = np.array([2, 2], np.float64)
    ground_truth_dx2 = np.array([2, 2], np.float64)
    assert np.allclose(dx1.asnumpy(), ground_truth_dx1)
    assert np.allclose(dx2.asnumpy(), ground_truth_dx2)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_sparse_add_grad_int8():
    """
    Feature: test SparseGrad ops
    Description: test int8 value input
    Expectation: Output matching expected values
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    val_grad = np.array([2, 2, 2]).astype(np.int8)
    x1_indices = np.array([[0, 1], [1, 2]]).astype(np.int64)
    x2_indices = np.array([[0, 1], [1, 3]]).astype(np.int64)
    sum_indices = np.array([[0, 1], [1, 2], [1, 3]]).astype(np.int64)
    dx1, dx2 = SparseAddGradNet()(Tensor(val_grad), Tensor(x1_indices), Tensor(x2_indices), Tensor(sum_indices))
    ground_truth_dx1 = np.array([2, 2], np.int8)
    ground_truth_dx2 = np.array([2, 2], np.int8)
    assert np.allclose(dx1.asnumpy(), ground_truth_dx1)
    assert np.allclose(dx2.asnumpy(), ground_truth_dx2)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_sparse_add_grad_int16():
    """
    Feature: test SparseGrad ops
    Description: test int16 value input
    Expectation: Output matching expected values
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    val_grad = np.array([1, 1, 4]).astype(np.int16)
    x1_indices = np.array([[0, 1], [1, 3]]).astype(np.int64)
    x2_indices = np.array([[1, 2], [1, 3]]).astype(np.int64)
    sum_indices = np.array([[0, 1], [1, 2], [1, 3]]).astype(np.int64)
    dx1, dx2 = SparseAddGradNet()(Tensor(val_grad), Tensor(x1_indices), Tensor(x2_indices), Tensor(sum_indices))
    ground_truth_dx1 = np.array([1, 4], np.int16)
    ground_truth_dx2 = np.array([1, 4], np.int16)
    assert np.allclose(dx1.asnumpy(), ground_truth_dx1)
    assert np.allclose(dx2.asnumpy(), ground_truth_dx2)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_sparse_add_grad_int32():
    """
    Feature: test SparseGrad ops
    Description: test int32 value input
    Expectation: Output matching expected values
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    val_grad = np.array([1, 1, 4]).astype(np.int32)
    x1_indices = np.array([[0, 1], [1, 3]]).astype(np.int64)
    x2_indices = np.array([[1, 2], [1, 3]]).astype(np.int64)
    sum_indices = np.array([[0, 1], [1, 2], [1, 3]]).astype(np.int64)
    dx1, dx2 = SparseAddGradNet()(Tensor(val_grad), Tensor(x1_indices), Tensor(x2_indices), Tensor(sum_indices))
    ground_truth_dx1 = np.array([1, 4], np.int32)
    ground_truth_dx2 = np.array([1, 4], np.int32)
    assert np.allclose(dx1.asnumpy(), ground_truth_dx1)
    assert np.allclose(dx2.asnumpy(), ground_truth_dx2)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_sparse_add_grad_int64():
    """
    Feature: test SparseGrad ops
    Description: test int64 value input
    Expectation: Output matching expected values
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    val_grad = np.array([2, 2, 2]).astype(np.int64)
    x1_indices = np.array([[0, 1], [1, 2]]).astype(np.int64)
    x2_indices = np.array([[0, 1], [1, 3]]).astype(np.int64)
    sum_indices = np.array([[0, 1], [1, 2], [1, 3]]).astype(np.int64)
    dx1, dx2 = SparseAddGradNet()(Tensor(val_grad), Tensor(x1_indices), Tensor(x2_indices), Tensor(sum_indices))
    ground_truth_dx1 = np.array([2, 2], np.int64)
    ground_truth_dx2 = np.array([2, 2], np.int64)
    assert np.allclose(dx1.asnumpy(), ground_truth_dx1)
    assert np.allclose(dx2.asnumpy(), ground_truth_dx2)
