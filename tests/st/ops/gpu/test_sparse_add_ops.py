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
from mindspore import Tensor
import mindspore as ms
from mindspore.ops.operations.sparse_ops import SparseAdd


class SparseAddNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.sparse_add = SparseAdd()

    def construct(self, a_indices, a_values, a_shape, b_indices, b_values, b_shape, thresh):
        return self.sparse_add(a_indices, a_values, a_shape, b_indices, b_values, b_shape, thresh)


def sparse_add_same_indices(value_type, thresh_type, np_type, thresh_value):
    a_indices = Tensor([[0, 1], [1, 2]], ms.int64)
    a_values = Tensor([1, 2], value_type)
    a_shape = Tensor([3, 4], ms.int64)
    b_indices = Tensor([[0, 1], [1, 2]], ms.int64)
    b_values = Tensor([1, 2], value_type)
    b_shape = Tensor([3, 4], ms.int64)
    thresh = Tensor(thresh_value, thresh_type)
    sparse_add = SparseAddNet()
    sum_indices, sum_value, sum_shape = sparse_add(a_indices, a_values, a_shape, b_indices, b_values, b_shape, thresh)

    ground_truth_indices = np.array([[0, 1], [1, 2]], np.int64)
    ground_truth_value = np.array([2, 4], np_type)
    ground_truth_shape = np.array([3, 4], np.int64)
    assert np.allclose(sum_indices.asnumpy(), ground_truth_indices)
    assert np.allclose(sum_value.asnumpy(), ground_truth_value)
    assert np.allclose(sum_shape.asnumpy(), ground_truth_shape)


def sparse_add_left_same_indices(value_type, thresh_type, np_type, thresh_value):
    a_indices = Tensor([[0, 1], [1, 2]], ms.int64)
    a_values = Tensor([1, 2], value_type)
    a_shape = Tensor([3, 4], ms.int64)
    b_indices = Tensor([[0, 1], [1, 3]], ms.int64)
    b_values = Tensor([1, 2], value_type)
    b_shape = Tensor([3, 4], ms.int64)
    thresh = Tensor(thresh_value, thresh_type)
    sparse_add = SparseAddNet()
    sum_indices, sum_value, sum_shape = sparse_add(a_indices, a_values, a_shape, b_indices, b_values, b_shape, thresh)

    ground_truth_indices = np.array([[0, 1], [1, 2], [1, 3]], np.int64)
    ground_truth_value = np.array([2, 2, 2], np_type)
    ground_truth_shape = np.array([3, 4], np.int64)
    assert np.allclose(sum_indices.asnumpy(), ground_truth_indices)
    assert np.allclose(sum_value.asnumpy(), ground_truth_value)
    assert np.allclose(sum_shape.asnumpy(), ground_truth_shape)


def sparse_add_right_same_indices(value_type, thresh_type, np_type, thresh_value):
    a_indices = Tensor([[0, 1], [1, 3]], ms.int64)
    a_values = Tensor([1, 2], value_type)
    a_shape = Tensor([3, 4], ms.int64)
    b_indices = Tensor([[1, 2], [1, 3]], ms.int64)
    b_values = Tensor([1, 2], value_type)
    b_shape = Tensor([3, 4], ms.int64)
    thresh = Tensor(thresh_value, thresh_type)
    sparse_add = SparseAddNet()
    sum_indices, sum_value, sum_shape = sparse_add(a_indices, a_values, a_shape, b_indices, b_values, b_shape, thresh)

    ground_truth_indices = np.array([[0, 1], [1, 2], [1, 3]], np.int64)
    ground_truth_value = np.array([1, 1, 4], np_type)
    ground_truth_shape = np.array([3, 4], np.int64)
    assert np.allclose(sum_indices.asnumpy(), ground_truth_indices)
    assert np.allclose(sum_value.asnumpy(), ground_truth_value)
    assert np.allclose(sum_shape.asnumpy(), ground_truth_shape)


def sparse_add_no_same_indices(value_type, thresh_type, np_type, thresh_value):
    a_indices = Tensor([[0, 1], [1, 3]], ms.int64)
    a_values = Tensor([1, 2], value_type)
    a_shape = Tensor([3, 4], ms.int64)
    b_indices = Tensor([[1, 2], [2, 2]], ms.int64)
    b_values = Tensor([1, 2], value_type)
    b_shape = Tensor([3, 4], ms.int64)
    thresh = Tensor(thresh_value, thresh_type)
    sparse_add = SparseAddNet()
    sum_indices, sum_value, sum_shape = sparse_add(a_indices, a_values, a_shape, b_indices, b_values, b_shape, thresh)

    ground_truth_indices = np.array([[0, 1], [1, 2], [1, 3], [2, 2]], np.int64)
    ground_truth_value = np.array([1, 1, 2, 2], np_type)
    ground_truth_shape = np.array([3, 4], np.int64)
    assert np.allclose(sum_indices.asnumpy(), ground_truth_indices)
    assert np.allclose(sum_value.asnumpy(), ground_truth_value)
    assert np.allclose(sum_shape.asnumpy(), ground_truth_shape)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_sparse_add_float32():
    """
    Feature: Sparse add ops
    Description: test float32 value input
    Expectation: Output matching expected values
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    value_type = ms.float32
    thresh_type = ms.float32
    np_type = np.float32
    sparse_add_same_indices(value_type, thresh_type, np_type, 0)
    sparse_add_same_indices(value_type, thresh_type, np_type, 1)
    sparse_add_left_same_indices(value_type, thresh_type, np_type, 0)
    sparse_add_left_same_indices(value_type, thresh_type, np_type, 1)
    sparse_add_right_same_indices(value_type, thresh_type, np_type, 0)
    sparse_add_right_same_indices(value_type, thresh_type, np_type, 1)
    sparse_add_no_same_indices(value_type, thresh_type, np_type, 0)
    sparse_add_no_same_indices(value_type, thresh_type, np_type, 1)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_sparse_add_float64():
    """
    Feature: Sparse add ops
    Description: test float64 value input
    Expectation: Output matching expected values
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    value_type = ms.float64
    thresh_type = ms.float64
    np_type = np.float64
    sparse_add_same_indices(value_type, thresh_type, np_type, 0)
    sparse_add_same_indices(value_type, thresh_type, np_type, 1)
    sparse_add_left_same_indices(value_type, thresh_type, np_type, 0)
    sparse_add_left_same_indices(value_type, thresh_type, np_type, 1)
    sparse_add_right_same_indices(value_type, thresh_type, np_type, 0)
    sparse_add_right_same_indices(value_type, thresh_type, np_type, 1)
    sparse_add_no_same_indices(value_type, thresh_type, np_type, 0)
    sparse_add_no_same_indices(value_type, thresh_type, np_type, 1)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_sparse_add_int8():
    """
    Feature: Sparse add ops
    Description: test int8 value input
    Expectation: Output matching expected values
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    value_type = ms.int8
    thresh_type = ms.int8
    np_type = np.int8
    sparse_add_same_indices(value_type, thresh_type, np_type, 0)
    sparse_add_same_indices(value_type, thresh_type, np_type, 1)
    sparse_add_left_same_indices(value_type, thresh_type, np_type, 0)
    sparse_add_left_same_indices(value_type, thresh_type, np_type, 1)
    sparse_add_right_same_indices(value_type, thresh_type, np_type, 0)
    sparse_add_right_same_indices(value_type, thresh_type, np_type, 1)
    sparse_add_no_same_indices(value_type, thresh_type, np_type, 0)
    sparse_add_no_same_indices(value_type, thresh_type, np_type, 1)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_sparse_add_int16():
    """
    Feature: Sparse add ops
    Description: test int16 value input
    Expectation: Output matching expected values
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    value_type = ms.int16
    thresh_type = ms.int16
    np_type = np.int16
    sparse_add_same_indices(value_type, thresh_type, np_type, 0)
    sparse_add_same_indices(value_type, thresh_type, np_type, 1)
    sparse_add_left_same_indices(value_type, thresh_type, np_type, 0)
    sparse_add_left_same_indices(value_type, thresh_type, np_type, 1)
    sparse_add_right_same_indices(value_type, thresh_type, np_type, 0)
    sparse_add_right_same_indices(value_type, thresh_type, np_type, 1)
    sparse_add_no_same_indices(value_type, thresh_type, np_type, 0)
    sparse_add_no_same_indices(value_type, thresh_type, np_type, 1)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_sparse_add_int32():
    """
    Feature: Sparse add ops
    Description: test int32 value input
    Expectation: Output matching expected values
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    value_type = ms.int32
    thresh_type = ms.int32
    np_type = np.int32
    sparse_add_same_indices(value_type, thresh_type, np_type, 0)
    sparse_add_same_indices(value_type, thresh_type, np_type, 1)
    sparse_add_left_same_indices(value_type, thresh_type, np_type, 0)
    sparse_add_left_same_indices(value_type, thresh_type, np_type, 1)
    sparse_add_right_same_indices(value_type, thresh_type, np_type, 0)
    sparse_add_right_same_indices(value_type, thresh_type, np_type, 1)
    sparse_add_no_same_indices(value_type, thresh_type, np_type, 0)
    sparse_add_no_same_indices(value_type, thresh_type, np_type, 1)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_sparse_add_int64():
    """
    Feature: Sparse add ops
    Description: test int64 value input
    Expectation: Output matching expected values
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    value_type = ms.int64
    thresh_type = ms.int64
    np_type = np.int64
    sparse_add_same_indices(value_type, thresh_type, np_type, 0)
    sparse_add_same_indices(value_type, thresh_type, np_type, 1)
    sparse_add_left_same_indices(value_type, thresh_type, np_type, 0)
    sparse_add_left_same_indices(value_type, thresh_type, np_type, 1)
    sparse_add_right_same_indices(value_type, thresh_type, np_type, 0)
    sparse_add_right_same_indices(value_type, thresh_type, np_type, 1)
    sparse_add_no_same_indices(value_type, thresh_type, np_type, 0)
    sparse_add_no_same_indices(value_type, thresh_type, np_type, 1)
