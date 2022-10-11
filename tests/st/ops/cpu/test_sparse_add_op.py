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

context.set_context(mode=context.GRAPH_MODE, device_target='CPU')


class SparseAddNet(nn.Cell):

    def __init__(self):
        super().__init__()
        self.sparse_add = SparseAdd()

    def construct(self, a_indices, a_values, a_shape, b_indices, b_values,
                  b_shape, thresh):
        return self.sparse_add(a_indices, a_values, a_shape, b_indices,
                               b_values, b_shape, thresh)


def sparse_add_same_indices(value_type, thresh_type, np_type, thresh_value):
    a_indices = Tensor([[0, 1], [1, 2]], ms.int64)
    a_values = Tensor([1, 2], value_type)
    a_shape = Tensor([3, 4], ms.int64)
    b_indices = Tensor([[0, 1], [1, 2]], ms.int64)
    b_values = Tensor([1, 2], value_type)
    b_shape = Tensor([3, 4], ms.int64)
    thresh = Tensor(thresh_value, thresh_type)
    sparse_add = SparseAddNet()
    sum_indices, sum_value, sum_shape = sparse_add(a_indices, a_values,
                                                   a_shape, b_indices,
                                                   b_values, b_shape, thresh)

    ground_truth_indices = np.array([[0, 1], [1, 2]], np.int64)
    ground_truth_value = np.array([2, 4], np_type)
    ground_truth_shape = np.array([3, 4], np.int64)
    assert np.allclose(sum_indices.asnumpy(), ground_truth_indices)
    assert np.allclose(sum_value.asnumpy(), ground_truth_value)
    assert np.allclose(sum_shape.asnumpy(), ground_truth_shape)


def sparse_add_left_same_indices(value_type, thresh_type, np_type,
                                 thresh_value):
    a_indices = Tensor([[0, 1], [1, 2]], ms.int64)
    a_values = Tensor([1, 2], value_type)
    a_shape = Tensor([3, 4], ms.int64)
    b_indices = Tensor([[0, 1], [1, 3]], ms.int64)
    b_values = Tensor([1, 2], value_type)
    b_shape = Tensor([3, 4], ms.int64)
    thresh = Tensor(thresh_value, thresh_type)
    sparse_add = SparseAddNet()
    sum_indices, sum_value, sum_shape = sparse_add(a_indices, a_values,
                                                   a_shape, b_indices,
                                                   b_values, b_shape, thresh)

    ground_truth_indices = np.array([[0, 1], [1, 2], [1, 3]], np.int64)
    ground_truth_value = np.array([2, 2, 2], np_type)
    ground_truth_shape = np.array([3, 4], np.int64)
    assert np.allclose(sum_indices.asnumpy(), ground_truth_indices)
    assert np.allclose(sum_value.asnumpy(), ground_truth_value)
    assert np.allclose(sum_shape.asnumpy(), ground_truth_shape)


def sparse_add_right_same_indices(value_type, thresh_type, np_type,
                                  thresh_value):
    a_indices = Tensor([[0, 1], [1, 3]], ms.int64)
    a_values = Tensor([1, 2], value_type)
    a_shape = Tensor([3, 4], ms.int64)
    b_indices = Tensor([[1, 2], [1, 3]], ms.int64)
    b_values = Tensor([1, 2], value_type)
    b_shape = Tensor([3, 4], ms.int64)
    thresh = Tensor(thresh_value, thresh_type)
    sparse_add = SparseAddNet()
    sum_indices, sum_value, sum_shape = sparse_add(a_indices, a_values,
                                                   a_shape, b_indices,
                                                   b_values, b_shape, thresh)

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
    sum_indices, sum_value, sum_shape = sparse_add(a_indices, a_values,
                                                   a_shape, b_indices,
                                                   b_values, b_shape, thresh)

    ground_truth_indices = np.array([[0, 1], [1, 2], [1, 3], [2, 2]], np.int64)
    ground_truth_value = np.array([1, 1, 2, 2], np_type)
    ground_truth_shape = np.array([3, 4], np.int64)
    assert np.allclose(sum_indices.asnumpy(), ground_truth_indices)
    assert np.allclose(sum_value.asnumpy(), ground_truth_value)
    assert np.allclose(sum_shape.asnumpy(), ground_truth_shape)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_sparse_add_dynamic_shape():
    """
    Feature: test SparseAdd op in cpu.
    Description: test the ops in dynamic shape.
    Expectation: expect correct shape result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    net = SparseAddNet()
    indics0_dyn = Tensor(shape=[None, None], dtype=ms.int64)
    values0_dyn = Tensor(shape=[None], dtype=ms.int32)
    shape0_dyn = Tensor(shape=[None], dtype=ms.int64)
    indics1_dyn = Tensor(shape=[None, None], dtype=ms.int64)
    values1_dyn = Tensor(shape=[None], dtype=ms.int32)
    shape1_dyn = Tensor(shape=[None], dtype=ms.int64)
    thres = Tensor(0, dtype=ms.int32)
    net.set_inputs(indics0_dyn, values0_dyn, shape0_dyn, indics1_dyn,
                   values1_dyn, shape1_dyn, thres)

    indics0 = Tensor([[0, 1], [1, 2]], dtype=ms.int64)
    values0 = Tensor([1, 2], dtype=ms.int32)
    shape0 = Tensor([3, 4], dtype=ms.int64)
    indics1 = Tensor([[0, 0], [1, 1]], dtype=ms.int64)
    values1 = Tensor([3, 4], dtype=ms.int32)
    shape1 = Tensor([3, 4], dtype=ms.int64)
    outputs = net(indics0, values0, shape0, indics1, values1, shape1, thres)
    print(outputs)
    sum_indices = outputs[0]
    sum_values = outputs[1]
    sum_shape = outputs[2]
    expect_indices_shape = (4, 2)
    assert sum_indices.asnumpy().shape == expect_indices_shape
    expect_sum_values_shape = (4,)
    assert sum_values.asnumpy().shape == expect_sum_values_shape
    expect_sum_shape_shape = (2,)
    assert sum_shape.asnumpy().shape == expect_sum_shape_shape


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_sparse_add_float32():
    """
    Feature: Sparse add ops
    Description: test float32 value input
    Expectation: Output matching expected values
    """
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


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_sparse_add_float64():
    """
    Feature: Sparse add ops
    Description: test float64 value input
    Expectation: Output matching expected values
    """
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


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_sparse_add_int8():
    """
    Feature: Sparse add ops
    Description: test int8 value input
    Expectation: Output matching expected values
    """
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


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_sparse_add_int16():
    """
    Feature: Sparse add ops
    Description: test int16 value input
    Expectation: Output matching expected values
    """
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


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_sparse_add_int32():
    """
    Feature: Sparse add ops
    Description: test int32 value input
    Expectation: Output matching expected values
    """
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


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_sparse_add_int64():
    """
    Feature: Sparse add ops
    Description: test int64 value input
    Expectation: Output matching expected values
    """
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
