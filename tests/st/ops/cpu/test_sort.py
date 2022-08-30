# Copyright 2021 Huawei Technologies Co., Ltd
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
from mindspore.ops import operations as P

class SortNet(nn.Cell):
    def __init__(self, axis, descending):
        super(SortNet, self).__init__()
        self.sort = P.Sort(axis, descending)

    def construct(self, x):
        return self.sort(x)


def sort_1d(descending, nptype):
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

    x_numpy = np.array([1, -2, 3, 4]).astype(nptype)
    x = Tensor(x_numpy)
    sort_net = SortNet(0, descending)
    output, indices = sort_net(x)

    expected_output = np.sort(x_numpy, 0)
    expected_indices = np.array([1, 0, 2, 3])
    if descending:
        expected_output = expected_output[::-1]
        expected_indices = expected_indices[::-1]

    np.testing.assert_array_equal(output.asnumpy(), expected_output)
    np.testing.assert_array_equal(indices.asnumpy(), expected_indices)

def sort_3d(descending, nptype):
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

    x_numpy = np.array([[[1, 2, 3, 4],
                         [8, 7, 2, 0],
                         [9, 4, 1, 8]],
                        [[5, 4, 1, 8],
                         [2, 9, 0, 7],
                         [6, 1, 7, 4]]]).astype(nptype)
    x = Tensor(x_numpy)

    axis = -1
    sort_net = SortNet(axis, descending)
    output, indices = sort_net(x)

    expected_output = np.sort(x_numpy, axis)
    expected_indices = np.array([[[0, 1, 2, 3],
                                  [3, 2, 1, 0],
                                  [2, 1, 3, 0]],
                                 [[2, 1, 0, 3],
                                  [2, 0, 3, 1],
                                  [1, 3, 0, 2]]])
    if descending:
        expected_output = expected_output[:, :, ::-1]
        expected_indices = expected_indices[:, :, ::-1]

    np.testing.assert_array_equal(output.asnumpy(), expected_output)
    np.testing.assert_array_equal(indices.asnumpy(), expected_indices)

    axis = 1
    sort_net = SortNet(axis, descending)
    output, indices = sort_net(x)

    expected_output = np.sort(x_numpy, axis)
    expected_indices = np.array([[[0, 0, 2, 1],
                                  [1, 2, 1, 0],
                                  [2, 1, 0, 2]],
                                 [[1, 2, 1, 2],
                                  [0, 0, 0, 1],
                                  [2, 1, 2, 0]]])
    if descending:
        expected_output = expected_output[:, ::-1, :]
        expected_indices = expected_indices[:, ::-1, :]

    np.testing.assert_array_equal(output.asnumpy(), expected_output)
    np.testing.assert_array_equal(indices.asnumpy(), expected_indices)

    axis = -3
    sort_net = SortNet(axis, descending)
    output, indices = sort_net(x)

    expected_output = np.sort(x_numpy, axis)
    expected_indices = np.array([[[0, 0, 1, 0],
                                  [1, 0, 1, 0],
                                  [1, 1, 0, 1]],
                                 [[1, 1, 0, 1],
                                  [0, 1, 0, 1],
                                  [0, 0, 1, 0]]])
    if descending:
        expected_output = expected_output[::-1, :, :]
        expected_indices = expected_indices[::-1, :, :]

    np.testing.assert_array_equal(output.asnumpy(), expected_output)
    np.testing.assert_array_equal(indices.asnumpy(), expected_indices)


def dynamic_sort_3d(descending, nptype):
    """
    Feature: test sort dynamic function interface.
    Description: test interface.
    Expectation: the result match with numpy result
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

    x_numpy = np.array([[[1, 2, 3, 4],
                         [8, 7, 2, 0],
                         [9, 4, 1, 8]],
                        [[5, 4, 1, 8],
                         [2, 9, 0, 7],
                         [6, 1, 7, 4]]]).astype(nptype)
    x = Tensor(x_numpy)

    axis = -1
    sort_net = SortNet(axis, descending)

    dy_shape = [None for _ in x_numpy.shape]
    input_dyn = Tensor(shape=dy_shape, dtype=x.dtype)
    sort_net.set_inputs(input_dyn)
    output, indices = sort_net(x)

    expected_output = np.sort(x_numpy, axis)
    expected_indices = np.array([[[0, 1, 2, 3],
                                  [3, 2, 1, 0],
                                  [2, 1, 3, 0]],
                                 [[2, 1, 0, 3],
                                  [2, 0, 3, 1],
                                  [1, 3, 0, 2]]])
    if descending:
        expected_output = expected_output[:, :, ::-1]
        expected_indices = expected_indices[:, :, ::-1]

    np.testing.assert_array_equal(output.asnumpy(), expected_output)
    np.testing.assert_array_equal(indices.asnumpy(), expected_indices)

    axis = 1
    sort_net = SortNet(axis, descending)
    dy_shape = [None for _ in x_numpy.shape]
    input_dyn = Tensor(shape=dy_shape, dtype=x.dtype)
    sort_net.set_inputs(input_dyn)
    output, indices = sort_net(x)

    expected_output = np.sort(x_numpy, axis)
    expected_indices = np.array([[[0, 0, 2, 1],
                                  [1, 2, 1, 0],
                                  [2, 1, 0, 2]],
                                 [[1, 2, 1, 2],
                                  [0, 0, 0, 1],
                                  [2, 1, 2, 0]]])
    if descending:
        expected_output = expected_output[:, ::-1, :]
        expected_indices = expected_indices[:, ::-1, :]

    np.testing.assert_array_equal(output.asnumpy(), expected_output)
    np.testing.assert_array_equal(indices.asnumpy(), expected_indices)

    axis = -3
    sort_net = SortNet(axis, descending)
    dy_shape = [None for _ in x_numpy.shape]
    input_dyn = Tensor(shape=dy_shape, dtype=x.dtype)
    sort_net.set_inputs(input_dyn)
    output, indices = sort_net(x)

    expected_output = np.sort(x_numpy, axis)
    expected_indices = np.array([[[0, 0, 1, 0],
                                  [1, 0, 1, 0],
                                  [1, 1, 0, 1]],
                                 [[1, 1, 0, 1],
                                  [0, 1, 0, 1],
                                  [0, 0, 1, 0]]])
    if descending:
        expected_output = expected_output[::-1, :, :]
        expected_indices = expected_indices[::-1, :, :]

    np.testing.assert_array_equal(output.asnumpy(), expected_output)
    np.testing.assert_array_equal(indices.asnumpy(), expected_indices)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_sort1d_float16():
    sort_1d(False, np.float16)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_sort1d_descending_float16():
    sort_1d(True, np.float16)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_sort1d_float32():
    sort_1d(False, np.float32)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_sort1d_descending_float32():
    sort_1d(True, np.float32)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_sort3d_float16():
    sort_3d(False, np.float16)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_sort3d_descending_float16():
    sort_3d(True, np.float16)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_sort3d_float32():
    sort_3d(False, np.float32)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_sort3d_descending_float32():
    sort_3d(True, np.float32)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_cpu_dynamic_sort3d_descending_float32():
    """
    Feature: test cpu sort dynamic function interface.
    Description: test interface.
    Expectation: the result match with numpy result
    """
    dynamic_sort_3d(True, np.float32)
