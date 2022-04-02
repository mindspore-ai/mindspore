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
from mindspore import Tensor, Parameter
import mindspore.common.dtype as mstype
import mindspore.ops as ops

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

func_map = {
    "update": ops.ScatterNdUpdate,
    "add": ops.ScatterNdAdd,
    "sub": ops.ScatterNdSub,
    "div": ops.ScatterNdDiv,
}

np_func_map = {
    "update": lambda a, b: b,
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "div": lambda a, b: a / b,
}


class TestScatterNdFuncNet(nn.Cell):
    def __init__(self, func, lock, inputx, indices, updates):
        super(TestScatterNdFuncNet, self).__init__()

        self.scatter_func = func_map[func](use_locking=lock)
        self.inputx = Parameter(inputx, name="inputx")
        self.indices = Parameter(indices, name="indices")
        self.updates = Parameter(updates, name="updates")

    def construct(self):
        out = self.scatter_func(self.inputx, self.indices, self.updates)
        return out


def scatter_nd_func_np(func, inputx, indices, updates):
    result = inputx.asnumpy().copy()
    updates_np = updates.asnumpy()

    f = np_func_map[func]

    for idx, _ in np.ndenumerate(np.zeros(indices.shape[:-1])):
        out_index = indices[idx]
        result[out_index] = f(result[out_index], updates_np[idx])

    return result


def compare_scatter_nd_func(func, lock, inputx, indices, updates):
    output = TestScatterNdFuncNet(func, lock, inputx, indices, updates)()
    expected = scatter_nd_func_np(func, inputx, indices, updates)
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('lock', [True, False])
@pytest.mark.parametrize('func', ['update', 'add', 'sub', 'div'])
@pytest.mark.parametrize('data_type',
                         [mstype.uint8, mstype.int8, mstype.int16, mstype.int32, mstype.float16, mstype.float32,
                          mstype.float64])
@pytest.mark.parametrize('index_type', [mstype.int32])
def test_scatter_nd_func_small(lock, func, data_type, index_type):
    """
    Feature: ALL To ALL
    Description: test cases for small input of ScatterNd* like functions
    Expectation: the result match to numpy implementation
    """
    inputx = Tensor(np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]), data_type)
    indices = Tensor(np.array([[0, 0], [1, 1]]), index_type)
    updates = Tensor(np.array([1.0, 2.2]), data_type)

    compare_scatter_nd_func(func, lock, inputx, indices, updates)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('lock', [True, False])
def test_scatter_nd_func_small_update(lock):
    """
    Feature: ALL To ALL
    Description: test cases for bool input of ScatterNdUpdate
    Expectation: the result match to numpy implementation
    """
    inputx = Tensor(np.array([True, False, True, False, True, True, False, True]), mstype.bool_)
    indices = Tensor(np.array([[False], [True], [False], [True]]), mstype.int32)
    updates = Tensor(np.array([9, 10, 11, 12]), mstype.bool_)

    compare_scatter_nd_func("update", lock, inputx, indices, updates)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('lock', [True, False])
@pytest.mark.parametrize('func', ['update', 'add', 'sub', 'div'])
@pytest.mark.parametrize('data_type',
                         [mstype.uint8, mstype.int8, mstype.int16, mstype.int32, mstype.float16, mstype.float32,
                          mstype.float64])
@pytest.mark.parametrize('index_type', [mstype.int32])
def test_scatter_nd_func_small_int(lock, func, data_type, index_type):
    """
    Feature: ALL To ALL
    Description: test cases for int input of ScatterNd* like functions
    Expectation: the result match to numpy implementation
    """
    inputx = Tensor(np.array([1, 2, 3, 4, 5, 6, 7, 8]), data_type)
    indices = Tensor(np.array([[4], [3], [1], [7]]), index_type)
    updates = Tensor(np.array([9, 10, 11, 12]), data_type)

    compare_scatter_nd_func(func, lock, inputx, indices, updates)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('lock', [True, False])
@pytest.mark.parametrize('func', ['update', 'add', 'sub', 'div'])
@pytest.mark.parametrize('data_type',
                         [mstype.uint8, mstype.int8, mstype.int16, mstype.int32, mstype.float16, mstype.float32,
                          mstype.float64])
@pytest.mark.parametrize('index_type', [mstype.int32])
def test_scatter_nd_func_small_negative(lock, func, data_type, index_type):
    """
    Feature: ALL To ALL
    Description: test cases for negative input of ScatterNd* like functions
    Expectation: the result match to numpy implementation
    """
    inputx = Tensor(np.array([-1, -2, -3, -4, -5, -6, -7, -8]), data_type)
    indices = Tensor(np.array([[4], [3], [1], [7]]), index_type)
    updates = Tensor(np.array([9, -10, 11, -12]), data_type)

    compare_scatter_nd_func(func, lock, inputx, indices, updates)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('lock', [True, False])
@pytest.mark.parametrize('func', ['update', 'add', 'sub', 'div'])
@pytest.mark.parametrize('data_type',
                         [mstype.uint8, mstype.int8, mstype.int16, mstype.int32, mstype.float16, mstype.float32,
                          mstype.float64])
@pytest.mark.parametrize('index_type', [mstype.int32])
def test_scatter_nd_func_multi_dims(lock, func, data_type, index_type):
    """
    Feature: ALL To ALL
    Description: test cases for multi-dims input of ScatterNd* like functions
    Expectation: the result match to numpy implementation
    """
    inputx = Tensor(np.zeros((4, 4, 4)), data_type)
    indices = Tensor(np.array([[0], [2]]), index_type)
    updates = Tensor(
        np.array(
            [
                [[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
                [[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
            ]
        ),
        data_type,
    )

    compare_scatter_nd_func(func, lock, inputx, indices, updates)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('lock', [True, False])
@pytest.mark.parametrize('func', ['update', 'add', 'sub', 'div'])
@pytest.mark.parametrize('data_type',
                         [mstype.uint8, mstype.int8, mstype.int16, mstype.int32, mstype.float16, mstype.float32,
                          mstype.float64])
@pytest.mark.parametrize('index_type', [mstype.int32])
def test_scatter_nd_func_one_value(lock, func, data_type, index_type):
    """
    Feature: ALL To ALL
    Description: test cases for one value modification of ScatterNd* like functions
    Expectation: the result match to numpy implementation
    """
    inputx = Tensor(np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]), data_type)
    indices = Tensor(np.array([[0, 1]]), index_type)
    updates = Tensor(np.array([1.0]), data_type)

    compare_scatter_nd_func(func, lock, inputx, indices, updates)
