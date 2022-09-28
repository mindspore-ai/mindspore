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
import mindspore.ops.operations as P
from mindspore import Tensor, Parameter
import mindspore.common.dtype as mstype
import mindspore.ops as ops
from mindspore.ops.operations.array_ops import ScatterNdMul
from mindspore.ops.operations.array_ops import ScatterNdMax
from mindspore.ops.functional import vmap

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

div = P.Div()

func_map = {
    "update": ops.ScatterNdUpdate,
    "add": ops.ScatterNdAdd,
    "sub": ops.ScatterNdSub,
    "mul": ScatterNdMul,
    "div": ops.ScatterNdDiv,
    "max": ScatterNdMax,
    "min": ops.ScatterNdMin,
}

np_func_map = {
    "update": lambda a, b: b,
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "div": lambda a, b: div(Tensor(np.array(a)), Tensor(np.array(b))).asnumpy(),
    "max": np.maximum,
    "min": np.minimum,
}


class TestScatterNdFuncNet(nn.Cell):
    def __init__(self, func, inputx, indices, updates):
        super(TestScatterNdFuncNet, self).__init__()

        self.scatter_func = func_map[func](use_locking=True)
        self.inputx = Parameter(inputx, name="inputx")
        self.indices = Parameter(indices, name="indices")
        self.updates = Parameter(updates, name="updates")

    def construct(self):
        out = self.scatter_func(self.inputx, self.indices, self.updates)
        return out


class DynamicShapeScatterNet(nn.Cell):
    def __init__(self, func, input_x, axis=0):
        super(DynamicShapeScatterNet, self).__init__()
        self.unique = P.Unique()
        self.gather = P.Gather()
        self.scatter_nd_func = func_map.get(func)()
        self.axis = axis
        self.input_x = Parameter(input_x)

    def construct(self, scatter_indices, update, indices):
        unique_indices, _ = self.unique(indices)
        real_input = self.gather(self.input_x, unique_indices, self.axis)
        return real_input, self.scatter_nd_func(self.input_x, scatter_indices, update)


class VmapScatterNet(nn.Cell):
    def __init__(self, func_name):
        super(VmapScatterNet, self).__init__()
        self.scatter_func = func_map.get(func_name)()

    def construct(self, input_x, indices, updates):
        self.scatter_func(input_x, indices, updates)
        return input_x


class VMapNet(nn.Cell):
    def __init__(self, net, input_x, in_axes, out_axes):
        super(VMapNet, self).__init__()
        self.input_x = Parameter(input_x, name="input_x")
        self.net = net
        self.in_axes = in_axes
        self.out_axes = out_axes

    def construct(self, indices, updates):
        return vmap(self.net, self.in_axes, self.out_axes)(self.input_x, indices, updates)


def scatter_nd_func_np(func, inputx, indices, updates):
    result = inputx.asnumpy().copy()
    updates_np = updates.asnumpy()

    f = np_func_map[func]

    for idx, _ in np.ndenumerate(np.zeros(indices.shape[:-1])):
        out_index = indices[idx]
        result[out_index] = f(result[out_index], updates_np[idx]).astype(result.dtype)

    return result


def compare_scatter_nd_func(func, inputx, indices, updates):
    output = TestScatterNdFuncNet(func, inputx, indices, updates)()
    expected = scatter_nd_func_np(func, inputx, indices, updates)
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('func', ['update', 'add', 'sub', 'div', 'mul', 'max', 'min'])
@pytest.mark.parametrize('data_type',
                         [mstype.uint8, mstype.int8, mstype.uint16, mstype.int16, mstype.uint32, mstype.int32,
                          mstype.uint64, mstype.int64, mstype.float16, mstype.float32, mstype.float64])
@pytest.mark.parametrize('index_type', [mstype.int32, mstype.int64])
def test_scatter_nd_func_small(func, data_type, index_type):
    """
    Feature: ALL To ALL
    Description: test cases for small input of ScatterNd* like functions
    Expectation: the result match to numpy implementation
    """
    inputx = Tensor(np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]), data_type)
    indices = Tensor(np.array([[0, 0], [1, 1]]), index_type)
    updates = Tensor(np.array([1.0, 2.2]), data_type)

    compare_scatter_nd_func(func, inputx, indices, updates)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_scatter_nd_func_small_update():
    """
    Feature: ALL To ALL
    Description: test cases for bool input of ScatterNdUpdate
    Expectation: the result match to numpy implementation
    """
    inputx = Tensor(np.array([True, False, True, False, True, True, False, True]), mstype.bool_)
    indices = Tensor(np.array([[False], [True], [False], [True]]), mstype.int32)
    updates = Tensor(np.array([9, 10, 11, 12]), mstype.bool_)

    compare_scatter_nd_func("update", inputx, indices, updates)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('func', ['update', 'add', 'sub', 'div', 'mul', 'max', 'min'])
@pytest.mark.parametrize('data_type',
                         [mstype.uint8, mstype.int8, mstype.uint16, mstype.int16, mstype.uint32, mstype.int32,
                          mstype.uint64, mstype.int64, mstype.float16, mstype.float32, mstype.float64])
@pytest.mark.parametrize('index_type', [mstype.int32, mstype.int64])
def test_scatter_nd_func_multi_dims(func, data_type, index_type):
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

    compare_scatter_nd_func(func, inputx, indices, updates)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('func', ['update', 'add', 'sub', 'div', 'mul', 'max', 'min'])
@pytest.mark.parametrize('data_type',
                         [mstype.uint8, mstype.int8, mstype.uint16, mstype.int16, mstype.uint32, mstype.int32,
                          mstype.uint64, mstype.int64, mstype.float16, mstype.float32, mstype.float64])
@pytest.mark.parametrize('index_type', [mstype.int32, mstype.int64])
def test_scatter_nd_func_one_value(func, data_type, index_type):
    """
    Feature: ALL To ALL
    Description: test cases for one value modification of ScatterNd* like functions
    Expectation: the result match to numpy implementation
    """
    inputx = Tensor(np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]), data_type)
    indices = Tensor(np.array([[0, 1]]), index_type)
    updates = Tensor(np.array([1.0]), data_type)

    compare_scatter_nd_func(func, inputx, indices, updates)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_gpu_training
@pytest.mark.parametrize('data_type', [mstype.int64])
@pytest.mark.parametrize('index_type', [mstype.int64])
def test_scatter_nd_div_division_by_zero(data_type, index_type):
    """
    Feature: Test ScatterNdSub && ScatterNdAdd DyNamicShape.
    Description: The input shape may need to broadcast.
    Expectation: raise ValueError.
    """
    context.set_context(mode=context.GRAPH_MODE)

    inputx = Tensor(np.array([[-2, 5, 3], [4, 5, -3]]), data_type)
    indices = Tensor(np.array([[0, 0], [1, 1]]), index_type)
    updates = Tensor(np.array([0, 2]), data_type)

    compare_scatter_nd_func('div', inputx, indices, updates)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_gpu_training
@pytest.mark.parametrize("func_name", ['update', 'add', 'sub', 'div', 'mul', 'max', 'min'])
def test_scatter_nd_dy_shape(func_name):
    """
    Feature: Test ScatterNdSub && ScatterNdAdd DyNamicShape.
    Description: The input shape may need to broadcast.
    Expectation: match to np benchmark.
    """
    context.set_context(mode=context.GRAPH_MODE)
    np.random.seed(1)

    input_x = Tensor(np.ones((4, 4, 4)).astype(np.float32))
    scatter_indices = Tensor(np.array([[0], [2]]).astype(np.int32))
    updates = Tensor(np.array([[[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
                               [[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]]]).astype(np.float32))
    indices = Tensor(np.array([i for i in range(0, 4)]).astype(np.int32))
    net = DynamicShapeScatterNet(func_name, input_x)
    real_input_x, ms_result = net(scatter_indices, updates, indices)
    np_result = scatter_nd_func_np(func_name, real_input_x, scatter_indices, updates)
    np.testing.assert_allclose(np_result, ms_result.asnumpy())

    context.set_context(mode=context.PYNATIVE_MODE)
    net = DynamicShapeScatterNet(func_name, input_x)
    real_input_x, ms_result = net(scatter_indices, updates, indices)
    np_result = scatter_nd_func_np(func_name, real_input_x, scatter_indices, updates)
    np.testing.assert_allclose(np_result, ms_result.asnumpy())


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('func', ['add', 'sub', 'div', 'min', 'max', 'mul'])
def test_scatter_func_indices_vmap(func):
    """
    Feature: test ScatterNd* vmap.
    Description: in_axes: (0, 0, None).
    Expectation: the result match with numpy result
    """
    context.set_context(mode=context.GRAPH_MODE)

    in_axes = (0, 0, None)
    out_axes = 0
    input_x = Tensor(
        np.array([[[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]], [[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]]).astype(np.float32))
    indices = Tensor(np.array([[[0, 0], [1, 1]], [[0, 0], [1, 1]]]).astype(np.int32))
    updates = Tensor(np.array([1.0, 2.2]).astype(np.float32))

    output = VMapNet(VmapScatterNet(func), input_x, in_axes, out_axes)(indices, updates)
    expected = np.zeros_like(input_x.asnumpy())
    for i in range(input_x.shape[0]):
        expected[i, :] = scatter_nd_func_np(func, input_x[i, :], indices[i, :], updates)

    np.testing.assert_array_almost_equal(output.asnumpy(), expected)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('func', ['add', 'sub', 'div', 'min', 'max', 'mul'])
def test_scatter_func_update_vmap(func):
    """
    Feature: test ScatterNd* vmap.
    Description: in_axes: (0,  None, 0).
    Expectation: the result match with numpy result
    """
    context.set_context(mode=context.GRAPH_MODE)
    in_axes = (0, None, 0)
    out_axes = 0
    input_x = Tensor(
        np.array([[[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]], [[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]]).astype(np.float32))
    indices = Tensor(np.array([[0, 0], [1, 1]]).astype(np.int32))
    updates = Tensor(np.array([[1.0, 2.2], [0.07, 1.23]]).astype(np.float32))

    output = VMapNet(VmapScatterNet(func), input_x, in_axes, out_axes)(indices, updates)
    expected = np.zeros_like(input_x.asnumpy())
    for i in range(input_x.shape[0]):
        expected[i, :] = scatter_nd_func_np(func, input_x[i, :], indices, updates[i, :])

    np.testing.assert_array_almost_equal(output.asnumpy(), expected)
