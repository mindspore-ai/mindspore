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
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.ops.operations as P
from mindspore.ops.operations.array_ops import ScatterNdMul
import mindspore.context as context
from mindspore.common import dtype as mstype
from mindspore.common import Tensor, Parameter
from mindspore.common.api import _pynative_executor
from mindspore.ops.functional import vmap


func_map = {
    "mul": ScatterNdMul,
    "add": ops.ScatterNdAdd,
    "sub": ops.ScatterNdSub,
    "div": ops.ScatterNdDiv,
}

np_func_map = {
    "mul": lambda a, b: a * b,
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "div": lambda a, b: a / b,
}


class TestScatterNdNet(nn.Cell):
    def __init__(self, func, lock, input_x, indices, updates):
        super(TestScatterNdNet, self).__init__()
        self.scatter_func = func_map.get(func)(use_locking=lock)
        self.input_x = Parameter(input_x, name="input_x")
        self.indices = Parameter(indices, name="indices")
        self.updates = Parameter(updates, name="updates")

    def construct(self):
        self.scatter_func(self.input_x, self.indices, self.updates)
        return self.input_x


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


def scatter_nd_np(func, input_x, indices, updates):
    result = input_x.asnumpy().copy()
    indices_np = indices.asnumpy().copy()
    updates_np = updates.asnumpy().copy()

    f = np_func_map.get(func)

    for idx, _ in np.ndenumerate(np.zeros(indices.shape[:-1])):
        upd_idx = tuple(idx)
        out_idx = tuple(indices_np[upd_idx])
        result[out_idx] = f(result[out_idx], updates_np[upd_idx]).astype(result.dtype)

    return result


def compare_with_numpy(func, lock, input_x, indices, updates):
    expected = scatter_nd_np(func, input_x, indices, updates)

    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    graph_output = TestScatterNdNet(func, lock, input_x, indices, updates)()
    np.testing.assert_array_almost_equal(graph_output.asnumpy(), expected)

    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    pynative_output = TestScatterNdNet(func, lock, input_x, indices, updates)()
    np.testing.assert_array_almost_equal(pynative_output.asnumpy(), expected)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('lock', [True, False])
@pytest.mark.parametrize('func', ['mul', 'sub', 'add', 'div'])
@pytest.mark.parametrize('data_type', [mstype.float32, mstype.float64])
@pytest.mark.parametrize('index_type', [mstype.int32])
def test_scatter_nd_small_float(lock, func, data_type, index_type):
    """
    Feature: ScatterNd* operators.
    Description: test cases for ScatterNd* operator
    Expectation: the result match numpy implementation.
    """
    input_x = Tensor(np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]), data_type)
    indices = Tensor(np.array([[0, 0], [1, 1]]), index_type)
    updates = Tensor(np.array([1.0, 2.2]), data_type)

    compare_with_numpy(func, lock, input_x, indices, updates)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('lock', [True, False])
@pytest.mark.parametrize('func', ['mul', 'sub', 'add', 'div'])
@pytest.mark.parametrize('data_type', [mstype.int8, mstype.int16, mstype.int32, mstype.int64])
@pytest.mark.parametrize('index_type', [mstype.int32])
def test_scatter_nd_small_int(lock, func, data_type, index_type):
    """
    Feature: ScatterNd* operators.
    Description: test cases for ScatterNd* operator
    Expectation: the result match numpy implementation.
    """
    input_x = Tensor(np.array([1, 2, 3, 4, 5, 6, 7, 8]), data_type)
    indices = Tensor(np.array([[4], [3], [1], [7]]), index_type)
    updates = Tensor(np.array([9, 10, 11, 12]), data_type)

    compare_with_numpy(func, lock, input_x, indices, updates)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('lock', [True, False])
@pytest.mark.parametrize('func', ['mul', 'sub', 'add', 'div'])
@pytest.mark.parametrize('data_type', [mstype.int8, mstype.int16, mstype.int32, mstype.int64])
@pytest.mark.parametrize('index_type', [mstype.int32])
def test_scatter_nd_multi_dims(lock, func, data_type, index_type):
    """
    Feature: ScatterNd* operators.
    Description: test cases for ScatterNd* operator
    Expectation: the result match numpy implementation.
    """
    input_x = Tensor(np.ones((4, 4, 4)), data_type)
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

    compare_with_numpy(func, lock, input_x, indices, updates)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('lock', [True, False])
@pytest.mark.parametrize('func', ['mul', 'sub', 'add', 'div'])
@pytest.mark.parametrize('data_type', [mstype.int8, mstype.int16, mstype.int32, mstype.int64])
@pytest.mark.parametrize('index_type', [mstype.int32])
def test_scatter_nd_indices_out_of_range(lock, func, data_type, index_type):
    """
    Feature: ScatterNd* operators.
    Description: test cases for ScatterNd* operator with invalid indices
    Expectation: raise RuntimeError
    """
    input_x = Tensor(np.ones((4, 4, 4)), data_type)
    indices = Tensor(np.array([[0], [4]]), index_type)
    updates = Tensor(
        np.array(
            [
                [[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
                [[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
            ]
        ),
        data_type,
    )

    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    with pytest.raises(RuntimeError):
        _ = TestScatterNdNet(func, lock, input_x, indices, updates)()

    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    with pytest.raises(RuntimeError):
        _ = TestScatterNdNet(func, lock, input_x, indices, updates)()
        _pynative_executor.sync()

@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('lock', [True, False])
@pytest.mark.parametrize('func', ['mul', 'sub', 'add', 'div'])
@pytest.mark.parametrize('data_type', [mstype.int8, mstype.int16, mstype.int32, mstype.int64])
@pytest.mark.parametrize('index_type', [mstype.int32])
def test_scatter_nd_one_value(lock, func, data_type, index_type):
    """
    Feature: ScatterNd* operators.
    Description: test cases for ScatterNd* operator
    Expectation: the result match numpy implementation.
    """
    input_x = Tensor(np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]), data_type)
    indices = Tensor(np.array([[0, 1]]), index_type)
    updates = Tensor(np.array([1.0]), data_type)

    compare_with_numpy(func, lock, input_x, indices, updates)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('lock', [True])
@pytest.mark.parametrize('func', ['mul', 'sub', 'add', 'div'])
@pytest.mark.parametrize('data_type', [mstype.int64])
@pytest.mark.parametrize('index_type', [mstype.int32])
def test_scatter_nd_lock(lock, func, data_type, index_type):
    """
    Feature: ScatterNd* operators.
    Description: test cases for ScatterNd* operator with use_locking is true.
    Expectation: the result match numpy implementation.
    """
    input_x = Tensor(np.ones((5, 4, 4)), data_type)
    indices = Tensor(np.zeros((30, 1)), index_type)
    updates = Tensor(np.random.randint(low=1, high=3, size=(30, 4, 4)), data_type)

    compare_with_numpy(func, lock, input_x, indices, updates)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize("func_name", ["add", "sub"])
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
    np_result = scatter_nd_np(func_name, real_input_x, scatter_indices, updates)
    np.testing.assert_allclose(np_result, ms_result.asnumpy())
    context.set_context(mode=context.PYNATIVE_MODE)
    net = DynamicShapeScatterNet(func_name, input_x)
    real_input_x, ms_result = net(scatter_indices, updates, indices)
    np_result = scatter_nd_np(func_name, real_input_x, scatter_indices, updates)
    np.testing.assert_allclose(np_result, ms_result.asnumpy())


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('data_type', [mstype.int64])
@pytest.mark.parametrize('index_type', [mstype.int32])
def test_scatter_nd_div_division_by_zero(data_type, index_type):
    """
    Feature: Test ScatterNdSub && ScatterNdAdd DyNamicShape.
    Description: The input shape may need to broadcast.
    Expectation: raise ValueError.
    """
    context.set_context(mode=context.GRAPH_MODE)
    np.random.seed(1)
    input_x = Tensor(np.array([[-2, 5, 3], [4, 5, -3]]), data_type)
    indices = Tensor(np.array([[0, 0], [1, 1]]), index_type)
    updates = Tensor(np.array([0, 2]), data_type)

    compare_with_numpy('div', False, input_x, indices, updates)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('func', ['add', 'sub', 'div', 'mul'])
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
        expected[i, :] = scatter_nd_np(func, input_x[i, :], indices[i, :], updates)

    np.testing.assert_array_almost_equal(output.asnumpy(), expected)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('func', ['add', 'sub', 'div', 'mul'])
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
        expected[i, :] = scatter_nd_np(func, input_x[i, :], indices, updates[i, :])

    np.testing.assert_array_almost_equal(output.asnumpy(), expected)
