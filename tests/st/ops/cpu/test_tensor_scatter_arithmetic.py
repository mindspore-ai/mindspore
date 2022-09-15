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

import pytest
import numpy as np
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.context as context
from mindspore.ops import functional as F
from mindspore.ops import vmap
from mindspore.common import dtype as mstype
from mindspore.common import Tensor
from mindspore.common import Parameter

func_map = {
    "add": ops.TensorScatterAdd,
    "sub": ops.TensorScatterSub,
    "div": ops.TensorScatterDiv,
    "max": ops.TensorScatterMax,
    "min": ops.TensorScatterMin,
    "mul": ops.TensorScatterMul,
}

function_func_map = {
    "div": F.tensor_scatter_div,
}

np_func_map = {
    "mul": lambda a, b: a * b,
    "div": lambda a, b: a / b,
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "max": np.maximum,
    "min": np.minimum,
}


class TestTensorScatterArithmeticNet(nn.Cell):
    def __init__(self, func, input_x, indices, updates):
        super(TestTensorScatterArithmeticNet, self).__init__()
        self.scatter_func = func_map.get(func)()
        self.input_x = Parameter(input_x, name="input_x")
        self.indices = Parameter(indices, name="indices")
        self.updates = Parameter(updates, name="updates")

    def construct(self):
        output = self.scatter_func(self.input_x, self.indices, self.updates)
        return output


class TestTensorScatterNet(nn.Cell):
    def __init__(self, func_name):
        super(TestTensorScatterNet, self).__init__()
        self.scatter_func = func_map.get(func_name)()

    def construct(self, input_x, indices, updates):
        out = self.scatter_func(input_x, indices, updates)
        return out


class DynamicShapeTensorScatterNet(nn.Cell):
    def __init__(self, func_name, axis=0):
        super(DynamicShapeTensorScatterNet, self).__init__()
        self.unique = ops.Unique()
        self.gather = ops.Gather()
        self.tensor_scatter_func = func_map.get(func_name)()
        self.axis = axis

    def construct(self, input_x, scatter_indices, update, indices):
        unique_indices, _ = self.unique(indices)
        # Only Dynamic input_x.
        real_input = self.gather(input_x, unique_indices, self.axis)
        return real_input, self.tensor_scatter_func(real_input, scatter_indices, update)


class VMapNet(nn.Cell):
    def __init__(self, net, in_axes, out_axes):
        super(VMapNet, self).__init__()
        self.net = net
        self.in_axes = in_axes
        self.out_axes = out_axes

    def construct(self, input_x, indices, updates):
        return vmap(self.net, self.in_axes, self.out_axes)(input_x, indices, updates)


def tensor_scatter_np(func, input_x, indices, updates):
    result = input_x.asnumpy().copy()
    indices_np = indices.asnumpy().copy()
    updates_np = updates.asnumpy().copy()

    f = np_func_map.get(func)

    for idx, _ in np.ndenumerate(np.zeros(indices.shape[:-1])):
        upd_idx = tuple(idx)
        out_idx = tuple(indices_np[upd_idx])
        result[out_idx] = f(result[out_idx], updates_np[upd_idx])

    return result


def compare_with_numpy(func, input_x, indices, updates):
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    graph_output = TestTensorScatterArithmeticNet(func, input_x, indices, updates)()
    expected = tensor_scatter_np(func, input_x, indices, updates)
    np.testing.assert_array_almost_equal(graph_output.asnumpy(), expected)

    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    pynative_output = TestTensorScatterArithmeticNet(func, input_x, indices, updates)()
    np.testing.assert_array_almost_equal(pynative_output.asnumpy(), expected)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('func', ['add', 'sub', 'div', 'mul', 'max', 'min'])
@pytest.mark.parametrize('data_type', [mstype.float32, mstype.float64])
@pytest.mark.parametrize('index_type', [mstype.int32])
def test_tensor_scatter_arithmetic_small_float(func, data_type, index_type):
    """
    Feature: TensorScatter* operators.
    Description: test cases for TensorScatter* operator
    Expectation: the result match numpy implementation.
    """
    input_x = Tensor(np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]), data_type)
    indices = Tensor(np.array([[0, 0], [1, 1]]), index_type)
    updates = Tensor(np.array([1.0, 2.2]), data_type)

    compare_with_numpy(func, input_x, indices, updates)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('func', ['add', 'sub', 'div', 'mul', 'max', 'min'])
@pytest.mark.parametrize('data_type', [mstype.int8, mstype.int16, mstype.int32, mstype.int64])
@pytest.mark.parametrize('index_type', [mstype.int32])
def test_tensor_scatter_arithmetic_small_int(func, data_type, index_type):
    """
    Feature: TensorScatter* operators.
    Description: test cases for TensorScatter* operator
    Expectation: the result match numpy implementation.
    """

    # tensor_scatter_div grad do not support int8 and uint8 currently
    # disable int8 and uint8 datatype
    if func == 'div' and data_type == mstype.int8:
        return

    input_x = Tensor(np.array([1, 2, 3, 4, 5, 6, 7, 8]), data_type)
    indices = Tensor(np.array([[4], [3], [1], [7]]), index_type)
    updates = Tensor(np.array([9, 10, 11, 12]), data_type)

    compare_with_numpy(func, input_x, indices, updates)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('func', ['add', 'sub', 'div', 'mul', 'max', 'min'])
@pytest.mark.parametrize('data_type', [mstype.int8, mstype.int16, mstype.int32, mstype.int64])
@pytest.mark.parametrize('index_type', [mstype.int32])
def test_tensor_scatter_arithmetic_multi_dims(func, data_type, index_type):
    """
    Feature: TensorScatter* operators.
    Description: test cases for TensorScatter* operator
    Expectation: the result match numpy implementation.
    """

    # tensor_scatter_div grad do not support int8 and uint8 currently
    # disable int8 and uint8 datatype
    if func == 'div' and data_type == mstype.int8:
        return

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

    compare_with_numpy(func, input_x, indices, updates)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('func', ['add', 'sub', 'div', 'mul', 'max', 'min'])
@pytest.mark.parametrize('data_type', [mstype.int8, mstype.int16, mstype.int32, mstype.int64])
@pytest.mark.parametrize('index_type', [mstype.int32])
def test_tensor_scatter_arithmetic_one_value(func, data_type, index_type):
    """
    Feature: TensorScatter* operators.
    Description: test cases for TensorScatter* operator
    Expectation: the result match numpy implementation.
    """

    # tensor_scatter_div grad do not support int8 and uint8 currently
    # disable int8 and uint8 datatype
    if func == 'div' and data_type == mstype.int8:
        return

    input_x = Tensor(np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]), data_type)
    indices = Tensor(np.array([[0, 1]]), index_type)
    updates = Tensor(np.array([1.0]), data_type)

    compare_with_numpy(func, input_x, indices, updates)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('func', ['add', 'sub', 'div', 'mul', 'max', 'min'])
@pytest.mark.parametrize('data_type', [mstype.int8])
@pytest.mark.parametrize('index_type', [mstype.int32])
def test_tensor_scatter_arithmetic_dim_check(func, data_type, index_type):
    """
    Feature: TensorScatter* operators.
    Description: test cases for invalid input.
    Expectation: raise ValueError.
    """

    # tensor_scatter_div grad do not support int8 and uint8 currently
    # disable int8 and uint8 datatype
    if func == 'div' and data_type == mstype.int8:
        return

    input_x = Tensor(np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]), data_type)
    indices = Tensor(np.array([[0, 1, 2]]), index_type)
    updates = Tensor(np.array([1.0]), data_type)

    with pytest.raises(ValueError):
        compare_with_numpy(func, input_x, indices, updates)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('func', ['add', 'sub', 'div', 'mul', 'max', 'min'])
@pytest.mark.parametrize('data_type', [mstype.int8, mstype.int16, mstype.int32, mstype.int64])
@pytest.mark.parametrize('index_type', [mstype.int8, mstype.int16])
def test_tensor_scatter_arithmetic_type_check(func, data_type, index_type):
    """
    Feature: TensorScatter* operators.
    Description: test cases for invalid input.
    Expectation: raise TypeError.
    """

    # tensor_scatter_div grad do not support int8 and uint8 currently
    # disable int8 and uint8 datatype
    if func == 'div' and data_type == mstype.int8:
        return

    input_x = Tensor(np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]), data_type)
    indices = Tensor(np.array([[0, 1]]), index_type)
    updates = Tensor(np.array([1.0]), data_type)

    with pytest.raises(TypeError):
        compare_with_numpy(func, input_x, indices, updates)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('func', ['add', 'sub', 'max', 'min'])
@pytest.mark.parametrize('data_type', [mstype.int32])
@pytest.mark.parametrize('index_type', [mstype.int32])
def test_tensor_scatter_arithmetic_indices_check(func, data_type, index_type):
    """
    Feature: TensorScatter* operators.
    Description: test cases for invalid indices.
    Expectation: raise RuntimeError.
    """
    input_x = Tensor(np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]), data_type)
    indices = Tensor(np.array([[10, 10]]), index_type)
    updates = Tensor(np.array([1.0]), data_type)

    with pytest.raises(RuntimeError):
        compare_with_numpy(func, input_x, indices, updates)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('func', ['div'])
@pytest.mark.parametrize('data_type', [mstype.float32])
@pytest.mark.parametrize('index_type', [mstype.int32])
def test_tensor_scatter_arithmetic_tensor_func_check(func, data_type, index_type):
    """
    Feature: TensorScatter* tensor operators.
    Description: test cases for invalid input.
    Expectation: raise TypeError.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

    input_x = Tensor(np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]), data_type)
    indices = Tensor(np.array([[0, 1]]), index_type)
    updates = Tensor(np.array([1.0]), data_type)
    expected = tensor_scatter_np(func, input_x, indices, updates)

    if func == 'div':
        output = input_x.scatter_div(indices, updates)

    np.testing.assert_allclose(output.asnumpy(), expected, rtol=1e-6)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('func', ['div'])
@pytest.mark.parametrize('data_type', [mstype.float32])
@pytest.mark.parametrize('index_type', [mstype.int32])
def test_tensor_scatter_arithmetic_functional_func_check(func, data_type, index_type):
    """
    Feature: TensorScatter* functional operators.
    Description: test cases for invalid input.
    Expectation: raise TypeError.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

    input_x = Tensor(np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]), data_type)
    indices = Tensor(np.array([[0, 1]]), index_type)
    updates = Tensor(np.array([1.0]), data_type)
    expected = tensor_scatter_np(func, input_x, indices, updates)
    output = function_func_map.get(func)(input_x, indices, updates)

    np.testing.assert_allclose(output.asnumpy(), expected, rtol=1e-6)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.parametrize("func_name", ["mul", "div"])
def test_scatter_nd_dy_shape(func_name):
    """
    Feature: Test TensorScatterOp DyNamicShape.
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
    net = DynamicShapeTensorScatterNet(func_name)
    real_input_x, ms_result = net(input_x, scatter_indices, updates, indices)
    np_result = tensor_scatter_np(func_name, real_input_x, scatter_indices, updates)
    np.testing.assert_allclose(np_result, ms_result.asnumpy(), atol=1e-6, rtol=1e-6)
    context.set_context(mode=context.PYNATIVE_MODE)
    net = DynamicShapeTensorScatterNet(func_name)
    real_input_x, ms_result = net(input_x, scatter_indices, updates, indices)
    np_result = tensor_scatter_np(func_name, real_input_x, scatter_indices, updates)
    np.testing.assert_allclose(np_result, ms_result.asnumpy(), atol=1e-6, rtol=1e-6)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize("func_name", ["mul", "div"])
def test_tensor_scatter_mul_func_indices_vmap(func_name):
    """
    Feature: test TensorScatterOp vmap.
    Description: in_axes: (0, 0, None).
    Expectation: the result match with numpy result
    """
    context.set_context(mode=context.GRAPH_MODE)
    in_axes = (0, 0, None)
    out_axes = 0
    in_np = np.array([[[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]], [[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]]).astype(np.float32)
    input_x = Tensor(in_np)
    indices = Tensor(np.array([[[0, 0], [1, 1]], [[0, 0], [1, 1]]]).astype(np.int32))
    updates = Tensor(np.array([1.0, 2.2]).astype(np.float32))
    output = VMapNet(TestTensorScatterNet(func_name), in_axes, out_axes)(input_x, indices, updates)
    benchmark_output = in_np
    f = np_func_map.get(func_name)
    benchmark_output[0][0][0] = f(in_np[0][0][0], 1.0)
    benchmark_output[1][0][0] = f(in_np[1][0][0], 1.0)
    benchmark_output[0][1][1] = f(in_np[0][1][1], 2.2)
    benchmark_output[1][1][1] = f(in_np[1][1][1], 2.2)
    np.testing.assert_allclose(output.asnumpy(), benchmark_output, atol=1e-6, rtol=1e-6)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize("func_name", ["mul", "div"])
def test_scatter_func_update_vmap(func_name):
    """
    Feature: test TensorScatterOp vmap.
    Description: in_axes: (0,  None, 0).
    Expectation: the result match with numpy result
    """
    context.set_context(mode=context.GRAPH_MODE)
    in_axes = (0, None, 0)
    out_axes = 0
    in_np = np.array([[[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]], [[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]]).astype(np.float32)
    input_x = Tensor(in_np)
    indices = Tensor(np.array([[0, 0], [1, 1]]).astype(np.int32))
    updates = Tensor(np.array([[1.0, 2.2], [0.07, 1.23]]).astype(np.float32))
    output = VMapNet(TestTensorScatterNet(func_name), in_axes, out_axes)(input_x, indices, updates)
    benchmark_output = in_np
    f = np_func_map.get(func_name)
    benchmark_output[0][0][0] = f(in_np[0][0][0], 1.0)
    benchmark_output[1][0][0] = f(in_np[1][0][0], 0.07)
    benchmark_output[0][1][1] = f(in_np[0][1][1], 2.2)
    benchmark_output[1][1][1] = f(in_np[1][1][1], 1.23)
    np.testing.assert_allclose(output.asnumpy(), benchmark_output, atol=1e-6, rtol=1e-6)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.parametrize("func_name", ["add", "max", "min", "sub"])
def test_tensor_scatter_dy_shape(func_name):
    """
    Feature: op dynamic shape
    Description: set input_shape None and input real tensor
    Expectation: success
    """

    context.set_context(mode=context.GRAPH_MODE)
    np.random.seed(1)

    input_x = Parameter(Tensor(np.ones((4, 4, 4)).astype(np.float32)))
    scatter_indices = Tensor(np.array([[0], [2]]).astype(np.int32))
    updates = Tensor(np.array([[[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
                               [[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]]]).astype(np.float32))
    input_x_dyn = Tensor(shape=[4, None, None], dtype=input_x.dtype)
    indice_dyn = Tensor(shape=[None, None], dtype=scatter_indices.dtype)
    update_dyn = Tensor(shape=[None, None, None], dtype=updates.dtype)
    net = TestTensorScatterNet(func_name)
    net.set_inputs(input_x_dyn, indice_dyn, update_dyn)
    ms_result = net(input_x, scatter_indices, updates)
    np_result = tensor_scatter_np(func_name, input_x, scatter_indices, updates)
    assert ms_result.asnumpy().shape == np_result.shape
