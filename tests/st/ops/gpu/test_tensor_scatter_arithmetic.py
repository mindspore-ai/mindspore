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
from mindspore.ops import vmap
from mindspore.ops import operations as P

tensor_scatter_func_map = {
    "update": P.TensorScatterUpdate,
    "min": P.TensorScatterMin,
    "max": P.TensorScatterMax,
    "add": P.TensorScatterAdd,
    "sub": P.TensorScatterSub,
    "mul": P.TensorScatterMul,
    "div": P.TensorScatterDiv,

}

np_benchmark_func_map = {
    "update": lambda a, b: b,
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "div": lambda a, b: a / b,
    "min": min,
    "max": max,
}


class TestTensorScatterFuncNet(nn.Cell):
    def __init__(self, func, input_x, indices, updates):
        super(TestTensorScatterFuncNet, self).__init__()

        self.scatter_func = tensor_scatter_func_map.get(func)()
        self.input_x = Tensor(input_x)
        self.indices = Tensor(indices)
        self.updates = Tensor(updates)

    def construct(self):
        out = self.scatter_func(self.input_x, self.indices, self.updates)
        return out


class TestTensorScatterNet(nn.Cell):
    def __init__(self, func_name):
        super(TestTensorScatterNet, self).__init__()
        self.scatter_func = tensor_scatter_func_map.get(func_name)()

    def construct(self, input_x, indices, updates):
        out = self.scatter_func(input_x, indices, updates)
        return out


class DynamicShapeTensorScatterNet(nn.Cell):
    def __init__(self, func_name, axis=0):
        super(DynamicShapeTensorScatterNet, self).__init__()
        self.unique = P.Unique()
        self.gather = P.Gather()
        self.tensor_scatter_func = tensor_scatter_func_map.get(func_name)()
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


def tensor_scatter_np_benchmark(np_func, input_x, indices, updates):
    """
    Feature: benchmark to generate result.
    Description: benchmark function to generate result.
    Expectation: match to tensor scatter binary op.
    """
    result = input_x.copy()
    benchmark_func = np_benchmark_func_map.get(np_func)
    for index, _ in np.ndenumerate(np.zeros(indices.shape[:-1])):
        out_index = tuple(indices[index])
        result[out_index] = benchmark_func(result[out_index], updates[index])
    return result


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('func_name', ["update", "min", "max", "add", "sub", "mul", "div"])
@pytest.mark.parametrize('input_data_type', [np.float16, np.float32, np.float64, np.int8, np.int32])
@pytest.mark.parametrize('index_data_type', [np.int32, np.int64])
def test_tensor_scatter(func_name, input_data_type, index_data_type):
    """
    Feature: test_tensor_scatter
    Description: Test the function of tensor scatter binary op.
    Expectation: match to numpy benchmark.
    """

    # tensor_scatter_div grad do not support int8 and uint8 currently
    # disable int8 and uint8 datatype
    if func_name == "div" and input_data_type == np.int8:
        return

    context.set_context(mode=context.GRAPH_MODE)
    arr_input = np.arange(21).reshape(3, 7).astype(input_data_type)
    arr_indices = np.array([[0, 1], [1, 1], [0, 5], [0, 2], [2, 1]]).astype(index_data_type)
    arr_update = np.array([3.2, 1.1, 5.3, -2.2, -1.0]).astype(input_data_type)
    tensor_scatter_net = TestTensorScatterFuncNet(func_name, arr_input, arr_indices, arr_update)
    out = tensor_scatter_net()
    expected = tensor_scatter_np_benchmark(func_name, arr_input, arr_indices, arr_update)
    np.testing.assert_allclose(out.asnumpy(), expected, rtol=1e-6)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_gpu_training
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
    np_result = tensor_scatter_np_benchmark(func_name, real_input_x.asnumpy(), scatter_indices.asnumpy(),
                                            updates.asnumpy())
    np.testing.assert_allclose(np_result, ms_result.asnumpy(), atol=1e-6, rtol=1e-6)
    context.set_context(mode=context.PYNATIVE_MODE)
    net = DynamicShapeTensorScatterNet(func_name)
    real_input_x, ms_result = net(input_x, scatter_indices, updates, indices)
    np_result = tensor_scatter_np_benchmark(func_name, real_input_x.asnumpy(), scatter_indices.asnumpy(),
                                            updates.asnumpy())
    np.testing.assert_allclose(np_result, ms_result.asnumpy(), atol=1e-6, rtol=1e-6)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
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
    f = np_benchmark_func_map.get(func_name)
    benchmark_output[0][0][0] = f(in_np[0][0][0], 1.0)
    benchmark_output[1][0][0] = f(in_np[1][0][0], 1.0)
    benchmark_output[0][1][1] = f(in_np[0][1][1], 2.2)
    benchmark_output[1][1][1] = f(in_np[1][1][1], 2.2)
    np.testing.assert_allclose(output.asnumpy(), benchmark_output, atol=1e-6, rtol=1e-6)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
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
    f = np_benchmark_func_map.get(func_name)
    benchmark_output[0][0][0] = f(in_np[0][0][0], 1.0)
    benchmark_output[1][0][0] = f(in_np[1][0][0], 0.07)
    benchmark_output[0][1][1] = f(in_np[0][1][1], 2.2)
    benchmark_output[1][1][1] = f(in_np[1][1][1], 1.23)
    np.testing.assert_allclose(output.asnumpy(), benchmark_output, atol=1e-6, rtol=1e-6)
