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
from mindspore.ops import operations as P

tensor_scatter_func_map = {
    "update": P.TensorScatterUpdate,
    "min": P.TensorScatterMin,
    "max": P.TensorScatterMax,
    "add": P.TensorScatterAdd,
    "sub": P.TensorScatterSub,
    "mul": P.TensorScatterMul,

}

np_benchmark_func_map = {
    "update": lambda a, b: b,
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
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


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('func_name', ["update", "min", "max", "add", "sub", "mul"])
@pytest.mark.parametrize('input_data_type', [np.float16, np.float32, np.float64, np.int8, np.int32])
@pytest.mark.parametrize('index_data_type', [np.int32, np.int64])
def test_tensor_scatter(func_name, input_data_type, index_data_type):
    """
    Feature: test_tensor_scatter
    Description: Test the function of tensor scatter binary op.
    Expectation: match to numpy benchmark.
    """
    context.set_context(mode=context.GRAPH_MODE)
    arr_input = np.arange(21).reshape(3, 7).astype(input_data_type)
    arr_indices = np.array([[0, 1], [1, 1], [0, 5], [0, 2], [2, 1]]).astype(index_data_type)
    arr_update = np.array([3.2, 1.1, 5.3, -2.2, -1.0]).astype(input_data_type)
    tensor_scatter_net = TestTensorScatterFuncNet(func_name, arr_input, arr_indices, arr_update)
    out = tensor_scatter_net()
    expected = tensor_scatter_np_benchmark(func_name, arr_input, arr_indices, arr_update)
    np.testing.assert_allclose(out.asnumpy(), expected, rtol=1e-6)
