# Copyright 2022-2025 Huawei Technologies Co., Ltd
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
import mindspore.ops as ops
from mindspore.common import Tensor, Parameter
from mindspore.common import dtype as mstype
from mindspore.ops.functional import vmap

import tests.st.utils.test_utils as test_utils
from tests.mark_utils import arg_mark
from tests.st.ops.test_tools.test_op import TEST_OP

op_map = {
    "add": ops.TensorScatterAdd,
    "sub": ops.TensorScatterSub,
    "max": ops.TensorScatterMax,
    "min": ops.TensorScatterMin,
    "div": ops.TensorScatterDiv,
}

func_map = {
    "add": ops.tensor_scatter_add,
    "sub": ops.tensor_scatter_sub,
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
        self.scatter_func = op_map.get(func)()
        self.input_x = Parameter(input_x, name="input_x")
        self.indices = Parameter(indices, name="indices")
        self.updates = Parameter(updates, name="updates")

    def construct(self):
        output = self.scatter_func(self.input_x, self.indices, self.updates)
        return output


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
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    graph_output = TestTensorScatterArithmeticNet(func, input_x, indices, updates)()
    expected = tensor_scatter_np(func, input_x, indices, updates)
    np.testing.assert_array_almost_equal(graph_output.asnumpy(), expected)

    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    pynative_output = TestTensorScatterArithmeticNet(func, input_x, indices, updates)()
    np.testing.assert_array_almost_equal(pynative_output.asnumpy(), expected)


@test_utils.run_with_cell
def tensor_scatter_add_forward_func(input_x, indices, updates):
    return ops.tensor_scatter_add(input_x, indices, updates)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential', mem_peak=1024.02)
@pytest.mark.parametrize('func', ['add', 'sub'])
@pytest.mark.parametrize('data_type', [mstype.float16, mstype.float32])
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

@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential', mem_peak=1024.02)
@pytest.mark.parametrize('func', ['add', 'sub'])
@pytest.mark.parametrize('data_type', [mstype.int32])
@pytest.mark.parametrize('index_type', [mstype.int32])
def test_tensor_scatter_arithmetic_small_int(func, data_type, index_type):
    """
    Feature: TensorScatter* operators.
    Description: test cases for TensorScatter* operator
    Expectation: the result match numpy implementation.
    """
    input_x = Tensor(np.array([5, 6, 7, 8, 9, 10, 11, 12]), data_type)
    indices = Tensor(np.array([[4], [3], [1], [7]]), index_type)
    updates = Tensor(np.array([1, 2, 3, 4]), data_type)

    compare_with_numpy(func, input_x, indices, updates)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential',
        mem_peak=1024.10)
@pytest.mark.parametrize('func', ['add', 'sub'])
@pytest.mark.parametrize('data_type', [mstype.int32, mstype.float32])
@pytest.mark.parametrize('index_type', [mstype.int32])
def test_tensor_scatter_arithmetic_multi_dims(func, data_type, index_type):
    """
    Feature: TensorScatter* operators.
    Description: test cases for TensorScatter* operator
    Expectation: the result match numpy implementation.
    """
    input_x = Tensor(np.ones((4, 4, 4)) * 10, data_type)
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


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=1024.01)
@pytest.mark.parametrize('func', ['add', 'sub'])
@pytest.mark.parametrize('data_type', [mstype.float32])
@pytest.mark.parametrize('index_type', [mstype.int32])
def test_tensor_scatter_arithmetic_function_op(func, data_type, index_type):
    """
    Feature: TensorScatter* functional operators.
    Description: test cases for ops.tensor_scatter_* api
    Expectation: the result match numpy implementation.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")

    input_x = Tensor(np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]), data_type)
    indices = Tensor(np.array([[0, 1]]), index_type)
    updates = Tensor(np.array([1.0]), data_type)
    expected = tensor_scatter_np(func, input_x, indices, updates)
    output = func_map.get(func)(input_x, indices, updates)

    np.testing.assert_allclose(output.asnumpy(), expected, rtol=1e-6)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential', mem_peak=1024.01)
@pytest.mark.parametrize('func', ['add', 'sub'])
@pytest.mark.parametrize('data_type', [mstype.float32])
@pytest.mark.parametrize('index_type', [mstype.int32])
def test_tensor_scatter_arithmetic_tensor_op(func, data_type, index_type):
    """
    Feature: TensorScatter* tensor operators.
    Description: test cases for tensor.tensor_scatter_* api
    Expectation: the result match numpy implementation.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")

    input_x = Tensor(np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]), data_type)
    indices = Tensor(np.array([[0, 1]]), index_type)
    updates = Tensor(np.array([1.0]), data_type)
    expected = tensor_scatter_np(func, input_x, indices, updates)

    if func == 'add':
        output = input_x.scatter_add(indices, updates)
    elif func == 'sub':
        output = input_x.scatter_sub(indices, updates)

    np.testing.assert_allclose(output.asnumpy(), expected, rtol=1e-6)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential', mem_peak=3072.01)
@pytest.mark.parametrize('data_type', [mstype.float32])
@pytest.mark.parametrize('index_type', [mstype.int32])
def test_tensor_scatter_add_dynamic(data_type, index_type):
    """
    Feature: Test tensor_scatter_add with dynamic shape.
    Description: call ops.tensor_scatter_add with valid input and index.
    Expectation: return the correct value.
    """
    input_x1 = Tensor(np.random.randn(8, 8, 8, 16), data_type)
    input_x2 = Tensor(np.random.randn(8, 8, 8, 1, 2), data_type)

    indices1 = Tensor(np.random.randint(8, size=(8, 2)), index_type)
    indices2 = Tensor(np.random.randint(8, size=(8, 3, 2)), index_type)

    updates1 = Tensor(np.random.randn(8, 8, 16), data_type)
    updates2 = Tensor(np.random.randn(8, 3, 8, 1, 2), data_type)

    TEST_OP(tensor_scatter_add_forward_func,
            [[input_x1, indices1, updates1], [input_x2, indices2, updates2]],
            disable_case=['EmptyTensor', 'ScalarTensor'],
            case_config={'deterministic_use_origin_inputs': True})


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=1040.01)
@pytest.mark.parametrize('func', ['add', 'sub'])
@pytest.mark.parametrize('data_type', [mstype.float32])
@pytest.mark.parametrize('index_type', [mstype.int32])
def test_tensor_scatter_arithmetic_vmap(func, data_type, index_type):
    """
    Feature: TensorScatter* tensor operators.
    Description: test cases for tensor.tensor_scatter_* api
    Expectation: the result match numpy implementation.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")

    input_x = Tensor(np.random.randn(8, 8, 8, 16), data_type)
    indices = Tensor(np.random.randint(8, size=(8, 8, 8, 1)), index_type)
    updates = Tensor(np.random.randn(8, 8, 8, 16), data_type)

    class TensorScatterOp(nn.Cell):
        def __init__(self):
            super().__init__()
            self.tensor_scatter_op = op_map.get(func)()

        def construct(self, input_x, indices, updates):
            return self.tensor_scatter_op(input_x, indices, updates)

    net = TensorScatterOp()
    vmap_net = vmap(vmap(net, in_axes=(0, 0, 0), out_axes=0), in_axes=(0, 0, 0), out_axes=0)
    vmap_net(input_x, indices, updates)
