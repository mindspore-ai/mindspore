# Copyright 2024 Huawei Technologies Co., Ltd
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

import mindspore as ms
from mindspore import Tensor, context
from mindspore.ops.function.array_func import index_fill_ext
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.mark_utils import arg_mark
from tests.st.utils.test_utils import run_with_cell

def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)

def _count_unequal_element(data_expected, data_me, rtol, atol):
    assert data_expected.shape == data_me.shape
    total_count = len(data_expected.flatten())
    error = np.abs(data_expected - data_me)
    greater = np.greater(error, atol + np.abs(data_me) * rtol)
    loss_count = np.count_nonzero(greater)
    assert (loss_count / total_count) < rtol, \
        "\ndata_expected_std:{0}\ndata_me_error:{1}\nloss:{2}". \
            format(data_expected[greater], data_me[greater], error[greater])

def allclose_nparray(data_expected, data_me, rtol, atol, equal_nan=True):
    if np.any(np.isnan(data_expected)):
        assert np.allclose(data_expected, data_me, rtol, atol, equal_nan=equal_nan)
    elif not np.allclose(data_expected, data_me, rtol, atol, equal_nan=equal_nan):
        _count_unequal_element(data_expected, data_me, rtol, atol)
    else:
        assert True

@run_with_cell
def index_fill_tensor_forward_func(input_x, dim, index, value):
    return index_fill_ext(input_x, dim, index, value)

@run_with_cell
def index_fill_tensor_backward(input_x, dim, index, value):
    grad_fn = ms.grad(index_fill_tensor_forward_func, grad_position=(0, 3))
    return grad_fn(input_x, dim, index, value)

@arg_mark(
    plat_marks=["platform_ascend"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential", mem_peak=2048.01
)
@pytest.mark.parametrize("mode", ["pynative", "KBK"])
def test_ops_index_fill_tensor(mode):
    """
    Feature: ops.index_fill_tensor
    Description: Verify the result of ops.index_fill_tensor with tensor value
    Expectation: success
    """
    input_x = Tensor(np.arange(25).reshape(5, 5), dtype=ms.float32)
    index = Tensor(np.array([0, 1, 1, 2]), dtype=ms.int64)
    value = Tensor(10.0, dtype=ms.float32)
    dim = 0

    expected_output = np.copy(input_x.asnumpy())
    expected_output[index.asnumpy(), :] = value

    expected_grad_input = np.ones((5, 5))
    expected_grad_input[index.asnumpy(), :] = 0

    if mode == "pynative":
        context.set_context(mode=ms.PYNATIVE_MODE)
    elif mode == "KBK":
        context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")

    # Forward computation
    output = index_fill_tensor_forward_func(input_x, dim, index, value)
    allclose_nparray(output.asnumpy(), expected_output, 1e-4, 1e-4)

    # Backward computation
    grads = index_fill_tensor_backward(input_x, dim, index, value)
    allclose_nparray(grads[0].asnumpy(), expected_grad_input, 1e-4, 1e-4)

@arg_mark(
    plat_marks=["platform_ascend", "platform_ascend910b"],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="unessential", mem_peak=2.01
)
def test_ops_index_fill_tensor_dynamic():
    """
    Feature: Test index_fill_ext with dynamic shape in graph mode using TEST_OP.
    Description: call ops.index_fill_ext with valid input, dim, index, and value.
    Expectation: return the correct value.
    """

    x1 = Tensor(np.arange(25).reshape(5, 5), dtype=ms.float32)
    dim1 = 0
    index1 = Tensor(np.array([0, 1]).astype(np.int64))
    value1 = Tensor(0, ms.float32)

    x2 = Tensor(np.arange(24).reshape(4, 6), dtype=ms.float32)
    dim2 = 0  # Choose a random dim to fill
    index2 = Tensor(np.array([0, 1]).astype(np.int64))
    value2 = Tensor(0, ms.float32)

    TEST_OP(
        index_fill_ext,
        [[Tensor(x1), dim1, index1, value1], [Tensor(x2), dim2, index2, value2]],
        disable_mode=["GRAPH_MODE_GE", "GRAPH_MODE_O0"],
        disable_case=['EmptyTensor', 'ScalarTensor'],
        case_config={'disable_input_check': True,
                     'disable_grad': True}
    )
