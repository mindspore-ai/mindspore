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
import random

import mindspore as ms
from mindspore import Tensor, jit
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
    assert (
        loss_count / total_count
    ) < rtol, "\ndata_expected_std:{0}\ndata_me_error:{1}\nloss:{2}".format(
        data_expected[greater], data_me[greater], error[greater]
    )


def allclose_nparray(data_expected, data_me, rtol, atol, equal_nan=True):
    if np.any(np.isnan(data_expected)):
        assert np.allclose(data_expected, data_me, rtol, atol, equal_nan=equal_nan)
    elif not np.allclose(data_expected, data_me, rtol, atol, equal_nan=equal_nan):
        _count_unequal_element(data_expected, data_me, rtol, atol)
    else:
        assert True


@run_with_cell
def fill__forward_func(input_x, value):
    temp = input_x + 0
    return temp.fill_(value)

@jit(backend="ms_backend")
def fill__backward_func_scalar(input_x, value):
    grad_fn = ms.grad(fill__forward_func, grad_position=(0))
    return grad_fn(input_x, value)


@run_with_cell
def fill__backward_func_tensor(input_x, value):
    grad_fn = ms.grad(fill__forward_func, grad_position=(0, 1))
    return grad_fn(input_x, value)


@arg_mark(
    plat_marks=["platform_ascend"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential", mem_peak=4.00
)
@pytest.mark.parametrize("mode", ["pynative", "KBK"])
def test_tensor_fill__normal(mode):
    """
    Feature: Tensor.fill_
    Description: Verify the result of Tensor.fill_
    Expectation: success
    """
    if mode == "pynative":
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    elif mode == "KBK":
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")

    input_x = Tensor(np.full((100, 100), 0, dtype=np.float32))
    value = 10
    except_out = Tensor(np.full((100, 100), 10, dtype=np.float32))
    except_x_grad = Tensor(np.full((100, 100), 0, dtype=np.float32))
    except_value_tensor_grad = Tensor(10000, ms.float32)
    output_scalar = fill__forward_func(input_x, value)
    output_tensor = fill__forward_func(input_x, Tensor(value))
    allclose_nparray(output_scalar.asnumpy(), except_out.asnumpy(), 1e-04, 1e-04)
    allclose_nparray(output_tensor.asnumpy(), except_out.asnumpy(), 1e-04, 1e-04)

    grads_scalar = fill__backward_func_scalar(input_x, value)
    grads_tensor = fill__backward_func_tensor(input_x, Tensor(value))
    allclose_nparray(grads_scalar[0].asnumpy(), except_x_grad.asnumpy(), 1e-04, 1e-04)
    allclose_nparray(grads_tensor[0].asnumpy(), except_x_grad.asnumpy(), 1e-04, 1e-04)
    allclose_nparray(
        grads_tensor[1].asnumpy(), except_value_tensor_grad.asnumpy(), 1e-04, 1e-04
    )


@arg_mark(
    plat_marks=["platform_ascend", "platform_ascend910b"],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="unessential", mem_peak=4.01
)
def test_tensor_fill_tensor_dynamic():
    """
    Feature: Test outer with dynamic shape in graph mode using TEST_OP.
    Description: call Tensor.fill_ with valid input and index.
    Expectation: return the correct value.
    """
    x1 = generate_random_input((2, 3), np.float32)
    y1 = random.random()
    x2 = generate_random_input((4, 5, 6), np.float32)
    y2 = random.random()
    TEST_OP(
        fill__forward_func,
        [[Tensor(x1), Tensor(y1)], [Tensor(x2), Tensor(y2)]],
        disable_mode=["GRAPH_MODE_GE"],
        case_config={'disable_input_check': True},
        inplace_update=True,
    )


@arg_mark(
    plat_marks=["platform_ascend", "platform_ascend910b"],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="unessential", mem_peak=4.01
)
def test_tensor_fill_scalar_dynamic():
    """
    Feature: Test outer with dynamic shape in graph mode using TEST_OP.
    Description: call Tensor.fill_ with valid input and index.
    Expectation: return the correct value.
    """
    x1 = generate_random_input((2, 3), np.float32)
    y1 = random.random()
    x2 = generate_random_input((4, 5, 6), np.float32)
    y2 = random.random()
    TEST_OP(
        fill__forward_func,
        [[Tensor(x1), y1], [Tensor(x2), y2]],
        disable_mode=["GRAPH_MODE_GE"],
        case_config={'disable_input_check': True},
        inplace_update=True,
    )
