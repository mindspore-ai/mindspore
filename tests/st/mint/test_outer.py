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
from mindspore import mint, Tensor, context
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.mark_utils import arg_mark
from tests.st.ops.test_tools.ops_binary_cases import ops_binary_cases, OpsBinaryCase
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
def outer_forward_func(input_x, vec2):
    return mint.outer(input_x, vec2)


@run_with_cell
def outer_backward_func(input_x, vec2):
    grad_fn = ms.grad(outer_forward_func, grad_position=(0, 1))
    return grad_fn(input_x, vec2)


def mint_outer_binary_compare(input_binary_data, output_binary_data, loss, mode):
    if mode == "pynative":
        context.set_context(mode=ms.PYNATIVE_MODE)
    elif mode == "KBK":
        context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    input_x = Tensor(input_binary_data[0])
    vec2 = Tensor(input_binary_data[1])

    output = outer_forward_func(input_x, vec2)
    allclose_nparray(output.asnumpy(), output_binary_data[0], loss, loss)

    grads = outer_backward_func(input_x, vec2)
    allclose_nparray(grads[0].asnumpy(), output_binary_data[1], loss, loss)
    allclose_nparray(grads[1].asnumpy(), output_binary_data[2], loss, loss)


@ops_binary_cases(
    OpsBinaryCase(
        input_info=[
            ((1000,), np.float32),
            ((2000,), np.float32),
        ],
        output_info=[
            ((1000, 2000), np.float32),
            ((1000,), np.float32),
            ((2000,), np.float32),
        ],
        extra_info="auto_drive",
    )
)
def mint_outer_binary_case1(input_binary_data=None, output_binary_data=None, loss=1e-04, mode="pynative"):
    mint_outer_binary_compare(input_binary_data, output_binary_data, loss, mode)


@arg_mark(
    plat_marks=["platform_ascend"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
    mem_peak=2.00
)
@pytest.mark.parametrize("mode", ["pynative", "KBK"])
def test_mint_outer_binary_cases(mode):
    """
    Feature: mint.outer
    Description: Verify the result of outer
    Expectation: success
    """
    mint_outer_binary_case1(loss=1e-04, mode=mode)


@arg_mark(
    plat_marks=["platform_ascend", "platform_ascend910b"],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="unessential", mem_peak=4.01
)
def test_mint_outer_dynamic():
    """
    Feature: Test outer with dynamic shape in graph mode using TEST_OP.
    Description: call mint.outer with valid input and index.
    Expectation: return the correct value.
    """
    x1 = generate_random_input((2,), np.float32)
    y1 = generate_random_input((3,), np.float32)
    x2 = generate_random_input((4,), np.float32)
    y2 = generate_random_input((5,), np.float32)
    # TEST_OP Todo GradByRequirement NULL pointer failure.
    TEST_OP(
        mint.outer,
        [[Tensor(x1), Tensor(y1)], [Tensor(x2), Tensor(y2)]],
        disable_mode=["GRAPH_MODE_GE"],
        disable_case=['ScalarTensor',
                      'GradByRequirement'],
        case_config={'disable_input_check': True}
    )
