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
from mindspore import Tensor
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.mark_utils import arg_mark
from tests.st.utils.test_utils import run_with_cell


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)


def np_masked_fill_forward_func(input_x, mask, value):
    input_x = np.ma.array(input_x, mask=mask, fill_value=value)
    return input_x.filled()


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
def masked_fill__forward_func(input_x, mask, value):
    temp = input_x + 0
    return temp.masked_fill_(mask, value)


@run_with_cell
def masked_fill__backward_func_scalar(input_x, mask, value):
    return ms.grad(masked_fill__forward_func, (0, 1))(input_x, mask, value)


@run_with_cell
def masked_fill__backward_func_tensor(input_x, mask, value):
    return ms.grad(masked_fill__forward_func, (0, 1, 2))(input_x, mask, value)


@arg_mark(
    plat_marks=["platform_ascend"],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="essential", mem_peak=2048.01
)
@pytest.mark.parametrize("mode", ["pynative", "KBK"])
def test_ops_masked_fill_normal(mode):
    """
    Feature: Tensor.masked_fill_
    Description: Verify the result of Tensor.masked_fill_
    Expectation: success
    """
    np.random.seed(42)
    input_x_np = generate_random_input((100, 100), np.float32)
    mask_np = np.random.choice([True, False], size=[100, 100])
    value = random.random()
    input_x = Tensor(input_x_np)
    mask = Tensor(mask_np)
    except_out = np_masked_fill_forward_func(input_x_np, mask_np, value)

    if mode == "pynative":
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    elif mode == "KBK":
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    output1 = masked_fill__forward_func(input_x, mask, value)
    output2 = masked_fill__forward_func(input_x, mask, Tensor(value))
    allclose_nparray(output1.asnumpy(), except_out, 1e-04, 1e-04)
    allclose_nparray(output2.asnumpy(), except_out, 1e-04, 1e-04)

    input_x1 = ms.Tensor(np.array([1.0, 2.0, 3.0, 4.0]).astype(np.float32))
    mask = ms.Tensor(np.array([True, True, False, False]).astype(np.bool_))
    expect_x_grad = np.asarray([0.0, 0.0, 1.0, 1.0]).astype(np.float32)
    expect_mask_grad = np.asarray([0.0, 0.0, 0.0, 0.0]).astype(np.float32)
    expect_value_grad = np.asarray(2.0).astype(np.float32)
    grads_scalar = masked_fill__backward_func_scalar(input_x1, mask, 0.5)
    grads_tensor = masked_fill__backward_func_tensor(input_x1, mask, Tensor(0.5))
    allclose_nparray(grads_scalar[0].asnumpy(), expect_x_grad, 1e-04, 1e-04)
    allclose_nparray(grads_scalar[1].asnumpy(), expect_mask_grad, 1e-04, 1e-04)
    allclose_nparray(grads_tensor[0].asnumpy(), expect_x_grad, 1e-04, 1e-04)
    allclose_nparray(grads_tensor[1].asnumpy(), expect_mask_grad, 1e-04, 1e-04)
    allclose_nparray(grads_tensor[2].asnumpy(), expect_value_grad, 1e-04, 1e-04)


@arg_mark(
    plat_marks=["platform_ascend", "platform_ascend910b"],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="unessential", mem_peak=20.01
)
def test_tensor_masked_fill_tensor_dynamic():
    """
    Feature: Test Tensor.masked_fill_ with dynamic shape in graph mode using TEST_OP.
    Description: call Tensor.masked_fill_ with valid input and index.
    Expectation: return the correct value.
    """
    x1 = generate_random_input((2, 3), np.float32)
    mask1 = np.random.choice([True, False], size=[2, 3])
    y1 = random.random()
    x2 = generate_random_input((4, 5, 6), np.float32)
    mask2 = np.random.choice([True, False], size=[4, 5, 6])
    y2 = random.random()
    TEST_OP(
        masked_fill__forward_func,
        [
            [Tensor(x1), Tensor(mask1), Tensor(y1)],
            [Tensor(x2), Tensor(mask2), Tensor(y2)],
        ],
        disable_mode=["GRAPH_MODE_GE"],
        disable_case=['ScalarTensor'],
        case_config={'disable_input_check': True, 'all_dim_zero': True},
        inplace_update=True,
    )


@arg_mark(
    plat_marks=["platform_ascend", "platform_ascend910b"],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="unessential", mem_peak=4.01
)
def test_tensor_masked_fill_scalar_dynamic():
    """
    Feature: Test Tensor.masked_fill_ with dynamic shape in graph mode using TEST_OP.
    Description: call Tensor.masked_fill_ with valid input and index.
    Expectation: return the correct value.
    """
    x1 = generate_random_input((2, 3), np.float32)
    mask1 = np.random.choice([True, False], size=[2, 3])
    y1 = random.random()
    x2 = generate_random_input((4, 5, 6), np.float32)
    mask2 = np.random.choice([True, False], size=[4, 5, 6])
    y2 = random.random()
    TEST_OP(
        masked_fill__forward_func,
        [[Tensor(x1), Tensor(mask1), y1], [Tensor(x2), Tensor(mask2), y2]],
        disable_mode=["GRAPH_MODE_GE"],
        case_config={'disable_input_check': True, 'all_dim_zero': True},
        inplace_update=True,
    )
