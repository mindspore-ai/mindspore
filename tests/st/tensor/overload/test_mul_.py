# Copyright 2025 Huawei Technologies Co., Ltd
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
# pylint: disable=unused-variable
import numpy as np
import pytest

import mindspore as ms
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.mark_utils import arg_mark
from tests.st.utils.test_utils import run_with_cell


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


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)


@run_with_cell
def inplace_mul_forward_func(x, y):
    temp = x + 0
    return temp.mul_(y)


@run_with_cell
def inplace_mul_forward_operator_func(x, y):
    temp = x + 0
    temp *= y
    return temp


@run_with_cell
def inplace_mul_backward_func_scalar(input_x, value):
    grad_fn = ms.grad(inplace_mul_forward_func, grad_position=(0))
    return grad_fn(input_x, value)


@run_with_cell
def inplace_mul_backward_func_tensor(input_x, value):
    grad_fn = ms.grad(inplace_mul_forward_func, grad_position=(0, 1))
    return grad_fn(input_x, value)


@arg_mark(
    plat_marks=["platform_ascend"],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="essential", mem_peak=1024.01
)
@pytest.mark.parametrize("mode", ["pynative"])
def test_tensor_mul__normal(mode):
    """
    Feature: standard forward, backward features.
    Description: test function copy.
    Expectation: expect correct result.
    """
    if mode == "pynative":
        ms.context.set_context(mode=ms.PYNATIVE_MODE)

    x = generate_random_input((2, 3, 4), np.float32)
    y = generate_random_input((2, 3, 4), np.float32)
    z = 10.0
    expect_out_tensor = x * y
    expect_out_scalar = x * z
    expect_x_grad = y
    expect_y_grad = x
    except_x_tensor_grad = np.full((2, 3, 4), z, dtype=np.float32)

    out_tensor = inplace_mul_forward_func(ms.Tensor(x), ms.Tensor(y))
    out_scalar = inplace_mul_forward_func(ms.Tensor(x), z)
    out_tensor_op = inplace_mul_forward_operator_func(ms.Tensor(x), ms.Tensor(y))
    out_scalar_op = inplace_mul_forward_operator_func(ms.Tensor(x), z)
    allclose_nparray(expect_out_tensor, out_tensor.asnumpy(), 1e-4, 1e-4)
    allclose_nparray(expect_out_scalar, out_scalar.asnumpy(), 1e-4, 1e-4)
    allclose_nparray(expect_out_tensor, out_tensor_op.asnumpy(), 1e-4, 1e-4)
    allclose_nparray(expect_out_scalar, out_scalar_op.asnumpy(), 1e-4, 1e-4)

    grads_tensor = inplace_mul_backward_func_tensor(ms.Tensor(x), ms.Tensor(y))
    grads_scalar = inplace_mul_backward_func_scalar(ms.Tensor(x), z)
    allclose_nparray(expect_x_grad, grads_tensor[0].asnumpy(), 1e-4, 1e-4)
    allclose_nparray(expect_y_grad, grads_tensor[1].asnumpy(), 1e-4, 1e-4)
    allclose_nparray(except_x_tensor_grad, grads_scalar[0].asnumpy(), 1e-4, 1e-4)


@arg_mark(
    plat_marks=["platform_ascend"],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="unessential", mem_peak=1024.01
)
def test_tensor_mul__dynamic():
    """
    Feature: dynamic shape forward, backward features.
    Description: test copy forward with dynamic shape.
    Expectation: expect correct result.
    """
    tensor_x1 = ms.Tensor(generate_random_input((2, 3), np.float32))
    tensor_y1 = ms.Tensor(generate_random_input((2, 3), np.float32))
    tensor_x2 = ms.Tensor(generate_random_input((3, 4, 5), np.float32))
    tensor_y2 = ms.Tensor(generate_random_input((1, 1, 5), np.float32))

    scalar_x1 = tensor_x1
    scalar_y1 = 10.0
    scalar_x2 = tensor_x2
    scalar_y2 = 30.0

    TEST_OP(
        inplace_mul_forward_func,
        [[tensor_x1, tensor_y1], [tensor_x2, tensor_y2]],
        disable_mode=["GRAPH_MODE_GE", "GRAPH_MODE_O0"],
        case_config={'all_dim_zero': True}
    )

    TEST_OP(
        inplace_mul_forward_func,
        [[scalar_x1, scalar_y1], [scalar_x2, scalar_y2]],
        disable_mode=["GRAPH_MODE_GE", "GRAPH_MODE_O0"],
        case_config={'all_dim_zero': True}
    )
