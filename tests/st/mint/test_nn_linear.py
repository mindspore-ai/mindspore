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
import mindspore.mint.nn as mnn
from mindspore import Tensor, context
from tests.mark_utils import arg_mark
from tests.st.utils import test_utils
from tests.st.ops.test_tools.ops_binary_cases import ops_binary_cases, OpsBinaryCase


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


def linear_forward_func(in_features, out_features, weight, bias, input_x):
    net = mnn.Linear(in_features, out_features, weight_init=weight, bias_init=bias)
    return net(input_x)


def linear_backward_func(in_features, out_features, weight, bias, input_x):
    net = mnn.Linear(in_features, out_features, weight_init=weight, bias_init=bias)
    weights = net.trainable_params()
    grad_fn = ms.grad(net, grad_position=0, weights=weights)
    return grad_fn(input_x)


def mint_nn_linear_binary_compare(input_binary_data, output_binary_data, loss, mode):
    if mode == "pynative":
        context.set_context(mode=ms.PYNATIVE_MODE)
    elif mode == "KBK":
        context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    input_x = Tensor(input_binary_data[0])
    weight = Tensor(input_binary_data[1])
    bias = Tensor(input_binary_data[2])

    output = linear_forward_func(100, 50, weight, bias, input_x)
    allclose_nparray(output.asnumpy(), output_binary_data[0], loss, loss)

    grad_input, grad_params = linear_backward_func(100, 50, weight, bias, input_x)
    allclose_nparray(grad_input.asnumpy(), output_binary_data[1], loss, loss)
    allclose_nparray(grad_params[0].asnumpy(), output_binary_data[2], loss, loss)
    allclose_nparray(grad_params[1].asnumpy(), output_binary_data[3], loss, loss)


@ops_binary_cases(
    OpsBinaryCase(
        input_info=[
            ((100, 100, 100), np.float32),
            ((50, 100), np.float32),
            ((50,), np.float32),
        ],
        output_info=[
            ((100, 100, 50), np.float32),
            ((100, 100, 100), np.float32),
            ((50, 100), np.float32),
            ((50,), np.float32),
        ],
        extra_info="auto_drive",
    )
)
def mint_nn_linear_binary_case1(input_binary_data=None, output_binary_data=None, loss=1e-04, mode="pynative"):
    mint_nn_linear_binary_compare(input_binary_data, output_binary_data, loss, mode)


@arg_mark(
    plat_marks=["platform_ascend"],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="essential",
    mem_peak=0.01
)
@pytest.mark.parametrize("mode", ["pynative", "KBK"])
def test_mint_nn_linear_binary_cases_910a(mode):
    """
    Feature: mint.nn.Linear
    Description: Verify the result of Linear
    Expectation: success
    """
    # This operator converts float32 to float16 on the 910a platform, so the loss is set to 1e-03
    mint_nn_linear_binary_case1(loss=1e-03, mode=mode)


@pytest.mark.parametrize("mode", ["pynative", "KBK"])
def test_mint_nn_linear_binary_cases_910b(mode):
    """
    Feature: mint.nn.Linear
    Description: Verify the result of Linear
    Expectation: success
    """
    mint_nn_linear_binary_case1(loss=1e-04, mode=mode)


@test_utils.run_with_cell
def linear_nn_functional_forward_func(input_x, weight, bias):
    return mnn.functional.linear(input_x, weight, bias)


@test_utils.run_with_cell
def linear_nn_functional_backward_func(input_x, weight, bias):
    return ms.grad(linear_nn_functional_forward_func, (0, 1, 2))(input_x, weight, bias)


@arg_mark(
    plat_marks=["platform_ascend"],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="essential",
    mem_peak=0.01
)
@pytest.mark.parametrize("mode", ["pynative", "KBK"])
@pytest.mark.parametrize("shape", [[(2, 0), (0, 0)], [(2, 0), (2, 0)], [(0, 2), (0, 2)]])
def test_mint_nn_functional_linear_empty_tensor(mode, shape):
    """
    Feature: mint.nn.Linear
    Description: Verify the result of Linear
    Expectation: success
    """
    if mode == "pynative":
        context.set_context(mode=ms.PYNATIVE_MODE)
    elif mode == "KBK":
        context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")

    input_x = Tensor(np.random.randn(*shape[0]), dtype=ms.float32)
    weight = Tensor(np.random.randn(*shape[1]), dtype=ms.float32)
    bias = Tensor(np.random.randn(), dtype=ms.float32)

    out = linear_nn_functional_forward_func(input_x, weight, bias)
    assert out.shape == (shape[0][0], shape[1][0])

    grad_input, grad_weight, grad_bias = linear_nn_functional_backward_func(input_x, weight, bias)
    assert grad_input.shape == input_x.shape
    assert grad_weight.shape == weight.shape
    assert grad_bias.shape == bias.shape
