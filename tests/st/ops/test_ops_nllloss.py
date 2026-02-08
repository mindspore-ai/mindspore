# Copyright 2024 Huawei Technocasties Co., Ltd
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
from mindspore import grad
from mindspore.mint.nn.functional import nll_loss
import mindspore as ms
import tests.st.utils.test_utils as test_utils
from tests.mark_utils import arg_mark
from tests.st.ops.test_tools.test_op import TEST_OP


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)


def get_input():
    inputx = ms.Tensor(np.array([[1.81770931, 0.31360114, 0.30931599, 0.18100776, 0.40601558],
                                 [-1.23061385, -0.88784365, 0.68879239, -0.07928044, 0.9060898],
                                 [-0.57779883, -0.54754656, 0.84325396, -1.59585763, 0.61647592]]), ms.float32)
    target = ms.Tensor(np.array([4, 3, 2]), ms.int64)
    weight = ms.Tensor(np.array([-0.75326879, 1.45076741, 1.19889046, 0.09059219, -1.53917865]), ms.float32)
    ignore_index = 3
    return inputx, target, weight, ignore_index


def get_output_forward(reduction):
    output_mean = np.array([1.134446])
    output_sum = np.array([-0.386039])
    output_none = np.array([0.6249305, 0., -1.0109692])
    output = {"mean": output_mean, "sum": output_sum, "none": output_none}
    return output[reduction]


def get_output_backward(reduction):
    input_grad_mean = np.array([[0.0000, 0.0000, 0.0000, 0.0000, -4.523162],
                                [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                [0.0000, 0.0000, 3.523162, 0.0000, 0.0000]])
    input_grad_sum = np.array([[0.0000, 0.0000, 0.0000, 0.0000, 1.539179],
                               [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                               [0.0000, 0.0000, -1.19889, 0.0000, 0.0000]])
    input_grad_none = np.array([[0.0000, 0.0000, 0.0000, 0.0000, 1.539179],
                                [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                [0.0000, 0.0000, -1.19889, 0.0000, 0.0000]])
    output = {"mean": [input_grad_mean],
              "sum": [input_grad_sum],
              "none": [input_grad_none]}
    return output[reduction]


@test_utils.run_with_cell
def nll_loss_forward_func(inputx, target, weight=None, ignore_index=-100, reduction="mean"):
    return nll_loss(inputx, target, weight, ignore_index, reduction)


@test_utils.run_with_cell
def nll_loss_backward_func(inputx, target, weight=None, ignore_index=-100, reduction="mean"):
    grad_op = grad(nll_loss_forward_func, (0, 1, 2))
    return grad_op(inputx, target, weight, ignore_index, reduction)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential', mem_peak=2048.01)
@pytest.mark.parametrize("mode", ["pynative", "KBK"])
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_ops_nll_loss(mode, reduction):
    """
    Feature: pyboost function.
    Description: test function nll_loss backward.
    Expectation: expect correct result.
    """
    inputx, target, weight, ignore_index = get_input()
    expect_forward = get_output_forward(reduction)
    expect_backward = get_output_backward(reduction)
    if mode == "pynative":
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    elif mode == "KBK":
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_config={"jit_level": "O0"})
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_config={"jit_level": "O2"})
    output_forward = nll_loss_forward_func(inputx, target, weight, ignore_index, reduction)
    output_backward = nll_loss_backward_func(inputx, target, weight, ignore_index, reduction)
    np.testing.assert_allclose(output_forward.asnumpy(), expect_forward, rtol=1e-3)
    np.testing.assert_allclose(output_backward[0].asnumpy(), expect_backward[0], rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=1024.0)
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_ops_nll_loss_dynamic_shape(reduction):
    """
    Feature: pyboost function.
    Description: test function nll_loss forward with dynamic shape.
    Expectation: expect correct result.
    """
    x1 = ms.Tensor(np.random.rand(7, 8, 9).astype(np.float32))
    target1 = ms.Tensor(generate_random_input((7, 9), np.int64))
    weight1 = ms.Tensor(generate_random_input((8,), np.float32))
    ignore_index1 = 3

    x2 = ms.Tensor(np.random.rand(9, 8).astype(np.float32))
    target2 = ms.Tensor(generate_random_input((9,), np.int64))
    weight2 = ms.Tensor(generate_random_input((8,), np.float32))
    ignore_index2 = 2


    test_cell = test_utils.to_cell_obj(nll_loss_forward_func)
    TEST_OP(test_cell, [[x1, target1, weight1, ignore_index1, reduction],
                        [x2, target2, weight2, ignore_index2, reduction]],
            disable_mode=['GRAPH_MODE_GE', 'GRAPH_MODE_O0'],
            disable_case=['ScalarTensor', 'Deterministic'],
            case_config={'disable_input_check': True,
                         'all_dim_zero': True})
