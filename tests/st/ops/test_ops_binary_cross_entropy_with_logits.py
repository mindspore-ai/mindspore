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
from mindspore import ops
from mindspore.mint.nn.functional import binary_cross_entropy_with_logits
import mindspore as ms
import tests.st.utils.test_utils as test_utils
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)

@test_utils.run_with_cell
def binary_cross_entropy_with_logits_forward_func(inputx, target, weight=None, posWeight=None, reduction="mean"):
    return binary_cross_entropy_with_logits(inputx, target, weight, reduction, posWeight)


@test_utils.run_with_cell
def binary_cross_entropy_with_logits_backward_func(inputx, target, weight=None, posWeight=None, reduction="mean"):
    grad_op = ops.grad(binary_cross_entropy_with_logits_forward_func, (0, 1, 2, 3, 4))
    return grad_op(inputx, target, weight, posWeight, reduction)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize("mode", ["pynative", "KBK"])
def test_ops_binary_cross_entropy_with_logits_forward(mode):
    """
    Feature: pyboost function.
    Description: test function binary_cross_entropy_with_logits forward.
    Expectation: expect correct result.
    """
    inputx = ms.Tensor(np.array([[-0.8, 1.2, 0.7], [-0.1, -0.4, 0.7]]), ms.float32)
    target = ms.Tensor(np.array([[0.3, 0.8, 1.2], [-0.6, 0.1, 2.2]]), ms.float32)
    weight = ms.Tensor(np.array([1.0, 1.0, 1.0]), ms.float32)
    pos_weight = ms.Tensor(np.array([1.0, 1.0, 1.0]), ms.float32)


    expect_sum = np.array([2.078167])
    expect_mean = np.array([0.3463612])
    expect_none = np.array([[0.6111006, 0.5032824, 0.26318604], [0.58439666, 0.55301523, -0.43681395]])

    if mode == "pynative":
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        out_mean = binary_cross_entropy_with_logits_forward_func(inputx, target, weight, pos_weight, "mean")
        out_sum = binary_cross_entropy_with_logits_forward_func(inputx, target, weight, pos_weight, "sum")
        out_none = binary_cross_entropy_with_logits_forward_func(inputx, target, weight, pos_weight, "none")
    elif mode == "KBK":
        ms.context.set_context(mode=ms.GRAPH_MODE)
        op = ms.jit(binary_cross_entropy_with_logits_forward_func, jit_config=ms.JitConfig(jit_level="O0"))
        out_mean = op(inputx, target, weight, pos_weight, "mean")
        out_sum = op(inputx, target, weight, pos_weight, "sum")
        out_none = op(inputx, target, weight, pos_weight, "none")
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE)
        out_mean = binary_cross_entropy_with_logits_forward_func(inputx, target, weight, pos_weight, "mean")
        out_sum = binary_cross_entropy_with_logits_forward_func(inputx, target, weight, pos_weight, "sum")
        out_none = binary_cross_entropy_with_logits_forward_func(inputx, target, weight, pos_weight, "none")

    np.testing.assert_allclose(out_mean.asnumpy(), expect_mean, rtol=1e-3)
    np.testing.assert_allclose(out_sum.asnumpy(), expect_sum, rtol=1e-3)
    np.testing.assert_allclose(out_none.asnumpy(), expect_none, rtol=1e-3)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize("mode", ["pynative", "KBK"])
def test_ops_binary_cross_entropy_with_logits_backward(mode):
    """
    Feature: pyboost function.
    Description: test function binary_cross_entropy_with_logits backward.
    Expectation: expect correct result.
    """
    inputx = ms.Tensor(np.array([[-0.8, 1.2, 0.7], [-0.1, -0.4, 0.7]]), ms.float32)
    target = ms.Tensor(np.array([[0.3, 0.8, 1.2], [-0.6, 0.1, 2.2]]), ms.float32)
    weight = ms.Tensor(np.array([1.0, 1.0, 1.0]), ms.float32)
    pos_weight = ms.Tensor(np.array([1.0, 1.0, 1.0]), ms.float32)


    expect_sum = np.array([[0.0100254714, -0.031475246, -0.53181231], [1.07502079, 0.301312357, -1.53181231]])
    expect_mean = np.array([[0.00167091191, -0.00524587464, -0.0886353850], [0.179170132, 0.0502187274, -0.255302072]])
    expect_none = np.array([[0.010025471, -0.031475246, -0.53181231], [1.07502079, 0.3013123578, -1.53181231]])


    if mode == "pynative":
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        out_mean = binary_cross_entropy_with_logits_backward_func(inputx, target, weight, pos_weight, "mean")
        out_sum = binary_cross_entropy_with_logits_backward_func(inputx, target, weight, pos_weight, "sum")
        out_none = binary_cross_entropy_with_logits_backward_func(inputx, target, weight, pos_weight, "none")
    elif mode == "KBK":
        ms.context.set_context(mode=ms.GRAPH_MODE)
        op = ms.jit(binary_cross_entropy_with_logits_backward_func, jit_config=ms.JitConfig(jit_level="O0"))
        out_mean = op(inputx, target, weight, pos_weight, "mean")
        out_sum = op(inputx, target, weight, pos_weight, "sum")
        out_none = op(inputx, target, weight, pos_weight, "none")
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE)
        out_mean = binary_cross_entropy_with_logits_backward_func(inputx, target, weight, pos_weight, "mean")
        out_sum = binary_cross_entropy_with_logits_backward_func(inputx, target, weight, pos_weight, "sum")
        out_none = binary_cross_entropy_with_logits_backward_func(inputx, target, weight, pos_weight, "none")

    np.testing.assert_allclose(out_mean[0].asnumpy(), expect_mean, rtol=1e-3)
    np.testing.assert_allclose(out_sum[0].asnumpy(), expect_sum, rtol=1e-3)
    np.testing.assert_allclose(out_none[0].asnumpy(), expect_none, rtol=1e-3)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize("context_mode", [ms.PYNATIVE_MODE])
def test_ops_binary_cross_entropy_with_logits_dynamic_shape(context_mode):
    """
    Feature: pyboost function.
    Description: test function binary_cross_entropy_with_logits forward with dynamic shape.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x1 = ms.Tensor(generate_random_input((7, 8, 9), np.float32))
    target1 = ms.Tensor(generate_random_input((7, 8, 9), np.float32))
    weight1 = ms.Tensor(generate_random_input((7, 8, 9), np.float32))
    pos_weight1 = ms.Tensor(generate_random_input((7, 8, 9), np.float32))

    x2 = ms.Tensor(generate_random_input((7, 8, 9), np.float32))
    target2 = ms.Tensor(generate_random_input((7, 8, 9), np.float32))
    weight2 = ms.Tensor(generate_random_input((7, 8, 9), np.float32))
    pos_weight2 = ms.Tensor(generate_random_input((7, 8, 9), np.float32))

    test_cell = test_utils.to_cell_obj(binary_cross_entropy_with_logits_forward_func)
    TEST_OP(test_cell, [[x1, target1, weight1, pos_weight1], [x2, target2, weight2, pos_weight2]], grad=True,
            jit_level="O0")
