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
from mindspore.mint.nn.functional import binary_cross_entropy
import mindspore as ms
import tests.st.utils.test_utils as test_utils
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)

def get_input():
    inputx = ms.Tensor(np.array([[0.1531, 0.3302, 0.7537], [0.2200, 0.6875, 0.2268], [0.5109, 0.5873, 0.9275]]),
                       ms.float32)
    target = ms.Tensor(np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]]), ms.float32)
    weight = ms.Tensor(np.array([[0.9000, 0.1000, 0.1000], [0.1000, 0.1000, 0.9000], [0.1000, 0.9000, 0.1000]]),
                       ms.float32)
    return inputx, target, weight

def get_output_forward(reduction):
    output_mean = np.array([0.4621])
    output_sum = np.array([4.1586])
    output_none = np.array([[1.6890, 0.0401, 0.1401], [0.02485, 0.1163, 1.3353], [0.0715, 0.4790, 0.2624]])
    output = {"mean": output_mean, "sum": output_sum, "none": output_none}
    return output[reduction]

def get_output_backward(reduction):
    input_grad_sum = np.array([[-5.8785, 0.1493, 0.406], [0.1282, 0.32, -3.9683], [0.2045, -1.5325, 1.3793]])
    input_grad_mean = np.array([[-0.6532, 0.0166, 0.0451], [0.01425, 0.03555, -0.4409], [0.0227, -0.1703, 0.1533]])
    input_grad_none = np.array([[-5.8785, 0.1493, 0.406], [0.1282, 0.32, -3.9683], [0.2045, -1.5325, 1.3793]])
    target_grad_sum = np.array([[1.5394, 0.0707, -0.1118], [0.1266, -0.0788, 1.1038], [-0.00436, -0.3175, -0.2549]])
    target_grad_mean = np.array([[0.171, 0.00786, -0.01243],
                                 [0.01406, -0.00876, 0.1226],
                                 [-0.000485, -0.0353, -0.0283]])
    target_grad_none = np.array([[1.539, 0.0707, -0.1118], [0.1266, -0.0788, 1.1038], [-0.00436, -0.3175, -0.2549]])
    output = {"mean": [input_grad_mean, target_grad_mean],
              "sum": [input_grad_sum, target_grad_sum],
              "none": [input_grad_none, target_grad_none]}
    return output[reduction]

@test_utils.run_with_cell
def binary_cross_entropy_forward_func(inputx, target, weight=None, reduction="mean"):
    return binary_cross_entropy(inputx, target, weight, reduction)


@test_utils.run_with_cell
def binary_cross_entropy_backward_func(inputx, target, weight=None, reduction="mean"):
    grad_op = ops.grad(binary_cross_entropy_forward_func, (0, 1, 2, 3))
    return grad_op(inputx, target, weight, reduction)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize("mode", ["pynative", "KBK", "graph"])
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_ops_binary_cross_entropy_normal(mode, reduction):
    """
    Feature: pyboost function.
    Description: test function binary_cross_entropy backward.
    Expectation: expect correct result.
    """
    inputx, target, weight = get_input()
    expect_forward = get_output_forward(reduction)
    expect_backward = get_output_backward(reduction)
    if mode == "pynative":
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        output_forward = binary_cross_entropy_forward_func(inputx, target, weight, reduction)
        output_backward = binary_cross_entropy_backward_func(inputx, target, weight, reduction)

    elif mode == "KBK":
        ms.context.set_context(mode=ms.GRAPH_MODE)
        op_froward = ms.jit(binary_cross_entropy_forward_func, jit_config=ms.JitConfig(jit_level="O0"))
        output_forward = op_froward(inputx, target, weight, reduction)
        op_backward = ms.jit(binary_cross_entropy_backward_func, jit_config=ms.JitConfig(jit_level="O0"))
        output_backward = op_backward(inputx, target, weight, reduction)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE)
        output_forward = binary_cross_entropy_forward_func(inputx, target, weight, reduction)
        output_backward = binary_cross_entropy_backward_func(inputx, target, weight, reduction)
    np.testing.assert_allclose(output_forward.asnumpy(), expect_forward, rtol=1e-3)
    np.testing.assert_allclose(output_backward[0].asnumpy(), expect_backward[0], rtol=1e-3)
    np.testing.assert_allclose(output_backward[1].asnumpy(), expect_backward[1], rtol=1e-3)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize("context_mode", [ms.PYNATIVE_MODE])
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_ops_binary_cross_entropy_dynamic_shape(context_mode, reduction):
    """
    Feature: pyboost function.
    Description: test function binary_cross_entropy forward with dynamic shape.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x1 = ms.Tensor(np.random.rand(7, 8, 9).astype(np.float32))
    target1 = ms.Tensor(generate_random_input((7, 8, 9), np.float32))
    weight1 = ms.Tensor(generate_random_input((7, 8, 9), np.float32))

    x2 = ms.Tensor(np.random.rand(9, 8).astype(np.float32))
    target2 = ms.Tensor(generate_random_input((9, 8), np.float32))
    weight2 = ms.Tensor(generate_random_input((9, 8), np.float32))

    test_cell = test_utils.to_cell_obj(binary_cross_entropy_forward_func)
    TEST_OP(test_cell, [[x1, target1, weight1, reduction], [x2, target2, weight2, reduction]],
            "binary_cross_entropy", disable_input_check=True)
