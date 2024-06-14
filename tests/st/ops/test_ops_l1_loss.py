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
from mindspore.mint.nn.functional import l1_loss
import mindspore as ms
import tests.st.utils.test_utils as test_utils
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)

def get_input():
    inputx = ms.Tensor(np.array([[1, 2, 3], [4, 5, 6]]), ms.float32)
    target = ms.Tensor(np.array([[6, 5, 4], [3, 2, 1]]), ms.float32)
    return inputx, target

def get_output_forward(reduction):
    output_mean = np.array([3.0])
    output_sum = np.array([18.0])
    output_none = np.array([[5.0, 3.0, 1.0], [1.0, 3.0, 5.0]])
    output = {"mean": output_mean, "sum": output_sum, "none": output_none}
    return output[reduction]

def get_output_backward(reduction):
    output_mean = np.array([[-0.16667, -0.16667, -0.16667], [0.16667, 0.16667, 0.16667]])
    output_sum = np.array([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]])
    output_none = np.array([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]])
    output = {"mean": output_mean, "sum": output_sum, "none": output_none}
    return output[reduction]

@test_utils.run_with_cell
def l1_loss_forward_func(inputx, target, reduction="mean"):
    return l1_loss(inputx, target, reduction)


@test_utils.run_with_cell
def l1_loss_backward_func(inputx, target, reduction="mean"):
    grad_op = ops.grad(l1_loss_forward_func, (0, 1, 2))
    return grad_op(inputx, target, reduction)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize("mode", ["pynative", "KBK"])
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_ops_l1_loss_forward(mode, reduction):
    """
    Feature: pyboost function.
    Description: test function l1_loss forward.
    Expectation: expect correct result.
    """
    inputx, target = get_input()
    expect_forward_value = get_output_forward(reduction)
    expect_backward_value = get_output_backward(reduction)

    if mode == "pynative":
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        output_forward_value = l1_loss_forward_func(inputx, target, reduction)
        output_backward_value = l1_loss_backward_func(inputx, target, reduction)
    elif mode == "KBK":
        ms.context.set_context(mode=ms.GRAPH_MODE)
        forward_op = ms.jit(l1_loss_forward_func, jit_config=ms.JitConfig(jit_level="O0"))
        backward_op = ms.jit(l1_loss_backward_func, jit_config=ms.JitConfig(jit_level="O0"))
        output_forward_value = forward_op(inputx, target, reduction)
        output_backward_value = backward_op(inputx, target, reduction)

    np.testing.assert_allclose(output_forward_value.asnumpy(), expect_forward_value, rtol=1e-3)
    np.testing.assert_allclose(output_backward_value[0].asnumpy(), expect_backward_value, rtol=1e-3)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize("context_mode", [ms.PYNATIVE_MODE])
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_ops_l1_loss_dynamic_shape(context_mode, reduction):
    """
    Feature: pyboost function.
    Description: test function l1_loss forward with dynamic shape.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x1 = ms.Tensor(generate_random_input((7, 8, 9), np.float32))
    target1 = ms.Tensor(generate_random_input((7, 8, 9), np.float32))

    x2 = ms.Tensor(generate_random_input((8, 9), np.float32))
    target2 = ms.Tensor(generate_random_input((8, 9), np.float32))

    test_cell = test_utils.to_cell_obj(l1_loss_forward_func)
    TEST_OP(test_cell, [[x1, target1, reduction], [x2, target2, reduction]], "l1_loss_ext", disable_grad=False,
            disable_input_check=True)
