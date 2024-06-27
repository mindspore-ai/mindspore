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
import mindspore.ops as ops
from mindspore import Tensor, mint
import tests.st.utils.test_utils as test_utils
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP


def generate_random_input(shape, dtype):
    return np.random.randint(1, 10, size=shape).astype(dtype)

def generate_expect_forward_output(x, shifts, dims):
    return np.roll(x, shifts, dims)

def generate_expect_backward_output(shifts, dims, grad):
    if isinstance(shifts, (tuple, list)):
        neg_shifts = [-i for i in shifts]
    else:
        neg_shifts = -shifts
    return np.roll(grad, neg_shifts, dims)

@test_utils.run_with_cell
def roll_forward_func(x, shifts, dims):
    return mint.roll(x, shifts, dims)

@test_utils.run_with_cell
def roll_backward_func(x, shifts, dims, grad):
    return ops.GradOperation(sens_param=True, get_all=True)(roll_forward_func)(x, shifts, dims, grad)
    #return ops.grad(roll_forward_func, (0, 1, 2))(x, shifts, dims)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.parametrize("mode", ["pynative", "KBK"])
def test_ops_roll_normal(mode):
    """
    Feature: pyboost function.
    Description: test function roll_ext forward.
    Expectation: expect correct result.
    """
    x1 = generate_random_input((2, 3, 4), np.float32)
    shifts1 = 3
    dims1 = None

    x2 = generate_random_input((2, 3, 4), np.float32)
    shifts2 = 3
    dims2 = 1

    x3 = generate_random_input((2, 3, 4), np.float32)
    shifts3 = (3, 2)
    dims3 = (1, 2)

    grad = generate_random_input((2, 3, 4), np.float32)

    expect_out1 = generate_expect_forward_output(x1, shifts1, dims1)
    expect_out2 = generate_expect_forward_output(x2, shifts2, dims2)
    expect_out3 = generate_expect_forward_output(x3, shifts3, dims3)

    expect_dout1 = generate_expect_backward_output(shifts1, dims1, grad)
    expect_dout2 = generate_expect_backward_output(shifts2, dims2, grad)
    expect_dout3 = generate_expect_backward_output(shifts3, dims3, grad)

    if mode == "pynative":
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        out1 = roll_forward_func(Tensor(x1), shifts1, dims1)
        out2 = roll_forward_func(Tensor(x2), shifts2, dims2)
        out3 = roll_forward_func(Tensor(x3), shifts3, dims3)
        dout1 = roll_backward_func(Tensor(x1), shifts1, dims1, Tensor(grad))
        dout2 = roll_backward_func(Tensor(x2), shifts2, dims2, Tensor(grad))
        dout3 = roll_backward_func(Tensor(x3), shifts3, dims3, Tensor(grad))
    elif mode == "KBK":
        ms.context.set_context(mode=ms.GRAPH_MODE)
        op = ms.jit(roll_forward_func, jit_config=ms.JitConfig(jit_level="O0"))
        out1 = op(Tensor(x1), shifts1, dims1)
        out2 = op(Tensor(x2), shifts2, dims2)
        out3 = op(Tensor(x3), shifts3, dims3)
        op = ms.jit(roll_backward_func, jit_config=ms.JitConfig(jit_level="O0"))
        dout1 = op(Tensor(x1), shifts1, dims1, Tensor(grad))
        dout2 = op(Tensor(x2), shifts2, dims2, Tensor(grad))
        dout3 = op(Tensor(x3), shifts3, dims3, Tensor(grad))
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE)
        out1 = roll_forward_func(Tensor(x1), shifts1, dims1)
        out2 = roll_forward_func(Tensor(x2), shifts2, dims2)
        out3 = roll_forward_func(Tensor(x3), shifts3, dims3)
        dout1 = roll_backward_func(Tensor(x1), shifts1, dims1, Tensor(grad))
        dout2 = roll_backward_func(Tensor(x2), shifts2, dims2, Tensor(grad))
        dout3 = roll_backward_func(Tensor(x3), shifts3, dims3, Tensor(grad))

    np.testing.assert_allclose(out1.asnumpy(), expect_out1, rtol=1e-3)
    np.testing.assert_allclose(out2.asnumpy(), expect_out2, rtol=1e-3)
    np.testing.assert_allclose(out3.asnumpy(), expect_out3, rtol=1e-3)
    np.testing.assert_allclose(dout1[0].asnumpy(), expect_dout1, rtol=1e-3)
    np.testing.assert_allclose(dout2[0].asnumpy(), expect_dout2, rtol=1e-3)
    np.testing.assert_allclose(dout3[0].asnumpy(), expect_dout3, rtol=1e-3)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
def test_ops_roll_dynamic_shape():
    """
    Feature: pyboost function.
    Description: test function roll_ext forward with dynamic shape.
    Expectation: expect correct result.
    """
    x1 = Tensor(generate_random_input((2, 3, 4), np.float32))
    shifts1 = (3, 1)
    dims1 = (1, 2)

    x2 = Tensor(generate_random_input((2, 3, 4, 5), np.float32))
    shifts2 = (2, 3)
    dims2 = (2, 3)

    test_cell = test_utils.to_cell_obj(roll_forward_func)
    TEST_OP(test_cell, [[x1, shifts1, dims1], [x2, shifts2, dims2]], "roll", disable_mode=['GRAPH_MODE'])
