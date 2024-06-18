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
# pylint: disable=unused-variable
import numpy as np
import pytest
import mindspore as ms
from mindspore import mint, jit, JitConfig
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)


def generate_expect_forward_output(x):
    return np.linalg.inv(x)


def generate_expect_backward_output(x):
    res = generate_expect_forward_output(x)
    temp = np.matmul(np.ones_like(res, np.float32), res.T)
    out = -1 * np.matmul(res.T, temp)
    return out


def inverse_forward_func(x):
    return mint.inverse(x)


def inverse_backward_func(x):
    input_grad = ms.ops.grad(inverse_forward_func, 0)(x)
    return input_grad


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_inverse_std(mode):
    """
    Feature: mint
    Description: Verify the result of mint function
    Expectation: success
    """
    x = generate_random_input((9, 9), np.float32)

    expect_forward = generate_expect_forward_output(x)
    expect_grad = generate_expect_backward_output(x)

    if mode == 'pynative':
        ms.set_context(mode=ms.PYNATIVE_MODE)
        output_forward = inverse_forward_func(ms.Tensor(x))
        output_grad = inverse_backward_func(ms.Tensor(x))
    else:
        output_forward = (jit(inverse_forward_func, jit_config=JitConfig(jit_level="O0")))(ms.Tensor(x))
        output_grad = (jit(inverse_backward_func, jit_config=JitConfig(jit_level="O0")))(ms.Tensor(x))

    assert np.allclose(output_forward.asnumpy(), expect_forward, 1e-2, 1e-2)
    assert np.allclose(output_grad.asnumpy(), expect_grad, 5e-1, 5e-1)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_inverse_dynamic_shape(mode):
    """
    Feature: Test leaky relu with dynamic shape in graph mode.
    Description: call mint.inverse with valid input and index.
    Expectation: return the correct value.
    """
    x1 = generate_random_input((3, 3), np.float32)
    x2 = generate_random_input((2, 4, 4), np.float32)

    TEST_OP(inverse_forward_func, [[ms.Tensor(x1)], [ms.Tensor(x2)]], 'matrix_inverse_ext',
            disable_input_check=True, disable_mode=['GRAPH_MODE'])
