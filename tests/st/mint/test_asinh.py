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
import pytest
import numpy as np
import mindspore as ms
from mindspore import ops, mint, jit, JitConfig
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)


def generate_expect_forward_output(x):
    return np.arcsinh(x)


def generate_expect_backward_output(x):
    return 1 / np.sqrt(np.square(x) + 1)


def asinh_forward_func(x):
    return mint.asinh(x)


def asinh_backward_func(x):
    return ops.grad(asinh_forward_func, (0,))(x)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_asinh_std(mode):
    """
    Feature: standard forward, backward features.
    Description: test function asinh.
    Expectation: expect correct result.
    """
    x = generate_random_input((2, 3, 4), np.float32)
    expect = generate_expect_forward_output(x)
    expect_grad = generate_expect_backward_output(x)
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        output = asinh_forward_func(ms.Tensor(x))
        output_grad = asinh_backward_func(ms.Tensor(x))
    else:
        output = (jit(asinh_forward_func, jit_config=JitConfig(jit_level="O0")))(ms.Tensor(x))
        output_grad = (jit(asinh_backward_func, jit_config=JitConfig(jit_level="O0")))(ms.Tensor(x))

    np.allclose(output.asnumpy(), expect, rtol=1e-5, equal_nan=True)
    np.allclose(output_grad.asnumpy(), expect_grad, rtol=1e-5, equal_nan=True)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
def test_asinh_dynamic_shape():
    """
    Feature: dynamic shape forward, backward features.
    Description: test asinh forward with dynamic shape.
    Expectation: expect correct result.
    """
    tensor_1 = ms.Tensor(generate_random_input((2, 3), np.float32))
    tensor_2 = ms.Tensor(generate_random_input((3, 4, 5), np.float32))

    TEST_OP(asinh_forward_func, [[tensor_1], [tensor_2]], 'asinh_ext', disable_mode=['GRAPH_MODE'])


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_asinh_bfloat16(mode):
    """
    Feature: test asinh functional API.
    Description: testcase for asinh functional API.
    Expectation: the result match with expected result.
    """
    x = generate_random_input((2, 3), np.float32)
    expect = generate_expect_forward_output(x).astype(np.float32)
    expect_grad = generate_expect_backward_output(x).astype(np.float32)

    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        output = asinh_forward_func(ms.Tensor(x, dtype=ms.bfloat16))
        output_grad = asinh_backward_func(ms.Tensor(x, dtype=ms.bfloat16))
    else:
        output = (jit(asinh_forward_func, jit_config=JitConfig(jit_level="O0")))(ms.Tensor(x, dtype=ms.bfloat16))
        output_grad = (jit(asinh_backward_func, jit_config=JitConfig(jit_level="O0")))(ms.Tensor(x, dtype=ms.bfloat16))

    np.allclose(output.float().asnumpy(), expect, 0.004, 0.004, equal_nan=True)
    np.allclose(output_grad.float().asnumpy(), expect_grad, 0.004, 0.004, equal_nan=True)
