# Copyright 2024 Huawei Technonarrowies Co., Ltd
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
import mindspore as ms
from mindspore import ops, jit, JitConfig
from mindspore.mint import narrow
import tests.st.utils.test_utils as test_utils
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)


def generate_expect_forward_output(x, dim, start, length):
    condition = np.zeros(x.shape[dim])
    if start < 0:
        start += x.shape[dim]
    condition[start:start+length] = 1
    return np.compress(condition, x, axis=dim)


@test_utils.run_with_cell
def narrow_forward_func(x, dim, start, length):
    return narrow(x, dim, start, length)


@test_utils.run_with_cell
def narrow_backward_func(x, dim, start, length):
    return ops.grad(narrow_forward_func, (0))(x, dim, start, length)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('context_mode', ['pynative', 'KBK'])
def test_ops_narrow_forward(context_mode):
    """
    Feature: pyboost function.
    Description: test function narrow forward.
    Expectation: expect correct result.
    """
    x = generate_random_input((3, 960, 64, 64), np.float16)
    dim = 2
    start = 0
    length = 64
    expect_forward = generate_expect_forward_output(x, dim, start, length)
    expect_grad = np.zeros_like(x)
    expect_grad[:, :, start:start+length, :] = 1

    if context_mode == 'pynative':
        ms.set_context(mode=ms.PYNATIVE_MODE)
        output_forward = narrow_forward_func(ms.Tensor(x), dim, start, length)
        output_grad = narrow_backward_func(ms.Tensor(x), dim, start, length)
    else:
        output_forward = \
            (jit(narrow_forward_func, jit_config=JitConfig(jit_level="O0")))(ms.Tensor(x), dim, start, length)
        output_grad = \
            (jit(narrow_backward_func, jit_config=JitConfig(jit_level="O0")))(ms.Tensor(x), dim, start, length)

    np.testing.assert_allclose(output_forward.asnumpy(), expect_forward, rtol=1e-3)
    np.testing.assert_allclose(output_grad.asnumpy(), expect_grad, rtol=1e-3)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('context_mode', ['pynative', 'KBK'])
def test_ops_narrow_forward_case01(context_mode):
    """
    Feature: pyboost function.
    Description: test function narrow forward.
    Expectation: expect correct result.
    """
    x = generate_random_input((3, 1920, 32, 32), np.float16)
    dim = 1
    start = 1280
    length = 640
    expect_forward = generate_expect_forward_output(x, dim, start, length)
    expect_grad = np.zeros_like(x)
    expect_grad[:, start:start+length, :, :] = 1

    if context_mode == 'pynative':
        ms.set_context(mode=ms.PYNATIVE_MODE)
        output_forward = narrow_forward_func(ms.Tensor(x), dim, start, length)
        output_grad = narrow_backward_func(ms.Tensor(x), dim, start, length)
    else:
        output_forward = \
            (jit(narrow_forward_func, jit_config=JitConfig(jit_level="O0")))(ms.Tensor(x), dim, start, length)
        output_grad = \
            (jit(narrow_backward_func, jit_config=JitConfig(jit_level="O0")))(ms.Tensor(x), dim, start, length)

    np.testing.assert_allclose(output_forward.asnumpy(), expect_forward, rtol=1e-3)
    np.testing.assert_allclose(output_grad.asnumpy(), expect_grad, rtol=1e-3)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
def test_ops_narrow_backward_dynamic_shape():
    """
    Feature: pyboost function.
    Description: test function narrow backward with dynamic shape.
    Expectation: expect correct result.
    """
    x1 = generate_random_input((2, 3, 4, 5), np.float32)
    dim1 = 1
    start1 = 0
    length1 = 2

    x2 = generate_random_input((2, 4, 5), np.float32)
    dim2 = 2
    start2 = 0
    length2 = 5

    TEST_OP(narrow_forward_func, [[ms.Tensor(x1), dim1, start1, length1], [ms.Tensor(x2), dim2, start2, length2]],
            '', disable_input_check=True, disable_yaml_check=True, disable_mode=['GRAPH_MODE'])
