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
import pytest
import numpy as np
import mindspore as ms
from mindspore import ops, jit, JitConfig
from mindspore.mint import sinc
from tests.st.utils import test_utils
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)


def generate_expect_forward_output(x):
    return np.sinc(x)


@test_utils.run_with_cell
def sinc_forward_func(x):
    return sinc(x)


@test_utils.run_with_cell
def sinc_backward_func(x):
    return ops.grad(sinc_forward_func, (0))(x)


@test_utils.run_with_cell
def sinc_vmap_func(x):
    return ops.vmap(sinc_forward_func)(x)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_ops_sinc_forward_backward(mode):
    """
    Feature: pyboost function.
    Description: test function sinc forward and backward.
    Expectation: expect correct result.
    """
    x = generate_random_input((2, 3, 4, 5), np.float32)
    if mode == 'pynative':
        ms.set_context(mode=ms.PYNATIVE_MODE)
        output = sinc_forward_func(ms.Tensor(x))
    else:
        output = (jit(sinc_forward_func, jit_config=JitConfig(jit_level="O0")))(ms.Tensor(x))
    expect = generate_expect_forward_output(x)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)

    x = np.array([[1., 2., 3.], [4., 5., 6.]]).astype(np.float32)
    if mode == 'pynative':
        ms.set_context(mode=ms.PYNATIVE_MODE)
        output = sinc_backward_func(ms.Tensor(x))
    else:
        output = (jit(sinc_backward_func, jit_config=JitConfig(jit_level="O0")))(ms.Tensor(x))
    expect = np.array([[-1., 0.5, -0.3333], [0.25, -0.2, 0.166667]])
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_ops_sinc_vmap(mode):
    """
    Feature: pyboost function.
    Description: test function sinc vmap feature.
    Expectation: expect correct result.
    """
    x = generate_random_input((2, 3, 4, 5), np.float32)
    if mode == 'pynative':
        ms.set_context(mode=ms.PYNATIVE_MODE)
        output = sinc_vmap_func(ms.Tensor(x))
    else:
        output = (jit(sinc_vmap_func, jit_config=JitConfig(jit_level="O0")))(ms.Tensor(x))
    expect = generate_expect_forward_output(x)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
def test_ops_sinc_dynamic_shape():
    """
    Feature: pyboost function.
    Description: test function sinc with dynamic shape and dynamic rank.
    Expectation: return the correct value.
    """
    x1 = generate_random_input((2, 3, 4, 5), np.float32)
    x2 = generate_random_input((4, 5), np.float32)
    TEST_OP(sinc_forward_func, [[ms.Tensor(x1)], [ms.Tensor(x2)]], 'sinc')
