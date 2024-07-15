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
from mindspore.mint import cosh
from tests.st.utils import test_utils
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)


def generate_expect_forward_output(x):
    return np.cosh(x)


@test_utils.run_with_cell
def cosh_forward_func(x):
    return cosh(x)


@test_utils.run_with_cell
def cosh_backward_func(x):
    return ops.grad(cosh_forward_func, (0))(x)


@test_utils.run_with_cell
def cosh_vmap_func(x):
    return ops.vmap(cosh_forward_func)(x)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_ops_cosh_forward_backward(mode):
    """
    Feature: pyboost function.
    Description: test function cosh forward and backward.
    Expectation: expect correct result.
    """
    x = generate_random_input((2, 3, 4, 5), np.float32)
    if mode == 'pynative':
        ms.set_context(mode=ms.PYNATIVE_MODE)
        output = cosh_forward_func(ms.Tensor(x))
    else:
        output = (jit(cosh_forward_func, jit_config=JitConfig(jit_level="O0")))(ms.Tensor(x))
    expect = generate_expect_forward_output(x)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)

    x = np.array([[1., 2., 3.], [4., 5., 6.]]).astype(np.float32)
    if mode == 'pynative':
        ms.set_context(mode=ms.PYNATIVE_MODE)
        output = cosh_backward_func(ms.Tensor(x))
    else:
        output = (jit(cosh_backward_func, jit_config=JitConfig(jit_level="O0")))(ms.Tensor(x))
    expect = np.array([[1.175201, 3.62682, 10.017875], [27.289917, 74.20321, 201.71315]])
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_ops_cosh_vmap(mode):
    """
    Feature: pyboost function.
    Description: test function cosh vmap feature.
    Expectation: expect correct result.
    """
    x = generate_random_input((2, 3, 4, 5), np.float32)
    if mode == 'pynative':
        ms.set_context(mode=ms.PYNATIVE_MODE)
        output = cosh_vmap_func(ms.Tensor(x))
    else:
        output = (jit(cosh_vmap_func, jit_config=JitConfig(jit_level="O0")))(ms.Tensor(x))
    expect = generate_expect_forward_output(x)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
def test_ops_cosh_dynamic_shape():
    """
    Feature: pyboost function.
    Description: test function cosh with dynamic shape and dynamic rank.
    Expectation: return the correct value.
    """
    x1 = generate_random_input((2, 3, 4, 5), np.float32)
    x2 = generate_random_input((4, 5), np.float32)
    TEST_OP(cosh_forward_func, [[ms.Tensor(x1)], [ms.Tensor(x2)]], 'cosh')
