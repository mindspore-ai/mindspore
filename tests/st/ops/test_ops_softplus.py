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
from mindspore import ops
from mindspore.mint.nn.functional import softplus
from tests.st.utils import test_utils
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)


def generate_expect_forward_output(x, beta=1, threshold=20):
    sacling_input = beta * x
    output = (1 / beta) * np.log(1 + np.exp(sacling_input))
    return np.select(sacling_input > threshold, x, output)


@test_utils.run_with_cell
def softplus_forward_func(x, beta=1, threshold=20):
    return softplus(x, beta, threshold)


@test_utils.run_with_cell
def softplus_backward_func(x, beta=1, threshold=20):
    return ops.grad(softplus_forward_func, (0))(x, beta, threshold)


@test_utils.run_with_cell
def softplus_vmap_func(x, beta=1, threshold=20):
    return ops.vmap(softplus_forward_func, in_axes=(0, None, None), out_axes=0)(x, beta, threshold)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_softplus_forward(context_mode):
    """
    Feature: pyboost function.
    Description: test function softplus forward.
    Expectation: expect correct result.
    """
    ms.set_context(jit_level='O0')
    ms.context.set_context(mode=context_mode)
    x = generate_random_input((2, 3, 4, 5), np.float32)
    output = softplus_forward_func(ms.Tensor(x))
    expect = generate_expect_forward_output(x)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-4)

    output2 = softplus_forward_func(ms.Tensor(x), 2, 12)
    expect2 = generate_expect_forward_output(x, 2, 12)
    np.testing.assert_allclose(output2.asnumpy(), expect2, rtol=1e-4)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_softplus_bf16(context_mode):
    """
    Feature: pyboost function.
    Description: test function softplus forward(bf16).
    Expectation: expect correct result.
    """
    ms.set_context(jit_level='O0')
    ms.context.set_context(mode=context_mode)
    x_tensor = ms.Tensor([0.1, 0.2, 30, 25], dtype=ms.bfloat16)
    output = softplus_forward_func(x_tensor)
    expect = np.array([0.7461, 0.7969, 30.0000, 25.0000])
    np.testing.assert_allclose(output.float().asnumpy(), expect, rtol=5e-3, atol=5e-3)

    output2 = softplus_forward_func(x_tensor, 0.3, 100)
    expect2 = np.array([2.3594, 2.4062, 30.0000, 25.0000])
    np.testing.assert_allclose(output2.float().asnumpy(), expect2, rtol=5e-3, atol=5e-3)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_softplus_backward(context_mode):
    """
    Feature: pyboost function.
    Description: test function softplus backward.
    Expectation: expect correct result.
    """
    ms.set_context(jit_level='O0')
    ms.context.set_context(mode=context_mode)
    x = np.array([0.1, 0.2, 0.3, 1, 2]).astype(np.float32)
    output1 = softplus_backward_func(ms.Tensor(x))
    expect1 = np.array([0.5249791, 0.5498339, 0.5744425, 0.7310585, 0.8807970]).astype(np.float32)
    np.testing.assert_allclose(output1.asnumpy(), expect1, rtol=1e-4)

    output2 = softplus_backward_func(ms.Tensor(x), 0.2, 0.2)
    expect2 = np.array([0.50499983, 0.5099986, 0.5149955, 0.5498339, 1.00000]).astype(np.float32)
    np.testing.assert_allclose(output2.asnumpy(), expect2, rtol=1e-4)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_softplus_vmap(context_mode):
    """
    Feature: pyboost function.
    Description: test function softplus vmap feature.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x = generate_random_input((2, 3, 4, 5), np.float32)
    output = softplus_vmap_func(ms.Tensor(x))
    expect = generate_expect_forward_output(x)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-4)

    output2 = softplus_vmap_func(ms.Tensor(x), 2, 12)
    expect2 = generate_expect_forward_output(x, 2, 12)
    np.testing.assert_allclose(output2.asnumpy(), expect2, rtol=1e-4)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
def test_softplus_dynamic_shape():
    """
    Feature: Test dynamic shape.
    Description: test function softplus  dynamic feature.
    Expectation: expect correct result.
    """
    ms_data1 = generate_random_input((2, 3, 4), np.float32)
    beta1 = 2
    threshold1 = 15
    ms_data2 = generate_random_input((3, 4, 5, 6), np.float32)
    beta2 = 3
    threshold2 = 18
    TEST_OP(softplus_forward_func,
            [[ms.Tensor(ms_data1), beta1, threshold1], [ms.Tensor(ms_data2), beta2, threshold2]], 'softplus_ext',
            disable_mode=['GRAPH_MODE'])
