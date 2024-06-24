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
from mindspore.ops.auto_generate import less
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP

import tests.st.utils.test_utils as test_utils


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype), np.random.randn(*shape).astype(dtype)


def generate_expect_forward_output(x, other):
    return np.less(x, other)


@test_utils.run_with_cell
def less_forward_func(x, other):
    return less(x, other)


@test_utils.run_with_cell
def less_backward_func(x, other):
    return ops.grad(less_forward_func, (0, 1))(x, other)


@test_utils.run_with_cell
def less_vmap_func(x, other):
    return ops.vmap(less_forward_func)(x, other)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_less_forward(context_mode):
    """
    Feature: pyboost function.
    Description: test function less forward.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x, other = generate_random_input((2, 3, 4, 5), np.float32)
    output = less_forward_func(ms.Tensor(x), ms.Tensor(other))
    expect = generate_expect_forward_output(x, other)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)

    x, _ = generate_random_input((2, 3, 4, 5), np.float32)
    other, _ = generate_random_input((2, 3, 4, 1), np.float32)
    output = less_forward_func(ms.Tensor(x), ms.Tensor(other))
    expect = generate_expect_forward_output(x, other)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_less_number_forward(context_mode):
    """
    Feature: pyboost function.
    Description: test function less forward.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x, _ = generate_random_input((2, 3, 4, 5), np.float32)
    other = 5
    output = less_forward_func(ms.Tensor(x), other)
    expect = generate_expect_forward_output(x, other)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_less_bool_forward(context_mode):
    """
    Feature: pyboost function.
    Description: test function less forward.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x, _ = generate_random_input((2, 3, 4, 5), np.float32)
    other = True
    output = less_forward_func(ms.Tensor(x), other)
    expect = generate_expect_forward_output(x, other)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_less_number_to_number_forward(context_mode):
    """
    Feature: pyboost function.
    Description: test function less forward.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x = 5
    other = 6
    output = less_forward_func(x, other)
    expect = generate_expect_forward_output(x, other)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_less_backward(context_mode):
    """
    Feature: pyboost function.
    Description: test function less backward.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x, other = generate_random_input((2, 3, 4, 5), np.float32)
    less_backward_func(ms.Tensor(x), ms.Tensor(other))


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_less_vmap(context_mode):
    """
    Feature: pyboost function.
    Description: test function less vmap feature.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x, other = generate_random_input((2, 3, 4, 5), np.float32)
    output = less_vmap_func(ms.Tensor(x), ms.Tensor(other))
    expect = generate_expect_forward_output(x, other)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_cpu_training
@pytest.mark.platform_x86_gpu_training
def test_less_dynamic_shape():
    """
    Feature: Test less with dynamic shape in graph mode.
    Description: call less with valid input and index.
    Expectation: return the correct value.
    """

    x1, other1 = generate_random_input((2, 3, 4), np.float32)
    x2, other2 = generate_random_input((3, 4, 5, 6), np.float32)

    TEST_OP(less_forward_func,
            [[ms.Tensor(x1), ms.Tensor(other1)], [ms.Tensor(x2), ms.Tensor(other2)]], 'less')
