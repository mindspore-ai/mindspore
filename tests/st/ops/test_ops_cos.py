# Copyright 2023 Huawei Technologies Co., Ltd
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
from mindspore.ops import cos

from tests.st.utils import test_utils
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)


def generate_expect_forward_output(x):
    return np.cos(x)


def generate_expect_backward_output(x):
    return -np.sin(x)


@test_utils.run_with_cell
def cos_forward_func(x):
    return cos(x)


@test_utils.run_with_cell
def cos_backward_func(x):
    return ops.grad(cos_forward_func, (0))(x)


@test_utils.run_with_cell
def cos_vmap_func(x):
    return ops.vmap(cos_forward_func)(x)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_cos_forward(context_mode):
    """
    Feature: pyboost function.
    Description: test function cos forward.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x1 = generate_random_input((64, 224), np.float32)
    output1 = cos_forward_func(ms.Tensor(x1))
    expect1 = generate_expect_forward_output(x1)
    np.testing.assert_allclose(output1.asnumpy(), expect1, rtol=1e-3, atol=1e-6)

    x2 = generate_random_input((384, 128), np.float32)
    output2 = cos_forward_func(ms.Tensor(x2))
    expect2 = generate_expect_forward_output(x2)
    np.testing.assert_allclose(output2.asnumpy(), expect2, rtol=1e-3, atol=1e-6)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_cos_backward(context_mode):
    """
    Feature: pyboost function.
    Description: test function cos backward.
    Expectation: expect correct result.
    """
    x1 = generate_random_input((64, 224), np.float32)
    output1 = cos_backward_func(ms.Tensor(x1))
    expect1 = generate_expect_backward_output(x1)
    np.testing.assert_allclose(output1.asnumpy(), expect1, rtol=1e-3)

    x2 = generate_random_input((384, 128), np.float32)
    output2 = cos_backward_func(ms.Tensor(x2))
    expect2 = generate_expect_backward_output(x2)
    np.testing.assert_allclose(output2.asnumpy(), expect2, rtol=1e-3)
    ms.context.set_context(mode=context_mode)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_cos_vmap(context_mode):
    """
    Feature: pyboost function.
    Description: test function cos vmap feature.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x = generate_random_input((2, 3, 4, 5), np.float32)
    output = cos_vmap_func(ms.Tensor(x))
    expect = generate_expect_forward_output(x)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
def test_ops_cos_dyn():
    """
    Feature: pyboost function.
    Description: test function cos with dynamic shape and dynamic rank by TEST_OP.
    Expectation: passed.
    """
    x1 = generate_random_input((4, 5, 6), np.float32)
    x2 = generate_random_input((2, 3, 4, 5), np.float32)

    TEST_OP(cos_forward_func, [[ms.Tensor(x1)], [ms.Tensor(x2)]], 'cos')
