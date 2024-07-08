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
from mindspore.mint.nn.functional import hardshrink
from tests.st.utils import test_utils
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)


def hardshrink_expect_forward_func(x, lambd=0.5):
    result = np.zeros_like(x, dtype=x.dtype)
    for index, _ in np.ndenumerate(x):
        if x[index] > lambd or x[index] < (-1 * lambd):
            result[index] = x[index]
        else:
            result[index] = 0
    return result


@test_utils.run_with_cell
def hardshrink_forward_func(x, lambd=0.5):
    return hardshrink(x, lambd)


@test_utils.run_with_cell
def hardshrink_backward_func(x, lambd=0.5):
    return ops.grad(hardshrink_forward_func, (0,))(x, lambd)


@test_utils.run_with_cell
def hardshrink_vmap_func(x, lambd=0.5):
    return ops.vmap(hardshrink_forward_func, in_axes=(0, None), out_axes=0)(x, lambd)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize('dtype', [np.float16, np.float32])
@test_utils.run_test_with_On
def test_ops_hardshrink_normal(context_mode, dtype):
    """
    Feature: pyboost function.
    Description: test function hardshrink forward.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    # forward
    x = generate_random_input((2, 3, 4, 5), dtype)
    output_f1 = hardshrink_forward_func(ms.Tensor(x))
    expect_f1 = hardshrink_expect_forward_func(x)
    np.testing.assert_allclose(output_f1.asnumpy(), expect_f1, rtol=1e-3)

    output_f2 = hardshrink_forward_func(ms.Tensor(x), 5)
    expect_f2 = hardshrink_expect_forward_func(x, 5)
    np.testing.assert_allclose(output_f2.asnumpy(), expect_f2, rtol=1e-3)

    output_f3 = hardshrink_forward_func(ms.Tensor(x), True)
    expect_f3 = hardshrink_expect_forward_func(x, True)
    np.testing.assert_allclose(output_f3.asnumpy(), expect_f3, rtol=1e-3)

    # backward
    x2 = np.array([[0.5, 1, 2.0], [0.0533, 0.0776, -2.1233]]).astype(np.float32)
    output_b2 = hardshrink_backward_func(ms.Tensor(x2))
    expect_b2 = np.array([[0., 1., 1.], [0., 0., 1]]).astype(np.float32)
    np.testing.assert_allclose(output_f2.asnumpy(), expect_f2, rtol=1e-3)

    output_b2 = hardshrink_backward_func(ms.Tensor(x2), 2)
    expect_b2 = np.array([[0., 0., 0.], [0., 0., 1]]).astype(np.float32)
    np.testing.assert_allclose(output_b2.asnumpy(), expect_b2, rtol=1e-3)

    output_b3 = hardshrink_backward_func(ms.Tensor(x2), True)
    expect_b3 = np.array([[0., 0., 1.], [0., 0., 1]]).astype(np.float32)
    np.testing.assert_allclose(output_b3.asnumpy(), expect_b3, rtol=1e-3)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_ops_hardshrink_vmap(context_mode):
    """
    Feature: pyboost function.
    Description: test function hardshrink vmap feature.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x = generate_random_input((2, 3, 4, 5), np.float32)
    output1 = hardshrink_vmap_func(ms.Tensor(x))
    expect1 = hardshrink_expect_forward_func(x)
    np.testing.assert_allclose(output1.asnumpy(), expect1, rtol=1e-3)

    output2 = hardshrink_vmap_func(ms.Tensor(x), 4)
    expect2 = hardshrink_expect_forward_func(x, 4)
    np.testing.assert_allclose(output2.asnumpy(), expect2, rtol=1e-3)

    output3 = hardshrink_vmap_func(ms.Tensor(x), False)
    expect3 = hardshrink_expect_forward_func(x, False)
    np.testing.assert_allclose(output3.asnumpy(), expect3, rtol=1e-3)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@test_utils.run_test_with_On
def test_hardshrink_dynamic_shape():
    """
    Feature: Test dynamic shape.
    Description: test function hardshrink  dynamic feature.
    Expectation: expect correct result.
    """
    ms_data1 = generate_random_input((2, 3, 4, 5), np.float32)
    lambd1 = 0.5
    ms_data2 = generate_random_input((3, 4, 5, 6, 7), np.float32)
    lambd2 = 0.6
    TEST_OP(hardshrink_forward_func
            , [[ms.Tensor(ms_data1), lambd1], [ms.Tensor(ms_data2), lambd2]]
            , 'hshrink')
