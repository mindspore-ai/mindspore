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
from tests.mark_utils import arg_mark
import pytest
import numpy as np
import mindspore as ms
from mindspore import ops
from mindspore.mint import less_equal
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP

import tests.st.utils.test_utils as test_utils


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype), np.random.randn(*shape).astype(dtype)


def generate_expect_forward_output(x, other):
    return np.less_equal(x, other)


@test_utils.run_with_cell
def less_equal_forward_func(x, other):
    return less_equal(x, other)


@test_utils.run_with_cell
def less_equal_backward_func(x, other):
    return ops.grad(less_equal_forward_func, (0, 1))(x, other)


@test_utils.run_with_cell
def less_equal_vmap_func(x, other):
    return ops.vmap(less_equal_forward_func)(x, other)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_less_equal_forward(context_mode):
    """
    Feature: pyboost function.
    Description: test function less_equal forward.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x, other = generate_random_input((8192, 2048), np.float32)
    output = less_equal_forward_func(ms.Tensor(x), ms.Tensor(other))
    expect = generate_expect_forward_output(x, other)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_less_equal_forward_case01(context_mode):
    """
    Feature: pyboost function.
    Description: test function less_equal forward.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x = np.array(1.0)
    other = np.random.randn(8192).astype(np.float32)
    output = less_equal_forward_func(ms.Tensor(x), ms.Tensor(other))
    expect = generate_expect_forward_output(x, other)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)

    x, _ = generate_random_input((2, 3, 4, 5), np.float32)
    other, _ = generate_random_input((2, 3, 4, 1), np.float32)
    output = less_equal_forward_func(ms.Tensor(x), ms.Tensor(other))
    expect = generate_expect_forward_output(x, other)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_less_equal_number_forward(context_mode):
    """
    Feature: pyboost function.
    Description: test function less_equal forward.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x, _ = generate_random_input((2, 3, 4, 5), np.float32)
    other = 5
    output = less_equal_forward_func(ms.Tensor(x), other)
    expect = generate_expect_forward_output(x, other)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_less_equal_bool_forward(context_mode):
    """
    Feature: pyboost function.
    Description: test function less_equal forward.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x, _ = generate_random_input((2, 3, 4, 5), np.float32)
    other = True
    output = less_equal_forward_func(ms.Tensor(x), other)
    expect = generate_expect_forward_output(x, other)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_less_equal_number_to_number_forward(context_mode):
    """
    Feature: pyboost function.
    Description: test function less_equal forward.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x = 5
    other = 6
    output = less_equal_forward_func(x, other)
    expect = generate_expect_forward_output(x, other)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_less_equal_backward(context_mode):
    """
    Feature: pyboost function.
    Description: test function less_equal backward.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x, other = generate_random_input((2, 3, 4, 5), np.float32)
    less_equal_backward_func(ms.Tensor(x), ms.Tensor(other))


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_less_equal_vmap(context_mode):
    """
    Feature: pyboost function.
    Description: test function less_equal vmap feature.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x, other = generate_random_input((2, 3, 4, 5), np.float32)
    output = less_equal_vmap_func(ms.Tensor(x), ms.Tensor(other))
    expect = generate_expect_forward_output(x, other)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_less_equal_dynamic_shape():
    """
    Feature: Test less_equal with dynamic shape in graph mode.
    Description: call less_equal with valid input and index.
    Expectation: return the correct value.
    """

    x1, other1 = generate_random_input((2, 3, 4), np.float32)
    x2, other2 = generate_random_input((3, 4, 5, 6), np.float32)

    TEST_OP(less_equal_forward_func,
            [[ms.Tensor(x1), ms.Tensor(other1)], [ms.Tensor(x2), ms.Tensor(other2)]], 'less_equal')
    