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
from tests.mark_utils import arg_mark
import pytest
import numpy as np
import mindspore as ms
from mindspore import ops
from mindspore.ops.auto_generate import not_equal
import tests.st.utils.test_utils as test_utils


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype), np.random.randn(*shape).astype(dtype)


def generate_expect_forward_output(x, other):
    return np.not_equal(x, other)


@test_utils.run_with_cell
def not_equal_forward_func(x, other):
    return not_equal(x, other)


@test_utils.run_with_cell
def not_equal_backward_func(x, other):
    return ops.grad(not_equal_forward_func, (0, 1))(x, other)


@test_utils.run_with_cell
def not_equal_vmap_func(x, other):
    return ops.vmap(not_equal_forward_func)(x, other)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_not_equal_normal(context_mode):
    """
    Feature: pyboost function.
    Description: test function not_equal forward.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x, other = generate_random_input((2, 3, 4, 5), np.float32)
    output = not_equal_forward_func(ms.Tensor(x), ms.Tensor(other))
    expect = generate_expect_forward_output(x, other)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)

    x, other = generate_random_input((2, 3, 4, 5), np.float32)
    not_equal_backward_func(ms.Tensor(x), ms.Tensor(other))


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_not_equal_bf16(context_mode):
    """
    Feature: pyboost function.
    Description: test function not_equal backward.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)

    x_np = np.random.randn(1, 4096)
    x = ms.Tensor(x_np, ms.bfloat16)
    other = ms.Tensor(0, ms.bfloat16)

    output = not_equal_forward_func(x, other)
    expect = generate_expect_forward_output(x_np, 0)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_not_equal_vmap(context_mode):
    """
    Feature: pyboost function.
    Description: test function not_equal vmap feature.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x, other = generate_random_input((2, 3, 4, 5), np.float32)
    output = not_equal_vmap_func(ms.Tensor(x), ms.Tensor(other))
    expect = generate_expect_forward_output(x, other)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_not_equal_forward_dynamic_shape(context_mode):
    """
    Feature: pyboost function.
    Description: test function not_equal forward with dynamic shape.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x_dyn = ms.Tensor(shape=[None, None, None, None], dtype=ms.float32)
    y_dyn = ms.Tensor(shape=[None, None, None, None], dtype=ms.float32)
    test_cell = test_utils.to_cell_obj(not_equal_forward_func)
    test_cell.set_inputs(x_dyn, y_dyn)
    x1, other1 = generate_random_input((2, 3, 4, 5), np.float32)
    output = test_cell(ms.Tensor(x1), ms.Tensor(other1))
    expect = generate_expect_forward_output(x1, other1)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)
    x2, other2 = generate_random_input((3, 4, 5, 6), np.float32)
    output = test_cell(ms.Tensor(x2), ms.Tensor(other2))
    expect = generate_expect_forward_output(x2, other2)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_not_equal_forward_dynamic_rank(context_mode):
    """
    Feature: pyboost function.
    Description: test function not_equal forward with dynamic rank.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x_dyn = ms.Tensor(shape=None, dtype=ms.float32)
    y_dyn = ms.Tensor(shape=None, dtype=ms.float32)
    test_cell = test_utils.to_cell_obj(not_equal_forward_func)
    test_cell.set_inputs(x_dyn, y_dyn)
    x1, other1 = generate_random_input((2, 3, 4, 5), np.float32)
    output = test_cell(ms.Tensor(x1), ms.Tensor(other1))
    expect = generate_expect_forward_output(x1, other1)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)
    x2, other2 = generate_random_input((3, 4, 5, 6), np.float32)
    output = test_cell(ms.Tensor(x2), ms.Tensor(other2))
    expect = generate_expect_forward_output(x2, other2)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)
