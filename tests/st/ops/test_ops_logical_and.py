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
import numpy as np
import pytest
import mindspore as ms
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP
from tests.st.utils import test_utils
from tests.mark_utils import arg_mark


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype), np.random.randn(*shape).astype(dtype)


def generate_expect_forward_output(x, other):
    return np.logical_and(x, other)


@test_utils.run_with_cell
def logical_and_forward_func(x, other):
    return ms.ops.logical_and(x, other)


@test_utils.run_with_cell
def logical_and_backward_func(x, other):
    return ms.ops.grad(logical_and_forward_func, 0)(x, other)


@test_utils.run_with_cell
def logical_and_vmap_func(x, other):
    return ms.ops.vmap(logical_and_forward_func, in_axes=0, out_axes=0)(x, other)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0',
          card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_logical_and_forward(context_mode):
    """
    Feature: pyboost function.
    Description: test function logical_and forward.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x, other = generate_random_input((2, 3, 4, 5), np.float32)
    output = logical_and_forward_func(ms.Tensor(x), ms.Tensor(other))
    expect = generate_expect_forward_output(x, other)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_logical_and_vmap(context_mode):
    """
    Feature: pyboost function.
    Description: test function logical_and vmap feature.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x, other = generate_random_input((2, 3, 4, 5), np.float32)
    output = logical_and_vmap_func(ms.Tensor(x), ms.Tensor(other))
    expect = generate_expect_forward_output(x, other)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_logical_and_forward_dynamic_shape(context_mode):
    """
    Feature: pyboost function.
    Description: test function logical_and forward with dynamic shape.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x_dyn = ms.Tensor(shape=[None, None, None, None], dtype=ms.float32)
    y_dyn = ms.Tensor(shape=[None, None, None, None], dtype=ms.float32)
    test_cell = test_utils.to_cell_obj(logical_and_forward_func)
    test_cell.set_inputs(x_dyn, y_dyn)
    x1, other1 = generate_random_input((2, 3, 4, 5), np.float32)
    output = test_cell(ms.Tensor(x1), ms.Tensor(other1))
    expect = generate_expect_forward_output(x1, other1)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)
    x2, other2 = generate_random_input((3, 4, 5, 6), np.float32)
    output = test_cell(ms.Tensor(x2), ms.Tensor(other2))
    expect = generate_expect_forward_output(x2, other2)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_logical_and_forward_dynamic_rank(context_mode):
    """
    Feature: pyboost function.
    Description: test function logical_and forward with dynamic rank.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x_dyn = ms.Tensor(shape=None, dtype=ms.float32)
    y_dyn = ms.Tensor(shape=None, dtype=ms.float32)
    test_cell = test_utils.to_cell_obj(logical_and_forward_func)
    test_cell.set_inputs(x_dyn, y_dyn)
    x1, other1 = generate_random_input((2, 3, 4, 5), np.float32)
    output = test_cell(ms.Tensor(x1), ms.Tensor(other1))
    expect = generate_expect_forward_output(x1, other1)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)
    x2, other2 = generate_random_input((3, 4, 5, 6), np.float32)
    output = test_cell(ms.Tensor(x2), ms.Tensor(other2))
    expect = generate_expect_forward_output(x2, other2)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential')
def test_logical_and():
    """
    Feature: Test logical_and op.
    Description: Test logical_and dynamic shape.
    Expectation: the result match with expected result.
    """
    input_case1 = ms.Tensor(generate_random_input((3, 4, 5, 6), np.float32))
    input_case2 = ms.Tensor(generate_random_input((3, 4, 5, 6), np.float32))
    input_case3 = ms.Tensor(generate_random_input((3, 4), np.float32))
    input_case4 = ms.Tensor(generate_random_input((3, 4), np.float32))
    TEST_OP(logical_and_forward_func, [[input_case1, input_case2], [input_case3, input_case4]], 'logical_and',
            disable_grad=True)
