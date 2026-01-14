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
from mindspore import ops, mint, jit
from tests.st.utils import test_utils
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.mark_utils import arg_mark

def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)

def generate_expect_forward_output(x):
    return np.round(x)

def generate_expect_forward_output_with_decimals(x, decimals):
    return np.round(x, decimals)

def generate_expect_backward_output(x):
    return np.zeros(x.shape).astype(np.float32)

@test_utils.run_with_cell
def round_forward_func(x):
    return ops.round(x)

@test_utils.run_with_cell
def round_forward_func_with_decimals(x, decimals_):
    return ops.round(x, decimals=decimals_)

@test_utils.run_with_cell
def round_forward_func_mint_specials(x):
    return mint.special.round(x)

@test_utils.run_with_cell
def round_backward_func(x):
    return ms.grad(round_forward_func, (0))(x)

@test_utils.run_with_cell
def round_vmap_func(x):
    return ops.vmap(round_forward_func)(x)

@arg_mark(plat_marks=['platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0',
          card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_round_normal_cpu_gpu(context_mode):
    """
    Feature: pyboost function.
    Description: test function round forward and backward on GPU and CPU.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)

    x = generate_random_input((2, 3, 4, 5), np.float32)
    output = round_forward_func(ms.Tensor(x))
    expect = generate_expect_forward_output(x)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)

    x = generate_random_input((2, 3, 4, 5), np.float32)
    output = round_backward_func(ms.Tensor(x))
    expect = generate_expect_backward_output(x)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1',
          card_mark='onecard', essential_mark='essential', mem_peak=4.0)
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_round_normal_ascend_pynative_kbk(context_mode):
    """
    Feature: pyboost function.
    Description: test function round forward and backward on Ascend.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x = generate_random_input((2, 3, 4, 5), np.float32)
    output = round_forward_func(ms.Tensor(x))
    expect = generate_expect_forward_output(x)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)

    x = generate_random_input((2, 3, 4, 5), np.float32)
    output = round_forward_func_mint_specials(ms.Tensor(x))
    expect = generate_expect_forward_output(x)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)

    x = generate_random_input((2, 3, 4, 5), np.float32)
    output = round_backward_func(ms.Tensor(x))
    expect = generate_expect_backward_output(x)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)

    decimals = 2
    x = generate_random_input((2, 3, 4, 5), np.float32)
    output = round_forward_func_with_decimals(ms.Tensor(x), decimals)
    expect = generate_expect_forward_output_with_decimals(x, decimals)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)

    x = generate_random_input((2, 3, 4, 5), np.float32)
    output = round_forward_func_with_decimals(ms.Tensor(x), ms.Tensor(decimals))
    expect = generate_expect_forward_output_with_decimals(x, decimals)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)

    decimals = -2
    x = generate_random_input((2, 3, 4, 5), np.float32)
    x = x * 1000
    output = round_forward_func_with_decimals(ms.Tensor(x), decimals)
    expect = generate_expect_forward_output_with_decimals(x, decimals)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)

@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
          card_mark='onecard', essential_mark='essential', mem_peak=1024.01)
@pytest.mark.parametrize('context_mode', ['GE'])
def test_ops_round_normal_ascend_GE(context_mode):
    """
    Feature: pyboost function.
    Description: test function round in GE mode in Ascend.
    Expectation: expect correct result.
    """
    x = generate_random_input((2, 3, 4, 5), np.float32)
    output = (jit(round_forward_func, backend="GE"))(ms.Tensor(x))
    expect = generate_expect_forward_output(x)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)

    x = generate_random_input((2, 3, 4, 5), np.float32)
    output = (jit(round_backward_func, backend="GE"))(ms.Tensor(x))
    expect = generate_expect_backward_output(x)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)

@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='essential', mem_peak=1024.01)
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_round_vmap(context_mode):
    """
    Feature: pyboost function.
    Description: test function round vmap feature.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x = generate_random_input((2, 3, 4, 5), np.float32)
    output = round_vmap_func(ms.Tensor(x))
    expect = generate_expect_forward_output(x)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)

@arg_mark(plat_marks=['platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_round_dynamic_shape_cpu_gpu(context_mode):
    """
    Feature: pyboost function.
    Description: test function round with dynamic shape on GPU and CPU.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x1 = generate_random_input((2, 3, 4, 5), np.float32)
    x2 = generate_random_input((6, 7, 8), np.float32)
    TEST_OP(round_forward_func, [[ms.Tensor(x1)], [ms.Tensor(x2)]])

@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
          card_mark='onecard', essential_mark='essential', mem_peak=3072.01)
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_round_dynamic_shape_ascend(context_mode):
    """
    Feature: pyboost function.
    Description: test function round with dynamic shape on Ascend.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x1 = generate_random_input((2, 3, 4, 5), np.float32)
    x2 = generate_random_input((6, 7, 8), np.float32)
    TEST_OP(round_forward_func, [[ms.Tensor(x1)], [ms.Tensor(x2)]])

    x1 = generate_random_input((2, 3, 4, 5), np.float32)
    x2 = generate_random_input((6, 7, 8), np.float32)
    TEST_OP(round_forward_func_with_decimals, [[ms.Tensor(x1), 3], [ms.Tensor(x2), 2]])
