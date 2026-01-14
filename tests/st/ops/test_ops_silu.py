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
from mindspore.ops import silu
from tests.st.utils import test_utils
from tests.st.ops.test_tools.ops_binary_cases import ops_binary_cases, OpsBinaryCase
from tests.mark_utils import arg_mark


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def generate_expect_forward_output(x):
    return x * sigmoid(x)


def generate_expect_backward_output(x):
    return sigmoid(x) * ((1 - sigmoid(x)) * x + 1)


@test_utils.run_with_cell
def silu_forward_func(x):
    return silu(x)


@test_utils.run_with_cell
def silu_backward_func(x):
    return ops.grad(silu_forward_func, (0,))(x)  # pylint: disable=not-callable


@test_utils.run_with_cell
def silu_vmap_func(x):
    return ops.vmap(silu_forward_func, in_axes=0, out_axes=0)(x)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential', mem_peak=1024.01)
@pytest.mark.parametrize('mode', [ms.context.GRAPH_MODE, ms.context.PYNATIVE_MODE])
def test_ops_silu_forward(mode):
    """
    Feature: Pyboost function.
    Description: Test function silu forward.
    Expectation: Correct result.
    """
    ms.context.set_context(mode=mode)
    x = generate_random_input((2, 3, 4, 5), np.float32)
    output = silu_forward_func(ms.Tensor(x))
    expect = generate_expect_forward_output(x)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential', mem_peak=1024.01)
@pytest.mark.parametrize('mode', [ms.context.GRAPH_MODE, ms.context.PYNATIVE_MODE])
def test_ops_silu_backward(mode):
    """
    Feature: Pyboost function.
    Description: Test function silu backward.
    Expectation: Correct result.
    """
    ms.context.set_context(mode=mode)
    x = generate_random_input((2, 3, 4, 5), np.float32)
    output = silu_backward_func(ms.Tensor(x))
    expect = generate_expect_backward_output(x)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential', mem_peak=1024.01)
@pytest.mark.parametrize('mode', [ms.context.GRAPH_MODE, ms.context.PYNATIVE_MODE])
def test_ops_silu_vmap(mode):
    """
    Feature: Pyboost function.
    Description: Test function silu vmap feature.
    Expectation: Correct result.
    """
    ms.context.set_context(mode=mode)
    x = generate_random_input((2, 3, 4, 5), np.float32)
    output = silu_vmap_func(ms.Tensor(x))
    expect = generate_expect_forward_output(x)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential', mem_peak=0.02)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.context.PYNATIVE_MODE])
def test_ops_silu_forward_dynamic_shape(mode):
    """
    Feature: Pyboost function.
    Description: Test function silu forward with dynamic shape.
    Expectation: Correct result.
    """
    ms.context.set_context(mode=mode)
    x_dyn = ms.Tensor(shape=[None, None, None, None], dtype=ms.float32)
    test_cell = test_utils.to_cell_obj(silu_forward_func)
    test_cell.set_inputs(x_dyn)

    x1 = generate_random_input((2, 3, 4, 5), np.float32)
    output = test_cell(ms.Tensor(x1))
    expect = generate_expect_forward_output(x1)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)

    x2 = generate_random_input((3, 4, 5, 6), np.float32)
    output = test_cell(ms.Tensor(x2))
    expect = generate_expect_forward_output(x2)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential', mem_peak=2048.00)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.context.PYNATIVE_MODE])
def test_ops_silu_forward_dynamic_rank(mode):
    """
    Feature: Pyboost function.
    Description: Test function silu forward with dynamic rank.
    Expectation: Correct result.
    """
    ms.set_context(jit_level='O0')
    ms.context.set_context(mode=mode)
    x_dyn = ms.Tensor(shape=None, dtype=ms.float32)
    test_cell = test_utils.to_cell_obj(silu_forward_func)
    test_cell.set_inputs(x_dyn)

    x1 = generate_random_input((2, 3, 4, 5), np.float32)
    output = test_cell(ms.Tensor(x1))
    expect = generate_expect_forward_output(x1)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)

    x2 = generate_random_input((3, 4, 5, 6), np.float32)
    output = test_cell(ms.Tensor(x2))
    expect = generate_expect_forward_output(x2)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential', mem_peak=0.02)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.context.PYNATIVE_MODE])
def test_ops_silu_backward_dynamic_shape(mode):
    """
    Feature: Pyboost function.
    Description: Test function silu backward with dynamic shape.
    Expectation: Correct result.
    """
    ms.context.set_context(mode=mode)
    x_dyn = ms.Tensor(shape=[None, None, None, None], dtype=ms.float32)
    test_cell = test_utils.to_cell_obj(silu_backward_func)
    test_cell.set_inputs(x_dyn)

    x1 = generate_random_input((2, 3, 4, 5), np.float32)
    output = test_cell(ms.Tensor(x1))
    expect = generate_expect_backward_output(x1)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)

    x2 = generate_random_input((3, 4, 5, 6), np.float32)
    output = test_cell(ms.Tensor(x2))
    expect = generate_expect_backward_output(x2)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential', mem_peak=2048.00)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.context.PYNATIVE_MODE])
def test_ops_silu_backward_dynamic_rank(mode):
    """
    Feature: Pyboost function.
    Description: Test function silu backward with dynamic rank.
    Expectation: Correct result.
    """
    ms.set_context(jit_level='O0')
    ms.context.set_context(mode=mode)
    x_dyn = ms.Tensor(shape=None, dtype=ms.float32)
    test_cell = test_utils.to_cell_obj(silu_backward_func)
    test_cell.set_inputs(x_dyn)

    x1 = generate_random_input((2, 3, 4, 5), np.float32)
    output = test_cell(ms.Tensor(x1))
    expect = generate_expect_backward_output(x1)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)

    x2 = generate_random_input((3, 4, 5, 6), np.float32)
    output = test_cell(ms.Tensor(x2))
    expect = generate_expect_backward_output(x2)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)


def ops_silu_binary_case_compare(input_binary_data, output_binary_data):
    output = silu_forward_func(ms.Tensor(input_binary_data[0]))
    np.testing.assert_allclose(output.asnumpy(), output_binary_data[0], rtol=1e-1)
    grads = silu_backward_func(ms.Tensor(input_binary_data[0]))
    np.testing.assert_allclose(grads.asnumpy(), output_binary_data[1], rtol=1e-1)


@ops_binary_cases(OpsBinaryCase(input_info=[((64, 448, 16, 16), np.float16)],
                                output_info=[((64, 448, 16, 16), np.float16), ((64, 448, 16, 16), np.float16)],
                                extra_info='SD5B'))
def ops_silu_binary_case1(input_binary_data=None, output_binary_data=None):
    ops_silu_binary_case_compare(input_binary_data, output_binary_data)


@ops_binary_cases(OpsBinaryCase(input_info=[((64, 448, 32, 32), np.float16)],
                                output_info=[((64, 448, 32, 32), np.float16), ((64, 448, 32, 32), np.float16)],
                                extra_info='SD5B'))
def ops_silu_binary_case2(input_binary_data=None, output_binary_data=None):
    ops_silu_binary_case_compare(input_binary_data, output_binary_data)


@ops_binary_cases(OpsBinaryCase(input_info=[((64, 448, 64, 64), np.float16)],
                                output_info=[((64, 448, 64, 64), np.float16), ((64, 448, 64, 64), np.float16)],
                                extra_info='SD5B'))
def ops_silu_binary_case3(input_binary_data=None, output_binary_data=None):
    ops_silu_binary_case_compare(input_binary_data, output_binary_data)


@ops_binary_cases(OpsBinaryCase(input_info=[((64, 896, 16, 16), np.float16)],
                                output_info=[((64, 896, 16, 16), np.float16), ((64, 896, 16, 16), np.float16)],
                                extra_info='SD5B'))
def ops_silu_binary_case4(input_binary_data=None, output_binary_data=None):
    ops_silu_binary_case_compare(input_binary_data, output_binary_data)


@ops_binary_cases(OpsBinaryCase(input_info=[((64, 896, 32, 32), np.float16)],
                                output_info=[((64, 896, 32, 32), np.float16), ((64, 896, 32, 32), np.float16)],
                                extra_info='SD5B'))
def ops_silu_binary_case5(input_binary_data=None, output_binary_data=None):
    ops_silu_binary_case_compare(input_binary_data, output_binary_data)


@ops_binary_cases(OpsBinaryCase(input_info=[((64, 896, 64, 64), np.float16)],
                                output_info=[((64, 896, 64, 64), np.float16), ((64, 896, 64, 64), np.float16)],
                                extra_info='SD5B'))
def ops_silu_binary_case6(input_binary_data=None, output_binary_data=None):
    ops_silu_binary_case_compare(input_binary_data, output_binary_data)


@ops_binary_cases(OpsBinaryCase(input_info=[((64, 1792), np.float16)],
                                output_info=[((64, 1792), np.float16), ((64, 1792), np.float16)],
                                extra_info='SD5B'))
def ops_silu_binary_case7(input_binary_data=None, output_binary_data=None):
    ops_silu_binary_case_compare(input_binary_data, output_binary_data)


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b'], level_mark='level1', card_mark='onecard',
          essential_mark='essential', mem_peak=6640.25)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.context.PYNATIVE_MODE])
def test_ops_silu_binary_cases(mode):
    """
    Feature: Pyboost function.
    Description: Test function silu binary cases.
    Expectation: Correct result.
    """
    ms.context.set_context(mode=mode)

    ops_silu_binary_case1()
    ops_silu_binary_case2()
    ops_silu_binary_case3()
    ops_silu_binary_case4()
    ops_silu_binary_case5()
    ops_silu_binary_case6()
    ops_silu_binary_case7()
