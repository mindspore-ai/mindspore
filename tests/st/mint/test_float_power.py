# Copyright 2024 Huawei Technomse_lossies Co., Ltd
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
from mindspore import mint
from tests.st.utils import test_utils
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.mark_utils import arg_mark


def generate_random_input(shape, dtype):
    return np.random.uniform(0.9, 1.0, size=shape).astype(dtype)


@test_utils.run_with_cell
def float_power_forward_func(x, exp):
    return mint.float_power(x, exp)


@test_utils.run_with_cell
def float_power_backward_func(x, exp):
    return ms.ops.grad(float_power_forward_func, (0, 1))(x, exp)


# Testcases for `mint.float_power(tensor, tensor)`
@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=2048.0)
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize("data_type", [np.float32, np.float64])
def test_float_power_tensor_tensor_forward(context_mode, data_type):
    """
    Feature: Ops.
    Description: test op float_power forward.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    if context_mode == ms.GRAPH_MODE:
        ms.set_context(jit_config={'jit_level': 'O0'})
    x_np = generate_random_input((2, 3, 4, 5), dtype=data_type)
    y_np = generate_random_input((2, 3, 4, 5), dtype=data_type)
    x = ms.Tensor(x_np)
    y = ms.Tensor(y_np)
    out = float_power_forward_func(x, y)
    expect_out = np.power(x_np, y_np).astype(np.float64)
    np.testing.assert_allclose(out.asnumpy(), expect_out, rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
        card_mark='onecard', essential_mark='essential', mem_peak=2048.01)
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize("data_type", [np.float32])
def test_float_power_tensor_tensor_backward(context_mode, data_type):
    """
    Feature: Auto grad.
    Description: test auto grad of op float_power.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    if context_mode == ms.GRAPH_MODE:
        ms.set_context(jit_config={'jit_level': 'O0'})
    x_np = generate_random_input((2, 3, 4, 5), dtype=data_type)
    y_np = generate_random_input((2, 3, 4, 5), dtype=data_type)

    x = ms.Tensor(x_np)
    y = ms.Tensor(y_np)
    x_grad, y_grad = float_power_backward_func(x, y)
    x_expect_out = np.power(x_np, y_np - 1) * y_np
    y_expect_out = np.power(x_np, y_np) * np.log(x_np)
    np.testing.assert_allclose(x_grad.asnumpy(), x_expect_out, rtol=1e-3)
    np.testing.assert_allclose(y_grad.asnumpy(), y_expect_out, rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential', mem_peak=4.0)
def test_float_power_tensor_tensor_dynamic_shape():
    """
    Feature: Test float_power with dynamic shape in graph mode.
    Description: call mint.float_power with valid input and index.
    Expectation: return the correct value.
    """
    x1 = generate_random_input((2, 3, 4), np.float32)
    x2 = generate_random_input((3, 4, 5, 6), np.float32)
    exp1 = generate_random_input((2, 3, 4), np.float32)
    exp2 = generate_random_input((3, 4, 5, 6), np.float32)

    TEST_OP(float_power_forward_func,
            [[ms.Tensor(x1), ms.Tensor(exp1)], [ms.Tensor(x2), ms.Tensor(exp2)]],
            disable_mode=['GRAPH_MODE_GE'],
            case_config={'disable_input_check': True,
                         'all_dim_zero': True})


# Testcases for `mint.float_power(tensor, scalar)`
@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=2048.0)
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize("data_type", [np.float32, np.float64])
def test_float_power_tensor_scalar_forward(context_mode, data_type):
    """
    Feature: Ops.
    Description: test op float_power forward.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    if context_mode == ms.GRAPH_MODE:
        ms.set_context(jit_config={'jit_level': 'O0'})
    x_np = generate_random_input((3, 4, 5), dtype=data_type)
    y_np = 2.3

    x = ms.Tensor(x_np)
    y = y_np
    out = float_power_forward_func(x, y)
    expect_out = np.power(x_np, y_np).astype(np.float64)
    np.testing.assert_allclose(out.asnumpy(), expect_out, rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=2048.0)
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize("data_type", [np.float32])
def test_float_power_tensor_scalar_backward(context_mode, data_type):
    """
    Feature: Auto grad.
    Description: test auto grad of op float_power.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    if context_mode == ms.GRAPH_MODE:
        ms.set_context(jit_config={'jit_level': 'O0'})
    x_np = generate_random_input((2, 3, 4, 5), dtype=data_type)
    y_np = 2.3

    x = ms.Tensor(x_np)
    y = y_np
    grads = float_power_backward_func(x, y)
    expect_out = np.power(x_np, y_np - 1) * y_np
    np.testing.assert_allclose(grads.asnumpy(), expect_out, rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential', mem_peak=4.0)
def test_float_power_tensor_scalar_dynamic_shape():
    """
    Feature: Test float_power with dynamic shape in graph mode.
    Description: call mint.float_power with valid input and index.
    Expectation: return the correct value.
    """
    x1 = generate_random_input((2, 3, 4), np.float32)
    x2 = generate_random_input((3, 4, 5, 6), np.float32)

    TEST_OP(float_power_forward_func,
            [[ms.Tensor(x1), 2.3], [ms.Tensor(x2), 0]],
            disable_mode=['GRAPH_MODE_GE'],
            case_config={'disable_input_check': True,
                         'disable_resize': True,
                         'all_dim_zero': True})


# Testcases for `mint.float_power(scalar, tensor)`
@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
        card_mark='onecard', essential_mark='essential', mem_peak=2048.00)
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize("data_type", [np.float32, np.float64])
def test_float_power_scalar_tensor_forward(context_mode, data_type):
    """
    Feature: Ops.
    Description: test op float_power forward.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    if context_mode == ms.GRAPH_MODE:
        ms.set_context(jit_config={'jit_level': 'O0'})
    x_np = 2.3
    y_np = generate_random_input((3, 4, 5), dtype=data_type)

    x = x_np
    y = ms.Tensor(y_np)
    out = float_power_forward_func(x, y)
    expect_out = np.power(x_np, y_np).astype(np.float64)
    np.testing.assert_allclose(out.asnumpy(), expect_out, rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=2048.01)
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize("data_type", [np.float32])
def test_float_power_scalar_tensor_backward(context_mode, data_type):
    """
    Feature: Auto grad.
    Description: test auto grad of op float_power.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    if context_mode == ms.GRAPH_MODE:
        ms.set_context(jit_config={'jit_level': 'O0'})
    y_np = generate_random_input((3, 4, 5), dtype=data_type)
    x = x_np = 2
    y = ms.Tensor(y_np)
    grads = float_power_backward_func(x, y)
    expect_out = np.power(x_np, y_np) * np.log(x_np)
    np.testing.assert_allclose(grads.asnumpy(), expect_out, rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential', mem_peak=4.0)
def test_float_power_scalar_tensor_dynamic_shape():
    """
    Feature: Test float_power with dynamic shape in graph mode.
    Description: call mint.float_power with valid input and index.
    Expectation: return the correct value.
    """
    exp1 = generate_random_input((2, 3, 4), np.float32)
    exp2 = generate_random_input((3, 4, 5, 6), np.float32)

    TEST_OP(float_power_forward_func, [[2.3, ms.Tensor(exp1)], [0, ms.Tensor(exp2)]],
            disable_mode=['GRAPH_MODE_GE'],
            case_config={'disable_input_check': True,
                         'disable_resize': True,
                         'all_dim_zero': True})
