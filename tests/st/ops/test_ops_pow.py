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
"""
Tests for ops.pow operation.
"""
import numpy as np
import pytest
from tests.st.utils import test_utils
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.mark_utils import arg_mark
from mindspore import ops
import mindspore as ms


def generate_random_input(shape, dtype):
    return np.random.uniform(0.9, 1.0, size=shape).astype(dtype), np.random.uniform(0.9, 1.0, size=shape).astype(dtype)


@test_utils.run_with_cell
def pow_forward_func(x, y):
    return ops.pow(x, y)


@test_utils.run_with_cell
def pow_backward_func(x, y):
    return ops.grad(pow_forward_func, (0, 1))(x, y)


@test_utils.run_with_cell
def pow_vmap_func(x, y):
    return ops.vmap(pow_forward_func, in_axes=0, out_axes=0)(x, y)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential', mem_peak=2048.10)
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize("data_type", [np.float32])
@test_utils.run_test_with_On
def test_pow_op_forward(context_mode, data_type):
    """
    Feature: Ops.
    Description: test op pow forward.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x_np, y_np = generate_random_input((2, 3, 4, 5), dtype=data_type)

    x = ms.Tensor(x_np)
    y = ms.Tensor(y_np)
    out = pow_forward_func(x, y)
    expect_out = np.power(x_np, y_np)
    np.testing.assert_allclose(out.asnumpy(), expect_out, rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential', mem_peak=1024.01)
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize("data_type", [np.float32])
@test_utils.run_test_with_On
def test_pow_op_backward(context_mode, data_type):
    """
    Feature: Auto grad.
    Description: test auto grad of op pow.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x = ms.Tensor(np.array([1, 2, 4]).astype(data_type))
    y = ms.Tensor(np.array([2, 4, 3]).astype(data_type))
    grads = pow_backward_func(x, y)
    expect_out = np.array([[2., 32., 48.], [0., 11.0903549, 88.7228394]]).astype(np.float32)
    np.testing.assert_allclose(grads[0].asnumpy(), expect_out[0], rtol=1e-3)
    np.testing.assert_allclose(grads[1].asnumpy(), expect_out[1], rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b',
                      'platform_gpu',
                      'cpu_linux', 'cpu_windows', 'cpu_macos'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='unessential', mem_peak=4.0)
@pytest.mark.parametrize("mode", ['pynative', 'kbk', 'ge'])
@pytest.mark.parametrize("dtype", [np.float32])
@test_utils.run_test_with_On
def test_pow_op_backward_zero_guard(mode, dtype):
    """
    Feature: Auto grad.
    Description: test auto grad of op pow with zero input/exponent to cover bprop guard branches.
    Expectation: expect correct result and no NaN/Inf.
    """
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    elif mode == 'kbk':
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level='O0')
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level='O2')

    x = ms.Tensor(np.array([0., 2., 3.]).astype(dtype))
    y = ms.Tensor(np.array([2., 0., 2.]).astype(dtype))

    grads = pow_backward_func(x, y)
    dx = grads[0].asnumpy()
    dy = grads[1].asnumpy()

    # dx = dout * (y * x^(y-1)); when y == 0, dx should be 0.
    expect_dx = np.array([0., 0., 6.]).astype(np.float32)
    # dy = dout * (out * log(x)); when x == 0 and y >= 0, dy should be 0 to avoid log(0).
    expect_dy = np.array([0., np.log(2.), 9. * np.log(3.)]).astype(np.float32)

    np.testing.assert_allclose(dx, expect_dx, rtol=1e-3)
    np.testing.assert_allclose(dy, expect_dy, rtol=1e-3)
    assert np.all(np.isfinite(dx))
    assert np.all(np.isfinite(dy))


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential', mem_peak=1024.01)
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize("data_type", [np.float32])
@test_utils.run_test_with_On
def test_pow_op_vmap(context_mode, data_type):
    """
    Feature: test vmap function.
    Description: test pow op vmap.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x_np, y_np = generate_random_input((2, 3, 4, 5), dtype=data_type)

    x = ms.Tensor(x_np)
    y = ms.Tensor(y_np)
    out = pow_vmap_func(x, y)
    expect_out = np.power(x_np, y_np)
    np.testing.assert_allclose(out.asnumpy(), expect_out, rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential', mem_peak=3072.00)
def test_pow_dynamic_shape():
    """
    Feature: Test pow with dynamic shape in graph mode.
    Description: call ops.pow with valid input and index.
    Expectation: return the correct value.
    """
    x1, other1 = generate_random_input((2, 3, 4), np.float32)
    x2, other2 = generate_random_input((3, 4, 5, 6), np.float32)

    TEST_OP(pow_forward_func,
            [[ms.Tensor(x1), ms.Tensor(other1)], [ms.Tensor(x2), ms.Tensor(other2)]],
            disable_case=['Deterministic'],
            case_config={'all_dim_zero': True})
