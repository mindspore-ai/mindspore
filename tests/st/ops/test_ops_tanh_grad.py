# Copyright 2024 Huawei Technotanhies Co., Ltd
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
from mindspore import context, Tensor, jit, JitConfig
from tests.st.utils import test_utils
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP
from tests.mark_utils import arg_mark

tanh_grad = ms.ops.auto_generate.TanhGrad()


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)

def generate_expect_forward_output(y, dy, dtype):
    return dy*(1 - np.power(y, 2)).astype(dtype)

def generate_expect_backward_output(y, dy, dtype):
    ydy = (-2*y*dy).astype(dtype)
    dydy = (1 - y*y).astype(dtype)
    return  ydy, dydy

@test_utils.run_with_cell
def tanh_grad_forward_func(y, dy):
    return tanh_grad(y, dy)

@test_utils.run_with_cell
def tanh_grad_backward_func(y, dy):
    return ms.ops.grad(tanh_grad_forward_func, (0, 1))(y, dy)

@test_utils.run_with_cell
def tanh_grad_vamp_func(y, dy):
    return ms.ops.vmap(tanh_grad_forward_func, in_axes=0, out_axes=0)(y, dy)




@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0',
          card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tanh_grad_forward(mode):
    """
    Feature: test tanh operator
    Description: test tanh run by pyboost
    Expectation: success
    """
    context.set_context(mode=mode)
    y_np = generate_random_input((2, 3, 4), np.float32)
    y_tensor = Tensor(y_np, ms.float32)
    dy_np = generate_random_input((2, 3, 4), np.float32)
    dy_tensor = Tensor(dy_np, ms.float32)
    output = tanh_grad_forward_func(y_tensor, dy_tensor)
    expect = generate_expect_forward_output(y_np, dy_np, np.float32)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-4, atol=1e-4)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0',
          card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tanh_grad_backward(mode):
    """
    Feature: test tanh operator
    Description: test tanh run by pyboost
    Expectation: success
    """
    context.set_context(mode=mode)
    y_np = generate_random_input((2, 3, 4), np.float32)
    y_tensor = Tensor(y_np, ms.float32)
    dy_np = generate_random_input((2, 3, 4), np.float32)
    dy_tensor = Tensor(dy_np, ms.float32)
    output = tanh_grad_backward_func(y_tensor, dy_tensor)
    expect = generate_expect_backward_output(y_np, dy_np, np.float32)
    np.testing.assert_allclose(output[0].asnumpy(), expect[0], rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(output[1].asnumpy(), expect[1], rtol=1e-4, atol=1e-4)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize("mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tanh_grad_vmap(mode):
    """
    Feature: pyboost function.
    Description: test function tanh vmap feature.
    Expectation: expect correct result.
    """
    context.set_context(mode=mode)
    y_np = generate_random_input((2, 3, 4, 5), np.float32)
    y_tensor = Tensor(y_np, ms.float32)
    dy_np = generate_random_input((2, 3, 4, 5), np.float32)
    dy_tensor = Tensor(dy_np, ms.float32)
    output = tanh_grad_vamp_func(y_tensor, dy_tensor)
    expect = generate_expect_forward_output(y_np, dy_np, np.float32)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-4, atol=1e-4)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize("mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tanh_grad_forward_dynamic_shape(mode):
    """
    Feature: pyboost function.
    Description: test function tanh forward with dynamic shape.
    Expectation: expect correct result.
    """
    context.set_context(mode=mode)

    y_dyn = Tensor(shape=[None, None, None, None], dtype=ms.float32)
    dy_dyn = Tensor(shape=[None, None, None, None], dtype=ms.float32)
    test_cell = test_utils.to_cell_obj(tanh_grad_forward_func)
    test_cell.set_inputs(y_dyn, dy_dyn)

    y1_np = generate_random_input((2, 3, 4, 5), np.float32)
    y1_tensor = Tensor(y1_np, ms.float32)
    dy1_np = generate_random_input((2, 3, 4, 5), np.float32)
    dy1_tensor = Tensor(dy1_np, ms.float32)

    output = test_cell(y1_tensor, dy1_tensor)
    expect = generate_expect_forward_output(y1_np, dy1_np, np.float32)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-4, atol=1e-4)

    y2_np = generate_random_input((3, 4, 5, 6), np.float32)
    y2_tensor = Tensor(y2_np, ms.float32)
    dy2_np = generate_random_input((3, 4, 5, 6), np.float32)
    dy2_tensor = Tensor(dy2_np, ms.float32)
    output = test_cell(y2_tensor, dy2_tensor)
    expect = generate_expect_forward_output(y2_np, dy2_np, np.float32)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-4, atol=1e-4)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize("mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tanh_grad_forward_dynamic_rank(mode):
    """
    Feature: pyboost function.
    Description: test function tanh forward with dynamic rank.
    Expectation: expect correct result.
    """
    context.set_context(mode=mode)

    y_dyn = Tensor(shape=None, dtype=ms.float32)
    dy_dyn = Tensor(shape=None, dtype=ms.float32)
    test_cell = test_utils.to_cell_obj(tanh_grad_forward_func)
    test_cell.set_inputs(y_dyn, dy_dyn)

    y1_np = generate_random_input((2, 3, 4, 5), np.float32)
    y1_tensor = Tensor(y1_np, ms.float32)
    dy1_np = generate_random_input((2, 3, 4, 5), np.float32)
    dy1_tensor = Tensor(dy1_np, ms.float32)
    output = test_cell(y1_tensor, dy1_tensor)
    expect = generate_expect_forward_output(y1_np, dy1_np, np.float32)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-4, atol=1e-4)

    y2_np = generate_random_input((3, 4, 5, 6), np.float32)
    y2_tensor = Tensor(y2_np, ms.float32)
    dy2_np = generate_random_input((3, 4, 5, 6), np.float32)
    dy2_tensor = Tensor(dy2_np, ms.float32)
    output = test_cell(y2_tensor, dy2_tensor)
    expect = generate_expect_forward_output(y2_np, dy2_np, np.float32)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-4, atol=1e-4)



@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize("mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tanh_grad_backward_dynamic_shape(mode):
    """
    Feature: pyboost function.
    Description: test function tanh backward with dynamic shape.
    Expectation: expect correct result.
    """
    context.set_context(mode=mode)

    y_dyn = Tensor(shape=[None, None, None, None], dtype=ms.float32)
    dy_dyn = Tensor(shape=[None, None, None, None], dtype=ms.float32)
    test_cell = test_utils.to_cell_obj(tanh_grad_backward_func)
    test_cell.set_inputs(y_dyn, dy_dyn)

    y1_np = generate_random_input((2, 3, 4, 5), np.float32)
    y1_tensor = Tensor(y1_np, ms.float32)
    dy1_np = generate_random_input((2, 3, 4, 5), np.float32)
    dy1_tensor = Tensor(dy1_np, ms.float32)
    output = test_cell(y1_tensor, dy1_tensor)
    expect = generate_expect_backward_output(y1_np, dy1_np, np.float32)
    np.testing.assert_allclose(output[0].asnumpy(), expect[0], rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(output[1].asnumpy(), expect[1], rtol=1e-4, atol=1e-4)

    y2_np = generate_random_input((3, 4, 5, 6), np.float32)
    y2_tensor = Tensor(y2_np, ms.float32)
    dy2_np = generate_random_input((3, 4, 5, 6), np.float32)
    dy2_tensor = Tensor(dy2_np, ms.float32)
    output = test_cell(y2_tensor, dy2_tensor)
    expect = generate_expect_backward_output(y2_np, dy2_np, np.float32)
    np.testing.assert_allclose(output[0].asnumpy(), expect[0], rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(output[1].asnumpy(), expect[1], rtol=1e-4, atol=1e-4)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize("mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tanh_grad_backward_dynamic_rank(mode):
    """
    Feature: pyboost function.
    Description: test function tanh backward with dynamic rank.
    Expectation: expect correct result.
    """
    context.set_context(mode=mode)

    y_dyn = Tensor(shape=None, dtype=ms.float32)
    dy_dyn = Tensor(shape=None, dtype=ms.float32)
    test_cell = test_utils.to_cell_obj(tanh_grad_backward_func)
    test_cell.set_inputs(y_dyn, dy_dyn)

    y1_np = generate_random_input((2, 3, 4, 5), np.float32)
    y1_tensor = Tensor(y1_np, ms.float32)
    dy1_np = generate_random_input((2, 3, 4, 5), np.float32)
    dy1_tensor = Tensor(dy1_np, ms.float32)
    output = test_cell(y1_tensor, dy1_tensor)
    expect = generate_expect_backward_output(y1_np, dy1_np, np.float32)
    np.testing.assert_allclose(output[0].asnumpy(), expect[0], rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(output[1].asnumpy(), expect[1], rtol=1e-4, atol=1e-4)

    y2_np = generate_random_input((3, 4, 5, 6), np.float32)
    y2_tensor = Tensor(y2_np, ms.float32)
    dy2_np = generate_random_input((3, 4, 5, 6), np.float32)
    dy2_tensor = Tensor(dy2_np, ms.float32)
    output = test_cell(y2_tensor, dy2_tensor)
    expect = generate_expect_backward_output(y2_np, dy2_np, np.float32)
    np.testing.assert_allclose(output[0].asnumpy(), expect[0], rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(output[1].asnumpy(), expect[1], rtol=1e-4, atol=1e-4)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_tanh_grad_dynamic_shape_testop():
    """
    Feature: Test tanh with dynamic shape in graph mode using TEST_OP.
    Description: call ops.tanh with valid input and index.
    Expectation: return the correct value.
    """
    y1 = generate_random_input((3, 4, 5), np.float32)
    dy1 = generate_random_input((3, 4, 5), np.float32)
    y2 = generate_random_input((3, 7, 8, 3), np.float32)
    dy2 = generate_random_input((3, 7, 8, 3), np.float32)

    TEST_OP(tanh_grad_forward_func, [[ms.Tensor(y1), ms.Tensor(dy1)], [ms.Tensor(y2), ms.Tensor(dy2)]], 'tanh_grad')


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK', 'GE'])
def test_tanh_grad_forward_static_shape(mode):
    """
    Feature: Test tanh with static shape in graph and pynative mode.
    Description: call ops.tanh with valid input and index.
    Expectation: return the correct value.
    """
    y_np = generate_random_input((3, 4, 5), np.float32)
    y_tensor = Tensor(y_np, ms.float32)
    dy_np = generate_random_input((3, 4, 5), np.float32)
    dy_tensor = Tensor(dy_np, ms.float32)

    if mode == 'pynative':
        context.set_context(mode=ms.PYNATIVE_MODE)
        output = tanh_grad_forward_func(y_tensor, dy_tensor)
    elif mode == 'KBK':
        context.set_context(mode=ms.GRAPH_MODE)
        output = (jit(tanh_grad_forward_func, jit_config=JitConfig(jit_level="O0")))(y_tensor, dy_tensor)
    else:
        context.set_context(mode=ms.GRAPH_MODE)
        output = (jit(tanh_grad_forward_func, jit_config=JitConfig(jit_level="O2")))(y_tensor, dy_tensor)

    expect = generate_expect_forward_output(y_np, dy_np, np.float32)
    assert np.allclose(output.asnumpy(), expect, rtol=1e-4, atol=1e-4)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK', 'GE'])
def test_tanh_grad_backward_static_shape(mode):
    """
    Feature: Test tanh with static shape in graph and pynative mode.
    Description: call ops.tanh with valid input and index.
    Expectation: return the correct value.
    """
    y_np = generate_random_input((3, 4, 5), np.float32)
    y_tensor = Tensor(y_np, ms.float32)
    dy_np = generate_random_input((3, 4, 5), np.float32)
    dy_tensor = Tensor(dy_np, ms.float32)

    if mode == 'pynative':
        context.set_context(mode=ms.PYNATIVE_MODE)
        output = tanh_grad_backward_func(y_tensor, dy_tensor)
    elif mode == 'KBK':
        context.set_context(mode=ms.GRAPH_MODE)
        output = (jit(tanh_grad_backward_func, jit_config=JitConfig(jit_level="O0")))(y_tensor, dy_tensor)
    else:
        context.set_context(mode=ms.GRAPH_MODE)
        output = (jit(tanh_grad_backward_func, jit_config=JitConfig(jit_level="O2")))(y_tensor, dy_tensor)

    expect = generate_expect_backward_output(y_np, dy_np, np.float32)
    assert np.allclose(output[0].asnumpy(), expect[0], rtol=1e-4, atol=1e-4)
    assert np.allclose(output[1].asnumpy(), expect[1], rtol=1e-4, atol=1e-4)
