# Copyright 2023 Huawei Technotanhies Co., Ltd
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


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)

def generate_expect_forward_output(x, dtype):
    return np.tanh(x).astype(dtype)

def generate_expect_backward_output(x, dtype):
    output = 1 - np.power(np.tanh(x), 2)
    return  output.astype(dtype)

@test_utils.run_with_cell
def tanh_forward_func(x):
    return ms.ops.tanh(x)

@test_utils.run_with_cell
def tanh_backward_func(x):
    return ms.ops.grad(tanh_forward_func, (0))(x)

@test_utils.run_with_cell
def tanh_vmap_func(x):
    return ms.ops.vmap(tanh_forward_func, in_axes=0, out_axes=0)(x)




@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize("mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tanh_forward(mode):
    """
    Feature: test tanh operator
    Description: test tanh run by pyboost
    Expectation: success
    """
    context.set_context(mode=mode)
    x_np = generate_random_input((2, 3, 4), np.float32)
    x_tensor = Tensor(x_np, ms.float32)
    output = tanh_forward_func(x_tensor)
    expect = generate_expect_forward_output(x_np, np.float32)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-4, atol=1e-4)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize("mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tanh_backward(mode):
    """
    Feature: test tanh operator
    Description: test tanh run by pyboost
    Expectation: success
    """
    context.set_context(mode=mode)
    x_np = generate_random_input((2, 3, 4), np.float32)
    x_tensor = Tensor(x_np, ms.float32)
    output = tanh_backward_func(x_tensor)
    expect = generate_expect_backward_output(x_np, np.float32)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-4, atol=1e-4)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize("mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tanh_vmap(mode):
    """
    Feature: pyboost function.
    Description: test function tanh vmap feature.
    Expectation: expect correct result.
    """
    context.set_context(mode=mode)
    x_np = generate_random_input((2, 3, 4, 5), np.float32)
    x_tensor = Tensor(x_np, ms.float32)
    output = tanh_vmap_func(x_tensor)
    expect = generate_expect_forward_output(x_np, np.float32)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-4, atol=1e-4)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize("mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tanh_forward_dynamic_shape(mode):
    """
    Feature: pyboost function.
    Description: test function tanh forward with dynamic shape.
    Expectation: expect correct result.
    """
    context.set_context(mode=mode)

    x_dyn = Tensor(shape=[None, None, None, None], dtype=ms.float32)
    test_cell = test_utils.to_cell_obj(tanh_forward_func)
    test_cell.set_inputs(x_dyn)

    x1_np = generate_random_input((2, 3, 4, 5), np.float32)
    x1_tensor = Tensor(x1_np, ms.float32)
    output = test_cell(x1_tensor)
    expect = generate_expect_forward_output(x1_np, np.float32)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-4, atol=1e-4)

    x2_np = generate_random_input((3, 4, 5, 6), np.float32)
    x2_tensor = Tensor(x2_np, ms.float32)
    output = test_cell(x2_tensor)
    expect = generate_expect_forward_output(x2_np, np.float32)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-4, atol=1e-4)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize("mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tanh_forward_dynamic_rank(mode):
    """
    Feature: pyboost function.
    Description: test function tanh forward with dynamic rank.
    Expectation: expect correct result.
    """
    context.set_context(mode=mode)

    x_dyn = Tensor(shape=None, dtype=ms.float32)
    test_cell = test_utils.to_cell_obj(tanh_forward_func)
    test_cell.set_inputs(x_dyn)

    x1_np = generate_random_input((2, 3, 4, 5), np.float32)
    x1_tensor = Tensor(x1_np, ms.float32)
    output = test_cell(x1_tensor)
    expect = generate_expect_forward_output(x1_np, np.float32)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-4, atol=1e-4)

    x2_np = generate_random_input((3, 4, 5, 6), np.float32)
    x2_tensor = Tensor(x2_np, ms.float32)
    output = test_cell(x2_tensor)
    expect = generate_expect_forward_output(x2_np, np.float32)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-4, atol=1e-4)



@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize("mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tanh_backward_dynamic_shape(mode):
    """
    Feature: pyboost function.
    Description: test function tanh backward with dynamic shape.
    Expectation: expect correct result.
    """
    context.set_context(mode=mode)

    x_dyn = Tensor(shape=[None, None, None, None], dtype=ms.float32)
    test_cell = test_utils.to_cell_obj(tanh_backward_func)
    test_cell.set_inputs(x_dyn)

    x1_np = generate_random_input((2, 3, 4, 5), np.float32)
    x1_tensor = Tensor(x1_np, ms.float32)
    output = test_cell(x1_tensor)
    expect = generate_expect_backward_output(x1_np, np.float32)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-4, atol=1e-4)

    x2_np = generate_random_input((3, 4, 5, 6), np.float32)
    x2_tensor = Tensor(x2_np, ms.float32)
    output = test_cell(x2_tensor)
    expect = generate_expect_backward_output(x2_np, np.float32)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-4, atol=1e-4)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize("mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tanh_backward_dynamic_rank(mode):
    """
    Feature: pyboost function.
    Description: test function tanh backward with dynamic rank.
    Expectation: expect correct result.
    """
    context.set_context(mode=mode)

    x_dyn = Tensor(shape=None, dtype=ms.float32)
    test_cell = test_utils.to_cell_obj(tanh_backward_func)
    test_cell.set_inputs(x_dyn)

    x1_np = generate_random_input((2, 3, 4, 5), np.float32)
    x1_tensor = Tensor(x1_np, ms.float32)
    output = test_cell(x1_tensor)
    expect = generate_expect_backward_output(x1_np, np.float32)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-4, atol=1e-4)

    x2_np = generate_random_input((3, 4, 5, 6), np.float32)
    x2_tensor = Tensor(x2_np, ms.float32)
    output = test_cell(x2_tensor)
    expect = generate_expect_backward_output(x2_np, np.float32)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-4, atol=1e-4)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.parametrize('jit_level', ["O0", "O2"])
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
def test_tanh_dynamic_shape_testop(jit_level):
    """
    Feature: Test tanh with dynamic shape in graph mode using TEST_OP.
    Description: call ops.tanh with valid input and index.
    Expectation: return the correct value.
    """
    x1 = generate_random_input((3, 4, 5), np.float32)
    x2 = generate_random_input((3, 7, 8, 3), np.float32)

    TEST_OP(tanh_forward_func, [[ms.Tensor(x1)], [ms.Tensor(x2)]], grad=True, jit_level=jit_level)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.parametrize('mode', ['pynative', 'KBK', 'GE'])
def test_tanh_forward_static_shape(mode):
    """
    Feature: Test tanh with static shape in graph and pynative mode.
    Description: call ops.tanh with valid input and index.
    Expectation: return the correct value.
    """
    x_np = generate_random_input((3, 4, 5), np.float32)
    x_tensor = Tensor(x_np, ms.float32)

    if mode == 'pynative':
        context.set_context(mode=ms.PYNATIVE_MODE)
        output = tanh_forward_func(x_tensor)
    elif mode == 'KBK':
        context.set_context(mode=ms.GRAPH_MODE)
        output = (jit(tanh_forward_func, jit_config=JitConfig(jit_level="O0")))(x_tensor)
    else:
        context.set_context(mode=ms.GRAPH_MODE)
        output = (jit(tanh_forward_func, jit_config=JitConfig(jit_level="O2")))(x_tensor)

    expect = generate_expect_forward_output(x_np, np.float32)
    assert np.allclose(output.asnumpy(), expect, rtol=1e-4, atol=1e-4)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.parametrize('mode', ['pynative', 'KBK', 'GE'])
def test_tanh_backward_static_shape(mode):
    """
    Feature: Test tanh with static shape in graph and pynative mode.
    Description: call ops.tanh with valid input and index.
    Expectation: return the correct value.
    """
    x_np = generate_random_input((3, 4, 5), np.float32)
    x_tensor = Tensor(x_np, ms.float32)

    if mode == 'pynative':
        context.set_context(mode=ms.PYNATIVE_MODE)
        output = tanh_backward_func(x_tensor)
    elif mode == 'KBK':
        context.set_context(mode=ms.GRAPH_MODE)
        output = (jit(tanh_backward_func, jit_config=JitConfig(jit_level="O0")))(x_tensor)
    else:
        context.set_context(mode=ms.GRAPH_MODE)
        output = (jit(tanh_backward_func, jit_config=JitConfig(jit_level="O2")))(x_tensor)

    expect = generate_expect_backward_output(x_np, np.float32)
    assert np.allclose(output.asnumpy(), expect, rtol=1e-4, atol=1e-4)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.parametrize("mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tanh_forward_bfloat16(mode):
    """
    Feature: test tanh operator
    Description: test tanh run by pyboost
    Expectation: success
    """
    context.set_context(mode=mode)
    x_tensor = Tensor(generate_random_input((2, 3, 4), np.float32), ms.bfloat16)
    x_np = x_tensor.float().asnumpy()
    output = tanh_forward_func(x_tensor)
    expect = generate_expect_forward_output(x_np, np.float32)
    np.testing.assert_allclose(output.float().asnumpy(), expect, rtol=4e-3, atol=4e-3)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.parametrize("mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tanh_backward_bfloat16(mode):
    """
    Feature: test tanh operator
    Description: test tanh run by pyboost
    Expectation: success
    """
    context.set_context(mode=mode)
    x_tensor = Tensor(generate_random_input((2, 3, 4), np.float32), ms.bfloat16)
    x_np = x_tensor.float().asnumpy()
    output = tanh_backward_func(x_tensor)
    expect = generate_expect_backward_output(x_np, np.float32)
    np.testing.assert_allclose(output.float().asnumpy(), expect, rtol=4e-3, atol=4e-3)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize("mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tanh_forward_float16(mode):
    """
    Feature: test tanh operator
    Description: test tanh run by pyboost
    Expectation: success
    """
    context.set_context(mode=mode)
    x_np = generate_random_input((2, 3, 4), np.float16)
    x_tensor = Tensor(x_np, ms.float16)
    output = tanh_forward_func(x_tensor)
    expect = generate_expect_forward_output(x_np, np.float16)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3, atol=1e-3)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize("mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tanh_backward_float16(mode):
    """
    Feature: test tanh operator
    Description: test tanh run by pyboost
    Expectation: success
    """
    context.set_context(mode=mode)
    x_np = generate_random_input((2, 3, 4), np.float16)
    x_tensor = Tensor(x_np, ms.float16)
    output = tanh_backward_func(x_tensor)
    expect = generate_expect_backward_output(x_np, np.float16)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3, atol=1e-3)
