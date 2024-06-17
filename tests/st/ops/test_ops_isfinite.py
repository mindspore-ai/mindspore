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
from mindspore import ops, jit, JitConfig
from mindspore.ops import isfinite
from tests.st.utils import test_utils
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)


def generate_expect_forward_output(x):
    return np.isfinite(x)


def generate_expect_backward_output(x):
    return 0


@test_utils.run_with_cell
def isfinite_forward_func(x):
    return isfinite(x)


@test_utils.run_with_cell
def isfinite_backward_func(x):
    return ops.grad(isfinite_forward_func, (0))(x)


@test_utils.run_with_cell
def isfinite_vmap_func(x, in_axes=0):
    return ops.vmap(isfinite_forward_func, in_axes, out_axes=0)(x)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_isfinite_forward(context_mode):
    """
    Feature: pyboost function.
    Description: test function isfinite forward.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x = generate_random_input((2, 3, 4, 5), np.float32)
    output = isfinite_forward_func(ms.Tensor(x))
    expect = generate_expect_forward_output(x)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_isfinite_backward(context_mode):
    """
    Feature: pyboost function.
    Description: test function isfinite backward.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x = generate_random_input((2, 3, 4, 5), np.float32)
    output = isfinite_backward_func(ms.Tensor(x))
    expect = generate_expect_backward_output(x)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_isfinite_vmap(context_mode):
    """
    Feature: pyboost function.
    Description: test function isfinite vmap feature.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x = generate_random_input((2, 3, 4, 5), np.float32)
    output = isfinite_vmap_func(ms.Tensor(x), 0)
    expect = generate_expect_forward_output(x)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_isfinite_forward_dynamic_shape(context_mode):
    """
    Feature: pyboost function.
    Description: test function isfinite forward with dynamic shape.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)

    x_dyn = ms.Tensor(shape=[None, None, None, None], dtype=ms.float32)
    test_cell = test_utils.to_cell_obj(isfinite_forward_func)
    test_cell.set_inputs(x_dyn)

    x1 = generate_random_input((2, 3, 4, 5), np.float32)
    output = test_cell(ms.Tensor(x1))
    expect = generate_expect_forward_output(x1)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)

    x2 = generate_random_input((3, 4, 5, 6), np.float32)
    output = test_cell(ms.Tensor(x2))
    expect = generate_expect_forward_output(x2)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_isfinite_forward_dynamic_rank(context_mode):
    """
    Feature: pyboost function.
    Description: test function isfinite forward with dynamic rank.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)

    x_dyn = ms.Tensor(shape=None, dtype=ms.float32)
    test_cell = test_utils.to_cell_obj(isfinite_forward_func)
    test_cell.set_inputs(x_dyn)

    x1 = generate_random_input((2, 3, 4, 5), np.float32)
    output = test_cell(ms.Tensor(x1))
    expect = generate_expect_forward_output(x1)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)

    x2 = generate_random_input((3, 4, 5, 6), np.float32)
    output = test_cell(ms.Tensor(x2))
    expect = generate_expect_forward_output(x2)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_isfinite_backward_dynamic_shape(context_mode):
    """
    Feature: pyboost function.
    Description: test function isfinite backward with dynamic shape.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)

    x_dyn = ms.Tensor(shape=[None, None, None, None], dtype=ms.float32)
    test_cell = test_utils.to_cell_obj(isfinite_backward_func)
    test_cell.set_inputs(x_dyn)

    x1 = generate_random_input((2, 3, 4, 5), np.float32)
    output = test_cell(ms.Tensor(x1))
    expect = generate_expect_backward_output(x1)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)

    x2 = generate_random_input((3, 4, 5, 6), np.float32)
    output = test_cell(ms.Tensor(x2))
    expect = generate_expect_backward_output(x2)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_isfinite_backward_dynamic_rank(context_mode):
    """
    Feature: pyboost function.
    Description: test function isfinite backward with dynamic rank.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)

    x_dyn = ms.Tensor(shape=None, dtype=ms.float32)
    test_cell = test_utils.to_cell_obj(isfinite_backward_func)
    test_cell.set_inputs(x_dyn)

    x1 = generate_random_input((2, 3, 4, 5), np.float32)
    output = test_cell(ms.Tensor(x1))
    expect = generate_expect_backward_output(x1)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)

    x2 = generate_random_input((3, 4, 5, 6), np.float32)
    output = test_cell(ms.Tensor(x2))
    expect = generate_expect_backward_output(x2)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.parametrize('mode', ['pynative', 'KBK', 'GE'])
def test_isfinite_forward_static_shape(mode):
    """
    Feature: Test isfinite with static shape in graph and pynative mode.
    Description: call ops.isfinite with valid input and index.
    Expectation: return the correct value.
    """
    x = generate_random_input((3, 4, 5), np.float32)

    if mode == 'pynative':
        output = isfinite_forward_func(ms.Tensor(x))
    elif mode == 'KBK':
        output = (jit(isfinite_forward_func, jit_config=JitConfig(jit_level="O0")))(ms.Tensor(x))
    else:
        output = (jit(isfinite_forward_func, jit_config=JitConfig(jit_level="O2")))(ms.Tensor(x))

    expect = generate_expect_forward_output(x)
    assert np.allclose(output.asnumpy(), expect, rtol=1e-4)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.parametrize("mode", ['pynative', 'GE', 'KBK'])
def test_isfinite_backward_static_shape(mode):
    """
    Feature: Test isfinite with backward.
    Description: call ops.isfinite with valid input and index.
    Expectation: return the correct value.
    """
    x = generate_random_input((3, 4, 5), np.float32)

    if mode == 'pynative':
        output = isfinite_backward_func(ms.Tensor(x))
    elif mode == 'KBK':
        output = (jit(isfinite_backward_func, jit_config=JitConfig(jit_level="O0")))(ms.Tensor(x))
    else:
        output = (jit(isfinite_backward_func, jit_config=JitConfig(jit_level="O2")))(ms.Tensor(x))

    expect = generate_expect_backward_output(x)
    assert np.allclose(output.asnumpy(), expect, rtol=1e-4)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
def test_isfinite_dynamic_shape_testop():
    """
    Feature: Test isfinite with dynamic shape in graph mode using TEST_OP.
    Description: call ops.isfinite with valid input and index.
    Expectation: return the correct value.
    """
    x1 = generate_random_input((3, 4, 5), np.float32)
    x2 = generate_random_input((3, 7, 8, 3), np.float32)

    TEST_OP(isfinite_forward_func, [[ms.Tensor(x1)], [ms.Tensor(x2)]], 'isfinite')


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.parametrize('param_jit_level', ["O2", "O0"])
def test_isfinite_vmap(param_jit_level):
    """
    Feature: Test isfinite with vmap.
    Description: call ops.isfinite with valid input and index.
    Expectation: return the correct value.
    """
    def _foreach_run(inputs, batch):
        out = []
        for i in range(inputs.shape[batch]):
            if batch == -1:
                input_inner = inputs[..., i]
            else:
                input_inner = inputs[i, ...]
            out.append(isfinite_forward_func(input_inner))
        out = ops.Stack()(out)
        return out

    ms.set_context(jit_level=param_jit_level)
    x = generate_random_input((4, 5, 6), np.float32)

    batch_axis = -1
    output = isfinite_vmap_func(ms.Tensor(x), batch_axis)
    expect = _foreach_run(ms.Tensor(x), batch_axis)
    assert np.allclose(output.asnumpy(), expect.asnumpy(), rtol=1e-4)

    batch_axis = 0
    output = isfinite_vmap_func(ms.Tensor(x), batch_axis)
    expect = _foreach_run(ms.Tensor(x), batch_axis)
    assert np.allclose(output.asnumpy(), expect.asnumpy(), rtol=1e-4)
