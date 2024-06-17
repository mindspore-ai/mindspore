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
import time
import numpy as np
import mindspore as ms
from mindspore import ops, Tensor, jit, JitConfig
from mindspore.ops import square
from mindspore.common.api import _pynative_executor
from tests.st.utils import test_utils
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)


def generate_expect_forward_output(x):
    return np.square(x)


def generate_expect_backward_output(x):
    return 2 * x


@test_utils.run_with_cell
def square_forward_func(x):
    return square(x)


@test_utils.run_with_cell
def square_backward_func(x):
    return ops.grad(square_forward_func, (0))(x)


@test_utils.run_with_cell
def square_vmap_func(x, in_axes=0):
    return ops.vmap(square_forward_func, in_axes, out_axes=0)(x)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_square_forward(context_mode):
    """
    Feature: pyboost function.
    Description: test function square forward.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x = generate_random_input((2, 3, 4, 5), np.float32)
    output = square_forward_func(ms.Tensor(x))
    expect = generate_expect_forward_output(x)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_square_backward(context_mode):
    """
    Feature: pyboost function.
    Description: test function square backward.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x = generate_random_input((2, 3, 4, 5), np.float32)
    output = square_backward_func(ms.Tensor(x))
    expect = generate_expect_backward_output(x)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_square_vmap(context_mode):
    """
    Feature: pyboost function.
    Description: test function square vmap feature.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x = generate_random_input((2, 3, 4, 5), np.float32)
    output = square_vmap_func(ms.Tensor(x), 0)
    expect = generate_expect_forward_output(x)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_square_forward_dynamic_shape(context_mode):
    """
    Feature: pyboost function.
    Description: test function square forward with dynamic shape.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)

    x_dyn = ms.Tensor(shape=[None, None, None, None], dtype=ms.float32)
    test_cell = test_utils.to_cell_obj(square_forward_func)
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
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_square_forward_dynamic_rank(context_mode):
    """
    Feature: pyboost function.
    Description: test function square forward with dynamic rank.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)

    x_dyn = ms.Tensor(shape=None, dtype=ms.float32)
    test_cell = test_utils.to_cell_obj(square_forward_func)
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
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_square_backward_dynamic_shape(context_mode):
    """
    Feature: pyboost function.
    Description: test function square backward with dynamic shape.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)

    x_dyn = ms.Tensor(shape=[None, None, None, None], dtype=ms.float32)
    test_cell = test_utils.to_cell_obj(square_backward_func)
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
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_square_backward_dynamic_rank(context_mode):
    """
    Feature: pyboost function.
    Description: test function square backward with dynamic rank.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)

    x_dyn = ms.Tensor(shape=None, dtype=ms.float32)
    test_cell = test_utils.to_cell_obj(square_backward_func)
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
@pytest.mark.platform_x86_cpu_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.parametrize('mode', ['pynative', 'KBK', 'GE'])
def test_square_forward_static_shape(mode):
    """
    Feature: Test square with static shape in graph and pynative mode.
    Description: call ops.square with valid input and index.
    Expectation: return the correct value.
    """
    x = generate_random_input((3, 4, 5), np.float32)

    if mode == 'pynative':
        output = square_forward_func(ms.Tensor(x))
    elif mode == 'KBK':
        output = (jit(square_forward_func, jit_config=JitConfig(jit_level="O0")))(ms.Tensor(x))
    else:
        output = (jit(square_forward_func, jit_config=JitConfig(jit_level="O2")))(ms.Tensor(x))

    expect = generate_expect_forward_output(x)
    assert np.allclose(output.asnumpy(), expect, rtol=1e-4)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_cpu_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.parametrize("mode", ['pynative', 'GE', 'KBK'])
def test_square_backward_static_shape(mode):
    """
    Feature: Test square with backward.
    Description: call ops.square with valid input and index.
    Expectation: return the correct value.
    """
    x = generate_random_input((3, 4, 5), np.float32)

    if mode == 'pynative':
        output = square_backward_func(ms.Tensor(x))
    elif mode == 'KBK':
        output = (jit(square_backward_func, jit_config=JitConfig(jit_level="O0")))(ms.Tensor(x))
    else:
        output = (jit(square_backward_func, jit_config=JitConfig(jit_level="O2")))(ms.Tensor(x))

    expect = generate_expect_backward_output(x)
    assert np.allclose(output.asnumpy(), expect, rtol=1e-4)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_cpu_training
@pytest.mark.platform_x86_gpu_training
def test_square_dynamic_shape_testop():
    """
    Feature: Test square with dynamic shape in graph mode using TEST_OP.
    Description: call ops.square with valid input and index.
    Expectation: return the correct value.
    """
    x1 = generate_random_input((3, 4, 5), np.float32)
    x2 = generate_random_input((3, 7, 8, 3), np.float32)

    TEST_OP(square_forward_func, [[ms.Tensor(x1)], [ms.Tensor(x2)]], 'square')


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_cpu_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.parametrize('param_jit_level', ["O2", "O0"])
def test_square_vmap(param_jit_level):
    """
    Feature: Test square with vmap.
    Description: call ops.square with valid input and index.
    Expectation: return the correct value.
    """
    def _foreach_run(inputs, batch):
        out = []
        for i in range(inputs.shape[batch]):
            if batch == -1:
                input_inner = inputs[..., i]
            else:
                input_inner = inputs[i, ...]
            out.append(square_forward_func(input_inner))
        out = ops.Stack()(out)
        return out

    ms.set_context(jit_level=param_jit_level)
    x = generate_random_input((4, 5, 6), np.float32)

    batch_axis = -1
    output = square_vmap_func(ms.Tensor(x), batch_axis)
    expect = _foreach_run(ms.Tensor(x), batch_axis)
    assert np.allclose(output.asnumpy(), expect.asnumpy(), rtol=1e-4)

    batch_axis = 0
    output = square_vmap_func(ms.Tensor(x), batch_axis)
    expect = _foreach_run(ms.Tensor(x), batch_axis)
    assert np.allclose(output.asnumpy(), expect.asnumpy(), rtol=1e-4)



@pytest.mark.parametrize('batch', [8, 16, 32, 64, 128])
def test_square_vmap_perf(batch):
    """
    Feature: Test square with vmap perf.
    Description: call ops.square with valid input and index.
    Expectation: return the correct value.
    """
    # @jit
    def _foreach_run(inputs):
        out = []
        for inputs_inner in inputs:
            out_inner = []
            for inputs_tensor in inputs_inner:
                out_inner.append(square_forward_func(inputs_tensor))
            out.append(out_inner)
        return out

    def _preprocess(inputs):
        out = []
        for i in range(inputs.shape[0]):
            out_inner = []
            for j in range(inputs.shape[1]):
                out_inner.append(inputs[i, j, ...])
            out.append(out_inner)
        return out

    def _postprocess(inputs, out_list):
        out_inner = []
        for i in range(inputs.shape[0]):
            out_inner.append(ops.Stack()(out_list[i]))
        out = ops.Stack()(out_inner)
        return out

    ops.Abs()(Tensor(5.0))
    _pynative_executor.sync()
    x = generate_random_input((batch, 4, 4, 4), np.float32)
    x_ori = _preprocess(ms.Tensor(x))

    _pynative_executor.sync()
    run_times = 100
    start = time.time()
    for _ in range(run_times):
        output = square_vmap_func(ms.Tensor(x), 0)
    _pynative_executor.sync()
    end = time.time()
    vmap_duration = end - start

    start = time.time()
    for _ in range(run_times):
        ori_out_list = _foreach_run(x_ori)
    _pynative_executor.sync()
    end = time.time()
    foreach_duration = end - start

    ori_out = _postprocess(ms.Tensor(x), ori_out_list)
    assert np.allclose(output.asnumpy(), ori_out.asnumpy(), rtol=1e-4)

    print(f"Testing vmap perf with batch={batch}:")
    print(f"foreach_duration: {foreach_duration / run_times}")
    print(f"vmap_duration: {vmap_duration / run_times}")
    print(f"improve_times: {foreach_duration / vmap_duration}\n")
