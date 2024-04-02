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
import os
import numpy as np
import mindspore as ms
from mindspore import ops, jit, JitConfig
from tests.st.utils import test_utils
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)

def generate_expect_forward_output(x):
    return np.all(x)

def generate_expect_forward_output_with_axis(x, in_axis):
    return np.all(x, axis=in_axis)

def generate_expect_backward_output(x):
    return np.zeros_like(x)

@test_utils.run_with_cell
def all_forward_func(x):
    return ms.ops.all(x)

@test_utils.run_with_cell
def all_forward_func_with_axis(x, in_axis):
    return ms.ops.all(x, axis=in_axis)


@test_utils.run_with_cell
def all_backward_func(x):
    return ms.ops.grad(all_forward_func, (0))(x)

@test_utils.run_with_cell
def all_vmap_func(x, in_axes=0):
    return ops.vmap(all_forward_func, in_axes, out_axes=0)(x)

@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_all_forward(context_mode):
    """
    Feature: pyboost function.
    Description: test function all forward.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x = generate_random_input((2, 3, 4, 5), np.float32)
    output = all_forward_func(ms.Tensor(x))
    expect = generate_expect_forward_output(x)
    np.testing.assert_equal(output.asnumpy(), expect)

@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_all_forward_with_axis(context_mode):
    """
    Feature: pyboost function.
    Description: test function all forward with axis.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x = generate_random_input((2, 3, 4, 5), np.float32)
    axis = 1
    output = all_forward_func_with_axis(ms.Tensor(x), axis)
    expect = generate_expect_forward_output_with_axis(x, axis)
    np.testing.assert_equal(output.asnumpy(), expect)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_all_backward(context_mode):
    """
    Feature: pyboost function.
    Description: test function all backward.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x = generate_random_input((2, 3, 4, 5), np.float32)
    output = all_backward_func(ms.Tensor(x))
    expect = generate_expect_backward_output(x)
    np.testing.assert_equal(output.asnumpy(), expect)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_all_vmap(context_mode):
    """
    Feature: pyboost function.
    Description: test function all vmap feature.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x = generate_random_input((2, 3, 4, 5), np.float32)
    output = all_vmap_func(ms.Tensor(x), 0)
    expect = generate_expect_forward_output(x)
    np.testing.assert_equal(output.asnumpy(), expect)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_all_forward_dynamic_shape(context_mode):
    """
    Feature: pyboost function.
    Description: test function all forward with dynamic shape.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)

    x_dyn = ms.Tensor(shape=[None, None, None, None], dtype=ms.float32)
    test_cell = test_utils.to_cell_obj(all_forward_func)
    test_cell.set_inputs(x_dyn)

    x1 = generate_random_input((2, 3, 4, 5), np.float32)
    output = test_cell(ms.Tensor(x1))
    expect = generate_expect_forward_output(x1)
    np.testing.assert_equal(output.asnumpy(), expect)

    x2 = generate_random_input((3, 4, 5, 6), np.float32)
    output = test_cell(ms.Tensor(x2))
    expect = generate_expect_forward_output(x2)
    np.testing.assert_equal(output.asnumpy(), expect)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_all_forward_dynamic_rank(context_mode):
    """
    Feature: pyboost function.
    Description: test function all forward with dynamic rank.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)

    x_dyn = ms.Tensor(shape=None, dtype=ms.float32)
    test_cell = test_utils.to_cell_obj(all_forward_func)
    test_cell.set_inputs(x_dyn)

    x1 = generate_random_input((2, 3, 4, 5), np.float32)
    output = test_cell(ms.Tensor(x1))
    expect = generate_expect_forward_output(x1)
    np.testing.assert_equal(output.asnumpy(), expect)

    x2 = generate_random_input((3, 4, 5, 6), np.float32)
    output = test_cell(ms.Tensor(x2))
    expect = generate_expect_forward_output(x2)
    np.testing.assert_equal(output.asnumpy(), expect)



@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_all_forward_dynamic_rank_axis_none(context_mode):
    """
    Feature: pyboost function.
    Description: test function all forward with dynamic rank axis None.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)

    axis = None
    x_dyn = ms.Tensor(shape=None, dtype=ms.float32)
    test_cell = test_utils.to_cell_obj(all_forward_func_with_axis)
    test_cell.set_inputs(x_dyn, axis)


    x1 = generate_random_input((2, 3, 4, 5), np.float32)
    output = test_cell(ms.Tensor(x1), axis)
    expect = generate_expect_forward_output_with_axis(x1, in_axis=axis)
    np.testing.assert_equal(output.asnumpy(), expect)

    x2 = generate_random_input((3, 4, 5, 6), np.float32)
    output = test_cell(ms.Tensor(x2), axis)
    expect = generate_expect_forward_output_with_axis(x2, in_axis=axis)
    np.testing.assert_equal(output.asnumpy(), expect)




@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.parametrize('mode', ['pynative', 'KBK', 'GE'])
def test_all_forward_static_shape(mode):
    """
    Feature: Test all with static shape in graph and pynative mode.
    Description: call ops.all with valid input and index.
    Expectation: return the correct value.
    """
    x = generate_random_input((3, 4, 5), np.float32)

    if mode == 'pynative':
        output = all_forward_func(ms.Tensor(x))
    elif mode == 'KBK':
        output = (jit(all_forward_func, jit_config=JitConfig(jit_level="O0")))(ms.Tensor(x))
    else:
        output = (jit(all_forward_func, jit_config=JitConfig(jit_level="O2")))(ms.Tensor(x))

    expect = generate_expect_forward_output(x)
    np.testing.assert_equal(output.asnumpy(), expect)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.parametrize('jit_level', ["O0", "O2"])
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
def test_all_dynamic_shape_testop(jit_level):
    """
    Feature: Test all with dynamic shape in graph mode using TEST_OP.
    Description: call ops.all with valid input and index.
    Expectation: return the correct value.
    """
    x1 = generate_random_input((3, 4, 5), np.float32)
    x2 = generate_random_input((3, 7, 8, 3), np.float32)
    axis1 = (0, 1)
    axis2 = (2)
    keep_dims1 = True
    keep_dims2 = False

    TEST_OP(all_forward_func, [[ms.Tensor(x1), axis1, keep_dims1], [ms.Tensor(x2), axis2, keep_dims2]],
            grad=True, jit_level=jit_level)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.parametrize('graph_level', ["0", "1"])
def test_all_vmap(graph_level):
    """
    Feature: Test all with vmap.
    Description: call ops.all with valid input and index.
    Expectation: return the correct value.
    """
    def _foreach_run(inputs, batch):
        out = []
        for i in range(inputs.shape[batch]):
            if batch == -1:
                input_inner = inputs[..., i]
            else:
                input_inner = inputs[i, ...]
            out.append(all_forward_func(input_inner))
        out = ops.Stack()(out)
        return out

    os.environ['GRAPH_OP_RUN'] = graph_level
    x = generate_random_input((4, 5, 6), np.float32)

    batch_axis = -1
    output = all_vmap_func(ms.Tensor(x), batch_axis)
    expect = _foreach_run(ms.Tensor(x), batch_axis)
    np.testing.assert_equal(output.asnumpy(), expect.asnumpy())

    batch_axis = 0
    output = all_vmap_func(ms.Tensor(x), batch_axis)
    expect = _foreach_run(ms.Tensor(x), batch_axis)
    np.testing.assert_equal(output.asnumpy(), expect.asnmpy())

    del os.environ['GRAPH_OP_RUN']
