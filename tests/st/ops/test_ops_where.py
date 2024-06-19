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


"""test where"""
import numpy as np
import pytest
import mindspore.common.dtype as mstype

from mindspore.ops import where
from mindspore import ops, Tensor, jit, JitConfig, context
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP
from tests.st.utils import test_utils


def generate_random_input(shape, dtype):
    return Tensor(np.random.randn(*shape).astype(dtype))


def generate_expect_forward_output(condition, x, y):
    return np.where(condition, x, y)


def generate_expect_backward_output(condition):
    return np.zeros(np.shape(condition), dtype=np.bool_),\
           np.where(condition, 1, 0), np.where(condition, 0, 1)


@test_utils.run_with_cell
def where_forward_func(condition, x, y):
    return where(condition, x, y)


@test_utils.run_with_cell
def where_backward_func(condition, x, y):
    return ops.grad(where_forward_func, (0, 1, 2))(condition, x, y)


@test_utils.run_with_cell
def where_vmap_func(condition, x, y, in_axes=0):
    return ops.vmap(where_forward_func, in_axes, out_axes=0)(condition, x, y)


@pytest.mark.level2
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_ops_where(mode):
    """
    Feature: ops.where
    Description: Verify the result of ops.where
    Expectation: success
    """
    context.set_context(mode=mode)
    x = Tensor(np.arange(4).reshape((2, 2)), mstype.float32)
    y = Tensor(np.ones((2, 2)), mstype.float32)
    condition = x < 3
    output = where_forward_func(condition, x, y)
    expected = np.array([[0, 1], [2, 1]], dtype=np.float32)
    assert np.allclose(output.asnumpy(), expected)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_where_float32(mode):
    """
    Feature: Test functional where operator. Support x or y is a float32 Tensor.
    Description: Operator where's inputs `x` and `y` are Tensor with float32 type.
    Expectation: Assert result.
    """
    context.set_context(mode=mode)
    cond = np.array([[True, False], [True, False]]).astype(np.bool)
    x = np.array([[1.2, 1], [1, 0]]).astype(np.float32)
    y = np.array([[1, 2], [3, 4.0]]).astype(np.float32)
    output = where_forward_func(Tensor(cond), Tensor(x), Tensor(y))
    print(output.asnumpy())
    expect = [[1.2, 2], [1, 4.0]]
    error = np.ones(shape=[2, 2]) * 1.0e-6
    diff = output.asnumpy() - expect
    assert np.all(diff < error)
    assert np.all(-diff < error)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_where_float16(mode):
    """
    Feature: Test functional where operator. Support x or y is a float16 Tensor.
    Description: Operator where's inputs `x` and `y` are Tensor with float16 type.
    Expectation: Assert result.
    """
    context.set_context(mode=mode)
    cond = np.array([[True, False], [True, False]]).astype(np.bool)
    x = np.array([[1.2, 1], [1, 0]]).astype(np.float16)
    y = np.array([[1, 2], [3, 4.0]]).astype(np.float16)
    output = where_forward_func(Tensor(cond), Tensor(x), Tensor(y))
    print(output.asnumpy())
    expect = [[1.2, 2], [1, 4.0]]
    error = np.ones(shape=[2, 2]) * 1.0e-3
    diff = output.asnumpy() - expect
    assert np.all(diff < error)
    assert np.all(-diff < error)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_where_int32(mode):
    """
    Feature: Test functional where operator. Support x or y is a int32 Tensor.
    Description: Operator where's inputs `x` and `y` are Tensor with int32 type.
    Expectation: Assert result.
    """
    context.set_context(mode=mode)
    cond = np.array([[True, False], [True, False]]).astype(np.bool)
    x = np.array([[12, 1], [1, 0]]).astype(np.int32)
    y = np.array([[1, 2], [3, 4]]).astype(np.int32)
    output = where_forward_func(Tensor(cond), Tensor(x), Tensor(y))
    print(output.asnumpy())
    expect = [[12, 2], [1, 4]]
    error = np.ones(shape=[2, 2]) * 1.0e-6
    diff = output.asnumpy() - expect
    assert np.all(diff < error)
    assert np.all(-diff < error)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_functional_where_scalar(mode):
    """
    Feature: Test functional where operator. Support x or y is a int/float.
    Description: Operator where's input `x` is a Tensor with int32 type, input `y` is a int.
    Expectation: Assert result.
    """
    context.set_context(mode=mode)
    cond = np.array([[True, False], [True, False]]).astype(np.bool)
    x = np.array([[12, 1], [1, 0]]).astype(np.int32)
    y = 2
    output = where_forward_func(Tensor(cond), Tensor(x), y)
    print(output.asnumpy())
    expect = [[12, 2], [1, 2]]
    error = np.ones(shape=[2, 2]) * 1.0e-6
    diff = output.asnumpy() - expect
    assert np.all(diff < error)
    assert np.all(-diff < error)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_functional_where_broadcast(mode):
    """
    Feature: Test functional where operator support broadcast input.
    Description: Operator where's support broadcast input.
    Expectation: Assert result.
    """
    context.set_context(mode=mode)
    cond = Tensor(np.random.rand(1, 65, 54, 12, 5, 2), dtype=mstype.bool_)
    x = Tensor(np.random.rand(5, 5, 65, 1, 12, 5, 2).astype(np.float32))
    y = Tensor(np.random.rand(65, 54, 1, 5, 2).astype(np.float32))
    ret = where_forward_func(cond, x, y)
    assert ret.shape == (5, 5, 65, 54, 12, 5, 2)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_cpu_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.parametrize('mode', ['pynative', 'KBK', 'GE'])
def test_where_ext_static_shape(mode):
    """
    Feature: Test where with static shape in graph and pynative mode.
    Description: call ops.where with valid input and index.
    Expectation: return the correct value.
    """
    x = generate_random_input((2, 3, 4, 5), np.float32)
    y = generate_random_input((2, 3, 4, 5), np.float32)
    cond = x > 0

    if mode == 'pynative':
        ms_out = where_forward_func(cond, x, y)
    elif mode == 'KBK':
        ms_out = (jit(where_forward_func, jit_config=JitConfig(jit_level="O0")))(cond, x, y)
    else:
        ms_out = (jit(where_forward_func, jit_config=JitConfig(jit_level="O2")))(cond, x, y)

    expect = generate_expect_forward_output(cond.asnumpy(), x.asnumpy(), y.asnumpy())
    assert np.allclose(ms_out.asnumpy(), expect, rtol=1e-4)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_cpu_training
@pytest.mark.platform_x86_gpu_training
def test_where_ext_dynamic_shape():
    """
    Feature: Test where with dynamic shape in graph mode.
    Description: call ops.where with valid input and index.
    Expectation: return the correct value.
    """
    x1 = generate_random_input((2, 3, 4, 5), np.float32)
    y1 = generate_random_input((2, 3, 4, 5), np.float32)
    cond1 = x1 > 0

    x2 = generate_random_input((6, 7, 8), np.float32)
    y2 = generate_random_input((6, 7, 8), np.float32)
    cond2 = x2 > 0
    TEST_OP(where_forward_func, [[cond1, x1, y1], [cond2, x2, y2]], '', disable_yaml_check=True)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_cpu_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.parametrize('param_jit_level', ["O2", "O0"])
def test_where_vmap(param_jit_level):
    """
    Feature: Test where with vmap.
    Description: call ops.where with valid input and index.
    Expectation: return the correct value.
    """
    def _foreach_run(condition, x, y, batch):
        out = []
        for i in range(condition.shape[batch]):
            if batch == -1:
                cond_inner = condition[..., i]
                x_inner = x[..., i]
                y_inner = y[..., i]
            else:
                cond_inner = condition[i, ...]
                x_inner = x[i, ...]
                y_inner = y[i, ...]
            out.append(where_forward_func(cond_inner, x_inner, y_inner))
        out = ops.Stack()(out)
        return out

    context.set_context(jit_level=param_jit_level)
    x = generate_random_input((2, 3, 4, 5), np.float32)
    y = generate_random_input((2, 3, 4, 5), np.float32)
    cond = x > 0

    batch_axis = -1
    output = where_vmap_func(cond, x, y, batch_axis)
    expect = _foreach_run(cond, x, y, batch_axis)
    assert np.allclose(output.asnumpy(), expect.asnumpy(), rtol=1e-4)

    batch_axis = 0
    output = where_vmap_func(cond, x, y, batch_axis)
    expect = _foreach_run(cond, x, y, batch_axis)
    assert np.allclose(output.asnumpy(), expect.asnumpy(), rtol=1e-4)



@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_cpu_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.parametrize("mode", ['pynative', 'GE', 'KBK'])
def test_where_ext_grad(mode):
    """
    Feature: Test where with backward.
    Description: call ops.where with valid input and index.
    Expectation: return the correct value.
    """
    x = generate_random_input((2, 3, 4, 5), np.float32)
    y = generate_random_input((2, 3, 4, 5), np.float32)
    cond = x > 0

    if mode == 'pynative':
        ms_cond, ms_x, ms_y = where_backward_func(cond, x, y)
    elif mode == 'KBK':
        ms_cond, ms_x, ms_y = (jit(where_backward_func, jit_config=JitConfig(jit_level="O0")))(cond, x, y)
    else:
        ms_cond, ms_x, ms_y = (jit(where_backward_func, jit_config=JitConfig(jit_level="O2")))(cond, x, y)
    expect_cond, expect_x, expect_y = generate_expect_backward_output(cond.asnumpy())
    assert np.allclose(ms_cond.asnumpy(), expect_cond, rtol=1e-4)
    assert np.allclose(ms_x.asnumpy(), expect_x, rtol=1e-4)
    assert np.allclose(ms_y.asnumpy(), expect_y, rtol=1e-4)
