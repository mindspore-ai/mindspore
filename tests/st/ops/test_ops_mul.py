# Copyright 2024 Huawei Technomulies Co., Ltd
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
from mindspore.ops import mul

from tests.st.utils import test_utils


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)


def generate_expect_forward_output(x, y):
    return np.multiply(x, y)


def generate_expect_backward_output(x, y):
    return y, x


@test_utils.run_with_cell
def mul_forward_func(x, y):
    return mul(x, y)


@test_utils.run_with_cell
def mul_backward_func(x, y):
    return ops.grad(mul_forward_func, (0, 1))(x, y)


@test_utils.run_with_cell
def mul_vmap_func(x, y):
    return ops.vmap(mul_forward_func, in_axes=0, out_axes=0)(x, y)


@test_utils.run_with_cell
def mul_infervalue_func1():
    x = ms.Tensor(np.array([1, 2, 4]).astype(np.float32))
    y = ms.Tensor(np.array([2, 4, 3]).astype(np.float32))
    return ops.auto_generate.mul(x, y)


@test_utils.run_with_cell
def mul_infervalue_func2():
    x = ms.Tensor(np.array([1, 2, 4]).astype(np.float32))
    y = ms.Tensor(np.array([3]).astype(np.float32))
    return ops.auto_generate.mul(x, y)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_mul_forward(context_mode):
    """
    Feature: pyboost function.
    Description: test function mul forward.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x = generate_random_input((2, 3, 4, 5), np.float32)
    y = generate_random_input((2, 3, 4, 5), np.float32)
    output = mul_forward_func(ms.Tensor(x), ms.Tensor(y))
    expect_out = generate_expect_forward_output(x, y)
    np.testing.assert_allclose(output.asnumpy(), expect_out, rtol=1e-3)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_mul_backward(context_mode):
    """
    Feature: pyboost function.
    Description: test function mul backward.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x = generate_random_input((2, 3, 4), np.float32)
    y = generate_random_input((2, 3, 4), np.float32)
    output = mul_backward_func(ms.Tensor(x), ms.Tensor(y))
    expect_out = generate_expect_backward_output(x, y)
    np.testing.assert_allclose(output[0].asnumpy(), expect_out[0], rtol=1e-3)
    np.testing.assert_allclose(output[1].asnumpy(), expect_out[1], rtol=1e-3)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_mul_vmap(context_mode):
    """
    Feature: pyboost function.
    Description: test function mul vmap feature.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x = generate_random_input((2, 3, 4, 5), np.float32)
    y = generate_random_input((2, 3, 4, 5), np.float32)
    output = mul_vmap_func(ms.Tensor(x), ms.Tensor(y))
    expect_out = generate_expect_forward_output(x, y)
    np.testing.assert_allclose(output.asnumpy(), expect_out, rtol=1e-3)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_mul_forward_dynamic_shape(context_mode):
    """
    Feature: pyboost function.
    Description: test function mul forward with dynamic shape.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x_dyn = ms.Tensor(shape=[None, None, None, None], dtype=ms.float32)
    y_dyn = ms.Tensor(shape=[None, None, None, None], dtype=ms.float32)
    test_cell = test_utils.to_cell_obj(mul_forward_func)
    test_cell.set_inputs(x_dyn, y_dyn)
    x1 = generate_random_input((2, 3, 4, 5), np.float32)
    y1 = generate_random_input((2, 3, 4, 5), np.float32)
    output = test_cell(ms.Tensor(x1), ms.Tensor(y1))
    expect_out = generate_expect_forward_output(x1, y1)
    np.testing.assert_allclose(output.asnumpy(), expect_out, rtol=1e-3)
    x2 = generate_random_input((3, 4, 5, 6), np.float32)
    y2 = generate_random_input((3, 4, 5, 6), np.float32)
    output = test_cell(ms.Tensor(x2), ms.Tensor(y2))
    expect_out = generate_expect_forward_output(x2, y2)
    np.testing.assert_allclose(output.asnumpy(), expect_out, rtol=1e-3)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_mul_forward_dynamic_rank(context_mode):
    """
    Feature: pyboost function.
    Description: test function mul forward with dynamic rank.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x_dyn = ms.Tensor(shape=None, dtype=ms.float32)
    y_dyn = ms.Tensor(shape=None, dtype=ms.float32)
    test_cell = test_utils.to_cell_obj(mul_forward_func)
    test_cell.set_inputs(x_dyn, y_dyn)
    x1 = generate_random_input((2, 3, 4, 5), np.float32)
    y1 = generate_random_input((2, 3, 4, 5), np.float32)
    output = test_cell(ms.Tensor(x1), ms.Tensor(y1))
    expect_out = generate_expect_forward_output(x1, y1)
    np.testing.assert_allclose(output.asnumpy(), expect_out, rtol=1e-3)
    x2 = generate_random_input((4, 5, 6), np.float32)
    y2 = generate_random_input((4, 5, 6), np.float32)
    output = test_cell(ms.Tensor(x2), ms.Tensor(y2))
    expect_out = generate_expect_forward_output(x2, y2)
    np.testing.assert_allclose(output.asnumpy(), expect_out, rtol=1e-3)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_mul_backward_dynamic_shape(context_mode):
    """
    Feature: pyboost function.
    Description: test function mul backward with dynamic shape.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x_dyn = ms.Tensor(shape=[None, None, None, None], dtype=ms.float32)
    y_dyn = ms.Tensor(shape=[None, None, None, None], dtype=ms.float32)
    test_cell = test_utils.to_cell_obj(mul_backward_func)
    test_cell.set_inputs(x_dyn, y_dyn)
    x1 = generate_random_input((2, 3, 4, 5), np.float32)
    y1 = generate_random_input((2, 3, 4, 5), np.float32)
    output = test_cell(ms.Tensor(x1), ms.Tensor(y1))
    expect_out = generate_expect_backward_output(x1, y1)
    np.testing.assert_allclose(output[0].asnumpy(), expect_out[0], rtol=1e-3)
    np.testing.assert_allclose(output[1].asnumpy(), expect_out[1], rtol=1e-3)
    x2 = generate_random_input((3, 4, 5, 6), np.float32)
    y2 = generate_random_input((3, 4, 5, 6), np.float32)
    output = test_cell(ms.Tensor(x2), ms.Tensor(y2))
    expect_out = generate_expect_backward_output(x2, y2)
    np.testing.assert_allclose(output[0].asnumpy(), expect_out[0], rtol=1e-3)
    np.testing.assert_allclose(output[1].asnumpy(), expect_out[1], rtol=1e-3)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_mul_backward_dynamic_rank(context_mode):
    """
    Feature: pyboost function.
    Description: test function mul backward with dynamic rank.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x_dyn = ms.Tensor(shape=None, dtype=ms.float32)
    y_dyn = ms.Tensor(shape=None, dtype=ms.float32)
    test_cell = test_utils.to_cell_obj(mul_backward_func)
    test_cell.set_inputs(x_dyn, y_dyn)
    x1 = generate_random_input((2, 3, 4, 5), np.float32)
    y1 = generate_random_input((2, 3, 4, 5), np.float32)
    output = test_cell(ms.Tensor(x1), ms.Tensor(y1))
    expect_out = generate_expect_backward_output(x1, y1)
    np.testing.assert_allclose(output[0].asnumpy(), expect_out[0], rtol=1e-3)
    np.testing.assert_allclose(output[1].asnumpy(), expect_out[1], rtol=1e-3)
    x2 = generate_random_input((4, 5, 6), np.float32)
    y2 = generate_random_input((4, 5, 6), np.float32)
    output = test_cell(ms.Tensor(x2), ms.Tensor(y2))
    expect_out = generate_expect_backward_output(x2, y2)
    np.testing.assert_allclose(output[0].asnumpy(), expect_out[0], rtol=1e-3)
    np.testing.assert_allclose(output[1].asnumpy(), expect_out[1], rtol=1e-3)
