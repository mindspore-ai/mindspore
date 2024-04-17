# Copyright 2023 Huawei Technoreluies Co., Ltd
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
from mindspore import context, Tensor
from mindspore.ops import relu
from tests.st.utils import test_utils

def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)

def generate_expect_forward_output(x, dtype):
    return np.maximum(x, 0).astype(dtype)

def generate_expect_backward_output(x, dtype):
    return np.where(x > 0, 1, 0).astype(dtype)

@test_utils.run_with_cell
def relu_forward_func(x):
    return relu(x)

@test_utils.run_with_cell
def relu_backward_func(x):
    return ms.ops.grad(relu_forward_func, (0))(x)

@test_utils.run_with_cell
def relu_vmap_func(x):
    return ms.ops.vmap(relu_forward_func, in_axes=0, out_axes=0)(x)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize("mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_relu_forward(mode):
    """
    Feature: test relu operator
    Description: test relu run by pyboost
    Expectation: success
    """
    context.set_context(mode=mode)
    np_array = np.random.rand(2, 3, 4)
    x = Tensor(np_array, ms.float32)
    output = relu_forward_func(x)
    expect = generate_expect_forward_output(np_array, np.float32)
    assert np.allclose(output.asnumpy(), expect)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize("mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_relu_backward(mode):
    """
    Feature: test relu operator
    Description: test relu run by pyboost
    Expectation: success
    """
    context.set_context(mode=mode)
    np_array = np.random.rand(2, 3, 4).astype(np.float32)
    x = Tensor(np_array, ms.float32)
    output = relu_backward_func(x)
    expect = generate_expect_backward_output(np_array, np.float32)
    assert np.allclose(output.asnumpy(), expect)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize("mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_relu_vmap(mode):
    """
    Feature: pyboost function.
    Description: test function relu vmap feature.
    Expectation: expect correct result.
    """
    context.set_context(mode=mode)
    x = generate_random_input((2, 3, 4, 5), np.float32)
    output = relu_vmap_func(Tensor(x))
    expect = generate_expect_forward_output(x, np.float32)
    np.testing.assert_allclose(output.asnumpy(), expect)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize("mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_relu_forward_dynamic_shape(mode):
    """
    Feature: pyboost function.
    Description: test function relu forward with dynamic shape.
    Expectation: expect correct result.
    """
    context.set_context(mode=mode)

    x_dyn = Tensor(shape=[None, None, None, None], dtype=ms.float32)
    test_cell = test_utils.to_cell_obj(relu_forward_func)
    test_cell.set_inputs(x_dyn)

    x1 = generate_random_input((2, 3, 4, 5), np.float32)
    output = test_cell(Tensor(x1))
    expect = generate_expect_forward_output(x1, np.float32)
    np.testing.assert_allclose(output.asnumpy(), expect)

    x2 = generate_random_input((3, 4, 5, 6), np.float32)
    output = test_cell(Tensor(x2))
    expect = generate_expect_forward_output(x2, np.float32)
    np.testing.assert_allclose(output.asnumpy(), expect)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize("mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_relu_forward_dynamic_rank(mode):
    """
    Feature: pyboost function.
    Description: test function relu forward with dynamic rank.
    Expectation: expect correct result.
    """
    context.set_context(mode=mode)

    x_dyn = Tensor(shape=None, dtype=ms.float32)
    test_cell = test_utils.to_cell_obj(relu_forward_func)
    test_cell.set_inputs(x_dyn)

    x1 = generate_random_input((2, 3, 4, 5), np.float32)
    output = test_cell(Tensor(x1))
    expect = generate_expect_forward_output(x1, np.float32)
    np.testing.assert_allclose(output.asnumpy(), expect)

    x2 = generate_random_input((3, 4, 5, 6), np.float32)
    output = test_cell(Tensor(x2))
    expect = generate_expect_forward_output(x2, np.float32)
    np.testing.assert_allclose(output.asnumpy(), expect)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize("mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_relu_backward_dynamic_shape(mode):
    """
    Feature: pyboost function.
    Description: test function relu backward with dynamic shape.
    Expectation: expect correct result.
    """
    context.set_context(mode=mode)

    x_dyn = Tensor(shape=[None, None, None, None], dtype=ms.float32)
    test_cell = test_utils.to_cell_obj(relu_backward_func)
    test_cell.set_inputs(x_dyn)

    x1 = generate_random_input((2, 3, 4, 5), np.float32)
    output = test_cell(Tensor(x1))
    expect = generate_expect_backward_output(x1, np.float32)
    np.testing.assert_allclose(output.asnumpy(), expect)

    x2 = generate_random_input((3, 4, 5, 6), np.float32)
    output = test_cell(Tensor(x2))
    expect = generate_expect_backward_output(x2, np.float32)
    np.testing.assert_allclose(output.asnumpy(), expect)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize("mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_relu_backward_dynamic_rank(mode):
    """
    Feature: pyboost function.
    Description: test function relu backward with dynamic rank.
    Expectation: expect correct result.
    """
    context.set_context(mode=mode)

    x_dyn = Tensor(shape=None, dtype=ms.float32)
    test_cell = test_utils.to_cell_obj(relu_backward_func)
    test_cell.set_inputs(x_dyn)

    x1 = generate_random_input((2, 3, 4, 5), np.float32)
    output = test_cell(Tensor(x1))
    expect = generate_expect_backward_output(x1, np.float32)
    np.testing.assert_allclose(output.asnumpy(), expect)

    x2 = generate_random_input((3, 4, 5, 6), np.float32)
    output = test_cell(Tensor(x2))
    expect = generate_expect_backward_output(x2, np.float32)
    np.testing.assert_allclose(output.asnumpy(), expect)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.parametrize("mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_relu_bfloat16(mode):
    """
    Feature: test relu operator
    Description: test relu run by pyboost
    Expectation: success
    """
    context.set_context(mode=mode)
    np_array = np.random.rand(2, 3, 4)
    x = Tensor(np_array, ms.bfloat16)
    output = relu_forward_func(x)
    expect = generate_expect_forward_output(np_array, np.float32)
    assert np.allclose(output.float().asnumpy(), expect)
