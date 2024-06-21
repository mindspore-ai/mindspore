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

import numpy as np
import pytest

import mindspore as ms
from mindspore import Tensor, ops, context
from mindspore.ops import layer_norm

import tests.st.utils.test_utils as test_utils
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP


@test_utils.run_with_cell
def layer_norm_forward_func(input_x, normalized_shape, gamma, beta, eps=1e-7):
    return layer_norm(input_x, normalized_shape, gamma, beta, eps)

@test_utils.run_with_cell
def layer_norm_backward_func(input_x, normalized_shape, gamma, beta, eps=1e-7):
    return ops.grad(layer_norm_forward_func, (0, 2, 3))(input_x, normalized_shape, gamma, beta, eps)

def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)

def layer_norm_forward_func_np(input_x, normalized_shape, gamma, beta, eps=1e-7):
    mean_np = np.mean(input_x, axis=-1, keepdims=True)
    var_np = np.var(input_x, axis=-1, keepdims=True)
    x_norm = (input_x - mean_np) / np.sqrt(var_np + eps)
    return gamma * x_norm + beta

@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_ops_layer_norm_forward(mode):
    """
    Feature: pyboost function.
    Description: test layer_norm_ext forward.
    Expectation: success.
    """
    context.set_context(mode=mode, device_target="Ascend")
    input_x = generate_random_input((64, 255, 448), np.float32)
    normalized_shape = (448,)
    gamma = generate_random_input(normalized_shape, np.float32)
    beta = generate_random_input(normalized_shape, np.float32)
    output = layer_norm_forward_func(Tensor(input_x), normalized_shape, Tensor(gamma), Tensor(beta), eps=1e-5)

    expect_output = layer_norm_forward_func_np(input_x, normalized_shape, gamma, beta, eps=1e-5)
    assert np.allclose(output.asnumpy(), expect_output, atol=1e-6)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_ops_layer_norm_ext_backward(mode):
    """
    Feature: pyboost function.
    Description: test layer_norm_ext backward.
    Expectation: success.
    """
    context.set_context(mode=mode, device_target="Ascend")
    input_x = Tensor(np.array([[1, 2, 3], [1, 2, 3]]).astype(np.float32))
    normalized_shape = (3,)
    gamma = Tensor(np.ones(normalized_shape), ms.float32)
    beta = Tensor(np.zeros(normalized_shape), ms.float32)

    grad_input_x, grad_gamma, grad_beta = layer_norm_backward_func(
        input_x, normalized_shape, gamma, beta)

    expect_grad_input_x = np.array([[0, 0, 0], [0, 0, 0]]).astype(np.float32)
    expect_grad_gamma = np.array([-2.4494894, 0., 2.4494894])
    expect_grad_beta = np.array([2., 2., 2.]).astype(np.float32)
    assert np.allclose(grad_input_x.asnumpy(), expect_grad_input_x, rtol=1e-6, atol=1e-6)
    assert np.allclose(grad_gamma.asnumpy(), expect_grad_gamma, rtol=1e-6, atol=1e-6)
    assert np.allclose(grad_beta.asnumpy(), expect_grad_beta, rtol=1e-6, atol=1e-6)

@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_layer_norm_ext_vmap(mode):
    """
    Feature: test vmap function.
    Description: test layer norm op vmap.
    Expectation: expect correct result.
    """
    context.set_context(mode=mode)
    input_x = ms.Tensor(np.array([[1, 2, 3], [1, 2, 3]]), ms.float32)
    normalized_shape = (3,)
    in_axes = (0, None, 0, 0)
    gamma = ms.Tensor(np.ones(normalized_shape), ms.float32)
    beta = ms.Tensor(np.zeros(normalized_shape), ms.float32)

    batch_input_x = ops.stack((input_x, input_x))
    batch_gamma = ops.stack((gamma, gamma))
    batch_beta = ops.stack((beta, beta))

    layer_norm_vmap = ops.vmap(layer_norm_forward_func, in_axes=in_axes)
    outputs = layer_norm_vmap(batch_input_x, normalized_shape, batch_gamma, batch_beta)
    expect_outputs = np.array([[[-1.22474468, 0., 1.22474468],
                                [-1.22474468, 0., 1.22474468]],
                               [[-1.22474468, 0., 1.22474468],
                                [-1.22474468, 0., 1.22474468]]]).astype(np.float32)
    assert np.allclose(outputs.asnumpy(), expect_outputs, rtol=1e-6, atol=1e-6)

@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
def test_layer_norm_ext_dyn():
    """
    Feature: test layer_norm_ext function.
    Description: test layer norm op dynamic_shape and dynamic_rank.
    Expectation: expect correct result.
    """
    input_x1 = np.random.randn(*(1, 2, 4, 4)).astype(np.float32)
    input_x2 = np.random.randn(*(1, 2, 4)).astype(np.float32)
    normalize_shape = (4,)
    gamma = Tensor(np.ones(normalize_shape), ms.float32)
    beta = Tensor(np.zeros(normalize_shape), ms.float32)
    eps = 1e-7

    TEST_OP(layer_norm_forward_func,
            [[Tensor(input_x1), normalize_shape, gamma, beta, eps],
             [Tensor(input_x2), normalize_shape, gamma, beta, eps]], 'layer_norm_ext', disable_input_check=True)
