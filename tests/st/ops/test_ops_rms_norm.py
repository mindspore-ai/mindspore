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
# pylint: disable=unused-variable
import pytest
import numpy as np
import mindspore as ms
from mindspore import ops, Tensor, jit, JitConfig
from tests.mark_utils import arg_mark
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP


def rms_norm_forward_func(x, gamma, epsilon):
    y, rstd = ops.rms_norm(x, gamma, epsilon)
    return y, rstd


def rms_norm_backward_func(x, gamma, epsilon):
    x_grad, gamma_grad = ops.grad(rms_norm_forward_func, (0, 1))(x, gamma, epsilon)
    return x_grad, gamma_grad

def rms_norm_grad_numpy_impl(x, gamma, epsilon, y_grad):
    x_dim = len(x.shape)
    gamma_dim = len(gamma.shape)
    reduce_dims_1 = tuple(range(x_dim - gamma_dim))
    reduce_dims_2 = tuple(range(x_dim - gamma_dim, x_dim))
    norm_ele_num = np.prod(gamma.shape)
    var = np.mean(np.power(x, 2), reduce_dims_2, keepdims=True)
    std = np.sqrt(var + epsilon)
    rstd = 1/ std
    np.broadcast(rstd, y_grad)
    np.broadcast(gamma, y_grad)
    dgamma = np.sum(y_grad * x * rstd, reduce_dims_1, keepdims=True)
    dxp1 = y_grad * gamma * rstd
    dxp2 = x * np.sum(rstd * rstd * rstd * y_grad * gamma * x, reduce_dims_2, keepdims=True) / norm_ele_num
    dx = dxp1 - dxp2
    dgamma = np.reshape(dgamma, gamma.shape)
    return dx, dgamma.astype(np.float32)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK', 'GE'])
@pytest.mark.parametrize('input_dtype', [np.float32, np.float16])
def test_rms_norm_forward(mode, input_dtype):
    """
    Feature: Ops.
    Description: test RmsNorm.
    Expectation: expect correct result.
    """
    x = Tensor(np.arange(16).reshape(2, 2, 2, 2).astype(input_dtype))
    gamma = Tensor(np.ones((2, 2, 2)).astype(input_dtype))
    eps = 1e-5
    expect_y = np.array([[[[0.0, 0.239046],
                           [0.478091, 0.717137]],
                          [[0.956183, 1.19523],
                           [1.43427, 1.67332]]],
                         [[[0.682242, 0.767523],
                           [0.852803, 0.938083]],
                          [[1.02336, 1.10864],
                           [1.19392, 1.2792]]]])
    expect_rstd = np.array([[[[0.239046]]],
                            [[[0.0852803]]]])
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        y, rstd = rms_norm_forward_func(x, gamma, eps)
    elif mode == 'KBK':
        ms.context.set_context(mode=ms.GRAPH_MODE)
        y, rstd = (jit(rms_norm_forward_func, jit_config=JitConfig(jit_level="O0")))(x, gamma, eps)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE)
        y, rstd = rms_norm_forward_func(x, gamma, eps)
    loss = 1e-4 if input_dtype == np.float32 else 1e-3
    np.testing.assert_allclose(y.asnumpy(), expect_y, rtol=loss)
    np.testing.assert_allclose(rstd.asnumpy(), expect_rstd, rtol=loss)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='unessential')
# the last dim of input must be integer multiple of 64 in 910a
@pytest.mark.parametrize('mode', ['pynative', 'KBK', 'GE'])
@pytest.mark.parametrize('input_dtype', [np.float32, np.float16])
def test_rms_norm_backward(mode, input_dtype):
    """
    Feature: Ops.
    Description: test RmsNormGrad.
    Expectation: expect correct result.
    """
    x = Tensor(np.arange(16).reshape(2, 2, 2, 2).astype(input_dtype))
    gamma = Tensor(np.ones((2, 2, 2)).astype(input_dtype))
    eps = 1e-7
    expect_x_grad = np.array([[[[0.23904569, 0.19123657],
                                [0.14342743, 0.0956183]],
                               [[0.04780918, 0.0],
                                [-0.047809094, -0.095618218]]],
                              [[[0.028220026, 0.021087488],
                                [0.013954958, 0.0068224263]],
                               [[-0.0003101, -0.0074426359],
                                [-0.014575172, -0.021707699]]]])
    expect_gamma_grad = np.array([[[0.6822423, 1.0065683],
                                   [1.3308942, 1.6552203]],
                                  [[1.9795461, 2.303872],
                                   [2.6281981, 2.9525242]]])
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        x_grad, gamma_grad = rms_norm_backward_func(x, gamma, eps)
    elif mode == 'KBK':
        ms.context.set_context(mode=ms.GRAPH_MODE)
        x_grad, gamma_grad = (jit(rms_norm_backward_func, jit_config=JitConfig(jit_level="O0")))(x, gamma, eps)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE)
        x_grad, gamma_grad = rms_norm_backward_func(x, gamma, eps)
    loss = 1e-4 if input_dtype == np.float32 else 1e-3
    np.testing.assert_allclose(x_grad.asnumpy(), expect_x_grad, rtol=loss)
    np.testing.assert_allclose(gamma_grad.asnumpy(), expect_gamma_grad, rtol=loss)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK', 'GE'])
@pytest.mark.parametrize('input_dtype', [np.float32, np.float16])
def test_rms_norm_backward_cmp_with_numpy(mode, input_dtype):
    """
    Feature: Ops.
    Description: test RmsNormGrad.
    Expectation: expect correct result.
    """
    x_np = np.random.randn(2, 3, 4, 64).astype(input_dtype)
    x = Tensor(x_np)
    gamma_np = np.random.randn(3, 4, 64).astype(input_dtype)
    gamma = Tensor(gamma_np)
    y_grad_np = np.ones((2, 3, 4, 64)).astype(input_dtype)
    eps = 1e-7
    expect_x_grad, expect_gamma_grad = rms_norm_grad_numpy_impl(x_np, gamma_np, eps, y_grad_np)
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        x_grad, gamma_grad = rms_norm_backward_func(x, gamma, eps)
    elif mode == 'KBK':
        ms.context.set_context(mode=ms.GRAPH_MODE)
        x_grad, gamma_grad = (jit(rms_norm_backward_func, jit_config=JitConfig(jit_level="O0")))(x, gamma, eps)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE)
        x_grad, gamma_grad = rms_norm_backward_func(x, gamma, eps)
    a_loss = 1e-4 if input_dtype == np.float32 else 0.2
    r_loss = 1e-3 if input_dtype == np.float32 else 5e-1
    np.testing.assert_allclose(x_grad.asnumpy(), expect_x_grad, rtol=r_loss, atol=a_loss)
    np.testing.assert_allclose(gamma_grad.asnumpy(), expect_gamma_grad, rtol=r_loss, atol=a_loss)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='unessential')
def test_rms_norm_dynamic_shape():
    """
    Feature: Test gather with dynamic shape in graph mode.
    Description: call ops.extend.gather with valid input and index.
    Expectation: return the correct value.
    """
    x_1 = Tensor(np.random.randn(3, 4, 64).astype(np.float32))
    gamma_1 = Tensor(np.random.randn(4, 64).astype(np.float32))
    eps_1 = 1e-4

    x_2 = Tensor(np.random.randn(2, 3, 4, 64).astype(np.float32))
    gamma_2 = Tensor(np.random.randn(3, 4, 64).astype(np.float32))
    eps_2 = 1e-5
    TEST_OP(rms_norm_forward_func, [[x_1, gamma_1, eps_1], [x_2, gamma_2, eps_2]], "rms_norm")
