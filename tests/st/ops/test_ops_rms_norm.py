# Copyright 2023 Huawei Technologies Co., Ltd
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
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP


def rms_norm_forward_func(x, gamma, epsilon):
    y, rstd = ops.rms_norm(x, gamma, epsilon)
    return y, rstd


def rms_norm_backward_func(x, gamma, epsilon):
    x_grad, gamma_grad = ops.grad(rms_norm_forward_func, (0, 1))(x, gamma, epsilon)
    return x_grad, gamma_grad


@pytest.mark.level0
@pytest.mark.env_onecard
# output result is wrong in 910a, waiting for cann package update.
@pytest.mark.platform_x86_ascend910b_training
@pytest.mark.platform_arm_ascend910b_training
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


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_ascend910b_training
@pytest.mark.platform_arm_ascend910b_training
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


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_ascend910b_training
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.parametrize('jit_level', ["O0", "O2"])
def test_rms_norm_dynamic_shape(jit_level):
    """
    Feature: Test gather with dynamic shape in graph mode.
    Description: call ops.extend.gather with valid input and index.
    Expectation: return the correct value.
    """
    x_1 = Tensor(np.arange(24).reshape(2, 3, 4).astype(np.float32))
    gamma_1 = Tensor(np.random.randn(3, 4).astype(np.float32))
    eps_1 = 1e-4

    x_2 = Tensor(np.arange(120).reshape(2, 3, 4, 5).astype(np.float32))
    gamma_2 = Tensor(np.random.randn(1, 4, 5).astype(np.float32))
    eps_2 = 1e-5
    TEST_OP(rms_norm_forward_func, [[x_1, gamma_1, eps_1], [x_2, gamma_2, eps_2]], grad=True, jit_level=jit_level)
