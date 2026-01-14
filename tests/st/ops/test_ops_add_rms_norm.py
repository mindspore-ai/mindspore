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
from mindspore import ops, Tensor
from tests.mark_utils import arg_mark
from tests.st.ops.test_tools.test_op import TEST_OP

def add_rms_norm_forward_func(x1, x2, gamma, epsilon):
    y, rstd, x_sum = ops.add_rms_norm(x1, x2, gamma, epsilon)
    return y, rstd, x_sum

def add_rms_norm_backward_func(x1, x2, gamma, epsilon):
    # pylint: disable=not-callable
    x1_grad, x2_grad, gamma_grad = ops.grad(add_rms_norm_forward_func, (0, 1, 2))(x1, x2, gamma, epsilon)
    return x1_grad, x2_grad, gamma_grad


@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_add_rms_norm_forward_backward(mode):
    """
    Feature: Ops.
    Description: test AddRmsNorm. Not support GE yet.
    Expectation: expect correct result.
    """
    np.random.seed(0)
    x1 = Tensor(np.random.rand(2, 2, 2, 2).astype(np.float32))
    x2 = Tensor(np.random.rand(2, 2, 2, 2).astype(np.float32))
    gamma = Tensor(np.random.rand(2, 2, 2).astype(np.float32))
    eps = 1e-6
    expect_y = np.array([[[[0.19422545, 0.65835315],
                           [0.019415665, 0.65391016]],
                          [[0.6422639, 0.6670892],
                           [0.63490605, 0.8531909]]],
                         [[[0.46267724, 0.5453529],
                           [0.016471693, 0.8532362]],
                          [[0.62541926, 0.7751672],
                           [0.2969172, 0.5505848]]]])
    expect_rstd = np.array([[[[0.7482755]]],
                            [[[0.93749315]]]])
    expect_x = np.array([[[[0.5690319, 1.5478091],
                           [1.3809202, 1.4148953]],
                          [[1.4022732, 1.4450526],
                           [0.89906657, 1.6723022]]],
                         [[[1.0819372, 1.0233625],
                           [0.9350783, 1.4735638]],
                          [[1.0898929, 1.3402586],
                           [0.33559167, 0.861363]]]])
    expect_dx1 = np.array([[[[0.169143, -0.043005686],
                             [-0.40379205, 0.034028947]],
                            [[0.033702962, 0.024378741],
                             [0.4341354, 0.0041682026]]],
                           [[[-0.06277209, 0.06904319],
                             [-0.40622783, -0.08889346]],
                            [[0.07981955, -0.029127851],
                             [0.7326436, 0.24877167]]]])
    expect_dx2 = expect_dx1
    expect_dgamma = np.array([[[1.4401014, 2.117583],
                               [1.9099383, 2.4401875]],
                              [[2.071054, 2.3377807],
                               [0.9873644, 2.0588646]]])

    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        y, rstd, x = add_rms_norm_forward_func(x1, x2, gamma, eps)
        dx1, dx2, dgamma = add_rms_norm_backward_func(x1, x2, gamma, eps)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE)
        y, rstd, x = add_rms_norm_forward_func(x1, x2, gamma, eps)
        dx1, dx2, dgamma = add_rms_norm_backward_func(x1, x2, gamma, eps)
    loss = 1e-4
    np.testing.assert_allclose(y.asnumpy(), expect_y, rtol=loss)
    np.testing.assert_allclose(rstd.asnumpy(), expect_rstd, rtol=loss)
    np.testing.assert_allclose(x.asnumpy(), expect_x, rtol=loss)
    np.testing.assert_allclose(dx1.asnumpy(), expect_dx1, rtol=loss)
    np.testing.assert_allclose(dx2.asnumpy(), expect_dx2, rtol=loss)
    np.testing.assert_allclose(dgamma.asnumpy(), expect_dgamma, rtol=loss)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=20.0)
def test_rms_norm_dynamic_shape():
    """
    Feature: Test gather with dynamic shape in graph mode.
    Description: call ops.extend.gather with valid input and index.
    Expectation: return the correct value.
    """
    x1_1 = Tensor(np.random.randn(3, 4, 64).astype(np.float32))
    x1_2 = Tensor(np.random.randn(3, 4, 64).astype(np.float32))
    gamma_1 = Tensor(np.random.randn(4, 64).astype(np.float32))
    eps_1 = 1e-4

    x2_1 = Tensor(np.random.randn(2, 3, 4, 64).astype(np.float32))
    x2_2 = Tensor(np.random.randn(2, 3, 4, 64).astype(np.float32))
    gamma_2 = Tensor(np.random.randn(3, 4, 64).astype(np.float32))
    eps_2 = 1e-5

    # TEST_OP Todo 3rd grad is not support Deterministic case.
    TEST_OP(add_rms_norm_forward_func, [[x1_1, x1_2, gamma_1, eps_1], [x2_1, x2_2, gamma_2, eps_2]],
            disable_mode=["GRAPH_MODE_GE"],
            disable_case=['EmptyTensor',
                          'ScalarTensor',
                          'Deterministic'])
