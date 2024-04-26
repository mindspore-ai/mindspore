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
import scipy
from mindspore.scipy.linalg import lstsq

import mindspore as ms
from mindspore import ops, Tensor, jit, JitConfig, context
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP
from tests.st.utils import test_utils


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)


def generate_expect_forward_output(A, B):
    return scipy.linalg.lstsq(A, B, lapack_driver='gelss')


def pinv_backward(grad, pinvA, A):
    m = A.shape[-2]
    n = A.shape[-1]
    pinvAh = pinvA.conj().T
    gradh = grad.conj().T
    if m <= n:
        K = gradh @ pinvA
        KpinvAh = K @ pinvAh
        gA = -(pinvA @ K).conj().T + KpinvAh - (A @ pinvA) @ KpinvAh + \
            (pinvAh @ pinvA) @ (gradh - K @ A)
    else:
        K = pinvA @ gradh
        pinvAhK = pinvAh @ K
        gA = -(K @ pinvA).conj().T + (gradh - A @ K) @ pinvA @ pinvAh + \
            pinvAhK - pinvAhK @ pinvA @ A
    return gA


def generate_expect_backward_output(A, B, dout):
    pinvA = np.linalg.pinv(A)
    pinvA_grad = dout @ B.conj().T
    grad_a = pinv_backward(pinvA_grad, pinvA, A)
    grad_b = pinvA.conj().T @ dout
    return grad_a, grad_b


@test_utils.run_with_cell
def lstsq_forward_func(A, B):
    return lstsq(A, B, driver='gelss')


@test_utils.run_with_cell
def lstsq_backward_func(A, B):
    return ops.grad(lstsq_forward_func, (0, 1))(A, B)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', ['GE', 'KBK', 'pynative'])
def test_ops_lstsq_forward(mode):
    """
    Feature: scipy.linalg.lstsq
    Description: test function lstsq forward.
    Expectation: success
    """
    A = generate_random_input((5, 5), np.float32)
    B = generate_random_input((5, 3), np.float32)
    expect_outs = generate_expect_forward_output(A, B)
    if mode == 'pynative':
        context.set_context(mode=ms.PYNATIVE_MODE)
        outs = lstsq_forward_func(Tensor(A), Tensor(B))
    elif mode == 'KBK':
        outs = (jit(lstsq_forward_func, jit_config=JitConfig(jit_level="O0")))(
            ms.Tensor(A), ms.Tensor(B))
    else:
        outs = (jit(lstsq_forward_func, jit_config=JitConfig(jit_level="O2")))(
            ms.Tensor(A), ms.Tensor(B))
    for i in range(4):
        np.testing.assert_allclose(
            outs[i].asnumpy(), expect_outs[i], rtol=1e-3, atol=1e-5)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', ['GE', 'KBK', 'pynative'])
def test_ops_lstsq_backward(mode):
    """
    Feature: scipy.linalg.lstsq
    Description: test function lstsq backward.
    Expectation: success
    """
    A = generate_random_input((4, 5), np.float32)
    B = generate_random_input((4, 6), np.float32)
    dout = np.ones((5, 6), np.float32)
    expect_grads = generate_expect_backward_output(A, B, dout)
    if mode == 'pynative':
        context.set_context(mode=ms.PYNATIVE_MODE)
        grads = lstsq_backward_func(Tensor(A), Tensor(B))
    elif mode == 'KBK':
        grads = (jit(lstsq_backward_func, jit_config=JitConfig(jit_level="O0")))(
            Tensor(A), Tensor(B))
    else:
        grads = (jit(lstsq_backward_func, jit_config=JitConfig(jit_level="O2")))(
            Tensor(A), Tensor(B))
    for i in range(2):
        np.testing.assert_allclose(
            grads[i].asnumpy(), expect_grads[i], rtol=1e-3, atol=1e-5)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('jit_level', ["O0", "O2"])
def test_ops_lstsq_dynamic(jit_level):
    """
    Feature: scipy.linalg.lstsq
    Description: test function lstsq with dynamic shape/rank.
    Expectation: success
    """
    A1 = generate_random_input((5, 5), np.float32)
    B1 = generate_random_input((5, 3), np.float32)
    A2 = generate_random_input((3, 6), np.float32)
    B2 = generate_random_input((3, 7), np.float32)
    inputs1 = [Tensor(A1), Tensor(B1)]
    inputs2 = [Tensor(A2), Tensor(B2)]

    TEST_OP(lstsq_forward_func, [inputs1, inputs2],
            grad=False, jit_level=jit_level)
    TEST_OP(lstsq_forward_func, [inputs1, inputs2],
            grad=True, jit_level=jit_level)
