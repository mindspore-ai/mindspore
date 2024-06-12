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
import mindspore as ms
from mindspore import ops, jit, JitConfig
from tests.st.utils import test_utils
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)


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


def generate_expect_forward_output(Dout, A, B):
    pinvA = np.linalg.pinv(A)
    pinvA_grad = Dout @ B.conj().T
    grad_a = pinv_backward(pinvA_grad, pinvA, A)
    grad_b = pinvA.conj().T @ Dout
    return grad_a, grad_b


@test_utils.run_with_cell
def lstsqv2_grad_forward_func(dx, a, b):
    return ops.auto_generate.lstsq_v2_grad_op(dx, a, b)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.parametrize('mode', ['GE', 'pynative', 'KBK'])
def test_lstsqv2_grad_like_forward(mode):
    """
    Feature: Ops.LstsqV2Grad
    Description: test Ops.LstsqV2Grad.
    Expectation: expect correct result.
    """
    A = generate_random_input((4, 2), np.float32)
    B = generate_random_input((4, 3), np.float32)
    Dout = generate_random_input((2, 3), np.float32)
    expect_grads = generate_expect_forward_output(Dout, A, B)
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        grads = lstsqv2_grad_forward_func(
            ms.Tensor(Dout), ms.Tensor(A), ms.Tensor(B))
    elif mode == 'KBK':
        grads = (jit(lstsqv2_grad_forward_func, jit_config=JitConfig(jit_level="O0")))(
            ms.Tensor(Dout), ms.Tensor(A), ms.Tensor(B))
    else:
        grads = (jit(lstsqv2_grad_forward_func, jit_config=JitConfig(jit_level="O2")))(
            ms.Tensor(Dout), ms.Tensor(A), ms.Tensor(B))
    for i in range(2):
        np.testing.assert_allclose(
            grads[i].asnumpy(), expect_grads[i], rtol=1e-3)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
def test_lstsqv2_grad_dynamic_shape():
    """
    Feature: Test Ops.LstsqV2Grad with dynamic shape in graph mode.
    Description: call Ops.LstsqV2Grad with valid input and index.
    Expectation: return the correct value.
    """
    A1 = generate_random_input((5, 5), np.float32)
    B1 = generate_random_input((5, 3), np.float32)
    Dout1 = generate_random_input((5, 3), np.float32)

    A2 = generate_random_input((2, 3, 6), np.float32)
    B2 = generate_random_input((2, 3, 7), np.float32)
    Dout2 = generate_random_input((2, 6, 7), np.float32)

    inputs1 = [ms.Tensor(Dout1), ms.Tensor(A1), ms.Tensor(B1)]
    inputs2 = [ms.Tensor(Dout2), ms.Tensor(A2), ms.Tensor(B2)]

    TEST_OP(lstsqv2_grad_forward_func, [
        inputs1, inputs2], 'lstsqv2_grad', disable_grad=True, disable_yaml_check=True)
