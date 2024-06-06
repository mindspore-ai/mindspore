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


def generate_expect_forward_output(a, offset=0, axis1=0, axis2=1, dout=None):
    da = np.zeros_like(a)
    perm = []
    for i in range(a.ndim):
        if (i != axis1 and i != axis2):
            perm.append(i)
    perm.append(axis1)
    perm.append(axis2)
    da = da.transpose(perm)
    min_size = min(da.shape[-1], da.shape[-2])
    row_st = 0
    col_st = 0
    diag_count = 0
    if offset < 0:
        row_st = -offset
        col_st = 0
        diag_count = min_size + offset
    else:
        row_st = 0
        col_st = offset
        diag_count = min_size - offset
    for i in range(diag_count):
        da[..., row_st + i, col_st + i] = dout
    rev_perm = np.arange(a.ndim)
    for i in range(a.ndim):
        rev_perm[perm[i]] = i
    da = da.transpose(rev_perm)
    return da


@test_utils.run_with_cell
def tracev2_grad_forward_func(dout, shape, offset=0, axis1=0, axis2=1):
    return ops.auto_generate.tracev2_grad_op(dout, shape, offset, axis1, axis2)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.parametrize('mode', ['GE', 'pynative', 'KBK'])
def test_tracev2_grad_like_forward(mode):
    """
    Feature: Ops.TraceV2Grad
    Description: test Ops.TraceV2Grad.
    Expectation: expect correct result.
    """
    A = generate_random_input((5, 4, 2), np.float32)
    Dout = generate_random_input((2,), np.float32)
    expect_grads = generate_expect_forward_output(A, dout=Dout)
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        grads = tracev2_grad_forward_func(
            ms.Tensor(Dout), ms.Tensor(A.shape))
    elif mode == 'KBK':
        grads = (jit(tracev2_grad_forward_func, jit_config=JitConfig(jit_level="O0")))(
            ms.Tensor(Dout), ms.Tensor(A.shape))
    else:
        grads = (jit(tracev2_grad_forward_func, jit_config=JitConfig(jit_level="O2")))(
            ms.Tensor(Dout), ms.Tensor(A.shape))
    for i in range(2):
        np.testing.assert_allclose(
            grads[i].asnumpy(), expect_grads[i], rtol=1e-3)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
def test_tracev2_grad_dynamic_shape():
    """
    Feature: Test Ops.TraceV2Grad with dynamic shape in graph mode.
    Description: call Ops.TraceV2Grad with valid input and index.
    Expectation: return the correct value.
    """
    A1 = generate_random_input((6, 5, 5), np.float32)
    Dout1 = generate_random_input((5,), np.float32)
    offset1 = -1
    axis11 = 0
    axis21 = 1

    A2 = generate_random_input((2, 3, 6, 8), np.float32)
    Dout2 = generate_random_input((3, 8), np.float32)
    offset2 = 0
    axis12 = 2
    axis22 = 0

    inputs1 = [ms.Tensor(Dout1), ms.Tensor(A1.shape), offset1, axis11, axis21]
    inputs2 = [ms.Tensor(Dout2), ms.Tensor(A2.shape), offset2, axis12, axis22]

    TEST_OP(tracev2_grad_forward_func, [
        inputs1, inputs2], 'tracev2_grad', disable_grad=True, disable_input_check=True)
