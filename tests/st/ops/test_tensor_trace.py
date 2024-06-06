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


@test_utils.run_with_cell
def trace_forward_func(a, offset=0, axis1=0, axis2=1):
    return a.trace(offset, axis1, axis2, dtype=None)


@test_utils.run_with_cell
def trace_backward_func(a, offset=0, axis1=0, axis2=1):
    return ops.grad(trace_forward_func, (0))(a, offset, axis1, axis2)


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)


def generate_expect_forward_output(a, offset=0, axis1=0, axis2=1, dtype=None):
    return np.trace(a, offset, axis1, axis2, dtype)


def generate_expect_backward_output(a, offset=0, axis1=0, axis2=1, dout=None):
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


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', ['GE', 'KBK', 'pynative'])
def test_ops_trace_forward(mode):
    """
    Feature: numpy.trace
    Description: test function trace forward.
    Expectation: success
    """
    a_np = generate_random_input((5, 5, 5), np.float32)
    if mode == 'pynative':
        output = trace_forward_func(ms.Tensor(a_np))
    elif mode == 'KBK':
        output = (jit(trace_forward_func, jit_config=JitConfig(jit_level="O0")))(
            ms.Tensor(a_np))
    else:
        output = (jit(trace_forward_func, jit_config=JitConfig(jit_level="O2")))(
            ms.Tensor(a_np))
    expect = generate_expect_forward_output(a_np)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3, atol=1e-5)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', ['GE', 'KBK', 'pynative'])
def test_ops_trace_backward(mode):
    """
    Feature: numpy.trace
    Description: test function trace backward.
    Expectation: success
    """
    in_np = generate_random_input((5, 5, 5), np.float32)
    dout = np.ones((5,), np.float32)
    if mode == 'pynative':
        grad = trace_backward_func(ms.Tensor(in_np))
    elif mode == 'KBK':
        grad = (jit(trace_backward_func, jit_config=JitConfig(jit_level="O0")))(
            ms.Tensor(in_np))
    else:
        grad = (jit(trace_backward_func, jit_config=JitConfig(jit_level="O2")))(
            ms.Tensor(in_np))
    expect = generate_expect_backward_output(
        in_np, offset=0, axis1=0, axis2=1, dout=dout)
    np.testing.assert_allclose(
        grad.asnumpy(), expect, rtol=1e-3, atol=1e-5)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_ops_trace_dynamic():
    """
    Feature: numpy.trace
    Description: test function trace forward with dynamic input.
    Expectation: success
    """
    A1 = generate_random_input((6, 5, 5), np.float32)
    offset1 = 0
    axis1 = 0
    axis2 = 1

    A2 = generate_random_input((2, 3, 6, 8), np.float32)
    offset2 = 1
    axis3 = -1
    axis4 = 0

    inputs1 = [ms.Tensor(A1), offset1, axis1, axis2]
    inputs2 = [ms.Tensor(A2), offset2, axis3, axis4]

    TEST_OP(trace_forward_func, [inputs1, inputs2],
            'trace_v2', disable_yaml_check=True)
