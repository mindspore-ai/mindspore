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
from mindspore import mint, jit
from mindspore.ops.auto_generate import BatchMatMul
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.mark_utils import arg_mark



def generate_random_input(shape, shape1, shape2):
    x = np.random.randn(*shape).astype(np.float32)
    batch1 = np.random.randn(*shape1).astype(np.float32)
    batch2 = np.random.randn(*shape2).astype(np.float32)
    return x, batch1, batch2


def batch_matmul_(batch1, batch2):
    return BatchMatMul()(batch1, batch2)


def generate_expect_forward_output(input1, batch1, batch2, beta=1, alpha=1):
    bmm_res = batch_matmul_(ms.Tensor(batch1), ms.Tensor(batch2))
    return beta * ms.Tensor(input1) + alpha * (bmm_res.sum(axis=0))


def generate_expect_backward_output(input1, batch1, batch2, beta=1, alpha=1):
    """Compute expected gradients for addbmm with batch broadcasting.

    y = beta * input1 + alpha * sum_b(batch1_b @ batch2_b), where b iterates over
    the broadcasted batch dimension. dy is assumed to be ones of shape (m, n).

    Returns numpy arrays: (grad_input, grad_batch1, grad_batch2).
    """
    # out_grad shape (m, n)
    out_grad = np.ones((batch1.shape[1], batch2.shape[2]), dtype=np.float32)
    input_grad_full = beta * out_grad
    in_shape = tuple(input1.shape)
    out_shape = input_grad_full.shape
    # align dims by prepending 1s on input side
    if len(in_shape) < len(out_shape):
        in_shape_aligned = (1,) * (len(out_shape) - len(in_shape)) + in_shape
    else:
        in_shape_aligned = in_shape
    # sum over axes where input dim is 1 but output dim > 1
    axes = tuple(i for i, (din, dout) in enumerate(zip(in_shape_aligned, out_shape)) if din == 1 and dout > 1)
    if axes:
        input_grad_full = input_grad_full.sum(axis=axes, keepdims=True)
    # finally reshape to exact input shape
    input_grad = input_grad_full.reshape(in_shape)
    # Initialize grads with zeros to accumulate over broadcasted batches
    b1_grad = np.zeros_like(batch1)
    b2_grad = np.zeros_like(batch2)
    B1 = batch1.shape[0]
    B2 = batch2.shape[0]
    B = max(B1, B2)
    for k in range(B):
        i1 = 0 if B1 == 1 else k
        i2 = 0 if B2 == 1 else k
        # d(batch1[i1]) = alpha * (out_grad @ batch2[i2].T)
        b1_grad[i1] += alpha * (out_grad @ batch2[i2].transpose((1, 0)))
        # d(batch2[i2]) = alpha * (batch1[i1].T @ out_grad)
        b2_grad[i2] += alpha * (batch1[i1].transpose((1, 0)) @ out_grad)
    return input_grad, b1_grad, b2_grad


def addbmm_forward_func(input1, batch1, batch2, beta=1, alpha=1):
    return mint.addbmm(input1, batch1, batch2, beta=beta, alpha=alpha)


def addbmm_backward_func(input1, batch1, batch2, beta=1, alpha=1):
    output_grad, b1_grad, b2_grad = ms.ops.grad(
        addbmm_forward_func, (0, 1, 2))(input1, batch1, batch2, beta, alpha)
    return output_grad, b1_grad, b2_grad


def addbmm_forward_func_tensor(input1, batch1, batch2, beta=1, alpha=1):
    return input1.addbmm(batch1, batch2, beta=beta, alpha=alpha)


def addbmm_backward_func_tensor(input1, batch1, batch2, beta=1, alpha=1):
    output_grad, b1_grad, b2_grad = ms.ops.grad(
        addbmm_forward_func_tensor, (0, 1, 2))(input1, batch1, batch2, beta, alpha)
    return output_grad, b1_grad, b2_grad


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=2.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_addbmm_input_with_first_dim_1(mode):
    """
    Feature: Ops.
    Description: test op addbmm with input whose first dim is 1 (broadcasted batches).
    Expectation: expect correct result.
    """
    input_shape = (15,)
    batch1_shape = (3, 1, 12)
    batch2_shape = (3, 12, 15)
    beta = 1
    alpha = 2.0
    input1, batch1, batch2 = generate_random_input(
        input_shape, batch1_shape, batch2_shape)
    expect_forward = generate_expect_forward_output(input1, batch1, batch2, beta, alpha)
    expect_grad, expect_b1_grad, expect_b2_grad = generate_expect_backward_output(
        input1, batch1, batch2, beta, alpha)
    if mode == 'pynative':
        ms.set_context(mode=ms.PYNATIVE_MODE)
        output_forward = addbmm_forward_func(
            ms.Tensor(input1), ms.Tensor(batch1), ms.Tensor(batch2), beta, alpha)
        input_grad, b1_grad, b2_grad = addbmm_backward_func(
            ms.Tensor(input1), ms.Tensor(batch1), ms.Tensor(batch2), beta, alpha)
    else:
        output_forward = (jit(addbmm_forward_func, backend="ms_backend", jit_level="O0"))(
            ms.Tensor(input1), ms.Tensor(batch1), ms.Tensor(batch2), beta, alpha)
        input_grad, b1_grad, b2_grad = (jit(addbmm_backward_func, backend="ms_backend", jit_level="O0"))(
            ms.Tensor(input1), ms.Tensor(batch1), ms.Tensor(batch2), beta, alpha)
    np.testing.assert_allclose(
        output_forward.asnumpy(), expect_forward.asnumpy(), 3e-3, 3e-3)
    np.testing.assert_allclose(input_grad.asnumpy(), expect_grad, 3e-3, 3e-3)
    np.testing.assert_allclose(b1_grad.asnumpy(), expect_b1_grad, 3e-3, 3e-3)
    np.testing.assert_allclose(b2_grad.asnumpy(), expect_b2_grad, 3e-3, 3e-3)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=2.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_addbmm_tensor(mode):
    """
    Feature: Ops.
    Description: test op addbmm tensor.
    Expectation: expect correct result.
    """
    input_shape1 = (4, 5)
    input_shape2 = (4, 7)
    batch1_shape = (3, 4, 2)
    batch2_shape = (3, 2, 5)
    batch3_shape = (6, 4, 3)
    batch4_shape = (6, 3, 7)
    beta = 1
    alpha = 2.0
    input1, batch1, batch2 = generate_random_input(
        input_shape1, batch1_shape, batch2_shape)
    input2, batch3, batch4 = generate_random_input(
        input_shape2, batch3_shape, batch4_shape)
    expect_forward_output1 = generate_expect_forward_output(
        input1, batch1, batch2)
    expect_forward_output2 = generate_expect_forward_output(
        input2, batch3, batch4, beta, alpha)
    if mode == 'pynative':
        ms.set_context(mode=ms.PYNATIVE_MODE)
        output_forward1 = addbmm_forward_func_tensor(
            ms.Tensor(input1), ms.Tensor(batch1), ms.Tensor(batch2))
        output_forward2 = addbmm_forward_func_tensor(
            ms.Tensor(input2), ms.Tensor(batch3), ms.Tensor(batch4), beta, alpha)
        output_grad1, b1_grad1, b2_grad1 = addbmm_backward_func_tensor(
            ms.Tensor(input1), ms.Tensor(batch1), ms.Tensor(batch2))
        output_grad2, b1_grad2, b2_grad2 = addbmm_backward_func_tensor(
            ms.Tensor(input2), ms.Tensor(batch3), ms.Tensor(batch4), beta, alpha)
    else:
        output_forward1 = (jit(addbmm_forward_func_tensor, backend="ms_backend", jit_level="O0"))(
            ms.Tensor(input1), ms.Tensor(batch1), ms.Tensor(batch2))
        output_forward2 = (jit(addbmm_forward_func_tensor, backend="ms_backend", jit_level="O0"))(
            ms.Tensor(input2), ms.Tensor(batch3), ms.Tensor(batch4), beta, alpha)
        output_grad1, b1_grad1, b2_grad1 = (jit(addbmm_backward_func_tensor, backend="ms_backend", jit_level="O0"))(
            ms.Tensor(input1), ms.Tensor(batch1), ms.Tensor(batch2))
        output_grad2, b1_grad2, b2_grad2 = (jit(addbmm_backward_func_tensor, backend="ms_backend", jit_level="O0"))(
            ms.Tensor(input2), ms.Tensor(batch3), ms.Tensor(batch4), beta, alpha)
    np.testing.assert_allclose(output_forward1.asnumpy(), expect_forward_output1.asnumpy(), 3e-3, 3e-3)
    np.testing.assert_allclose(output_forward2.asnumpy(), expect_forward_output2.asnumpy(), 3e-3, 3e-3)
    expect_grad1, expect_b1_grad1, expect_b2_grad1 = generate_expect_backward_output(
        input1, batch1, batch2)
    expect_grad2, expect_b1_grad2, expect_b2_grad2 = generate_expect_backward_output(
        input2, batch3, batch4, beta, alpha)
    np.testing.assert_allclose(output_grad1.asnumpy(), expect_grad1, 3e-3, 3e-3)
    np.testing.assert_allclose(b1_grad1.asnumpy(), expect_b1_grad1, 3e-3, 3e-3)
    np.testing.assert_allclose(b2_grad1.asnumpy(), expect_b2_grad1, 3e-3, 3e-3)
    np.testing.assert_allclose(output_grad2.asnumpy(), expect_grad2, 3e-3, 3e-3)
    np.testing.assert_allclose(b1_grad2.asnumpy(), expect_b1_grad2, 3e-3, 3e-3)
    np.testing.assert_allclose(b2_grad2.asnumpy(), expect_b2_grad2, 3e-3, 3e-3)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=2.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_addbmm_normal(mode):
    """
    Feature: Ops.
    Description: test op addbmm.
    Expectation: expect correct result.
    """
    input_shape1 = (4, 5)
    input_shape2 = (4, 7)
    batch1_shape = (3, 4, 2)
    batch2_shape = (3, 2, 5)
    batch3_shape = (6, 4, 3)
    batch4_shape = (6, 3, 7)
    beta = 1
    alpha = 2.0
    input1, batch1, batch2 = generate_random_input(
        input_shape1, batch1_shape, batch2_shape)
    input2, batch3, batch4 = generate_random_input(
        input_shape2, batch3_shape, batch4_shape)
    expect_forward_output1 = generate_expect_forward_output(
        input1, batch1, batch2)
    expect_forward_output2 = generate_expect_forward_output(
        input2, batch3, batch4, beta, alpha)
    if mode == 'pynative':
        ms.set_context(mode=ms.PYNATIVE_MODE)
        output_forward1 = addbmm_forward_func(
            ms.Tensor(input1), ms.Tensor(batch1), ms.Tensor(batch2))
        output_forward2 = addbmm_forward_func(
            ms.Tensor(input2), ms.Tensor(batch3), ms.Tensor(batch4), beta, alpha)
        output_grad1, b1_grad1, b2_grad1 = addbmm_backward_func(
            ms.Tensor(input1), ms.Tensor(batch1), ms.Tensor(batch2))
        output_grad2, b1_grad2, b2_grad2 = addbmm_backward_func(
            ms.Tensor(input2), ms.Tensor(batch3), ms.Tensor(batch4), beta, alpha)
    else:
        output_forward1 = (jit(addbmm_forward_func, backend="ms_backend", jit_level="O0"))(
            ms.Tensor(input1), ms.Tensor(batch1), ms.Tensor(batch2))
        output_forward2 = (jit(addbmm_forward_func, backend="ms_backend", jit_level="O0"))(
            ms.Tensor(input2), ms.Tensor(batch3), ms.Tensor(batch4), beta, alpha)
        output_grad1, b1_grad1, b2_grad1 = (jit(addbmm_backward_func, backend="ms_backend", jit_level="O0"))(
            ms.Tensor(input1), ms.Tensor(batch1), ms.Tensor(batch2))
        output_grad2, b1_grad2, b2_grad2 = (jit(addbmm_backward_func, backend="ms_backend", jit_level="O0"))(
            ms.Tensor(input2), ms.Tensor(batch3), ms.Tensor(batch4), beta, alpha)
    np.testing.assert_allclose(output_forward1.asnumpy(), expect_forward_output1.asnumpy(), 3e-3, 3e-3)
    np.testing.assert_allclose(output_forward2.asnumpy(), expect_forward_output2.asnumpy(), 3e-3, 3e-3)
    expect_grad1, expect_b1_grad1, expect_b2_grad1 = generate_expect_backward_output(
        input1, batch1, batch2)
    expect_grad2, expect_b1_grad2, expect_b2_grad2 = generate_expect_backward_output(
        input2, batch3, batch4, beta, alpha)
    np.testing.assert_allclose(output_grad1.asnumpy(), expect_grad1, 3e-3, 3e-3)
    np.testing.assert_allclose(b1_grad1.asnumpy(), expect_b1_grad1, 3e-3, 3e-3)
    np.testing.assert_allclose(b2_grad1.asnumpy(), expect_b2_grad1, 3e-3, 3e-3)
    np.testing.assert_allclose(output_grad2.asnumpy(), expect_grad2, 3e-3, 3e-3)
    np.testing.assert_allclose(b1_grad2.asnumpy(), expect_b1_grad2, 3e-3, 3e-3)
    np.testing.assert_allclose(b2_grad2.asnumpy(), expect_b2_grad2, 3e-3, 3e-3)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=2048.0)
def test_addbmm_dynamic_shape():
    """
    Feature: Test dynamic shape.
    Description: test function div dynamic feature.
    Expectation: expect correct result.
    """
    input_shape1 = (4, 5)
    batch1_shape = (3, 4, 2)
    batch2_shape = (3, 2, 5)
    beta = 1.0
    alpha = 0.5
    input_shape2 = (4, 4)
    batch1_shape2 = (5, 4, 2)
    batch2_shape2 = (5, 2, 4)
    beta2 = 1.0
    alpha2 = 2.0
    input1, batch1, batch2 = generate_random_input(
        input_shape1, batch1_shape, batch2_shape)
    input2, batch1_2, batch2_2 = generate_random_input(
        input_shape2, batch1_shape2, batch2_shape2)
    TEST_OP(addbmm_forward_func, [[ms.Tensor(input1), ms.Tensor(batch1), ms.Tensor(batch2), beta, alpha],
                                  [ms.Tensor(input2), ms.Tensor(batch1_2), ms.Tensor(batch2_2), beta2, alpha2]],
            disable_mode=['GRAPH_MODE_GE'],
            disable_case=['EmptyTensor', 'ScalarTensor'],
            case_config={'disable_input_check': True})
