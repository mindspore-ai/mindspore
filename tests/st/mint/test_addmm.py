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

"""Tests for mint.addmm forward/backward, including column-major transpose views."""

import pytest
import torch
import numpy as np
import mindspore as ms
from mindspore import mint, jit
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.mark_utils import arg_mark


def generate_random_input(shape, shape1, shape2):
    x = np.random.randn(*shape).astype(np.float32)
    batch1 = np.random.randn(*shape1).astype(np.float32)
    batch2 = np.random.randn(*shape2).astype(np.float32)
    return x, batch1, batch2


def generate_expect_forward_output(input1, batch1, batch2, beta=1, alpha=1):
    input1 = torch.tensor(input1)
    batch1 = torch.tensor(batch1)
    batch2 = torch.tensor(batch2)
    return torch.addmm(input1, batch1, batch2, beta=beta, alpha=alpha).numpy()


def generate_expect_backward_output(input1, batch1, batch2, beta=1, alpha=1):
    # Use PyTorch autograd as the benchmark for backward results
    t_input = torch.tensor(input1, dtype=torch.float32, requires_grad=True)
    t_b1 = torch.tensor(batch1, dtype=torch.float32, requires_grad=True)
    t_b2 = torch.tensor(batch2, dtype=torch.float32, requires_grad=True)
    out = torch.addmm(t_input, t_b1, t_b2, beta=beta, alpha=alpha)
    out.backward(torch.ones_like(out))
    return t_input.grad.detach().numpy(), t_b1.grad.detach().numpy(), t_b2.grad.detach().numpy()


def addmm_forward_func(input1, batch1, batch2, beta=1, alpha=1):
    return mint.addmm(input1, batch1, batch2, beta=beta, alpha=alpha)


def addmm_backward_func(input1, batch1, batch2, beta=1, alpha=1):
    output_grad, b1_grad, b2_grad = ms.ops.grad(
        addmm_forward_func, (0, 1, 2))(input1, batch1, batch2, beta, alpha)
    return output_grad, b1_grad, b2_grad


def addmm_forward_func_tensor(input1, batch1, batch2, beta=1, alpha=1):
    return input1.addmm(batch1, batch2, beta=beta, alpha=alpha)


def addmm_backward_func_tensor(input1, batch1, batch2, beta=1, alpha=1):
    output_grad, b1_grad, b2_grad = ms.ops.grad(
        addmm_forward_func_tensor, (0, 1, 2))(input1, batch1, batch2, beta, alpha)
    return output_grad, b1_grad, b2_grad


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_addmm_tensor(mode):
    """
    Feature: Ops.
    Description: test op addmm tensor.
    Expectation: expect correct result.
    """
    input_shape1 = (5, 6)
    input_shape2 = (7, 8)
    batch1_shape = (5, 7)
    batch2_shape = (7, 6)
    batch3_shape = (6, 8)
    beta = 1
    alpha = 2.0
    input1, batch1, batch2 = generate_random_input(
        input_shape1, batch1_shape, batch2_shape)
    input2, batch3, batch4 = generate_random_input(
        input_shape2, batch2_shape, batch3_shape)
    expect_forward = generate_expect_forward_output(input1, batch1, batch2)
    expect_forward2 = generate_expect_forward_output(
        input2, batch3, batch4, beta, alpha)
    expect_grad, expect_b1_grad, expect_b2_grad = generate_expect_backward_output(
        input1, batch1, batch2)
    expect_grad2, expect_b1_grad2, expect_b2_grad2 = generate_expect_backward_output(input2, batch3,
                                                                                     batch4, beta, alpha)
    if mode == 'pynative':
        ms.set_context(mode=ms.PYNATIVE_MODE)
        output_forward = addmm_forward_func_tensor(
            ms.Tensor(input1), ms.Tensor(batch1), ms.Tensor(batch2))
        output_forward2 = addmm_forward_func_tensor(
            ms.Tensor(input2), ms.Tensor(batch3), ms.Tensor(batch4), beta, alpha)
        output_grad, b1_grad, b2_grad = addmm_backward_func_tensor(
            ms.Tensor(input1), ms.Tensor(batch1), ms.Tensor(batch2))
        output_grad2, b1_grad2, b2_grad2 = addmm_backward_func_tensor(
            ms.Tensor(input2), ms.Tensor(batch3), ms.Tensor(batch4), beta, alpha)
    else:
        output_forward = (jit(addmm_forward_func_tensor, jit_level="O0"))(
            ms.Tensor(input1), ms.Tensor(batch1), ms.Tensor(batch2))
        output_forward2 = (jit(addmm_forward_func_tensor, jit_level="O0"))(
            ms.Tensor(input2), ms.Tensor(batch3), ms.Tensor(batch4), beta, alpha)
        output_grad, b1_grad, b2_grad = (jit(addmm_backward_func_tensor, jit_level="O0"))(
            ms.Tensor(input1), ms.Tensor(batch1), ms.Tensor(batch2))
        output_grad2, b1_grad2, b2_grad2 = (jit(addmm_backward_func_tensor, jit_level="O0"))(
            ms.Tensor(input2), ms.Tensor(batch3), ms.Tensor(batch4), beta, alpha)
    np.testing.assert_allclose(
        output_forward.asnumpy(), expect_forward, 1e-4, 1e-4)
    np.testing.assert_allclose(output_grad.asnumpy(), expect_grad, 1e-4, 1e-4)
    np.testing.assert_allclose(b1_grad.asnumpy(), expect_b1_grad, 1e-4, 1e-4)
    np.testing.assert_allclose(b2_grad.asnumpy(), expect_b2_grad, 1e-4, 1e-4)
    np.testing.assert_allclose(
        output_forward2.asnumpy(), expect_forward2, 1e-4, 1e-4)
    np.testing.assert_allclose(
        output_grad2.asnumpy(), expect_grad2, 1e-4, 1e-4)
    np.testing.assert_allclose(b1_grad2.asnumpy(), expect_b1_grad2, 1e-4, 1e-4)
    np.testing.assert_allclose(b2_grad2.asnumpy(), expect_b2_grad2, 1e-4, 1e-4)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_addmm_normal(mode):
    """
    Feature: Ops.
    Description: test op addmm.
    Expectation: expect correct result.
    """
    input_shape1 = (5, 6)
    input_shape2 = (7, 8)
    batch1_shape = (5, 7)
    batch2_shape = (7, 6)
    batch3_shape = (6, 8)
    beta = 1
    alpha = 2.0
    input1, batch1, batch2 = generate_random_input(
        input_shape1, batch1_shape, batch2_shape)
    input2, batch3, batch4 = generate_random_input(
        input_shape2, batch2_shape, batch3_shape)
    expect_forward = generate_expect_forward_output(input1, batch1, batch2)
    expect_forward2 = generate_expect_forward_output(
        input2, batch3, batch4, beta, alpha)
    expect_grad, expect_b1_grad, expect_b2_grad = generate_expect_backward_output(
        input1, batch1, batch2)
    expect_grad2, expect_b1_grad2, expect_b2_grad2 = generate_expect_backward_output(input2, batch3,
                                                                                     batch4, beta, alpha)
    if mode == 'pynative':
        ms.set_context(mode=ms.PYNATIVE_MODE)
        output_forward = addmm_forward_func(
            ms.Tensor(input1), ms.Tensor(batch1), ms.Tensor(batch2))
        output_forward2 = addmm_forward_func(
            ms.Tensor(input2), ms.Tensor(batch3), ms.Tensor(batch4), beta, alpha)
        output_grad, b1_grad, b2_grad = addmm_backward_func(
            ms.Tensor(input1), ms.Tensor(batch1), ms.Tensor(batch2))
        output_grad2, b1_grad2, b2_grad2 = addmm_backward_func(
            ms.Tensor(input2), ms.Tensor(batch3), ms.Tensor(batch4), beta, alpha)
    else:
        output_forward = (jit(addmm_forward_func, backend="ms_backend", jit_level="O0"))(
            ms.Tensor(input1), ms.Tensor(batch1), ms.Tensor(batch2))
        output_forward2 = (jit(addmm_forward_func, backend="ms_backend", jit_level="O0"))(
            ms.Tensor(input2), ms.Tensor(batch3), ms.Tensor(batch4), beta, alpha)
        output_grad, b1_grad, b2_grad = (jit(addmm_backward_func, backend="ms_backend", jit_level="O0"))(
            ms.Tensor(input1), ms.Tensor(batch1), ms.Tensor(batch2))
        output_grad2, b1_grad2, b2_grad2 = (jit(addmm_backward_func, backend="ms_backend", jit_level="O0"))(
            ms.Tensor(input2), ms.Tensor(batch3), ms.Tensor(batch4), beta, alpha)
    np.testing.assert_allclose(
        output_forward.asnumpy(), expect_forward, 1e-4, 1e-4)
    np.testing.assert_allclose(output_grad.asnumpy(), expect_grad, 1e-4, 1e-4)
    np.testing.assert_allclose(b1_grad.asnumpy(), expect_b1_grad, 1e-4, 1e-4)
    np.testing.assert_allclose(b2_grad.asnumpy(), expect_b2_grad, 1e-4, 1e-4)
    np.testing.assert_allclose(
        output_forward2.asnumpy(), expect_forward2, 1e-4, 1e-4)
    np.testing.assert_allclose(
        output_grad2.asnumpy(), expect_grad2, 1e-4, 1e-4)
    np.testing.assert_allclose(b1_grad2.asnumpy(), expect_b1_grad2, 1e-4, 1e-4)
    np.testing.assert_allclose(b2_grad2.asnumpy(), expect_b2_grad2, 1e-4, 1e-4)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_addmm_column_major_views(mode):
    """
    Feature: Ops.
    Description: test addmm with column-major like views (created by transpose) for mat1 and mat2.
    Expectation: expect correct forward and backward results.
    """
    M, K, N = 5, 7, 6
    input_shape = (M, N)
    mat1_base_shape = (K, M)
    mat2_base_shape = (N, K)

    beta = 1.0
    alpha = 2.0

    # numpy inputs
    input_np = np.random.randn(*input_shape).astype(np.float32)
    mat1_base_np = np.random.randn(*mat1_base_shape).astype(np.float32)
    mat2_base_np = np.random.randn(*mat2_base_shape).astype(np.float32)

    # column-major-like views by transpose
    mat1_cm_np = mat1_base_np.transpose((1, 0))  # shape [M, K]
    mat2_cm_np = mat2_base_np.transpose((1, 0))  # shape [K, N]

    # expected with actual numeric values
    expect_forward = generate_expect_forward_output(input_np, mat1_cm_np, mat2_cm_np, beta, alpha)
    expect_grad, expect_mat1_grad, expect_mat2_grad = generate_expect_backward_output(
        input_np, mat1_cm_np, mat2_cm_np, beta, alpha
    )

    input_ms = ms.Tensor(input_np)
    # create views via mint.transpose to simulate column-major stride
    mat1_ms = mint.transpose(ms.Tensor(mat1_base_np), 0, 1)  # [M, K]
    mat2_ms = mint.transpose(ms.Tensor(mat2_base_np), 0, 1)  # [K, N]

    if mode == 'pynative':
        ms.set_context(mode=ms.PYNATIVE_MODE)
        out = addmm_forward_func(input_ms, mat1_ms, mat2_ms, beta, alpha)
        in_grad, m1_grad, m2_grad = addmm_backward_func(input_ms, mat1_ms, mat2_ms, beta, alpha)
    else:
        out = (jit(addmm_forward_func, backend="ms_backend", jit_level="O0"))(input_ms, mat1_ms, mat2_ms, beta, alpha)
        in_grad, m1_grad, m2_grad = (jit(addmm_backward_func, backend="ms_backend", jit_level="O0"))(
            input_ms, mat1_ms, mat2_ms, beta, alpha
        )

    np.testing.assert_allclose(out.asnumpy(), expect_forward, 1e-4, 1e-4)
    np.testing.assert_allclose(in_grad.asnumpy(), expect_grad, 1e-4, 1e-4)
    np.testing.assert_allclose(m1_grad.asnumpy(), expect_mat1_grad, 1e-4, 1e-4)
    np.testing.assert_allclose(m2_grad.asnumpy(), expect_mat2_grad, 1e-4, 1e-4)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_addmm_input_with_first_dim_1(mode):
    """
    Feature: Ops.
    Description: test op addmm with input whose first dim is 1.
    Expectation: expect correct result.
    """
    input_shape = (15,)
    mat1_shape = (1, 12)
    mat2_shape = (12, 15)
    beta = 1
    alpha = 2.0
    input1, mat1, mat2 = generate_random_input(
        input_shape, mat1_shape, mat2_shape)
    expect_forward = generate_expect_forward_output(input1, mat1, mat2, beta, alpha)
    expect_grad, expect_mat1_grad, expect_mat2_grad = generate_expect_backward_output(
        input1, mat1, mat2, beta, alpha)
    if mode == 'pynative':
        ms.set_context(mode=ms.PYNATIVE_MODE)
        output_forward = addmm_forward_func(
            ms.Tensor(input1), ms.Tensor(mat1), ms.Tensor(mat2), beta, alpha)
        input_grad, mat1_grad, mat2_grad = addmm_backward_func(
            ms.Tensor(input1), ms.Tensor(mat1), ms.Tensor(mat2), beta, alpha)
    else:
        output_forward = (jit(addmm_forward_func, backend="ms_backend", jit_level="O0"))(
            ms.Tensor(input1), ms.Tensor(mat1), ms.Tensor(mat2), beta, alpha)
        input_grad, mat1_grad, mat2_grad = (jit(addmm_backward_func, backend="ms_backend", jit_level="O0"))(
            ms.Tensor(input1), ms.Tensor(mat1), ms.Tensor(mat2), beta, alpha)
    np.testing.assert_allclose(
        output_forward.asnumpy(), expect_forward, 1e-4, 1e-4)
    np.testing.assert_allclose(input_grad.asnumpy(), expect_grad, 1e-4, 1e-4)
    np.testing.assert_allclose(mat1_grad.asnumpy(), expect_mat1_grad, 1e-4, 1e-4)
    np.testing.assert_allclose(mat2_grad.asnumpy(), expect_mat2_grad, 1e-4, 1e-4)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
        card_mark='onecard', essential_mark='unessential', mem_peak=0.01)
def test_addmm_dynamic_shape():
    """
    Feature: Test dynamic shape.
    Description: test function div dynamic feature.
    Expectation: expect correct result.
    """
    input_shape1 = (3, 4)
    batch1_shape = (3, 2)
    batch2_shape = (2, 4)
    beta = 1.0
    alpha = 0.5
    input_shape2 = (5, 5)
    batch1_shape2 = (5, 4)
    batch2_shape2 = (4, 5)
    beta2 = 1.0
    alpha2 = 2.0
    input1, batch1, batch2 = generate_random_input(
        input_shape1, batch1_shape, batch2_shape)
    input2, batch1_2, batch2_2 = generate_random_input(
        input_shape2, batch1_shape2, batch2_shape2)
    TEST_OP(addmm_forward_func, [[ms.Tensor(input1), ms.Tensor(batch1), ms.Tensor(batch2), beta, alpha],
                                 [ms.Tensor(input2), ms.Tensor(batch1_2), ms.Tensor(batch2_2), beta2, alpha2]],
            disable_mode=['GRAPH_MODE_GE'],
            disable_case=['EmptyTensor', 'ScalarTensor'],
            case_config={'disable_input_check': True})
