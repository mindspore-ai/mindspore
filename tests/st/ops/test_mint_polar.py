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
"""Tests for mint.polar forward/backward."""

import pytest
import torch
import numpy as np
import mindspore as ms
from mindspore import mint, jit
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.mark_utils import arg_mark


def generate_random_input(shape):
    return np.random.randn(*shape).astype(np.float32), np.random.randn(*shape).astype(np.float32)


def generate_expect_forward_output(abs_np, angle_np):
    abs_torch = torch.tensor(abs_np)
    angle_torch = torch.tensor(angle_np)
    return torch.polar(abs_torch, angle_torch).numpy()


def generate_expect_backward_output(abs_np, angle_np):
    abs_torch = torch.tensor(abs_np, dtype=torch.float32, requires_grad=True)
    angle_torch = torch.tensor(angle_np, dtype=torch.float32, requires_grad=True)
    out = torch.polar(abs_torch, angle_torch)
    out.backward(torch.ones_like(out))
    return abs_torch.grad.detach().numpy(), angle_torch.grad.detach().numpy()


def polar_forward_func(input_abs, input_angle):
    return mint.polar(input_abs, input_angle)


def polar_backward_func(input_abs, input_angle):
    input_grad = ms.ops.grad(polar_forward_func, (0, 1))(input_abs, input_angle)
    return input_grad


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_polar_normal(mode):
    """
    Feature: Ops.
    Description: test polar forward and backward.
    Expectation: expect correct result.
    """
    abs_np, angle_np = generate_random_input((2, 3))
    expect_forward = generate_expect_forward_output(abs_np, angle_np)
    expect_grad_abs, expect_grad_angle = generate_expect_backward_output(abs_np, angle_np)

    if mode == 'pynative':
        ms.set_context(mode=ms.PYNATIVE_MODE)
        output_forward = polar_forward_func(ms.Tensor(abs_np), ms.Tensor(angle_np))
        output_grad_abs, output_grad_angle = polar_backward_func(ms.Tensor(abs_np), ms.Tensor(angle_np))
    else:
        output_forward = (jit(polar_forward_func, backend="ms_backend", jit_level="O0"))(
            ms.Tensor(abs_np), ms.Tensor(angle_np))
        output_grad_abs, output_grad_angle = (jit(polar_backward_func, backend="ms_backend", jit_level="O0"))(
            ms.Tensor(abs_np), ms.Tensor(angle_np))

    np.testing.assert_allclose(output_forward.asnumpy(), expect_forward, 1e-4, 1e-4)
    np.testing.assert_allclose(output_grad_abs.asnumpy(), expect_grad_abs, 1e-4, 1e-4)
    np.testing.assert_allclose(output_grad_angle.asnumpy(), expect_grad_angle, 1e-4, 1e-4)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
@pytest.mark.parametrize('mode', ['pynative'])
def test_polar_backward_pynative(mode):
    """
    Feature: Ops.
    Description: test polar backward in pynative mode to verify the fix for device address issue.
    Expectation: expect correct result without coredump.
    """
    ms.set_context(mode=ms.PYNATIVE_MODE)
    abs_np, angle_np = generate_random_input((3, 4))
    expect_forward = generate_expect_forward_output(abs_np, angle_np)
    expect_grad_abs, expect_grad_angle = generate_expect_backward_output(abs_np, angle_np)

    output_forward = polar_forward_func(ms.Tensor(abs_np), ms.Tensor(angle_np))
    output_grad_abs, output_grad_angle = polar_backward_func(ms.Tensor(abs_np), ms.Tensor(angle_np))

    np.testing.assert_allclose(output_forward.asnumpy(), expect_forward, 1e-4, 1e-4)
    np.testing.assert_allclose(output_grad_abs.asnumpy(), expect_grad_abs, 1e-4, 1e-4)
    np.testing.assert_allclose(output_grad_angle.asnumpy(), expect_grad_angle, 1e-4, 1e-4)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_polar_3d(mode):
    """
    Feature: Ops.
    Description: test polar with 3D input.
    Expectation: expect correct result.
    """
    abs_np, angle_np = generate_random_input((2, 3, 4))
    expect_forward = generate_expect_forward_output(abs_np, angle_np)
    expect_grad_abs, expect_grad_angle = generate_expect_backward_output(abs_np, angle_np)

    if mode == 'pynative':
        ms.set_context(mode=ms.PYNATIVE_MODE)
        output_forward = polar_forward_func(ms.Tensor(abs_np), ms.Tensor(angle_np))
        output_grad_abs, output_grad_angle = polar_backward_func(ms.Tensor(abs_np), ms.Tensor(angle_np))
    else:
        output_forward = (jit(polar_forward_func, backend="ms_backend", jit_level="O0"))(
            ms.Tensor(abs_np), ms.Tensor(angle_np))
        output_grad_abs, output_grad_angle = (jit(polar_backward_func, backend="ms_backend", jit_level="O0"))(
            ms.Tensor(abs_np), ms.Tensor(angle_np))

    np.testing.assert_allclose(output_forward.asnumpy(), expect_forward, 1e-4, 1e-4)
    np.testing.assert_allclose(output_grad_abs.asnumpy(), expect_grad_abs, 1e-4, 1e-4)
    np.testing.assert_allclose(output_grad_angle.asnumpy(), expect_grad_angle, 1e-4, 1e-4)


@arg_mark(plat_marks=['platform_ascend910b', 'platform_ascend'], level_mark='level0', card_mark='onecard',
          essential_mark='essential', mem_peak=4.00)
def test_polar_dynamic_shape():
    """
    Feature: Test polar with dynamic shape in graph mode.
    Description: call mint.polar with valid input and dynamic shape.
    Expectation: return the correct value.
    """
    inputs1_abs = ms.Tensor(np.array([[1, 10, 2], [0, 6, 1]], np.float32))
    inputs1_angle = ms.Tensor(np.array([[1.0, 3.5, 2.2], [0, 0.1, 0.2]], np.float32))

    inputs2_abs = ms.Tensor(np.array([[[5, 0.1], [0, 5.5]], [[0.1, 0.8], [5, 6]]], np.float32))
    inputs2_angle = ms.Tensor(np.array([[[5.3, -0.1], [0.3, -2.5]], [[1.2, 5.6], [3, 5]]], np.float32))

    TEST_OP(polar_forward_func, [[inputs1_abs, inputs1_angle], [inputs2_abs, inputs2_angle]],
            disable_mode=['GRAPH_MODE_GE'],
            case_config={'disable_grad': True,
                         'all_dim_zero': True})
