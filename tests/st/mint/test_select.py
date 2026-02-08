# Copyright 2026 Huawei Technologies Co., Ltd
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
"""select op test case"""
import pytest
import numpy as np
import torch
import mindspore as ms
from mindspore import ops, mint, jit
from mindspore import context
from tests.mark_utils import arg_mark
from tools import allclose_nparray, needs_backward


# Set fixed random seed for reproducibility
np.random.seed(42)


def generate_random_input(shape, dtype):
    """Generate random input data"""
    if np.issubdtype(dtype, np.complexfloating):
        # For complex types, generate real and imaginary parts
        real_dtype = np.float32 if dtype == np.complex64 else np.float64
        real_part = np.random.uniform(-1, 1, shape).astype(real_dtype)
        imag_part = np.random.uniform(-1, 1, shape).astype(real_dtype)
        return real_part + 1j * imag_part
    if np.issubdtype(dtype, np.integer):
        return np.random.randint(-100, 100, shape).astype(dtype)
    # For float types
    return np.random.uniform(-1, 1, shape).astype(dtype)


def generate_special_input(shape, dtype, special_type):
    """Generate special input data"""
    if special_type == "inf":
        x = np.ones(shape, dtype=dtype)
        x[0] = np.inf
        x[-1] = -np.inf
        return x
    if special_type == "nan":
        x = np.ones(shape, dtype=dtype)
        x[0] = np.nan
        return x
    if special_type == "zero":
        return np.zeros(shape, dtype=dtype)
    if special_type == "large":
        return np.random.uniform(1e6, 1e9, shape).astype(dtype)
    if special_type == "small":
        return np.random.uniform(-1e-9, 1e-9, shape).astype(dtype)
    return generate_random_input(shape, dtype)


def generate_ones_grad(shape, dtype):
    """Generate gradient input with all ones"""
    return np.ones(shape).astype(dtype)


def generate_expect_forward_output(x, dim, index):
    """Generate expected forward calculation results using PyTorch"""
    return torch.select(x, dim, index)


def generate_expect_backward_output(x, dim, index, grad):
    """Generate expected backward propagation results using PyTorch"""
    x.requires_grad = True
    out = torch.select(x, dim, index)
    out.backward(grad)
    dx = x.grad
    return dx


def select_forward_func(x, dim, index):
    """MindSpore forward calculation function"""
    return mint.select(x, dim, index)


def select_backward_func(x, dim, index):
    """MindSpore backward propagation function"""
    return ops.grad(select_forward_func, (0,))(x, dim, index)


KBK_SELECT_FORWARD = jit(select_forward_func, backend="ms_backend", jit_level="O0")
KBK_SELECT_BACKWARD = jit(select_backward_func, backend="ms_backend", jit_level="O0")


def select_forward(mode, x, dim, index):
    """Select forward wrapper.

    In KBK mode, this wrapper uses jit-wrapped callable.
    """
    if mode == 'pynative':
        return select_forward_func(x, dim, index)
    return KBK_SELECT_FORWARD(x, dim, index)


def select_backward(mode, x, dim, index):
    """Select backward wrapper.

    In KBK mode, this wrapper uses jit-wrapped callable.
    """
    if mode == 'pynative':
        return select_backward_func(x, dim, index)
    return KBK_SELECT_BACKWARD(x, dim, index)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=1024.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_select_std(mode):
    """
    Feature: standard forward, backward features.
    Description: test standard cases for select.
    Expectation: expect correct result.
    """

    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")

    x = generate_random_input((2, 3, 4), np.float32)
    dim = 1
    index = 1
    expect = generate_expect_forward_output(torch.Tensor(x), dim, index)
    grad = generate_ones_grad(expect.shape, expect.numpy().dtype)
    expect_grad = generate_expect_backward_output(
        torch.Tensor(x), dim, index, torch.Tensor(grad))
    output = select_forward(mode, ms.Tensor(x), dim, index)
    output_grad = select_backward(mode, ms.Tensor(x), dim, index)

    allclose_nparray(expect.detach().numpy(), output.asnumpy(),
                     equal_nan=True)
    allclose_nparray(expect_grad.detach().numpy(), output_grad.asnumpy(),
                     equal_nan=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=1024.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
@pytest.mark.parametrize('dtype', [
    np.float16, np.float32, np.float64,
    np.int32, np.int64
])
def test_select_dtype_coverage(mode, dtype):
    """
    Feature: dtype coverage for select operator.
    Description: test select with various dtypes.
    Expectation: results match PyTorch implementation with appropriate
                 precision.
    """

    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    x = generate_random_input((3, 4, 5), dtype)
    dim = 1
    index = 2
    # Handle type mapping supported by PyTorch and MindSpore
    torch_dtype_map = {
        np.int32: torch.int32,
        np.int64: torch.int64,
        np.float16: torch.float16,
        np.float32: torch.float32,
        np.float64: torch.float64
    }
    torch_x = torch.tensor(x, dtype=torch_dtype_map.get(dtype, torch.float32))
    expect = generate_expect_forward_output(torch_x, dim, index)
    ms_x = ms.Tensor(x)
    output = select_forward(mode, ms_x, dim, index)
    allclose_nparray(expect.detach().numpy(), output.asnumpy(),
                     equal_nan=True)
    # Test backward only for float types
    if np.issubdtype(dtype, np.floating) and dtype != np.float16:
        grad = generate_ones_grad(expect.shape, expect.numpy().dtype)
        expect_grad = generate_expect_backward_output(
            torch_x, dim, index, torch.tensor(grad))
        output_grad = select_backward(mode, ms_x, dim, index)
        allclose_nparray(expect_grad.detach().numpy(),
                         output_grad.asnumpy(), equal_nan=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=1024.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
@pytest.mark.parametrize('input_shape,dim,index', [
    ((24,), 0, 10),  # 1D tensor
    ((2, 3, 4), 0, 1),  # 3D to 2D, select along dim 0
    ((2, 3, 4), 1, 2),  # 3D to 2D, select along dim 1
    ((2, 3, 4), 2, 3),  # 3D to 2D, select along dim 2
    ((2, 3, 4, 5), 0, 0),  # 4D to 3D, select along dim 0
    ((2, 3, 4, 5), 1, 1),  # 4D to 3D, select along dim 1
    ((2, 3, 4, 5), 2, 2),  # 4D to 3D, select along dim 2
    ((2, 3, 4, 5), 3, 4),  # 4D to 3D, select along dim 3
    ((2, 3, 4, 5, 6), 2, 1),  # 5D to 4D
])
def test_select_dimensions(mode, input_shape, dim, index):
    """
    Feature: dimension coverage for select operator.
    Description: test select with various dimension selections.
    Expectation: results match PyTorch implementation.
    """

    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")

    dtype = np.float32
    x = generate_random_input(input_shape, dtype)
    torch_x = torch.tensor(x, dtype=torch.float32)
    expect = generate_expect_forward_output(torch_x, dim, index)
    grad = generate_ones_grad(expect.shape, dtype)
    torch_grad = torch.tensor(grad, dtype=torch.float32)
    expect_grad = generate_expect_backward_output(
        torch_x, dim, index, torch_grad)
    ms_x = ms.Tensor(x, dtype=ms.float32)
    output = select_forward(mode, ms_x, dim, index)
    output_grad = select_backward(mode, ms_x, dim, index)
    allclose_nparray(expect.detach().numpy(), output.asnumpy(),
                     equal_nan=True)
    allclose_nparray(expect_grad.detach().numpy(),
                     output_grad.asnumpy(), equal_nan=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=1024.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
@pytest.mark.parametrize('special_type',
                         ['inf', 'nan', 'zero', 'large', 'small'])
def test_select_special_values(mode, special_type):
    """
    Feature: special value handling for select operator.
    Description: test select with inf, nan, zero, large and small values.
    Expectation: results match PyTorch implementation.
    """

    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    x = generate_special_input((2, 3, 4), np.float32, special_type)
    dim = 1
    index = 1
    expect = generate_expect_forward_output(torch.Tensor(x), dim, index)
    grad = generate_ones_grad(expect.shape, expect.numpy().dtype)
    expect_grad = generate_expect_backward_output(
        torch.Tensor(x), dim, index, torch.Tensor(grad))
    ms_x = ms.Tensor(x)
    output = select_forward(mode, ms_x, dim, index)
    output_grad = select_backward(mode, ms_x, dim, index)
    allclose_nparray(expect.detach().numpy(), output.asnumpy(),
                     equal_nan=True)
    allclose_nparray(expect_grad.detach().numpy(),
                     output_grad.asnumpy(), equal_nan=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=1024.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
@pytest.mark.parametrize('has_dynamic_rank', [False, True])
def test_select_dynamic_shape_advanced(mode, has_dynamic_rank):
    """
    Feature: dynamic shape and rank support for select operator.
    Description: test select with dynamic dimensions and dynamic rank inputs.
    Expectation: results match PyTorch implementation and shape info is
                 correctly handled.
    """

    if mode == 'pynative':
        context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    base_shapes = [
        ((12,), 0, 5),
        ((2, 6), 1, 3),
        ((2, 3, 4), 2, 2),
    ]
    if has_dynamic_rank:
        base_shapes.extend([
            ((24,), 0, 10),
            ((2, 3, 4, 5), 3, 3)
        ])
    for input_shape, dim, index in base_shapes:
        x_np = generate_random_input(input_shape, np.float32)
        x_torch = torch.Tensor(x_np)
        expect_forward = generate_expect_forward_output(x_torch, dim, index)
        x_ms = ms.Tensor(x_np)
        output_forward = select_forward(mode, x_ms, dim, index)
        allclose_nparray(expect_forward.detach().numpy(),
                         output_forward.asnumpy(), equal_nan=True)
        # Verify output shape
        expected_shape = list(input_shape)
        expected_shape.pop(dim)
        assert output_forward.shape == tuple(expected_shape), \
            (f"Output shape mismatch: expected {tuple(expected_shape)}, "
             f"got {output_forward.shape}")
        grad_np = generate_ones_grad(tuple(expected_shape), np.float32)
        expect_backward = generate_expect_backward_output(
            x_torch, dim, index, torch.Tensor(grad_np))
        output_backward = select_backward(mode, x_ms, dim, index)
        allclose_nparray(expect_backward.detach().numpy(),
                         output_backward.asnumpy(), equal_nan=True)
        assert output_backward.shape == input_shape, \
            (f"Gradient shape mismatch: expected {input_shape}, "
             f"got {output_backward.shape}")


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=1024.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_select_negative_dim(mode):
    """
    Feature: negative dimension support for select operator.
    Description: test select with negative dim index.
    Expectation: results match PyTorch implementation.
    """

    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")

    dtype = np.float32
    test_cases = [
        ((2, 3, 4), -1, 2),  # Select along last dimension
        ((2, 3, 4), -2, 1),  # Select along second-to-last dimension
        ((2, 3, 4), -3, 0),  # Select along first dimension
        ((2, 3, 4, 5), -1, 3),  # 4D tensor, last dimension
        ((2, 3, 4, 5), -2, 2),  # 4D tensor, second-to-last dimension
    ]

    for input_shape, dim, index in test_cases:
        x = generate_random_input(input_shape, dtype)
        torch_x = torch.tensor(x, dtype=torch.float32)
        expect = generate_expect_forward_output(torch_x, dim, index)
        ms_x = ms.Tensor(x, dtype=ms.float32)
        output = select_forward(mode, ms_x, dim, index)
        allclose_nparray(expect.detach().numpy(), output.asnumpy(),
                         equal_nan=True)

        # Test backward
        grad = generate_ones_grad(expect.shape, dtype)
        expect_grad = generate_expect_backward_output(
            torch_x, dim, index, torch.tensor(grad))
        output_grad = select_backward(mode, ms_x, dim, index)
        allclose_nparray(expect_grad.detach().numpy(),
                         output_grad.asnumpy(), equal_nan=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=1024.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_select_negative_index(mode):
    """
    Feature: negative index support for select operator.
    Description: test select with negative index.
    Expectation: results match PyTorch implementation.
    """

    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")

    dtype = np.float32
    test_cases = [
        ((2, 3, 4), 1, -1),  # Select last index along dim 1
        ((2, 3, 4), 1, -2),  # Select second-to-last index along dim 1
        ((2, 3, 4), 2, -3),  # Select third-to-last index along dim 2
        ((5, 6, 7), 0, -1),  # Select last index along dim 0
    ]

    for input_shape, dim, index in test_cases:
        x = generate_random_input(input_shape, dtype)
        torch_x = torch.tensor(x, dtype=torch.float32)
        expect = generate_expect_forward_output(torch_x, dim, index)
        ms_x = ms.Tensor(x, dtype=ms.float32)
        output = select_forward(mode, ms_x, dim, index)
        allclose_nparray(expect.detach().numpy(), output.asnumpy(),
                         equal_nan=True)

        # Test backward
        grad = generate_ones_grad(expect.shape, dtype)
        expect_grad = generate_expect_backward_output(
            torch_x, dim, index, torch.tensor(grad))
        output_grad = select_backward(mode, ms_x, dim, index)
        allclose_nparray(expect_grad.detach().numpy(),
                         output_grad.asnumpy(), equal_nan=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=1024.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_select_zero_bias(mode):
    """
    Feature: zero bias with torch_cpu.
    Description: test zero bias between mint.select and torch_cpu results.
    Expectation: complete consistency with torch_cpu results.
    """

    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    x = generate_random_input((2, 3, 4), np.float32)
    dim = 1
    index = 2
    torch_x = torch.tensor(x)
    expect = generate_expect_forward_output(torch_x, dim, index)
    ms_x = ms.Tensor(x)
    output = select_forward(mode, ms_x, dim, index)
    output_grad = select_backward(mode, ms_x, dim, index)
    # Verify zero bias with strict precision requirements
    allclose_nparray(expect.detach().numpy(), output.asnumpy(),
                     equal_nan=True)
    # Verify backward propagation zero bias
    grad = generate_ones_grad(expect.shape, np.float32)
    expect_grad = generate_expect_backward_output(
        torch_x, dim, index, torch.tensor(grad))
    allclose_nparray(expect_grad.detach().numpy(),
                     output_grad.asnumpy(), equal_nan=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=1024.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_select_non_contiguous(mode):
    """
    Feature: non-contiguous input support for select operator.
    Description: test select with non-contiguous tensor input.
    Expectation: results match PyTorch implementation.
    """

    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    dtype = np.float32
    x = generate_random_input((4, 6, 8), dtype)
    # Create non-contiguous tensor (via transpose and slicing)
    x_non_contiguous = np.transpose(x, (1, 0, 2))[:, :, ::2]
    assert not x_non_contiguous.flags.contiguous, "Input is still contiguous"
    dim = 1
    index = 2
    pt_x = torch.tensor(x_non_contiguous)
    expect = generate_expect_forward_output(pt_x, dim, index)
    output = select_forward(mode, ms.Tensor(x_non_contiguous), dim, index)
    allclose_nparray(expect.detach().numpy(), output.asnumpy(),
                     equal_nan=True)
    # Non-contiguous input backward test
    grad = np.ones(expect.shape, dtype=expect.numpy().dtype)
    expect_grad = generate_expect_backward_output(
        pt_x, dim, index, torch.tensor(grad))
    output_grad = select_backward(mode, ms.Tensor(x_non_contiguous), dim, index)
    allclose_nparray(expect_grad.detach().numpy(),
                     output_grad.asnumpy(), equal_nan=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=1024.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_select_functional_interface(mode):
    """
    Feature: Functional interface test
    Description: Test mint.select functional interface
    Expectation: Expect correct results
    """
    dtype = np.float32
    x = generate_random_input((2, 3, 4), dtype)
    dim = 1
    index = 2
    ms_x = ms.tensor(x)
    pt_x = torch.tensor(x)

    if mode == 'pynative':
        context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    # Test functional interface
    ms_output = select_forward(mode, ms_x, dim, index)
    expect = generate_expect_forward_output(pt_x, dim, index)
    allclose_nparray(expect.detach().numpy(), ms_output.asnumpy(),
                     equal_nan=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=1024.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_select_jit_mode(mode):
    """
    Feature: JIT mode test
    Description: Test mint.select functionality in JIT mode
    Expectation: Expect correct results
    """

    if mode == 'pynative':
        context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    dtype = np.float32
    x = generate_random_input((2, 3, 4), dtype)
    dim = 1
    index = 1
    ms_x = ms.tensor(x)
    pt_x = torch.tensor(x)
    # Test forward
    ms_output = select_forward(mode, ms_x, dim, index)
    expect = generate_expect_forward_output(pt_x, dim, index)
    allclose_nparray(expect.detach().numpy(), ms_output.asnumpy(),
                     equal_nan=True)
    # Test backward
    if needs_backward(dtype):
        grad = generate_ones_grad(ms_output.shape, dtype)
        pt_grad = torch.tensor(grad)
        expect_grad = generate_expect_backward_output(
            pt_x, dim, index, pt_grad)
        ms_grad = select_backward(mode, ms_x, dim, index)
        allclose_nparray(expect_grad.detach().numpy(),
                         ms_grad.asnumpy(), equal_nan=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=1024.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_select_large_tensors(mode):
    """
    Feature: Large tensor select test
    Description: Test mint.select handling of large tensors
    Expectation: Expect correct results
    """

    if mode == 'pynative':
        context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    dtype = np.float32
    x = generate_random_input((100, 50, 20), dtype)
    dim = 1
    index = 25
    ms_tensor = ms.tensor(x)
    pt_tensor = torch.tensor(x)
    expect_forward = generate_expect_forward_output(pt_tensor, dim, index)
    ms_forward = select_forward(mode, ms_tensor, dim, index)
    allclose_nparray(expect_forward.detach().numpy(),
                     ms_forward.asnumpy(), equal_nan=True)
    if needs_backward(dtype):
        grad = generate_ones_grad(expect_forward.shape, dtype)
        pt_grad = torch.tensor(grad)
        expect_backward = generate_expect_backward_output(
            pt_tensor, dim, index, pt_grad)
        ms_grad = select_backward(mode, ms_tensor, dim, index)
        allclose_nparray(expect_backward.detach().numpy(),
                         ms_grad.asnumpy(), equal_nan=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=1024.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_select_high_dimension(mode):
    """
    Feature: High dimension select test
    Description: Test mint.select handling of high-dimensional tensors
    Expectation: Expect correct results
    """

    if mode == 'pynative':
        context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    dtype = np.float32
    x = generate_random_input((2, 3, 4, 5, 6), dtype)
    dim = 3
    index = 2
    ms_tensor = ms.tensor(x)
    pt_tensor = torch.tensor(x)
    expect_forward = generate_expect_forward_output(pt_tensor, dim, index)
    ms_output_forward = select_forward(mode, ms_tensor, dim, index)
    allclose_nparray(expect_forward.detach().numpy(),
                     ms_output_forward.asnumpy(), equal_nan=True)
    if needs_backward(dtype):
        grad = generate_ones_grad(expect_forward.shape, dtype)
        pt_grad = torch.tensor(grad)
        expect_backward = generate_expect_backward_output(
            pt_tensor, dim, index, pt_grad)
        ms_grad = select_backward(mode, ms_tensor, dim, index)
        allclose_nparray(expect_backward.detach().numpy(),
                         ms_grad.asnumpy(), equal_nan=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=1024.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_select_complex_dtypes(mode, dtype):
    """
    Feature: complex dtype support for select operator.
    Description: test select with complex dtypes.
    Expectation: results match PyTorch implementation.
    """

    if mode == 'pynative':
        context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    x = generate_random_input((3, 4, 5), dtype)
    dim = 1
    index = 2
    torch_x = torch.tensor(x)
    expect = torch.select(torch_x, dim, index)
    ms_x = ms.Tensor(x)
    output = select_forward(mode, ms_x, dim, index)
    output_np = output.asnumpy()
    allclose_nparray(expect.detach().numpy(), output_np, equal_nan=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=1024.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_select_inplace_view(mode):
    """
    Feature: select view with inplace operation
    Description: test select as view followed by inplace modification
    Expectation: results match PyTorch implementation
    """

    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")

    # Test select view + copy_ inplace operation
    dst = generate_random_input((2, 3, 4), np.float32)
    src = generate_random_input((2, 4), np.float32)
    dim = 1
    index = 1

    # PyTorch: select creates a view, then inplace copy
    torch_dst = torch.Tensor(dst)
    torch_src = torch.Tensor(src)
    torch_dst_view = torch_dst.select(dim, index)
    torch_dst_view.copy_(torch_src)
    expect = torch_dst  # Original tensor should be modified

    # MindSpore: select creates a view, then inplace copy
    ms_dst = ms.Tensor(dst)
    ms_src = ms.Tensor(src)
    ms_dst_view = ms_dst.select(dim, index)
    ms_dst_view.copy_(ms_src)
    output_result = ms_dst  # Original tensor should be modified

    allclose_nparray(expect.detach().numpy(), output_result.asnumpy(),
                     equal_nan=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=1024.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_select_view_backward(mode):
    """
    Feature: select view with backward propagation
    Description: test gradient propagation through select view and inplace
    Expectation: results match PyTorch implementation
    """

    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")

    dst = generate_random_input((2, 3, 4), np.float32)
    src = generate_random_input((2, 4), np.float32)
    dim = 1
    index = 1

    # PyTorch backward
    torch_src = torch.Tensor(src)
    torch_src.requires_grad = True
    # Multiply to create non-leaf tensor for dst before select
    torch_dst = torch.Tensor(dst) * 1
    torch_dst_view = torch_dst.select(dim, index)
    torch_out = torch_dst_view.copy_(torch_src)
    grad = generate_ones_grad(torch_out.shape, np.float32)
    torch_out.backward(torch.Tensor(grad))
    expect_grad = torch_src.grad

    # MindSpore backward
    def forward_func(dst, src):
        # Multiply dst to create a working copy
        """Helper function."""
        dst_copy = dst * 1
        dst_view = dst_copy.select(dim, index)
        out = dst_view.copy_(src)
        return out

    ms_dst = ms.Tensor(dst)
    ms_src = ms.Tensor(src)
    output_grad = ops.grad(forward_func, (1,))(ms_dst, ms_src)

    allclose_nparray(expect_grad.detach().numpy(),
                     output_grad.asnumpy(), equal_nan=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=1024.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
@pytest.mark.parametrize('input_shape,dim,index', [
    ((24,), 0, 10),
    ((2, 3, 4), 1, 2),
    ((2, 2, 3, 4), 2, 1),
])
def test_select_view_multiple_shapes(mode, input_shape, dim, index):
    """
    Feature: select view with various shapes
    Description: test select view + inplace with different shape combinations
    Expectation: results match PyTorch implementation
    """

    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")

    # Calculate selected slice shape
    selected_shape = list(input_shape)
    selected_shape.pop(dim)
    selected_shape = tuple(selected_shape)

    dst = generate_random_input(input_shape, np.float32)
    src = generate_random_input(selected_shape, np.float32)

    # PyTorch
    torch_dst = torch.Tensor(dst)
    torch_src = torch.Tensor(src)
    torch_dst_view = torch_dst.select(dim, index)
    torch_dst_view.copy_(torch_src)
    expect = torch_dst

    # MindSpore
    ms_dst = ms.Tensor(dst)
    ms_src = ms.Tensor(src)
    ms_dst_view = ms_dst.select(dim, index)
    ms_dst_view.copy_(ms_src)
    output_result = ms_dst

    allclose_nparray(expect.detach().numpy(), output_result.asnumpy(),
                     equal_nan=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=1024.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_select_view_chain(mode):
    """
    Feature: chained select views with inplace operation
    Description: test multiple select views followed by inplace modification
    Expectation: results match PyTorch implementation
    """

    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")

    x = generate_random_input((2, 3, 4, 5), np.float32)
    src = generate_random_input((4,), np.float32)

    # PyTorch: chain of selects
    torch_x = torch.Tensor(x)
    torch_src = torch.Tensor(src)
    torch_view1 = torch_x.select(0, 1)  # Shape: (3, 4, 5)
    torch_view2 = torch_view1.select(0, 2)  # Shape: (4, 5)
    torch_view3 = torch_view2.select(1, 3)  # Shape: (4,)
    torch_view3.copy_(torch_src)
    expect = torch_x

    # MindSpore: chain of selects
    ms_x = ms.Tensor(x)
    ms_src = ms.Tensor(src)
    ms_view1 = ms_x.select(0, 1)
    ms_view2 = ms_view1.select(0, 2)
    ms_view3 = ms_view2.select(1, 3)
    ms_view3.copy_(ms_src)
    output_result = ms_x

    allclose_nparray(expect.detach().numpy(), output_result.asnumpy(),
                     equal_nan=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=1024.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
@pytest.mark.parametrize('dtype', [np.float32, np.float64, np.int32, np.int64])
def test_select_view_dtypes(mode, dtype):
    """
    Feature: select view with various data types
    Description: test select view + inplace with different dtypes
    Expectation: results match PyTorch implementation
    """

    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")

    dst = generate_random_input((2, 3, 4), dtype)
    src = generate_random_input((2, 4), dtype)
    dim = 1
    index = 1

    torch_dtype_map = {
        np.int32: torch.int32,
        np.int64: torch.int64,
        np.float32: torch.float32,
        np.float64: torch.float64
    }

    # PyTorch
    torch_dst = torch.tensor(dst, dtype=torch_dtype_map[dtype])
    torch_src = torch.tensor(src, dtype=torch_dtype_map[dtype])
    torch_dst_view = torch_dst.select(dim, index)
    torch_dst_view.copy_(torch_src)
    expect = torch_dst

    # MindSpore
    ms_dst = ms.Tensor(dst)
    ms_src = ms.Tensor(src)
    ms_dst_view = ms_dst.select(dim, index)
    ms_dst_view.copy_(ms_src)
    output_result = ms_dst

    allclose_nparray(expect.detach().numpy(), output_result.asnumpy(),
                     equal_nan=True)
