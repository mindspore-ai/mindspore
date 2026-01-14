# Copyright 2025-2026 Huawei Technologies Co., Ltd
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
import numpy as np
import pytest
import torch
import mindspore as ms
from mindspore import ops, mint, jit, context
from tests.mark_utils import arg_mark
from tools import allclose_nparray, needs_backward

# Set fixed random seed for reproducibility
np.random.seed(42)


def unsqueeze_forward_func(x, dim):
    """MindSpore forward calculation function"""
    return mint.unsqueeze(x, dim)


def unsqueeze_backward_func(x, dim):
    """MindSpore backward propagation function"""
    return ops.grad(unsqueeze_forward_func, (0,))(x, dim)


KBK_UNSQUEEZE_FORWARD = jit(unsqueeze_forward_func, backend="ms_backend", jit_level="O0")
KBK_UNSQUEEZE_BACKWARD = jit(unsqueeze_backward_func, backend="ms_backend", jit_level="O0")


def unsqueeze_forward(mode, x, dim):
    """Unsqueeze forward wrapper.

    In KBK mode, this wrapper uses jit-wrapped callable.
    """
    if mode == 'pynative':
        return unsqueeze_forward_func(x, dim)
    return KBK_UNSQUEEZE_FORWARD(x, dim)


def unsqueeze_backward(mode, x, dim):
    """Unsqueeze backward wrapper.

    In KBK mode, this wrapper uses jit-wrapped callable.
    """
    if mode == 'pynative':
        return unsqueeze_backward_func(x, dim)
    return KBK_UNSQUEEZE_BACKWARD(x, dim)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=2048.00)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_mint_unsqueeze_normal(mode):
    """
    Feature: mint.unqueeze
    Description: Verify the result of mint.unqueeze
    Expectation: success
    """
    ms.set_context(mode=mode)
    ms.context.set_context(jit_level='O0')

    x = ms.Tensor(np.arange(2 * 3).reshape((2, 3)), dtype=ms.float32)
    dim = 0
    expect_out = np.array(np.arange(2 * 3).reshape((1, 2, 3)))

    out = unsqueeze_forward(mode, x, dim)
    assert out.shape == expect_out.shape
    assert np.allclose(out.asnumpy(), expect_out)

    expect_grad = np.ones((2, 3)).astype(np.float32)
    grad = unsqueeze_backward(mode, x, 0)
    assert grad.shape == expect_grad.shape
    assert np.allclose(grad.asnumpy(), expect_grad)


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


def generate_expect_forward_output(x, dim):
    """Generate expected forward calculation results using PyTorch"""
    return torch.unsqueeze(x, dim)


def generate_expect_backward_output(x, dim, grad):
    """Generate expected backward propagation results using PyTorch"""
    x.requires_grad = True
    out = torch.unsqueeze(x, dim)
    out.backward(grad)
    dx = x.grad
    return dx


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=2048.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_unsqueeze_std(mode):
    """
    Feature: standard forward, backward features.
    Description: test standard cases for unsqueeze.
    Expectation: expect correct result.
    """

    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")

    x = generate_random_input((2, 3, 4), np.float32)
    dim = 1
    expect = generate_expect_forward_output(torch.Tensor(x), dim)
    grad = generate_ones_grad(expect.shape, expect.numpy().dtype)
    expect_grad = generate_expect_backward_output(
        torch.Tensor(x), dim, torch.Tensor(grad))
    output = unsqueeze_forward(mode, ms.Tensor(x), dim)
    output_grad = unsqueeze_backward(mode, ms.Tensor(x), dim)

    allclose_nparray(expect.detach().numpy(), output.asnumpy(),
                     equal_nan=True)
    allclose_nparray(expect_grad.detach().numpy(), output_grad.asnumpy(),
                     equal_nan=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=2048.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
@pytest.mark.parametrize('dtype', [
    np.float16, np.float32, np.float64,
    np.int32, np.int64
])
def test_unsqueeze_dtype_coverage(mode, dtype):
    """
    Feature: dtype coverage for unsqueeze operator.
    Description: test unsqueeze with various dtypes.
    Expectation: results match PyTorch implementation with appropriate
                 precision.
    """

    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    x = generate_random_input((3, 4), dtype)
    dim = 0
    # Handle type mapping supported by PyTorch and MindSpore
    torch_dtype_map = {
        np.int32: torch.int32,
        np.int64: torch.int64,
        np.float16: torch.float16,
        np.float32: torch.float32,
        np.float64: torch.float64
    }
    torch_x = torch.tensor(x, dtype=torch_dtype_map.get(dtype, torch.float32))
    expect = generate_expect_forward_output(torch_x, dim)
    ms_x = ms.Tensor(x)
    output = unsqueeze_forward(mode, ms_x, dim)
    allclose_nparray(expect.detach().numpy(), output.asnumpy(),
                     equal_nan=True)
    # Test backward only for float types
    if np.issubdtype(dtype, np.floating) and dtype != np.float16:
        grad = generate_ones_grad(expect.shape, expect.numpy().dtype)
        expect_grad = generate_expect_backward_output(
            torch_x, dim, torch.tensor(grad))
        output_grad = unsqueeze_backward(mode, ms_x, dim)
        allclose_nparray(expect_grad.detach().numpy(),
                         output_grad.asnumpy(), equal_nan=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=2048.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
@pytest.mark.parametrize('input_shape,dim', [
    ((24,), 0),  # 1D, insert at beginning
    ((24,), 1),  # 1D, insert at end
    ((24,), -1),  # 1D, negative index at end
    ((2, 3, 4), 0),  # 3D, insert at beginning
    ((2, 3, 4), 1),  # 3D, insert in middle
    ((2, 3, 4), 3),  # 3D, insert at end
    ((2, 3, 4), -1),  # 3D, negative index
    ((2, 3, 4, 5), 2),  # 4D, insert in middle
    ((2, 3, 4, 5, 6), 3),  # 5D, insert in middle
])
def test_unsqueeze_dimensions(mode, input_shape, dim):
    """
    Feature: dimension coverage for unsqueeze operator.
    Description: test unsqueeze with various dimension positions.
    Expectation: results match PyTorch implementation.
    """

    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")

    dtype = np.float32
    x = generate_random_input(input_shape, dtype)
    torch_x = torch.tensor(x, dtype=torch.float32)
    expect = generate_expect_forward_output(torch_x, dim)
    grad = generate_ones_grad(expect.shape, dtype)
    torch_grad = torch.tensor(grad, dtype=torch.float32)
    expect_grad = generate_expect_backward_output(
        torch_x, dim, torch_grad)
    ms_x = ms.Tensor(x, dtype=ms.float32)
    output = unsqueeze_forward(mode, ms_x, dim)
    output_grad = unsqueeze_backward(mode, ms_x, dim)
    allclose_nparray(expect.detach().numpy(), output.asnumpy(),
                     equal_nan=True)
    allclose_nparray(expect_grad.detach().numpy(),
                     output_grad.asnumpy(), equal_nan=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=2048.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
@pytest.mark.parametrize('special_type',
                         ['inf', 'nan', 'zero', 'large', 'small'])
def test_unsqueeze_special_values(mode, special_type):
    """
    Feature: special value handling for unsqueeze operator.
    Description: test unsqueeze with inf, nan, zero, large and small values.
    Expectation: results match PyTorch implementation.
    """

    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    x = generate_special_input((2, 3, 4), np.float32, special_type)
    dim = 1
    expect = generate_expect_forward_output(torch.Tensor(x), dim)
    grad = generate_ones_grad(expect.shape, expect.numpy().dtype)
    expect_grad = generate_expect_backward_output(
        torch.Tensor(x), dim, torch.Tensor(grad))
    ms_x = ms.Tensor(x)
    output = unsqueeze_forward(mode, ms_x, dim)
    output_grad = unsqueeze_backward(mode, ms_x, dim)
    allclose_nparray(expect.detach().numpy(), output.asnumpy(),
                     equal_nan=True)
    allclose_nparray(expect_grad.detach().numpy(),
                     output_grad.asnumpy(), equal_nan=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=2048.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_unsqueeze_empty_tensor(mode):
    """
    Feature: empty tensor handling for unsqueeze operator.
    Description: test unsqueeze with empty tensor input.
    Expectation: results match PyTorch implementation.
    """

    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    # Test empty tensors
    test_cases = [
        ((0,), 0),
        ((0,), 1),
        ((2, 0), 0),
        ((2, 0), 2),
        ((0, 3, 4), 1)
    ]
    for shape, dim in test_cases:
        x = generate_random_input(shape, np.float32)
        expect = generate_expect_forward_output(torch.Tensor(x), dim)
        ms_x = ms.Tensor(x)
        output = unsqueeze_forward(mode, ms_x, dim)
        assert output.shape == expect.shape
        allclose_nparray(expect.detach().numpy(), output.asnumpy(),
                         equal_nan=True)
        # Test backward propagation
        grad = generate_ones_grad(expect.shape, expect.numpy().dtype)
        expect_grad = generate_expect_backward_output(
            torch.Tensor(x), dim, torch.Tensor(grad))
        output_grad = unsqueeze_backward(mode, ms_x, dim)
        allclose_nparray(expect_grad.detach().numpy(),
                         output_grad.asnumpy(), equal_nan=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=2048.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
@pytest.mark.parametrize('has_dynamic_rank', [False, True])
def test_unsqueeze_dynamic_shape_advanced(mode, has_dynamic_rank):
    """
    Feature: dynamic shape and rank support for unsqueeze operator.
    Description: test unsqueeze with dynamic dimensions and dynamic rank
                 inputs.
    Expectation: results match PyTorch implementation and shape info is
                 correctly handled.
    """

    if mode == 'pynative':
        context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    base_shapes = [
        ((12,), 0),
        ((2, 6), 1),
        ((2, 3, 4), 2),
    ]
    if has_dynamic_rank:
        base_shapes.extend([
            ((24,), -1),
            ((2, 3, 4, 5), 3)
        ])
    for input_shape, dim in base_shapes:
        x_np = generate_random_input(input_shape, np.float32)
        x_torch = torch.Tensor(x_np)
        expect_forward = generate_expect_forward_output(x_torch, dim)
        x_ms = ms.Tensor(x_np)
        output_forward = unsqueeze_forward(mode, x_ms, dim)
        allclose_nparray(expect_forward.detach().numpy(),
                         output_forward.asnumpy(), equal_nan=True)
        assert output_forward.shape == expect_forward.shape, \
            (f"Output shape mismatch: expected {expect_forward.shape}, "
             f"got {output_forward.shape}")
        grad_np = generate_ones_grad(expect_forward.shape, np.float32)
        expect_backward = generate_expect_backward_output(
            x_torch, dim, torch.Tensor(grad_np))
        output_backward = unsqueeze_backward(mode, x_ms, dim)
        allclose_nparray(expect_backward.detach().numpy(),
                         output_backward.asnumpy(), equal_nan=True)
        assert output_backward.shape == input_shape, \
            (f"Gradient shape mismatch: expected {input_shape}, "
             f"got {output_backward.shape}")


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=2048.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_unsqueeze_zero_bias(mode):
    """
    Feature: zero bias with torch_cpu.
    Description: test zero bias between mint.unsqueeze and torch_cpu results.
    Expectation: complete consistency with torch_cpu results.
    """

    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    x = generate_random_input((2, 3, 4), np.float32)
    dim = 1
    torch_x = torch.tensor(x)
    expect = generate_expect_forward_output(torch_x, dim)
    ms_x = ms.Tensor(x)
    output = unsqueeze_forward(mode, ms_x, dim)
    output_grad = unsqueeze_backward(mode, ms_x, dim)
    # Verify zero bias with strict precision requirements
    allclose_nparray(expect.detach().numpy(), output.asnumpy(),
                     equal_nan=True)
    # Verify backward propagation zero bias
    grad = generate_ones_grad(expect.shape, np.float32)
    expect_grad = generate_expect_backward_output(
        torch_x, dim, torch.tensor(grad))
    allclose_nparray(expect_grad.detach().numpy(),
                     output_grad.asnumpy(), equal_nan=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=2048.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_unsqueeze_non_contiguous(mode):
    """
    Feature: non-contiguous input support for unsqueeze operator.
    Description: test unsqueeze with non-contiguous tensor input.
    Expectation: results match PyTorch implementation.
    """

    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    dtype = np.float32
    x = generate_random_input((2, 3, 4), dtype)
    # Create non-contiguous tensor (via transpose and slicing)
    x_contiguous = np.transpose(x, (1, 0, 2))[:, :, ::2]
    assert not x_contiguous.flags.contiguous, "Input is still contiguous"
    dim = 1
    pt_x = torch.tensor(x_contiguous)
    expect = generate_expect_forward_output(pt_x, dim)
    output = unsqueeze_forward(mode, ms.Tensor(x_contiguous), dim)
    allclose_nparray(expect.detach().numpy(), output.asnumpy(),
                     equal_nan=True)
    # Non-contiguous input backward test
    grad = np.ones(expect.shape, dtype=expect.numpy().dtype)
    expect_grad = generate_expect_backward_output(
        pt_x, dim, torch.tensor(grad))
    output_grad = unsqueeze_backward(mode, ms.Tensor(x_contiguous), dim)
    allclose_nparray(expect_grad.detach().numpy(),
                     output_grad.asnumpy(), equal_nan=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=2048.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_unsqueeze_functional_interface(mode):
    """
    Feature: Functional interface test
    Description: Test mint.unsqueeze functional interface
    Expectation: Expect correct results
    """
    dtype = np.float32
    x = generate_random_input((2, 3, 4), dtype)
    dim = 1
    ms_x = ms.tensor(x)
    pt_x = torch.tensor(x)

    if mode == 'pynative':
        context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    # Test functional interface
    ms_output = unsqueeze_forward(mode, ms_x, dim)
    expect = generate_expect_forward_output(pt_x, dim)
    allclose_nparray(expect.detach().numpy(), ms_output.asnumpy(),
                     equal_nan=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=2048.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_unsqueeze_jit_mode(mode):
    """
    Feature: JIT mode test
    Description: Test mint.unsqueeze functionality in JIT mode
    Expectation: Expect correct results
    """

    if mode == 'pynative':
        context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    dtype = np.float32
    x = generate_random_input((2, 3, 4), dtype)
    dim = 1
    ms_x = ms.tensor(x)
    pt_x = torch.tensor(x)
    # Test forward
    ms_output = unsqueeze_forward(mode, ms_x, dim)
    expect = generate_expect_forward_output(pt_x, dim)
    allclose_nparray(expect.detach().numpy(), ms_output.asnumpy(),
                     equal_nan=True)
    # Test backward
    if needs_backward(dtype):
        grad = generate_ones_grad(ms_output.shape, dtype)
        pt_grad = torch.tensor(grad)
        expect_grad = generate_expect_backward_output(
            pt_x, dim, pt_grad)
        ms_grad = unsqueeze_backward(mode, ms_x, dim)
        allclose_nparray(expect_grad.detach().numpy(),
                         ms_grad.asnumpy(), equal_nan=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=2048.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_unsqueeze_large_tensors(mode):
    """
    Feature: Large tensor unsqueeze test
    Description: Test mint.unsqueeze handling of large tensors
    Expectation: Expect correct results
    """

    if mode == 'pynative':
        context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    dtype = np.float32
    x = generate_random_input((100, 50), dtype)
    dim = 1
    ms_tensor = ms.tensor(x)
    pt_tensor = torch.tensor(x)
    expect_forward = generate_expect_forward_output(pt_tensor, dim)
    ms_forward = unsqueeze_forward(mode, ms_tensor, dim)
    allclose_nparray(expect_forward.detach().numpy(),
                     ms_forward.asnumpy(), equal_nan=True)
    if needs_backward(dtype):
        grad = generate_ones_grad(expect_forward.shape, dtype)
        pt_grad = torch.tensor(grad)
        expect_backward = generate_expect_backward_output(
            pt_tensor, dim, pt_grad)
        ms_grad = unsqueeze_backward(mode, ms_tensor, dim)
        allclose_nparray(expect_backward.detach().numpy(),
                         ms_grad.asnumpy(), equal_nan=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=2048.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_unsqueeze_high_dimension(mode):
    """
    Feature: High dimension unsqueeze test
    Description: Test mint.unsqueeze handling of high-dimensional tensors
    Expectation: Expect correct results
    """

    if mode == 'pynative':
        context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    dtype = np.float32
    x = generate_random_input((2, 3, 4, 5), dtype)
    dim = 2
    ms_tensor = ms.tensor(x)
    pt_tensor = torch.tensor(x)
    expect_forward = generate_expect_forward_output(pt_tensor, dim)
    ms_output_forward = unsqueeze_forward(mode, ms_tensor, dim)
    allclose_nparray(expect_forward.detach().numpy(),
                     ms_output_forward.asnumpy(), equal_nan=True)
    if needs_backward(dtype):
        grad = generate_ones_grad(expect_forward.shape, dtype)
        pt_grad = torch.tensor(grad)
        expect_backward = generate_expect_backward_output(
            pt_tensor, dim, pt_grad)
        ms_grad = unsqueeze_backward(mode, ms_tensor, dim)
        allclose_nparray(expect_backward.detach().numpy(),
                         ms_grad.asnumpy(), equal_nan=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=2048.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_unsqueeze_very_large_tensors(mode):
    """
    Feature: Very large tensor unsqueeze test
    Description: Test mint.unsqueeze handling of very large tensors
    Expectation: Expect correct results
    """
    dtype = np.float32
    x = generate_random_input((1000, 100), dtype)
    dim = 1
    ms_tensor = ms.tensor(x)
    pt_tensor = torch.tensor(x)
    expect_forward = generate_expect_forward_output(pt_tensor, dim)
    if mode == 'pynative':
        context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    ms_output_forward = unsqueeze_forward(mode, ms_tensor, dim)
    allclose_nparray(expect_forward.detach().numpy(),
                     ms_output_forward.asnumpy(), equal_nan=True)
    if needs_backward(dtype):
        grad = generate_ones_grad(expect_forward.shape, dtype)
        pt_grad = torch.tensor(grad)
        expect_backward = generate_expect_backward_output(
            pt_tensor, dim, pt_grad)
        ms_grad = unsqueeze_backward(mode, ms_tensor, dim)
        allclose_nparray(expect_backward.detach().numpy(),
                         ms_grad.asnumpy(), equal_nan=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=2048.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_unsqueeze_complex_dtypes(mode, dtype):
    """
    Feature: complex dtype support for unsqueeze operator.
    Description: test unsqueeze with complex dtypes.
    Expectation: results match PyTorch implementation.
    """

    if mode == 'pynative':
        context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    x = generate_random_input((3, 4), dtype)
    dim = 1
    torch_x = torch.tensor(x)
    expect = torch.unsqueeze(torch_x, dim)
    ms_x = ms.Tensor(x)
    output = unsqueeze_forward(mode, ms_x, dim)
    output_np = output.asnumpy()
    allclose_nparray(expect.detach().numpy(), output_np, equal_nan=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=2048.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_unsqueeze_0d_tensor(mode):
    """
    Feature: unsqueeze 0D tensor
    Description: test unsqueeze on scalar/0D tensor
    Expectation: results match PyTorch implementation
    """

    if mode == 'pynative':
        context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")

    dtype = np.float32
    x = np.array(5.0, dtype=dtype)

    torch_x = torch.tensor(x)
    expect = generate_expect_forward_output(torch_x, 0)
    ms_x = ms.Tensor(x)
    output = unsqueeze_forward(mode, ms_x, 0)

    allclose_nparray(expect.detach().numpy(), output.asnumpy(), equal_nan=True)

    # Test backward
    grad = generate_ones_grad(expect.shape, dtype)
    expect_grad = generate_expect_backward_output(
        torch_x, 0, torch.tensor(grad))
    output_grad = unsqueeze_backward(mode, ms_x, 0)
    allclose_nparray(expect_grad.detach().numpy(),
                     output_grad.asnumpy(), equal_nan=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=2048.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_unsqueeze_inplace_view(mode):
    """
    Feature: unsqueeze view with inplace operation
    Description: test unsqueeze as view followed by inplace modification
    Expectation: results match PyTorch implementation
    """

    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")

    # Test unsqueeze view + copy_ inplace operation
    dst = generate_random_input((2, 3, 4), np.float32)
    src = generate_random_input((2, 3, 4), np.float32)
    dim = 1

    # PyTorch: unsqueeze creates a view, then inplace copy
    torch_dst = torch.Tensor(dst)
    torch_src = torch.Tensor(src)
    torch_dst_view = torch_dst.unsqueeze(dim)
    torch_dst_view.copy_(torch_src.unsqueeze(dim))
    expect = torch_dst  # Original tensor should be modified

    # MindSpore: unsqueeze creates a view, then inplace copy
    ms_dst = ms.Tensor(dst)
    ms_src = ms.Tensor(src)
    ms_dst_view = ms_dst.unsqueeze(dim)
    ms_dst_view.copy_(ms_src.unsqueeze(dim))
    output = ms_dst  # Original tensor should be modified

    allclose_nparray(expect.detach().numpy(), output.asnumpy(),
                     equal_nan=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=2048.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_unsqueeze_view_backward(mode):
    """
    Feature: unsqueeze view with backward propagation
    Description: test gradient propagation through unsqueeze view and inplace
    Expectation: results match PyTorch implementation
    """

    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")

    dst = generate_random_input((2, 3, 4), np.float32)
    src = generate_random_input((2, 3, 4), np.float32)
    dim = 1

    # PyTorch backward - following copy_ test pattern
    torch_src = torch.Tensor(src)
    torch_src.requires_grad = True
    # Multiply to create non-leaf tensor for dst before unsqueeze
    torch_dst = torch.Tensor(dst) * 1
    torch_dst_view = torch_dst.unsqueeze(dim)
    torch_out = torch_dst_view.copy_(torch_src.unsqueeze(dim))
    grad = generate_ones_grad(torch_out.shape, np.float32)
    torch_out.backward(torch.Tensor(grad))
    expect_grad = torch_src.grad

    # MindSpore backward - following copy_ test pattern
    def forward_func(dst, src):
        # Multiply dst to create a working copy
        """Helper function."""
        dst_copy = dst * 1
        dst_view = dst_copy.unsqueeze(dim)
        out = dst_view.copy_(src.unsqueeze(dim))
        return out

    ms_dst = ms.Tensor(dst)
    ms_src = ms.Tensor(src)
    output_grad = ops.grad(forward_func, (1,))(ms_dst, ms_src)

    allclose_nparray(expect_grad.detach().numpy(),
                     output_grad.asnumpy(), equal_nan=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=2048.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
@pytest.mark.parametrize('input_shape,dim', [
    ((24,), 0),
    ((2, 3, 4), 1),
    ((2, 2, 3, 4), 2),
])
def test_unsqueeze_view_multiple_shapes(mode, input_shape, dim):
    """
    Feature: unsqueeze view with various shapes
    Description: test unsqueeze view + inplace with different shape
                 combinations
    Expectation: results match PyTorch implementation
    """

    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")

    dst = generate_random_input(input_shape, np.float32)
    src = generate_random_input(input_shape, np.float32)

    # PyTorch
    torch_dst = torch.Tensor(dst)
    torch_src = torch.Tensor(src)
    torch_dst_view = torch_dst.unsqueeze(dim)
    torch_dst_view.copy_(torch_src.unsqueeze(dim))
    expect = torch_dst

    # MindSpore
    ms_dst = ms.Tensor(dst)
    ms_src = ms.Tensor(src)
    ms_dst_view = ms_dst.unsqueeze(dim)
    ms_dst_view.copy_(ms_src.unsqueeze(dim))
    output = ms_dst

    allclose_nparray(expect.detach().numpy(), output.asnumpy(),
                     equal_nan=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=2048.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_unsqueeze_view_chain(mode):
    """
    Feature: chained unsqueeze views with inplace operation
    Description: test multiple unsqueeze views followed by inplace modification
    Expectation: results match PyTorch implementation
    """

    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")

    x = generate_random_input((2, 3, 4), np.float32)
    src = generate_random_input((2, 3, 4), np.float32)

    # PyTorch: chain of unsqueezes
    torch_x = torch.Tensor(x)
    torch_src = torch.Tensor(src)
    torch_view1 = torch_x.unsqueeze(1)
    torch_view2 = torch_view1.unsqueeze(3)
    torch_src_view = torch_src.unsqueeze(1).unsqueeze(3)
    torch_view2.copy_(torch_src_view)
    expect = torch_x

    # MindSpore: chain of unsqueezes
    ms_x = ms.Tensor(x)
    ms_src = ms.Tensor(src)
    ms_view1 = ms_x.unsqueeze(1)
    ms_view2 = ms_view1.unsqueeze(3)
    ms_src_view = ms_src.unsqueeze(1).unsqueeze(3)
    ms_view2.copy_(ms_src_view)
    output = ms_x

    allclose_nparray(expect.detach().numpy(), output.asnumpy(),
                     equal_nan=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=2048.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
@pytest.mark.parametrize('dtype', [np.float32, np.float64, np.int32, np.int64])
def test_unsqueeze_view_dtypes(mode, dtype):
    """
    Feature: unsqueeze view with various data types
    Description: test unsqueeze view + inplace with different dtypes
    Expectation: results match PyTorch implementation
    """

    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")

    dst = generate_random_input((2, 3, 4), dtype)
    src = generate_random_input((2, 3, 4), dtype)
    dim = 1

    torch_dtype_map = {
        np.int32: torch.int32,
        np.int64: torch.int64,
        np.float32: torch.float32,
        np.float64: torch.float64
    }

    # PyTorch
    torch_dst = torch.tensor(dst, dtype=torch_dtype_map[dtype])
    torch_src = torch.tensor(src, dtype=torch_dtype_map[dtype])
    torch_dst_view = torch_dst.unsqueeze(dim)
    torch_dst_view.copy_(torch_src.unsqueeze(dim))
    expect = torch_dst

    # MindSpore
    ms_dst = ms.Tensor(dst)
    ms_src = ms.Tensor(src)
    ms_dst_view = ms_dst.unsqueeze(dim)
    ms_dst_view.copy_(ms_src.unsqueeze(dim))
    output = ms_dst

    allclose_nparray(expect.detach().numpy(), output.asnumpy(),
                     equal_nan=True)
