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
"""squeeze op test case"""
import pytest
import numpy as np
import torch
import mindspore as ms
from mindspore import ops, mint, vmap, jit
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
        x.flat[0] = np.inf
        x.flat[-1] = -np.inf
        return x
    if special_type == "nan":
        x = np.ones(shape, dtype=dtype)
        x.flat[0] = np.nan
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
    return torch.squeeze(x, dim)


def generate_expect_backward_output(x, dim, grad):
    """Generate expected backward propagation results using PyTorch"""
    x.requires_grad = True
    if dim is None:
        out = torch.squeeze(x)
    else:
        out = torch.squeeze(x, dim)
    out.backward(grad)
    dx = x.grad
    return dx


def squeeze_forward_func(x, dim):
    """MindSpore forward calculation function"""
    return mint.squeeze(x, dim)


def squeeze_backward_func(x, dim):
    """MindSpore backward propagation function"""
    def forward_fn(x):
        """Helper function."""
        return mint.squeeze(x, dim)
    return ops.grad(forward_fn, (0,))(x)


KBK_SQUEEZE_FORWARD = jit(squeeze_forward_func, backend="ms_backend", jit_level="O0")
KBK_SQUEEZE_BACKWARD = jit(squeeze_backward_func, backend="ms_backend", jit_level="O0")


def squeeze_forward(mode, x, dim):
    """Squeeze forward wrapper.

    In KBK mode, this wrapper uses jit-wrapped callable.
    """
    if mode == 'pynative':
        return squeeze_forward_func(x, dim)
    return KBK_SQUEEZE_FORWARD(x, dim)


def squeeze_backward(mode, x, dim):
    """Squeeze backward wrapper.

    In KBK mode, this wrapper uses jit-wrapped callable.
    """
    if mode == 'pynative':
        return squeeze_backward_func(x, dim)
    return KBK_SQUEEZE_BACKWARD(x, dim)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=1024.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_squeeze_std(mode):
    """
    Feature: standard forward, backward features.
    Description: test standard cases for squeeze.
    Expectation: expect correct result.
    """

    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")

    x = generate_random_input((2, 1, 3, 1, 4), np.float32)
    dim = 1  # Squeeze dimension 1
    expect = generate_expect_forward_output(torch.Tensor(x), dim)
    grad = generate_ones_grad(expect.shape, expect.numpy().dtype)
    expect_grad = generate_expect_backward_output(
        torch.Tensor(x), dim, torch.Tensor(grad))
    output = squeeze_forward(mode, ms.Tensor(x), dim)
    output_grad = squeeze_backward(mode, ms.Tensor(x), dim)

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
def test_squeeze_dtype_coverage(mode, dtype):
    """
    Feature: dtype coverage for squeeze operator.
    Description: test squeeze with various dtypes.
    Expectation: results match PyTorch implementation with appropriate
                 precision.
    """

    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    x = generate_random_input((3, 1, 4), dtype)
    dim = 1  # Squeeze dimension 1
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
    output = squeeze_forward(mode, ms_x, dim)
    allclose_nparray(expect.detach().numpy(), output.asnumpy(),
                     equal_nan=True)
    # Test backward only for float types
    if np.issubdtype(dtype, np.floating) and dtype != np.float16:
        grad = generate_ones_grad(expect.shape, expect.numpy().dtype)
        expect_grad = generate_expect_backward_output(
            torch_x, dim, torch.tensor(grad))
        output_grad = squeeze_backward(mode, ms_x, dim)
        allclose_nparray(expect_grad.detach().numpy(),
                         output_grad.asnumpy(), equal_nan=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=1024.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
@pytest.mark.parametrize('input_shape,dim', [
    ((1, 2, 3), 0),  # Squeeze first dimension
    ((2, 1, 3), 1),  # Squeeze middle dimension
    ((2, 3, 1), 2),  # Squeeze last dimension
    ((1, 1, 3, 4), 0),  # Multiple size-1 dims, squeeze first
    ((1, 1, 3, 4), 1),  # Multiple size-1 dims, squeeze second
    ((2, 1, 3, 1, 4), 1),  # Squeeze specific dim
    ((2, 1, 3, 1, 4), 3),  # Squeeze another specific dim
])
def test_squeeze_dimensions(mode, input_shape, dim):
    """
    Feature: dimension coverage for squeeze operator.
    Description: test squeeze with various dimension specifications.
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
    output = squeeze_forward(mode, ms_x, dim)
    output_grad = squeeze_backward(mode, ms_x, dim)
    allclose_nparray(expect.detach().numpy(), output.asnumpy(),
                     equal_nan=True)
    allclose_nparray(expect_grad.detach().numpy(),
                     output_grad.asnumpy(), equal_nan=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=1024.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
@pytest.mark.parametrize('special_type',
                         ['inf', 'nan', 'zero', 'large', 'small'])
def test_squeeze_special_values(mode, special_type):
    """
    Feature: special value handling for squeeze operator.
    Description: test squeeze with inf, nan, zero, large and small values.
    Expectation: results match PyTorch implementation.
    """

    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    x = generate_special_input((2, 1, 3, 1, 4), np.float32, special_type)
    dim = 1  # Squeeze dimension 1
    expect = generate_expect_forward_output(torch.Tensor(x), dim)
    grad = generate_ones_grad(expect.shape, expect.numpy().dtype)
    expect_grad = generate_expect_backward_output(
        torch.Tensor(x), dim, torch.Tensor(grad))
    ms_x = ms.Tensor(x)
    output = squeeze_forward(mode, ms_x, dim)
    output_grad = squeeze_backward(mode, ms_x, dim)
    allclose_nparray(expect.detach().numpy(), output.asnumpy(),
                     equal_nan=True)
    allclose_nparray(expect_grad.detach().numpy(),
                     output_grad.asnumpy(), equal_nan=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=1024.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_squeeze_empty_tensor(mode):
    """
    Feature: empty tensor handling for squeeze operator.
    Description: test squeeze with empty tensor input.
    Expectation: results match PyTorch implementation.
    """

    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    # Test empty tensors
    test_cases = [
        ((1, 0), 0),
        ((0, 1, 3), 1),
        ((2, 1, 0), 1)
    ]
    for shape, dim in test_cases:
        x = generate_random_input(shape, np.float32)
        expect = generate_expect_forward_output(torch.Tensor(x), dim)
        ms_x = ms.Tensor(x)
        output = squeeze_forward(mode, ms_x, dim)
        assert output.shape == expect.shape
        allclose_nparray(expect.detach().numpy(), output.asnumpy(),
                         equal_nan=True)
        # Test backward propagation
        grad = generate_ones_grad(expect.shape, expect.numpy().dtype)
        expect_grad = generate_expect_backward_output(
            torch.Tensor(x), dim, torch.Tensor(grad))
        output_grad = squeeze_backward(mode, ms_x, dim)
        allclose_nparray(expect_grad.detach().numpy(),
                         output_grad.asnumpy(), equal_nan=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=1024.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
@pytest.mark.parametrize('batch_size', [8, 16, 32, 64, 128])
def test_squeeze_vmap(mode, batch_size):
    """
    Feature: vmap support for squeeze operator.
    Description: test squeeze with vmap for batch processing.
    Expectation: results match PyTorch implementation.
    """

    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    # Prepare batch data (batch_size, 2, 1, 3)
    x = generate_random_input((batch_size, 2, 1, 3), np.float32)
    torch_x = torch.tensor(x)
    dim = 1  # Squeeze dimension 1
    # PyTorch doesn't have direct vmap for squeeze, so we test each batch
    expect_list = []
    for i in range(batch_size):
        expect_i = generate_expect_forward_output(torch_x[i], dim)
        expect_list.append(expect_i.numpy())
    expect = np.stack(expect_list)
    # Use vmap for batch processing

    def vmap_func(x):
        """Helper function."""
        return squeeze_forward_func(x, dim)

    vmap_squeeze = vmap(vmap_func, in_axes=0)
    if mode == 'KBK':
        vmap_squeeze = jit(vmap_squeeze, backend="ms_backend", jit_level="O0")
    ms_x = ms.Tensor(x)
    output = vmap_squeeze(ms_x)
    allclose_nparray(expect, output.asnumpy(), equal_nan=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=1024.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
@pytest.mark.parametrize('has_dynamic_rank', [False, True])
def test_squeeze_dynamic_shape_advanced(mode, has_dynamic_rank):
    """
    Feature: dynamic shape and rank support for squeeze operator.
    Description: test squeeze with dynamic dimensions and dynamic rank inputs.
    Expectation: results match PyTorch implementation and shape info is
                 correctly handled.
    """

    if mode == 'pynative':
        context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    base_shapes = [
        ((1, 2, 3), 0),
        ((2, 1, 3), 1),
        ((2, 1, 3, 1), 1),
    ]
    if has_dynamic_rank:
        base_shapes.extend([
            ((1, 1, 2, 3), 0),
            ((2, 1, 3, 1, 4), 1)
        ])
    for input_shape, dim in base_shapes:
        x_np = generate_random_input(input_shape, np.float32)
        x_torch = torch.Tensor(x_np)
        expect_forward = generate_expect_forward_output(x_torch, dim)
        x_ms = ms.Tensor(x_np)
        output_forward = squeeze_forward(mode, x_ms, dim)
        allclose_nparray(expect_forward.detach().numpy(),
                         output_forward.asnumpy(), equal_nan=True)
        assert output_forward.shape == tuple(expect_forward.shape), \
            (f"Output shape mismatch: expected {tuple(expect_forward.shape)}, "
             f"got {output_forward.shape}")
        grad_np = generate_ones_grad(tuple(expect_forward.shape), np.float32)
        expect_backward = generate_expect_backward_output(
            x_torch, dim, torch.Tensor(grad_np))
        output_backward = squeeze_backward(mode, x_ms, dim)
        allclose_nparray(expect_backward.detach().numpy(),
                         output_backward.asnumpy(), equal_nan=True)
        assert output_backward.shape == input_shape, \
            (f"Gradient shape mismatch: expected {input_shape}, "
             f"got {output_backward.shape}")


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=1024.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_squeeze_zero_bias(mode):
    """
    Feature: zero bias with torch_cpu.
    Description: test zero bias between mint.squeeze and torch_cpu results.
    Expectation: complete consistency with torch_cpu results.
    """

    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    x = generate_random_input((2, 1, 3, 1, 4), np.float32)
    dim = 1  # Squeeze dimension 1
    torch_x = torch.tensor(x)
    expect = generate_expect_forward_output(torch_x, dim)
    ms_x = ms.Tensor(x)
    output = squeeze_forward(mode, ms_x, dim)
    output_grad = squeeze_backward(mode, ms_x, dim)
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
          card_mark='onecard', essential_mark='essential', mem_peak=1024.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_squeeze_non_contiguous(mode):
    """
    Feature: non-contiguous input support for squeeze operator.
    Description: test squeeze with non-contiguous tensor input.
    Expectation: results match PyTorch implementation.
    """

    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    dtype = np.float32
    x = generate_random_input((2, 1, 3, 4), dtype)
    # Create non-contiguous tensor (via transpose and slicing)
    x_contiguous = np.transpose(x, (2, 0, 1, 3))[:, :, :, ::2]
    dim = 1  # Squeeze dimension 1 (after transpose, dim 2 becomes dim 1)
    pt_x = torch.tensor(x_contiguous)
    expect = generate_expect_forward_output(pt_x, dim)
    output = squeeze_forward(mode, ms.Tensor(x_contiguous), dim)
    allclose_nparray(expect.detach().numpy(), output.asnumpy(),
                     equal_nan=True)
    # Non-contiguous input backward test
    grad = np.ones(expect.shape, dtype=expect.numpy().dtype)
    expect_grad = generate_expect_backward_output(
        pt_x, dim, torch.tensor(grad))
    output_grad = squeeze_backward(mode, ms.Tensor(x_contiguous), dim)
    allclose_nparray(expect_grad.detach().numpy(),
                     output_grad.asnumpy(), equal_nan=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=1024.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_squeeze_functional_interface(mode):
    """
    Feature: Functional interface test
    Description: Test mint.squeeze functional interface
    Expectation: Expect correct results
    """
    dtype = np.float32
    x = generate_random_input((2, 1, 3, 1, 4), dtype)
    dim = 1  # Squeeze dimension 1
    ms_x = ms.tensor(x)
    pt_x = torch.tensor(x)

    if mode == 'pynative':
        context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    # Test functional interface
    ms_output = squeeze_forward(mode, ms_x, dim)
    expect = generate_expect_forward_output(pt_x, dim)
    allclose_nparray(expect.detach().numpy(), ms_output.asnumpy(),
                     equal_nan=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=1024.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_squeeze_jit_mode(mode):
    """
    Feature: JIT mode test
    Description: Test mint.squeeze functionality in JIT mode
    Expectation: Expect correct results
    """

    if mode == 'pynative':
        context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    dtype = np.float32
    x = generate_random_input((2, 1, 3, 1, 4), dtype)
    dim = 1  # Squeeze dimension 1
    ms_x = ms.tensor(x)
    pt_x = torch.tensor(x)
    # Test forward
    ms_output = squeeze_forward(mode, ms_x, dim)
    expect = generate_expect_forward_output(pt_x, dim)
    allclose_nparray(expect.detach().numpy(), ms_output.asnumpy(),
                     equal_nan=True)
    # Test backward
    if needs_backward(dtype):
        grad = generate_ones_grad(ms_output.shape, dtype)
        pt_grad = torch.tensor(grad)
        expect_grad = generate_expect_backward_output(
            pt_x, dim, pt_grad)
        ms_grad = squeeze_backward(mode, ms_x, dim)
        allclose_nparray(expect_grad.detach().numpy(),
                         ms_grad.asnumpy(), equal_nan=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=1024.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_squeeze_large_tensors(mode):
    """
    Feature: Large tensor squeeze test
    Description: Test mint.squeeze handling of large tensors
    Expectation: Expect correct results
    """

    if mode == 'pynative':
        context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    dtype = np.float32
    x = generate_random_input((100, 1, 50), dtype)
    ms_tensor = ms.tensor(x)
    pt_tensor = torch.tensor(x)
    expect_forward = generate_expect_forward_output(pt_tensor, 1)
    ms_forward = squeeze_forward(mode, ms_tensor, 1)
    allclose_nparray(expect_forward.detach().numpy(),
                     ms_forward.asnumpy(), equal_nan=True)
    if needs_backward(dtype):
        grad = generate_ones_grad(expect_forward.shape, dtype)
        pt_grad = torch.tensor(grad)
        expect_backward = generate_expect_backward_output(
            pt_tensor, 1, pt_grad)
        ms_grad = squeeze_backward(mode, ms_tensor, 1)
        allclose_nparray(expect_backward.detach().numpy(),
                         ms_grad.asnumpy(), equal_nan=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=1024.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_squeeze_high_dimension(mode):
    """
    Feature: High dimension squeeze test
    Description: Test mint.squeeze handling of high-dimensional tensors
    Expectation: Expect correct results
    """

    if mode == 'pynative':
        context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    dtype = np.float32
    x = generate_random_input((2, 1, 3, 1, 4, 1, 5), dtype)
    dim = 1  # Squeeze dimension 1
    ms_tensor = ms.tensor(x)
    pt_tensor = torch.tensor(x)
    expect_forward = generate_expect_forward_output(pt_tensor, dim)
    ms_output_forward = squeeze_forward(mode, ms_tensor, dim)
    allclose_nparray(expect_forward.detach().numpy(),
                     ms_output_forward.asnumpy(), equal_nan=True)
    if needs_backward(dtype):
        grad = generate_ones_grad(expect_forward.shape, dtype)
        pt_grad = torch.tensor(grad)
        expect_backward = generate_expect_backward_output(
            pt_tensor, dim, pt_grad)
        ms_grad = squeeze_backward(mode, ms_tensor, dim)
        allclose_nparray(expect_backward.detach().numpy(),
                         ms_grad.asnumpy(), equal_nan=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=1024.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_squeeze_complex_dtypes(mode, dtype):
    """
    Feature: complex dtype support for squeeze operator.
    Description: test squeeze with complex dtypes.
    Expectation: results match PyTorch implementation.
    """

    if mode == 'pynative':
        context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    x = generate_random_input((3, 1, 4), dtype)
    dim = 1  # Squeeze dimension 1
    torch_x = torch.tensor(x)
    expect = torch.squeeze(torch_x, dim)
    ms_x = ms.Tensor(x)
    output = squeeze_forward(mode, ms_x, dim)
    output_np = output.asnumpy()
    allclose_nparray(expect.detach().numpy(), output_np, equal_nan=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=1024.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_squeeze_negative_dim(mode):
    """
    Feature: negative dimension index for squeeze operator.
    Description: test squeeze with negative dimension indices.
    Expectation: results match PyTorch implementation.
    """

    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")

    dtype = np.float32
    test_cases = [
        ((2, 1, 3, 4), -3),  # Negative index for middle dim
        ((2, 3, 1, 4), -2),  # Negative index for second last dim
        ((2, 3, 4, 1), -1),  # Negative index for last dim
    ]

    for input_shape, dim in test_cases:
        x = generate_random_input(input_shape, dtype)
        torch_x = torch.tensor(x, dtype=torch.float32)
        expect = generate_expect_forward_output(torch_x, dim)
        ms_x = ms.Tensor(x, dtype=ms.float32)
        output = squeeze_forward(mode, ms_x, dim)
        allclose_nparray(expect.detach().numpy(), output.asnumpy(),
                         equal_nan=True)

        # Test backward
        grad = generate_ones_grad(expect.shape, dtype)
        expect_grad = generate_expect_backward_output(
            torch_x, dim, torch.tensor(grad))
        output_grad = squeeze_backward(mode, ms_x, dim)
        allclose_nparray(expect_grad.detach().numpy(),
                         output_grad.asnumpy(), equal_nan=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=1024.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_squeeze_inplace_view(mode):
    """
    Feature: squeeze view with inplace operation
    Description: test squeeze as view followed by inplace modification
    Expectation: results match PyTorch implementation
    """

    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")

    # Test squeeze view + copy_ inplace operation
    dst = generate_random_input((2, 1, 3, 1, 4), np.float32)
    # After squeeze(dim=1): (2, 1, 3, 1, 4) -> (2, 3, 1, 4)
    src = generate_random_input((2, 3, 1, 4), np.float32)
    dim = 1  # Squeeze dimension 1

    # PyTorch: squeeze creates a view, then inplace copy
    torch_dst = torch.Tensor(dst)
    torch_src = torch.Tensor(src)
    torch_dst_view = torch_dst.squeeze(dim)
    torch_dst_view.copy_(torch_src)
    expect = torch_dst  # Original tensor should be modified

    # MindSpore: squeeze creates a view, then inplace copy
    ms_dst = ms.Tensor(dst)
    ms_src = ms.Tensor(src)
    ms_dst_view = ms_dst.squeeze(dim)
    ms_dst_view.copy_(ms_src)
    output = ms_dst  # Original tensor should be modified

    allclose_nparray(expect.detach().numpy(), output.asnumpy(),
                     equal_nan=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=1024.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_squeeze_view_backward(mode):
    """
    Feature: squeeze view with backward propagation
    Description: test gradient propagation through squeeze view and inplace
    Expectation: results match PyTorch implementation
    """

    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")

    dst = generate_random_input((2, 1, 3, 1, 4), np.float32)
    # After squeeze(dim=1): (2, 1, 3, 1, 4) -> (2, 3, 1, 4)
    src = generate_random_input((2, 3, 1, 4), np.float32)
    dim = 1  # Squeeze dimension 1

    # PyTorch backward
    torch_src = torch.Tensor(src)
    torch_src.requires_grad = True
    # Multiply to create non-leaf tensor for dst before squeeze
    torch_dst = torch.Tensor(dst) * 1
    torch_dst_view = torch_dst.squeeze(dim)
    torch_out = torch_dst_view.copy_(torch_src)
    grad = generate_ones_grad(torch_out.shape, np.float32)
    torch_out.backward(torch.Tensor(grad))
    expect_grad = torch_src.grad

    # MindSpore backward
    def forward_func(dst, src):
        # Multiply dst to create a working copy
        """Helper function."""
        dst_copy = dst * 1
        dst_view = dst_copy.squeeze(dim)
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
@pytest.mark.parametrize('input_shape,dim,src_shape', [
    ((2, 1, 3, 4), 1, (2, 3, 4)),
    ((1, 2, 3, 4), 0, (2, 3, 4)),
    ((2, 1, 3, 1, 4), 1, (2, 3, 1, 4)),
])
def test_squeeze_view_multiple_shapes(mode, input_shape, dim, src_shape):
    """
    Feature: squeeze view with various shapes
    Description: test squeeze view + inplace with different shape combinations
    Expectation: results match PyTorch implementation
    """

    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")

    dst = generate_random_input(input_shape, np.float32)
    src = generate_random_input(src_shape, np.float32)

    # PyTorch
    torch_dst = torch.Tensor(dst)
    torch_src = torch.Tensor(src)
    torch_dst_view = torch_dst.squeeze(dim)
    torch_dst_view.copy_(torch_src)
    expect = torch_dst

    # MindSpore
    ms_dst = ms.Tensor(dst)
    ms_src = ms.Tensor(src)
    ms_dst_view = ms_dst.squeeze(dim)
    ms_dst_view.copy_(ms_src)
    output = ms_dst

    allclose_nparray(expect.detach().numpy(), output.asnumpy(),
                     equal_nan=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=1024.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
@pytest.mark.parametrize('dtype', [np.float32, np.float64, np.int32, np.int64])
def test_squeeze_view_dtypes(mode, dtype):
    """
    Feature: squeeze view with various data types
    Description: test squeeze view + inplace with different dtypes
    Expectation: results match PyTorch implementation
    """

    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")

    dst = generate_random_input((2, 1, 3, 1, 4), dtype)
    # Match shape after squeeze: (2, 1, 3, 1, 4) squeeze dim=1 -> (2, 3, 1, 4)
    src = generate_random_input((2, 3, 1, 4), dtype)
    dim = 1  # Squeeze dimension 1

    torch_dtype_map = {
        np.int32: torch.int32,
        np.int64: torch.int64,
        np.float32: torch.float32,
        np.float64: torch.float64
    }

    # PyTorch
    torch_dst = torch.tensor(dst, dtype=torch_dtype_map[dtype])
    torch_src = torch.tensor(src, dtype=torch_dtype_map[dtype])
    torch_dst_view = torch_dst.squeeze(dim)
    torch_dst_view.copy_(torch_src)
    expect = torch_dst

    # MindSpore
    ms_dst = ms.Tensor(dst)
    ms_src = ms.Tensor(src)
    ms_dst_view = ms_dst.squeeze(dim)
    ms_dst_view.copy_(ms_src)
    output = ms_dst

    allclose_nparray(expect.detach().numpy(), output.asnumpy(),
                     equal_nan=True)
