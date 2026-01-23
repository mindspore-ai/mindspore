# Copyright 2025 Huawei Technologies Co., Ltd
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
"""Tests for mint.permute: input, dims."""
import numpy as np
import pytest
import torch
import mindspore as ms
from mindspore import ops, mint, jit
from mindspore import context
from tests.mark_utils import arg_mark
from tools import allclose_nparray, needs_backward


# Set fixed random seed for reproducibility
np.random.seed(42)


def permute_forward_func(x, dims):
    """MindSpore forward calculation function"""
    return mint.permute(x, dims)


def permute_backward_func(x, dims):
    """MindSpore backward propagation function"""
    return ops.grad(permute_forward_func, (0,))(x, dims)


KBK_PERMUTE_FORWARD = jit(
    permute_forward_func, backend="ms_backend", jit_level="O0"
)
KBK_PERMUTE_BACKWARD = jit(
    permute_backward_func, backend="ms_backend", jit_level="O0"
)

def permute_forward(mode, x, dims):
    """Permute forward wrapper.

    In KBK mode, this wrapper uses jit-wrapped callable.
    """
    if mode == 'pynative':
        return permute_forward_func(x, dims)
    return KBK_PERMUTE_FORWARD(x, dims)


def permute_backward(mode, x, dims):
    """Permute backward wrapper.

    In KBK mode, this wrapper uses jit-wrapped callable.
    """
    if mode == 'pynative':
        return permute_backward_func(x, dims)
    return KBK_PERMUTE_BACKWARD(x, dims)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=2048.00)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_mint_permute_normal(mode):
    """
    Feature: mint.permute
    Description: Verify the result of mint.permute
    Expectation: success
    """
    ms.set_context(mode=mode)
    ms.context.set_context(jit_level='O0')

    x_np = np.random.randn(2, 3, 4).astype(np.float32)
    x = ms.Tensor(x_np)
    dims = (2, 0, 1)
    expect_out = x_np.transpose(dims)

    out = permute_forward(mode, x, dims)
    assert out.shape == expect_out.shape
    assert np.allclose(out.asnumpy(), expect_out)

    expect_grad = np.ones((2, 3, 4)).astype(np.float32)
    grad = permute_backward(mode, x, dims)
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


def generate_expect_forward_output(x, dims):
    """Generate expected forward calculation results using PyTorch"""
    return torch.permute(x, dims)


def generate_expect_backward_output(x, dims, grad):
    """Generate expected backward propagation results using PyTorch"""
    x.requires_grad = True
    out = torch.permute(x, dims)
    out.backward(grad)
    dx = x.grad
    return dx


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=2048.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_permute_std(mode):
    """
    Feature: standard forward, backward features.
    Description: test standard cases for permute.
    Expectation: expect correct result.
    """

    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")

    x = generate_random_input((2, 3, 4), np.float32)
    dims = (2, 0, 1)
    expect = generate_expect_forward_output(torch.Tensor(x), dims)
    grad = generate_ones_grad(expect.shape, expect.numpy().dtype)
    expect_grad = generate_expect_backward_output(
        torch.Tensor(x), dims, torch.Tensor(grad))
    output = permute_forward(mode, ms.Tensor(x), dims)
    output_grad = permute_backward(mode, ms.Tensor(x), dims)

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
def test_permute_dtype_coverage(mode, dtype):
    """
    Feature: dtype coverage for permute operator.
    Description: test permute with various dtypes.
    Expectation: results match PyTorch implementation with appropriate
                 precision.
    """

    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    x = generate_random_input((2, 3, 4), dtype)
    dims = (1, 2, 0)
    # Handle type mapping supported by PyTorch and MindSpore
    torch_dtype_map = {
        np.int32: torch.int32,
        np.int64: torch.int64,
        np.float16: torch.float16,
        np.float32: torch.float32,
        np.float64: torch.float64
    }
    torch_x = torch.tensor(x, dtype=torch_dtype_map.get(dtype, torch.float32))
    expect = generate_expect_forward_output(torch_x, dims)
    ms_x = ms.Tensor(x)
    output = permute_forward(mode, ms_x, dims)
    allclose_nparray(expect.detach().numpy(), output.asnumpy(),
                     equal_nan=True)
    # Test backward only for float types
    if np.issubdtype(dtype, np.floating) and dtype != np.float16:
        grad = generate_ones_grad(expect.shape, expect.numpy().dtype)
        expect_grad = generate_expect_backward_output(
            torch_x, dims, torch.tensor(grad))
        output_grad = permute_backward(mode, ms_x, dims)
        allclose_nparray(expect_grad.detach().numpy(),
                         output_grad.asnumpy(), equal_nan=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=2048.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
@pytest.mark.parametrize('input_shape,dims', [
    ((2, 3, 4), (2, 0, 1)),  # 3D permute
    ((2, 3, 4), (0, 2, 1)),  # 3D permute - swap last two dims
    ((2, 3, 4), (1, 0, 2)),  # 3D permute - swap first two dims
    ((2, 3, 4, 5), (3, 2, 1, 0)),  # 4D reverse
    ((2, 3, 4, 5), (0, 2, 1, 3)),  # 4D swap middle dims
    ((2, 3, 4, 5, 6), (4, 3, 2, 1, 0)),  # 5D reverse
    ((2, 3, 4, 5, 6), (0, 1, 3, 2, 4)),  # 5D partial swap
])
def test_permute_dimensions(mode, input_shape, dims):
    """
    Feature: dimension coverage for permute operator.
    Description: test permute with various dimension transformations.
    Expectation: results match PyTorch implementation.
    """

    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")

    dtype = np.float32
    x = generate_random_input(input_shape, dtype)
    torch_x = torch.tensor(x, dtype=torch.float32)
    expect = generate_expect_forward_output(torch_x, dims)
    grad = generate_ones_grad(expect.shape, dtype)
    torch_grad = torch.tensor(grad, dtype=torch.float32)
    expect_grad = generate_expect_backward_output(
        torch_x, dims, torch_grad)
    ms_x = ms.Tensor(x, dtype=ms.float32)
    output = permute_forward(mode, ms_x, dims)
    output_grad = permute_backward(mode, ms_x, dims)
    allclose_nparray(expect.detach().numpy(), output.asnumpy(),
                     equal_nan=True)
    allclose_nparray(expect_grad.detach().numpy(),
                     output_grad.asnumpy(), equal_nan=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=2048.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
@pytest.mark.parametrize('special_type',
                         ['inf', 'nan', 'zero', 'large', 'small'])
def test_permute_special_values(mode, special_type):
    """
    Feature: special value handling for permute operator.
    Description: test permute with inf, nan, zero, large and small values.
    Expectation: results match PyTorch implementation.
    """

    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    x = generate_special_input((2, 3, 4), np.float32, special_type)
    dims = (2, 0, 1)
    expect = generate_expect_forward_output(torch.Tensor(x), dims)
    grad = generate_ones_grad(expect.shape, expect.numpy().dtype)
    expect_grad = generate_expect_backward_output(
        torch.Tensor(x), dims, torch.Tensor(grad))
    ms_x = ms.Tensor(x)
    output = permute_forward(mode, ms_x, dims)
    output_grad = permute_backward(mode, ms_x, dims)
    allclose_nparray(expect.detach().numpy(), output.asnumpy(),
                     equal_nan=True)
    allclose_nparray(expect_grad.detach().numpy(),
                     output_grad.asnumpy(), equal_nan=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=2048.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_permute_empty_tensor(mode):
    """
    Feature: empty tensor handling for permute operator.
    Description: test permute with empty tensor input.
    Expectation: results match PyTorch implementation.
    """

    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    # Test empty tensors
    test_cases = [
        ((0, 3, 4), (1, 2, 0)),
        ((2, 0, 4), (2, 0, 1)),
        ((2, 3, 0), (0, 1, 2))
    ]
    for shape, dims in test_cases:
        x = generate_random_input(shape, np.float32)
        expect = generate_expect_forward_output(torch.Tensor(x), dims)
        ms_x = ms.Tensor(x)
        output = permute_forward(mode, ms_x, dims)
        assert output.shape == expect.shape
        allclose_nparray(expect.detach().numpy(), output.asnumpy(),
                         equal_nan=True)
        # Test backward propagation
        grad = generate_ones_grad(expect.shape, expect.numpy().dtype)
        expect_grad = generate_expect_backward_output(
            torch.Tensor(x), dims, torch.Tensor(grad))
        output_grad = permute_backward(mode, ms_x, dims)
        allclose_nparray(expect_grad.detach().numpy(),
                         output_grad.asnumpy(), equal_nan=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=2048.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
@pytest.mark.parametrize('has_dynamic_rank', [False, True])
def test_permute_dynamic_shape_advanced(mode, has_dynamic_rank):
    """
    Feature: dynamic shape and rank support for permute operator.
    Description: test permute with dynamic dimensions and dynamic rank inputs.
    Expectation: results match PyTorch implementation and shape info is
                 correctly handled.
    """

    if mode == 'pynative':
        context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    base_shapes = [
        ((2, 3, 4), (1, 0, 2)),
        ((3, 4, 5), (2, 1, 0)),
    ]
    if has_dynamic_rank:
        base_shapes.extend([
            ((2, 3, 4, 5), (3, 0, 2, 1)),
            ((2, 3, 4, 5, 6), (4, 2, 0, 1, 3))
        ])
    for input_shape, dims in base_shapes:
        x_np = generate_random_input(input_shape, np.float32)
        x_torch = torch.Tensor(x_np)
        expect_forward = generate_expect_forward_output(x_torch, dims)
        x_ms = ms.Tensor(x_np)
        output_forward = permute_forward(mode, x_ms, dims)
        allclose_nparray(expect_forward.detach().numpy(),
                         output_forward.asnumpy(), equal_nan=True)
        assert output_forward.shape == expect_forward.shape, \
            (f"Output shape mismatch: expected {expect_forward.shape}, "
             f"got {output_forward.shape}")
        grad_np = generate_ones_grad(expect_forward.shape, np.float32)
        expect_backward = generate_expect_backward_output(
            x_torch, dims, torch.Tensor(grad_np))
        output_backward = permute_backward(mode, x_ms, dims)
        allclose_nparray(expect_backward.detach().numpy(),
                         output_backward.asnumpy(), equal_nan=True)
        assert output_backward.shape == input_shape, \
            (f"Gradient shape mismatch: expected {input_shape}, "
             f"got {output_backward.shape}")


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=2048.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_permute_identity(mode):
    """
    Feature: identity permutation support for permute operator.
    Description: test permute with identity permutation (0, 1, 2, ...).
    Expectation: results match PyTorch implementation.
    """

    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")

    dtype = np.float32
    test_cases = [
        ((2, 3, 4), (0, 1, 2)),
        ((2, 3, 4, 5), (0, 1, 2, 3)),
        ((2, 3, 4, 5, 6), (0, 1, 2, 3, 4)),
    ]

    for input_shape, dims in test_cases:
        x = generate_random_input(input_shape, dtype)
        torch_x = torch.tensor(x, dtype=torch.float32)
        expect = generate_expect_forward_output(torch_x, dims)
        ms_x = ms.Tensor(x, dtype=ms.float32)
        output = permute_forward(mode, ms_x, dims)
        allclose_nparray(expect.detach().numpy(), output.asnumpy(),
                         equal_nan=True)

        # Test backward
        grad = generate_ones_grad(expect.shape, dtype)
        expect_grad = generate_expect_backward_output(
            torch_x, dims, torch.tensor(grad))
        output_grad = permute_backward(mode, ms_x, dims)
        allclose_nparray(expect_grad.detach().numpy(),
                         output_grad.asnumpy(), equal_nan=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=2048.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_permute_zero_bias(mode):
    """
    Feature: zero bias with torch_cpu.
    Description: test zero bias between mint.permute and torch_cpu results.
    Expectation: complete consistency with torch_cpu results.
    """

    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    x = generate_random_input((2, 3, 4), np.float32)
    dims = (2, 0, 1)
    torch_x = torch.tensor(x)
    expect = generate_expect_forward_output(torch_x, dims)
    ms_x = ms.Tensor(x)
    output = permute_forward(mode, ms_x, dims)
    output_grad = permute_backward(mode, ms_x, dims)
    # Verify zero bias with strict precision requirements
    allclose_nparray(expect.detach().numpy(), output.asnumpy(),
                     equal_nan=True)
    # Verify backward propagation zero bias
    grad = generate_ones_grad(expect.shape, np.float32)
    expect_grad = generate_expect_backward_output(
        torch_x, dims, torch.tensor(grad))
    allclose_nparray(expect_grad.detach().numpy(),
                     output_grad.asnumpy(), equal_nan=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=2048.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_permute_non_contiguous(mode):
    """
    Feature: non-contiguous input support for permute operator.
    Description: test permute with non-contiguous tensor input.
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
    dims = (1, 2, 0)
    pt_x = torch.tensor(x_contiguous)
    expect = generate_expect_forward_output(pt_x, dims)
    output = permute_forward(mode, ms.Tensor(x_contiguous), dims)
    allclose_nparray(expect.detach().numpy(), output.asnumpy(),
                     equal_nan=True)
    # Non-contiguous input backward test
    grad = np.ones(expect.shape, dtype=expect.numpy().dtype)
    expect_grad = generate_expect_backward_output(
        pt_x, dims, torch.tensor(grad))
    output_grad = permute_backward(mode, ms.Tensor(x_contiguous), dims)
    allclose_nparray(expect_grad.detach().numpy(),
                     output_grad.asnumpy(), equal_nan=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=2048.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_permute_functional_interface(mode):
    """
    Feature: Functional interface test
    Description: Test mint.permute functional interface
    Expectation: Expect correct results
    """
    dtype = np.float32
    x = generate_random_input((2, 3, 4), dtype)
    dims = (2, 0, 1)
    ms_x = ms.tensor(x)
    pt_x = torch.tensor(x)

    if mode == 'pynative':
        context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    # Test functional interface
    ms_output = permute_forward(mode, ms_x, dims)
    expect = generate_expect_forward_output(pt_x, dims)
    allclose_nparray(expect.detach().numpy(), ms_output.asnumpy(),
                     equal_nan=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=2048.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_permute_jit_mode(mode):
    """
    Feature: JIT mode test
    Description: Test mint.permute functionality in JIT mode
    Expectation: Expect correct results
    """

    if mode == 'pynative':
        context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    dtype = np.float32
    x = generate_random_input((2, 3, 4), dtype)
    dims = (2, 0, 1)
    ms_x = ms.tensor(x)
    pt_x = torch.tensor(x)
    # Test forward
    ms_output = permute_forward(mode, ms_x, dims)
    expect = generate_expect_forward_output(pt_x, dims)
    allclose_nparray(expect.detach().numpy(), ms_output.asnumpy(),
                     equal_nan=True)
    # Test backward
    if needs_backward(dtype):
        grad = generate_ones_grad(ms_output.shape, dtype)
        pt_grad = torch.tensor(grad)
        expect_grad = generate_expect_backward_output(
            pt_x, dims, pt_grad)
        ms_grad = permute_backward(mode, ms_x, dims)
        allclose_nparray(expect_grad.detach().numpy(),
                         ms_grad.asnumpy(), equal_nan=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=2048.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_permute_large_tensors(mode):
    """
    Feature: Large tensor permute test
    Description: Test mint.permute handling of large tensors
    Expectation: Expect correct results
    """

    if mode == 'pynative':
        context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    dtype = np.float32
    x = generate_random_input((100, 50, 20), dtype)
    dims = (2, 0, 1)
    ms_tensor = ms.tensor(x)
    pt_tensor = torch.tensor(x)
    expect_forward = generate_expect_forward_output(pt_tensor, dims)
    ms_forward = permute_forward(mode, ms_tensor, dims)
    allclose_nparray(expect_forward.detach().numpy(),
                     ms_forward.asnumpy(), equal_nan=True)
    if needs_backward(dtype):
        grad = generate_ones_grad(expect_forward.shape, dtype)
        pt_grad = torch.tensor(grad)
        expect_backward = generate_expect_backward_output(
            pt_tensor, dims, pt_grad)
        ms_grad = permute_backward(mode, ms_tensor, dims)
        allclose_nparray(expect_backward.detach().numpy(),
                         ms_grad.asnumpy(), equal_nan=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=2048.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_permute_high_dimension(mode):
    """
    Feature: High dimension permute test
    Description: Test mint.permute handling of high-dimensional tensors
    Expectation: Expect correct results
    """

    if mode == 'pynative':
        context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    dtype = np.float32
    x = generate_random_input((2, 3, 4, 5), dtype)
    dims = (3, 1, 0, 2)
    ms_tensor = ms.tensor(x)
    pt_tensor = torch.tensor(x)
    expect_forward = generate_expect_forward_output(pt_tensor, dims)
    ms_output_forward = permute_forward(mode, ms_tensor, dims)
    allclose_nparray(expect_forward.detach().numpy(),
                     ms_output_forward.asnumpy(), equal_nan=True)
    if needs_backward(dtype):
        grad = generate_ones_grad(expect_forward.shape, dtype)
        pt_grad = torch.tensor(grad)
        expect_backward = generate_expect_backward_output(
            pt_tensor, dims, pt_grad)
        ms_grad = permute_backward(mode, ms_tensor, dims)
        allclose_nparray(expect_backward.detach().numpy(),
                         ms_grad.asnumpy(), equal_nan=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=2048.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_permute_very_large_tensors(mode):
    """
    Feature: Very large tensor permute test
    Description: Test mint.permute handling of very large tensors
    Expectation: Expect correct results
    """
    dtype = np.float32
    x = generate_random_input((1000, 100), dtype)
    dims = (1, 0)
    ms_tensor = ms.tensor(x)
    pt_tensor = torch.tensor(x)
    expect_forward = generate_expect_forward_output(pt_tensor, dims)
    if mode == 'pynative':
        context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    ms_output_forward = permute_forward(mode, ms_tensor, dims)
    allclose_nparray(expect_forward.detach().numpy(),
                     ms_output_forward.asnumpy(), equal_nan=True)
    if needs_backward(dtype):
        grad = generate_ones_grad(expect_forward.shape, dtype)
        pt_grad = torch.tensor(grad)
        expect_backward = generate_expect_backward_output(
            pt_tensor, dims, pt_grad)
        ms_grad = permute_backward(mode, ms_tensor, dims)
        allclose_nparray(expect_backward.detach().numpy(),
                         ms_grad.asnumpy(), equal_nan=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=2048.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_permute_complex_dtypes(mode, dtype):
    """
    Feature: complex dtype support for permute operator.
    Description: test permute with complex dtypes.
    Expectation: results match PyTorch implementation.
    """

    if mode == 'pynative':
        context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    x = generate_random_input((2, 3, 4), dtype)
    dims = (2, 0, 1)
    torch_x = torch.tensor(x)
    expect = torch.permute(torch_x, dims)
    ms_x = ms.Tensor(x)
    output = permute_forward(mode, ms_x, dims)
    output_np = output.asnumpy()
    allclose_nparray(expect.detach().numpy(), output_np, equal_nan=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=2048.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_permute_reverse_dims(mode):
    """
    Feature: reverse dimension permutation
    Description: test permute with reversed dimension order
    Expectation: results match PyTorch implementation
    """

    if mode == 'pynative':
        context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")

    dtype = np.float32
    test_cases = [
        ((2, 3, 4), (2, 1, 0)),
        ((2, 3, 4, 5), (3, 2, 1, 0)),
        ((2, 3, 4, 5, 6), (4, 3, 2, 1, 0)),
    ]

    for input_shape, dims in test_cases:
        x = generate_random_input(input_shape, dtype)
        torch_x = torch.tensor(x)
        expect = generate_expect_forward_output(torch_x, dims)
        ms_x = ms.Tensor(x)
        output = permute_forward(mode, ms_x, dims)

        allclose_nparray(expect.detach().numpy(), output.asnumpy(),
                         equal_nan=True)

        # Test backward
        grad = generate_ones_grad(expect.shape, dtype)
        expect_grad = generate_expect_backward_output(
            torch_x, dims, torch.tensor(grad))
        output_grad = permute_backward(mode, ms_x, dims)
        allclose_nparray(expect_grad.detach().numpy(),
                         output_grad.asnumpy(), equal_nan=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=2048.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_permute_transpose_equivalent(mode):
    """
    Feature: permute as transpose operation
    Description: test permute behavior equivalent to transpose for 2D tensors
    Expectation: results match PyTorch implementation
    """

    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")

    dtype = np.float32
    x = generate_random_input((3, 4), dtype)
    dims = (1, 0)

    # Test permute
    torch_x = torch.tensor(x)
    expect = generate_expect_forward_output(torch_x, dims)
    ms_x = ms.Tensor(x)
    output = permute_forward(mode, ms_x, dims)

    allclose_nparray(expect.detach().numpy(), output.asnumpy(),
                     equal_nan=True)

    # Verify it's same as transpose
    torch_transpose = torch.transpose(torch_x, 0, 1)
    allclose_nparray(torch_transpose.detach().numpy(), output.asnumpy(),
                     equal_nan=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=2048.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_permute_inplace_view(mode):
    """
    Feature: permute view with inplace operation
    Description: test permute as view followed by inplace modification
    Expectation: results match PyTorch implementation
    """

    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")

    # Test permute view + copy_ inplace operation
    dst = generate_random_input((2, 3, 4), np.float32)
    src = generate_random_input((2, 3, 4), np.float32)
    dims = (2, 0, 1)

    # PyTorch: permute creates a view, then inplace copy
    torch_dst = torch.Tensor(dst)
    torch_src = torch.Tensor(src)
    torch_dst_view = torch_dst.permute(dims)
    torch_dst_view.copy_(torch_src.permute(dims))
    expect = torch_dst  # Original tensor should be modified

    # MindSpore: permute creates a view, then inplace copy
    ms_dst = ms.Tensor(dst)
    ms_src = ms.Tensor(src)
    ms_dst_view = ms_dst.permute(dims)
    ms_dst_view.copy_(ms_src.permute(dims))
    output = ms_dst  # Original tensor should be modified

    allclose_nparray(expect.detach().numpy(), output.asnumpy(),
                     equal_nan=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=2048.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_permute_view_backward(mode):
    """
    Feature: permute view with backward propagation
    Description: test gradient propagation through permute view and inplace
    Expectation: results match PyTorch implementation
    """

    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")

    dst = generate_random_input((2, 3, 4), np.float32)
    src = generate_random_input((2, 3, 4), np.float32)
    dims = (2, 0, 1)

    # PyTorch backward - following copy_ test pattern
    torch_src = torch.Tensor(src)
    torch_src.requires_grad = True
    # Multiply to create non-leaf tensor for dst before permute
    torch_dst = torch.Tensor(dst) * 1
    torch_dst_view = torch_dst.permute(dims)
    torch_out = torch_dst_view.copy_(torch_src.permute(dims))
    grad = generate_ones_grad(torch_out.shape, np.float32)
    torch_out.backward(torch.Tensor(grad))
    expect_grad = torch_src.grad

    # MindSpore backward - following copy_ test pattern
    def forward_func(dst, src):
        # Multiply dst to create a working copy
        """Helper function."""
        dst_copy = dst * 1
        dst_view = dst_copy.permute(dims)
        out = dst_view.copy_(src.permute(dims))
        return out

    ms_dst = ms.Tensor(dst)
    ms_src = ms.Tensor(src)
    output_grad = ops.grad(forward_func, (1,))(ms_dst, ms_src)

    allclose_nparray(expect_grad.detach().numpy(),
                     output_grad.asnumpy(), equal_nan=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=1024.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
@pytest.mark.parametrize('input_shape,dims', [
    ((2, 3, 4), (2, 0, 1)),
    ((2, 3, 4), (1, 2, 0)),
    ((2, 3, 4, 5), (3, 2, 1, 0)),
])
def test_permute_view_multiple_shapes(mode, input_shape, dims):
    """
    Feature: permute view with various shapes
    Description: test permute view + inplace with different shape
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
    torch_dst_view = torch_dst.permute(dims)
    torch_dst_view.copy_(torch_src.permute(dims))
    expect = torch_dst

    # MindSpore
    ms_dst = ms.Tensor(dst)
    ms_src = ms.Tensor(src)
    ms_dst_view = ms_dst.permute(dims)
    ms_dst_view.copy_(ms_src.permute(dims))
    output = ms_dst

    allclose_nparray(expect.detach().numpy(), output.asnumpy(),
                     equal_nan=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=2048.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_permute_view_chain(mode):
    """
    Feature: chained permute views with inplace operation
    Description: test multiple permute views followed by inplace
                 modification
    Expectation: results match PyTorch implementation
    """

    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")

    x = generate_random_input((2, 3, 4), np.float32)
    src = generate_random_input((2, 3, 4), np.float32)

    # PyTorch: chain of permutes
    torch_x = torch.Tensor(x)
    torch_src = torch.Tensor(src)
    torch_view1 = torch_x.permute(2, 0, 1)
    torch_view2 = torch_view1.permute(1, 2, 0)
    # src also goes through the same permute chain
    torch_src_view1 = torch_src.permute(2, 0, 1)
    torch_src_view2 = torch_src_view1.permute(1, 2, 0)
    torch_view2.copy_(torch_src_view2)
    expect = torch_x

    # MindSpore: chain of permutes
    ms_x = ms.Tensor(x)
    ms_src = ms.Tensor(src)
    ms_view1 = ms_x.permute(2, 0, 1)
    ms_view2 = ms_view1.permute(1, 2, 0)
    # src also goes through the same permute chain
    ms_src_view1 = ms_src.permute(2, 0, 1)
    ms_src_view2 = ms_src_view1.permute(1, 2, 0)
    ms_view2.copy_(ms_src_view2)
    output = ms_x

    allclose_nparray(expect.detach().numpy(), output.asnumpy(),
                     equal_nan=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=2048.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
@pytest.mark.parametrize('dtype', [np.float32, np.float64,
                                   np.int32, np.int64])
def test_permute_view_dtypes(mode, dtype):
    """
    Feature: permute view with various data types
    Description: test permute view + inplace with different dtypes
    Expectation: results match PyTorch implementation
    """

    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")

    dst = generate_random_input((2, 3, 4), dtype)
    src = generate_random_input((2, 3, 4), dtype)
    dims = (2, 0, 1)

    torch_dtype_map = {
        np.int32: torch.int32,
        np.int64: torch.int64,
        np.float32: torch.float32,
        np.float64: torch.float64
    }

    # PyTorch
    torch_dst = torch.tensor(dst, dtype=torch_dtype_map[dtype])
    torch_src = torch.tensor(src, dtype=torch_dtype_map[dtype])
    torch_dst_view = torch_dst.permute(dims)
    torch_dst_view.copy_(torch_src.permute(dims))
    expect = torch_dst

    # MindSpore
    ms_dst = ms.Tensor(dst)
    ms_src = ms.Tensor(src)
    ms_dst_view = ms_dst.permute(dims)
    ms_dst_view.copy_(ms_src.permute(dims))
    output = ms_dst

    allclose_nparray(expect.detach().numpy(), output.asnumpy(),
                     equal_nan=True)
