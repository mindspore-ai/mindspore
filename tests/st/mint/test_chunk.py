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
# pylint: disable=unused-variable
"""Tests for mint.chunk: input, chunks, dim."""
import pytest
import numpy as np
import torch
from tests.st.utils import test_utils
from tests.mark_utils import arg_mark

import mindspore as ms
from mindspore import mint, Tensor, jit, context, ops
from mindspore.common.api import _pynative_executor
from tools import allclose_nparray, needs_backward

# Set fixed random seed for reproducibility
np.random.seed(42)


def chunk_forward_func(x, chunks, dim):
    """MindSpore forward calculation function"""
    return mint.chunk(x, chunks, dim)


def chunk_backward_func(x, chunks, dim, chunk_idx=0):
    """MindSpore backward propagation function
    Args:
        x: input tensor
        chunks: number of chunks
        dim: dimension to chunk along
        chunk_idx: which chunk to compute gradient for (default: 0)
    """
    def forward_select_chunk(x):
        """Helper function."""
        chunks_result = mint.chunk(x, chunks, dim)
        return chunks_result[chunk_idx]
    grad_fn = ops.grad(forward_select_chunk, (0,))
    return grad_fn(x)


KBK_CHUNK_FORWARD = jit(chunk_forward_func, backend="ms_backend", jit_level="O0")
KBK_CHUNK_BACKWARD = jit(chunk_backward_func, backend="ms_backend", jit_level="O0")


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_chunk_forward_dynamic_shape(context_mode):
    """
    Feature: chunk ops.
    Description: test ops chunk with dynamic shape tensor input.
    Expectation: output the right result.
    """
    context.set_context(mode=context_mode)
    context.set_context(jit_level='O0')
    input_dyn = Tensor(shape=[4, None, None], dtype=ms.int64)
    chunks = 3
    dims = 0
    test_cell = test_utils.to_cell_obj(mint.chunk)
    test_cell.set_inputs(input_dyn, chunks, dims)
    input_tensor = Tensor(np.arange(60).reshape((4, 3, 5)).astype(np.int64))
    out = test_cell(input_tensor, chunks, dims)
    expect_output = [np.array(np.arange(30).reshape((2, 3, 5)), dtype=np.float32),
                     np.array(np.arange(30, 60).reshape((2, 3, 5)), dtype=np.float32)]
    for res, exp in zip(out, expect_output):
        assert np.allclose(res.asnumpy(), exp)

    input_tensor = Tensor(np.arange(24).reshape((4, 2, 3)).astype(np.int64))
    out = test_cell(input_tensor, chunks, dims)
    expect_output = [np.array(np.arange(12).reshape((2, 2, 3)), dtype=np.float32),
                     np.array(np.arange(12, 24).reshape((2, 2, 3)), dtype=np.float32)]
    for res, exp in zip(out, expect_output):
        assert np.allclose(res.asnumpy(), exp)

    if context_mode == ms.GRAPH_MODE:
        dims = 2
        test_cell.set_inputs(input_dyn, chunks, dims)
        input_tensor = Tensor(np.arange(24).reshape((4, 2, 3)).astype(np.int64))
        with pytest.raises(RuntimeError):
            _ = test_cell(input_tensor, chunks, dims)
            _pynative_executor.sync()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE])
def test_chunk_forward_dynamic_rank(context_mode):
    """
    Feature: chunk ops.
    Description: test ops chunk with dynamic shape tensor input.
    Expectation: output the right result.
    """
    context.set_context(mode=context_mode)
    input_dyn = Tensor(shape=None, dtype=ms.int64)
    chunks = 3
    dims = 0
    test_cell = test_utils.to_cell_obj(mint.chunk)
    test_cell.set_inputs(input_dyn, chunks, dims)
    input_tensor = Tensor(np.arange(24).reshape((4, 2, 3)).astype(np.int64))
    with pytest.raises(RuntimeError):
        _ = test_cell(input_tensor, chunks, dims)
        _pynative_executor.sync()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_chunk_backward_dynamic_shape(context_mode):
    """
    Feature: chunk ops.
    Description: test ops chunk with dynamic shape tensor input.
    Expectation: output the right result.
    """
    context.set_context(mode=context_mode)
    context.set_context(jit_level='O0')
    input_dyn = Tensor(shape=[None, 4, None], dtype=ms.float32)
    chunks = 3
    dims = 1
    test_cell = test_utils.to_cell_obj(ops.grad(mint.chunk, (0,)))
    test_cell.set_inputs(input_dyn, chunks, dims)
    input_tensor = Tensor(np.arange(60).reshape((3, 4, 5)).astype(np.float32))
    out = test_cell(input_tensor, chunks, dims)
    expect_output = np.ones((3, 4, 5))
    assert np.allclose(out.asnumpy(), expect_output)

    input_tensor = Tensor(np.arange(24).reshape((2, 4, 3)).astype(np.float32))
    out = test_cell(input_tensor, chunks, dims)
    expect_output = np.ones((2, 4, 3))
    assert np.allclose(out.asnumpy(), expect_output)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE])
def test_chunk_backward_dynamic_rank(context_mode):
    """
    Feature: chunk.
    Description: test mint chunk with dynamic shape tensor input.
    Expectation: output the right result.
    """
    context.set_context(mode=context_mode)
    input_dyn = Tensor(shape=None, dtype=ms.float64)
    chunks = 3
    dims = 1
    test_cell = test_utils.to_cell_obj(ops.grad(mint.chunk, (0,)))
    test_cell.set_inputs(input_dyn, chunks, dims)
    input_tensor = Tensor(np.arange(24).reshape((4, 2, 3)).astype(np.float64))
    with pytest.raises(RuntimeError):
        _ = test_cell(input_tensor, chunks, dims)
        _pynative_executor.sync()


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


def generate_ones_grad(shapes, dtype):
    """Generate gradient input with all ones for list of tensors"""
    return [np.ones(shape).astype(dtype) for shape in shapes]


def generate_expect_forward_output(x, chunks, dim):
    """Generate expected forward calculation results using PyTorch"""
    return torch.chunk(x, chunks, dim)


def generate_expect_backward_output(x, chunks, dim, grad_idx=0):
    """Generate expected backward propagation results using PyTorch
    Args:
        x: input tensor
        chunks: number of chunks
        dim: dimension to chunk along
        grad_idx: which chunk to compute gradient for (default: 0)
    """
    x.requires_grad = True
    outs = torch.chunk(x, chunks, dim)
    # Compute gradient for the specified chunk
    outs[grad_idx].backward(torch.ones_like(outs[grad_idx]))
    dx = x.grad
    return dx


def chunk_forward(mode, x, chunks, dim):
    """Chunk forward wrapper.

    In KBK mode, this wrapper uses jit-wrapped callable.
    """
    if mode == 'pynative':
        return chunk_forward_func(x, chunks, dim)
    return KBK_CHUNK_FORWARD(x, chunks, dim)


def chunk_backward(mode, x, chunks, dim, chunk_idx=0):
    """Chunk backward wrapper.

    In KBK mode, this wrapper uses jit-wrapped callable.
    """
    if mode == 'pynative':
        return chunk_backward_func(x, chunks, dim, chunk_idx=chunk_idx)
    return KBK_CHUNK_BACKWARD(x, chunks, dim, chunk_idx=chunk_idx)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_chunk_std(mode):
    """
    Feature: standard forward, backward features.
    Description: test standard cases for chunk.
    Expectation: expect correct result.
    """

    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")

    x = generate_random_input((6, 4, 8), np.float32)
    chunks = 3
    dim = 0
    torch_x = torch.Tensor(x)
    expect = generate_expect_forward_output(torch_x, chunks, dim)

    ms_x = ms.Tensor(x)
    output = chunk_forward(mode, ms_x, chunks, dim)

    # Compare each chunk
    assert len(output) == len(expect)
    for out_chunk, exp_chunk in zip(output, expect):
        allclose_nparray(exp_chunk.detach().numpy(), out_chunk.asnumpy(),
                         equal_nan=True)

    # Test backward
    expect_grad = generate_expect_backward_output(torch_x, chunks, dim)
    output_grad = chunk_backward(mode, ms_x, chunks, dim)

    allclose_nparray(expect_grad.detach().numpy(), output_grad.asnumpy(),
                     equal_nan=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
@pytest.mark.parametrize('dtype', [
    np.float16, np.float32, np.float64,
    np.int32, np.int64
])
def test_chunk_dtype_coverage(mode, dtype):
    """
    Feature: dtype coverage for chunk operator.
    Description: test chunk with various dtypes.
    Expectation: results match PyTorch implementation with appropriate
                 precision.
    """

    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")

    x = generate_random_input((8, 6), dtype)
    chunks = 2
    dim = 1

    # Handle type mapping supported by PyTorch and MindSpore
    torch_dtype_map = {
        np.int32: torch.int32,
        np.int64: torch.int64,
        np.float16: torch.float16,
        np.float32: torch.float32,
        np.float64: torch.float64
    }
    torch_x = torch.tensor(x, dtype=torch_dtype_map.get(dtype, torch.float32))
    expect = generate_expect_forward_output(torch_x, chunks, dim)
    ms_x = ms.Tensor(x)
    output = chunk_forward(mode, ms_x, chunks, dim)

    assert len(output) == len(expect)
    for out_chunk, exp_chunk in zip(output, expect):
        allclose_nparray(exp_chunk.detach().numpy(), out_chunk.asnumpy(),
                         equal_nan=True)

    # Test backward only for float types
    if np.issubdtype(dtype, np.floating) and dtype != np.float16:
        expect_grad = generate_expect_backward_output(
            torch_x, chunks, dim)
        output_grad = chunk_backward(mode, ms_x, chunks, dim)
        allclose_nparray(expect_grad.detach().numpy(),
                         output_grad.asnumpy(), equal_nan=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
@pytest.mark.parametrize('input_shape,chunks,dim', [
    ((12,), 3, 0),  # 1D tensor
    ((6, 8), 2, 0),  # 2D tensor, split along dim 0
    ((6, 8), 4, 1),  # 2D tensor, split along dim 1
    ((4, 6, 8), 2, 0),  # 3D tensor, split along dim 0
    ((4, 6, 8), 3, 1),  # 3D tensor, split along dim 1
    ((4, 6, 8), 4, 2),  # 3D tensor, split along dim 2
    ((2, 3, 4, 6), 2, 3),  # 4D tensor
    ((10,), 3, 0),  # Non-divisible case
    ((7, 5), 2, 0),  # Non-divisible case
])
def test_chunk_dimensions(mode, input_shape, chunks, dim):
    """
    Feature: dimension coverage for chunk operator.
    Description: test chunk with various dimension configurations.
    Expectation: results match PyTorch implementation.
    """

    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")

    dtype = np.float32
    x = generate_random_input(input_shape, dtype)
    torch_x = torch.tensor(x, dtype=torch.float32)
    expect = generate_expect_forward_output(torch_x, chunks, dim)

    ms_x = ms.Tensor(x, dtype=ms.float32)
    output = chunk_forward(mode, ms_x, chunks, dim)

    assert len(output) == len(expect)
    for out_chunk, exp_chunk in zip(output, expect):
        allclose_nparray(exp_chunk.detach().numpy(), out_chunk.asnumpy(),
                         equal_nan=True)

    # Test backward
    expect_grad = generate_expect_backward_output(
        torch_x, chunks, dim)
    output_grad = chunk_backward(mode, ms_x, chunks, dim)
    allclose_nparray(expect_grad.detach().numpy(),
                     output_grad.asnumpy(), equal_nan=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
@pytest.mark.parametrize('special_type',
                         ['inf', 'nan', 'zero', 'large', 'small'])
def test_chunk_special_values(mode, special_type):
    """
    Feature: special value handling for chunk operator.
    Description: test chunk with inf, nan, zero, large and small values.
    Expectation: results match PyTorch implementation.
    """

    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")

    x = generate_special_input((6, 4, 8), np.float32, special_type)
    chunks = 2
    dim = 0

    torch_x = torch.Tensor(x)
    expect = generate_expect_forward_output(torch_x, chunks, dim)

    ms_x = ms.Tensor(x)
    output = chunk_forward(mode, ms_x, chunks, dim)

    assert len(output) == len(expect)
    for out_chunk, exp_chunk in zip(output, expect):
        allclose_nparray(exp_chunk.detach().numpy(), out_chunk.asnumpy(),
                         equal_nan=True)

    # Test backward
    expect_grad = generate_expect_backward_output(torch_x, chunks, dim)
    output_grad = chunk_backward(mode, ms_x, chunks, dim)
    allclose_nparray(expect_grad.detach().numpy(),
                     output_grad.asnumpy(), equal_nan=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_chunk_empty_tensor(mode):
    """
    Feature: empty tensor handling for chunk operator.
    Description: test chunk with empty tensor input.
    Expectation: results match PyTorch implementation.
    """

    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")

    # Test empty tensors
    test_cases = [
        ((0, 3), 2, 0),
        ((4, 0), 2, 1),
        ((0, 3, 4), 2, 0)
    ]
    for shape, chunks, dim in test_cases:
        x = generate_random_input(shape, np.float32)
        torch_x = torch.Tensor(x)
        expect = generate_expect_forward_output(torch_x, chunks, dim)

        ms_x = ms.Tensor(x)
        output = chunk_forward(mode, ms_x, chunks, dim)

        assert len(output) == len(expect)
        for out_chunk, exp_chunk in zip(output, expect):
            assert out_chunk.shape == exp_chunk.shape
            allclose_nparray(exp_chunk.detach().numpy(), out_chunk.asnumpy(),
                             equal_nan=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_chunk_negative_dim(mode):
    """
    Feature: negative dimension support for chunk operator.
    Description: test chunk with negative dimension indices.
    Expectation: results match PyTorch implementation.
    """

    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")

    dtype = np.float32
    x = generate_random_input((4, 6, 8), dtype)
    chunks = 2

    test_dims = [-1, -2, -3]  # Test negative dimensions

    for dim in test_dims:
        torch_x = torch.tensor(x, dtype=torch.float32)
        expect = generate_expect_forward_output(torch_x, chunks, dim)

        ms_x = ms.Tensor(x, dtype=ms.float32)
        output = chunk_forward(mode, ms_x, chunks, dim)

        assert len(output) == len(expect)
        for out_chunk, exp_chunk in zip(output, expect):
            allclose_nparray(exp_chunk.detach().numpy(), out_chunk.asnumpy(),
                             equal_nan=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_chunk_zero_bias(mode):
    """
    Feature: zero bias with torch_cpu.
    Description: test zero bias between mint.chunk and torch_cpu results.
    Expectation: complete consistency with torch_cpu results.
    """

    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")

    x = generate_random_input((6, 8, 4), np.float32)
    chunks = 3
    dim = 0

    torch_x = torch.tensor(x)
    expect = generate_expect_forward_output(torch_x, chunks, dim)

    ms_x = ms.Tensor(x)
    output = chunk_forward(mode, ms_x, chunks, dim)

    # Verify zero bias with strict precision requirements
    assert len(output) == len(expect)
    for out_chunk, exp_chunk in zip(output, expect):
        allclose_nparray(exp_chunk.detach().numpy(), out_chunk.asnumpy(),
                         equal_nan=True)

    # Verify backward propagation zero bias
    expect_grad = generate_expect_backward_output(torch_x, chunks, dim)
    output_grad = chunk_backward(mode, ms_x, chunks, dim)
    allclose_nparray(expect_grad.detach().numpy(),
                     output_grad.asnumpy(), equal_nan=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_chunk_non_contiguous(mode):
    """
    Feature: non-contiguous input support for chunk operator.
    Description: test chunk with non-contiguous tensor input.
    Expectation: results match PyTorch implementation.
    """

    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")

    dtype = np.float32
    x = generate_random_input((4, 6, 8), dtype)
    # Create non-contiguous tensor (via transpose)
    x_transposed = np.transpose(x, (1, 0, 2))

    chunks = 2
    dim = 0

    pt_x = torch.tensor(x_transposed)
    expect = generate_expect_forward_output(pt_x, chunks, dim)

    ms_x = ms.Tensor(x_transposed)
    output = chunk_forward(mode, ms_x, chunks, dim)

    assert len(output) == len(expect)
    for out_chunk, exp_chunk in zip(output, expect):
        allclose_nparray(exp_chunk.detach().numpy(), out_chunk.asnumpy(),
                         equal_nan=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_chunk_functional_interface(mode):
    """
    Feature: Functional interface test
    Description: Test mint.chunk functional interface
    Expectation: Expect correct results
    """
    dtype = np.float32
    x = generate_random_input((6, 4, 8), dtype)
    chunks = 3
    dim = 0

    ms_x = ms.tensor(x)
    pt_x = torch.tensor(x)

    if mode == 'pynative':
        context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")

    # Test functional interface
    ms_output = chunk_forward(mode, ms_x, chunks, dim)
    expect = generate_expect_forward_output(pt_x, chunks, dim)

    assert len(ms_output) == len(expect)
    for out_chunk, exp_chunk in zip(ms_output, expect):
        allclose_nparray(exp_chunk.detach().numpy(), out_chunk.asnumpy(),
                         equal_nan=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_chunk_large_tensors(mode):
    """
    Feature: Large tensor chunk test
    Description: Test mint.chunk handling of large tensors
    Expectation: Expect correct results
    """

    if mode == 'pynative':
        context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")

    dtype = np.float32
    x = generate_random_input((100, 50), dtype)
    chunks = 5
    dim = 0

    ms_tensor = ms.tensor(x)
    pt_tensor = torch.tensor(x)

    expect_forward = generate_expect_forward_output(pt_tensor, chunks, dim)
    ms_forward = chunk_forward(mode, ms_tensor, chunks, dim)

    assert len(ms_forward) == len(expect_forward)
    for out_chunk, exp_chunk in zip(ms_forward, expect_forward):
        allclose_nparray(exp_chunk.detach().numpy(),
                         out_chunk.asnumpy(), equal_nan=True)

    if needs_backward(dtype):
        expect_backward = generate_expect_backward_output(
            pt_tensor, chunks, dim)
        ms_grad = chunk_backward(mode, ms_tensor, chunks, dim)
        allclose_nparray(expect_backward.detach().numpy(),
                         ms_grad.asnumpy(), equal_nan=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_chunk_high_dimension(mode):
    """
    Feature: High dimension chunk test
    Description: Test mint.chunk handling of high-dimensional tensors
    Expectation: Expect correct results
    """

    if mode == 'pynative':
        context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")

    dtype = np.float32
    x = generate_random_input((8, 6, 4, 10), dtype)
    chunks = 2
    dim = 2

    ms_tensor = ms.tensor(x)
    pt_tensor = torch.tensor(x)

    expect_forward = generate_expect_forward_output(pt_tensor, chunks, dim)
    ms_output_forward = chunk_forward(mode, ms_tensor, chunks, dim)

    assert len(ms_output_forward) == len(expect_forward)
    for out_chunk, exp_chunk in zip(ms_output_forward, expect_forward):
        allclose_nparray(exp_chunk.detach().numpy(),
                         out_chunk.asnumpy(), equal_nan=True)

    if needs_backward(dtype):
        expect_backward = generate_expect_backward_output(
            pt_tensor, chunks, dim)
        ms_grad = chunk_backward(mode, ms_tensor, chunks, dim)
        allclose_nparray(expect_backward.detach().numpy(),
                         ms_grad.asnumpy(), equal_nan=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_chunk_complex_dtypes(mode, dtype):
    """
    Feature: complex dtype support for chunk operator.
    Description: test chunk with complex dtypes.
    Expectation: results match PyTorch implementation.
    """

    if mode == 'pynative':
        context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")

    x = generate_random_input((6, 8), dtype)
    chunks = 3
    dim = 0

    torch_x = torch.tensor(x)
    expect = torch.chunk(torch_x, chunks, dim)

    ms_x = ms.Tensor(x)
    output = chunk_forward(mode, ms_x, chunks, dim)

    assert len(output) == len(expect)
    for out_chunk, exp_chunk in zip(output, expect):
        output_np = out_chunk.asnumpy()
        allclose_nparray(exp_chunk.detach().numpy(), output_np,
                         equal_nan=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_chunk_single_chunk(mode):
    """
    Feature: single chunk case
    Description: test chunk when chunks=1 (no splitting)
    Expectation: results match PyTorch implementation
    """

    if mode == 'pynative':
        context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")

    dtype = np.float32
    x = generate_random_input((6, 8, 4), dtype)
    chunks = 1
    dim = 0

    torch_x = torch.tensor(x)
    expect = generate_expect_forward_output(torch_x, chunks, dim)
    ms_x = ms.Tensor(x)
    output = chunk_forward(mode, ms_x, chunks, dim)

    assert len(output) == len(expect) == 1
    allclose_nparray(expect[0].detach().numpy(), output[0].asnumpy(),
                     equal_nan=True)

    # Test backward
    expect_grad = generate_expect_backward_output(
        torch_x, chunks, dim)
    output_grad = chunk_backward(mode, ms_x, chunks, dim)
    allclose_nparray(expect_grad.detach().numpy(),
                     output_grad.asnumpy(), equal_nan=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_chunk_more_chunks_than_size(mode):
    """
    Feature: chunks greater than dimension size
    Description: test chunk when chunks > size along dimension
    Expectation: results match PyTorch implementation
    """

    if mode == 'pynative':
        context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")

    dtype = np.float32
    x = generate_random_input((3, 4), dtype)
    chunks = 5  # More chunks than size along dim 0
    dim = 0

    torch_x = torch.tensor(x)
    expect = generate_expect_forward_output(torch_x, chunks, dim)
    ms_x = ms.Tensor(x)
    output = chunk_forward(mode, ms_x, chunks, dim)

    assert len(output) == len(expect)
    for out_chunk, exp_chunk in zip(output, expect):
        allclose_nparray(exp_chunk.detach().numpy(), out_chunk.asnumpy(),
                         equal_nan=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_chunk_inplace_view(mode):
    """
    Feature: chunk view with inplace operation
    Description: test chunk as view followed by inplace modification
    Expectation: results match PyTorch implementation
    """

    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")

    # Test chunk view + copy_ inplace operation
    x = generate_random_input((6, 8), np.float32)
    src_data = [generate_random_input((3, 8), np.float32) for _ in range(2)]
    chunks = 2
    dim = 0

    # PyTorch: chunk creates views, then inplace copy on each chunk
    torch_x = torch.Tensor(x)
    torch_chunks = torch.chunk(torch_x, chunks, dim)
    for i, chunk in enumerate(torch_chunks):
        chunk.copy_(torch.Tensor(src_data[i]))
    expect = torch_x  # Original tensor should be modified

    # MindSpore: chunk creates views, then inplace copy on each chunk
    ms_x = ms.Tensor(x)
    ms_chunks = mint.chunk(ms_x, chunks, dim)
    for i, chunk in enumerate(ms_chunks):
        chunk.copy_(ms.Tensor(src_data[i]))
    output = ms_x  # Original tensor should be modified

    allclose_nparray(expect.detach().numpy(), output.asnumpy(),
                     equal_nan=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
@pytest.mark.parametrize('input_shape,chunks,dim', [
    ((12,), 3, 0),
    ((6, 8), 2, 0),
    ((8, 12), 4, 1),
])
def test_chunk_view_multiple_shapes(mode, input_shape, chunks, dim):
    """
    Feature: chunk view with various shapes
    Description: test chunk view + inplace with different shape combinations
    Expectation: results match PyTorch implementation
    """

    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")

    dtype = np.float32
    x = generate_random_input(input_shape, dtype)

    # PyTorch: get chunk shapes first
    torch_x_temp = torch.tensor(x, dtype=torch.float32)
    torch_chunks_temp = torch.chunk(torch_x_temp, chunks, dim)
    src_data = [generate_random_input(chunk.shape, dtype)
                for chunk in torch_chunks_temp]

    # PyTorch
    torch_x = torch.tensor(x, dtype=torch.float32)
    torch_chunks = torch.chunk(torch_x, chunks, dim)
    for i, chunk in enumerate(torch_chunks):
        chunk.copy_(torch.tensor(src_data[i]))
    expect = torch_x

    # MindSpore
    ms_x = ms.Tensor(x, dtype=ms.float32)
    ms_chunks = mint.chunk(ms_x, chunks, dim)
    for i, chunk in enumerate(ms_chunks):
        chunk.copy_(ms.Tensor(src_data[i]))
    output = ms_x

    allclose_nparray(expect.detach().numpy(), output.asnumpy(),
                     equal_nan=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_chunk_view_chain(mode):
    """
    Feature: chained chunk views with inplace operation
    Description: test chunk followed by another operation and inplace
                 modification
    Expectation: results match PyTorch implementation
    """

    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")

    x = generate_random_input((6, 8), np.float32)
    src1 = generate_random_input((3, 8), np.float32)
    src2 = generate_random_input((3, 8), np.float32)

    # PyTorch: chunk then modify first chunk
    torch_x = torch.Tensor(x)
    torch_chunks = torch.chunk(torch_x, 2, 0)
    torch_chunks[0].copy_(torch.Tensor(src1))
    torch_chunks[1].copy_(torch.Tensor(src2))
    expect = torch_x

    # MindSpore: chunk then modify first chunk
    ms_x = ms.Tensor(x)
    ms_chunks = mint.chunk(ms_x, 2, 0)
    ms_chunks[0].copy_(ms.Tensor(src1))
    ms_chunks[1].copy_(ms.Tensor(src2))
    output = ms_x

    allclose_nparray(expect.detach().numpy(), output.asnumpy(),
                     equal_nan=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
@pytest.mark.parametrize('dtype', [np.float32, np.float64, np.int32, np.int64])
def test_chunk_view_dtypes(mode, dtype):
    """
    Feature: chunk view with various data types
    Description: test chunk view + inplace with different dtypes
    Expectation: results match PyTorch implementation
    """

    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")

    x = generate_random_input((6, 8), dtype)
    src_data = [generate_random_input((3, 8), dtype) for _ in range(2)]
    chunks = 2
    dim = 0

    torch_dtype_map = {
        np.int32: torch.int32,
        np.int64: torch.int64,
        np.float32: torch.float32,
        np.float64: torch.float64
    }

    # PyTorch
    torch_x = torch.tensor(x, dtype=torch_dtype_map[dtype])
    torch_chunks = torch.chunk(torch_x, chunks, dim)
    for i, chunk in enumerate(torch_chunks):
        chunk.copy_(torch.tensor(src_data[i], dtype=torch_dtype_map[dtype]))
    expect = torch_x

    # MindSpore
    ms_x = ms.Tensor(x)
    ms_chunks = mint.chunk(ms_x, chunks, dim)
    for i, chunk in enumerate(ms_chunks):
        chunk.copy_(ms.Tensor(src_data[i]))
    output = ms_x

    allclose_nparray(expect.detach().numpy(), output.asnumpy(),
                     equal_nan=True)
