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
"""reshape op test case"""
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


def generate_expect_forward_output(x, new_shape):
    """Generate expected forward calculation results using PyTorch"""
    return torch.reshape(x, new_shape)


def generate_expect_backward_output(x, new_shape, grad):
    """Generate expected backward propagation results using PyTorch"""
    x.requires_grad = True
    out = torch.reshape(x, new_shape)
    out.backward(grad)
    dx = x.grad
    return dx


def reshape_forward_func(x, new_shape):
    """MindSpore forward calculation function"""
    return mint.reshape(x, new_shape)


def reshape_backward_func(x, new_shape):
    """MindSpore backward propagation function"""
    return ops.grad(reshape_forward_func, (0,))(x, new_shape)


KBK_RESHAPE_FORWARD = jit(reshape_forward_func, backend="ms_backend", jit_level="O0")
KBK_RESHAPE_BACKWARD = jit(reshape_backward_func, backend="ms_backend", jit_level="O0")


def reshape_forward(mode, x, new_shape):
    """Reshape forward wrapper.

    In KBK mode, this wrapper uses jit-wrapped callable.
    """
    if mode == 'pynative':
        return reshape_forward_func(x, new_shape)
    return KBK_RESHAPE_FORWARD(x, new_shape)


def reshape_backward(mode, x, new_shape):
    """Reshape backward wrapper.

    In KBK mode, this wrapper uses jit-wrapped callable.
    """
    if mode == 'pynative':
        return reshape_backward_func(x, new_shape)
    return KBK_RESHAPE_BACKWARD(x, new_shape)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=1024.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_reshape_std(mode):
    """
    Feature: standard forward, backward features.
    Description: test standard cases for reshape.
    Expectation: expect correct result.
    """

    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")

    x = generate_random_input((2, 3, 4), np.float32)
    new_shape = (6, 4)
    expect = generate_expect_forward_output(torch.Tensor(x), new_shape)
    grad = generate_ones_grad(expect.shape, expect.numpy().dtype)
    expect_grad = generate_expect_backward_output(
        torch.Tensor(x), new_shape, torch.Tensor(grad))
    output = reshape_forward(mode, ms.Tensor(x), new_shape)
    output_grad = reshape_backward(mode, ms.Tensor(x), new_shape)

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
def test_reshape_dtype_coverage(mode, dtype):
    """
    Feature: dtype coverage for reshape operator.
    Description: test reshape with various dtypes.
    Expectation: results match PyTorch implementation with appropriate
                 precision.
    """

    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    x = generate_random_input((3, 4), dtype)
    new_shape = (2, 6)
    # Handle type mapping supported by PyTorch and MindSpore
    torch_dtype_map = {
        np.int32: torch.int32,
        np.int64: torch.int64,
        np.float16: torch.float16,
        np.float32: torch.float32,
        np.float64: torch.float64
    }
    torch_x = torch.tensor(x, dtype=torch_dtype_map.get(dtype, torch.float32))
    expect = generate_expect_forward_output(torch_x, new_shape)
    ms_x = ms.Tensor(x)
    output = reshape_forward(mode, ms_x, new_shape)
    allclose_nparray(expect.detach().numpy(), output.asnumpy(),
                     equal_nan=True)
    # Test backward only for float types
    if np.issubdtype(dtype, np.floating) and dtype != np.float16:
        grad = generate_ones_grad(expect.shape, expect.numpy().dtype)
        expect_grad = generate_expect_backward_output(
            torch_x, new_shape, torch.tensor(grad))
        output_grad = reshape_backward(mode, ms_x, new_shape)
        allclose_nparray(expect_grad.detach().numpy(),
                         output_grad.asnumpy(), equal_nan=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=1024.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
@pytest.mark.parametrize('input_shape,new_shape', [
    ((24,), (2, 3, 4)),  # 1D to 3D
    ((2, 3, 4), (24,)),  # 3D to 1D
    ((2, 3, 4), (6, 4)),  # 3D to 2D
    ((6, 4), (2, 3, 4)),  # 2D to 3D
    ((2, 2, 3, 4), (4, 12)),  # 4D to 2D
    ((2, 3, 4, 5), (6, 20)),  # 4D to 2D
    ((2, 3, 4, 5, 6), (6, 120)),  # 5D to 2D
    ((120,), (2, 3, 4, 5)),  # 1D to 4D
    ((24,), (2, 2, 2, 3)),  # 1D to 4D
])
def test_reshape_dimensions(mode, input_shape, new_shape):
    """
    Feature: dimension coverage for reshape operator.
    Description: test reshape with various dimension transformations.
    Expectation: results match PyTorch implementation.
    """

    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")

    dtype = np.float32
    x = generate_random_input(input_shape, dtype)
    torch_x = torch.tensor(x, dtype=torch.float32)
    expect = generate_expect_forward_output(torch_x, new_shape)
    grad = generate_ones_grad(expect.shape, dtype)
    torch_grad = torch.tensor(grad, dtype=torch.float32)
    expect_grad = generate_expect_backward_output(
        torch_x, new_shape, torch_grad)
    ms_x = ms.Tensor(x, dtype=ms.float32)
    output = reshape_forward(mode, ms_x, new_shape)
    output_grad = reshape_backward(mode, ms_x, new_shape)
    allclose_nparray(expect.detach().numpy(), output.asnumpy(),
                     equal_nan=True)
    allclose_nparray(expect_grad.detach().numpy(),
                     output_grad.asnumpy(), equal_nan=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=1024.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
@pytest.mark.parametrize('special_type',
                         ['inf', 'nan', 'zero', 'large', 'small'])
def test_reshape_special_values(mode, special_type):
    """
    Feature: special value handling for reshape operator.
    Description: test reshape with inf, nan, zero, large and small values.
    Expectation: results match PyTorch implementation.
    """

    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    x = generate_special_input((2, 3, 4), np.float32, special_type)
    new_shape = (6, 4)
    expect = generate_expect_forward_output(torch.Tensor(x), new_shape)
    grad = generate_ones_grad(expect.shape, expect.numpy().dtype)
    expect_grad = generate_expect_backward_output(
        torch.Tensor(x), new_shape, torch.Tensor(grad))
    ms_x = ms.Tensor(x)
    output = reshape_forward(mode, ms_x, new_shape)
    output_grad = reshape_backward(mode, ms_x, new_shape)
    allclose_nparray(expect.detach().numpy(), output.asnumpy(),
                     equal_nan=True)
    allclose_nparray(expect_grad.detach().numpy(),
                     output_grad.asnumpy(), equal_nan=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=1024.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_reshape_empty_tensor(mode):
    """
    Feature: empty tensor handling for reshape operator.
    Description: test reshape with empty tensor input.
    Expectation: results match PyTorch implementation.
    """

    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    # Test empty tensors
    test_cases = [
        ((0,), (0, 3)),
        ((2, 0), (0,)),
        ((0, 3, 4), (0, 12))
    ]
    for shape, new_shape in test_cases:
        x = generate_random_input(shape, np.float32)
        expect = generate_expect_forward_output(torch.Tensor(x), new_shape)
        ms_x = ms.Tensor(x)
        output = reshape_forward(mode, ms_x, new_shape)
        assert output.shape == expect.shape
        allclose_nparray(expect.detach().numpy(), output.asnumpy(),
                         equal_nan=True)
        # Test backward propagation
        grad = generate_ones_grad(expect.shape, expect.numpy().dtype)
        expect_grad = generate_expect_backward_output(
            torch.Tensor(x), new_shape, torch.Tensor(grad))
        output_grad = reshape_backward(mode, ms_x, new_shape)
        allclose_nparray(expect_grad.detach().numpy(),
                         output_grad.asnumpy(), equal_nan=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=1024.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
@pytest.mark.parametrize('batch_size', [8, 16, 32, 64, 128])
def test_reshape_vmap(mode, batch_size):
    """
    Feature: vmap support for reshape operator.
    Description: test reshape with vmap for batch processing.
    Expectation: results match PyTorch implementation.
    """

    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    # Prepare batch data (batch_size, 2, 3)
    x = generate_random_input((batch_size, 2, 3), np.float32)
    torch_x = torch.tensor(x)
    new_shape = (6,)
    # PyTorch doesn't have direct vmap for reshape, so we test each batch
    expect_list = []
    for i in range(batch_size):
        expect_i = generate_expect_forward_output(torch_x[i], new_shape)
        expect_list.append(expect_i.numpy())
    expect = np.stack(expect_list)
    # Use vmap for batch processing

    def vmap_func(x):
        """Helper function."""
        return reshape_forward_func(x, new_shape)

    vmap_reshape = vmap(vmap_func, in_axes=0)
    if mode == 'KBK':
        vmap_reshape = jit(vmap_reshape, backend="ms_backend", jit_level="O0")
    ms_x = ms.Tensor(x)
    output = vmap_reshape(ms_x)
    allclose_nparray(expect, output.asnumpy(), equal_nan=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=1024.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
@pytest.mark.parametrize('has_dynamic_rank', [False, True])
def test_reshape_dynamic_shape_advanced(mode, has_dynamic_rank):
    """
    Feature: dynamic shape and rank support for reshape operator.
    Description: test reshape with dynamic dimensions and dynamic rank inputs.
    Expectation: results match PyTorch implementation and shape info is
                 correctly handled.
    """

    if mode == 'pynative':
        context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    base_shapes = [
        ((12,), (3, 4)),
        ((2, 6), (3, 4)),
        ((2, 3, 4), (6, 4)),
    ]
    if has_dynamic_rank:
        base_shapes.extend([
            ((24,), (2, 3, 4)),
            ((2, 3, 4, 5), (10, 12))
        ])
    for input_shape, new_shape in base_shapes:
        x_np = generate_random_input(input_shape, np.float32)
        x_torch = torch.Tensor(x_np)
        expect_forward = generate_expect_forward_output(x_torch, new_shape)
        x_ms = ms.Tensor(x_np)
        output_forward = reshape_forward(mode, x_ms, new_shape)
        allclose_nparray(expect_forward.detach().numpy(),
                         output_forward.asnumpy(), equal_nan=True)
        assert output_forward.shape == new_shape, \
            (f"Output shape mismatch: expected {new_shape}, "
             f"got {output_forward.shape}")
        grad_np = generate_ones_grad(new_shape, np.float32)
        expect_backward = generate_expect_backward_output(
            x_torch, new_shape, torch.Tensor(grad_np))
        output_backward = reshape_backward(mode, x_ms, new_shape)
        allclose_nparray(expect_backward.detach().numpy(),
                         output_backward.asnumpy(), equal_nan=True)
        assert output_backward.shape == input_shape, \
            (f"Gradient shape mismatch: expected {input_shape}, "
             f"got {output_backward.shape}")


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=1024.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_reshape_infer_dimension(mode):
    """
    Feature: inferred dimension support for reshape operator.
    Description: test reshape with -1 dimension inference.
    Expectation: results match PyTorch implementation.
    """

    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")

    dtype = np.float32
    test_cases = [
        ((2, 3, 4), (-1, 4)),  # Infer first dimension
        ((2, 3, 4), (2, -1)),  # Infer last dimension
        ((2, 3, 4), (6, -1)),  # Infer middle dimension
        ((24,), (-1, 6)),      # 1D to 2D with inference
        ((2, 3, 4, 5), (-1, 20)),  # 4D to 2D with inference
    ]

    for input_shape, new_shape in test_cases:
        x = generate_random_input(input_shape, dtype)
        torch_x = torch.tensor(x, dtype=torch.float32)
        expect = generate_expect_forward_output(torch_x, new_shape)
        ms_x = ms.Tensor(x, dtype=ms.float32)
        output = reshape_forward(mode, ms_x, new_shape)
        allclose_nparray(expect.detach().numpy(), output.asnumpy(),
                         equal_nan=True)

        # Test backward
        grad = generate_ones_grad(expect.shape, dtype)
        expect_grad = generate_expect_backward_output(
            torch_x, new_shape, torch.tensor(grad))
        output_grad = reshape_backward(mode, ms_x, new_shape)
        allclose_nparray(expect_grad.detach().numpy(),
                         output_grad.asnumpy(), equal_nan=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=1024.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_reshape_zero_bias(mode):
    """
    Feature: zero bias with torch_cpu.
    Description: test zero bias between mint.reshape and torch_cpu results.
    Expectation: complete consistency with torch_cpu results.
    """

    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    x = generate_random_input((2, 3, 4), np.float32)
    new_shape = (6, 4)
    torch_x = torch.tensor(x)
    expect = generate_expect_forward_output(torch_x, new_shape)
    ms_x = ms.Tensor(x)
    output = reshape_forward(mode, ms_x, new_shape)
    output_grad = reshape_backward(mode, ms_x, new_shape)
    # Verify zero bias with strict precision requirements
    allclose_nparray(expect.detach().numpy(), output.asnumpy(),
                     equal_nan=True)
    # Verify backward propagation zero bias
    grad = generate_ones_grad(expect.shape, np.float32)
    expect_grad = generate_expect_backward_output(
        torch_x, new_shape, torch.tensor(grad))
    allclose_nparray(expect_grad.detach().numpy(),
                     output_grad.asnumpy(), equal_nan=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=1024.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_reshape_non_contiguous(mode):
    """
    Feature: non-contiguous input support for reshape operator.
    Description: test reshape with non-contiguous tensor input.
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
    new_shape = (6, 2)
    pt_x = torch.tensor(x_contiguous)
    expect = generate_expect_forward_output(pt_x, new_shape)
    output = reshape_forward(mode, ms.Tensor(x_contiguous), new_shape)
    allclose_nparray(expect.detach().numpy(), output.asnumpy(),
                     equal_nan=True)
    # Non-contiguous input backward test
    grad = np.ones(expect.shape, dtype=expect.numpy().dtype)
    expect_grad = generate_expect_backward_output(
        pt_x, new_shape, torch.tensor(grad))
    output_grad = reshape_backward(mode, ms.Tensor(x_contiguous), new_shape)
    allclose_nparray(expect_grad.detach().numpy(),
                     output_grad.asnumpy(), equal_nan=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=1024.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_reshape_functional_interface(mode):
    """
    Feature: Functional interface test
    Description: Test mint.reshape functional interface
    Expectation: Expect correct results
    """
    dtype = np.float32
    x = generate_random_input((2, 3, 4), dtype)
    new_shape = (6, 4)
    ms_x = ms.tensor(x)
    pt_x = torch.tensor(x)

    if mode == 'pynative':
        context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    # Test functional interface
    ms_output = reshape_forward(mode, ms_x, new_shape)
    expect = generate_expect_forward_output(pt_x, new_shape)
    allclose_nparray(expect.detach().numpy(), ms_output.asnumpy(),
                     equal_nan=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=1024.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_reshape_jit_mode(mode):
    """
    Feature: JIT mode test
    Description: Test mint.reshape functionality in JIT mode
    Expectation: Expect correct results
    """

    if mode == 'pynative':
        context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    dtype = np.float32
    x = generate_random_input((2, 3, 4), dtype)
    new_shape = (6, 4)
    ms_x = ms.tensor(x)
    pt_x = torch.tensor(x)
    # Test forward
    ms_output = reshape_forward(mode, ms_x, new_shape)
    expect = generate_expect_forward_output(pt_x, new_shape)
    allclose_nparray(expect.detach().numpy(), ms_output.asnumpy(),
                     equal_nan=True)
    # Test backward
    if needs_backward(dtype):
        grad = generate_ones_grad(ms_output.shape, dtype)
        pt_grad = torch.tensor(grad)
        expect_grad = generate_expect_backward_output(
            pt_x, new_shape, pt_grad)
        ms_grad = reshape_backward(mode, ms_x, new_shape)
        allclose_nparray(expect_grad.detach().numpy(),
                         ms_grad.asnumpy(), equal_nan=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=1024.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_reshape_large_tensors(mode):
    """
    Feature: Large tensor reshape test
    Description: Test mint.reshape handling of large tensors
    Expectation: Expect correct results
    """

    if mode == 'pynative':
        context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    dtype = np.float32
    x = generate_random_input((100, 50), dtype)
    new_shape = (50, 100)
    ms_tensor = ms.tensor(x)
    pt_tensor = torch.tensor(x)
    expect_forward = generate_expect_forward_output(pt_tensor, new_shape)
    ms_forward = reshape_forward(mode, ms_tensor, new_shape)
    allclose_nparray(expect_forward.detach().numpy(),
                     ms_forward.asnumpy(), equal_nan=True)
    if needs_backward(dtype):
        grad = generate_ones_grad(expect_forward.shape, dtype)
        pt_grad = torch.tensor(grad)
        expect_backward = generate_expect_backward_output(pt_tensor, new_shape, pt_grad)
        ms_grad = reshape_backward(mode, ms_tensor, new_shape)
        allclose_nparray(expect_backward.detach().numpy(),
                         ms_grad.asnumpy(), equal_nan=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=1024.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_reshape_high_dimension(mode):
    """
    Feature: High dimension reshape test
    Description: Test mint.reshape handling of high-dimensional tensors
    Expectation: Expect correct results
    """

    if mode == 'pynative':
        context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    dtype = np.float32
    x = generate_random_input((2, 3, 4, 5), dtype)
    new_shape = (6, 20)
    ms_tensor = ms.tensor(x)
    pt_tensor = torch.tensor(x)
    expect_forward = generate_expect_forward_output(pt_tensor, new_shape)
    ms_output_forward = reshape_forward(mode, ms_tensor, new_shape)
    allclose_nparray(expect_forward.detach().numpy(),
                     ms_output_forward.asnumpy(), equal_nan=True)
    if needs_backward(dtype):
        grad = generate_ones_grad(expect_forward.shape, dtype)
        pt_grad = torch.tensor(grad)
        expect_backward = generate_expect_backward_output(pt_tensor, new_shape, pt_grad)
        ms_grad = reshape_backward(mode, ms_tensor, new_shape)
        allclose_nparray(expect_backward.detach().numpy(),
                         ms_grad.asnumpy(), equal_nan=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=1024.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_reshape_very_large_tensors(mode):
    """
    Feature: Very large tensor reshape test
    Description: Test mint.reshape handling of very large tensors
    Expectation: Expect correct results
    """
    dtype = np.float32
    x = generate_random_input((1000, 100), dtype)
    new_shape = (100, 1000)
    ms_tensor = ms.tensor(x)
    pt_tensor = torch.tensor(x)
    expect_forward = generate_expect_forward_output(pt_tensor, new_shape)
    if mode == 'pynative':
        context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    ms_output_forward = reshape_forward(mode, ms_tensor, new_shape)
    allclose_nparray(expect_forward.detach().numpy(), ms_output_forward.asnumpy(), equal_nan=True)
    if needs_backward(dtype):
        grad = generate_ones_grad(expect_forward.shape, dtype)
        pt_grad = torch.tensor(grad)
        expect_backward = generate_expect_backward_output(pt_tensor, new_shape, pt_grad)
        ms_grad = reshape_backward(mode, ms_tensor, new_shape)
        allclose_nparray(expect_backward.detach().numpy(), ms_grad.asnumpy(), equal_nan=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=1024.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_reshape_complex_dtypes(mode, dtype):
    """
    Feature: complex dtype support for reshape operator.
    Description: test reshape with complex dtypes.
    Expectation: results match PyTorch implementation.
    """

    if mode == 'pynative':
        context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    x = generate_random_input((3, 4), dtype)
    new_shape = (2, 6)
    torch_x = torch.tensor(x)
    expect = torch.reshape(torch_x, new_shape)
    ms_x = ms.Tensor(x)
    output = reshape_forward(mode, ms_x, new_shape)
    output_np = output.asnumpy()
    allclose_nparray(expect.detach().numpy(), output_np, equal_nan=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=1024.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_reshape_same_shape(mode):
    """
    Feature: reshape to same shape
    Description: test reshape when new shape equals original shape
    Expectation: results match PyTorch implementation
    """

    if mode == 'pynative':
        context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")

    dtype = np.float32
    x = generate_random_input((2, 3, 4), dtype)
    new_shape = (2, 3, 4)

    torch_x = torch.tensor(x)
    expect = generate_expect_forward_output(torch_x, new_shape)
    ms_x = ms.Tensor(x)
    output = reshape_forward(mode, ms_x, new_shape)

    allclose_nparray(expect.detach().numpy(), output.asnumpy(), equal_nan=True)

    # Test backward
    grad = generate_ones_grad(expect.shape, dtype)
    expect_grad = generate_expect_backward_output(
        torch_x, new_shape, torch.tensor(grad))
    output_grad = reshape_backward(mode, ms_x, new_shape)
    allclose_nparray(expect_grad.detach().numpy(),
                     output_grad.asnumpy(), equal_nan=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=1024.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_reshape_flatten(mode):
    """
    Feature: flatten operation via reshape
    Description: test flattening multi-dimensional tensors to 1D
    Expectation: results match PyTorch implementation
    """

    if mode == 'pynative':
        context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")

    dtype = np.float32
    test_cases = [
        ((2, 3, 4), (-1,)),
        ((5, 6, 7, 8), (-1,)),
        ((10, 10), (-1,))
    ]

    for input_shape, new_shape in test_cases:
        x = generate_random_input(input_shape, dtype)
        torch_x = torch.tensor(x)
        expect = generate_expect_forward_output(torch_x, new_shape)
        ms_x = ms.Tensor(x)
        output = reshape_forward(mode, ms_x, new_shape)

        allclose_nparray(expect.detach().numpy(), output.asnumpy(),
                         equal_nan=True)

        # Test backward
        grad = generate_ones_grad(expect.shape, dtype)
        expect_grad = generate_expect_backward_output(
            torch_x, new_shape, torch.tensor(grad))
        output_grad = reshape_backward(mode, ms_x, new_shape)
        allclose_nparray(expect_grad.detach().numpy(),
                         output_grad.asnumpy(), equal_nan=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=1024.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_reshape_inplace_view(mode):
    """
    Feature: reshape view with inplace operation
    Description: test reshape as view followed by inplace modification
    Expectation: results match PyTorch implementation
    """

    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")

    # Test reshape view + copy_ inplace operation
    dst = generate_random_input((2, 3, 4), np.float32)
    src = generate_random_input((2, 3, 4), np.float32)
    new_shape = (6, 4)

    # PyTorch: reshape creates a view, then inplace copy
    torch_dst = torch.Tensor(dst)
    torch_src = torch.Tensor(src)
    torch_dst_view = torch_dst.reshape(new_shape)
    torch_dst_view.copy_(torch_src.reshape(new_shape))
    expect = torch_dst  # Original tensor should be modified

    # MindSpore: reshape creates a view, then inplace copy
    ms_dst = ms.Tensor(dst)
    ms_src = ms.Tensor(src)
    ms_dst_view = ms_dst.reshape(new_shape)
    ms_dst_view.copy_(ms_src.reshape(new_shape))
    output = ms_dst  # Original tensor should be modified

    allclose_nparray(expect.detach().numpy(), output.asnumpy(),
                     equal_nan=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=1024.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_reshape_view_backward(mode):
    """
    Feature: reshape view with backward propagation
    Description: test gradient propagation through reshape view and inplace
    Expectation: results match PyTorch implementation
    """

    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")

    dst = generate_random_input((2, 3, 4), np.float32)
    src = generate_random_input((2, 3, 4), np.float32)
    new_shape = (6, 4)

    # PyTorch backward - following copy_ test pattern
    torch_src = torch.Tensor(src)
    torch_src.requires_grad = True
    # Multiply to create non-leaf tensor for dst before reshape
    torch_dst = torch.Tensor(dst) * 1
    torch_dst_view = torch_dst.reshape(new_shape)
    torch_out = torch_dst_view.copy_(torch_src.reshape(new_shape))
    grad = generate_ones_grad(torch_out.shape, np.float32)
    torch_out.backward(torch.Tensor(grad))
    expect_grad = torch_src.grad

    # MindSpore backward - following copy_ test pattern
    def forward_func(dst, src):
        # Multiply dst to create a working copy
        """Helper function."""
        dst_copy = dst * 1
        dst_view = dst_copy.reshape(new_shape)
        out = dst_view.copy_(src.reshape(new_shape))
        return out

    ms_dst = ms.Tensor(dst)
    ms_src = ms.Tensor(src)
    output_grad = ops.grad(forward_func, (1,))(ms_dst, ms_src)

    allclose_nparray(expect_grad.detach().numpy(),
                     output_grad.asnumpy(), equal_nan=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=1024.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
@pytest.mark.parametrize('input_shape,new_shape', [
    ((24,), (2, 3, 4)),
    ((2, 3, 4), (6, 4)),
    ((2, 2, 3, 4), (4, 12)),
])
def test_reshape_view_multiple_shapes(mode, input_shape, new_shape):
    """
    Feature: reshape view with various shapes
    Description: test reshape view + inplace with different shape combinations
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
    torch_dst_view = torch_dst.reshape(new_shape)
    torch_dst_view.copy_(torch_src.reshape(new_shape))
    expect = torch_dst

    # MindSpore
    ms_dst = ms.Tensor(dst)
    ms_src = ms.Tensor(src)
    ms_dst_view = ms_dst.reshape(new_shape)
    ms_dst_view.copy_(ms_src.reshape(new_shape))
    output = ms_dst

    allclose_nparray(expect.detach().numpy(), output.asnumpy(),
                     equal_nan=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=1024.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_reshape_view_chain(mode):
    """
    Feature: chained reshape views with inplace operation
    Description: test multiple reshape views followed by inplace modification
    Expectation: results match PyTorch implementation
    """

    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")

    x = generate_random_input((2, 3, 4), np.float32)
    src = generate_random_input((2, 3, 4), np.float32)

    # PyTorch: chain of reshapes
    torch_x = torch.Tensor(x)
    torch_src = torch.Tensor(src)
    torch_view1 = torch_x.reshape(6, 4)
    torch_view2 = torch_view1.reshape(2, 12)
    torch_view2.copy_(torch_src.reshape(2, 12))
    expect = torch_x

    # MindSpore: chain of reshapes
    ms_x = ms.Tensor(x)
    ms_src = ms.Tensor(src)
    ms_view1 = ms_x.reshape(6, 4)
    ms_view2 = ms_view1.reshape(2, 12)
    ms_view2.copy_(ms_src.reshape(2, 12))
    output = ms_x

    allclose_nparray(expect.detach().numpy(), output.asnumpy(),
                     equal_nan=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=1024.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
@pytest.mark.parametrize('dtype', [np.float32, np.float64, np.int32, np.int64])
def test_reshape_view_dtypes(mode, dtype):
    """
    Feature: reshape view with various data types
    Description: test reshape view + inplace with different dtypes
    Expectation: results match PyTorch implementation
    """

    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")

    dst = generate_random_input((2, 3, 4), dtype)
    src = generate_random_input((2, 3, 4), dtype)
    new_shape = (6, 4)

    torch_dtype_map = {
        np.int32: torch.int32,
        np.int64: torch.int64,
        np.float32: torch.float32,
        np.float64: torch.float64
    }

    # PyTorch
    torch_dst = torch.tensor(dst, dtype=torch_dtype_map[dtype])
    torch_src = torch.tensor(src, dtype=torch_dtype_map[dtype])
    torch_dst_view = torch_dst.reshape(new_shape)
    torch_dst_view.copy_(torch_src.reshape(new_shape))
    expect = torch_dst

    # MindSpore
    ms_dst = ms.Tensor(dst)
    ms_src = ms.Tensor(src)
    ms_dst_view = ms_dst.reshape(new_shape)
    ms_dst_view.copy_(ms_src.reshape(new_shape))
    output = ms_dst

    allclose_nparray(expect.detach().numpy(), output.asnumpy(),
                     equal_nan=True)
