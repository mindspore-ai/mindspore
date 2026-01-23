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
"""Tests for mint.narrow: input, dim, start, length."""
import numpy as np
import pytest
import torch
import mindspore as ms
from mindspore import ops, mint, jit
from tests.mark_utils import arg_mark
from tools import allclose_nparray, needs_backward


np.random.seed(42)


def narrow_forward(mode, x, dim, start, length):
    """Narrow forward wrapper.

    In KBK mode, this wrapper uses jit-wrapped callable.
    """
    if mode == "pynative":
        return narrow_forward_func(x, dim, start, length)
    return KBK_NARROW_FORWARD(x, dim, start, length)


def narrow_backward(mode, x, dim, start, length):
    """Narrow backward wrapper.

    In KBK mode, this wrapper uses jit-wrapped callable.
    """
    if mode == "pynative":
        return narrow_backward_func(x, dim, start, length)
    return KBK_NARROW_BACKWARD(x, dim, start, length)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential', mem_peak=2048.00)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_mint_narrow_normal(mode):
    """
    Feature: mint.narrow
    Description: Verify the result of mint.narrow
    Expectation: success
    """
    ms.set_context(mode=mode)
    ms.context.set_context(jit_level='O0')

    x_np = np.random.randn(2, 4).astype(np.float32)
    x = ms.Tensor(x_np)
    expect_out = x_np[:, 1:3]

    out = narrow_forward(mode, x, -1, 1, 2)
    assert out.shape == expect_out.shape
    assert np.allclose(out.asnumpy(), expect_out)

    expect_grad = np.array([[0., 1., 1., 0.],
                            [0., 1., 1., 0.]]).astype(np.float32)
    grad = narrow_backward(mode, x, -1, 1, 2)
    assert grad.shape == expect_grad.shape
    assert np.allclose(grad.asnumpy(), expect_grad)


def generate_random_input(shape, dtype):
    """Generate random input data."""
    if np.issubdtype(dtype, np.complexfloating):
        real_part = np.random.uniform(-1, 1, shape).astype(
            np.float32 if dtype == np.complex64 else np.float64
        )
        imag_part = np.random.uniform(-1, 1, shape).astype(
            np.float32 if dtype == np.complex64 else np.float64
        )
        return real_part + 1j * imag_part
    if dtype == np.bool_:
        return (np.random.rand(*shape) > 0.5).astype(np.bool_)
    if np.issubdtype(dtype, np.integer):
        return np.random.randint(-100, 100, shape).astype(dtype)
    return np.random.uniform(-1, 1, shape).astype(dtype)


def generate_special_input(shape, dtype, special_type):
    """Generate special input data."""
    if special_type == "inf":
        x = np.ones(shape, dtype=dtype)
        x.reshape(-1)[0] = np.inf
        x.reshape(-1)[-1] = -np.inf
        return x
    if special_type == "nan":
        x = np.ones(shape, dtype=dtype)
        x.reshape(-1)[0] = np.nan
        return x
    if special_type == "zero":
        return np.zeros(shape, dtype=dtype)
    if special_type == "large":
        return np.random.uniform(1e6, 1e9, shape).astype(dtype)
    if special_type == "small":
        return np.random.uniform(-1e-9, 1e-9, shape).astype(dtype)
    return generate_random_input(shape, dtype)


def generate_ones_grad(shape, dtype):
    """Generate gradient input with all ones."""
    return np.ones(shape).astype(dtype)


def generate_expect_forward_output(x, dim, start, length):
    """Generate expected forward output using PyTorch."""
    return torch.narrow(x, dim=dim, start=start, length=length)


def generate_expect_backward_output(x, dim, start, length, grad):
    """Generate expected backward output using PyTorch."""
    if x.is_leaf:
        x.requires_grad_(True)
        out = torch.narrow(x, dim=dim, start=start, length=length)
        out.backward(grad)
        return x.grad

    x_leaf = x.detach().clone().requires_grad_(True)
    out = torch.narrow(x_leaf, dim=dim, start=start, length=length)
    out.backward(grad)
    return x_leaf.grad


def narrow_forward_func(x, dim, start, length):
    """MindSpore forward calculation function."""
    return mint.narrow(x, dim, start, length)


def narrow_backward_func(x, dim, start, length):
    """MindSpore backward propagation function."""
    def forward_fn(x):
        return mint.narrow(x, dim, start, length)
    return ops.grad(forward_fn, (0,))(x)


KBK_NARROW_FORWARD = jit(narrow_forward_func, backend="ms_backend", jit_level="O0")
KBK_NARROW_BACKWARD = jit(narrow_backward_func, backend="ms_backend", jit_level="O0")


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
        card_mark="onecard", essential_mark="essential", mem_peak=1024.00)
@pytest.mark.parametrize("mode", ["pynative", "KBK"])
def test_narrow_std(mode):
    """
    Feature: standard forward, backward features.
    Description: test standard cases for narrow.
    Expectation: expect correct result.
    """
    if mode == "pynative":
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")

    x = generate_random_input((2, 3, 4), np.float32)
    dim, start, length = 1, 1, 2

    torch_x = torch.tensor(x, dtype=torch.float32)
    expect = generate_expect_forward_output(torch_x, dim, start, length)
    grad = generate_ones_grad(expect.shape, np.float32)
    expect_grad = generate_expect_backward_output(
        torch.tensor(x, dtype=torch.float32),
        dim,
        start,
        length,
        torch.tensor(grad, dtype=torch.float32),
    )

    ms_x = ms.Tensor(x, dtype=ms.float32)
    output = narrow_forward(mode, ms_x, dim, start, length)
    output_grad = narrow_backward(mode, ms_x, dim, start, length)
    allclose_nparray(expect.detach().numpy(), output.asnumpy(), equal_nan=True)
    allclose_nparray(expect_grad.detach().numpy(), output_grad.asnumpy(), equal_nan=True)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
        card_mark="onecard", essential_mark="essential", mem_peak=1024.00)
@pytest.mark.parametrize("mode", ["pynative", "KBK"])
@pytest.mark.parametrize(
    "dtype",
    [np.float16, np.float32, np.int8, np.uint8, np.int32, np.int64, np.bool_],
)
def test_narrow_dtype_coverage(mode, dtype):
    """
    Feature: dtype coverage for narrow operator.
    Description: test narrow with various dtypes.
    Expectation: results match PyTorch implementation.
    """
    if mode == "pynative":
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")

    x = generate_random_input((6, 9), dtype)
    dim, start, length = 1, 2, 5

    torch_dtype_map = {
        np.float16: torch.float16,
        np.float32: torch.float32,
        np.int8: torch.int8,
        np.uint8: torch.uint8,
        np.int32: torch.int32,
        np.int64: torch.int64,
        np.bool_: torch.bool,
    }
    torch_dtype = torch_dtype_map.get(dtype, torch.float32)
    torch_x = torch.tensor(x, dtype=torch_dtype)
    expect = generate_expect_forward_output(torch_x, dim, start, length)

    ms_x = ms.Tensor(x)
    output = narrow_forward(mode, ms_x, dim, start, length)
    allclose_nparray(expect.detach().numpy(), output.asnumpy(), equal_nan=True)

    if np.issubdtype(dtype, np.floating) and dtype != np.float16 and needs_backward(dtype):
        grad = generate_ones_grad(expect.shape, expect.numpy().dtype)
        expect_grad = generate_expect_backward_output(
            torch.tensor(x, dtype=torch_dtype),
            dim,
            start,
            length,
            torch.tensor(grad, dtype=torch_dtype),
        )
        output_grad = narrow_backward(mode, ms_x, dim, start, length)
        allclose_nparray(expect_grad.detach().numpy(), output_grad.asnumpy(), equal_nan=True)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
        card_mark="onecard", essential_mark="essential", mem_peak=1024.00)
@pytest.mark.parametrize("mode", ["pynative", "KBK"])
@pytest.mark.parametrize(
    "shape",
    [
        (1,),
        (2, 3),
        (2, 3, 4),
        (2, 2, 3, 4),
        (2, 2, 2, 3, 4),
        (2, 1, 2, 2, 3, 4),
        (1, 2, 1, 2, 2, 3, 4),
        (2, 1, 2, 1, 2, 2, 3, 4),
    ],
)
def test_narrow_dimensions(mode, shape):
    """
    Feature: dimension coverage for narrow operator.
    Description: test narrow with 1D to 8D inputs.
    Expectation: results match PyTorch implementation.
    """
    if mode == "pynative":
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")

    x = generate_random_input(shape, np.float32)
    ms_x = ms.Tensor(x, dtype=ms.float32)
    torch_x = torch.tensor(x, dtype=torch.float32)
    for dim in [0, -1]:
        axis = dim if dim >= 0 else dim + len(shape)
        size = shape[axis]
        start = 0
        length = min(3, size)

        expect = generate_expect_forward_output(torch_x, dim, start, length)
        grad = generate_ones_grad(expect.shape, np.float32)
        expect_grad = generate_expect_backward_output(
            torch.tensor(x, dtype=torch.float32),
            dim,
            start,
            length,
            torch.tensor(grad, dtype=torch.float32),
        )

        output = narrow_forward(mode, ms_x, dim, start, length)
        output_grad = narrow_backward(mode, ms_x, dim, start, length)
        allclose_nparray(expect.detach().numpy(), output.asnumpy(), equal_nan=True)
        allclose_nparray(expect_grad.detach().numpy(), output_grad.asnumpy(), equal_nan=True)
        assert tuple(output.shape) == tuple(expect.shape)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
        card_mark="onecard", essential_mark="essential", mem_peak=1024.00)
@pytest.mark.parametrize("mode", ["pynative", "KBK"])
@pytest.mark.parametrize("special_type", ["inf", "nan", "zero", "large", "small"])
def test_narrow_special_values(mode, special_type):
    """
    Feature: special value handling for narrow operator.
    Description: test narrow with inf, nan, zero, large and small values.
    Expectation: results match PyTorch implementation.
    """
    if mode == "pynative":
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")

    x = generate_special_input((2, 3, 10), np.float32, special_type)
    dim, start, length = 2, 1, 7
    torch_x = torch.tensor(x, dtype=torch.float32)
    expect = generate_expect_forward_output(torch_x, dim, start, length)
    grad = generate_ones_grad(expect.shape, np.float32)
    expect_grad = generate_expect_backward_output(
        torch.tensor(x, dtype=torch.float32),
        dim,
        start,
        length,
        torch.tensor(grad, dtype=torch.float32),
    )
    ms_x = ms.Tensor(x, dtype=ms.float32)
    output = narrow_forward(mode, ms_x, dim, start, length)
    output_grad = narrow_backward(mode, ms_x, dim, start, length)
    allclose_nparray(expect.detach().numpy(), output.asnumpy(), equal_nan=True)
    allclose_nparray(expect_grad.detach().numpy(), output_grad.asnumpy(), equal_nan=True)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
        card_mark="onecard", essential_mark="essential", mem_peak=1024.00)
@pytest.mark.parametrize("mode", ["pynative", "KBK"])
def test_narrow_empty_tensor(mode):
    """
    Feature: empty tensor handling for narrow operator.
    Description: test narrow with empty tensor input.
    Expectation: results match PyTorch implementation.
    """
    if mode == "pynative":
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")

    shapes = [(0,), (2, 0), (0, 3, 4), (2, 0, 4)]
    for shape in shapes:
        x = generate_random_input(shape, np.float32)
        ms_x = ms.Tensor(x, dtype=ms.float32)
        torch_x = torch.tensor(x, dtype=torch.float32)

        if len(shape) == 1:
            dim = 0
        else:
            dim = 0 if shape[0] == 0 else 1
        start, length = 0, 0

        expect = generate_expect_forward_output(torch_x, dim, start, length)
        output = narrow_forward(mode, ms_x, dim, start, length)
        assert tuple(output.shape) == tuple(expect.shape)
        allclose_nparray(expect.detach().numpy(), output.asnumpy(), equal_nan=True)

        grad = generate_ones_grad(expect.shape, np.float32)
        expect_grad = generate_expect_backward_output(
            torch.tensor(x, dtype=torch.float32),
            dim,
            start,
            length,
            torch.tensor(grad, dtype=torch.float32),
        )
        output_grad = narrow_backward(mode, ms_x, dim, start, length)
        allclose_nparray(expect_grad.detach().numpy(), output_grad.asnumpy(), equal_nan=True)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
        card_mark="onecard", essential_mark="essential", mem_peak=1024.00)
@pytest.mark.parametrize("mode", ["pynative", "KBK"])
def test_narrow_non_contiguous(mode):
    """
    Feature: non-contiguous input support for narrow operator.
    Description: test narrow with non-contiguous tensor input.
    Expectation: results match PyTorch implementation.
    """
    if mode == "pynative":
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")

    x = generate_random_input((2, 3, 4), np.float32)
    torch_x_base = torch.tensor(x, dtype=torch.float32)
    torch_x = torch_x_base.transpose(0, 1)
    assert not torch_x.is_contiguous()

    ms_x_base = ms.Tensor(x, dtype=ms.float32)
    ms_x = mint.transpose(ms_x_base, 0, 1)
    assert not ms_x.is_contiguous()

    dim, start, length = 1, 0, 2
    expect = generate_expect_forward_output(torch_x, dim, start, length)
    grad = generate_ones_grad(expect.shape, np.float32)
    expect_grad = generate_expect_backward_output(
        torch_x,
        dim,
        start,
        length,
        torch.tensor(grad, dtype=torch.float32),
    )
    output = narrow_forward(mode, ms_x, dim, start, length)
    output_grad = narrow_backward(mode, ms_x, dim, start, length)
    allclose_nparray(expect.detach().numpy(), output.asnumpy(), equal_nan=True)
    allclose_nparray(expect_grad.detach().numpy(), output_grad.asnumpy(), equal_nan=True)
