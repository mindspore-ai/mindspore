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
# WITHOUT WARRANTIES OR ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
Test cases for Tensor.fill_ operator on Ascend backend.
Benchmark against torch_cpu implementation.
"""
import numpy as np
import pytest

import mindspore as ms
from mindspore import Tensor, ops, jit
from tests.mark_utils import arg_mark
from tests.st.pynative.utils import allclose_nparray
import torch


def generate_expect_forward_output(x, value):
    """Generate expected forward output using PyTorch"""
    x = x * 1
    x.fill_(value)
    return x


def generate_expect_backward_output(x, value, grad):
    """Generate expected backward output using PyTorch"""
    x.requires_grad = True
    x_new = x * 1
    out = x_new.fill_(value)
    out.backward(grad)
    return x.grad

@jit(backend="ms_backend")
def fill_scalar_forward_func(x, value):
    """MindSpore forward function for fill_ with scalar"""
    x = x * 1
    x.fill_(value)
    return x

@jit(backend="ms_backend")
def fill_scalar_backward_func(x, value):
    """MindSpore backward function for fill_ with scalar"""
    return ops.grad(fill_scalar_forward_func, (0,))(x, value)

@jit(backend="ms_backend")
def fill_tensor_forward_func(x, value):
    """MindSpore forward function for fill_ with tensor"""
    x = x * 1
    x.fill_(value)
    return x

@jit(backend="ms_backend")
def fill_tensor_backward_func(x, value):
    """MindSpore backward function for fill_ with tensor"""
    return ops.grad(fill_tensor_forward_func, (0,))(x, value)


def generate_expect_forward_output_tensor(x, value):
    """Generate expected forward output using PyTorch for tensor value"""
    x = x * 1
    x.fill_(value)
    return x


def generate_expect_backward_output_tensor(x, value, grad):
    """Generate expected backward output using PyTorch for tensor value"""
    x.requires_grad = True
    x_new = x * 1
    out = x_new.fill_(value)
    out.backward(grad)
    return x.grad


@arg_mark(
    plat_marks=["platform_ascend"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential", mem_peak=1.00)
@pytest.mark.parametrize("mode", ["pynative", "KBK"])
def test_fill__forward_scalar(mode):
    """
    Feature: Tensor.fill_ forward with scalar value
    Description: Test fill_ operator with scalar value in pynative and KBK mode, benchmark against torch_cpu
    Expectation: Output matches expected result and torch_cpu result
    """
    if mode == "pynative":
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    elif mode == "KBK":
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")

    # Test case 1: Basic scalar fill
    input_x = Tensor(np.full((10, 10), 0, dtype=np.float32))
    value = 5.0
    output = fill_scalar_forward_func(input_x, value)
    torch_input = torch.full((10, 10), 0.0, dtype=torch.float32)
    expected = generate_expect_forward_output(torch_input, value)
    allclose_nparray(expected.numpy(), output.asnumpy(), 0, 0)

    # Test case 2: Different shape
    input_x = Tensor(np.full((3, 4, 5), 1, dtype=np.float32))
    value = 2.5
    output = fill_scalar_forward_func(input_x, value)
    torch_input = torch.full((3, 4, 5), 1.0, dtype=torch.float32)
    expected = generate_expect_forward_output(torch_input, value)
    allclose_nparray(expected.numpy(), output.asnumpy(), 0, 0)

    # Test case 3: Integer value
    input_x = Tensor(np.full((5, 5), 0, dtype=np.float32))
    value = 7
    output = fill_scalar_forward_func(input_x, value)
    torch_input = torch.full((5, 5), 0.0, dtype=torch.float32)
    expected = generate_expect_forward_output(torch_input, value)
    allclose_nparray(expected.numpy(), output.asnumpy(), 0, 0)


@arg_mark(
    plat_marks=["platform_ascend"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential", mem_peak=2048.00
)
@pytest.mark.parametrize("mode", ["pynative", "KBK"])
def test_fill__forward_tensor(mode):
    """
    Feature: Tensor.fill_ forward with tensor value
    Description: Test fill_ operator with tensor value in pynative and KBK mode
    Expectation: Output matches expected result
    """
    if mode == "pynative":
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    elif mode == "KBK":
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")

    # Test case 1: Basic tensor fill
    input_x = Tensor(np.full((10, 10), 0, dtype=np.float32))
    value = Tensor(5.0, ms.float32)
    output = fill_tensor_forward_func(input_x, value)
    torch_input = torch.full((10, 10), 0.0, dtype=torch.float32)
    torch_value = torch.tensor(5.0, dtype=torch.float32)
    expected = generate_expect_forward_output_tensor(torch_input, torch_value)
    allclose_nparray(expected.numpy(), output.asnumpy(), 0, 0)

    # Test case 2: Different shape
    input_x = Tensor(np.full((2, 3, 4), 1, dtype=np.float32))
    value = Tensor(3.14, ms.float32)
    output = fill_tensor_forward_func(input_x, value)
    torch_input = torch.full((2, 3, 4), 1.0, dtype=torch.float32)
    torch_value = torch.tensor(3.14, dtype=torch.float32)
    expected = generate_expect_forward_output_tensor(torch_input, torch_value)
    allclose_nparray(expected.numpy(), output.asnumpy(), 0, 0)


@arg_mark(
    plat_marks=["platform_ascend"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential", mem_peak=2048.01)
@pytest.mark.parametrize("mode", ["pynative", "KBK"])
def test_fill__backward_scalar(mode):
    """
    Feature: Tensor.fill_ backward with scalar value
    Description: Test fill_ operator backward with scalar value in pynative and KBK mode, benchmark against torch_cpu
    Expectation: Gradient matches expected result and torch_cpu result
    """
    if mode == "pynative":
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    elif mode == "KBK":
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")

    # Test case 1: Basic backward
    input_x = Tensor(np.full((10, 10), 0, dtype=np.float32))
    value = 5.0
    grads = fill_scalar_backward_func(input_x, value)
    torch_input = torch.full((10, 10), 0.0, dtype=torch.float32)
    grad = torch.ones((10, 10), dtype=torch.float32)
    expected_grad = generate_expect_backward_output(torch_input, value, grad)
    allclose_nparray(expected_grad.numpy(), grads.asnumpy(), 0, 0)

    # Test case 2: Different shape
    input_x = Tensor(np.full((3, 4, 5), 1, dtype=np.float32))
    value = 2.5
    grads = fill_scalar_backward_func(input_x, value)
    torch_input = torch.full((3, 4, 5), 1.0, dtype=torch.float32)
    grad = torch.ones((3, 4, 5), dtype=torch.float32)
    expected_grad = generate_expect_backward_output(torch_input, value, grad)
    allclose_nparray(expected_grad.numpy(), grads.asnumpy(), 0, 0)


@arg_mark(
    plat_marks=["platform_ascend"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential", mem_peak=2048.01)
@pytest.mark.parametrize("mode", ["pynative", "KBK"])
def test_fill__backward_tensor(mode):
    """
    Feature: Tensor.fill_ backward with tensor value
    Description: Test fill_ operator backward with tensor value in pynative and KBK mode
    Expectation: Gradient matches expected result
    """
    if mode == "pynative":
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    elif mode == "KBK":
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")

    # Test case 1: Basic backward with tensor value
    input_x = Tensor(np.full((10, 10), 0, dtype=np.float32))
    value = Tensor(5.0, ms.float32)
    grads = fill_tensor_backward_func(input_x, value)
    torch_input = torch.full((10, 10), 0.0, dtype=torch.float32)
    torch_value = torch.tensor(5.0, dtype=torch.float32)
    grad = torch.ones((10, 10), dtype=torch.float32)
    expected_grad = generate_expect_backward_output_tensor(torch_input, torch_value, grad)
    allclose_nparray(expected_grad.numpy(), grads.asnumpy(), 0, 0)

    # Test case 2: Different shape
    input_x = Tensor(np.full((2, 3, 4), 1, dtype=np.float32))
    value = Tensor(3.14, ms.float32)
    grads = fill_tensor_backward_func(input_x, value)
    torch_input = torch.full((2, 3, 4), 1.0, dtype=torch.float32)
    torch_value = torch.tensor(3.14, dtype=torch.float32)
    grad = torch.ones((2, 3, 4), dtype=torch.float32)
    expected_grad = generate_expect_backward_output_tensor(torch_input, torch_value, grad)
    allclose_nparray(expected_grad.numpy(), grads.asnumpy(), 0, 0)


@arg_mark(
    plat_marks=["platform_ascend"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="unessential", mem_peak=2048.00
)
@pytest.mark.parametrize("mode", ["pynative", "KBK"])
@pytest.mark.parametrize("dtype", [np.float32, np.float16, np.int32, np.int64])
def test_fill__dtype(mode, dtype):
    """
    Feature: Tensor.fill_ with different data types
    Description: Test fill_ operator with various data types
    Expectation: Output matches expected result for each dtype
    """
    if mode == "pynative":
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    elif mode == "KBK":
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")

    input_x = Tensor(np.full((5, 5), 0, dtype=dtype))
    value = 10
    output = fill_scalar_forward_func(input_x, value)
    if dtype in [np.float32, np.float16]:
        torch_dtype = torch.float32 if dtype == np.float32 else torch.float16
        torch_input = torch.full((5, 5), 0, dtype=torch_dtype)
        expected = generate_expect_forward_output(torch_input, value)
        allclose_nparray(expected.numpy(), output.asnumpy(), 0, 0)
    else:
        expected = Tensor(np.full((5, 5), 10, dtype=dtype))
        allclose_nparray(expected.asnumpy(), output.asnumpy(), 0, 0)


@arg_mark(
    plat_marks=["platform_ascend"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="unessential", mem_peak=2048.00
)
@pytest.mark.parametrize("mode", ["pynative", "KBK"])
@pytest.mark.parametrize("shape", [(1,), (10,), (5, 5), (2, 3, 4), (1, 2, 3, 4)])
def test_fill__shape(mode, shape):
    """
    Feature: Tensor.fill_ with different shapes
    Description: Test fill_ operator with various tensor shapes
    Expectation: Output matches expected result for each shape
    """
    if mode == "pynative":
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    elif mode == "KBK":
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")

    input_x = Tensor(np.full(shape, 0, dtype=np.float32))
    value = 7.5
    output = fill_scalar_forward_func(input_x, value)
    torch_input = torch.full(shape, 0.0, dtype=torch.float32)
    expected = generate_expect_forward_output(torch_input, value)
    allclose_nparray(expected.numpy(), output.asnumpy(), 0, 0)
