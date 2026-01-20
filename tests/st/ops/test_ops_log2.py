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
"""ST cases for log2."""
import pytest
import numpy as np
import mindspore as ms
from mindspore import mint
from tests.st.utils import test_utils
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.mark_utils import arg_mark


def generate_random_input(shape, dtype):
    return np.random.randint(1, 10, size=shape).astype(dtype)


def generate_random_complex_input(shape, dtype):
    real = np.random.rand(*shape).astype(np.float32) + 0.5
    imag = np.random.rand(*shape).astype(np.float32) + 0.5
    return (real + 1j * imag).astype(dtype)


def generate_expect_forward_output(x):
    return np.log2(x)


def generate_expect_backward_output(x):
    return 1 / (np.log(2) * x)


def generate_expect_backward_output_complex(x):
    return 1 / (np.log(2) * np.conj(x))


def log2_func(x):
    return mint.log2(x)


@test_utils.run_with_cell
def log2_forward_func(x):
    return mint.log2(x)


@test_utils.run_with_cell
def log2_backward_func(x):
    return ms.grad(log2_forward_func, (0))(x)


@arg_mark(
    plat_marks=['platform_ascend'],
    level_mark='level1',
    card_mark='onecard',
    essential_mark='essential'
)
@pytest.mark.parametrize("context_mode", ['pynative', 'KBK'])
def test_ops_log2_normal(context_mode):
    """
    Feature: pyboost function.
    Description: test function log2 forward and backward.
    Expectation: expect correct result.
    """
    x = generate_random_input((2, 3, 4, 5), np.float32)
    if context_mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        output = log2_forward_func(ms.Tensor(x))
        output_grad = log2_backward_func(ms.Tensor(x))
    elif context_mode == 'KBK':
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
        output = log2_forward_func(ms.Tensor(x))
        output_grad = log2_backward_func(ms.Tensor(x))
    else:
        raise ValueError(f"Unsupported context_mode: {context_mode}")
    expect_forward = generate_expect_forward_output(x)
    expect_backward = generate_expect_backward_output(x)
    np.testing.assert_allclose(output.asnumpy(), expect_forward, rtol=1e-4)
    np.testing.assert_allclose(output_grad.asnumpy(), expect_backward, rtol=1e-4)


@arg_mark(
    plat_marks=['platform_ascend'],
    level_mark='level1',
    card_mark='onecard',
    essential_mark='essential'
)
@pytest.mark.parametrize("context_mode", ['pynative', 'KBK'])
def test_ops_log2_complex(context_mode):
    """
    Feature: pyboost function.
    Description: test complex input for log2 forward and backward.
    Expectation: expect correct result.
    """
    x = generate_random_complex_input((2, 3, 4, 5), np.complex64)
    if context_mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        output = log2_forward_func(ms.Tensor(x))
        output_grad = log2_backward_func(ms.Tensor(x))
    elif context_mode == 'KBK':
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
        output = log2_forward_func(ms.Tensor(x))
        output_grad = log2_backward_func(ms.Tensor(x))
    else:
        raise ValueError(f"Unsupported context_mode: {context_mode}")
    expect_forward = generate_expect_forward_output(x)
    expect_backward = generate_expect_backward_output_complex(x)
    np.testing.assert_allclose(output.asnumpy(), expect_forward, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(output_grad.asnumpy(), expect_backward, rtol=1e-4, atol=1e-4)


@arg_mark(
    plat_marks=['platform_ascend'],
    level_mark='level1',
    card_mark='onecard',
    essential_mark='unessential'
)
def test_mint_log2_dynamic():
    """
    Feature: pyboost function.
    Description: test function log2 forward and backward.
    Expectation: expect correct result.
    """
    input1 = generate_random_input((2, 3, 4, 5), np.float32)
    input2 = generate_random_input((2, 3, 4), np.float32)
    TEST_OP(log2_func, [[ms.Tensor(input1)], [ms.Tensor(input2)]], disable_mode=['GRAPH_MODE_GE'])
