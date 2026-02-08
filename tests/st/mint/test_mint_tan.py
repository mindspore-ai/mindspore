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
import pytest
import numpy as np
import mindspore as ms
from mindspore import ops, mint, jit
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.mark_utils import arg_mark


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)


def generate_expect_forward_output(x):
    return np.tan(x)


def generate_expect_backward_output(x):
    return 1 + np.square(x)


def tan_forward_func(x):
    return mint.tan(x)


def tan_backward_func(x):
    return ops.grad(tan_forward_func, (0,))(x)


@arg_mark(
    plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
    card_mark='onecard', essential_mark='essential', mem_peak=2048.25)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_tan_std(mode):
    """
    Feature: standard forward, backward features.
    Description: test function tan.
    Expectation: expect correct result.
    """
    x = generate_random_input((2, 3, 4), np.float32)
    expect = generate_expect_forward_output(x)
    expect_grad = generate_expect_backward_output(x)
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        output = tan_forward_func(ms.Tensor(x))
        output_grad = tan_backward_func(ms.Tensor(x))
    else:
        output = (jit(tan_forward_func, jit_level="O0"))(ms.Tensor(x))
        output_grad = (jit(tan_backward_func, jit_level="O0"))(ms.Tensor(x))

    np.allclose(output.asnumpy(), expect, rtol=1e-5, equal_nan=True)
    np.allclose(output_grad.asnumpy(), expect_grad, rtol=1e-5, equal_nan=True)


@arg_mark(
    plat_marks=['platform_ascend', 'platform_ascend910b', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'],
    level_mark='level1', card_mark='onecard', essential_mark='unessential', mem_peak=4.0)
def test_tan_dynamic_shape():
    """
    Feature: dynamic shape forward, backward features.
    Description: test tan forward with dynamic shape.
    Expectation: expect correct result.
    """
    tensor_1 = ms.Tensor(generate_random_input((2, 3), np.float32))
    tensor_2 = ms.Tensor(generate_random_input((2, 3, 4), np.float32))

    TEST_OP(tan_forward_func, [[tensor_1], [tensor_2]])


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=4.0)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_tan_bfloat16(mode):
    """
    Feature: test tan functional API.
    Description: testcase for tan functional API.
    Expectation: the result match with expected result.
    """
    x = generate_random_input((2, 3), np.float32)
    expect = generate_expect_forward_output(x).astype(np.float32)
    expect_grad = generate_expect_backward_output(x).astype(np.float32)

    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        output = tan_forward_func(ms.Tensor(x, dtype=ms.bfloat16))
        output_grad = tan_backward_func(ms.Tensor(x, dtype=ms.bfloat16))
    else:
        output = (jit(tan_forward_func, jit_level="O0"))(ms.Tensor(x, dtype=ms.bfloat16))
        output_grad = (jit(tan_backward_func, jit_level="O0"))(ms.Tensor(x, dtype=ms.bfloat16))

    np.allclose(output.float().asnumpy(), expect, 0.004, 0.004, equal_nan=True)
    np.allclose(output_grad.float().asnumpy(), expect_grad, 0.004, 0.004, equal_nan=True)
