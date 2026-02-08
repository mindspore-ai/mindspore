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
from mindspore import mint, jit
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.st.utils import test_utils
from tests.mark_utils import arg_mark


def generate_random_input(shape):
    return np.random.randn(*shape).astype(np.float32)


def generate_expect_forward_output(x):
    return np.minimum(np.maximum(0, x), 6)


def generate_expect_backward_output(x):
    return np.where((x > 0, x < 6), 1., 0.)[0]


def relu6_forward_func(x, inplace=False):
    out = mint.nn.functional.relu6(x, inplace=inplace)
    if inplace:
        return x
    return out


@test_utils.run_with_cell
def inplace_relu6_forward_func(x):
    mint.nn.functional.relu6(x, inplace=True)
    return x


def relu6_backward_func(x, inplace=False):
    return ms.ops.grad(relu6_forward_func, (0))(x, inplace=inplace)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_relu6_std(mode):
    """
    Feature: standard forward, backward features.
    Description: test function relu6
    Expectation: expect correct result.
    """
    x_np = generate_random_input((2, 3))
    x = ms.Tensor(x_np)
    expect = generate_expect_forward_output(x_np)
    expect_grad = generate_expect_backward_output(x_np)

    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        output = relu6_forward_func(x)
        output_grad = relu6_backward_func(x)
        inplace_x = inplace_relu6_forward_func(ms.Tensor(x_np, dtype=ms.float32))
    else:
        output = (jit(relu6_forward_func, backend="ms_backend", jit_level="O0"))(x)
        output_grad = (jit(relu6_backward_func, backend="ms_backend", jit_level="O0"))(x)
        inplace_x = (jit(inplace_relu6_forward_func, backend="ms_backend", jit_level="O0"))(
            ms.Tensor(x_np, dtype=ms.float32))
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(output_grad.asnumpy(), expect_grad, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(inplace_x.asnumpy(), expect, rtol=1e-5, atol=1e-5)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=2048.01)
def test_relu6_dynamic_shape():
    """
    Feature: dynamic shape forward, backward features.
    Description: test relu6 forward with dynamic shape.
    Expectation: expect correct result.
    """
    x1 = generate_random_input((3, 4, 5))
    x2 = generate_random_input((3, 7, 8, 3))
    TEST_OP(relu6_forward_func, [[ms.Tensor(x1)], [ms.Tensor(x2)]],
            disable_mode=['GRAPH_MODE_GE'],
            case_config={'disable_input_check': True})
    TEST_OP(inplace_relu6_forward_func, [[ms.Tensor(x1)], [ms.Tensor(x2)]],
            disable_mode=['GRAPH_MODE_GE'],
            case_config={'disable_input_check': True,
                         'disable_grad': True,
                         'all_dim_zero': True})


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=4.0)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_relu6_bfloat16(mode):
    """
    Feature: test relu6 functional API.
    Description: testcase for relu6 functional API.
    Expectation: the result match with expected result.
    """
    x_np = generate_random_input((2, 3, 4))
    x = ms.Tensor(x_np, dtype=ms.bfloat16)
    expect = generate_expect_forward_output(x_np)
    expect_grad = generate_expect_backward_output(x_np)

    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        output = relu6_forward_func(x)
        output_grad = relu6_backward_func(x)
        inplace_x = inplace_relu6_forward_func(ms.Tensor(x_np, dtype=ms.bfloat16))
    else:
        output = (jit(relu6_forward_func, jit_level="O0"))(x)
        output_grad = (jit(relu6_backward_func, jit_level="O0"))(x)
        inplace_x = (jit(inplace_relu6_forward_func, jit_level="O0"))(
            ms.Tensor(x_np, dtype=ms.bfloat16))
    np.allclose(output.float().asnumpy(), expect, 4e-3, 4e-3, equal_nan=True)
    np.allclose(output_grad.float().asnumpy(), expect_grad, 4e-3, 4e-3, equal_nan=True)
    np.testing.assert_allclose(inplace_x.asnumpy(), expect, rtol=4e-3, atol=4e-3)
