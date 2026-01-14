# Copyright 2024 Huawei Technocasties Co., Ltd
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
import pytest
import numpy as np
from mindspore import ops, Tensor, jit, dtype as mstype
import mindspore as ms
import tests.st.utils.test_utils as test_utils
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.mark_utils import arg_mark


def generate_random_input(shape, dtype):
    return np.random.randint(1, 10, size=shape).astype(dtype)


def generate_expect_forward_output(x, min_, max_):
    return np.clip(x, min_, max_)


@test_utils.run_with_cell
def clamp_forward_func(x, min_, max_):
    return x.clamp_(min_, max_)


@test_utils.run_with_cell
def clamp_forward_func_grad(x, min_, max_):
    x = x * 1
    return x.clamp_(min_, max_)


def clamp_backward_func(x, min_, max_):
    grad = ops.GradOperation(get_all=True)
    return grad(clamp_forward_func_grad)(x, min_, max_)


def expect_backward_func(x, min_, max_):
    grad = ops.GradOperation(get_all=True)
    return grad(generate_expect_forward_output)(x, min_, max_)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=1024.01)
@pytest.mark.parametrize("context_mode", [ms.PYNATIVE_MODE])
def test_mint_clamp_normal0(context_mode):
    """
    Feature: pyboost function.
    Description: test function clamp forward and backward.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x = np.array([[10, 0, 0, -2], [5.3, 5.2, 11, 8]])
    x = Tensor(x, mstype.float32)
    # input_min & input_max
    clamp_forward_func(x, -1, 9)
    expect = Tensor(
        np.array([[9, 0, 0, -1], [5.3, 5.2, 9, 8]]), mstype.float32)

    assert np.allclose(x.asnumpy(), expect.asnumpy(), rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=2048.00)
@pytest.mark.parametrize("context_mode", ['pynative', 'KBK'])
def test_mint_clamp_normal1(context_mode):
    """
    Feature: pyboost function.
    Description: test function clamp_ forward and backward.
    Expectation: expect correct result.
    """
    x = ms.Tensor(generate_random_input((3, 4, 5), np.float32))
    input_min = 3
    input_max = 6
    expect_out = generate_expect_forward_output(x, input_min, input_max)
    expect_grad = expect_backward_func(x, input_min, input_max)
    if context_mode == 'pynative':
        output = clamp_forward_func(x, input_min, input_max)
        grad = clamp_backward_func(x, input_min, input_max)
    else:
        output = (jit(clamp_forward_func, backend="ms_backend", jit_level="O0"))(x, input_min, input_max)
        grad = (jit(clamp_backward_func, backend="ms_backend", jit_level="O0"))(x, input_min, input_max)
    np.allclose(expect_out, output.asnumpy(), rtol=1e-5)
    np.allclose(expect_grad, grad[0].asnumpy(), rtol=1e-5)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=1024.01)
@pytest.mark.parametrize("context_mode", ['pynative', 'KBK'])
def test_mint_clamp_normal2(context_mode):
    """
    Feature: pyboost function.
    Description: test function clamp_ forward and backward.
    Expectation: expect correct result.
    """
    x = ms.Tensor(generate_random_input((3, 4, 5), np.float32))
    input_min = ms.Tensor(generate_random_input((3, 4, 1), np.float32))
    input_max = ms.Tensor(generate_random_input((3, 1, 1), np.float32))
    expect_out = generate_expect_forward_output(x, input_min, input_max)
    expect_grad, expect_min, expect_max = expect_backward_func(
        x, input_min, input_max)
    if context_mode == 'pynative':
        output = clamp_forward_func(x, input_min, input_max)
        grad, grad_min, grad_max = clamp_backward_func(x, input_min, input_max)
    else:
        output = (jit(clamp_forward_func, backend="ms_backend", jit_level="O0"))(x, input_min, input_max)
        grad, grad_min, grad_max = (jit(clamp_backward_func, backend="ms_backend", jit_level="O0"))(
            x, input_min, input_max)
    np.allclose(expect_out, output.asnumpy(), rtol=1e-5)
    np.allclose(expect_grad, grad.asnumpy(), rtol=1e-5)
    np.allclose(expect_min, grad_min.asnumpy(), rtol=1e-5)
    np.allclose(expect_max, grad_max.asnumpy(), rtol=1e-5)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=1024.01)
def test_mint_clamp_min_max_tensor_dynamic_shape():
    """
    Feature: pyboost function.
    Description: test function clamp forward with dynamic shape.
    Expectation: expect correct result.
    """
    x1 = ms.Tensor(generate_random_input((3, 4, 5), np.float32))
    min1 = ms.Tensor(generate_random_input((3, 4, 1), np.float32))
    max1 = ms.Tensor(generate_random_input((3, 1, 1), np.float32))

    x2 = ms.Tensor(generate_random_input((3, 4, 5, 6), np.float32))
    min2 = ms.Tensor(generate_random_input((3, 4, 5, 1), np.float32))
    max2 = ms.Tensor(generate_random_input((3, 4, 1, 6), np.float32))

    TEST_OP(clamp_forward_func_grad,
            [[x1, min1, max1], [x2, min2, max2]],
            disable_mode=['GRAPH_MODE_GE', 'GRAPH_MODE_O0'],
            case_config={'all_dim_zero': True})


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=1024.0)
def test_mint_clamp_min_max_scalar_dynamic_shape():
    """
    Feature: pyboost function.
    Description: test function clamp forward with dynamic shape.
    Expectation: expect correct result.
    """
    x1 = ms.Tensor(generate_random_input((3, 4, 5), np.float32))
    min1 = 2
    max1 = 7

    x2 = ms.Tensor(generate_random_input((3, 4, 5, 6), np.float32))
    min2 = 3
    max2 = 8

    TEST_OP(clamp_forward_func_grad,
            [[x1, min1, max1], [x2, min2, max2]],
            disable_mode=['GRAPH_MODE_GE', 'GRAPH_MODE_O0'],
            case_config={'all_dim_zero': True})
