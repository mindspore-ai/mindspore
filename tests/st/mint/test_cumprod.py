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

import pytest
import numpy as np
import mindspore as ms
from mindspore import ops, jit
from mindspore.mint import cumprod
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.st.utils import test_utils
from tests.mark_utils import arg_mark

def generate_random_input(shape, dim):
    x = np.random.randn(*shape)
    return x, np.cumprod(x, dim)


def cumprod_func(x, dim):
    return cumprod(x, dim)


@test_utils.run_with_cell
def cumprod_forward_func(x, dim):
    return cumprod_func(x, dim)


def cumprod_bwd_func(x, dim):
    return ops.grad(cumprod_func, (0,))(x, dim)


@test_utils.run_with_cell
def cumprod_backward_func(x, dim):
    return cumprod_bwd_func(x, dim)

def cumprod_grad(x, axis):
    cumprod_res = np.cumprod(x, axis=axis)
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'])
    while not it.finished:
        idx = list(it.multi_index)
        i = idx[axis]
        current_value = 0
        for j in range(i, x.shape[axis]):
            idx[axis] = j
            current_value += cumprod_res[tuple(idx)] / x[it.multi_index]
        grad[it.multi_index] = current_value
        it.iternext()
    return grad

def generate_random_input_backward(shape, dim):
    x = np.random.randn(*shape)
    return x, cumprod_grad(x, dim)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=3072.0)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_cumprod_normal(mode):
    """
    Feature: Ops.
    Description: test op cumprod.
    Expectation: expect correct result.
    """
    test_shape = (2, 3, 4, 5)
    dim1 = 2
    dim2 = 0

    x, expect = generate_random_input(test_shape, dim1)
    expect1 = cumprod_grad(x, dim2)
    if mode == 'pynative':
        ms.set_context(mode=ms.PYNATIVE_MODE)
        output = cumprod_forward_func(ms.Tensor(x), dim1)
        output1 = cumprod_backward_func(ms.Tensor(x), dim2)
    else:
        output = (jit(cumprod_forward_func, jit_level="O0"))(ms.Tensor(x), dim1)
        output1 = (jit(cumprod_backward_func, jit_level="O0"))(ms.Tensor(x), dim2)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-4)
    np.testing.assert_allclose(output1.asnumpy(), expect1, rtol=1e-4)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=4.0)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_cumprod_bfloat16(mode):
    """
    Feature: test ne functional API.
    Description: testcase for ne functional API.
    Expectation: the result match with expected result.
    """
    test_shape = (2, 3, 4)
    dim1 = 1
    x, expect = generate_random_input(test_shape, dim1)
    if mode == 'pynative':
        ms.set_context(mode=ms.PYNATIVE_MODE)
        output = cumprod_forward_func(ms.Tensor(x), dim1)
    else:
        output = (jit(cumprod_forward_func, jit_level="O0"))(ms.Tensor(x), dim1)
    np.testing.assert_allclose(output.float().asnumpy(), expect, rtol=5e-3, atol=5e-3)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=2048.01)
def test_cumprod_dynamic_shape():
    """
    Feature: Test dynamic shape.
    Description: test function div dynamic feature.
    Expectation: expect correct result.
    """
    dim1 = 0
    ms_data1, _ = generate_random_input((2, 3, 4), dim1)
    dim2 = 1
    ms_data2, _ = generate_random_input((3, 4, 5, 6), dim2)
    TEST_OP(cumprod_forward_func, [[ms.Tensor(ms_data1), dim1], [ms.Tensor(ms_data2), dim2]],
            disable_mode=['GRAPH_MODE_GE'], disable_case=['ScalarTensor'])
