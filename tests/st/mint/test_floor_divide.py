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
from mindspore import mint, jit
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.mark_utils import arg_mark


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)


def floor_divide_forward_func(x, other):
    y = mint.floor_divide(x, other)
    return y


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=2.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_floor_divide_normal0(mode):
    """
    Feature: mint.
    Description: test floor_divide.
    Expectation: expect correct result.
    """
    x = ms.Tensor(generate_random_input((2, 3), np.float32))
    other = ms.Tensor(generate_random_input((2, 3), np.float32))
    expect_y = np.floor_divide(x, other)

    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        y = floor_divide_forward_func(x, other)
    else:
        y = (jit(floor_divide_forward_func, backend="ms_backend"))(x, other)

    np.testing.assert_allclose(y.asnumpy(), expect_y, rtol=1e-5)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
        card_mark='onecard', essential_mark='unessential', mem_peak=1024.01)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_floor_divide_normal1(mode):
    """
    Feature: mint.
    Description: test floor_divide.
    Expectation: expect correct result.
    """
    x = ms.Tensor(generate_random_input((2, 3), np.float32))
    other = 2.0
    expect_y = np.floor_divide(x, other)

    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        y = floor_divide_forward_func(x, other)
    else:
        y = (jit(floor_divide_forward_func, backend="ms_backend"))(x, other)

    np.testing.assert_allclose(y.asnumpy(), expect_y, rtol=1e-5)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=2048.0)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_floor_divide_normal2(mode):
    """
    Feature: mint.
    Description: test floor_divide.
    Expectation: expect correct result.
    """
    x = 2.0
    other = ms.Tensor(generate_random_input((2, 3), np.float32))
    expect_y = np.floor_divide(x, other)

    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        y = floor_divide_forward_func(x, other)
    else:
        y = (jit(floor_divide_forward_func, backend="ms_backend"))(x, other)

    np.testing.assert_allclose(y.asnumpy(), expect_y, rtol=1e-5)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
        card_mark='onecard', essential_mark='unessential', mem_peak=3072.00)
def test_floor_divide_dynamic_shape():
    """
    Feature: Test floor_divide with dynamic shape in graph mode.
    Description: call mint.floor_divide with valid input and index.
    Expectation: return the correct value.
    """
    x1 = generate_random_input((2, 3), np.float32)
    other1 = generate_random_input((2, 3), np.float32)
    x2 = generate_random_input((2, 3, 4), np.float32)
    other2 = generate_random_input((2, 3, 4), np.float32)

    TEST_OP(floor_divide_forward_func,
            [[ms.Tensor(x1), ms.Tensor(other1)], [ms.Tensor(x2), ms.Tensor(other2)]],
            case_config={'all_dim_zero': True})
