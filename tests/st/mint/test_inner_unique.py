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
import numpy as np
import pytest

import mindspore as ms
from mindspore import Tensor, ops
from tests.mark_utils import arg_mark
from tests.st.utils.test_utils import run_with_cell
from tests.st.ops.test_tools.test_op import TEST_OP


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)


@run_with_cell
def inner_unique_forward_func(input_x, sorted_=True, return_inverse=False):
    return ops.auto_generate.inner_unique_op(input_x, sorted_, return_inverse)


def inner_unique_benchmark(input_np, return_inverse=False):
    expect = np.unique(input_np, return_inverse=return_inverse)
    if return_inverse:
        expect = list(expect)
        expect[1] = np.reshape(expect[1], input_np.shape)
    return expect


def compare_func(expect, actual):
    if expect.dtype == np.float32:
        loss = 1e-4
    elif expect.dtype == np.int64:
        loss = 0
    np.testing.assert_allclose(expect, actual, rtol=loss)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=2048.01)
@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
@pytest.mark.parametrize('dtype', [np.float32, np.int64])
@pytest.mark.parametrize('return_inverse', [True, False])
def test_inner_unique_normal(mode, dtype, return_inverse):
    """
    Feature: inner_unique
    Description: Verify the result of inner_unique
    Expectation: success
    """
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    elif mode == 'kbk':
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level='O0')

    x_np = generate_random_input((10, 10), dtype)
    expect = inner_unique_benchmark(x_np, return_inverse)
    actual = inner_unique_forward_func(Tensor(x_np), True, return_inverse)

    if return_inverse:
        compare_func(expect[0], actual[0].asnumpy())
        compare_func(expect[1], actual[1].asnumpy())
    else:
        compare_func(expect, actual[0].asnumpy())
        assert actual[1].shape == (0,)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
        card_mark='onecard', essential_mark='unessential', mem_peak=2048.01)
def test_inner_unique_dynamic():
    """
    Feature: inner_unique
    Description: Verify the result of inner_unique
    Expectation: success
    """
    input1 = Tensor(generate_random_input((2, 10, 10), np.float32))
    sorted1 = True
    return_inverse1 = True

    input2 = Tensor(generate_random_input((10, 10), np.float32))
    sorted2 = False
    return_inverse2 = False

    TEST_OP(
        inner_unique_forward_func,
        [[input1, sorted1, return_inverse1], [input2, sorted2, return_inverse2]],
        disable_mode=['GRAPH_MODE_GE'],
        case_config={
            'disable_grad': True,
        }
    )
