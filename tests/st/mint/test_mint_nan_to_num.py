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


def generate_random_input(dtype):
    return np.array([float('nan'), float('inf'), -float('inf'), 3.14]).astype(dtype)


def generate_expect_forward_output(x, nan_num, inf_num, neg_inf_num):
    x[np.isnan(x)] = nan_num
    x[x == np.inf] = inf_num
    x[x == -np.inf] = neg_inf_num
    return x


def generate_expect_backward_output(x, nan_num, inf_num, neg_inf_num):
    return x[np.isfinite(x)]


def nan_to_num_forward_func(x, nan_num, inf_num, neg_inf_num):
    return mint.nan_to_num(x, nan_num, inf_num, neg_inf_num)


def nan_to_num_backward_func(x, nan_num, inf_num, neg_inf_num):
    return ops.grad(nan_to_num_forward_func, (0))(x, nan_num, inf_num, neg_inf_num)


@arg_mark(
    plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard',
    essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK', 'GE'])
def test_nan_to_num_std(mode):
    """
    Feature: standard forward, backward features.
    Description: test function nan_to_num.
    Expectation: expect correct result.
    """
    x = generate_random_input(np.float32)
    nan_num = 1.0
    inf_num = 2.0
    neg_inf_num = 3.0
    expect = generate_expect_forward_output(x, nan_num, inf_num, neg_inf_num)
    expect_grad = generate_expect_backward_output(x, nan_num, inf_num, neg_inf_num)

    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        output = nan_to_num_forward_func(ms.Tensor(x), nan_num, inf_num, neg_inf_num)
        output_grad = nan_to_num_backward_func(ms.Tensor(x), nan_num, inf_num, neg_inf_num)

        output_nan_n = nan_to_num_forward_func(ms.Tensor(x), None, inf_num, neg_inf_num)
        output_nan_n_grad = nan_to_num_backward_func(ms.Tensor(x), None, inf_num, neg_inf_num)

        # nan=None will update to -> nan=0.0
        expect_nan_n = generate_expect_forward_output(x, None, inf_num, neg_inf_num)
        expect_nan_n_grad = generate_expect_backward_output(x, None, inf_num, neg_inf_num)

        np.allclose(output_nan_n.asnumpy(), expect_nan_n, rtol=1e-5, equal_nan=True)
        np.allclose(output_nan_n_grad.asnumpy(), expect_nan_n_grad, rtol=1e-5, equal_nan=True)

    elif mode == 'KBK':
        output = (jit(nan_to_num_forward_func, jit_level="O0"))(ms.Tensor(x), nan_num, inf_num, neg_inf_num)
        output_grad = (jit(nan_to_num_backward_func, jit_level="O0"))(ms.Tensor(x), nan_num, inf_num, neg_inf_num)
    else:
        output = (jit(nan_to_num_forward_func, backend="GE"))(ms.Tensor(x), nan_num, inf_num, neg_inf_num)
        output_grad = (jit(nan_to_num_backward_func, backend="GE"))(ms.Tensor(x), nan_num, inf_num, neg_inf_num)
    np.allclose(output.asnumpy(), expect, rtol=1e-5, equal_nan=True)
    np.allclose(output_grad.asnumpy(), expect_grad, rtol=1e-5, equal_nan=True)


@arg_mark(
    plat_marks=['platform_ascend910b', 'cpu_linux', 'cpu_windows', 'cpu_macos'],
    level_mark='level1', card_mark='onecard', essential_mark='unessential', mem_peak=36.0)
def test_nan_to_num_dynamic_shape():
    """
    Feature: dynamic shape forward, backward features.
    Description: test nan_to_num forward with dynamic shape.
    Expectation: expect correct result.
    """
    tensor_1 = ms.Tensor(np.array([float('nan'), float('inf'), -float('inf'), 3.14]).astype(np.float32))
    nan_num_1 = 1.0
    inf_num_1 = 2.0
    neg_inf_num_1 = 3.0
    tensor_2 = ms.Tensor(
        np.array([[float('nan'), float('inf'), -float('inf')], [3.14, float('inf'), -float('inf')]]).astype(np.float32))

    nan_num_2 = 0.0
    inf_num_2 = -255.0
    neg_inf_num_2 = 256.0
    TEST_OP(nan_to_num_forward_func, [[tensor_1, nan_num_1, inf_num_1, neg_inf_num_1],
                                      [tensor_2, nan_num_2, inf_num_2, neg_inf_num_2]])


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=36.0)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_nan_to_num_bfloat16(mode):
    """
    Feature: test nan_to_num functional API.
    Description: testcase for nan_to_num functional API.
    Expectation: the result match with expected result.
    """
    x = generate_random_input(np.float32)
    nan_num = 1.0
    inf_num = 2.0
    neg_inf_num = 3.0
    expect = generate_expect_forward_output(x, nan_num, inf_num, neg_inf_num)
    expect_grad = generate_expect_backward_output(x, nan_num, inf_num, neg_inf_num)

    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        output = nan_to_num_forward_func(ms.Tensor(x, dtype=ms.bfloat16), nan_num, inf_num, neg_inf_num)
        output_grad = nan_to_num_backward_func(ms.Tensor(x, dtype=ms.bfloat16), nan_num, inf_num, neg_inf_num)
    else:
        output = (jit(nan_to_num_forward_func, jit_level="O0"))(ms.Tensor(x, dtype=ms.bfloat16),
                                                                nan_num, inf_num, neg_inf_num)
        output_grad = (jit(nan_to_num_backward_func, jit_level="O0"))(
            ms.Tensor(x, dtype=ms.bfloat16), nan_num, inf_num, neg_inf_num)

    np.allclose(output.float().asnumpy(), expect, 0.004, 0.004, equal_nan=True)
    np.allclose(output_grad.float().asnumpy(), expect_grad, 0.004, 0.004, equal_nan=True)
