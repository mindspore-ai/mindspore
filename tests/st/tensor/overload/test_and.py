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
from mindspore import Tensor
from tests.mark_utils import arg_mark
from tests.st.utils.test_utils import run_with_cell
from tests.st.ops.test_tools.test_op import TEST_OP


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)


def generate_expect_forward_output(input_x, other):
    assert isinstance(input_x, np.ndarray)
    return input_x & other


@run_with_cell
def and_forward_func(input_x, other):
    return input_x & other


@arg_mark(
    plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend', 'platform_ascend910b'],
    level_mark='level1',
    card_mark='onecard',
    essential_mark='essential', mem_peak=1024.00)
@pytest.mark.parametrize('mode', ['pynative'])
def test_tensor_and_normal(mode):
    """
    Feature: tensor.__and__
    Description: verify the result of tensor.__and__
    Expectation: success
    """
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    elif mode == 'KBK':
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level='O0')

    input_list = []
    input_list.append([generate_random_input((2, 3), np.int32), generate_random_input((2, 3), np.int32)])
    input_list.append([generate_random_input((2, 3, 4), np.int32), generate_random_input((2, 3, 4), np.int8)])
    input_list.append([generate_random_input((2, 3, 4), np.uint8), generate_random_input((2, 3, 1), np.uint8)])
    input_list.append([generate_random_input((2, 3, 4, 5), np.int32), 3])
    input_list.append([generate_random_input((2, 3, 4, 5, 6), np.int32), True])

    for i in range(len(input_list)):
        input_np = input_list[i]
        inputs = [Tensor(x) if isinstance(x, np.ndarray) else x for x in input_np]
        expect = generate_expect_forward_output(*input_np)
        output = and_forward_func(*inputs)
        np.testing.assert_allclose(expect, output.asnumpy(), rtol=0, atol=0)


@arg_mark(
    plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend', 'platform_ascend910b'],
    level_mark='level1',
    card_mark='onecard',
    essential_mark='unessential', mem_peak=2.0)
def test_tensor_and_dynamic():
    """
    Feature: tensor.__and__
    Description: test the and_forward_func with dynamic shape
    Expectation: success
    """
    input1 = Tensor(generate_random_input((2, 3), np.int32))
    other1 = Tensor(generate_random_input((2, 3), np.int32))
    input2 = Tensor(generate_random_input((4, 5, 6), np.int32))
    other2 = Tensor(generate_random_input((4, 5, 6), np.int32))
    TEST_OP(
        and_forward_func,
        [[input1, other1], [input2, other2]],
        disable_mode=["GRAPH_MODE_GE", "GRAPH_MODE_O0"],
        case_config={'disable_grad': True,
                     'all_dim_zero': True}
    )

    input3 = Tensor(generate_random_input((2, 3), np.int32))
    other3 = 3
    input4 = Tensor(generate_random_input((2, 1, 4), np.int32))
    other4 = -5
    TEST_OP(
        and_forward_func,
        [[input3, other3], [input4, other4]],
        disable_mode=["GRAPH_MODE_GE", "GRAPH_MODE_O0"],
        case_config={'disable_grad': True,
                     'all_dim_zero': True}
    )
