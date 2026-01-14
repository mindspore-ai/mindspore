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

import numpy as np
import pytest

import mindspore as ms
from mindspore import Tensor
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.mark_utils import arg_mark
from tests.st.utils.test_utils import run_with_cell

def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)


def generate_expect_forward_output(input_x, other):
    assert isinstance(input_x, np.ndarray)
    input_x //= other
    return input_x


@run_with_cell
def floor_divide__forward_func(input_x, other):
    return input_x.floor_divide_(other)


@run_with_cell
def ifloordiv_forward_func(input_x, other):
    input_x //= other
    return input_x


@arg_mark(
    plat_marks=['platform_ascend', 'platform_ascend910b'],
    level_mark='level1',
    card_mark='onecard',
    essential_mark='essential', mem_peak=4.01
)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_tensor_floor_divide__normal(mode):
    """
    Feature: Tensor.floor_divide_
    Description: Verify the result of Tensor.floor_divide_
    Expectation: success
    """
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    elif mode == 'KBK':
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level='O0')

    input_list = []
    input_list.append([generate_random_input((2, 3), np.float16), generate_random_input((2, 3), np.float16)])
    input_list.append([generate_random_input((2, 3, 4), np.float32), generate_random_input((2, 1, 4), np.float32)])
    input_list.append([generate_random_input((2, 3, 4), np.float64), 0.5])

    for i in range(len(input_list)):
        input_np = input_list[i]
        inputs = [Tensor(x) if isinstance(x, np.ndarray) else x for x in input_np]
        expect = generate_expect_forward_output(*input_np)
        output = floor_divide__forward_func(*inputs)
        np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-4)


@arg_mark(
    plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend', 'platform_ascend910b'],
    level_mark='level1',
    card_mark='onecard',
    essential_mark='unessential', mem_peak=4.01
)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_tensor_ifloordiv_normal(mode):
    """
    Feature: Tensor.__ifloordiv__
    Description: Verify the result of Tensor.__ifloordiv__
    Expectation: success
    """
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    elif mode == 'KBK':
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level='O0')

    input_list = []
    input_list.append([generate_random_input((2, 3), np.float16), generate_random_input((2, 3), np.float16)])
    input_list.append([generate_random_input((2, 3, 4), np.float32), generate_random_input((2, 1, 4), np.float32)])
    input_list.append([generate_random_input((2, 3, 4), np.float64), 0.5])

    for i in range(len(input_list)):
        input_np = input_list[i]
        inputs = [Tensor(x) if isinstance(x, np.ndarray) else x for x in input_np]
        expect = generate_expect_forward_output(*input_np)
        output = ifloordiv_forward_func(*inputs)
        np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-4)


@arg_mark(
    plat_marks=['platform_ascend', 'platform_ascend910b'],
    level_mark='level1',
    card_mark='onecard',
    essential_mark='unessential',
    mem_peak=4.01
)
def test_tensor_floor_divide__dynamic():
    """
    Feature: Tensor.floor_divide_
    Description: test Tensor.floor_divide_ with dynamic shape
    Expectation: success
    """
    x1 = generate_random_input((2, 3), np.float32)
    y1 = generate_random_input((2, 3), np.float32)
    x2 = generate_random_input((4, 5, 6), np.float32)
    y2 = generate_random_input((4, 5, 6), np.float32)
    TEST_OP(
        floor_divide__forward_func,
        [[Tensor(x1), Tensor(y1)], [Tensor(x2), Tensor(y2)]],
        disable_mode=["GRAPH_MODE_GE"],
        case_config={'disable_grad': True, 'all_dim_zero': True},
        inplace_update=True
    )

    x3 = generate_random_input((2, 2), np.float32)
    y3 = np.random.randn()
    x4 = generate_random_input((3, 4, 5), np.float32)
    y4 = np.random.randn()
    TEST_OP(
        floor_divide__forward_func,
        [[Tensor(x3), y3], [Tensor(x4), y4]],
        disable_mode=["GRAPH_MODE_GE"],
        case_config={'disable_grad': True, 'all_dim_zero': True},
        inplace_update=True
    )


@arg_mark(
    plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend', 'platform_ascend910b'],
    level_mark='level1',
    card_mark='onecard',
    essential_mark='unessential', mem_peak=2048.01
)
def test_tensor_ifloordiv_dynamic():
    """
    Feature: Tensor.__ifloordiv__
    Description: test Tensor.__ifloordiv__ with dynamic shape
    Expectation: success
    """
    x1 = generate_random_input((2, 3), np.float32)
    y1 = generate_random_input((2, 3), np.float32)
    x2 = generate_random_input((4, 5, 6), np.float32)
    y2 = generate_random_input((4, 5, 6), np.float32)
    TEST_OP(
        ifloordiv_forward_func,
        [[Tensor(x1), Tensor(y1)], [Tensor(x2), Tensor(y2)]],
        case_config={'disable_grad': True, 'all_dim_zero': True},
        inplace_update=True
    )
