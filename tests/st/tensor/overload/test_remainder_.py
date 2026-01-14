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
from tests.mark_utils import arg_mark
from tests.st.utils.test_utils import run_with_cell
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.st.ops.test_tools.ops_binary_cases import ops_binary_cases, OpsBinaryCase


def _count_unequal_element(data_expected, data_me, rtol, atol):
    assert data_expected.shape == data_me.shape
    total_count = len(data_expected.flatten())
    error = np.abs(data_expected - data_me)
    greater = np.greater(error, atol + np.abs(data_me) * rtol)
    loss_count = np.count_nonzero(greater)
    assert (
        loss_count / total_count
    ) < rtol, "\ndata_expected_std:{0}\ndata_me_error:{1}\nloss:{2}".format(
        data_expected[greater], data_me[greater], error[greater]
    )


def allclose_nparray(data_expected, data_me, rtol, atol, equal_nan=True):
    if np.any(np.isnan(data_expected)):
        assert np.allclose(data_expected, data_me, rtol, atol, equal_nan=equal_nan)
    elif not np.allclose(data_expected, data_me, rtol, atol, equal_nan=equal_nan):
        _count_unequal_element(data_expected, data_me, rtol, atol)
    else:
        assert True


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)


@run_with_cell
def remainder__forward_func(input_x, other):
    temp = input_x * 1
    temp.remainder_(other)
    return temp


@run_with_cell
def imod_forward_func(input_x, other):
    temp = input_x * 1
    temp %= other
    return temp


@run_with_cell
def remainder__backward_func(input_x, other):
    if isinstance(other, ms.Tensor):
        grad_fn = ms.grad(remainder__forward_func, grad_position=(0, 1))
    else:
        grad_fn = ms.grad(remainder__forward_func, grad_position=(0,))
    return grad_fn(input_x, other)


@run_with_cell
def imod_backward_func(input_x, other):
    if isinstance(other, ms.Tensor):
        grad_fn = ms.grad(imod_forward_func, grad_position=(0, 1))
    else:
        grad_fn = ms.grad(imod_forward_func, grad_position=(0,))
    return grad_fn(input_x, other)


@ops_binary_cases(OpsBinaryCase(input_info=[((6, 64, 88, 160), np.float32), ((6, 64, 88, 160), np.float32)],
                                output_info=[((6, 64, 88, 160), np.float32), ((6, 64, 88, 160), np.float32),
                                             ((6, 64, 88, 160), np.float32)],
                                extra_info='SD5B'))
def remainder__binary_case1(input_binary_data=None, output_binary_data=None):
    input_x = Tensor(input_binary_data[0])
    other = Tensor(input_binary_data[1])
    output = remainder__forward_func(input_x, other)
    allclose_nparray(output_binary_data[0], output.asnumpy(), 1e-4, 1e-4)
    grads = remainder__backward_func(input_x, other)
    allclose_nparray(output_binary_data[1], grads[0].asnumpy(), 1e-4, 1e-4)
    allclose_nparray(output_binary_data[2], grads[1].asnumpy(), 1e-4, 1e-4)


@ops_binary_cases(OpsBinaryCase(input_info=[((6, 64, 88, 160), np.float32)],
                                output_info=[((6, 64, 88, 160), np.float32), ((6, 64, 88, 160), np.float32)],
                                extra_info='SD5B'))
def remainder__binary_case2(input_binary_data=None, output_binary_data=None):
    input_x = Tensor(input_binary_data[0])
    other = 9
    output = remainder__forward_func(input_x, other)
    allclose_nparray(output_binary_data[0], output.asnumpy(), 1e-4, 1e-4)
    grads = remainder__backward_func(input_x, other)
    allclose_nparray(output_binary_data[1], grads[0].asnumpy(), 1e-4, 1e-4)


@ops_binary_cases(OpsBinaryCase(input_info=[((8, 16, 64), np.float32), ((8, 16, 64), np.float32)],
                                output_info=[((8, 16, 64), np.float32), ((8, 16, 64), np.float32),
                                             ((8, 16, 64), np.float32)],
                                extra_info='SD5B'))
def imod_binary_case1(input_binary_data=None, output_binary_data=None):
    input_x = Tensor(input_binary_data[0])
    other = Tensor(input_binary_data[1])
    output = imod_forward_func(input_x, other)
    allclose_nparray(output_binary_data[0], output.asnumpy(), 1e-4, 1e-4)
    grads = imod_backward_func(input_x, other)
    allclose_nparray(output_binary_data[1], grads[0].asnumpy(), 1e-4, 1e-4)
    allclose_nparray(output_binary_data[2], grads[1].asnumpy(), 1e-4, 1e-4)


@ops_binary_cases(OpsBinaryCase(input_info=[((8, 16, 64), np.float32)],
                                output_info=[((8, 16, 64), np.float32), ((8, 16, 64), np.float32)],
                                extra_info='SD5B'))
def imod_binary_case2(input_binary_data=None, output_binary_data=None):
    input_x = Tensor(input_binary_data[0])
    other = 6
    output = imod_forward_func(input_x, other)
    allclose_nparray(output_binary_data[0], output.asnumpy(), 1e-4, 1e-4)
    grads = imod_backward_func(input_x, other)
    allclose_nparray(output_binary_data[1], grads[0].asnumpy(), 1e-4, 1e-4)


@arg_mark(
    plat_marks=['platform_ascend', 'platform_ascend910b'],
    level_mark='level0',
    card_mark='onecard',
    essential_mark='essential', mem_peak=166.01)
@pytest.mark.parametrize('mode', ['pynative']) # KBK mode is not support.
def test_tensor_remainder__normal(mode):
    """
    Feature: tensor.remainder_
    Description: Verify the result of tensor.remainder_
    Expectation: success
    """
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)

    remainder__binary_case1()
    remainder__binary_case2()


@arg_mark(
    plat_marks=['platform_gpu', 'platform_ascend', 'platform_ascend910b'],
    level_mark='level1',
    card_mark='onecard',
    essential_mark='unessential', mem_peak=4.0)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_tensor_imod_normal(mode):
    """
    Feature: tensor.__imod__
    Description: Verify the result of tensor.__imod__
    Expectation: success
    """
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    elif mode == 'KBK':
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level='O0')

    imod_binary_case1()
    imod_binary_case2()


@arg_mark(
    plat_marks=['platform_ascend', 'platform_ascend910b'],
    level_mark='level1',
    card_mark='onecard',
    essential_mark='unessential', mem_peak=2.0)
def test_tensor_remainder__dynamic():
    """
    Feature: tensor.remainder_
    Description: test tensor.remainder_ with dynamic shape
    Expectation: success
    """
    x1 = generate_random_input((2, 3), np.float32)
    y1 = generate_random_input((2, 3), np.float32)
    x2 = generate_random_input((4, 5, 6), np.float32)
    y2 = generate_random_input((4, 5, 6), np.float32)
    TEST_OP(
        remainder__forward_func,
        [[Tensor(x1), Tensor(y1)], [Tensor(x2), Tensor(y2)]],
        disable_mode=["GRAPH_MODE_GE", "GRAPH_MODE_O0"],
        case_config={'all_dim_zero': True},
        inplace_update=True
    )

    x3 = generate_random_input((2, 2), np.float32)
    y3 = np.random.randn()
    x4 = generate_random_input((3, 4, 5), np.float32)
    y4 = np.random.randn()
    TEST_OP(
        remainder__forward_func,
        [[Tensor(x3), y3], [Tensor(x4), y4]],
        disable_mode=["GRAPH_MODE_GE", "GRAPH_MODE_O0"],
        inplace_update=True
    )


@arg_mark(
    plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend', 'platform_ascend910b'],
    level_mark='level1',
    card_mark='onecard',
    essential_mark='unessential', mem_peak=4.0)
def test_tensor_imod_dynamic():
    """
    Feature: tensor.__imod__
    Description: test tensor.__imod__with dynamic shape
    Expectation: success
    """
    x1 = generate_random_input((2, 3, 4), np.float32)
    y1 = generate_random_input((2, 3, 4), np.float32)
    x2 = generate_random_input((5, 6), np.float32)
    y2 = generate_random_input((5, 6), np.float32)
    TEST_OP(
        imod_forward_func,
        [[Tensor(x1), Tensor(y1)], [Tensor(x2), Tensor(y2)]],
        case_config={'all_dim_zero': True},
        inplace_update=True
    )

    x3 = generate_random_input((3, 4, 5), np.float32)
    y3 = np.random.randn()
    x4 = generate_random_input((2, 6), np.float32)
    y4 = np.random.randn()
    TEST_OP(
        imod_forward_func,
        [[Tensor(x3), y3], [Tensor(x4), y4]],
        inplace_update=True
    )
