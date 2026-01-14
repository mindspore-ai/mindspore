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
def index_fill__forward_func(input_x, dim, index, value):
    temp = input_x * 1
    return temp.index_fill_(dim, index, value)


@run_with_cell
def index_fill__backward_func(input_x, dim, index, value):
    if isinstance(value, ms.Tensor):
        grad_fn = ms.grad(index_fill__forward_func, grad_position=(0, 3))
    else:
        grad_fn = ms.grad(index_fill__forward_func, grad_position=(0,))
    return grad_fn(input_x, dim, index, value)


@ops_binary_cases(OpsBinaryCase(input_info=[((6, 64, 88, 160), np.float32), ((3,), np.int64), ((), np.float32)],
                                output_info=[((6, 64, 88, 160), np.float32), ((6, 64, 88, 160), np.float32),
                                             ((), np.float32)],
                                extra_info='SD5B'))
def index_fill__binary_case1(input_binary_data=None, output_binary_data=None):
    input_x = Tensor(input_binary_data[0])
    dim = 1
    index = Tensor(input_binary_data[1])
    value = Tensor(input_binary_data[2])
    output = index_fill__forward_func(input_x, dim, index, value)
    allclose_nparray(output_binary_data[0], output.asnumpy(), 1e-4, 1e-4)
    grads = index_fill__backward_func(input_x, dim, index, value)
    allclose_nparray(output_binary_data[1], grads[0].asnumpy(), 1e-4, 1e-4)
    allclose_nparray(output_binary_data[2], grads[1].asnumpy(), 1e-4, 1e-4)


@ops_binary_cases(OpsBinaryCase(input_info=[((6, 64, 88, 160), np.float32), ((5,), np.int64)],
                                output_info=[((6, 64, 88, 160), np.float32), ((6, 64, 88, 160), np.float32)],
                                extra_info='SD5B'))
def index_fill__binary_case2(input_binary_data=None, output_binary_data=None):
    input_x = Tensor(input_binary_data[0])
    dim = -1
    index = Tensor(input_binary_data[1])
    value = 3.5
    output = index_fill__forward_func(input_x, dim, index, value)
    allclose_nparray(output_binary_data[0], output.asnumpy(), 1e-4, 1e-4)
    grads = index_fill__backward_func(input_x, dim, index, value)
    allclose_nparray(output_binary_data[1], grads[0].asnumpy(), 1e-4, 1e-4)


@arg_mark(
    plat_marks=['platform_ascend'],
    level_mark='level1',
    card_mark='onecard',
    essential_mark='essential', mem_peak=2048.01)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_tensor_index_fill__normal(mode):
    """
    Feature: tensor.index_fill_
    Description: Verify the result of tensor.index_fill_
    Expectation: success
    """
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    elif mode == 'KBK':
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level='O0')

    index_fill__binary_case1()
    index_fill__binary_case2()


@arg_mark(
    plat_marks=['platform_ascend'],
    level_mark='level1',
    card_mark='onecard',
    essential_mark='unessential', mem_peak=2048.00)
def test_tensor_index_fill__dynamic():
    """
    Feature: tensor.index_fill_
    Description: test tensor.index_fill_ with dynamic shape
    Expectation: success
    """
    input1 = Tensor(generate_random_input((2, 3), np.float32))
    dim1 = 0
    index1 = Tensor([0], dtype=ms.int64)
    value1 = Tensor(np.random.randn(), dtype=ms.float32)
    input2 = Tensor(generate_random_input((4, 5, 6), np.float32))
    dim2 = 1
    index2 = Tensor([1, 2], dtype=ms.int64)
    value2 = Tensor(np.random.randn(), dtype=ms.float32)

    TEST_OP(
        index_fill__forward_func,
        [[input1, dim1, index1, value1], [input2, dim2, index2, value2]],
        disable_mode=['GRAPH_MODE_GE'],
        disable_case=['ScalarTensor'],
        case_config={
            'disable_input_check': True,
        },
        inplace_update=True
    )

    input3 = Tensor(generate_random_input((2, 2), np.float32))
    dim3 = 0
    index3 = Tensor([0], dtype=ms.int64)
    value3 = np.random.randn()
    input4 = Tensor(generate_random_input((3, 4, 5), np.float32))
    dim4 = 1
    index4 = Tensor([1, 2], dtype=ms.int64)
    value4 = np.random.randn()
    TEST_OP(
        index_fill__forward_func,
        [[input3, dim3, index3, value3], [input4, dim4, index4, value4]],
        disable_mode=['GRAPH_MODE_GE'],
        disable_case=['ScalarTensor'],
        case_config={
            'disable_input_check': True,
        },
        inplace_update=True
    )
