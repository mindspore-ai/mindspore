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
from mindspore import mint
from tests.st.utils import test_utils
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.st.ops.test_tools.ops_binary_cases import ops_binary_cases, OpsBinaryCase
from tests.mark_utils import arg_mark

def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)

def generate_expect_forward_output(input_x, dim=None, correction=1, keepdim=False):
    if isinstance(input_x, ms.Tensor):
        input_x = input_x.asnumpy()
    return np.var(input_x, axis=dim, ddof=correction, keepdims=keepdim)

@test_utils.run_with_cell
def var_forward_func(input_x, dim=None, correction=1, keepdim=False):
    return mint.var(input_x, dim=dim, correction=correction, keepdim=keepdim)

@test_utils.run_with_cell
def var_backward_func(input_x, dim=None, correction=1, keepdim=False):
    input_grad = ms.grad(var_forward_func, (0,))(input_x, dim=dim, correction=correction, keepdim=keepdim)
    return input_grad


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='essential', mem_peak=1024.01)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_ops_var_normal(mode):
    """
    Feature: mint.var
    Description: verify the result of var
    Expectation: success
    """
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    elif mode == 'KBK':
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level='O0')
    # random forward
    input_list = []
    input_list.append([ms.Tensor(generate_random_input((2, 3), np.float32)), None, 0, False])
    input_list.append([ms.Tensor(generate_random_input((2, 3, 4), np.float32)), None, 1, True])
    input_list.append([ms.Tensor(generate_random_input((2, 3, 4, 5), np.float32)), None, 3, True])
    input_list.append([ms.Tensor(generate_random_input((2, 3, 4, 5, 6), np.float32)), None, 3, False])
    input_list.append([ms.Tensor(generate_random_input((2, 3), np.float32)), 0, 0, False])
    input_list.append([ms.Tensor(generate_random_input((2, 3, 4), np.float32)), (0, 1, 2), 1, True])
    input_list.append([ms.Tensor(generate_random_input((2, 3, 4, 5), np.float32)), (0, 1, 2), 3, True])
    input_list.append([ms.Tensor(generate_random_input((2, 3, 4, 5, 6), np.float32)), (0, 4), 3, False])

    for i in range(len(input_list)):
        input_x = input_list[i]
        expect = generate_expect_forward_output(*input_x)
        output = var_forward_func(*input_x)
        np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-4)

def ops_var_binary_compare(input_binary_data, output_binary_data, dim, correction, keepdim):
    output = var_forward_func(ms.Tensor(input_binary_data[0]), dim, correction, keepdim)
    assert np.allclose(output.asnumpy(), output_binary_data[0], 1e-04, 1e-04)
    grad = var_backward_func(ms.Tensor(input_binary_data[0]), dim, correction, keepdim)
    assert np.allclose(grad.asnumpy(), output_binary_data[1], 1e-04, 1e-04)

@ops_binary_cases(OpsBinaryCase(input_info=[((2, 3), np.float32)],
                                output_info=[((), np.float32), ((2, 3), np.float32)],
                                extra_info='SD5B'))
def ops_var_binary_case1(input_binary_data=None, output_binary_data=None):
    ops_var_binary_compare(input_binary_data, output_binary_data, None, 0, False)

@ops_binary_cases(OpsBinaryCase(input_info=[((2, 3, 4), np.float32)],
                                output_info=[((1, 1, 1), np.float32), ((2, 3, 4), np.float32)],
                                extra_info='SD5B'))
def ops_var_binary_case2(input_binary_data=None, output_binary_data=None):
    ops_var_binary_compare(input_binary_data, output_binary_data, None, 1, True)

@ops_binary_cases(OpsBinaryCase(input_info=[((2, 3, 4, 5), np.float32)],
                                output_info=[((1, 1, 1, 1), np.float32), ((2, 3, 4, 5), np.float32)],
                                extra_info='SD5B'))
def ops_var_binary_case3(input_binary_data=None, output_binary_data=None):
    ops_var_binary_compare(input_binary_data, output_binary_data, None, 3, True)

@ops_binary_cases(OpsBinaryCase(input_info=[((2, 3, 4, 5, 6), np.float32)],
                                output_info=[((), np.float32), ((2, 3, 4, 5, 6), np.float32)],
                                extra_info='SD5B'))
def ops_var_binary_case4(input_binary_data=None, output_binary_data=None):
    ops_var_binary_compare(input_binary_data, output_binary_data, None, 3, False)

@ops_binary_cases(OpsBinaryCase(input_info=[((2, 3), np.float32)],
                                output_info=[((3,), np.float32), ((2, 3), np.float32)],
                                extra_info='SD5B'))
def ops_var_binary_case5(input_binary_data=None, output_binary_data=None):
    ops_var_binary_compare(input_binary_data, output_binary_data, 0, 0, False)

@ops_binary_cases(OpsBinaryCase(input_info=[((2, 3, 4), np.float32)],
                                output_info=[((1, 1, 1), np.float32), ((2, 3, 4), np.float32)],
                                extra_info='SD5B'))
def ops_var_binary_case6(input_binary_data=None, output_binary_data=None):
    ops_var_binary_compare(input_binary_data, output_binary_data, (0, 1, 2), 1, True)

@ops_binary_cases(OpsBinaryCase(input_info=[((2, 3, 4, 5), np.float32)],
                                output_info=[((1, 1, 1, 5), np.float32), ((2, 3, 4, 5), np.float32)],
                                extra_info='SD5B'))
def ops_var_binary_case7(input_binary_data=None, output_binary_data=None):
    ops_var_binary_compare(input_binary_data, output_binary_data, (0, 1, 2), 3, True)

@ops_binary_cases(OpsBinaryCase(input_info=[((2, 3, 4, 5, 6), np.float32)],
                                output_info=[((3, 4, 5), np.float32), ((2, 3, 4, 5, 6), np.float32)],
                                extra_info='SD5B'))
def ops_var_binary_case8(input_binary_data=None, output_binary_data=None):
    ops_var_binary_compare(input_binary_data, output_binary_data, (0, 4), 3, False)

@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential', mem_peak=4.0)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_ops_var_binary_cases(mode):
    """
    Feature: mint.var
    Description: verify the result of var with binary cases
    Expectation: success
    """
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    elif mode == 'KBK':
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level='O0')

    ops_var_binary_case1()
    ops_var_binary_case2()
    ops_var_binary_case3()
    ops_var_binary_case4()
    ops_var_binary_case5()
    ops_var_binary_case6()
    ops_var_binary_case7()
    ops_var_binary_case8()

@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential', mem_peak=2048.00)
def test_ops_var_dynamic():
    """
    Feature: mint.var
    Description: test function var_forward_func with dynamic shape and dynamic rank
    Expectation: success
    """
    input1 = generate_random_input((2, 3, 4, 5), np.float32)
    input2 = generate_random_input((3, 5, 2), np.float32)
    TEST_OP(
        var_forward_func,
        [[ms.Tensor(input1), 0, 1, False], [ms.Tensor(input2), 1, 3, True]],
        case_config={'disable_input_check': True},
        disable_mode=["GRAPH_MODE_GE"],
    )
