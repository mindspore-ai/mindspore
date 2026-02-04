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
from tests.mark_utils import arg_mark

def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)

def generate_expect_backward_output(input_x, dim=0):
    if isinstance(input_x, ms.Tensor):
        input_x = input_x.asnumpy()
    return np.ones_like(input_x)

@test_utils.run_with_cell
def unbind_forward_func(input_x, dim=0):
    return mint.unbind(input_x, dim=dim)

@test_utils.run_with_cell
def unbind_backward_func(input_x, dim=0):
    input_grad = ms.grad(unbind_forward_func, (0,))(input_x, dim=dim)
    return input_grad

def ops_unbind_compare(output_tuple, expect_tuple):
    for output, expect in zip(output_tuple, expect_tuple):
        np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-4)

def ops_unbind_float32_1d_case():
    x_np = generate_random_input((2,), np.float32)
    input_x = ms.Tensor(x_np)
    dim = 0

    expect_tuple = []
    for i in range(input_x.shape[dim]):
        expect_tuple.append(x_np[i])
    output_tuple = unbind_forward_func(input_x, dim)
    ops_unbind_compare(output_tuple, expect_tuple)

def ops_unbind_float32_5d_case():
    x_np = generate_random_input((2, 3, 4, 5, 6), np.float32)
    input_x = ms.Tensor(x_np)
    dim = 3

    expect_tuple = []
    for i in range(input_x.shape[dim]):
        expect_tuple.append(x_np[:, :, :, i, :])
    output_tuple = unbind_forward_func(input_x, dim)
    ops_unbind_compare(output_tuple, expect_tuple)

def ops_unbind_int64_8d_case():
    x_np = generate_random_input((2, 3, 4, 5, 6, 7, 8, 9), np.int64)
    input_x = ms.Tensor(x_np)
    dim = 3

    expect_tuple = []
    for i in range(input_x.shape[dim]):
        expect_tuple.append(x_np[:, :, :, i, :, :, :, :])
    output_tuple = unbind_forward_func(input_x, dim)
    ops_unbind_compare(output_tuple, expect_tuple)

def ops_unbind_float32_dim_neg_case():
    x_np = generate_random_input((2, 3, 4), np.float32)
    input_x = ms.Tensor(x_np)
    dim = -2

    expect_tuple = []
    for i in range(input_x.shape[dim]):
        expect_tuple.append(x_np[:, i, :])
    output_tuple = unbind_forward_func(input_x, dim)
    ops_unbind_compare(output_tuple, expect_tuple)

@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='essential', mem_peak=1024.01)
@pytest.mark.parametrize('mode', ['pynative']) # Only support pynative mode
def test_ops_unbind_forward(mode):
    """
    Feature: mint.unbind
    Description: verify the result of unbind
    Expectation: success
    """
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    elif mode == 'KBK':
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level='O0')

    ops_unbind_float32_1d_case()
    ops_unbind_float32_5d_case()
    ops_unbind_int64_8d_case()
    ops_unbind_float32_dim_neg_case()

@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='essential', mem_peak=0.01)
@pytest.mark.parametrize('mode', ['pynative']) # Only support pynative mode
def test_ops_unbind_backward(mode):
    """
    Feature: mint.unbind
    Description: verify the result of unbind
    Expectation: success
    """
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    elif mode == 'KBK':
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level='O0')

    input_list = []
    input_list.append([ms.Tensor(generate_random_input((2,), np.float32)), 0])
    input_list.append([ms.Tensor(generate_random_input((2, 3, 4, 5, 6), np.float32)), 3])
    input_list.append([ms.Tensor(generate_random_input((2, 3, 4, 5, 6, 7, 8, 9), np.int64)), 3])
    input_list.append([ms.Tensor(generate_random_input((2, 3, 4), np.float32)), -2])

    for i in range(len(input_list)):
        input_x = input_list[i]
        expect = generate_expect_backward_output(*input_x)
        output = unbind_backward_func(*input_x)
        np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-4)

@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential', mem_peak=2.0)
def test_ops_unbind_dynamic():
    """
    Feature: mint.unbind
    Description: test function unbind_forward_func with dynamic shape and dynamic rank
    Expectation: success
    """
    input1 = generate_random_input((2, 3, 4, 5), np.float32)
    input2 = generate_random_input((3, 5, 2), np.float32)
    TEST_OP(
        unbind_forward_func,
        [[ms.Tensor(input1), 0], [ms.Tensor(input2), 1]],
        disable_mode=["GRAPH_MODE_GE", "GRAPH_MODE_O0"],
        disable_case=['EmptyTensor', 'ScalarTensor'],
        case_config={'disable_input_check': True},
    )
