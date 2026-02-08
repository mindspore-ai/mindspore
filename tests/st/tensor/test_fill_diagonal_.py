# Copyright 2022 Huawei Technologies Co., Ltd
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
from mindspore import Tensor, jit, JitConfig
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.mark_utils import arg_mark


def generate_random_input(shape, dtype):
    return np.random.randint(1, 9, shape).astype(dtype)


def fill_diagonal_forward_func(input_x, fill_value, wrap=False):
    return input_x.fill_diagonal_(fill_value, wrap)


def fill_diagonal__backward_func(input_x, fill_value, wrap=False):
    output_grad = ms.ops.grad(
        fill_diagonal_forward_func_withx1, (0,))(input_x, fill_value, wrap)
    return output_grad


def generate_expect_grad(grad_input):
    grad = np.ones(grad_input.shape)
    grad_out = Tensor(grad).fill_diagonal_(0, wrap=False)
    return grad_out


def fill_diagonal_forward_func_withx1(input_x, fill_value, wrap=False):
    input_x = input_x * 1
    return input_x.fill_diagonal_(fill_value, wrap)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='essential', mem_peak=1024.01)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
@pytest.mark.parametrize('wrap', [True, False])
def test_tensor_fill_diagonal(mode, wrap):
    """
    Feature: tensor.fill_diagonal
    Description: Verify the result of fill_diagonal
    Expectation: success
    """

    input_x = Tensor(np.zeros((6, 3)), ms.float32)
    fill_value = 5.0
    expect_grad_output = generate_expect_grad(input_x)
    expect_forward_output = np.array([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0],
                                      [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    if wrap:
        expect_forward_output = np.array([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0],
                                          [0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [0.0, 5.0, 0.0]])
    if mode == 'pynative':
        output_grad = fill_diagonal__backward_func(
            input_x, fill_value, wrap=wrap)
        output_forward = fill_diagonal_forward_func(
            input_x, fill_value, wrap=wrap)
    else:
        output_grad = (jit(fill_diagonal__backward_func, backend="ms_backend", jit_config=JitConfig(jit_level="O0")))(
            input_x, fill_value, wrap)
        output_forward = (jit(fill_diagonal_forward_func, backend="ms_backend", jit_config=JitConfig(jit_level="O0")))(
            input_x, fill_value, wrap)
    assert np.allclose(output_forward.asnumpy(), expect_forward_output)
    assert np.allclose(output_grad.asnumpy(), expect_grad_output.asnumpy())


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential', mem_peak=2048.01)
def test_fill_diagonal_dynamic_shape():
    """
    Feature: dynamic shape forward, backward features.
    Description: test zero forward with dynamic shape.
    Expectation: expect correct result.
    """
    tensor_x1 = ms.Tensor(generate_random_input((2, 3), np.float32))
    tensor_x2 = ms.Tensor(generate_random_input((3, 3, 3), np.float32))
    fill_value1 = 3
    fill_value2 = 4
    wrap1 = False
    wrap2 = True
    TEST_OP(fill_diagonal_forward_func_withx1,
            [[tensor_x1, fill_value1, wrap1],
             [tensor_x2, fill_value2, wrap2]],
            disable_mode=['GRAPH_MODE_GE'],
            disable_case=['ScalarTensor'])
