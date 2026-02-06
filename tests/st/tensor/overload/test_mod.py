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


@run_with_cell
def mod_forward_func(input_x, other):
    return input_x % other


@run_with_cell
def mod_backward_func(input_x, other):
    if isinstance(other, Tensor):
        grad_fn = ms.grad(mod_forward_func, grad_position=(0, 1))
    else:
        grad_fn = ms.grad(mod_forward_func, grad_position=(0,))
    return grad_fn(input_x, other)


@arg_mark(
    plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend', 'platform_ascend910b'],
    level_mark='level1',
    card_mark='onecard',
    essential_mark='essential', mem_peak=4.0)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_tensor_mod_normal(mode):
    """
    Feature: tensor.__mod__
    Description: verify the result of tensor.__mod__
    Expectation: success
    """
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    elif mode == 'KBK':
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level='O0')

    input_list = []
    input_list.append([Tensor(np.array([1.0, 2.0, 3.0]), ms.float32), Tensor(np.array([2.0, 2.0, -2.0]), ms.float32)])
    input_list.append([Tensor(np.array([3.5, 2.5, -4.5]), ms.float32), 3.0])
    input_list.append([Tensor(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, -6.0]]), ms.float32),
                       Tensor(np.array([2.0, -2.0]), ms.float32)])

    expect_list = []
    expect_list.append(np.array([1.0, 0.0, -1.0], dtype=np.float32))
    expect_list.append(np.array([0.5, 2.5, 1.5], dtype=np.float32))
    expect_list.append(np.array([[1.0, 0.0], [1.0, 0.0], [1.0, -0.0]], dtype=np.float32))

    expect_grad_list = []
    expect_grad_list.append([np.array([1.0, 1.0, 1.0], dtype=np.float32),
                             np.array([-0.0, -1.0, 2.0], dtype=np.float32)])
    expect_grad_list.append([np.array([1.0, 1.0, 1.0], dtype=np.float32)])
    expect_grad_list.append([np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]], dtype=np.float32),
                             np.array([-3.0, 0.0], dtype=np.float32)])
    for i in range(len(input_list)):
        input_x = input_list[i]
        expect = expect_list[i]
        expect_grads = expect_grad_list[i]
        output = mod_forward_func(*input_x)
        grads = mod_backward_func(*input_x)
        np.testing.assert_allclose(expect, output.asnumpy(), rtol=1e-4)
        for expect_grad, grad in zip(expect_grads, grads):
            np.testing.assert_allclose(expect_grad, grad.asnumpy(), rtol=1e-4)


@arg_mark(
    plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend', 'platform_ascend910b'],
    level_mark='level1',
    card_mark='onecard',
    essential_mark='unessential', mem_peak=4.0)
def test_tensor_mod_dynamic():
    """
    Feature: tensor.__mod__
    Description: test the mod_forward_func with dynamic shape
    Expectation: success
    """
    input1 = Tensor(np.array([1.0, 2.0, 3.0]), ms.float32)
    other1 = Tensor(np.array([2.0, 3.0, 5.0]), ms.float32)
    input2 = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), ms.float32)
    other2 = Tensor(np.array([[3.0, 4.0], [5.0, 6.0]]), ms.float32)
    TEST_OP(
        mod_forward_func,
        [[input1, other1], [input2, other2]]
    )

    input3 = Tensor(np.array([1.0, 2.0, 3.0]), ms.float32)
    other3 = 3.0
    input4 = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), ms.float32)
    other4 = -5.0
    TEST_OP(
        mod_forward_func,
        [[input3, other3], [input4, other4]]
    )
