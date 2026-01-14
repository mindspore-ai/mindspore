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
from mindspore import ops, mint, jit
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.mark_utils import arg_mark

def generate_random_input(shape, dtype):
    return np.random.randint(1, 10, size=shape).astype(dtype)

def mm_forward_func(input_x, mat2):
    y = mint.mm(input_x, mat2)
    return y

def mm_backward_func(input_x, mat2):
    grad = ops.grad(mm_forward_func)(input_x, mat2)
    return grad

def generate_expect_forward_output(input_x, mat2):
    y = np.matmul(input_x, mat2)
    return y

def generate_expect_backward_output(input_x, mat2):
    grad = ops.grad(ops.matmul)(input_x, mat2)
    return grad

@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
        card_mark='onecard', essential_mark='essential', mem_peak=1024.01)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
@pytest.mark.parametrize('dtype', [np.float16, np.float32])
def test_ops(mode, dtype):
    """
    Feature: mint.mm
    Description: test mint.mm
    Expectation: expect correct shape result.
    """
    input_x = generate_random_input((2, 3), dtype)
    mat2 = generate_random_input((3, 4), dtype)
    expect_forward = generate_expect_forward_output(input_x, mat2)
    expect_grad = generate_expect_backward_output(ms.Tensor(input_x), ms.Tensor(mat2))
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        res = mm_forward_func(ms.Tensor(input_x), ms.Tensor(mat2))
        res_grad = mm_backward_func(ms.Tensor(input_x), ms.Tensor(mat2))
    elif mode == 'KBK':
        res = (jit(mm_forward_func, backend="ms_backend", jit_level="O0"))(ms.Tensor(input_x), ms.Tensor(mat2))
        res_grad = (jit(mm_backward_func, backend="ms_backend", jit_level="O0"))(ms.Tensor(input_x), ms.Tensor(mat2))
    else:
        res = (jit(mm_forward_func, backend="GE"))(ms.Tensor(input_x), ms.Tensor(mat2))
        res_grad = (jit(mm_backward_func, backend="GE"))(ms.Tensor(input_x), ms.Tensor(mat2))
    np.testing.assert_allclose(res.asnumpy(), expect_forward, rtol=1e-5)
    np.testing.assert_allclose(res_grad.asnumpy(), expect_grad.asnumpy(), rtol=1e-5)

@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
        card_mark='onecard', essential_mark='unessential', mem_peak=2048.00)
def test_mm_dynamic_shape():
    """
    Feature: Test mm with dynamic shape in graph mode.
    Description: call mint.mm with valid input_x.
    Expectation: return the correct value.
    """
    input1 = ms.Tensor(generate_random_input((2, 3), np.float32))
    mat21 = ms.Tensor(generate_random_input((3, 4), np.float32))

    input2 = ms.Tensor(generate_random_input((4, 5), np.float32))
    mat22 = ms.Tensor(generate_random_input((5, 6), np.float32))

    TEST_OP(mm_forward_func, [[input1, mat21], [input2, mat22]],
            disable_mode=['GRAPH_MODE_GE'],
            disable_case=['ScalarTensor'],
            case_config={'disable_input_check': True,
                         'disable_resize': True,
                         'all_dim_zero': True})
