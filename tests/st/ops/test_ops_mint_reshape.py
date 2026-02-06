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
from mindspore import mint, jit
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.st.utils import test_utils
from tests.mark_utils import arg_mark


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)


def generate_expect_forward_output(x, shape):
    return np.reshape(x, shape)


@test_utils.run_with_cell
def reshape_forward_func(x, shape):
    return mint.reshape(x, shape)


@test_utils.run_with_cell
def reshape_backward_func(x, shape):
    return ms.grad(reshape_forward_func)(x, shape)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
          card_mark='onecard', essential_mark='essential', mem_peak=1024.01)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_ops_mint_reshape(mode):
    """
    Feature: pyboost function.
    Description: test function reshape forward.
    Expectation: expect correct result.
    """
    input_tensor_list = [
        generate_random_input((2, 3, 4, 5), np.float32)
    ]
    shape_list = [(6, 2, 2, 5)]

    for i in range(len(input_tensor_list)):
        x = input_tensor_list[i]
        shape = shape_list[i]
        if mode == 'pynative':
            ms.context.set_context(mode=ms.PYNATIVE_MODE)
            output = reshape_forward_func(ms.Tensor(x), shape)
            out_grad = reshape_backward_func(ms.Tensor(x), shape)
        elif mode == 'KBK':
            output = (jit(reshape_forward_func, jit_level="O0"))(ms.Tensor(x), shape)
            out_grad = (jit(reshape_backward_func, jit_level="O0"))(ms.Tensor(x), shape)
        else:
            output = (jit(reshape_forward_func, backend="GE"))(ms.Tensor(x), shape)
            out_grad = (jit(reshape_backward_func, backend="GE"))(ms.Tensor(x), shape)
        expect = generate_expect_forward_output(x, shape)
        np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)
        np.testing.assert_allclose(out_grad.asnumpy(), 1, rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential', mem_peak=2048.00)
def test_mint_reshape_dynamic_shape():
    """
    Feature: Test mint.reshape with dynamic shape in pynative mode and KBK mode.
    Description: call mint.reshape with valid shape.
    Expectation: return the correct value.
    """
    ms_data1 = ms.Tensor(generate_random_input((2, 3, 4, 5), np.float32))
    shape1 = (6, 4, 5)

    ms_data2 = ms.Tensor(generate_random_input((5, 8, 7), np.float32))
    shape2 = (5, 2, 4, 7)
    TEST_OP(reshape_forward_func, [[ms_data1, shape1], [ms_data2, shape2]],
            disable_mode=['GRAPH_MODE_GE'],
            disable_case=['EmptyTensor', 'ScalarTensor'])
