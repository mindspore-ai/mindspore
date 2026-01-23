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
"""
Tests for fmod operation.
"""
import numpy as np
import pytest
import mindspore as ms
from mindspore import mint, Tensor, context
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.st.utils import test_utils
from tests.mark_utils import arg_mark


def generate_random_input(shape, dtype):
    x = np.random.randn(*shape).astype(dtype)
    y = np.random.randn(*shape).astype(dtype)
    return x, y


@test_utils.run_with_cell
def fmod_forward_func(x, y):
    return mint.fmod(x, y)


@test_utils.run_with_cell
def fmod_backward_func(x, y):
    grad_fn = ms.grad(fmod_forward_func, grad_position=(0, 1))
    return grad_fn(x, y)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=2048.00)
@pytest.mark.parametrize("mode", ['pynative', 'KBK'])
def test_fmod_forward_backward(mode):
    """
    Feature: pyboost function.
    Description: test function fmod forward.
    Expectation: expect correct result.
    """
    x = Tensor(np.array([5.0, 10.0, 15.0], dtype=np.float32))
    y = Tensor(np.array([3.0, 4.0, 5.0], dtype=np.float32))
    y2 = 3.0
    expect = [2.0, 2.0, 0.0]
    expect2 = [2.0, 1.0, 0.0]
    if mode == 'pynative':
        context.set_context(mode=ms.PYNATIVE_MODE)
    elif mode == 'KBK':
        context.set_context(mode=ms.GRAPH_MODE, jit_config={"jit_level": "O0"})
    else:
        context.set_context(mode=ms.GRAPH_MODE, jit_config={"jit_level": "O2"})
    output = fmod_forward_func(x, y)
    output2 = fmod_forward_func(x, y2)
    grad = fmod_backward_func(x, y)
    grad2 = fmod_backward_func(x, y2)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)
    np.testing.assert_allclose(output2.asnumpy(), expect2, rtol=1e-3)
    expect_backward = [1.0, 1.0, 1.0]
    expect_backward_2 = [1.0, 1.0, 1.0]
    np.testing.assert_allclose(grad[0].asnumpy(), expect_backward, rtol=1e-3)
    np.testing.assert_allclose(grad2[0].asnumpy(), expect_backward_2, rtol=1e-3)
    np.testing.assert_allclose(grad2[0].asnumpy(), expect_backward_2, rtol=1e-3)

    x_i = x.astype(ms.int64)
    y_f = 3.0
    output_i = fmod_forward_func(x_i, y_f)
    grad_i = fmod_backward_func(x_i, y_f)
    np.testing.assert_allclose(output_i.asnumpy(), expect2, rtol=1e-3)
    expect_backward_i = [1, 1, 1]
    np.testing.assert_allclose(grad_i[0].asnumpy(), expect_backward_i, rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
        card_mark='onecard', essential_mark='unessential', mem_peak=0.01)
def test_fmod_dynamic_shape_scalar():
    """
    Feature: Test fmod op.
    Description: Test fmod dynamic shape.
    Expectation: the result match with expected result.
    """
    x, y = generate_random_input((3, 4, 5, 6), np.float16)
    x = Tensor(x, dtype=ms.float16)
    y = 6
    x2, y2 = generate_random_input((3, 4), np.float16)
    x2 = Tensor(x2, dtype=ms.float16)
    y2 = 3
    TEST_OP(fmod_forward_func, [[x, y], [x2, y2]], disable_mode=['GRAPH_MODE_GE'])


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=2048.0)
def test_fmod_dynamic_shape_tensor():
    """
    Feature: Test fmod op.
    Description: Test fmod dynamic shape.
    Expectation: the result match with expected result.
    """
    x, y = generate_random_input((3, 4, 5, 6), np.float16)
    x = Tensor(x, dtype=ms.float16)
    y = Tensor(y, dtype=ms.float16)
    x2, y2 = generate_random_input((3, 4), np.float16)
    x2 = Tensor(x2, dtype=ms.float16)
    y2 = Tensor(y2, dtype=ms.float16)
    TEST_OP(fmod_forward_func, [[x, y], [x2, y2]], disable_mode=['GRAPH_MODE_GE'], case_config={'all_dim_zero': True})
