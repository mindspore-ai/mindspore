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
from mindspore import Tensor
from mindspore.ops import swiglu
from tests.st.utils import test_utils
from tests.mark_utils import arg_mark
from tests.st.ops.test_tools.test_op import TEST_OP

def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)

def generate_expect_forward_output(input_np, dim):
    in_np_0, in_np_1 = np.split(input_np, 2, axis=dim)
    input_0_cast = in_np_0.astype(np.float32)
    input_1_cast = in_np_1.astype(np.float32)
    expected = input_0_cast / (1 + np.exp(-input_0_cast))
    expected *= input_1_cast
    return expected

def generate_expect_backward_output(input_np, dim):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    in_np_0, in_np_1 = np.split(input_np, 2, axis=dim)
    input_0_cast = in_np_0.astype(np.float32)
    input_1_cast = in_np_1.astype(np.float32)
    swish_input_0 = input_0_cast / (1 + np.exp(-input_0_cast))
    sigmoid_input_0 = sigmoid(input_0_cast)
    dSwish_input_0 = sigmoid_input_0 + input_0_cast * sigmoid_input_0 * (1 - sigmoid_input_0)
    d_in_np_0 = dSwish_input_0 * input_1_cast
    d_in_np_1 = swish_input_0
    return np.concatenate([d_in_np_0, d_in_np_1], axis=dim)

@test_utils.run_with_cell
def swiglu_forward_func(x, dim):
    return swiglu(x, dim)

@test_utils.run_with_cell
def swiglu_backward_func(x, dim):
    return ms.grad(swiglu_forward_func, (0))(x, dim)


@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize('dim', [0, 2, -1])
@test_utils.run_test_with_On
def test_ops_swiglu_normal(context_mode, dim):
    """
    Feature: pyboost function.
    Description: test function swiglu forward and backward.
    Expectation: expect correct result.
    """
    ms.set_context(jit_level='O0')
    ms.context.set_context(mode=context_mode)
    # forward
    dim = dim
    x_np = generate_random_input((2, 3, 4, 6), np.float32)
    x_ms = Tensor(x_np, ms.float32)
    out_ms = swiglu_forward_func(x_ms, dim)
    out_expect = generate_expect_forward_output(x_np, dim)
    np.testing.assert_allclose(out_ms.asnumpy(), out_expect, atol=1e-4, rtol=1e-4)

    # backward
    out_grad_ms = swiglu_backward_func(x_ms, dim)
    out_grad_expect = generate_expect_backward_output(x_np, dim)
    np.testing.assert_allclose(out_grad_ms.asnumpy(), out_grad_expect, atol=1e-4, rtol=1e-4)


@arg_mark(plat_marks=['platform_ascend910b'],
          level_mark='level1', card_mark='onecard', essential_mark='unessential', mem_peak=20.0)
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize('dim', [2])
@test_utils.run_test_with_On
def test_ops_swiglu_bfloat16(context_mode, dim):
    """
    Feature: pyboost function.
    Description: test function swiglu vmap feature.
    Expectation: expect correct result.
    """
    ms.set_context(jit_level='O0')
    ms.context.set_context(mode=context_mode)
    # forward
    np.random.seed(4312)
    dim = dim
    x_np = generate_random_input((2, 3, 4), np.float32)
    x_ms = Tensor(x_np, ms.bfloat16)
    out_ms = swiglu_forward_func(x_ms, dim).float()
    out_expect = generate_expect_forward_output(x_np, dim)
    np.testing.assert_allclose(out_ms.asnumpy(), out_expect, atol=5e-3, rtol=5e-3)

    # backward
    out_grad_ms = swiglu_backward_func(x_ms, dim).float()
    out_grad_expect = generate_expect_backward_output(x_np, dim)
    np.testing.assert_allclose(out_grad_ms.asnumpy(), out_grad_expect, atol=5e-3, rtol=5e-3)


@arg_mark(plat_marks=['platform_ascend910b'],
          level_mark='level1', card_mark='onecard', essential_mark='unessential', mem_peak=20.0)
def test_ops_swiglu_dynamic_shape():
    """
    Feature: pyboost function.
    Description: test function swiglu with dynamic shape and dynamic rank.
    Expectation: return the correct value.
    """
    x1 = generate_random_input((2, 8, 4), np.float32)
    dim1 = 1
    x2 = generate_random_input((4, 5, 7, 9), np.float32)
    dim2 = -4
    TEST_OP(swiglu_forward_func, [[ms.Tensor(x1), dim1], [ms.Tensor(x2), dim2]],
            disable_case=['ScalarTensor'],
            disable_mode=['GRAPH_MODE_GE'])
