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
from mindspore.ops.auto_generate import kthvalue
from tests.st.utils import test_utils
from tests.mark_utils import arg_mark
from tests.st.ops.test_tools.test_op import TEST_OP

def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)

def generate_expect_forward_output(input_array, k, axis=-1, keepdim=False):
    sorted_indices = np.argsort(input_array, axis=axis)
    sorted_array = np.take_along_axis(input_array, sorted_indices, axis=axis)
    k_index = k - 1  # Convert 1-based index to 0-based
    values = np.take(sorted_array, k_index, axis=axis)
    indices = np.take(sorted_indices, k_index, axis=axis)
    if keepdim:
        values = np.expand_dims(values, axis=axis)
        indices = np.expand_dims(indices, axis=axis)
    return values, indices

def generate_expect_backward_output(input_array, indices, grad_output, axis=-1, keepdim=False):
    grad_input = np.zeros_like(input_array)
    if not keepdim:
        expanded_grad_output = np.expand_dims(grad_output, axis=axis)
        expanded_indices = np.expand_dims(indices, axis=axis)
    else:
        expanded_grad_output = grad_output
        expanded_indices = indices
    np.put_along_axis(grad_input, expanded_indices, expanded_grad_output, axis=axis)
    return grad_input


@test_utils.run_with_cell
def kthvalue_forward_func(x, k, dim, keepdim):
    return kthvalue(x, k, dim, keepdim)

@test_utils.run_with_cell
def kthvalue_backward_func(x, k, dim, keepdim):
    return ms.grad(kthvalue_forward_func, (0))(x, k, dim, keepdim)

@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level0', card_mark='onecard', essential_mark='essential', mem_peak=2048.01)
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize('dim', [0, 2])
@pytest.mark.parametrize('keepdim', [True, False])
@test_utils.run_test_with_On
def test_ops_kthvalue_normal(context_mode, dim, keepdim):
    """
    Feature: pyboost function.
    Description: test function kthvalue forward and backward.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    if context_mode == ms.GRAPH_MODE:
        ms.set_context(jit_config={'jit_level': 'O0'})
    # forward
    shape = (2, 3, 4, 6)
    dim = dim
    k = shape[dim]-1
    keepdim = keepdim
    x_np = generate_random_input(shape, np.float32)
    x_ms = Tensor(x_np, ms.float32)
    out_ms = kthvalue_forward_func(x_ms, k, dim, keepdim)
    out_expect = generate_expect_forward_output(x_np, k, dim, keepdim)
    np.testing.assert_allclose(out_ms[0].asnumpy(), out_expect[0], atol=1e-4, rtol=1e-4)
    np.testing.assert_array_equal(out_ms[1].asnumpy(), out_expect[1])

    # backward
    grad_output = np.ones_like(out_expect[0])
    out_grad_ms = kthvalue_backward_func(x_ms, k, dim, keepdim)
    out_grad_expect = generate_expect_backward_output(x_np, out_expect[1], grad_output, dim, keepdim)
    np.testing.assert_allclose(out_grad_ms[0].asnumpy(), out_grad_expect[0], atol=1e-4, rtol=1e-4)
    np.testing.assert_array_equal(out_grad_ms[1].asnumpy(), out_grad_expect[1])


@arg_mark(plat_marks=['platform_ascend910b'],
          level_mark='level1', card_mark='onecard', essential_mark='unessential', mem_peak=36.0)
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize('dim', [0])
@pytest.mark.parametrize('keepdim', [True, False])
@test_utils.run_test_with_On
def test_ops_kthvalue_bfloat16(context_mode, dim, keepdim):
    """
    Feature: pyboost function.
    Description: test function kthvalue vmap feature.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    if context_mode == ms.GRAPH_MODE:
        ms.set_context(jit_config={'jit_level': 'O0'})
    # forward
    shape = (2, 3)
    dim = dim
    k = shape[dim]-1
    keepdim = keepdim
    x_np = generate_random_input(shape, np.float32)
    x_ms = Tensor(x_np, ms.bfloat16)
    out_ms = kthvalue_forward_func(x_ms, k, dim, keepdim)
    out_expect = generate_expect_forward_output(x_np, k, dim, keepdim)
    np.testing.assert_allclose(out_ms[0].asnumpy(), out_expect[0], atol=4e-3, rtol=4e-3)
    np.testing.assert_allclose(out_ms[1].asnumpy(), out_expect[1], atol=4e-3, rtol=4e-3)

    # backward
    grad_output = np.ones_like(out_ms[0].asnumpy())
    out_grad_ms = kthvalue_backward_func(x_ms, k, dim, keepdim)
    out_grad_expect = generate_expect_backward_output(x_np, out_ms[1].asnumpy(), grad_output, dim, keepdim)
    np.testing.assert_allclose(out_grad_ms[0].asnumpy(), out_grad_expect[0], atol=4e-3, rtol=4e-3)
    np.testing.assert_allclose(out_grad_ms[1].asnumpy(), out_grad_expect[1], atol=4e-3, rtol=4e-3)


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level1', card_mark='onecard', essential_mark='unessential', mem_peak=2048.00)
def test_ops_kthvalue_dynamic_shape():
    """
    Feature: pyboost function.
    Description: test function kthvalue with dynamic shape and dynamic rank.
    Expectation: return the correct value.
    """
    x1 = generate_random_input((2, 8, 4), np.float32)
    k1 = 4
    dim1 = 1
    keepdim1 = False
    x2 = generate_random_input((4, 5, 7, 9), np.float32)
    k2 = 3
    dim2 = 0
    keepdim2 = True
    TEST_OP(kthvalue_forward_func,
            [[ms.Tensor(x1), k1, dim1, keepdim1], [ms.Tensor(x2), k2, dim2, keepdim2]],
            disable_mode=['GRAPH_MODE_GE'],
            disable_case=['EmptyTensor', 'ScalarTensor'])
