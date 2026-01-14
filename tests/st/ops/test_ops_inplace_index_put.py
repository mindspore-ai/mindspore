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
from mindspore import nn
from mindspore.common import mutable
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.st.utils import test_utils
from tests.mark_utils import arg_mark
from tests.st.pynative.utils import GradOfAllInputs


class InplaceIndexPutNet(nn.Cell):
    def construct(self, x, indices, values, accumulate):
        y = x * 1
        return y.index_put_(indices, values, accumulate)

def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)

def generate_expect_forward_output(x, indices, values, accumulate):
    out = np.copy(x)
    indices = tuple(indices)
    if accumulate:
        out[indices] += values
    else:
        out[indices] = values
    return out

def generate_expect_backward_output(x, indices, values, accumulate):
    input_grad = np.ones_like(x, np.float32)
    indices = tuple(indices)
    values_grad = input_grad[indices]
    if values.shape != values_grad.shape:
        values_grad = [np.sum(values_grad)]
    if not accumulate:
        zeros = np.zeros(values.shape)
        input_grad[indices] = zeros
    return input_grad, values_grad

@test_utils.run_with_cell
def inplace_index_put_forward_func(x, indices, values, accumulate=False):
    net = InplaceIndexPutNet()
    return net(x, indices, values, accumulate)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='essential', mem_peak=2048.01)
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_inplace_index_put_forward(context_mode):
    """
    Feature: mint
    Description: Verify the result of mint function
    Expectation: success
    """
    ms.context.set_context(mode=context_mode)
    if context_mode == ms.GRAPH_MODE:
        ms.set_context(jit_level='O0')
    test_shape = (2, 3, 4)
    indices = [np.array([0, 1, 1]).astype(np.int32), np.array([1, 0, 1]).astype(np.int32),
               np.array([1, 2, 1]).astype(np.int32)]
    values = np.array([3]).astype(np.float32)
    accumulate = False
    indices2 = [np.array([0, 1]).astype(np.int64), np.array([1, 2]).astype(np.int64),
                np.array([2, 3]).astype(np.int64)]
    values2 = np.array([2, 3]).astype(np.float32)
    accumulate2 = True
    x = generate_random_input(test_shape, np.float32)

    expect_forward = generate_expect_forward_output(x, indices, values, accumulate)
    expect_forward2 = generate_expect_forward_output(x, indices2, values2, accumulate2)

    output_forward = inplace_index_put_forward_func(
        ms.Tensor(x), [ms.Tensor(i) for i in indices], ms.Tensor(values), accumulate)
    output_forward2 = inplace_index_put_forward_func(
        ms.Tensor(x), [ms.Tensor(i) for i in indices2], ms.Tensor(values2), accumulate2)

    np.testing.assert_allclose(output_forward.asnumpy(), expect_forward, rtol=1e-5)
    np.testing.assert_allclose(output_forward2.asnumpy(), expect_forward2, rtol=1e-5)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_inplace_index_put_backward(context_mode):
    """
    Feature: mint
    Description: Verify the result of mint function
    Expectation: success
    """
    ms.context.set_context(mode=context_mode)
    if context_mode == ms.GRAPH_MODE:
        ms.set_context(jit_level='O0')
    test_shape = (2, 3, 4)
    indices = [np.array([0, 1, 1]).astype(np.int32), np.array([1, 0, 1]).astype(np.int32),
               np.array([1, 2, 1]).astype(np.int32)]
    values = np.array([3]).astype(np.float32)
    accumulate = False
    indices2 = [np.array([0, 1]).astype(np.int64), np.array([1, 2]).astype(np.int64),
                np.array([2, 3]).astype(np.int64)]
    values2 = np.array([2, 3]).astype(np.float32)
    accumulate2 = True
    x = generate_random_input(test_shape, np.float32)

    expect_grad, expect_v_grad = generate_expect_backward_output(x, indices, values, accumulate)
    expect_grad2, expect_v_grad2 = generate_expect_backward_output(x, indices2, values2, accumulate2)

    test_cell = InplaceIndexPutNet()
    grad_func = GradOfAllInputs(test_cell, sens_param=False)
    output_grad, output_v_grad = grad_func(ms.Tensor(x),
                                           [ms.Tensor(i) for i in indices], ms.Tensor(values), accumulate)
    output_grad2, output_v_grad2 = grad_func(ms.Tensor(x),
                                             [ms.Tensor(i) for i in indices2], ms.Tensor(values2), accumulate2)

    np.testing.assert_allclose(output_grad.asnumpy(), expect_grad, rtol=1e-5)
    np.testing.assert_allclose(output_v_grad.asnumpy(), expect_v_grad, rtol=1e-5)
    np.testing.assert_allclose(output_grad2.asnumpy(), expect_grad2, rtol=1e-5)
    np.testing.assert_allclose(output_v_grad2.asnumpy(), expect_v_grad2, rtol=1e-5)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=4.0)
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_inplace_index_put_bfloat16(context_mode):
    """
    Feature: test ne functional API.
    Description: testcase for ne functional API.
    Expectation: the result match with expected result.
    """
    ms.context.set_context(mode=context_mode)
    if context_mode == ms.GRAPH_MODE:
        ms.set_context(jit_level='O0')
    test_shape = (2, 3)
    indices = [np.array([0, 1]).astype(np.int64), np.array([1, 2]).astype(np.int64)]
    values = np.array([3.23]).astype(np.float32)
    accumulate = True
    x = generate_random_input(test_shape, np.float32)
    expect = generate_expect_forward_output(x, indices, values, accumulate)
    output = inplace_index_put_forward_func(
        ms.Tensor(x, dtype=ms.bfloat16), [ms.Tensor(i) for i in indices],
        ms.Tensor(values, dtype=ms.bfloat16), accumulate)
    assert np.allclose(output.float().asnumpy(), expect, 0.004, 0.004)


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b'],
          level_mark='level1', card_mark='onecard', essential_mark='essential', mem_peak=2048.00)
def test_inplace_index_put_dynamic_shape():
    """
    Feature: Test leaky relu with dynamic shape in graph mode.
    Description: call mint.inplace_index_put with valid input and index.
    Expectation: return the correct value.
    """
    x = generate_random_input((3, 4, 5, 6), np.float32)
    indices = [np.array([0, 1, 1]).astype(np.int32), np.array([1, 0, 1]).astype(np.int32),
               np.array([1, 2, 1]).astype(np.int32)]
    values = np.array([3]).astype(np.float32)
    accumulate = False
    x2 = generate_random_input((2, 3), np.float32)
    indices2 = [np.array([[1, 1, 1], [1, 0, 1]]).astype(np.bool_)]
    values2 = np.array([[2, 3]]).astype(np.float32)
    accumulate2 = True
    TEST_OP(inplace_index_put_forward_func,
            [[ms.Tensor(x), mutable([ms.Tensor(i) for i in indices]), ms.Tensor(values), accumulate],
             [ms.Tensor(x2), mutable([ms.Tensor(i) for i in indices2]), ms.Tensor(values2), accumulate2]],
            disable_mode=['GRAPH_MODE_GE'],
            disable_case=['ScalarTensor', 'GradByRequirement'],
            case_config={'disable_resize': True})
