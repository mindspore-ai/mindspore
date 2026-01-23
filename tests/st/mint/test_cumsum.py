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
""" test_cumsum """
import pytest
import numpy as np
import mindspore as ms
from mindspore import ops, jit
from mindspore.mint import cumsum
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.st.ops.ascend.grad import GradOfFirstInput
from tests.st.utils import test_utils
from tests.mark_utils import arg_mark

def generate_random_input(shape, dim):
    x = np.random.randn(*shape)
    return x, np.cumsum(x, dim)


def cumsum_func(x, dim):
    return cumsum(x, dim)


@test_utils.run_with_cell
def cumsum_forward_func(x, dim):
    return cumsum_func(x, dim)


def cumsum_bwd_func(x, dim):
    return ops.grad(cumsum_func, (0,))(x, dim)


@test_utils.run_with_cell
def cumsum_backward_func(x, dim):
    return cumsum_bwd_func(x, dim)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=2.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_cumsum_normal(mode):
    """
    Feature: Ops.
    Description: test op cumsum.
    Expectation: expect correct result.
    """
    test_shape = (2, 3, 4, 5)
    dim1 = 2
    dim2 = 0
    dim3 = -2

    x, expect = generate_random_input(test_shape, dim1)
    expect1 = np.flip(np.cumsum(np.flip(np.ones(test_shape), dim2), dim2), dim2)
    expect2 = np.flip(np.cumsum(np.flip(np.ones(test_shape), dim3), dim3), dim3)
    if mode == 'pynative':
        ms.set_context(mode=ms.PYNATIVE_MODE)
        output = cumsum_forward_func(ms.Tensor(x), dim1)
        output1 = cumsum_backward_func(ms.Tensor(x), dim2)
        output2 = cumsum_backward_func(ms.Tensor(x), dim3)
    else:
        output = (jit(cumsum_forward_func, backend="ms_backend", jit_level="O0"))(ms.Tensor(x), dim1)
        output1 = (jit(cumsum_backward_func, backend="ms_backend", jit_level="O0"))(ms.Tensor(x), dim2)
        output2 = (jit(cumsum_backward_func, backend="ms_backend", jit_level="O0"))(ms.Tensor(x), dim3)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-4)
    np.testing.assert_allclose(output1.asnumpy(), expect1, rtol=1e-4)
    np.testing.assert_allclose(output2.asnumpy(), expect2, rtol=1e-4)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=2.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_cumsum_rank_0(mode):
    """
    Feature: mint.cumsum operator
    Description: Validate cumsum behavior on zero-dimensional (scalar) inputs.
    Expectation: Correct forward output and gradient results.
    """
    dim1 = 0
    dim2 = -1

    x1, expect1 = generate_random_input([], dim1)
    x2, expect2 = generate_random_input([], dim2)
    if mode == 'pynative':
        output1 = cumsum_forward_func(ms.Tensor(x1), dim1)
        output2 = cumsum_forward_func(ms.Tensor(x2), dim2)
        grad1 = cumsum_backward_func(ms.Tensor(x1), dim1)
        grad2 = cumsum_backward_func(ms.Tensor(x2), dim2)
    else:
        output1 = (jit(cumsum_forward_func, backend="ms_backend", jit_level="O0"))(ms.Tensor(x1), dim1)
        output2 = (jit(cumsum_forward_func, backend="ms_backend", jit_level="O0"))(ms.Tensor(x2), dim2)
        grad1 = (jit(cumsum_backward_func, backend="ms_backend", jit_level="O0"))(ms.Tensor(x1), dim1)
        grad2 = (jit(cumsum_backward_func, backend="ms_backend", jit_level="O0"))(ms.Tensor(x2), dim2)

    np.testing.assert_allclose(output1.asnumpy(), expect1, rtol=1e-4)
    np.testing.assert_allclose(output2.asnumpy(), expect2, rtol=1e-4)

    np.testing.assert_allclose(grad1.asnumpy(), np.ones([]), rtol=1e-4)
    np.testing.assert_allclose(grad2.asnumpy(), np.ones([]), rtol=1e-4)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=4.0)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_cumsum_bfloat16(mode):
    """
    Feature: test ne functional API.
    Description: testcase for ne functional API.
    Expectation: the result match with expected result.
    """
    test_shape = (2, 3, 4)
    dim1 = 1
    x, expect = generate_random_input(test_shape, dim1)
    if mode == 'pynative':
        ms.set_context(mode=ms.PYNATIVE_MODE)
        output = cumsum_forward_func(ms.Tensor(x), dim1)
    else:
        output = (jit(cumsum_forward_func, jit_level="O0"))(ms.Tensor(x), dim1)
    np.testing.assert_allclose(output.float().asnumpy(), expect, rtol=5e-3, atol=5e-3)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
        card_mark='onecard', essential_mark='unessential', mem_peak=2048.01)
def test_cumsum_dynamic_shape():
    """
    Feature: Test dynamic shape.
    Description: test function div dynamic feature.
    Expectation: expect correct result.
    """
    dim1 = 0
    ms_data1, _ = generate_random_input((2, 3, 4), dim1)
    dim2 = 1
    ms_data2, _ = generate_random_input((3, 4, 5, 6), dim2)
    TEST_OP(cumsum_forward_func, [[ms.Tensor(ms_data1), dim1], [ms.Tensor(ms_data2), dim2]],
            disable_mode=['GRAPH_MODE_GE'])

@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='unessential', mem_peak=4.00)
def test_cumsum_grad_dynamic_rank():
    """
    Feature: mint.cumsum operator
    Description: Test cumsum ops grad with dynamic rank.
    Expectation: expect correct result.
    """
    ms.set_context(mode=ms.GRAPH_MODE, jit_config={"jit_level": "O0"})

    grad_op = GradOfFirstInput(ms.mint.cumsum, sens_param=False)
    input_placeholder = ms.Tensor(shape=None, dtype=ms.float32)

    test_shape1 = (10, 3, 4)
    dim1 = 0
    input1 = ms.ops.rand(*test_shape1, dtype=ms.float32)
    grad_op.set_inputs(input_placeholder, dim1)
    grad1 = grad_op(input1, dim1)
    expect_grad1 = np.flip(np.cumsum(np.flip(np.ones(test_shape1), dim1), dim1), dim1)
    np.testing.assert_allclose(grad1.asnumpy(), expect_grad1)

    test_shape2 = (5, 6, 9, 2)
    dim2 = -3
    input2 = ms.ops.rand(*test_shape2, dtype=ms.float32)
    grad_op.set_inputs(input_placeholder, dim2)
    grad2 = grad_op(input2, dim2)
    expect_grad2 = np.flip(np.cumsum(np.flip(np.ones(test_shape2), dim2), dim2), dim2)
    np.testing.assert_allclose(grad2.asnumpy(), expect_grad2)
