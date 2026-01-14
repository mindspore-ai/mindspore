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
from tests.st.common.random_generator import generate_numpy_ndarray_by_randn
from tests.mark_utils import arg_mark

def generate_random_input(shape, shape1, shape2):
    x = generate_numpy_ndarray_by_randn(shape, np.float32, 'x', seed=0)
    batch1 = generate_numpy_ndarray_by_randn(shape1, np.float32, 'batch1', seed=0)
    batch2 = generate_numpy_ndarray_by_randn(shape2, np.float32, 'batch2', seed=0)
    return x, batch1, batch2


def generate_expect_forward_output(input1, batch1, batch2, beta=1, alpha=1):
    return beta * input1 + alpha * (batch1 @ batch2)


def generate_expect_backward_output(input1, batch1, batch2, beta=1, alpha=1):
    out_grad = np.ones((batch1 @ batch2).shape)
    input_grad = beta * out_grad
    if input1.size != out_grad.size:
        input_grad = (beta * out_grad).sum(0).reshape(input1.shape)
    b1_grad = alpha * (out_grad @ batch2.transpose((0, 2, 1)))
    b2_grad = alpha * (batch1.transpose((0, 2, 1)) @ out_grad)
    return input_grad, b1_grad, b2_grad


def baddbmm_forward_func(input1, batch1, batch2, beta=1, alpha=1):
    return mint.baddbmm(input1, batch1, batch2, beta=beta, alpha=alpha)


def baddbmm_backward_func(input1, batch1, batch2, beta=1, alpha=1):
    output_grad, b1_grad, b2_grad = ms.ops.grad(baddbmm_forward_func, (0, 1, 2))(input1, batch1, batch2, beta, alpha)
    return output_grad, b1_grad, b2_grad


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=2.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_baddbmm_normal(mode):
    """
    Feature: Ops.
    Description: test op baddbmm.
    Expectation: expect correct result.
    """
    input_shape1 = (3, 4, 5)
    input_shape2 = (4, 5)
    batch1_shape = (3, 4, 2)
    batch2_shape = (3, 2, 5)
    beta = 1
    alpha = 2.0
    input1, batch1, batch2 = generate_random_input(input_shape1, batch1_shape, batch2_shape)
    input2, batch1, batch2 = generate_random_input(input_shape2, batch1_shape, batch2_shape)
    expect_forward = generate_expect_forward_output(input1, batch1, batch2)
    expect_forward2 = generate_expect_forward_output(input2, batch1, batch2, beta, alpha)
    expect_grad, expect_b1_grad, expect_b2_grad = generate_expect_backward_output(input1, batch1, batch2)
    expect_grad2, expect_b1_grad2, expect_b2_grad2 = generate_expect_backward_output(input2, batch1,
                                                                                     batch2, beta, alpha)

    if mode == 'pynative':
        ms.set_context(mode=ms.PYNATIVE_MODE)
        output_forward = baddbmm_forward_func(ms.Tensor(input1), ms.Tensor(batch1), ms.Tensor(batch2))
        output_forward2 = baddbmm_forward_func(ms.Tensor(input2), ms.Tensor(batch1), ms.Tensor(batch2), beta, alpha)
        output_grad, b1_grad, b2_grad = baddbmm_backward_func(ms.Tensor(input1),
                                                              ms.Tensor(batch1), ms.Tensor(batch2))
        output_grad2, b1_grad2, b2_grad2 = baddbmm_backward_func(ms.Tensor(input2),
                                                                 ms.Tensor(batch1), ms.Tensor(batch2), beta, alpha)
    else:
        output_forward = (jit(baddbmm_forward_func, backend="ms_backend", jit_level="O0"))(
            ms.Tensor(input1), ms.Tensor(batch1), ms.Tensor(batch2))
        output_forward2 = (jit(baddbmm_forward_func, backend="ms_backend", jit_level="O0"))(
            ms.Tensor(input2), ms.Tensor(batch1), ms.Tensor(batch2), beta, alpha)
        output_grad, b1_grad, b2_grad = (jit(baddbmm_backward_func, backend="ms_backend", jit_level="O0"))(
            ms.Tensor(input1), ms.Tensor(batch1), ms.Tensor(batch2))
        output_grad2, b1_grad2, b2_grad2 = (jit(baddbmm_backward_func, backend="ms_backend", jit_level="O0"))(
            ms.Tensor(input2), ms.Tensor(batch1), ms.Tensor(batch2), beta, alpha)
    np.testing.assert_allclose(output_forward.asnumpy(), expect_forward, 3e-3, 3e-3)
    np.testing.assert_allclose(output_grad.asnumpy(), expect_grad, 3e-3, 3e-3)
    np.testing.assert_allclose(b1_grad.asnumpy(), expect_b1_grad, 3e-3, 3e-3)
    np.testing.assert_allclose(b2_grad.asnumpy(), expect_b2_grad, 3e-3, 3e-3)
    np.testing.assert_allclose(output_forward2.asnumpy(), expect_forward2, 3e-3, 3e-3)
    np.testing.assert_allclose(output_grad2.asnumpy(), expect_grad2, 3e-3, 3e-3)
    np.testing.assert_allclose(b1_grad2.asnumpy(), expect_b1_grad2, 3e-3, 3e-3)
    np.testing.assert_allclose(b2_grad2.asnumpy(), expect_b2_grad2, 3e-3, 3e-3)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=2.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_baddbmm_input_with_first_dim_1(mode):
    """
    Feature: Ops.
    Description: test op baddbmm with input whose first dim is 1 (broadcasted batches).
    Expectation: expect correct result.
    """
    input_shape = (15,)
    batch1_shape = (3, 1, 12)
    batch2_shape = (1, 12, 15)
    beta = 1
    alpha = 2.0
    input1, batch1, batch2 = generate_random_input(input_shape, batch1_shape, batch2_shape)
    expect_forward = generate_expect_forward_output(input1, batch1, batch2, beta, alpha)
    expect_grad, expect_b1_grad, expect_b2_grad = generate_expect_backward_output(input1, batch1, batch2, beta, alpha)

    if mode == 'pynative':
        ms.set_context(mode=ms.PYNATIVE_MODE)
        output_forward = baddbmm_forward_func(ms.Tensor(input1), ms.Tensor(batch1), ms.Tensor(batch2), beta, alpha)
        input_grad, b1_grad, b2_grad = baddbmm_backward_func(
            ms.Tensor(input1), ms.Tensor(batch1), ms.Tensor(batch2), beta, alpha)
    else:
        output_forward = (jit(baddbmm_forward_func, backend="ms_backend", jit_level="O0"))(
            ms.Tensor(input1), ms.Tensor(batch1), ms.Tensor(batch2), beta, alpha)
        input_grad, b1_grad, b2_grad = (jit(baddbmm_backward_func, backend="ms_backend", jit_level="O0"))(
            ms.Tensor(input1), ms.Tensor(batch1), ms.Tensor(batch2), beta, alpha)

    np.testing.assert_allclose(output_forward.asnumpy(), expect_forward, 3e-3, 3e-3)
    np.testing.assert_allclose(input_grad.asnumpy(), expect_grad, 3e-3, 3e-3)
    np.testing.assert_allclose(b1_grad.asnumpy(), expect_b1_grad, 3e-3, 3e-3)
    np.testing.assert_allclose(b2_grad.asnumpy(), expect_b2_grad, 3e-3, 3e-3)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=24.0)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_baddbmm_bfloat16(mode):
    """
    Feature: test ne functional API.
    Description: testcase for ne functional API.
    Expectation: the result match with expected result.
    """
    input_shape1 = (1, 4, 5)
    batch1_shape = (3, 4, 2)
    batch2_shape = (3, 2, 5)
    beta = 1.0
    alpha = 0.5
    input1, batch1, batch2 = generate_random_input(input_shape1, batch1_shape, batch2_shape)
    expect_forward = generate_expect_forward_output(input1, batch1, batch2, beta, alpha)

    if mode == 'pynative':
        ms.set_context(mode=ms.PYNATIVE_MODE)
        output_forward = baddbmm_forward_func(ms.Tensor(input1, dtype=ms.bfloat16),
                                              ms.Tensor(batch1, dtype=ms.bfloat16),
                                              ms.Tensor(batch2, dtype=ms.bfloat16), beta, alpha)
    else:
        output_forward = (jit(baddbmm_forward_func, backend="ms_backend", jit_level="O0"))(
            ms.Tensor(input1, dtype=ms.bfloat16), ms.Tensor(batch1, dtype=ms.bfloat16),
            ms.Tensor(batch2, dtype=ms.bfloat16), beta, alpha)
    np.testing.assert_allclose(output_forward.float().asnumpy(), expect_forward, 0.125, 0.125)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
        card_mark='onecard', essential_mark='essential', mem_peak=2048.01)
def test_baddbmm_dynamic_shape():
    """
    Feature: Test dynamic shape.
    Description: test function div dynamic feature.
    Expectation: expect correct result.
    """
    input_shape1 = (3, 4, 5)
    batch1_shape = (3, 4, 2)
    batch2_shape = (3, 2, 5)
    beta = 1.0
    alpha = 0.5
    input_shape2 = (4, 4)
    batch1_shape2 = (5, 4, 2)
    batch2_shape2 = (5, 2, 4)
    beta2 = 1.0
    alpha2 = 2.0
    input1, batch1, batch2 = generate_random_input(input_shape1, batch1_shape, batch2_shape)
    input2, batch1_2, batch2_2 = generate_random_input(input_shape2, batch1_shape2, batch2_shape2)
    TEST_OP(baddbmm_forward_func,
            [[ms.Tensor(input1), ms.Tensor(batch1), ms.Tensor(batch2), beta, alpha],
             [ms.Tensor(input2), ms.Tensor(batch1_2), ms.Tensor(batch2_2), beta2, alpha2]],
            disable_mode=['GRAPH_MODE_GE'],
            disable_case=['EmptyTensor', 'ScalarTensor'],
            case_config={'disable_input_check': True})
