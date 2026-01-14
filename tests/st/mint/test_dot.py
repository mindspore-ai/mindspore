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
# pylint: disable=unused-variable
import numpy as np
import pytest
import mindspore as ms
from mindspore import mint, jit
from tests.mark_utils import arg_mark
from tests.st.ops.test_tools.test_op import TEST_OP


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype), np.random.randn(*shape).astype(dtype)


def generate_expect_forward_output(x, y):
    return np.sum(x * y.transpose())


def generate_expect_backward_output(x, y):
    return y, x


def dot_forward_func(x, y):
    return mint.dot(x, y)


def dot_backward_func(x, y):
    input_grad = ms.ops.grad(dot_forward_func, (0, 1))(x, y)
    return input_grad


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=2048.0)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_dot_std(mode):
    """
    Feature: mint
    Description: Verify the result of mint function
    Expectation: success
    """
    x, y = generate_random_input((10,), np.float32)
    expect_forward = generate_expect_forward_output(x, y)
    expect_grad = generate_expect_backward_output(x, y)

    if mode == 'pynative':
        ms.set_context(mode=ms.PYNATIVE_MODE)
        output_forward = dot_forward_func(ms.Tensor(x), ms.Tensor(y))
        output_grad = dot_backward_func(ms.Tensor(x), ms.Tensor(y))
    else:
        output_forward = (jit(dot_forward_func, backend="ms_backend", jit_level="O0"))(ms.Tensor(x), ms.Tensor(y))
        output_grad = (jit(dot_backward_func, backend="ms_backend", jit_level="O0"))(ms.Tensor(x), ms.Tensor(y))

    np.testing.assert_allclose(output_forward.asnumpy(), expect_forward, 1e-5, 1e-5)
    np.testing.assert_allclose(output_grad[0].asnumpy(), expect_grad[0], 1e-5, 1e-5)
    np.testing.assert_allclose(output_grad[1].asnumpy(), expect_grad[1], 1e-5, 1e-5)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
        card_mark='onecard', essential_mark='unessential', mem_peak=2048.00)
def test_dot_dynamic_shape():
    """
    Feature: Test dot with dynamic shape in graph mode.
    Description: call mint.dot with valid input and other.
    Expectation: return the correct value.
    """
    input1, other1 = generate_random_input((10,), np.float32)
    input2, other2 = generate_random_input((20,), np.float32)
    TEST_OP(dot_forward_func, [[ms.Tensor(input1), ms.Tensor(other1)], [ms.Tensor(input2), ms.Tensor(other2)]],
            disable_mode=['GRAPH_MODE_GE'], disable_case=['ScalarTensor'], case_config={'disable_input_check': True})


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=2.0)
def test_dot_bfloat16():
    """
    Feature: test dot functional API.
    Description: testcase for dot functional API.
    Expectation: the result match with expected result.
    """
    ms.set_context(mode=ms.PYNATIVE_MODE)
    x, y = generate_random_input((10,), np.float32)
    output = dot_forward_func(ms.Tensor(x, dtype=ms.bfloat16), ms.Tensor(y, dtype=ms.bfloat16))
    expect = generate_expect_forward_output(x, y).astype(np.float32)
    assert np.allclose(output.float().asnumpy(), expect, 0.004, 0.004)
