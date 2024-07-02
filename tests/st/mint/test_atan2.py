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
from mindspore import mint, jit, JitConfig
from tests.mark_utils import arg_mark
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype), np.random.randn(*shape).astype(dtype)


def generate_expect_forward_output(x, y):
    return np.arctan2(x, y)


def generate_expect_backward_output(x, y):
    recip = x * x + y * y
    return y / recip, -x / recip


def atan2_forward_func(x, y):
    return mint.atan2(x, y)


def atan2_backward_func(x, y):
    input_grad = ms.ops.grad(atan2_forward_func, (0, 1))(x, y)
    return input_grad


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_atan2_std(mode):
    """
    Feature: mint
    Description: Verify the result of mint function
    Expectation: success
    """
    x, y = generate_random_input((2, 3), np.float32)
    grad, _ = generate_random_input((2, 3), np.float32)

    expect_forward = generate_expect_forward_output(x, y)
    expect_grad = generate_expect_backward_output(x, y)

    if mode == 'pynative':
        ms.set_context(mode=ms.PYNATIVE_MODE)
        output_forward = atan2_forward_func(ms.Tensor(x), ms.Tensor(y))
        output_grad = atan2_backward_func(ms.Tensor(x), ms.Tensor(y))
    else:
        output_forward = (jit(atan2_forward_func, jit_config=JitConfig(jit_level="O0")))(ms.Tensor(x),
                                                                                         ms.Tensor(y))
        output_grad = (jit(atan2_backward_func, jit_config=JitConfig(jit_level="O0")))(ms.Tensor(x),
                                                                                       ms.Tensor(y))

    np.testing.assert_allclose(output_forward.asnumpy(), expect_forward, 1e-5, 1e-5)
    np.testing.assert_allclose(output_grad[0].asnumpy(), expect_grad[0], 1e-5, 1e-5)
    np.testing.assert_allclose(output_grad[1].asnumpy(), expect_grad[1], 1e-5, 1e-5)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_atan2_dynamic_shape():
    """
    Feature: Test atan2 with dynamic shape in graph mode.
    Description: call mint.atan2 with valid input and other.
    Expectation: return the correct value.
    """
    input1, other1 = generate_random_input((2, 3), np.float32)
    input2, other2 = generate_random_input((2, 3, 4), np.float32)

    TEST_OP(atan2_forward_func, [[ms.Tensor(input1), ms.Tensor(other1)], [ms.Tensor(input2), ms.Tensor(other2)]],
            'atan2_ext', disable_mode=['GRAPH_MODE'])


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_atan2_bfloat16(mode):
    """
    Feature: test atan2 functional API.
    Description: testcase for atan2 functional API.
    Expectation: the result match with expected result.
    """
    ms.set_context(mode=mode)
    x, y = generate_random_input((2, 3), np.float32)
    output = atan2_forward_func(ms.Tensor(x, dtype=ms.bfloat16), ms.Tensor(y, dtype=ms.bfloat16))
    expect = generate_expect_forward_output(x, y).astype(np.float32)
    assert np.allclose(output.float().asnumpy(), expect, 0.004, 0.004)
