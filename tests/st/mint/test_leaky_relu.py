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
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP
from tests.mark_utils import arg_mark


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype), 0.1


def generate_expect_forward_output(x, negative_slope):
    return np.where(x > 0, x, negative_slope * x)


def generate_expect_backward_output(x, negative_slope):
    return np.where(x > 0, 1, negative_slope)


def leaky_relu_forward_func(x, negative_slope):
    return mint.leaky_relu(x, negative_slope)


def leaky_relu_backward_func(x, negative_slope):
    input_grad = ms.ops.grad(leaky_relu_forward_func, 0)(x, negative_slope)
    return input_grad


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_leaky_relu_std(mode):
    """
    Feature: mint
    Description: Verify the result of mint function
    Expectation: success
    """
    x, negative_slope = generate_random_input((2, 3), np.float32)
    grad, _ = generate_random_input((2, 3), np.float32)

    expect_forward = generate_expect_forward_output(x, negative_slope)
    expect_grad = generate_expect_backward_output(x, negative_slope)

    if mode == 'pynative':
        ms.set_context(mode=ms.PYNATIVE_MODE)
        output_forward = leaky_relu_forward_func(ms.Tensor(x), negative_slope)
        output_grad = leaky_relu_backward_func(ms.Tensor(x), negative_slope)
    else:
        output_forward = (jit(leaky_relu_forward_func, jit_config=JitConfig(jit_level="O0")))(ms.Tensor(x),
                                                                                              negative_slope)
        output_grad = (jit(leaky_relu_backward_func, jit_config=JitConfig(jit_level="O0")))(ms.Tensor(x),
                                                                                            negative_slope)

    assert np.allclose(output_forward.asnumpy(), expect_forward)
    assert np.allclose(output_grad.asnumpy(), expect_grad)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_leaky_relu_dynamic_shape():
    """
    Feature: Test leaky relu with dynamic shape in graph mode.
    Description: call mint.leaky_relu with valid input and index.
    Expectation: return the correct value.
    """
    x1, ng1 = generate_random_input((2, 3), np.float32)
    x2, ng2 = generate_random_input((2, 3, 4), np.float32)

    TEST_OP(leaky_relu_forward_func, [[ms.Tensor(x1), ng1], [ms.Tensor(x2), ng2]], 'leaky_relu_ext',
            disable_input_check=True, disable_mode=['GRAPH_MODE'])


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_leaky_relu_bfloat16(mode):
    """
    Feature: test ne functional API.
    Description: testcase for ne functional API.
    Expectation: the result match with expected result.
    """
    ms.set_context(mode=mode, device_target="Ascend")
    x, negative_slope = generate_random_input((2, 3), np.float32)
    output = leaky_relu_forward_func(ms.Tensor(x, dtype=ms.bfloat16), negative_slope)
    expect = generate_expect_forward_output(x, negative_slope).astype(np.float32)
    assert np.allclose(output.float().asnumpy(), expect, 0.004, 0.004)
