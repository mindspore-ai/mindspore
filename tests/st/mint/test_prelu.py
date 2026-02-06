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
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.mark_utils import arg_mark


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype), np.random.randn(shape[1]).astype(dtype)


def generate_expect_forward_output(x, weight):
    return np.where(x > 0, x, weight * x)


def generate_expect_backward_output(x, weight):
    input_grad = np.where(x > 0, 1, weight)
    axis = 0 if weight.size == x.shape[1] else None
    w_grad = np.where(x > 0, 0, x).sum(axis=axis)
    return input_grad, w_grad


def prelu_forward_func(x, weight):
    return mint.nn.functional.prelu(x, weight)


def prelu_forward_func2(x, num_parameters, init, dtype=None):
    m = mint.nn.PReLU(num_parameters=num_parameters, init=init, dtype=dtype)
    out = m(x)
    return out


def prelu_backward_func(x, weight):
    input_grad, weight_grad = ms.ops.grad(prelu_forward_func, (0, 1))(x, weight)
    return input_grad, weight_grad


def prelu_backward_func2(x, num_parameters, init, dtype=None):
    input_grad = ms.ops.grad(prelu_forward_func2, (0,))(x, num_parameters, init, dtype)
    return input_grad


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=2048.25)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_prelu_std(mode):
    """
    Feature: mint
    Description: Verify the result of mint function
    Expectation: success
    """
    x, weight = generate_random_input((2, 3), np.float32)

    expect_forward = generate_expect_forward_output(x, weight)
    expect_grad, expect_w_grad = generate_expect_backward_output(x, weight)
    weight2 = np.array(2.0).astype(np.float32)
    expect_forward2 = generate_expect_forward_output(x, weight2)
    expect_grad2, expect_w_grad2 = generate_expect_backward_output(x, weight2)

    if mode == 'pynative':
        ms.set_context(mode=ms.PYNATIVE_MODE)
        output_forward = prelu_forward_func(ms.Tensor(x), ms.Tensor(weight))
        output_grad, output_w_grad = prelu_backward_func(ms.Tensor(x), ms.Tensor(weight))
        output_forward2 = prelu_forward_func(ms.Tensor(x), ms.Tensor(weight2))
        output_grad2, output_w_grad2 = prelu_backward_func(ms.Tensor(x), ms.Tensor(weight2))
    else:
        output_forward = (jit(prelu_forward_func, jit_level="O0"))(
            ms.Tensor(x), ms.Tensor(weight))
        output_grad, output_w_grad = (jit(prelu_backward_func, jit_level="O0"))(
            ms.Tensor(x), ms.Tensor(weight))
        output_forward2 = (jit(prelu_forward_func, jit_level="O0"))(
            ms.Tensor(x), ms.Tensor(weight2))
        output_grad2, output_w_grad2 = (jit(prelu_backward_func, jit_level="O0"))(
            ms.Tensor(x), ms.Tensor(weight2))

    assert np.allclose(output_forward.asnumpy(), expect_forward)
    assert np.allclose(output_grad.asnumpy(), expect_grad)
    assert np.allclose(output_w_grad.asnumpy(), expect_w_grad)
    assert np.allclose(output_forward2.asnumpy(), expect_forward2)
    assert np.allclose(output_grad2.asnumpy(), expect_grad2)
    assert np.allclose(output_w_grad2.asnumpy(), expect_w_grad2)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
        card_mark='onecard', essential_mark='essential', mem_peak=3072.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_nn_prelu(mode):
    """
    Feature: mint
    Description: Verify the result of mint function
    Expectation: success
    """
    x, weight = generate_random_input((2, 3), np.float32)
    num_parameters = 3
    init = 0.5
    dtype = ms.float32
    weight = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    num_parameters2 = 1
    init2 = 1.0
    weight2 = np.array([1.0])

    expect_forward = generate_expect_forward_output(x, weight)
    expect_grad, _ = generate_expect_backward_output(x, weight)
    expect_forward2 = generate_expect_forward_output(x, weight2)
    expect_grad2, _ = generate_expect_backward_output(x, weight2)

    if mode == 'pynative':
        ms.set_context(mode=ms.PYNATIVE_MODE)
        output_forward = prelu_forward_func2(ms.Tensor(x), num_parameters, init, dtype)
        output_grad = prelu_backward_func2(ms.Tensor(x), num_parameters, init, dtype)
        output_forward2 = prelu_forward_func2(ms.Tensor(x), num_parameters2, init2)
        output_grad2 = prelu_backward_func2(ms.Tensor(x), num_parameters2, init2)
    else:
        output_forward = (jit(prelu_forward_func2, jit_level="O0"))(
            ms.Tensor(x), num_parameters, init, dtype)
        output_grad = (jit(prelu_backward_func2, jit_level="O0"))(
            ms.Tensor(x), num_parameters, init, dtype)
        output_forward2 = (jit(prelu_forward_func2, jit_level="O0"))(
            ms.Tensor(x), num_parameters2, init2)
        output_grad2 = (jit(prelu_backward_func2, jit_level="O0"))(
            ms.Tensor(x), num_parameters2, init2)

    assert np.allclose(output_forward.asnumpy(), expect_forward)
    assert np.allclose(output_grad.asnumpy(), expect_grad)
    assert np.allclose(output_forward2.asnumpy(), expect_forward2)
    assert np.allclose(output_grad2.asnumpy(), expect_grad2)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=2048.01)
def test_prelu_dynamic_shape():
    """
    Feature: Test leaky relu with dynamic shape in graph mode.
    Description: call mint.prelu with valid input and index.
    Expectation: return the correct value.
    """
    x1, w1 = generate_random_input((1, 5), np.float32)
    x2, w2 = generate_random_input((2, 3, 4), np.float32)

    TEST_OP(prelu_forward_func, [[ms.Tensor(x1), ms.Tensor(w1)], [ms.Tensor(x2), ms.Tensor(w2)]],
            disable_mode=['GRAPH_MODE_GE'],
            case_config={'disable_input_check': True,
                         'all_dim_zero': True})


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=2.0)
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_prelu_bfloat16(mode):
    """
    Feature: test ne functional API.
    Description: testcase for ne functional API.
    Expectation: the result match with expected result.
    """
    ms.set_context(mode=mode, device_target="Ascend")
    x, weight = generate_random_input((2, 3), np.float32)
    output = prelu_forward_func(ms.Tensor(x, dtype=ms.bfloat16), ms.Tensor(weight))
    expect = generate_expect_forward_output(x, weight).astype(np.float32)
    assert np.allclose(output.float().asnumpy(), expect, 0.004, 0.004)
