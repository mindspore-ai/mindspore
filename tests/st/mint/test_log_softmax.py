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
import numpy as np
import pytest
import mindspore as ms
from mindspore import mint, jit
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.mark_utils import arg_mark

def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)


def generate_expect_forward_output(x, axis):
    axis = 1 if axis is None else axis
    return x - np.log(np.sum(np.exp(x), axis=axis, keepdims=True))


def log_softmax_forward_func(x, dim):
    return mint.nn.functional.log_softmax(x, dim)


def log_softmax_backward_func(x, dim):
    input_grad = ms.ops.grad(log_softmax_forward_func, 0)(x, dim)
    return input_grad


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
        card_mark='onecard', essential_mark='essential', mem_peak=1024.01)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_log_softmax_forward_backward(mode):
    """
    Feature: mint
    Description: Verify the result of mint function
    Expectation: success
    """
    x = np.array([[1, 2, 3], [2, 4, 5]]).astype(np.float32)
    dim = None
    expect_forward = generate_expect_forward_output(x, dim)
    expect_grad = np.array([[0.7299082, 0.2658146, -0.9957228],
                            [0.89464295, 0.22151065, -1.1161535]])
    if mode == 'pynative':
        ms.set_context(mode=ms.PYNATIVE_MODE)
        output_forward = log_softmax_forward_func(ms.Tensor(x), dim)
        output_grad = log_softmax_backward_func(ms.Tensor(x), dim)
    else:
        output_forward = (jit(log_softmax_forward_func, jit_level="O0"))(ms.Tensor(x), dim)
        output_grad = (jit(log_softmax_backward_func, jit_level="O0"))(ms.Tensor(x), dim)
    assert np.allclose(output_forward.asnumpy(), expect_forward)
    assert np.allclose(output_grad.asnumpy(), expect_grad)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=2.0)
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_log_softmax_bfloat16(mode):
    """
    Feature: test ne functional API.
    Description: testcase for ne functional API.
    Expectation: the result match with expected result.
    """
    ms.set_context(mode=mode, device_target="Ascend")
    x = generate_random_input((2, 3), np.float32)
    dim = 0
    output = log_softmax_forward_func(ms.Tensor(x, dtype=ms.bfloat16), dim)
    expect = generate_expect_forward_output(x, dim).astype(np.float32)
    assert np.allclose(output.float().asnumpy(), expect, 0.004, 0.004)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=2048.0)
def test_log_softmax_dynamic_shape():
    """
    Feature: Test log_softmax with dynamic shape in graph mode.
    Description: call mint.log_softmax with valid input and index.
    Expectation: return the correct value.
    """
    x1 = ms.Tensor(generate_random_input((2, 3), np.float32))
    dim1 = 0
    x2 = ms.Tensor(generate_random_input((2, 3, 4), np.float32))
    dim2 = 1
    TEST_OP(log_softmax_forward_func, [[x1, dim1], [x2, dim2]], disable_mode=['GRAPH_MODE_GE'])
