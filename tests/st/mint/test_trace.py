# Copyright 2023 Huawei Technologies Co., Ltd
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
from mindspore import mint, Tensor, jit
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.mark_utils import arg_mark

def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)

def trace_forward_func(x):
    return mint.trace(x)

def trace_backward_func(x):
    return ms.grad(trace_forward_func, (0))(x)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=2048.0)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_mint_trace_ext_normal(mode):
    """
    Feature: mint.trace
    Description: Verify the result of mint.trace
    Expectation: success
    """
    x = Tensor(np.array([[10, 11, 12], [13, 14, 15], [16, 17, 18]]), ms.float32)
    expect_grad = Tensor(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), ms.float32)
    expect_output = Tensor(np.asarray([42]), ms.float32)

    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        output = trace_forward_func(x)
        grad = trace_backward_func(x)
    elif mode == 'KBK':
        output = (jit(trace_forward_func, backend="ms_backend", jit_level="O0"))(x)
        grad = (jit(trace_backward_func, backend="ms_backend", jit_level="O0"))(x)
    assert np.allclose(output.asnumpy(), expect_output.asnumpy())
    assert np.allclose(grad.asnumpy(), expect_grad.asnumpy())


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
        card_mark='onecard', essential_mark='essential', mem_peak=2048.01)
def test_mint_trace_ext_dynamic():
    """
    Feature: Test trace with dynamic shape in graph mode using TEST_OP.
    Description: call mint.trace with valid input and index.
    Expectation: return the correct value.
    """
    x1 = generate_random_input((3, 3), np.float32)
    x2 = generate_random_input((3, 4), np.float32)
    TEST_OP(mint.trace, [[ms.Tensor(x1)], [ms.Tensor(x2)]],
            disable_mode=["GRAPH_MODE_GE"],
            disable_case=['ScalarTensor'],
            case_config={'disable_input_check': True})
