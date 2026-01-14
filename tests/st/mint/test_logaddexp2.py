# Copyright 2025 Huawei Technologies Co., Ltd
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
import torch
import numpy as np
from tests.st.utils import test_utils
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.mark_utils import arg_mark
import mindspore as ms
from mindspore import mint, Tensor, jit, context, JitConfig, ops


@test_utils.run_with_cell
def logaddexp2_forward_func(input_x, other):
    return mint.logaddexp2(input_x, other)

@test_utils.run_with_cell
def logaddexp2_backward_func(input_x, other):
    return ops.grad(logaddexp2_forward_func, (0, 1))(input_x, other)

@test_utils.run_with_cell
def generate_expect_forward_output(input_x, other):
    x = torch.tensor(input_x, dtype=torch.float32)
    y = torch.tensor(other, dtype=torch.float32)
    return torch.logaddexp2(x, y)

@test_utils.run_with_cell
def generate_expect_backward_output(input_x, other):
    x = torch.tensor(input_x, requires_grad=True, dtype=torch.float32)
    y = torch.tensor(other, requires_grad=True, dtype=torch.float32)
    result = torch.logaddexp2(x, y)
    grad_output = torch.ones_like(result)
    result.backward(grad_output)
    return x.grad, y.grad

@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=2048.0)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_logaddexp2_forward_backward(mode):
    """
    Feature: logaddexp2
    Description: test op LogAddExp2
    Expectation: expect correct result.
    """
    # logaddexp2 forward
    x = np.array([-100, -200, -300])
    y = np.array([-1, -2, -3])
    input_x = Tensor(x, dtype=ms.float32)
    other = Tensor(y, dtype=ms.float32)
    expect = generate_expect_forward_output(x, y)
    if mode == 'pynative':
        context.set_context(mode=ms.PYNATIVE_MODE)
        out = logaddexp2_forward_func(input_x, other)
    elif mode == 'KBK':
        context.set_context(mode=ms.GRAPH_MODE)
        out = (jit(logaddexp2_forward_func, backend="ms_backend", jit_config=JitConfig(jit_level="O0")))(input_x, other)
    else:
        context.set_context(mode=ms.GRAPH_MODE)
        out = logaddexp2_forward_func(input_x, other)
    assert np.allclose(out.asnumpy(), expect.numpy())

    # logaddexp2 backward
    input_x = Tensor(np.array(x), dtype=ms.float32)
    other = Tensor(np.array(y), dtype=ms.float32)
    expect_input_grad, expect_other_grad = generate_expect_backward_output(x, y)
    if mode == 'pynative':
        context.set_context(mode=ms.PYNATIVE_MODE)
        input_grad, other_grad = logaddexp2_backward_func(input_x, other)
    elif mode == 'KBK':
        context.set_context(mode=ms.GRAPH_MODE)
        input_grad, other_grad = \
            (jit(logaddexp2_backward_func, backend="ms_backend", jit_config=JitConfig(jit_level="O0")))(input_x, other)
    else:
        context.set_context(mode=ms.GRAPH_MODE)
        input_grad, other_grad = logaddexp2_backward_func(input_x, other)
    assert np.allclose(input_grad.asnumpy(), expect_input_grad.detach().numpy())
    assert np.allclose(other_grad.asnumpy(), expect_other_grad.detach().numpy())

@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
        card_mark='onecard', essential_mark='unessential', mem_peak=2048.01)
def test_f_logaddexp2_dynamic():
    """
    Feature: test dynamiclogaddexp2.
    Description: test auto grad of op LogAddExp2.
    Expectation: expect correct result.
    """
    input_1 = Tensor(np.zeros((5, 5)), dtype=ms.float32)
    other_1 = Tensor(np.zeros((5, 5)), dtype=ms.float32)
    input_2 = Tensor(np.ones((3, 4, 5)), dtype=ms.float32)
    other_2 = Tensor(np.ones((3, 4, 5)), dtype=ms.float32)
    # dynamic string is not supported
    TEST_OP(mint.logaddexp2, [[input_1, other_1], [input_2, other_2]],
            disable_mode=["GRAPH_MODE_GE"],
            case_config={'all_dim_zero': True})
