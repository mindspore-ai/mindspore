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
from tests.st.utils import test_utils
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.mark_utils import arg_mark
import mindspore as ms
from mindspore import mint, Tensor, jit, context, ops


@test_utils.run_with_cell
def logaddexp_forward_func(input_x, other):
    return mint.logaddexp(input_x, other)

@test_utils.run_with_cell
def logaddexp_backward_func(input_x, other):
    return ops.grad(logaddexp_forward_func, (0, 1))(input_x, other)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
        card_mark='onecard', essential_mark='essential', mem_peak=2048.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_logaddexp_forward_backward(mode):
    """
    Feature: logaddexp
    Description: test op LogAddExp
    Expectation: expect correct result.
    """

    # logaddexp forward
    input_x = Tensor(np.array([-100, -200, -300]), dtype=ms.float32)
    other = Tensor(np.array([-1, -2, -3]), dtype=ms.float32)
    expect = np.array([-1, -2, -3])
    if mode == 'pynative':
        context.set_context(mode=ms.PYNATIVE_MODE)
        out = logaddexp_forward_func(input_x, other)
    elif mode == 'KBK':
        context.set_context(mode=ms.GRAPH_MODE)
        out = (jit(logaddexp_forward_func, backend="ms_backend", jit_level="O0"))(input_x, other)
    else:
        context.set_context(mode=ms.GRAPH_MODE)
        out = logaddexp_forward_func(input_x, other)
    assert np.allclose(out.asnumpy(), expect)

    # logaddexp backward
    input_x = Tensor(np.array([-100, -200, -300]), dtype=ms.float32)
    other = Tensor(np.array([-1, -2, -3]), dtype=ms.float32)
    expect_input_grad = np.array([2.93874e-39, 2.93874e-39, 2.93874e-39])
    expect_other_grad = np.array([1, 1, 1])
    if mode == 'pynative':
        context.set_context(mode=ms.PYNATIVE_MODE)
        input_grad, other_grad = logaddexp_backward_func(input_x, other)
    elif mode == 'KBK':
        context.set_context(mode=ms.GRAPH_MODE)
        input_grad, other_grad = \
            (jit(logaddexp_backward_func, backend="ms_backend", jit_level="O0"))(input_x, other)
    else:
        context.set_context(mode=ms.GRAPH_MODE)
        input_grad, other_grad = logaddexp_backward_func(input_x, other)
    assert np.allclose(input_grad.asnumpy(), expect_input_grad)
    assert np.allclose(other_grad.asnumpy(), expect_other_grad)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
        card_mark='onecard', essential_mark='unessential', mem_peak=2048.00)
def test_f_logaddexp_dynamic():
    """
    Feature: test dynamiclogaddexp.
    Description: test auto grad of op LogAddExp.
    Expectation: expect correct result.
    """
    input_1 = Tensor(np.zeros((5, 5)), dtype=ms.float32)
    other_1 = Tensor(np.zeros((5, 5)), dtype=ms.float32)
    input_2 = Tensor(np.ones((3, 4, 5)), dtype=ms.float32)
    other_2 = Tensor(np.ones((3, 4, 5)), dtype=ms.float32)
    # dynamic string is not supported
    TEST_OP(mint.logaddexp, [[input_1, other_1], [input_2, other_2]],
            disable_mode=["GRAPH_MODE_GE"],
            case_config={'all_dim_zero': True})
