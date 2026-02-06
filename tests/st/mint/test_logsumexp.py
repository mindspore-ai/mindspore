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
def logsumexp_forward_func(input_x, dim, keepdim=False):
    return mint.logsumexp(input_x, dim, keepdim=keepdim)

@test_utils.run_with_cell
def logsumexp_backward_func(input_x, dim, keepdim=False):
    return ops.grad(logsumexp_forward_func, (0))(input_x, dim, keepdim=keepdim)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
        card_mark='onecard', essential_mark='essential', mem_peak=1024.01)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_logsumexp_forward_backward(mode):
    """
    Feature: logsumexp
    Description: test op LogSumExp
    Expectation: expect correct result.
    """

    # logsumexp forward
    input_x = Tensor(np.array([[1, 2, 3], [4, 5, 6]]), dtype=ms.float32)
    dim = 0
    keepdim = False
    expect = np.array([4.04858732, 5.04858732, 6.04858732])
    if mode == 'pynative':
        context.set_context(mode=ms.PYNATIVE_MODE)
        out = logsumexp_forward_func(input_x, dim, keepdim)
    elif mode == 'KBK':
        context.set_context(mode=ms.GRAPH_MODE)
        out = (jit(logsumexp_forward_func, backend="ms_backend", jit_level="O0"))(input_x, dim, keepdim)
    else:
        context.set_context(mode=ms.GRAPH_MODE)
        out = logsumexp_forward_func(input_x, dim, keepdim)
    assert np.allclose(out.asnumpy(), expect)

    # logsumexp backward
    input_x = Tensor(np.array([[1, 2, 3], [4, 5, 6]]), dtype=ms.float32)
    dim = 0
    keepdim = False
    expect_input_grad = np.array([[0.0474258736, 0.0474258736, 0.0474258736], [0.952574134, 0.952574134, 0.952574134]])
    if mode == 'pynative':
        context.set_context(mode=ms.PYNATIVE_MODE)
        input_grad = logsumexp_backward_func(input_x, dim, keepdim)
    elif mode == 'KBK':
        context.set_context(mode=ms.GRAPH_MODE)
        input_grad = \
            (jit(logsumexp_backward_func, backend="ms_backend", jit_level="O0"))(input_x, dim, keepdim)
    else:
        context.set_context(mode=ms.GRAPH_MODE)
        input_grad = logsumexp_backward_func(input_x, dim, keepdim)
    assert np.allclose(input_grad.asnumpy(), expect_input_grad)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
        card_mark='onecard', essential_mark='unessential', mem_peak=2048.01)
def test_f_logsumexp_dynamic():
    """
    Feature: test dynamiclogsumexp.
    Description: test auto grad of op LogSumExp.
    Expectation: expect correct result.
    """
    input_1 = Tensor(np.zeros((5, 5)), dtype=ms.float32)
    dim_1 = (0)
    keepdim_1 = False
    input_2 = Tensor(np.ones((3, 4, 5)), dtype=ms.float32)
    dim_2 = (2)
    keepdim_2 = True
    # dynamic string is not supported
    TEST_OP(mint.logsumexp, [[input_1, dim_1, keepdim_1], [input_2, dim_2, keepdim_2]], disable_mode=["GRAPH_MODE_GE"])
