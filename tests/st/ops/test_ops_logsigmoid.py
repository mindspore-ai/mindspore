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
from mindspore import mint

from tests.st.utils import test_utils
from tests.mark_utils import arg_mark
from tests.st.ops.test_tools.test_op import TEST_OP

def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)

def generate_expect_forward_output(x):
    return np.log(1./(1.+np.exp(-x)))

def generate_expect_backward_output(x):
    return 1./(1.+np.exp(x))

@test_utils.run_with_cell
def logsigmoid_forward_func(x):
    return mint.nn.functional.logsigmoid(x)

@test_utils.run_with_cell
def logsigmoid_backward_func(x):
    return ms.grad(logsigmoid_forward_func, (0,))(x)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential',
        mem_peak=2048.00)
@pytest.mark.parametrize('mode', ['KBK', 'pynative'])
def test_mint_logsigmoid_normal(mode):
    """
    Feature: logsigmoid
    Description: Verify the result of mint.nn.functional.logsigmoid
    Expectation: success
    """
    if mode == 'pynative':
        ms.set_context(mode=ms.PYNATIVE_MODE)
    elif mode == 'KBK':
        ms.set_context(mode=ms.GRAPH_MODE)
        ms.set_context(jit_level='O0')

    input_np = generate_random_input((2, 3, 4), np.float32)
    x = ms.Tensor(input_np)

    expect_forward_out = generate_expect_forward_output(input_np)
    forward_out = logsigmoid_forward_func(x)
    assert np.allclose(forward_out.asnumpy(), expect_forward_out, rtol=1e-4, atol=1e-4)

    expect_backward_out = generate_expect_backward_output(input_np)
    backward_out = logsigmoid_backward_func(x)
    assert np.allclose(backward_out.asnumpy(), expect_backward_out, rtol=1e-4, atol=1e-4)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential', mem_peak=2048.01)
def test_mint_logsigmoid_dyn():
    """
    Feature: pyboost function.
    Description: test mint.nn.functional.sigmoid with dynamic rank/shape.
    Expectation: success.
    """
    input1 = ms.Tensor(np.random.randn(2, 3, 4), ms.float32)
    input2 = ms.Tensor(np.random.randn(2, 3, 4, 4), ms.float32)
    TEST_OP(logsigmoid_forward_func, [[input1], [input2]], disable_mode=["GRAPH_MODE_GE"])
