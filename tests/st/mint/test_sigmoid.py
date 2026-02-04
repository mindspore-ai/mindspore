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
import numpy as np
import pytest

import mindspore as ms
from mindspore import mint
from tests.st.utils import test_utils
from tests.mark_utils import arg_mark
from tests.st.ops.test_tools.test_op import TEST_OP


class SigmoidNet(mint.nn.Sigmoid):
    def __init__(self):
        mint.nn.Sigmoid.__init__(self)

def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)

def generate_expect_forward_output(x):
    return 1./(1.+ np.exp(-x))

def generate_expect_backward_output(x):
    sig = generate_expect_forward_output(x)
    return sig * (1 - sig)

@test_utils.run_with_cell
def sigmoid_forward_func(x):
    net = SigmoidNet()
    return net(x)

@test_utils.run_with_cell
def sigmoid_backward_func(x):
    return ms.grad(sigmoid_forward_func, (0,))(x)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='unessential', mem_peak=2.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_mint_sigmoid_normal(mode):
    """
    Feature: sigmoid
    Description: Verify the result of mint.nn.functional.sigmoid
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
    forward_out = sigmoid_forward_func(x)
    assert np.allclose(forward_out.asnumpy(), expect_forward_out, rtol=1e-4, atol=1e-4)

    expect_backward_out = generate_expect_backward_output(input_np)
    backward_out = sigmoid_backward_func(x)
    assert np.allclose(backward_out.asnumpy(), expect_backward_out, rtol=1e-4, atol=1e-4)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=2048.0)
def test_mint_sigmoid_dyn():
    """
    Feature: pyboost function.
    Description: test mint.nn.functional.sigmoid with dynamic rank/shape.
    Expectation: success.
    """
    input1 = ms.Tensor(np.random.randn(2, 3, 4), ms.float32)
    input2 = ms.Tensor(np.random.randn(2, 3, 4, 4), ms.float32)
    TEST_OP(sigmoid_forward_func, [[input1], [input2]], disable_mode=["GRAPH_MODE_GE"])
