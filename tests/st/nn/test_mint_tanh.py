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
from mindspore import Tensor, mint
from tests.st.utils import test_utils
from tests.mark_utils import arg_mark
from tests.st.ops.test_tools.test_op import TEST_OP


@test_utils.run_with_cell
def Tanh_forward(x):
    op = mint.nn.Tanh()
    return op(x)


@test_utils.run_with_cell
def Tanh_grad(x):
    op = mint.nn.Tanh()
    return ms.grad(op)(x)


def np_tanh(x):
    return np.tanh(x)


def np_tanh_grad(x):
    return 1 - np.power(np.tanh(x), 2)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=2.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK', 'GE'])
def test_tanh_net(mode):
    """
    Feature: Tanh
    Description: Verify the result of mint.nn.Tanh network.
    Expectation: success
    """
    x_np = np.random.randn(1, 2, 4, 4).astype(np.float32)
    x = Tensor(x_np, dtype=ms.float32)
    output = Tanh_forward(x)
    expect_output_shape = (1, 2, 4, 4)
    expect_output = np_tanh(x_np)
    expect_output_grad = np_tanh_grad(x_np)
    assert np.allclose(expect_output_shape, output.shape)

    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        output = Tanh_forward(x)
        out_grad = Tanh_grad(x)
    elif mode == 'KBK':
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
        output = Tanh_forward(x)
        out_grad = Tanh_grad(x)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O2")
        output = Tanh_forward(x)
        out_grad = Tanh_grad(x)

    np.testing.assert_allclose(output.asnumpy(), expect_output, rtol=1e-4)
    np.testing.assert_allclose(out_grad.asnumpy(), expect_output_grad, rtol=1e-4)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential', mem_peak=3072.00)
def test_mint_tanh_dyn():
    """
    Feature: Dynamic shape of Tanh.
    Description: test Tanh with dynamic rank/shape.
    Expectation: success.
    """
    in1 = Tensor(np.random.randn(1, 2, 4, 4).astype(np.float32))
    in2 = Tensor(np.random.randn(2, 1, 4).astype(np.float32))
    TEST_OP(Tanh_forward, [[in1], [in2]])
