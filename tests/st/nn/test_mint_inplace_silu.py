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
from mindspore import mint, ops
from mindspore import Tensor
from tests.st.utils import test_utils
from tests.mark_utils import arg_mark
from tests.st.ops.test_tools.test_op import TEST_OP

class Net(ms.nn.Cell):
    def construct(self, x, inplace=False):
        op = mint.nn.SiLU(inplace)
        return op(x.clone())

@test_utils.run_with_cell
def SiLU_forward(x, inplace=False):
    return Net()(x, inplace)


@test_utils.run_with_cell
def SiLU_grad(x, inplace):
    grad = ops.GradOperation(get_all=True)
    return grad(Net())(x, inplace)


def np_sigmoid(x):
    return 1 / (1 + np.exp(-x))


def np_silu(x):
    return x * np_sigmoid(x)


def np_silu_grad(x):
    return (1 + np.exp(-x) + x * np.exp(-x)) * np.power(np_sigmoid(x), 2)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0',
          card_mark='onecard', essential_mark='unessential', mem_peak=2.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_silu_net(mode):
    """
    Feature: SiLU
    Description: Verify the result of mint.nn.SiLU network.
    Expectation: success
    """
    x_np = np.random.randn(1, 2, 4, 4).astype(np.float32)
    x = Tensor(x_np, dtype=ms.float32)
    output = SiLU_forward(x, inplace=False)
    expect_output_shape = (1, 2, 4, 4)
    expect_output = np_silu(x_np)
    expect_output_grad = np_silu_grad(x_np)
    assert np.allclose(expect_output_shape, output.shape)

    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        output = SiLU_forward(x, inplace=True)
        out_grad = SiLU_grad(x, inplace=True)
    elif mode == 'KBK':
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
        output = SiLU_forward(x, inplace=True)
        out_grad = SiLU_grad(x, inplace=True)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O2")
        output = SiLU_forward(x, inplace=False)
        out_grad = SiLU_grad(x, inplace=False)

    np.testing.assert_allclose(output.asnumpy(), expect_output, rtol=1e-3)
    np.testing.assert_allclose(out_grad[0].asnumpy(), expect_output_grad, rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential', mem_peak=2048.00)
def test_mint_silu_dyn():
    """
    Feature: Dynamic shape of SiLU.
    Description: test SiLU with dynamic rank/shape.
    Expectation: success.
    """
    in1 = Tensor(np.random.randn(1, 2, 4, 4).astype(np.float32))
    in2 = Tensor(np.random.randn(2, 1, 4).astype(np.float32))

    TEST_OP(SiLU_forward, [[in1], [in2]], disable_mode=["GRAPH_MODE_GE"])
