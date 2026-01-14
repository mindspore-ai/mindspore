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
from mindspore import Tensor
from tests.st.utils import test_utils
from tests.mark_utils import arg_mark
from tests.st.ops.test_tools.test_op import TEST_OP


@test_utils.run_with_cell
def SiLU_forward(x):
    op = mint.nn.SiLU()
    return op(x)


@test_utils.run_with_cell
def SiLU_grad(x):
    op = mint.nn.SiLU()
    return ms.grad(op)(x)


def np_sigmoid(x):
    return 1 / (1 + np.exp(-x))


def np_silu(x):
    return x * np_sigmoid(x)


def np_silu_grad(x):
    return (1 + np.exp(-x) + x * np.exp(-x)) * np.power(np_sigmoid(x), 2)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential', mem_peak=4.0)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_silu_net(mode):
    """
    Feature: SiLU
    Description: Verify the result of mint.nn.SiLU network.
    Expectation: success
    """
    x_np = np.random.randn(1, 2, 4, 4).astype(np.float32)
    x = Tensor(x_np, dtype=ms.float32)
    output = SiLU_forward(x)
    expect_output_shape = (1, 2, 4, 4)
    expect_output = np_silu(x_np)
    expect_output_grad = np_silu_grad(x_np)
    assert np.allclose(expect_output_shape, output.shape)

    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        output = SiLU_forward(x)
        out_grad = SiLU_grad(x)
    elif mode == 'KBK':
        output = (jit(SiLU_forward, jit_level="O0"))(x)
        out_grad = (jit(SiLU_grad, jit_level="O0"))(x)
    else:
        output = (jit(SiLU_forward, backend="GE"))(x)
        out_grad = (jit(SiLU_grad, backend="GE"))(x)

    np.testing.assert_allclose(output.asnumpy(), expect_output, rtol=1e-3)
    np.testing.assert_allclose(out_grad.asnumpy(), expect_output_grad, rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential', mem_peak=4.0)
def test_mint_silu_dyn():
    """
    Feature: Dynamic shape of SiLU.
    Description: test SiLU with dynamic rank/shape.
    Expectation: success.
    """
    in1 = Tensor(np.random.randn(1, 2, 4, 4).astype(np.float32))
    in2 = Tensor(np.random.randn(2, 1, 4).astype(np.float32))
    TEST_OP(SiLU_forward, [[in1], [in2]])
