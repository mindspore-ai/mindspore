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
def GELU_forward(x):
    op = mint.nn.GELU()
    return op(x)


@test_utils.run_with_cell
def GELU_grad(x):
    op = mint.nn.GELU()
    return ms.grad(op)(x)


def np_gelu(x):
    coeff = 0.044715
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + coeff * np.power(x, 3))))


def np_gelu_grad(x):
    u = np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)
    tanh_u = np.tanh(u)
    dtanhu = 1 - tanh_u ** 2
    du_dx = np.sqrt(2 / np.pi) * (1 + 3 * 0.044715 * x ** 2)
    dy_dx = 0.5 * (1 + tanh_u) + 0.5 * x * dtanhu * du_dx
    return dy_dx


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential', mem_peak=2048.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_gelu_net(mode):
    """
    Feature: GELU
    Description: Verify the result of mint.nn.GELU network.
    Expectation: success
    """
    x_np = np.array([[1, 2], [3, 4]], np.float32)
    x = Tensor(x_np, dtype=ms.float32)
    output = GELU_forward(x)
    expect_output_shape = (2, 2)
    expect_output = np.array([[0.8413, 1.9545],
                              [2.9959, 3.9999]], np.float32)
    expect_output_grad = np.array([[1.0833, 1.0852],
                                   [1.0119, 1.0005]], np.float32)
    assert np.allclose(expect_output_shape, output.shape)

    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        output = GELU_forward(x)
        out_grad = GELU_grad(x)
    elif mode == 'KBK':
        output = (jit(GELU_forward, backend="ms_backend", jit_level="O0"))(x)
        out_grad = (jit(GELU_grad, backend="ms_backend", jit_level="O0"))(x)
    else:
        output = (jit(GELU_forward, backend="GE"))(x)
        out_grad = (jit(GELU_grad, backend="GE"))(x)

    np.testing.assert_allclose(output.asnumpy(), expect_output, rtol=1e-4)
    np.testing.assert_allclose(out_grad.asnumpy(), expect_output_grad, rtol=1e-4)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential', mem_peak=2048.0)
def test_mint_gelu_dyn():
    """
    Feature: Dynamic shape of GELU.
    Description: test GELU with dynamic rank/shape.
    Expectation: success.
    """
    in1 = Tensor(np.random.randn(1, 2, 4, 4).astype(np.float32))
    in2 = Tensor(np.random.randn(2, 1, 4).astype(np.float32))
    TEST_OP(GELU_forward, [[in1], [in2]], disable_mode=["GRAPH_MODE_GE"])
