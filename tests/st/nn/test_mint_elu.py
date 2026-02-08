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
from mindspore import Tensor, mint, ops, jit
from tests.st.utils import test_utils
from tests.mark_utils import arg_mark
from tests.st.ops.test_tools.test_op import TEST_OP


@test_utils.run_with_cell
def ELU_forward(x, alpha):
    op = mint.nn.ELU(alpha)
    return op(x)


@test_utils.run_with_cell
def ELU_grad(x, alpha):
    op = mint.nn.ELU(alpha)
    return ms.grad(op, (0,))(x)


@test_utils.run_with_cell
def ELU_forward_for_dyn(x):
    op = mint.nn.ELU()
    return op(x)


class ELU_Inplace(ms.nn.Cell):
    def __init__(self, alpha):
        super(ELU_Inplace, self).__init__()
        self.op = mint.nn.ELU(alpha, True)

    def construct(self, x):
        y = x + 0
        return self.op(y)


def Inplace_ELU_forward(x, alpha):
    return ELU_Inplace(alpha)(x)


@jit(backend="ms_backend")
def Inplace_ELU_grad(x, alpha):
    grad = ops.GradOperation(get_all=True)
    return grad(ELU_Inplace(alpha))(x)


def Inplace_ELU_forward_for_dyn(x):
    return ELU_Inplace(1.0)(x)

def np_elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))


def np_elu_grad(x, alpha=1.0):
    return np.where(x > 0, 1, alpha * np.exp(x))


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=2.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_elu_net(mode):
    """
    Feature: ELU
    Description: Verify the result of mint.nn.ELU network.
    Expectation: success
    """
    x_np = np.random.randn(1, 2, 4, 4).astype(np.float32)
    x = Tensor(x_np, dtype=ms.float32)
    alpha = np.random.uniform(0.5, 2)

    expect_output = np_elu(x_np, alpha)
    expect_output_grad = np_elu_grad(x_np, alpha)

    output = ELU_forward(x, alpha)
    assert np.allclose(expect_output.shape, output.shape)

    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    elif mode == 'KBK':
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    output = ELU_forward(x, alpha)
    out_grad = ELU_grad(x, alpha)
    np.testing.assert_allclose(output.asnumpy(), expect_output, rtol=1e-4)
    np.testing.assert_allclose(out_grad.asnumpy(), expect_output_grad, rtol=1e-4)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=2.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_elu_net_inplace(mode):
    """
    Feature: ELU
    Description: Verify the result of mint.nn.ELU network.
    Expectation: success
    """
    x_np = np.random.randn(1, 2, 4, 4).astype(np.float32)
    x = Tensor(x_np, dtype=ms.float32)
    alpha = np.random.uniform(0.5, 2)

    expect_output = np_elu(x_np, alpha)
    expect_output_grad = np_elu_grad(x_np, alpha)

    output = Inplace_ELU_forward(x, alpha)
    assert np.allclose(expect_output.shape, output.shape)

    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    elif mode == 'KBK':
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    x = Tensor(x_np, dtype=ms.float32)
    output = Inplace_ELU_forward(x, alpha)
    x = Tensor(x_np, dtype=ms.float32)
    out_grad = Inplace_ELU_grad(x, alpha)
    np.testing.assert_allclose(output, expect_output, rtol=1e-4)
    np.testing.assert_allclose(out_grad[0], expect_output_grad, rtol=1e-4)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential', mem_peak=2048.00)
def test_mint_elu_dyn():
    """
    Feature: Dynamic shape of ELU.
    Description: test ELU with dynamic rank/shape.
    Expectation: success.
    """
    in1 = Tensor(np.random.randn(1, 2, 4, 4).astype(np.float32))
    in2 = Tensor(np.random.randn(2, 1, 4).astype(np.float32))
    TEST_OP(ELU_forward_for_dyn, [[in1], [in2]], disable_mode=['GRAPH_MODE_GE'])
    TEST_OP(Inplace_ELU_forward_for_dyn, [[in1], [in2]], disable_mode=['GRAPH_MODE_GE'], inplace_update=True)
