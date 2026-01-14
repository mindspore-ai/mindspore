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
def ReLU_forward(x):
    op = mint.nn.ReLU()
    return op(x)


@test_utils.run_with_cell
def ReLU_grad(x):
    op = mint.nn.ReLU()
    return ms.grad(op)(x)


def np_relu(x):
    return np.maximum(0, x)


def np_relu_grad(x):
    return np.where(x > 0, 1, 0)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='essential', mem_peak=2048.05)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_relu_net(mode):
    """
    Feature: ReLU
    Description: Verify the result of mint.nn.ReLU network.
    Expectation: success
    """
    x_np = np.random.randn(1, 2, 4, 4).astype(np.float32)
    x = Tensor(x_np, dtype=ms.float32)
    output = ReLU_forward(x)
    expect_output_shape = (1, 2, 4, 4)
    expect_output = np_relu(x_np)
    expect_output_grad = np_relu_grad(x_np)
    assert np.allclose(expect_output_shape, output.shape)

    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        output = ReLU_forward(x)
        out_grad = ReLU_grad(x)
    elif mode == 'KBK':
        output = (jit(ReLU_forward, jit_level="O0"))(x)
        out_grad = (jit(ReLU_grad, jit_level="O0"))(x)
    else:
        output = (jit(ReLU_forward, backend="GE"))(x)
        out_grad = (jit(ReLU_grad, backend="GE"))(x)

    np.testing.assert_allclose(output.asnumpy(), expect_output, rtol=1e-3)
    np.testing.assert_allclose(out_grad.asnumpy(), expect_output_grad, rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential', mem_peak=3072.00)
def test_mint_relu_dyn():
    """
    Feature: Dynamic shape of relu.
    Description: test relu with dynamic rank/shape.
    Expectation: success.
    """
    in1 = Tensor(np.random.randn(1, 2, 4, 4).astype(np.float32))
    in2 = Tensor(np.random.randn(2, 1, 4).astype(np.float32))
    TEST_OP(ReLU_forward, [[in1], [in2]])
