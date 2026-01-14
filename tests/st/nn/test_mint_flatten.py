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
def Flatten_forward(x, start_dim=0, end_dim=-1):
    op = mint.nn.Flatten(start_dim, end_dim)
    return op(x)


@test_utils.run_with_cell
def Flatten_grad(x, start_dim=0, end_dim=-1):
    op = mint.nn.Flatten(start_dim, end_dim)
    return ms.grad(op, (0,))(x)

@test_utils.run_with_cell
def Flatten_forward_for_dyn(x):
    op = mint.nn.Flatten(2, 3)
    return op(x)

@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
          card_mark='onecard', essential_mark='essential', mem_peak=2048.0)
@pytest.mark.parametrize('mode', ['pynative', 'KBK', 'GE'])
def test_flatten_normal(mode):
    """
    Feature: Flatten
    Description: Verify the result of mint.nn.Flatten network.
    Expectation: success
    """
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    elif mode == 'KBK':
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level='O0')
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level='O2')
    x_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    x = Tensor(x_np, dtype=ms.float32)
    output = Flatten_forward(x, 1, 2)
    expect_output = x_np.reshape(2, 12, 5)
    np.testing.assert_allclose(output.asnumpy(), expect_output, rtol=1e-4)

    output2 = Flatten_forward(x, 0, 2)
    expect_output2 = x_np.reshape(24, 5)
    np.testing.assert_allclose(output2.asnumpy(), expect_output2, rtol=1e-4)

    output3 = Flatten_grad(x, 1, 3)
    expect_output3 = np.ones((2, 3, 4, 5)).astype(np.float32)
    np.testing.assert_allclose(output3.asnumpy(), expect_output3, rtol=1e-4)

    output4 = Flatten_grad(x, 0, 2)
    expect_output4 = np.ones((2, 3, 4, 5)).astype(np.float32)
    np.testing.assert_allclose(output4.asnumpy(), expect_output4, rtol=1e-4)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential', mem_peak=2048.01)
def test_mint_flatten_dyn():
    """
    Feature: Dynamic shape of Flatten.
    Description: test Flatten with dynamic rank/shape.
    Expectation: success.
    """
    in1 = Tensor(np.random.randn(2, 3, 4, 5, 6).astype(np.float32))
    in2 = Tensor(np.random.randn(3, 4, 5, 6).astype(np.float32))
    TEST_OP(Flatten_forward_for_dyn, [[in1], [in2]], disable_case=['ScalarTensor'])
