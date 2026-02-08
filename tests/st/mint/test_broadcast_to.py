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
"""Tests for mint.broadcast_to: input, shape."""
import numpy as np
import pytest

import mindspore as ms
from tests.st.utils import test_utils
from tests.mark_utils import arg_mark


@test_utils.run_with_cell
def broadcast_to_forward(x, shape):
    out = ms.mint.broadcast_to(x, shape)
    return out


@test_utils.run_with_cell
def broadcast_to_backward(x, shape):
    return ms.grad(broadcast_to_forward, (0,))(x, shape)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential', mem_peak=2048.00)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_mint_broadcast_to_normal(mode):
    """
    Feature: mint.broadcast_to
    Description: Verify the result of mint.broadcast_to
    Expectation: success
    """
    ms.set_context(mode=mode)
    ms.context.set_context(jit_level='O0')

    x_np = np.random.rand(2, 1, 4).astype(np.float32)
    x = ms.Tensor(x_np)
    shape = (2, 5, 2, 5, 4)
    expect_out = np.broadcast_to(x_np, shape)

    out = broadcast_to_forward(x, shape)
    assert out.shape == expect_out.shape
    assert np.allclose(out.asnumpy(), expect_out)

    expect_grad = np.broadcast_to(np.array([50]).astype(np.float32), (2, 1, 4))
    grad = broadcast_to_backward(x, shape)
    assert grad.shape == expect_grad.shape
    assert np.allclose(grad.asnumpy(), expect_grad)
