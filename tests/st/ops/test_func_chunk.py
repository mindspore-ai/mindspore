# Copyright 2022 Huawei Technologies Co., Ltd
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
import mindspore.ops as ops
from mindspore import Tensor
import tests.st.utils.test_utils as test_utils
from tests.mark_utils import arg_mark

@test_utils.run_with_cell
def forward_func(x, chunks, axis):
    return ops.chunk(x, chunks, axis)


@test_utils.run_with_cell
def backward_func(x, chunks, axis):
    return ms.grad(forward_func, (0))(x, chunks, axis)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2',
          card_mark='onecard')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_chunk_normal(mode):
    """
    Feature: ops.chunk
    Description: Verify the result of chunk
    Expectation: success
    """
    ms.set_context(mode=mode)
    net = forward_func
    x = Tensor([[[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]]])
    chunks = 6
    axis = 1
    out = net(x, chunks, axis)
    expect_out_1 = np.array([[[[0, 1, 2, 3],
                               [4, 5, 6, 7],
                               [8, 9, 10, 11]]]])
    expect_out_2 = np.array([[[[0, 1, 2, 3],
                               [4, 5, 6, 7],
                               [8, 9, 10, 11]]]])
    assert np.allclose(out[0].asnumpy(), expect_out_1)
    assert np.allclose(out[1].asnumpy(), expect_out_2)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2',
          card_mark='onecard')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize('dtype', [ms.bfloat16])
def test_chunk_bfloat16(mode, dtype):
    """
    Feature: ops.chunk
    Description: Verify the result of chunk
    Expectation: success
    """
    ms.set_context(mode=mode)
    net = forward_func
    x_np = np.random.randn(1, 2, 3, 4).astype(np.float32)
    x = Tensor(x_np, dtype)
    chunks = 6
    axis = 1
    out = net(x, chunks, axis)
    expect_out_1, expect_out_2 = np.split(x_np, 2, axis=1)
    x_grad = backward_func(x, chunks, axis)
    if dtype == ms.float32:
        assert np.allclose(out[0].asnumpy(), expect_out_1)
        assert np.allclose(out[1].asnumpy(), expect_out_2)
        assert x_grad.asnumpy().shape == x.shape
    else:
        assert np.allclose(out[0].float().asnumpy(), expect_out_1, 4e-3)
        assert np.allclose(out[1].float().asnumpy(), expect_out_2, 4e-3)
        assert x_grad.float().asnumpy().shape == x.shape
