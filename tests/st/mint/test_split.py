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
"""Tests for mint.split: input, split_size, dim."""
import numpy as np
import pytest

import mindspore as ms
from tests.st.utils import test_utils
from tests.mark_utils import arg_mark


@test_utils.run_with_cell
def split_forward(x, split_size, dim):
    out = ms.mint.split(x, split_size, dim)
    return out


@test_utils.run_with_cell
def split_backward(x, split_size, dim):
    return ms.grad(split_forward, (0,))(x, split_size, dim)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential', mem_peak=2048.01)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_mint_split_normal(mode):
    """
    Feature: mint.split
    Description: Verify the result of mint.split
    Expectation: success
    """
    ms.set_context(mode=mode)
    ms.context.set_context(jit_level='O0')

    split_size_list = [3, (2, 3, 1, 2)]
    indices_list = [(3, 6, 9), (2, 5, 6)]
    dim_list = [1, -1]
    for split_size, indices, dim in zip(split_size_list, indices_list, dim_list):
        x_np = np.random.randn(4, 10, 2, 8).astype(np.float32)
        x = ms.Tensor(x_np)

        expect_outs = np.split(x_np, indices, dim)

        outs = split_forward(x, split_size, dim)
        for out, expect_out in zip(outs, expect_outs):
            assert out.shape == expect_out.shape
            assert np.allclose(out.asnumpy(), expect_out)

        expect_grad = np.ones(x.shape).astype(np.float32)
        grad = split_backward(x, split_size, dim)
        assert grad.shape == expect_grad.shape
        assert np.allclose(grad.asnumpy(), expect_grad)
