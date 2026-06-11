# Copyright 2026 Huawei Technologies Co., Ltd
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
"""Tests for mint softmax."""

import numpy as np
import pytest
import torch

import mindspore as ms
from mindspore import Tensor, mint
from tests.mark_utils import arg_mark
from tests.st.utils import test_utils


@test_utils.run_with_cell
def softmax_forward_func(x):
    return mint.nn.Softmax(dim=None)(x)


def _set_mode(context_mode):
    ms.context.set_context(mode=context_mode)
    if context_mode == ms.GRAPH_MODE:
        ms.set_context(jit_level="O0")


@arg_mark(plat_marks=["platform_ascend", "platform_ascend910b"],
          level_mark="level0",
          card_mark="onecard", mem_peak=4.01)
@pytest.mark.parametrize("context_mode", [ms.PYNATIVE_MODE, ms.GRAPH_MODE])
def test_mint_softmax_dim_none_scalar(context_mode):
    """
    Feature: mint.nn.Softmax
    Description: Verify scalar input follows torch when dim is None.
    Expectation: success
    """
    _set_mode(context_mode)
    x = Tensor(1.0, ms.float16)
    out = softmax_forward_func(x)
    assert out.shape == ()
    np.testing.assert_allclose(out.asnumpy(), np.array(1.0, dtype=np.float16), atol=0.0, rtol=0.0)


@test_utils.run_with_cell
def softmax_invalid_dim_forward_func(x):
    return mint.nn.Softmax(dim=1)(x)


@arg_mark(plat_marks=["platform_ascend", "platform_ascend910b"],
          level_mark="level0",
          card_mark="onecard", mem_peak=4.01)
@pytest.mark.parametrize("context_mode", [ms.PYNATIVE_MODE, ms.GRAPH_MODE])
def test_mint_softmax_scalar_invalid_dim(context_mode):
    """
    Feature: mint.nn.Softmax
    Description: Verify scalar input still raises for invalid dim.
    Expectation: raise IndexError
    """
    _set_mode(context_mode)
    x = Tensor(1.0, ms.float32)
    with pytest.raises(IndexError, match="Dimension out of range"):
        softmax_invalid_dim_forward_func(x)


@arg_mark(plat_marks=["platform_ascend", "platform_ascend910b"],
          level_mark="level0",
          card_mark="onecard", mem_peak=4.01)
@pytest.mark.parametrize("context_mode", [ms.PYNATIVE_MODE, ms.GRAPH_MODE])
def test_mint_softmax_dim_none_5d(context_mode):
    """
    Feature: mint.nn.Softmax
    Description: Verify implicit dim selection matches torch for 5D input.
    Expectation: success
    """
    _set_mode(context_mode)
    x_np = np.arange(32, dtype=np.float32).reshape(2, 2, 2, 2, 2)
    out = softmax_forward_func(Tensor(x_np))
    expect = torch.nn.Softmax(dim=None)(torch.tensor(x_np)).detach().numpy()
    np.testing.assert_allclose(out.asnumpy(), expect, atol=1e-4, rtol=1e-4)
