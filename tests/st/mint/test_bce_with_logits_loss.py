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
"""Non-opinfo coverage for mint.nn.BCEWithLogitsLoss."""

import numpy as np
import pytest

import mindspore as ms
from mindspore import Tensor, mint

from tests.mark_utils import arg_mark
from tests.st.utils import test_utils


@test_utils.run_with_cell
def bce_with_logits_loss_self_target_forward(x):
    return mint.nn.BCEWithLogitsLoss(reduction="none")(x, x)


@test_utils.run_with_cell
def bce_with_logits_loss_self_target_grad(x):
    return ms.grad(bce_with_logits_loss_self_target_forward, (0,))(x)


def _set_mode(mode):
    if mode == "pynative":
        ms.set_context(mode=ms.PYNATIVE_MODE)
    else:
        ms.set_context(mode=ms.GRAPH_MODE, jit_config={"jit_level": "O0"})

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', mem_peak=2.00)
@pytest.mark.parametrize("mode", ["pynative", "KBK"])
@pytest.mark.parametrize(
    "dtype,rtol,atol",
    [
        (ms.bfloat16, 5e-2, 5e-2),
        (ms.float16, 5e-3, 5e-3),
    ],
)
def test_mint_nn_bce_with_logits_loss_self_target_grad_accumulates_like_pytorch(mode, dtype, rtol, atol):
    """
    Feature: mint.nn.BCEWithLogitsLoss
    Description: verify alias inputs accumulate input and target gradients like native PyTorch.
    Expectation: gradients stay finite and match sigmoid(x) - 2 * x for loss(x, x).
    """
    _set_mode(mode)
    x_np = np.array([[8.8, 8.43, -7.2], [6.5, -5.4, 4.25]], dtype=np.float32)
    x = Tensor(x_np, dtype)

    grad = bce_with_logits_loss_self_target_grad(x).astype(ms.float32).asnumpy()
    expected = 1.0 / (1.0 + np.exp(-x_np)) - 2.0 * x_np

    assert np.isfinite(grad).all()
    np.testing.assert_allclose(grad, expected, rtol=rtol, atol=atol)
