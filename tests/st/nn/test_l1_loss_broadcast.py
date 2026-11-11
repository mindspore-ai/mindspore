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
"""L1 loss broadcast value and gradient regression."""
import numpy as np
import pytest

import mindspore as ms
from mindspore import nn, ops, Tensor
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=["cpu_linux"], level_mark="level1", card_mark="onecard")
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize('reduction', ['none', 'mean', 'sum'])
@pytest.mark.parametrize('dtype', [np.float16, np.float32])
def test_l1_loss_broadcast_values_and_gradients(mode, reduction, dtype):
    """
    Feature: L1Loss with broadcast labels.
    Description: Compare loss values and gradients across modes, reductions and floating-point types.
    Expectation: Values and gradients match the NumPy reference, including at zero error.
    """
    previous_mode = ms.get_context('mode')
    try:
        ms.set_context(mode=mode)
        logits = np.array([[-2, 0, 2], [3, 5, -1]], dtype=dtype)
        labels = np.array([[1, -2, 2]], dtype=dtype)
        difference = logits - labels
        expected = np.abs(difference)
        scale = difference.size if reduction == 'mean' else 1
        if reduction == 'mean':
            expected = expected.mean(dtype=np.float32).astype(dtype)
        elif reduction == 'sum':
            expected = expected.sum(dtype=np.float32).astype(dtype)
        loss = nn.L1Loss(reduction=reduction)
        output = loss(Tensor(logits), Tensor(labels))
        grad_logits, grad_labels = ops.grad(loss, (0, 1))(Tensor(logits), Tensor(labels))
        expected_grad = np.sign(difference) / scale
        np.testing.assert_allclose(output.asnumpy(), expected, rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(grad_logits.asnumpy(), expected_grad, rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(grad_labels.asnumpy(), -expected_grad.sum(axis=0, keepdims=True),
                                   rtol=1e-3, atol=1e-3)
        assert output.dtype == Tensor(logits).dtype
    finally:
        ms.set_context(mode=previous_mode)
