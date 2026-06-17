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

"""Tests for ops.nn.CTCLoss forward/backward."""

import numpy as np
import pytest

import mindspore as ms
from mindspore import Tensor, ops, nn
from mindspore import dtype as mstype
from tests.mark_utils import arg_mark


class CTCLossNet(nn.Cell):
    def __init__(self, reduction="none"):
        super().__init__()
        self.ctcloss = nn.CTCLoss(blank=0, reduction=reduction, zero_infinity=False)

    def construct(self, log_probs, target, input_length, target_length):
        return self.ctcloss(log_probs, target, input_length, target_length)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level2',
          card_mark='onecard')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize('reduct', ["none", "mean", "sum"])
def test_ctc_loss_tnc(mode, reduct):
    """
    Feature: test CTCLoss op with input shape (T, N, C).
    Description: Verify the result of CTCLoss.
    Expectation: expect correct forward result.
    """
    ms.set_context(mode=mode)
    loss = CTCLossNet(reduction=reduct)
    t = 10
    c = 4
    n = 2
    s = 5
    s_min = 3
    arr = np.arange(t * n * c).reshape((t, n, c))
    inputs = Tensor(arr, dtype=mstype.float32)
    input_lengths = np.full(shape=(n), fill_value=t)
    input_lengths = Tensor(input_lengths, dtype=mstype.int32)
    target_lengths = np.full(shape=(n), fill_value=s_min)
    target_lengths = Tensor(target_lengths, dtype=mstype.int32)
    arr = np.arange(n * s).reshape((n, s))
    targets = Tensor(arr, dtype=mstype.int32)
    if reduct == "none":
        expect_output = np.array([-378.18414, -460.60648])
    elif reduct == "mean":
        expect_output = np.array([-139.79843])
    else:
        expect_output = np.array([-838.79065])
    output = loss(inputs, targets, input_lengths, target_lengths)
    assert np.allclose(output.asnumpy(), expect_output)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_ascend'],
          level_mark='level1',
          card_mark='onecard', mem_peak=1025.0)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ctc_loss_mean_grad(mode):
    """
    Feature: test CTCLoss backward with reduction='mean'.
    Description: Verify backward does not raise dtype mismatch in AddN.
    Expectation: expect correct backward result matching PyTorch.
    """
    import os
    os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    import torch
    ms.set_context(mode=mode)
    np.random.seed(0)
    T, N, C, S = 50, 16, 20, 30

    log_probs_np = np.random.randn(T, N, C).astype(np.float32)
    targets_np = np.random.randint(low=1, high=C, size=(N, S)).astype(np.int32)
    input_lengths_np = np.full(shape=(N,), fill_value=T).astype(np.int32)
    target_lengths_np = np.random.randint(low=10, high=S, size=(N,)).astype(np.int32)

    # PyTorch reference
    log_probs_torch = torch.tensor(log_probs_np, requires_grad=True)
    targets_torch = torch.tensor(targets_np, dtype=torch.int32)
    input_lengths_torch = torch.tensor(input_lengths_np, dtype=torch.int32)
    target_lengths_torch = torch.tensor(target_lengths_np, dtype=torch.int32)
    torch_loss = torch.nn.CTCLoss(blank=0, reduction='mean', zero_infinity=False)(
        log_probs_torch, targets_torch, input_lengths_torch, target_lengths_torch)
    torch_loss.backward()
    expected_grad = log_probs_torch.grad.numpy()

    # MindSpore
    net = CTCLossNet(reduction='mean')
    grad_fn = ops.grad(net)
    ms_grad = grad_fn(
        Tensor(log_probs_np),
        Tensor(targets_np, dtype=mstype.int32),
        Tensor(input_lengths_np, dtype=mstype.int32),
        Tensor(target_lengths_np, dtype=mstype.int32),
    )
    assert np.allclose(ms_grad.asnumpy(), expected_grad, rtol=1e-3, atol=1e-3)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level2',
          card_mark='onecard')
@pytest.mark.skip(reason="has bug, already record by 30032396")
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize('reduct', ["none", "mean", "sum"])
def test_ctc_loss_tc(mode, reduct):
    """
    Feature: test CTCLoss op with input shape (T, C).
    Description: Verify the result of CTCLoss.
    Expectation: expect correct forward result.
    """
    ms.set_context(mode=mode)
    loss = CTCLossNet(reduction=reduct)
    t = 10
    c = 4
    s_min = 3
    arr = np.arange(t * c).reshape((t, c))
    inputs = Tensor(arr, dtype=mstype.float32)
    input_lengths = Tensor([t], dtype=mstype.int32)
    target_lengths = Tensor([s_min], dtype=mstype.int32)
    arr = np.arange(s_min).reshape((s_min,))
    targets = Tensor(arr, dtype=mstype.int32)
    if reduct == "mean":
        expect_output = -66.061386
    else:
        expect_output = -198.18416
    output = loss(inputs, targets, input_lengths, target_lengths)
    assert np.allclose(output.asnumpy(), expect_output)
