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
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import dtype as mstype


class CTCLossNet(nn.Cell):
    def __init__(self, reduction="none"):
        super(CTCLossNet, self).__init__()
        self.ctcloss = nn.CTCLoss(blank=0, reduction=reduction, zero_infinity=False)

    def construct(self, log_probs, target, input_length, target_length):
        return self.ctcloss(log_probs, target, input_length, target_length)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.skip(reason="has bug, already record by 30032396")
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


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
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
