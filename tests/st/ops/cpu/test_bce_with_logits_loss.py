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
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P


class BceWithLogitsLossNet(nn.Cell):
    def __init__(self, reduction):
        super(BceWithLogitsLossNet, self).__init__()
        self.loss = P.BCEWithLogitsLoss(reduction=reduction)

    def construct(self, logits, label, weight, pos_weight):
        return self.loss(logits, label, weight, pos_weight)


def bec_np_bencmark(data_type, reduction):
    """
    Feature: generate a BCEWithLogitsLoss numpy benchmark.
    Description: The benchmark generate by different data type.
    Expectation: match to np mindspore BCEWithLogitsLoss.
    """
    if reduction == "none":
        expected = np.array([[0.6111006, 0.5032824, 0.26318598],
                             [0.58439666, 0.55301523, -0.436814]]).astype(data_type)
    elif reduction == "mean":
        expected = 0.3463612
    else:
        expected = 2.0781672
    return expected


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
@pytest.mark.parametrize("data_type", [np.float32, np.float16])
def test_bce_with_logits_loss(reduction, data_type):
    """
    Feature: test BCEWithLogitsLoss.
    Description: The output generate by different data type and reduction.
    Expectation: match to expected benchmark output.
    """
    context.set_context(mode=context.GRAPH_MODE)
    loss = BceWithLogitsLossNet(reduction)
    logits = Tensor(np.array([[-0.8, 1.2, 0.7], [-0.1, -0.4, 0.7]]).astype(data_type))
    label = Tensor(np.array([[0.3, 0.8, 1.2], [-0.6, 0.1, 2.2]]).astype(data_type))
    weight = Tensor(np.array([1.0, 1.0, 1.0]).astype(data_type))
    pos_weight = Tensor(np.array([1.0, 1.0, 1.0]).astype(data_type))
    error = 1e-6
    if data_type == np.float16:
        error = 1e-3
    benchmark = bec_np_bencmark(data_type, reduction)
    output = loss(logits, label, weight, pos_weight)
    np.testing.assert_allclose(benchmark, output.asnumpy(), rtol=error)
    context.set_context(mode=context.PYNATIVE_MODE)
    output = loss(logits, label, weight, pos_weight)
    np.testing.assert_allclose(benchmark, output.asnumpy(), rtol=error)
