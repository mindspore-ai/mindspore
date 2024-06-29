# Copyright 2019 Huawei Technologies Co., Ltd
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
from tests.mark_utils import arg_mark

import numpy as np
import pytest

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor


class NetSoftmaxCrossEntropyWithLogits(nn.Cell):
    def __init__(self):
        super(NetSoftmaxCrossEntropyWithLogits, self).__init__()
        self.loss = nn.SoftmaxCrossEntropyWithLogits(sparse=False)

    def construct(self, logits, labels):
        return self.loss(logits, labels)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_softmax_cross_entropy_with_logits():
    """
    Feature: template
    Description: template
    Expectation: template
    """
    logits = Tensor(np.array([[1, 1, 10],
                              [1, 10, 1],
                              [10, 1, 1]]).astype(np.float32))
    labels = Tensor(np.array([[0, 0, 1],
                              [0, 1, 0],
                              [1, 0, 0]]).astype(np.float32))
    expect_loss = [0.00024673, 0.00024673, 0.00024673]

    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    softmax_cross_entropy_with_logits = NetSoftmaxCrossEntropyWithLogits()
    output = softmax_cross_entropy_with_logits(logits, labels)
    error0 = 1.0e-6
    diff0 = output.asnumpy() - expect_loss
    assert np.all(abs(diff0) < error0)

    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
    softmax_cross_entropy_with_logits = NetSoftmaxCrossEntropyWithLogits()
    output = softmax_cross_entropy_with_logits(logits, labels)
    error0 = 1.0e-6
    diff0 = output.asnumpy() - expect_loss
    assert np.all(abs(diff0) < error0)
