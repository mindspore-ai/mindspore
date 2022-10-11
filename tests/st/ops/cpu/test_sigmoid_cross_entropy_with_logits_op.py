# Copyright 2020 Huawei Technologies Co., Ltd
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
import mindspore as ms
from mindspore import Tensor
from mindspore.ops import operations as P


class NetSigmoidCrossEntropyWithLogits(nn.Cell):

    def __init__(self):
        super(NetSigmoidCrossEntropyWithLogits, self).__init__()
        self.loss = P.SigmoidCrossEntropyWithLogits()

    def construct(self, logits, labels):
        return self.loss(logits, labels)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_sigmoid_cross_entropy_with_logits_dynamic_shape():
    """
    Feature: test SigmoidCrossEntropyWithLogits op in cpu.
    Description: test the ops in dynamic shape.
    Expectation: expect correct shape result.
    """
    net = NetSigmoidCrossEntropyWithLogits()

    input_logits_dyn = Tensor(shape=[2, None], dtype=ms.float32)
    input_label_dyn = Tensor(shape=[2, None], dtype=ms.float32)

    net.set_inputs(input_logits_dyn, input_label_dyn)

    input_logits = Tensor(
        np.array([[-0.8, 1.2, 0.7], [-0.1, -0.4, 0.7]]).astype(np.float32))
    input_labels = Tensor(
        np.array([[0.3, 0.8, 1.2], [-0.6, 0.1, 2.2]]).astype(np.float32))

    output = net(input_logits, input_labels)
    expect_shape = (2, 3)
    assert output.asnumpy().shape == expect_shape


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_sigmoid_cross_entropy_with_logits():
    logits = Tensor(
        np.array([[1, 1, 2], [1, 2, 1], [2, 1, 1]]).astype(np.float32))
    labels = Tensor(
        np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]]).astype(np.float32))
    expect_loss = np.array([[1.313262, 1.313262, 0.126928],
                            [1.313262, 0.126928, 1.313262],
                            [0.126928, 1.313262, 1.313262]]).astype(np.float32)

    error = np.ones(shape=[3, 3]) * 1.0e-6

    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    sigmoid_cross_entropy_with_logits = NetSigmoidCrossEntropyWithLogits()
    output = sigmoid_cross_entropy_with_logits(logits, labels)
    diff = output.asnumpy() - expect_loss
    assert np.all(abs(diff) < error)
