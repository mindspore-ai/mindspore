# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
from mindspore.ops.operations import _grad_ops as G


class NetSigmoidCrossEntropyWithLogits(nn.Cell):
    def __init__(self):
        super(NetSigmoidCrossEntropyWithLogits, self).__init__()
        self.sigmoid_cross_entropy_with_logits_grad = G.SigmoidCrossEntropyWithLogitsGrad()

    def construct(self, logits, labels, dout):
        return self.sigmoid_cross_entropy_with_logits_grad(logits, labels, dout)


def sigmoid_cross_entropy_with_logits_grad(nptype):
    logits = Tensor(np.array([[1, 1, 2],
                              [1, 2, 1],
                              [2, 1, 1]]).astype(nptype))
    labels = Tensor(np.array([[0, 0, 1],
                              [0, 1, 0],
                              [1, 0, 0]]).astype(nptype))
    dout = Tensor(np.ones(shape=[3, 3]).astype(nptype))

    expect = np.array([[0.731059, 0.731059, -0.119203],
                       [0.731059, -0.119203, 0.731059],
                       [-0.119203, 0.731059, 0.731059]]).astype(nptype)

    error = np.ones(shape=[3, 3]) * 1.0e-6

    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    net = NetSigmoidCrossEntropyWithLogits()
    output = net(logits, labels, dout)
    diff = output.asnumpy() - expect
    assert np.all(abs(diff) < error)

    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
    net = NetSigmoidCrossEntropyWithLogits()
    output = net(logits, labels, dout)
    diff = output.asnumpy() - expect
    assert np.all(abs(diff) < error)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_sigmoid_cross_entropy_with_logits_float32():
    sigmoid_cross_entropy_with_logits_grad(np.float32)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_sigmoid_cross_entropy_with_logits_float64():
    sigmoid_cross_entropy_with_logits_grad(np.float64)
