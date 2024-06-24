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
from tests.mark_utils import arg_mark

"""test NLLLoss forward and backward"""

import pytest
import numpy as np

from mindspore import context
from mindspore import Tensor
from mindspore import nn
from mindspore.ops import operations as P
from mindspore.ops import composite as C
from mindspore import export, load

context.set_context(mode=context.GRAPH_MODE, device_target="CPU", save_graphs=True)


class NLLLoss(nn.Cell):
    def __init__(self, reduction="none"):
        super(NLLLoss, self).__init__()
        self.nllloss = P.NLLLoss(reduction=reduction)

    def construct(self, x, t, w):
        return self.nllloss(x, t, w)


class NLLLossGrad(nn.Cell):
    def __init__(self, forward, sens):
        super(NLLLossGrad, self).__init__()
        self.forward = forward
        self.grad = C.GradOperation(get_all=True, sens_param=True)
        self.sens = sens

    def construct(self, x, t, w):
        return self.grad(self.forward)(x, t, w, self.sens)


np_type = np.float32
logits = Tensor(np.array([[-1.3739, -2.2700, -3.2333, -2.4589, -0.6566],
                          [-1.2156, -2.6026, -1.2200, -1.8731, -1.7119],
                          [-0.7130, -3.3672, -1.5368, -1.8289, -2.3058]]).astype(np_type))
target = Tensor(np.array([1, 0, 4]).astype(np.int32))
weight = Tensor(np.array([0.2, 0.3, 0.1, 0.15, 0.25]).astype(np_type))


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_NLLLoss_none():
    """
    Feature: test nlllosss op with reduction none.
    Description: including forward and backward.
    Expectation: expect correct forward and backward result.
    """
    nllloss = NLLLoss(reduction="none")
    actual_output = nllloss(logits, target, weight)
    expect_loss = np.array([0.681, 0.24312, 0.57645]).astype(np_type)
    expect_total_weight = np.array(0.75).astype(np_type)
    assert np.allclose(actual_output[0].asnumpy(), expect_loss)
    assert np.allclose(actual_output[1].asnumpy(), expect_total_weight)

    nllloss_grad = NLLLossGrad(nllloss, sens=(actual_output[0] + 0.5, actual_output[1] + 0.5))
    expect_grad = nllloss_grad(logits, target, weight)
    expect_dx = np.array([[0.0000, -0.35430002, 0.0000, 0.0000, 0.0000],
                          [-0.148624, 0.0000, 0.0000, 0.0000, 0.0000],
                          [0.0000, 0.0000, 0.0000, 0.0000, -0.2691125]]).astype(np_type)
    assert np.allclose(expect_grad[0].asnumpy(), expect_dx)

    export(nllloss_grad, logits, target, weight, file_name="nllloss_none", file_format='MINDIR')
    net = nn.GraphCell(load("nllloss_none.mindir"))
    assert np.allclose(net(logits, target, weight)[0].asnumpy(), expect_dx)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_NLLLoss_sum():
    """
    Feature: test nlllosss op with reduction sum.
    Description: including forward and backward.
    Expectation: expect correct forward and backward result.
    """
    nllloss = NLLLoss(reduction="sum")
    actual_output = nllloss(logits, target, weight)
    expect_loss = np.array(1.50057).astype(np_type)
    expect_total_weight = np.array(0.75).astype(np_type)
    assert np.allclose(actual_output[0].asnumpy(), expect_loss)
    assert np.allclose(actual_output[1].asnumpy(), expect_total_weight)

    nllloss_grad = NLLLossGrad(nllloss, sens=(actual_output[0] + 0.5, actual_output[1] + 0.5))
    expect_grad = nllloss_grad(logits, target, weight)
    expect_dx = np.array([[0.0000, -0.600171, 0.0000, 0.0000, 0.0000],
                          [-0.40011403, 0.0000, 0.0000, 0.0000, 0.0000],
                          [0.0000, 0.0000, 0.0000, 0.0000, -0.5001425]]).astype(np_type)
    assert np.allclose(expect_grad[0].asnumpy(), expect_dx)

    export(nllloss_grad, logits, target, weight, file_name="nllloss_sum", file_format='MINDIR')
    net = nn.GraphCell(load("nllloss_sum.mindir"))
    assert np.allclose(net(logits, target, weight)[0].asnumpy(), expect_dx)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_NLLLoss_mean():
    """
    Feature: test nllloss op with reduction mean.
    Description: including forward and backward.
    Expectation: expect correct forward and backward result.
    """
    nllloss = NLLLoss("mean")
    actual_output = nllloss(logits, target, weight)
    expect_loss = np.array(2.00076).astype(np_type)
    expect_total_weight = np.array(0.75).astype(np_type)
    assert np.allclose(actual_output[0].asnumpy(), expect_loss)
    assert np.allclose(actual_output[1].asnumpy(), expect_total_weight)

    nllloss_grad = NLLLossGrad(nllloss, sens=(actual_output[0] + 0.5, actual_output[1] + 0.5))
    expect_grad = nllloss_grad(logits, target, weight)
    expect_dx = np.array([[0.0000, -1.0003041, 0.0000, 0.0000, 0.0000],
                          [-0.6668694, 0.0000, 0.0000, 0.0000, 0.0000],
                          [0.0000, 0.0000, 0.0000, 0.0000, -0.8335867]]).astype(np_type)
    assert np.allclose(expect_grad[0].asnumpy(), expect_dx)

    export(nllloss_grad, logits, target, weight, file_name="nllloss_mean", file_format='MINDIR')
    net = nn.GraphCell(load("nllloss_mean.mindir"))
    assert np.allclose(net(logits, target, weight)[0].asnumpy(), expect_dx)
