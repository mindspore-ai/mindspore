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

import os
import numpy as np
import pytest

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.ops.operations import _grad_ops as G


os.environ['CUDA_VISIBLE_DEVICES'] = '2, 3'


class MultiMraginLossWeight(nn.Cell):
    def __init__(self, p, margin, reduction):
        super(MultiMraginLossWeight, self).__init__()
        self.loss = P.MultiMarginLoss(p=p, margin=margin, reduction=reduction)

    def construct(self, x, target, weight):
        return self.loss(x, target, weight)


class MultiMraginLoss(nn.Cell):
    def __init__(self, p, margin, reduction):
        super(MultiMraginLoss, self).__init__()
        self.loss = P.MultiMarginLoss(p=p, margin=margin, reduction=reduction)

    def construct(self, x, target):
        return self.loss(x, target)


class MultiMraginLossGradWeight(nn.Cell):
    def __init__(self, p, margin, reduction):
        super(MultiMraginLossGradWeight, self).__init__()
        self.grad = G.MultiMarginLossGrad(p=p, margin=margin, reduction=reduction)

    def construct(self, y_grad, x, target, weight):
        gout = self.grad(y_grad, x, target, weight)
        return gout


class MultiMraginLossGrad(nn.Cell):
    def __init__(self, p, margin, reduction):
        super(MultiMraginLossGrad, self).__init__()
        self.grad = G.MultiMarginLossGrad(p=p, margin=margin, reduction=reduction)

    def construct(self, y_grad, x, target):
        gout = self.grad(y_grad, x, target)
        return gout


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_multi_margin_loss_fp64():
    """
    Feature: FractionalMaxPool
    Description: Test of input
    Expectation: The results are as expected
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    type_i = np.float64
    ertol_loss = 1e-05
    x = Tensor(np.array([[12.0, 2.0, 13.0], [22.0, 2.0, 1.0], [3.0, 2.0, 1.0]]).astype(type_i))
    target = Tensor(np.array([1, 2, 1]).astype(np.int64))
    weight = Tensor(np.array([1.0, 2.0, 3.0]).astype(type_i))
    net = MultiMraginLossWeight(1, 1.0, 'none')
    output = net(x, target, weight)
    output = output.asnumpy()
    expect_output = np.array([15.33333333, 24.0, 1.33333333]).astype(type_i)
    assert np.allclose(output, expect_output, ertol_loss)

    net = MultiMraginLoss(2, 1.0, 'mean')
    output = net(x, target)
    output = output.asnumpy()
    expect_output = np.array([84.1111111111111]).astype(type_i)
    assert np.allclose(output, expect_output, ertol_loss)

    net = MultiMraginLoss(2, 2.0, 'sum')
    output = net(x, target)
    output = output.asnumpy()
    expect_output = np.array([287.0]).astype(type_i)
    assert np.allclose(output, expect_output, ertol_loss)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_multi_margin_loss_grad_fp64():
    """
    Feature: FractionalMaxPool
    Description: Test of input
    Expectation: The results are as expected
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    type_i = np.float64
    ertol_loss = 1e-05
    y_grad = Tensor(np.array([1.0, 2.0, 3.0]).astype(type_i))
    x = Tensor(np.array([[12.0, 2.0, 13.0], [22.0, 2.0, 1.0], [3.0, 2.0, 1.0]]).astype(type_i))
    target = Tensor(np.array([1, 2, 1]).astype(np.int64))
    weight = Tensor(np.array([1.0, 2.0, 3.0]).astype(type_i))
    net = MultiMraginLossGradWeight(2, 1.0, 'none')
    output = net(y_grad, x, target, weight)
    output = output.asnumpy()
    expect_output = np.array([[14.66666667, -30.66666667, 16.0],
                              [88.0, 8.0, -96.0],
                              [8.0, -8.0, 0.0]]).astype(type_i)
    assert np.allclose(output, expect_output, ertol_loss)

    y_grad = Tensor(np.array([2.0]).astype(type_i))
    net = MultiMraginLossGrad(1, 1.0, 'mean')
    output = net(y_grad, x, target)
    output = output.asnumpy()
    expect_output = np.array([[0.22222222, -0.44444444, 0.22222222],
                              [0.22222222, 0.22222222, -0.44444444],
                              [0.22222222, -0.22222222, 0.0]]).astype(type_i)
    assert np.allclose(output, expect_output, ertol_loss)

    net = MultiMraginLossGrad(2, 2.0, 'sum')
    output = net(y_grad, x, target)
    output = output.asnumpy()
    expect_output = np.array([[16.0, -33.33333333, 17.33333333],
                              [30.66666667, 4.0, -34.66666667],
                              [4.0, -5.33333333, 1.33333333]]).astype(type_i)
    assert np.allclose(output, expect_output, ertol_loss)
