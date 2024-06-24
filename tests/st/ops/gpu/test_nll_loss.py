# Copyright 2021 Huawei Technologies Co., Ltd
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
from mindspore.ops import operations as P


class Net(nn.Cell):
    def __init__(self, reduction):
        super(Net, self).__init__()
        self.loss = P.NLLLoss(reduction=reduction)

    def construct(self, predict, target, weight):
        return self.loss(predict, target, weight)


class NLLLossGradNet(nn.Cell):
    def __init__(self, reduction):
        super(NLLLossGradNet, self).__init__()
        self.grad = G.NLLLossGrad(reduction=reduction)

    def construct(self, x, dout_x, target, weight, total_weight):
        gout = self.grad(x, dout_x, target, weight, total_weight)
        return gout


def nll_loss_template(nptype_input, nptype_weight, reduction):
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

    nll_loss_net = Net(reduction)

    predict = Tensor(
        np.array([[0.53, 0.74, -2.12], [1.29, -0.34, -1.13]]).astype(nptype_input))

    target = Tensor(np.array([0, 1]).astype(np.int32))

    weight = Tensor(np.array([0.45, -0.32, 1.21]).astype(nptype_weight))

    loss, total_weight = nll_loss_net(predict, target, weight)

    loss_np = loss.asnumpy()
    total_weight_np = total_weight.asnumpy()

    expected_tot_weight = np.array(0.129999995)

    if reduction == 'none':
        expected_loss = np.array([-0.238499984, -0.108800001])
    elif reduction == 'mean':
        expected_loss = np.array(-2.67153859)
    elif reduction == 'sum':
        expected_loss = np.array(-0.347299993)

    if nptype_input == np.float32 and nptype_weight == np.float32:
        ertol_loss = 1e-06
    elif nptype_input == np.float16 or nptype_weight == np.float16:
        ertol_loss = 1e-03

    if nptype_weight == np.float32:
        ertol_weight = 1e-06
    elif nptype_weight == np.float16:
        ertol_weight = 1e-03

    np.testing.assert_allclose(loss_np, expected_loss, ertol_loss)
    np.testing.assert_allclose(
        total_weight_np, expected_tot_weight, ertol_weight)


def nll_loss_grad_template(nptype_input, nptype_weight, reduction):
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

    nll_loss_grad_net = NLLLossGradNet(reduction)

    x = Tensor(
        np.array([[0.53, 0.74, -2.12], [1.29, -0.34, -1.13]]).astype(nptype_input))

    if reduction == "none":
        dloss = Tensor(
            np.array([3.24, -2.13]).astype(nptype_input))
    else:
        dloss = Tensor(np.array(1.23).astype(nptype_input))

    target = Tensor(np.array([0, 1]).astype(np.int32))
    weight = Tensor(np.array([0.45, -0.32, 1.21]).astype(nptype_weight))

    total_weight = Tensor(np.array(0.13).astype(nptype_weight))

    dx = nll_loss_grad_net(x, dloss, target, weight, total_weight)

    dx_np = dx.asnumpy()

    print(dx)

    if reduction == "none":
        dx_expected = np.array([[-1.45799994, 0, 0], [0, -0.681600034, 0]])
    elif reduction == "mean":
        dx_expected = np.array([[-4.25769234, 0, 0], [0, 3.02769232, 0]])
    else:
        dx_expected = np.array([[-0.553499997, 0, 0], [0, 0.393599987, 0]])

    if nptype_input == np.float32 and nptype_weight == np.float32:
        ertol_loss = 1e-06
    else:
        ertol_loss = 1e-02

    np.testing.assert_allclose(dx_np, dx_expected, ertol_loss)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_nll_loss_no_reduction():
    # Four combinations of fp32 and fp16 inputs and weights
    nll_loss_template(np.float32, np.float32, "none")
    nll_loss_template(np.float32, np.float16, "none")
    nll_loss_template(np.float16, np.float32, "none")
    nll_loss_template(np.float16, np.float16, "none")


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_nll_loss_mean_reduction():
    # Four combinations of fp32 and fp16 inputs and weights
    nll_loss_template(np.float32, np.float32, "mean")
    nll_loss_template(np.float32, np.float16, "mean")
    nll_loss_template(np.float16, np.float32, "mean")
    nll_loss_template(np.float16, np.float16, "mean")


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_nll_loss_sum_reduction():
    # Four combinations of fp32 and fp16 inputs and weights
    nll_loss_template(np.float32, np.float32, "sum")
    nll_loss_template(np.float32, np.float16, "sum")
    nll_loss_template(np.float16, np.float32, "sum")
    nll_loss_template(np.float16, np.float16, "sum")


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_nll_loss_grad_mean_reduction():
    # Four combinations of fp32 and fp16 inputs and weights
    nll_loss_grad_template(np.float32, np.float32, "mean")
    nll_loss_grad_template(np.float32, np.float16, "mean")
    nll_loss_grad_template(np.float16, np.float32, "mean")
    nll_loss_grad_template(np.float16, np.float16, "mean")


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_nll_loss_grad_sum_reduction():
    # Four combinations of fp32 and fp16 inputs and weights
    nll_loss_grad_template(np.float32, np.float32, "sum")
    nll_loss_grad_template(np.float32, np.float16, "sum")
    nll_loss_grad_template(np.float16, np.float32, "sum")
    nll_loss_grad_template(np.float16, np.float16, "sum")


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_nll_loss_grad_no_reduction():
    # Four combinations of fp32 and fp16 inputs and weights
    nll_loss_grad_template(np.float32, np.float32, "none")
    nll_loss_grad_template(np.float32, np.float16, "none")
    nll_loss_grad_template(np.float16, np.float32, "none")
    nll_loss_grad_template(np.float16, np.float16, "none")
