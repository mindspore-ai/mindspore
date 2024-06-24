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

import numpy as np
import pytest

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops.operations import _grad_ops as G
from mindspore.ops.functional import vmap


class NLLLossGradNet(nn.Cell):
    def __init__(self, reduction):
        super(NLLLossGradNet, self).__init__()
        self.grad = G.NLLLossGrad(reduction=reduction)

    def construct(self, x, dout_x, target, weight, total_weight):
        gout = self.grad(x, dout_x, target, weight, total_weight)
        return gout


def get_grad_inputs_and_output(nptype_input, nptype_weight, reduction, input_type="Tensor"):
    """Get inputs and outputs for nll loss grad."""
    x = np.array([[0.53, 0.74, -2.12], [1.29, -0.34, -1.13]]).astype(nptype_input)

    if reduction == "none":
        dloss = np.array([3.24, -2.13]).astype(nptype_input)
    else:
        dloss = np.array(1.23).astype(nptype_input)

    target = np.array([0, 1]).astype(np.int32)
    weight = np.array([0.45, -0.32, 1.21]).astype(nptype_weight)

    total_weight = np.array(0.13).astype(nptype_weight)

    inputs = (x, dloss, target, weight, total_weight)
    if input_type == "Tensor":
        inputs = (Tensor(input_element) for input_element in inputs)

    if reduction == "none":
        dx_expected = np.array([[-1.45799994, 0, 0], [0, -0.681600034, 0]])
    elif reduction == "mean":
        dx_expected = np.array([[-4.25769234, 0, 0], [0, 3.02769232, 0]])
    else:
        dx_expected = np.array([[-0.553499997, 0, 0], [0, 0.393599987, 0]])

    outputs = (dx_expected,)

    return inputs, outputs


def nll_loss_grad_template(nptype_input, nptype_weight, reduction, dynamic=False):
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")

    nll_loss_grad_net = NLLLossGradNet(reduction)

    inputs, expected_outputs = get_grad_inputs_and_output(nptype_input, nptype_weight, reduction)
    x, dloss, target, weight, total_weight = inputs

    if dynamic:
        x_dyn = Tensor(shape=[x.shape[0], None], dtype=x.dtype)
        nll_loss_grad_net.set_inputs(x_dyn, dloss, target, weight, total_weight)

    dx = nll_loss_grad_net(x, dloss, target, weight, total_weight)

    dx_np = dx.asnumpy()

    print(dx)

    dx_expected = expected_outputs[0]

    if nptype_input == np.float32 and nptype_weight == np.float32:
        ertol_loss = 1e-06
    else:
        ertol_loss = 1e-02

    np.testing.assert_allclose(dx_np, dx_expected, ertol_loss)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_nll_loss_grad_vmap():
    """
    Feature: test NLLLossGrad vmap interface.
    Description: test the rightness of NLLLossGrad kernel.
    Expectation: the result match with numpy result
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    reduction = "none"
    def cal_nll_loss_grad(x, dout_x, target, weight, total_weight):
        return G.NLLLossGrad(reduction)(x, dout_x, target, weight, total_weight)

    inputs, expected_outputs = get_grad_inputs_and_output(np.float32, np.float32, "none", "numpy")
    x, dloss, target, weight, total_weight = inputs
    dim_size = 3
    stack_x = np.stack([x] * dim_size)
    stack_dloss = np.stack([dloss] * dim_size)
    stack_target = np.stack([target] * dim_size)

    outputs = vmap(cal_nll_loss_grad, in_axes=(0, 0, 0, None, None), out_axes=0)(
        Tensor(stack_x), Tensor(stack_dloss), Tensor(stack_target), Tensor(weight), Tensor(total_weight))
    expect = np.stack([expected_outputs[0]] * dim_size)
    ertol_loss = 1e-06
    np.testing.assert_allclose(outputs.asnumpy(), expect, ertol_loss)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_nll_loss_grad_no_reduction():
    """
    Feature: test NLLLossGrad kernel.
    Description: test the rightness of NLLLossGrad kernel.
    Expectation: the result match with numpy result
    """
    # Four combinations of fp32 and fp16 inputs and weights
    nll_loss_grad_template(np.float32, np.float32, "mean")


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_nll_loss_grad_no_reduction_dynamic():
    """
    Feature: test NLLLossGrad kernel with dynamic case.
    Description: test the rightness of NLLLossGrad kernel.
    Expectation: the result match with numpy result
    """
    # Four combinations of fp32 and fp16 inputs and weights
    nll_loss_grad_template(np.float32, np.float32, "mean", True)
