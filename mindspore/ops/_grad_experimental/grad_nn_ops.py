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


"""Define the grad rules of neural network related operations."""
from .._grad.grad_base import bprop_getters
from .. import operations as P
from ..composite.multitype_ops.zeros_like_impl import zeros_like
from ..operations import _grad_ops as G

@bprop_getters.register(P.CTCLossV2)
def get_bprop_ctc_loss_v2(self):
    """Grad definition for `CTCLossV2` operation"""
    transpose = P.Transpose()
    ctc_loss_grad = P.CTCLossV2Grad(self.blank, self.reduction, self.zero_infinity)

    def bprop(log_probs, targets, input_lengths, target_lengths, out, dout):
        grad = ctc_loss_grad(dout[1], log_probs, targets, input_lengths, target_lengths, out[0], out[1])
        grad = transpose(grad, (1, 0, 2))
        return grad, zeros_like(targets), zeros_like(input_lengths), zeros_like(target_lengths)

    return bprop


@bprop_getters.register(P.SoftMarginLoss)
def get_bprop_soft_margin_loss(self):
    """Grad definition for `SoftMarginLoss` operation."""
    grad = G.SoftMarginLossGrad(reduction=self.reduction)

    def bprop(predict, label, out, dout):
        dx = grad(predict, label, dout)
        dy = grad(label, predict, dout)
        return dx, dy

    return bprop


@bprop_getters.register(P.SoftShrink)
def get_bprop_softshrink(self):
    """Grad definition for `SoftShrink` operation."""
    input_grad = G.SoftShrinkGrad(self.lambd)

    def bprop(input_x, out, dout):
        dx = input_grad(dout, input_x)
        return (dx,)

    return bprop


@bprop_getters.register(P.HShrink)
def get_bprop_hshrink(self):
    """Grad definition for `HShrinkGrad` operation."""
    grad = G.HShrinkGrad(self.lambd)

    def bprop(features, out, gradients):
        dx = grad(gradients, features)
        return (dx,)

    return bprop
