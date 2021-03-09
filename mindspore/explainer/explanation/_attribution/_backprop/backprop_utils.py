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
"""Providing utility functions."""

from mindspore.nn import Cell
from mindspore.ops.composite import GradOperation
from mindspore.explainer._utils import unify_inputs, unify_targets, generate_one_hot


def get_bp_weights(model, inputs, targets=None, weights=None):
    r"""
    Compute the gradient of output w.r.t input.

    Args:
        model (Cell): Differentiable black-box model.
        inputs (Tensor): Input to calculate gradient and explanation.
        targets (int, optional): Target label id specifying which category to compute gradient. Default: None.
        weights (Tensor, optional): Custom weights for computing gradients. The shape of weights should match the model
            outputs. If None is provided, an one-hot weights with one in targets positions will be used instead.
            Default: None.

    Returns:
        Tensor, signal to be back-propagated to the input.
    """
    inputs = unify_inputs(inputs)
    if targets is None and weights is None:
        raise ValueError('Must provide one of targets or weights')
    if weights is None:
        targets = unify_targets(targets)
        output = model(*inputs)
        num_categories = output.shape[-1]
        weights = generate_one_hot(targets, num_categories)
    return weights


class GradNet(Cell):
    """
    Network for gradient calculation.

    Args:
        network (Cell): The network to generate backpropagated gradients.
    """

    def __init__(self, network):
        super(GradNet, self).__init__()
        self.network = network
        self.grad = GradOperation(get_all=True, sens_param=True)(network)

    def construct(self, *input_data):
        """
        Get backpropgated gradients.

        Returns:
            Tensor, output gradients.
        """
        gout = self.grad(*input_data)[0]
        return gout
