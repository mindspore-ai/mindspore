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

"""GradCAM."""

from mindspore.ops import operations as op
from mindspore.explainer._utils import ForwardProbe, retrieve_layer, unify_inputs, unify_targets

from .backprop_utils import get_bp_weights, GradNet
from .intermediate_layer import IntermediateLayerAttribution


def _gradcam_aggregation(attributions):
    """
    Aggregate the gradient and activation to get the final _attribution.

    Args:
        attributions (Tensor): the _attribution with channel dimension.

    Returns:
        Tensor: the _attribution with channel dimension aggregated.
    """
    sum_ = op.ReduceSum(keep_dims=True)
    relu_ = op.ReLU()
    attributions = relu_(sum_(attributions, 1))
    return attributions


class GradCAM(IntermediateLayerAttribution):
    r"""
    Provides GradCAM explanation method.

    `GradCAM` generates saliency map at intermediate layer. The attribution is obtained as:

    .. math::

        \alpha_k^c = \frac{1}{Z} \sum_i \sum_j \frac{\partial{y^c}}{\partial{A_{i,j}^k}}

        attribution = ReLU(\sum_k \alpha_k^c A^k)

    For more details, please refer to the original paper: `GradCAM <https://openaccess.thecvf.com/content_ICCV_2017/
    papers/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.pdf>`_.

    Note:
        The parsed `network` will be set to eval mode through `network.set_grad(False)` and `network.set_train(False)`.
        If you want to train the `network` afterwards, please reset it back to training mode through the opposite
        operations.

    Args:
        network (Cell): The black-box model to be explained.
        layer (str, optional): The layer name to generate the explanation, usually chosen as the last convolutional
            layer for better practice. If it is '', the explanation will be generated at the input layer.
            Default: ''.

    Inputs:
        - **inputs** (Tensor) - The input data to be explained, a 4D tensor of shape :math:`(N, C, H, W)`.
        - **targets** (Tensor, int) - The label of interest. It should be a 1D or 0D tensor, or an integer.
          If it is a 1D tensor, its length should be the same as `inputs`.

    Outputs:
        Tensor, a 4D tensor of shape :math:`(N, 1, H, W)`.

    Examples:
        >>> import numpy as np
        >>> import mindspore as ms
        >>> from mindspore.explainer.explanation import GradCAM
        >>>
        >>> # The detail of LeNet5 is shown in model_zoo.official.cv.lenet.src.lenet.py
        >>> net = LeNet5(10, num_channel=3)
        >>> # specify a layer name to generate explanation, usually the layer can be set as the last conv layer.
        >>> layer_name = 'conv2'
        >>> # init GradCAM with a trained network and specify the layer to obtain attribution
        >>> gradcam = GradCAM(net, layer=layer_name)
        >>> inputs = ms.Tensor(np.random.rand(1, 3, 32, 32), ms.float32)
        >>> label = 5
        >>> saliency = gradcam(inputs, label)
        >>> print(saliency.shape)
        (1, 1, 32, 32)
    """

    def __init__(self, network, layer=""):
        super(GradCAM, self).__init__(network, layer)

        self._saliency_cell = retrieve_layer(self._backward_model, target_layer=layer)
        self._avgpool = op.ReduceMean(keep_dims=True)
        self._intermediate_grad = None
        self._aggregation_fn = _gradcam_aggregation
        self._resize_mode = 'bilinear'

    def _hook_cell(self):
        if self._saliency_cell:
            self._saliency_cell.register_backward_hook(self._cell_hook_fn)
            self._saliency_cell.enable_hook = True
        self._intermediate_grad = None

    def _cell_hook_fn(self, _, grad_input, grad_output):
        """
        Hook function to deal with the backward gradient.

        The arguments are set as required by `Cell.register_backward_hook`.
        """
        self._intermediate_grad = grad_input

    def __call__(self, inputs, targets):
        """Call function for `GradCAM`."""
        self._verify_data(inputs, targets)
        self._hook_cell()

        with ForwardProbe(self._saliency_cell) as probe:

            inputs = unify_inputs(inputs)
            targets = unify_targets(targets)

            weights = get_bp_weights(self._backward_model, *inputs, targets)
            grad_net = GradNet(self._backward_model)
            gradients = grad_net(*inputs, weights)
            # get intermediate activation
            activation = (probe.value,)

            if self._layer == "":
                activation = inputs
                self._intermediate_grad = unify_inputs(gradients)
            if self._intermediate_grad is not None:
                # average pooling on gradients
                intermediate_grad = unify_inputs(
                    self._avgpool(self._intermediate_grad[0], (2, 3)))
            else:
                raise ValueError("Gradient for intermediate layer is not "
                                 "obtained")
            mul = op.Mul()
            attribution = self._aggregation_fn(
                mul(*intermediate_grad, *activation))
            if self._resize:
                attribution = self._resize_fn(attribution, *inputs,
                                              mode=self._resize_mode)
            self._intermediate_grad = None

        return attribution
