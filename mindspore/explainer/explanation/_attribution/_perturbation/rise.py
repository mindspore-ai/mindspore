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
"""RISE."""
import math

import numpy as np

from mindspore import Tensor
from mindspore.train._utils import check_value_type

from .perturbation import PerturbationAttribution
from .... import _operators as op
from ...._utils import resize


class RISE(PerturbationAttribution):
    r"""
    RISE: Randomized Input Sampling for Explanation of Black-box Model.

    RISE is a perturbation-based method that generates attribution maps by sampling on multiple random binary masks.
    The original image is randomly masked, and then fed into the black-box model to get predictions. The final
    attribution map is the weighted sum of these random masks, with the weights being the corresponding output on the
    node of interest:

    .. math::
        attribution = \sum_{i}f_c(I\odot M_i)  M_i

    For more details, please refer to the original paper via: `RISE <https://arxiv.org/abs/1806.07421>`_.

    Args:
        network (Cell): The black-box model to be explained.
        activation_fn (Cell): The activation layer that transforms logits to prediction probabilities. For
            single label classification tasks, `nn.Softmax` is usually applied. As for multi-label classification tasks,
            `nn.Sigmoid` is usually be applied. Users can also pass their own customized `activation_fn` as long as
            when combining this function with network, the final output is the probability of the input.
        perturbation_per_eval (int, optional): Number of perturbations for each inference during inferring the
            perturbed samples. Within the memory capacity, usually the larger this number is, the faster the
            explanation is obtained. Default: 32.

    Inputs:
        - **inputs** (Tensor) - The input data to be explained, a 4D tensor of shape :math:`(N, C, H, W)`.
        - **targets** (Tensor, int) - The labels of interest to be explained. When `targets` is an integer,
          all of the inputs will generates attribution map w.r.t this integer. When `targets` is a tensor, it
          should be of shape :math:`(N, l)` (l being the number of labels for each sample) or :math:`(N,)` :math:`()`.

    Outputs:
        Tensor, a 4D tensor of shape :math:`(N, ?, H, W)`.

    Examples:
        >>> import numpy as np
        >>> import mindspore as ms
        >>> from mindspore.explainer.explanation import RISE
        >>>
        >>> # The detail of LeNet5 is shown in model_zoo.official.cv.lenet.src.lenet.py
        >>> net = LeNet5(10, num_channel=3)
        >>> # initialize RISE explainer with the pretrained model and activation function
        >>> activation_fn = ms.nn.Softmax() # softmax layer is applied to transform logits to probabilities
        >>> rise = RISE(net, activation_fn=activation_fn)
        >>> # given an instance of RISE, saliency map can be generate
        >>> inputs = ms.Tensor(np.random.rand(2, 3, 32, 32), ms.float32)
        >>> # when `targets` is an integer
        >>> targets = 5
        >>> saliency = rise(inputs, targets)
        >>> print(saliency.shape)
        (2, 1, 32, 32)
        >>> # `targets` can also be a 2D tensor
        >>> targets = ms.Tensor([[5], [1]], ms.int32)
        >>> saliency = rise(inputs, targets)
        >>> print(saliency.shape)
        (2, 1, 32, 32)
"""

    def __init__(self,
                 network,
                 activation_fn,
                 perturbation_per_eval=32):
        super(RISE, self).__init__(network, activation_fn, perturbation_per_eval)

        self._num_masks = 6000  # number of masks to be sampled
        self._mask_probability = 0.5  # ratio of inputs to be masked
        self._down_sample_size = 10  # the original size of binary masks
        self._resize_mode = 'bilinear'  # mode choice to resize the down-sized binary masks to size of the inputs
        self._perturbation_mode = 'constant'  # setting the perturbed pixels to a constant value
        self._base_value = 0  # setting the perturbed pixels to this constant value
        self._num_classes = None  # placeholder of self._num_classes just for future assignment in other methods

    def _generate_masks(self, data, batch_size):
        """Generate a batch of binary masks for data."""

        height, width = data.shape[2], data.shape[3]

        mask_size = (self._down_sample_size, self._down_sample_size)

        up_size = (height + mask_size[0], width + mask_size[1])
        mask = np.random.random((batch_size, 1) + mask_size) < self._mask_probability
        upsample = resize(op.Tensor(mask, data.dtype), up_size,
                          self._resize_mode).asnumpy()
        shift_x = np.random.randint(0, mask_size[0] + 1, size=batch_size)
        shift_y = np.random.randint(0, mask_size[1] + 1, size=batch_size)

        masks = [sample[:, x_i: x_i + height, y_i: y_i + width] for sample, x_i, y_i
                 in zip(upsample, shift_x, shift_y)]
        masks = Tensor(np.array(masks), data.dtype)
        return masks

    def __call__(self, inputs, targets):
        """Generates attribution maps for inputs."""
        self._verify_data(inputs, targets)
        height, width = inputs.shape[2], inputs.shape[3]

        batch_size = inputs.shape[0]

        if self._num_classes is None:
            logits = self.network(inputs)
            num_classes = logits.shape[1]
            self._num_classes = num_classes

        # Due to the unsupported Op of slice assignment, we use numpy array here
        targets = self._unify_targets(inputs, targets)

        attr_np = np.zeros(shape=(batch_size, targets.shape[1], height, width))

        cal_times = math.ceil(self._num_masks / self._perturbation_per_eval)

        for idx, data in enumerate(inputs):
            bg_data = data * 0 + self._base_value
            for j in range(cal_times):
                bs = min(self._num_masks - j * self._perturbation_per_eval,
                         self._perturbation_per_eval)
                data = op.reshape(data, (1, -1, height, width))
                masks = self._generate_masks(data, bs)

                masked_input = masks * data + (1 - masks) * bg_data
                weights = self._activation_fn(self.network(masked_input))
                while len(weights.shape) > 2:
                    weights = op.mean(weights, axis=2)

                weights = np.expand_dims(np.expand_dims(weights.asnumpy()[:, targets[idx]], 2), 3)

                attr_np[idx] += np.sum(weights * masks.asnumpy(), axis=0)

        attr_np = attr_np / self._num_masks

        return op.Tensor(attr_np, dtype=inputs.dtype)

    @staticmethod
    def _verify_data(inputs, targets):
        """Verify the validity of the parsed inputs."""
        check_value_type('inputs', inputs, Tensor)
        if len(inputs.shape) != 4:
            raise ValueError(f'Argument inputs must be 4D Tensor, but got {len(inputs.shape)}D Tensor.')
        check_value_type('targets', targets, (Tensor, int, tuple, list))
        if isinstance(targets, Tensor):
            if len(targets.shape) > 2:
                raise ValueError('Dimension invalid. If `targets` is a Tensor, it should be 0D, 1D or 2D. '
                                 'But got {}D.'.format(len(targets.shape)))
            if targets.shape and len(targets) != len(inputs):
                raise ValueError(
                    'If `targets` is a 2D, 1D Tensor, it should have the same length as inputs {}. But got {}.'.format(
                        len(inputs), len(targets)))

    @staticmethod
    def _unify_targets(inputs, targets):
        """To unify targets to be 2D numpy.ndarray."""
        if isinstance(targets, int):
            return np.array([[targets] for _ in inputs]).astype(np.int)
        if isinstance(targets, Tensor):
            if not targets.shape:
                return np.array([[targets.asnumpy()] for _ in inputs]).astype(np.int)
            if len(targets.shape) == 1:
                return np.array([[t.asnumpy()] for t in targets]).astype(np.int)
            if len(targets.shape) == 2:
                return np.array([t.asnumpy() for t in targets]).astype(np.int)
        return targets
