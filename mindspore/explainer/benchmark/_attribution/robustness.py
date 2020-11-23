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
"""Robustness."""

from typing import Optional, Union

import numpy as np

import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import log
from .metric import LabelSensitiveMetric
from ...explanation._attribution import Attribution
from ...explanation._attribution._perturbation.replacement import RandomPerturb

_Array = np.ndarray
_Label = Union[ms.Tensor, int]


class Robustness(LabelSensitiveMetric):
    """
    Robustness perturbs the inputs by adding random noise and choose the maximum sensitivity as evaluation score from
    the perturbations.

    Args:
        num_labels (int): Number of classes in the dataset.

    Examples:
    >>> from mindspore.explainer.benchmark import Robustness
    >>> num_labels = 100
    >>> robustness = Robustness(num_labels)
    """

    def __init__(self, num_labels: int, activation_fn=nn.Softmax()):
        super().__init__(num_labels)

        self._perturb = RandomPerturb()
        self._num_perturbations = 100  # number of perturbations used in evaluation
        self._threshold = 0.1  # threshold to generate perturbation
        self._activation_fn = activation_fn

    def evaluate(self,
                 explainer: Attribution,
                 inputs: Tensor,
                 targets: _Label,
                 saliency: Optional[Tensor] = None
                 ) -> _Array:
        """
        Evaluate robustness on single sample.

        Note:
            Currently only single sample (:math:`N=1`) at each call is supported.

        Args:
            explainer (Explanation): The explainer to be evaluated, see `mindspore.explainer.explanation`.
            inputs (Tensor): A data sample, a 4D tensor of shape :math:`(N, C, H, W)`.
            targets (Tensor, int): The label of interest. It should be a 1D or 0D tensor, or an integer.
                If `targets` is a 1D tensor, its length should be the same as `inputs`.
            saliency (Tensor, optional): The saliency map to be evaluated, a 4D tensor of shape :math:`(N, 1, H, W)`.
                If it is None, the parsed `explainer` will generate the saliency map with `inputs` and `targets` and
                continue the evaluation. Default: None.

        Returns:
            numpy.ndarray, 1D array of shape :math:`(N,)`, result of localization evaluated on `explainer`.

        Raises:
            ValueError: If batch_size is larger than 1.

        Examples:
            >>> # init an explainer, the network should contain the output activation function.
            >>> from mindspore.explainer.explanation import Gradient
            >>> from mindspore.explainer.benchmark import Robustness
            >>> gradient = Gradient(network)
            >>> input_x = ms.Tensor(np.random.rand(1, 3, 224, 224), ms.float32)
            >>> target_label = 5
            >>> robustness = Robustness(num_labels=10)
            >>> res = robustness.evaluate(gradient, input_x, target_label)
        """

        self._check_evaluate_param(explainer, inputs, targets, saliency)
        if inputs.shape[0] > 1:
            raise ValueError('Robustness only support a sample each time, but receive {}'.format(inputs.shape[0]))

        inputs_np = inputs.asnumpy()
        if isinstance(targets, int):
            targets = ms.Tensor(targets, ms.int32)
        if saliency is None:
            saliency = explainer(inputs, targets)
        saliency_np = saliency.asnumpy()
        norm = np.sqrt(np.sum(np.square(saliency_np), axis=tuple(range(1, len(saliency_np.shape)))))
        if norm == 0:
            log.warning('Get saliency norm equals 0, robustness return NaN for zero-norm saliency currently.')
            return np.array([np.nan])

        perturbations = []
        for sample in inputs_np:
            sample = np.expand_dims(sample, axis=0)
            perturbations_per_input = []
            for _ in range(self._num_perturbations):
                perturbation = self._perturb(sample)
                perturbations_per_input.append(perturbation)
            perturbations_per_input = np.vstack(perturbations_per_input)
            perturbations.append(perturbations_per_input)
        perturbations = np.stack(perturbations, axis=0)

        perturbations = np.reshape(perturbations, (-1,) + inputs_np.shape[1:])
        perturbations = ms.Tensor(perturbations, ms.float32)

        repeated_targets = np.repeat(targets.asnumpy(), repeats=self._num_perturbations, axis=0)
        repeated_targets = ms.Tensor(repeated_targets, ms.int32)
        saliency_of_perturbations = explainer(perturbations, repeated_targets)
        perturbations_saliency = saliency_of_perturbations.asnumpy()

        repeated_saliency = np.repeat(saliency_np, repeats=self._num_perturbations, axis=0)

        sensitivities = np.sum((repeated_saliency - perturbations_saliency) ** 2,
                               axis=tuple(range(1, len(repeated_saliency.shape))))

        max_sensitivity = np.max(sensitivities.reshape((norm.shape[0], -1)), axis=1) / norm
        robustness_res = 1 / np.exp(max_sensitivity)
        return robustness_res
