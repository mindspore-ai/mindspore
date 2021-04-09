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
"""Robustness."""

import numpy as np

import mindspore as ms
import mindspore.nn as nn
from mindspore.train._utils import check_value_type
from mindspore import log
from .metric import LabelSensitiveMetric
from ...explanation._attribution._perturbation.replacement import RandomPerturb


class Robustness(LabelSensitiveMetric):
    """
    Robustness perturbs the inputs by adding random noise and choose the maximum sensitivity as evaluation score from
    the perturbations.

    Args:
        num_labels (int): Number of classes in the dataset.
        activation_fn (Cell): The activation layer that transforms logits to prediction probabilities. For
            single label classification tasks, `nn.Softmax` is usually applied. As for multi-label classification tasks,
            `nn.Sigmoid` is usually be applied. Users can also pass their own customized `activation_fn` as long as
            when combining this function with network, the final output is the probability of the input.
    """

    def __init__(self, num_labels, activation_fn):
        super().__init__(num_labels)
        check_value_type("activation_fn", activation_fn, nn.Cell)
        self._perturb = RandomPerturb()
        self._num_perturbations = 10  # number of perturbations used in evaluation
        self._threshold = 0.1  # threshold to generate perturbation
        self._activation_fn = activation_fn

    def evaluate(self, explainer, inputs, targets, saliency=None):
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
            >>> import numpy as np
            >>> import mindspore as ms
            >>> from mindspore import nn
            >>> from mindspore.explainer.explanation import Gradient
            >>> from mindspore.explainer.benchmark import Robustness
            >>>
            >>> # Initialize a Robustness benchmarker passing num_labels of the dataset.
            >>> num_labels = 10
            >>> activation_fn = nn.Softmax()
            >>> robustness = Robustness(num_labels, activation_fn)
            >>>
            >>> # The detail of LeNet5 is shown in model_zoo.official.cv.lenet.src.lenet.py
            >>> net = LeNet5(10, num_channel=3)
            >>> # prepare your explainer to be evaluated, e.g., Gradient.
            >>> gradient = Gradient(net)
            >>> input_x = ms.Tensor(np.random.rand(1, 3, 32, 32), ms.float32)
            >>> target_label = ms.Tensor([0], ms.int32)
            >>> # robustness is a Robustness instance
            >>> res = robustness.evaluate(gradient, input_x, target_label)
            >>> print(res.shape)
            (1,)
        """

        self._check_evaluate_param(explainer, inputs, targets, saliency)
        if inputs.shape[0] > 1:
            raise ValueError('Robustness only support a sample each time, but receive {}'.format(inputs.shape[0]))

        inputs_np = inputs.asnumpy()
        if isinstance(targets, int):
            targets = ms.Tensor([targets], ms.int32)
        if saliency is None:
            saliency = explainer(inputs, targets)
        saliency_np = saliency.asnumpy()

        norm = np.sqrt(np.sum(np.square(saliency_np), axis=tuple(range(1, len(saliency_np.shape)))))
        if (norm == 0).any():
            log.warning('Get saliency norm equals 0, robustness return NaN for zero-norm saliency currently.')
            norm[norm == 0] = np.nan

        full_network = nn.SequentialCell([explainer.network, self._activation_fn])
        original_outputs = full_network(inputs).asnumpy()
        sensitivities = []
        for _ in range(self._num_perturbations):
            perturbations = []
            for j, sample in enumerate(inputs_np):
                perturbation_on_single_sample = self._perturb_with_threshold(full_network,
                                                                             np.expand_dims(sample, axis=0),
                                                                             original_outputs[j])
                perturbations.append(perturbation_on_single_sample)
            perturbations = np.vstack(perturbations)
            perturbations_saliency = explainer(ms.Tensor(perturbations, ms.float32), targets).asnumpy()
            sensitivity = np.sqrt(np.sum((perturbations_saliency - saliency_np) ** 2,
                                         axis=tuple(range(1, len(saliency_np.shape)))))
            sensitivities.append(sensitivity)
        sensitivities = np.stack(sensitivities, axis=-1)
        max_sensitivity = np.max(sensitivities, axis=1) / norm
        robustness_res = 1 / np.exp(max_sensitivity)
        return robustness_res

    def _perturb_with_threshold(self, network: nn.Cell, sample: np.ndarray, original_output: np.ndarray) -> np.ndarray:
        """
        Generate the perturbation until the L2-distance between original_output and perturbation_output is lower than
        the given self._threshold or until the attempt reaches the max_attempt_time.
        """
        # the maximum time attempt to get a perturbation with perturb_error low than self._threshold
        max_attempt_time = 3
        perturbation = None
        for _ in range(max_attempt_time):
            perturbation = self._perturb(sample)
            perturbation_output = self._activation_fn(network(ms.Tensor(sample, ms.float32))).asnumpy()
            perturb_error = np.linalg.norm(original_output - perturbation_output)
            if perturb_error <= self._threshold:
                return perturbation
        return perturbation
