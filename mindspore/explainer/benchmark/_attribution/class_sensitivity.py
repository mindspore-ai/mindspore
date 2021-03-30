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
"""Class Sensitivity."""

import numpy as np

from mindspore.explainer.explanation import RISE
from .metric import LabelAgnosticMetric
from ... import _operators as ops
from ..._utils import calc_correlation


class ClassSensitivity(LabelAgnosticMetric):
    """
    Class sensitivity metric used to evaluate attribution-based explanations.

    Reasonable atrribution-based explainers are expected to generate distinct saliency maps for different labels,
    especially for labels of highest confidence and low confidence. ClassSensitivity evaluates the explainer through
    computing the correlation between saliency maps of highest-confidence and lowest-confidence labels. Explainer with
    better class sensitivity will receive lower correlation score. To make the evaluation results intuitive, the
    returned score will take negative on correlation and normalize.
    """

    def evaluate(self, explainer, inputs):
        """
        Evaluate class sensitivity on a single data sample.

        Args:
            explainer (Explanation): The explainer to be evaluated, see `mindspore.explainer.explanation`.
            inputs (Tensor): A data sample, a 4D tensor of shape :math:`(N, C, H, W)`.

        Returns:
            numpy.ndarray, 1D array of shape :math:`(N,)`, result of class sensitivity evaluated on `explainer`.

        Examples:
            >>> import numpy as np
            >>> import mindspore as ms
            >>> from mindspore.explainer.benchmark import ClassSensitivity
            >>> from mindspore.explainer.explanation import Gradient
            >>>
            >>> # The detail of LeNet5 is shown in model_zoo.official.cv.lenet.src.lenet.py
            >>> net = LeNet5(10, num_channel=3)
            >>> # prepare your explainer to be evaluated, e.g., Gradient.
            >>> gradient = Gradient(net)
            >>> input_x = ms.Tensor(np.random.rand(1, 3, 32, 32), ms.float32)
            >>> class_sensitivity = ClassSensitivity()
            >>> res = class_sensitivity.evaluate(gradient, input_x)
            >>> print(res.shape)
            (1,)
        """
        self._check_evaluate_param(explainer, inputs)

        outputs = explainer.network(inputs)

        max_confidence_label = ops.argmax(outputs)
        min_confidence_label = ops.argmin(outputs)
        if isinstance(explainer, RISE):
            labels = ops.stack([max_confidence_label, min_confidence_label], axis=1)
            full_saliency = explainer(inputs, labels)
            max_confidence_saliency = full_saliency[:, max_confidence_label].asnumpy()
            min_confidence_saliency = full_saliency[:, min_confidence_label].asnumpy()
        else:
            max_confidence_saliency = explainer(inputs, max_confidence_label).asnumpy()
            min_confidence_saliency = explainer(inputs, min_confidence_label).asnumpy()

        correlations = []
        for i in range(inputs.shape[0]):
            correlation = calc_correlation(max_confidence_saliency[i].reshape(-1),
                                           min_confidence_saliency[i].reshape(-1))
            normalized_correlation = (-correlation + 1) / 2
            correlations.append(normalized_correlation)
        return np.array(correlations, np.float)
