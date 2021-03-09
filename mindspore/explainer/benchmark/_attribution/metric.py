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
"""Base class for XAI metrics."""

import copy
import math
from typing import Callable

import numpy as np

import mindspore as ms
from mindspore import log as logger
from mindspore.train._utils import check_value_type
from ..._operators import Tensor
from ..._utils import format_tensor_to_ndarray
from ...explanation._attribution.attribution import Attribution

_Explainer = Attribution


def verify_argument(inputs, arg_name):
    """Verify the validity of the parsed arguments."""
    check_value_type(arg_name, inputs, Tensor)
    if len(inputs.shape) != 4:
        raise ValueError('Argument {} must be a 4D Tensor.'.format(arg_name))
    if len(inputs) > 1:
        raise ValueError('Support single data evaluation only, but got {}.'.format(len(inputs)))


def verify_targets(targets, num_labels):
    """Verify the validity of the parsed targets."""
    check_value_type('targets', targets, (int, Tensor))

    if isinstance(targets, Tensor):
        if len(targets.shape) > 1 or (len(targets.shape) == 1 and len(targets) != 1):
            raise ValueError('Argument targets must be a 1D or 0D Tensor. If it is a 1D Tensor, '
                             'it should have the length = 1 as we only support single evaluation now.')
        targets = int(targets.asnumpy()[0]) if len(targets.shape) == 1 else int(targets.asnumpy())
    if targets > num_labels - 1 or targets < 0:
        raise ValueError('Parsed targets exceed the label range.')


class AttributionMetric:
    """Super class of XAI metric class used in classification scenarios."""

    def __init__(self):
        self._explainer = None

    evaluate: Callable
    """
    This method evaluates the explainer on the given attribution and returns the evaluation results.
    Derived class should implement this method according to specific algorithms of the metric.
    """

    def _record_explainer(self, explainer: _Explainer):
        """Record the explainer in current evaluation."""
        if self._explainer is None:
            self._explainer = explainer
        elif self._explainer is not explainer:
            logger.info('Provided explainer is not the same as previously evaluated one. Please reset the evaluated '
                        'results. Previous explainer: %s, current explainer: %s', self._explainer, explainer)
            self._explainer = explainer


class LabelAgnosticMetric(AttributionMetric):
    """Super class add functions for label-agnostic metric."""

    def __init__(self):
        super().__init__()
        self._global_results = []

    @property
    def performance(self) -> float:
        """
        Return the average evaluation result.

        Return:
            float, averaged result. If no result is aggregate in the global_results, 0.0 will be returned.
        """
        result_sum, count = 0, 0
        for res in self._global_results:
            if math.isfinite(res):
                result_sum += res
                count += 1
        return 0. if count == 0 else result_sum / count

    def aggregate(self, result):
        """Aggregate single evaluation result to global results."""
        if isinstance(result, float):
            self._global_results.append(result)
        elif isinstance(result, (ms.Tensor, np.ndarray)):
            result = format_tensor_to_ndarray(result)
            self._global_results.extend([float(res) for res in result.reshape(-1)])
        else:
            raise TypeError('result should have type of float, ms.Tensor or np.ndarray, but receive %s' % type(result))

    def get_results(self):
        """Return the global results."""
        return self._global_results.copy()

    def reset(self):
        """Reset global results."""
        self._global_results.clear()

    def _check_evaluate_param(self, explainer, inputs):
        """Check the evaluate parameters."""
        check_value_type('explainer', explainer, Attribution)
        self._record_explainer(explainer)
        verify_argument(inputs, 'inputs')


class LabelSensitiveMetric(AttributionMetric):
    """Super class add functions for label-sensitive metrics."""

    def __init__(self, num_labels: int):
        super().__init__()
        LabelSensitiveMetric._verify_params(num_labels)
        self._num_labels = num_labels
        self._global_results = {i: [] for i in range(num_labels)}

    @property
    def num_labels(self):
        """Number of labels used in evaluation."""
        return self._num_labels

    @staticmethod
    def _verify_params(num_labels):
        """Checks whether num_labels is valid."""
        check_value_type("num_labels", num_labels, int)
        if num_labels < 1:
            raise ValueError("Argument num_labels must be parsed with a integer > 0.")

    def aggregate(self, result, targets):
        """Aggregates single result to global_results."""
        if isinstance(result, float):
            if isinstance(targets, int):
                self._global_results[targets].append(result)
            else:
                target_np = format_tensor_to_ndarray(targets)
                if len(target_np) > 1:
                    raise ValueError("One result can not be aggreated to multiple targets.")
        elif isinstance(result, (ms.Tensor, np.ndarray)):
            result_np = format_tensor_to_ndarray(result).reshape(-1)
            if isinstance(targets, int):
                for res in result_np:
                    self._global_results[targets].append(float(res))
            else:
                target_np = format_tensor_to_ndarray(targets).reshape(-1)
                if len(target_np) != len(result_np):
                    raise ValueError("Length of result does not match with length of targets.")
                for tar, res in zip(target_np, result_np):
                    self._global_results[int(tar)].append(float(res))
        else:
            raise TypeError('Result should have type of float, ms.Tensor or np.ndarray, but receive %s' % type(result))

    def reset(self):
        """Resets global_result."""
        self._global_results = {i: [] for i in range(self._num_labels)}

    @property
    def class_performances(self):
        """
        Get the class performances by global result.

        Returns:
            (:class:`list`): a list of performances where each value is the average score of specific class.
        """
        results_on_labels = []
        for label_id in range(self._num_labels):
            sum_of_label, count_of_label = 0, 0
            for res in self._global_results[label_id]:
                if math.isfinite(res):
                    sum_of_label += res
                    count_of_label += 1
            results_on_labels.append(0. if count_of_label == 0 else sum_of_label / count_of_label)
        return results_on_labels

    @property
    def performance(self):
        """
        Get the performance by global result.

        Returns:
            (:class:`float`): mean performance.
        """
        result_sum, count = 0, 0
        for label_id in range(self._num_labels):
            for res in self._global_results[label_id]:
                if math.isfinite(res):
                    result_sum += res
                    count += 1
        return 0. if count == 0 else result_sum / count

    def get_results(self):
        """Global result of the metric can be return"""
        return copy.deepcopy(self._global_results)

    def _check_evaluate_param(self, explainer, inputs, targets, saliency):
        """Check the evaluate parameters."""
        check_value_type('explainer', explainer, Attribution)
        self._record_explainer(explainer)
        verify_argument(inputs, 'inputs')
        output = explainer.network(inputs)
        check_value_type("output of explainer model", output, Tensor)
        output_dim = explainer.network(inputs).shape[1]
        if output_dim != self._num_labels:
            raise ValueError("The output dimension of of black-box model in explainer does not match the dimension "
                             "of num_labels set in the __init__, please check explainer and num_labels again.")
        verify_targets(targets, self._num_labels)
        check_value_type('saliency', saliency, (Tensor, type(None)))
