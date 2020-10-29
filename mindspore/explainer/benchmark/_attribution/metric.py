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
import numpy as np

from mindspore.train._utils import check_value_type
from ..._operators import Tensor
from ..._utils import format_tensor_to_ndarray
from ...explanation._attribution._attribution import Attribution


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

    def __init__(self, num_labels=None):
        self._verify_params(num_labels)
        self._num_labels = num_labels
        self._global_results = {i: [] for i in range(num_labels)}

    @staticmethod
    def _verify_params(num_labels):
        check_value_type("num_labels", num_labels, int)
        if num_labels < 1:
            raise ValueError("Argument num_labels must be parsed with a integer > 0.")

    def evaluate(self, explainer, inputs, targets, saliency=None):
        """This function evaluates on a single sample and return the result."""
        raise NotImplementedError

    def aggregate(self, result, targets):
        """Aggregates single result to global_results."""
        if isinstance(result, float):
            if isinstance(targets, int):
                self._global_results[targets].append(result)
            else:
                target_np = format_tensor_to_ndarray(targets)
                if len(target_np) > 1:
                    raise ValueError("One result can not be aggreated to multiple targets.")
        else:
            result_np = format_tensor_to_ndarray(result)
            if isinstance(targets, int):
                for res in result_np:
                    self._global_results[targets].append(float(res))
            else:
                target_np = format_tensor_to_ndarray(targets)
                if len(target_np) != len(result_np):
                    raise ValueError("Length of result does not match with length of targets.")
                for tar, res in zip(target_np, result_np):
                    self._global_results[int(tar)].append(float(res))

    def reset(self):
        """Resets global_result."""
        self._global_results = {i: [] for i in range(self._num_labels)}

    @property
    def class_performances(self):
        """
        Get the class performances by global result.


        Returns:
            (:class:`np.ndarray`): :attr:`num_labels`-dimensional vector
                containing per-class performance.
        """
        count = np.array(
            [len(self._global_results[i]) for i in range(self._num_labels)])
        result_sum = np.array(
            [sum(self._global_results[i]) for i in range(self._num_labels)])
        return result_sum / count.clip(min=1)

    @property
    def performance(self):
        """
        Get the performance by global result.

        Returns:
            (:class:`float`): mean performance.
        """
        count = sum(
            [len(self._global_results[i]) for i in range(self._num_labels)])
        result_sum = sum(
            [sum(self._global_results[i]) for i in range(self._num_labels)])
        if count == 0:
            return 0
        return result_sum / count

    def get_results(self):
        """Global result of the metric can be return"""
        return self._global_results

    def _check_evaluate_param(self, explainer, inputs, targets, saliency):
        """Check the evaluate parameters."""
        check_value_type('explainer', explainer, Attribution)
        verify_argument(inputs, 'inputs')
        output = explainer.model(inputs)
        check_value_type("output of explainer model", output, Tensor)
        output_dim = explainer.model(inputs).shape[1]
        if output_dim > self._num_labels:
            raise ValueError("The output dimension of of black-box model in explainer should not exceed the dimension "
                             "of num_labels set in the __init__, please set num_labels larger.")
        verify_targets(targets, self._num_labels)
        check_value_type('saliency', saliency, (Tensor, type(None)))
