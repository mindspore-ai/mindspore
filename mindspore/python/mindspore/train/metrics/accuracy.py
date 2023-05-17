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
"""Accuracy."""
from __future__ import absolute_import

import numpy as np

from mindspore.train.metrics.metric import EvaluationBase, rearrange_inputs, _check_onehot_data


class Accuracy(EvaluationBase):
    r"""
    Calculates the accuracy for classification and multilabel data.

    The accuracy class creates two local variables, the correct number and the total number that are used to
    compute the frequency with which y_pred matches y. This frequency is the accuracy.

    .. math::
        \text{accuracy} =\frac{\text{true_positive} + \text{true_negative}}
        {\text{true_positive} + \text{true_negative} + \text{false_positive} + \text{false_negative}}

    Args:
        eval_type (str): The metric to calculate the accuracy over a dataset. Supports 'classification' and
          'multilabel'. 'classification' means the dataset label is single. 'multilabel' means the dataset has multiple
          labels. Default: ``'classification'`` .

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore
        >>> from mindspore import Tensor
        >>> from mindspore.train import Accuracy
        >>>
        >>> x = Tensor(np.array([[0.2, 0.5], [0.3, 0.1], [0.9, 0.6]]), mindspore.float32)
        >>> y = Tensor(np.array([1, 0, 1]), mindspore.float32)
        >>> metric = Accuracy('classification')
        >>> metric.clear()
        >>> metric.update(x, y)
        >>> accuracy = metric.eval()
        >>> print(accuracy)
        0.6666666666666666
    """
    def __init__(self, eval_type='classification'):
        super(Accuracy, self).__init__(eval_type)
        self.clear()

    def clear(self):
        """Clears the internal evaluation result."""
        self._correct_num = 0
        self._total_num = 0
        self._class_num = 0

    @rearrange_inputs
    def update(self, *inputs):
        """
        Updates the local variables. For 'classification', if the index of the maximum of the predict value
        matches the label, the predict result is correct. For 'multilabel', the predict value match the label,
        the predict result is correct.

        Args:
            inputs: Logits and labels. `y_pred` stands for logits, `y` stands for labels. `y_pred` and `y` must be a
                `Tensor`, a list or an array.

                - For the 'classification' evaluation type, `y_pred` is a list of floating numbers in range
                  :math:`[0, 1]` and the shape is :math:`(N, C)` in most cases (not strictly), where :math:`N`
                  is the number of cases and :math:`C` is the number of categories. `y` must be in one-hot format
                  that shape is :math:`(N, C)`, or can be transformed to one-hot format that shape is :math:`(N,)`.
                - For 'multilabel' evaluation type, the value of `y_pred` and `y` can only be 0 or 1, indices with 1
                  indicate the positive category. The shape of `y_pred` and `y` are both :math:`(N, C)`.

        Raises:
            ValueError: If the number of the inputs is not 2.
            ValueError: class numbers of last input predicted data and current predicted data not match.

        """
        if len(inputs) != 2:
            raise ValueError("For 'Accuracy.update', it needs 2 inputs (predicted value, true value), "
                             "but got {}".format(len(inputs)))
        y_pred = self._convert_data(inputs[0])
        y = self._convert_data(inputs[1])
        if self._type == 'classification' and y_pred.ndim == y.ndim and _check_onehot_data(y):
            y = y.argmax(axis=1)
        self._check_shape(y_pred, y)
        self._check_value(y_pred, y)

        if self._class_num == 0:
            self._class_num = y_pred.shape[1]
        elif y_pred.shape[1] != self._class_num:
            raise ValueError("For 'Accuracy.update', class number not match, last input predicted data contain {} "
                             "classes, but current predicted data contain {} classes, please check your predicted "
                             "value(inputs[0]).".format(self._class_num, y_pred.shape[1]))

        if self._type == 'classification':
            indices = y_pred.argmax(axis=1)
            result = (np.equal(indices, y) * 1).reshape(-1)
        elif self._type == 'multilabel':
            dimension_index = y_pred.ndim - 1
            y_pred = y_pred.swapaxes(1, dimension_index).reshape(-1, self._class_num)
            y = y.swapaxes(1, dimension_index).reshape(-1, self._class_num)
            result = np.equal(y_pred, y).all(axis=1) * 1

        self._correct_num += result.sum()
        self._total_num += result.shape[0]

    def eval(self):
        """
        Computes the accuracy.

        Returns:
            np.float64, the computed result.

        Raises:
            RuntimeError: If the sample size is 0.
        """
        if self._total_num == 0:
            raise RuntimeError("The 'Accuracy' can not be calculated, because the number of samples is 0, "
                               "please check whether your inputs(predicted value, true value) are empty, "
                               "or has called update method before calling eval method.")
        return self._correct_num / self._total_num
