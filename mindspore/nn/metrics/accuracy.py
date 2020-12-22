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
"""Accuracy."""
import numpy as np
from .metric import EvaluationBase


class Accuracy(EvaluationBase):
    r"""
    Calculates the accuracy for classification and multilabel data.

    The accuracy class creates two local variables, the correct number and the total number that are used to compute the
    frequency with which predictions matches labels. This frequency is ultimately returned as the accuracy: an
    idempotent operation that simply divides the correct number by the total number.

    .. math::
        \text{accuracy} =\frac{\text{true_positive} + \text{true_negative}}
        {\text{true_positive} + \text{true_negative} + \text{false_positive} + \text{false_negative}}

    Args:
        eval_type (str): Metric to calculate the accuracy over a dataset, for
            classification (single-label), and multilabel (multilabel classification).
            Default: 'classification'.

    Examples:
        >>> x = Tensor(np.array([[0.2, 0.5], [0.3, 0.1], [0.9, 0.6]]), mindspore.float32)
        >>> y = Tensor(np.array([1, 0, 1]), mindspore.float32)
        >>> metric = nn.Accuracy('classification')
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

    def update(self, *inputs):
        """
        Updates the internal evaluation result :math:`y_{pred}` and :math:`y`.

        Args:
            inputs: Input `y_pred` and `y`. `y_pred` and `y` are a `Tensor`, a list or an array.
                For the 'classification' evaluation type, `y_pred` is in most cases (not strictly) a list
                of floating numbers in range :math:`[0, 1]`
                and the shape is :math:`(N, C)`, where :math:`N` is the number of cases and :math:`C`
                is the number of categories. Shape of `y` can be :math:`(N, C)` with values 0 and 1 if one-hot
                encoding is used or the shape is :math:`(N,)` with integer values if index of category is used.
                For 'multilabel' evaluation type, `y_pred` and `y` can only be one-hot encoding with
                values 0 or 1. Indices with 1 indicate the positive category. The shape of `y_pred` and `y`
                are both :math:`(N, C)`.

        Raises:
            ValueError: If the number of the inputs is not 2.
        """
        if len(inputs) != 2:
            raise ValueError('Accuracy need 2 inputs (y_pred, y), but got {}'.format(len(inputs)))
        y_pred = self._convert_data(inputs[0])
        y = self._convert_data(inputs[1])
        if self._type == 'classification' and y_pred.ndim == y.ndim and self._check_onehot_data(y):
            y = y.argmax(axis=1)
        self._check_shape(y_pred, y)
        self._check_value(y_pred, y)

        if self._class_num == 0:
            self._class_num = y_pred.shape[1]
        elif y_pred.shape[1] != self._class_num:
            raise ValueError('Class number not match, last input data contain {} classes, but current data contain {} '
                             'classes'.format(self._class_num, y_pred.shape[1]))

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
            Float, the computed result.

        Raises:
            RuntimeError: If the sample size is 0.
        """
        if self._total_num == 0:
            raise RuntimeError('Accuary can not be calculated, because the number of samples is 0.')
        return self._correct_num / self._total_num
