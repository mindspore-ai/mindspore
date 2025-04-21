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
"""Fbeta."""
from __future__ import absolute_import

import sys
import numpy as np

from mindspore import _checkparam as validator
from mindspore.train.metrics.metric import Metric, rearrange_inputs, _check_onehot_data


class Fbeta(Metric):
    r"""
    Calculates the Fbeta score.

    Fbeta score is a weighted mean of precision and recall.

    .. math::
        F_\beta=\frac{(1+\beta^2) \cdot true\_positive}
                {(1+\beta^2) \cdot true\_positive +\beta^2 \cdot false\_negative + false\_positive}

    Args:
        beta (Union[float, int]): Beta coefficient in the F measure. `beta` should be greater than 0.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> from mindspore.train import Fbeta
        >>>
        >>> x = Tensor(np.array([[0.2, 0.5], [0.3, 0.1], [0.9, 0.6]]))
        >>> y = Tensor(np.array([1, 0, 1]))
        >>> metric = Fbeta(1)
        >>> metric.clear()
        >>> metric.update(x, y)
        >>> fbeta = metric.eval()
        >>> print(fbeta)
        [0.66666667 0.66666667]
    """
    def __init__(self, beta):
        super(Fbeta, self).__init__()
        self.eps = sys.float_info.min
        if not beta > 0:
            raise ValueError("For 'Fbeta', the argument 'beta' must be greater than 0, but got {}.".format(beta))
        self.beta = beta
        self.clear()

    def clear(self):
        """Clears the internal evaluation result."""
        self._true_positives = 0
        self._actual_positives = 0
        self._positives = 0
        self._class_num = 0

    @rearrange_inputs
    def update(self, *inputs):
        """
        Updates the internal evaluation result `y_pred` and `y`.

        Args:
            inputs: Input `y_pred` and `y`. `y_pred` and `y` are Tensor, list or numpy.ndarray.
                `y_pred` is in most cases (not strictly) a list of floating numbers in range :math:`[0, 1]`
                and the shape is :math:`(N, C)`, where :math:`N` is the number of cases and :math:`C`
                is the number of categories. y contains values of integers. The shape is :math:`(N, C)`
                if one-hot encoding is used. Shape can also be :math:`(N,)` if category index is used.

        Raises:
            ValueError: class numbers of last input predicted data and current predicted data not match.
            ValueError: If the predicted value and true value contain different classes.
        """
        if len(inputs) != 2:
            raise ValueError("For 'Fbeta.update', it needs 2 inputs (predicted value, true value), "
                             "but got {}.".format(len(inputs)))
        y_pred = self._convert_data(inputs[0])
        y = self._convert_data(inputs[1])
        if y_pred.ndim == y.ndim and _check_onehot_data(y):
            y = y.argmax(axis=1)

        if self._class_num == 0:
            self._class_num = y_pred.shape[1]
        elif y_pred.shape[1] != self._class_num:
            raise ValueError("For 'Fbeta.update', class number not match, last input predicted data contain {} "
                             "classes, but current predicted data contain {} classes, please check your "
                             "predicted value(inputs[0]).".format(self._class_num, y_pred.shape[1]))
        class_num = self._class_num

        if y.max() + 1 > class_num:
            raise ValueError("For 'Fbeta.update', predicted value(inputs[0]) and true value(inputs[1]) "
                             "should contain same classes, but got predicted value contains {} classes"
                             " and true value contains {} classes.".format(class_num, y.max() + 1))
        y = np.eye(class_num)[y.reshape(-1)]
        indices = y_pred.argmax(axis=1).reshape(-1)
        y_pred = np.eye(class_num)[indices]

        positives = y_pred.sum(axis=0)
        actual_positives = y.sum(axis=0)
        true_positives = (y * y_pred).sum(axis=0)

        self._true_positives += true_positives
        self._positives += positives
        self._actual_positives += actual_positives

    def eval(self, average=False):
        """
        Computes the fbeta.

        Args:
            average (bool): Whether to calculate the average fbeta. Default: ``False``.

        Returns:
            numpy.ndarray or numpy.float64, the computed result.
        """
        validator.check_value_type("average", average, [bool], self.__class__.__name__)
        if self._class_num == 0:
            raise RuntimeError("The 'Fbeta' can not be calculated, because the number of samples is 0, "
                               "please check whether your inputs(predicted value, true value) are empty, "
                               "or has called update method before calling eval method.")

        fbeta = (1.0 + self.beta ** 2) * self._true_positives / \
                (self.beta ** 2 * self._actual_positives + self._positives + self.eps)

        if average:
            return fbeta.mean()
        return fbeta


class F1(Fbeta):
    r"""
    Calculates the F1 score. F1 is a special case of Fbeta when beta is 1.
    Refer to class :class:`mindspore.train.Fbeta` for more details.

    .. math::
        F_1=\frac{2\cdot true\_positive}{2\cdot true\_positive + false\_negative + false\_positive}

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> from mindspore.train import F1
        >>>
        >>> x = Tensor(np.array([[0.2, 0.5], [0.3, 0.1], [0.9, 0.6]]))
        >>> y = Tensor(np.array([1, 0, 1]))
        >>> metric = F1()
        >>> metric.update(x, y)
        >>> result = metric.eval()
        >>> print(result)
        [0.66666667 0.66666667]
    """
    def __init__(self):
        super(F1, self).__init__(1.0)
