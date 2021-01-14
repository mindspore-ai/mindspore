# Copyright 2021 Huawei Technologies Co., Ltd
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
"""Dice"""
import numpy as np
from mindspore._checkparam import Validator as validator
from .metric import Metric


class Dice(Metric):
    r"""
    The Dice coefficient is a set similarity metric. It is used to calculate the similarity between two samples. The
    value of the Dice coefficient is 1 when the segmentation result is the best and 0 when the segmentation result
    is the worst. The Dice coefficient indicates the ratio of the area between two objects to the total area.
    The function is shown as follows:

    .. math::
            dice = \frac{2 * (pred \bigcap true)}{pred \bigcup true}

    Args:
        smooth (float): A term added to the denominator to improve numerical stability. Should be greater than 0.
                        Default: 1e-5.
        threshold (float): A threshold, which is used to compare with the input tensor. Default: 0.5.

    Examples:
        >>> x = Tensor(np.array([[0.2, 0.5], [0.3, 0.1], [0.9, 0.6]]))
        >>> y = Tensor(np.array([[0, 1], [1, 0], [0, 1]]))
        >>> metric = Dice(smooth=1e-5, threshold=0.5)
        >>> metric.clear()
        >>> metric.update(x, y)
        >>> dice = metric.eval()
        0.22222926
    """

    def __init__(self, smooth=1e-5, threshold=0.5):
        super(Dice, self).__init__()

        self.smooth = validator.check_positive_float(smooth, "smooth")
        self.threshold = validator.check_value_type("threshold", threshold, [float])
        self.clear()

    def clear(self):
        """Clears the internal evaluation result."""
        self._dim = 0
        self.intersection = 0
        self.unionset = 0

    def update(self, *inputs):
        """
        Updates the internal evaluation result :math:`y_pred` and :math:`y`.

        Args:
            inputs: Input `y_pred` and `y`. `y_pred` and `y` are Tensor, list or numpy.ndarray. `y_pred` is the
                    predicted value, `y` is the true value. The shape of `y_pred` and `y` are both :math:`(N, C)`.

        Raises:
            ValueError: If the number of the inputs is not 2.
        """
        if len(inputs) != 2:
            raise ValueError('Dice need 2 inputs (y_pred, y), but got {}'.format(len(inputs)))

        y_pred = self._convert_data(inputs[0])
        y = self._convert_data(inputs[1])

        if y_pred.shape != y.shape:
            raise RuntimeError('y_pred and y should have same the dimension, but the shape of y_pred is{}, '
                               'the shape of y is {}.'.format(y_pred.shape, y.shape))

        y_pred = (y_pred > self.threshold).astype(int)
        self._dim = y.shape
        pred_flat = np.reshape(y_pred, (self._dim[0], -1))
        true_flat = np.reshape(y, (self._dim[0], -1))
        self.intersection = np.sum((pred_flat * true_flat), axis=1)
        self.unionset = np.sum(pred_flat, axis=1) + np.sum(true_flat, axis=1)

    def eval(self):
        r"""
        Computes the Dice.

        Returns:
            Float, the computed result.

        Raises:
            RuntimeError: If the sample size is 0.
        """
        if self._dim[0] == 0:
            raise RuntimeError('Dice can not be calculated, because the number of samples is 0.')

        dice = (2 * self.intersection + self.smooth) / (self.unionset + self.smooth)

        return np.sum(dice) / self._dim[0]
