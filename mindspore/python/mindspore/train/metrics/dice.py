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
from __future__ import absolute_import

import numpy as np

from mindspore._checkparam import Validator as validator
from mindspore.train.metrics.metric import Metric, rearrange_inputs


class Dice(Metric):
    r"""
    The Dice coefficient is a set similarity metric. It is used to calculate the similarity between two samples. The
    value of the Dice coefficient is 1 when the segmentation result is the best and is 0 when the segmentation result
    is the worst. The Dice coefficient indicates the ratio of the area between two objects to the total area.
    The function is shown as follows:

    .. math::
        dice = \frac{2 * (pred \bigcap true)}{pred \bigcup true}

    Args:
        smooth (float): A term added to the denominator to improve numerical stability. Should be greater than 0.
                        Default: 1e-5.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> from mindspore.train import Dice
        >>>
        >>> x = Tensor(np.array([[0.2, 0.5], [0.3, 0.1], [0.9, 0.6]]))
        >>> y = Tensor(np.array([[0, 1], [1, 0], [0, 1]]))
        >>> metric = Dice(smooth=1e-5)
        >>> metric.clear()
        >>> metric.update(x, y)
        >>> dice = metric.eval()
        >>> print(dice)
        0.20467791371802546
    """

    def __init__(self, smooth=1e-5):
        super(Dice, self).__init__()

        self.smooth = validator.check_positive_float(smooth, "smooth")
        self._dice_coeff_sum = 0
        self._samples_num = 0
        self.clear()

    def clear(self):
        """Clears the internal evaluation result."""
        self._dice_coeff_sum = 0
        self._samples_num = 0

    @rearrange_inputs
    def update(self, *inputs):
        r"""
        Updates the internal evaluation result `y_pred` and `y`.

        Args:
            inputs (tuple): Input `y_pred` and `y`. `y_pred` and `y` are Tensor, list or numpy.ndarray. `y_pred` is the
                predicted value, `y` is the true value. The shape of `y_pred` and `y` are both :math:`(N, ...)`.

        Raises:
            ValueError: If the number of the inputs is not 2.
            ValueError: If y_pred and y do not have the same shape.
        """
        if len(inputs) != 2:
            raise ValueError("For 'Dice.update', it needs 2 inputs (predicted value, true value), "
                             "but got {}.".format(len(inputs)))

        y_pred = self._convert_data(inputs[0])
        y = self._convert_data(inputs[1])
        self._samples_num += y.shape[0]

        if y_pred.shape != y.shape:
            raise ValueError(f"For 'Dice.update', predicted value (input[0]) and true value (input[1]) "
                             f"should have same shape, but got predicted value shape: {y_pred.shape}, "
                             f"true value shape: {y.shape}.")

        intersection = np.dot(y_pred.flatten(), y.flatten())
        unionset = np.dot(y_pred.flatten(), y_pred.flatten()) + np.dot(y.flatten(), y.flatten())

        single_dice_coeff = 2 * float(intersection) / float(unionset + self.smooth)
        self._dice_coeff_sum += single_dice_coeff

    def eval(self):
        r"""
        Computes the Dice.

        Returns:
            Float, the computed result.

        Raises:
            RuntimeError: If the total number of samples is 0.
        """
        if self._samples_num == 0:
            raise RuntimeError("The 'Dice coefficient' can not be calculated, because the number of samples is 0, "
                               "please check whether your inputs(predicted value, true value) are empty, or has "
                               "called update method before calling eval method.")

        return self._dice_coeff_sum / float(self._samples_num)
