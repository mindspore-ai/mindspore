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
"""Error."""
from __future__ import absolute_import

import numpy as np

from mindspore.train.metrics.metric import Metric, rearrange_inputs


class MAE(Metric):
    r"""
    Calculates the mean absolute error(MAE).

    Creates a criterion that measures the MAE between each element
    in the input: :math:`x` and the target: :math:`y`.

    .. math::
        \text{MAE} = \frac{\sum_{i=1}^n \|{y\_pred}_i - y_i\|}{n}

    where :math:`n` is batch size.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore
        >>> from mindspore import Tensor
        >>> from mindspore.train import MAE
        >>>
        >>> x = Tensor(np.array([0.1, 0.2, 0.6, 0.9]), mindspore.float32)
        >>> y = Tensor(np.array([0.1, 0.25, 0.7, 0.9]), mindspore.float32)
        >>> error = MAE()
        >>> error.clear()
        >>> error.update(x, y)
        >>> result = error.eval()
        >>> print(result)
        0.037499990314245224
    """
    def __init__(self):
        super(MAE, self).__init__()
        self.clear()

    def clear(self):
        """Clears the internal evaluation result."""
        self._abs_error_sum = 0
        self._samples_num = 0

    @rearrange_inputs
    def update(self, *inputs):
        """
        Updates the internal evaluation result :math:`y_{pred}` and :math:`y`.

        Args:
            inputs: Input `y_pred` and `y` for calculating MAE where the shape of
                `y_pred` and `y` are both N-D and the shape should be the same.

        Raises:
            ValueError: If the number of the input is not 2.
        """
        if len(inputs) != 2:
            raise ValueError("For 'MAE.update', it needs 2 inputs (predicted value, true value), "
                             "but got {}.".format(len(inputs)))
        y_pred = self._convert_data(inputs[0])
        y = self._convert_data(inputs[1])
        abs_error_sum = np.abs(y.reshape(y_pred.shape) - y_pred)
        self._abs_error_sum += abs_error_sum.sum()
        self._samples_num += y.shape[0]

    def eval(self):
        """
        Computes the mean absolute error(MAE).

        Returns:
            numpy.float64. The computed result.

        Raises:
            RuntimeError: If the total number of samples is 0.
        """
        if self._samples_num == 0:
            raise RuntimeError("The 'MAE' can not be calculated, because the number of samples is 0, "
                               "please check whether your inputs(predicted value, true value) are empty, "
                               "or has called update method before calling eval method.")
        return self._abs_error_sum / self._samples_num


class MSE(Metric):
    r"""
    Measures the mean squared error(MSE).

    Creates a criterion that measures the MSE (squared L2 norm) between
    each element in the prediction and the ground truth: :math:`x` and: :math:`y`.

    .. math::
        \text{MSE}(x,\ y) = \frac{\sum_{i=1}^n({y\_pred}_i - y_i)^2}{n}

    where :math:`n` is batch size.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore
        >>> from mindspore import Tensor
        >>> from mindspore.train import MSE
        >>>
        >>> x = Tensor(np.array([0.1, 0.2, 0.6, 0.9]), mindspore.float32)
        >>> y = Tensor(np.array([0.1, 0.25, 0.5, 0.9]), mindspore.float32)
        >>> error = MSE()
        >>> error.clear()
        >>> error.update(x, y)
        >>> result = error.eval()
        >>> print(result)
        0.0031250009778887033
    """
    def __init__(self):
        super(MSE, self).__init__()
        self.clear()

    def clear(self):
        """Clear the internal evaluation result."""
        self._squared_error_sum = 0
        self._samples_num = 0

    @rearrange_inputs
    def update(self, *inputs):
        """
        Updates the internal evaluation result :math:`y_{pred}` and :math:`y`.

        Args:
            inputs: Input `y_pred` and `y` for calculating the MSE where the shape of
                `y_pred` and `y` are both N-D and the shape should be the same.

        Raises:
            ValueError: If the number of inputs is not 2.
        """
        if len(inputs) != 2:
            raise ValueError("For 'MSE.update', it needs 2 inputs (predicted value, true value), "
                             "but got {}.".format(len(inputs)))

        y_pred = self._convert_data(inputs[0])
        y = self._convert_data(inputs[1])
        squared_error_sum = np.power(y.reshape(y_pred.shape) - y_pred, 2)
        self._squared_error_sum += squared_error_sum.sum()
        self._samples_num += y.shape[0]

    def eval(self):
        """
        Computes the mean squared error(MSE).

        Returns:
            numpy.float64. The computed result.

        Raises:
            RuntimeError: If the number of samples is 0.
        """
        if self._samples_num == 0:
            raise RuntimeError("The 'MSE' can not be calculated, because the number of samples is 0, "
                               "please check whether your inputs(predicted value, true value) are empty, "
                               "or has called update method before calling eval method.")
        return self._squared_error_sum / self._samples_num
