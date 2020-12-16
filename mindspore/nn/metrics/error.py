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
"""Error."""
import numpy as np
from .metric import Metric


class MAE(Metric):
    r"""
    Calculates the mean absolute error.

    Creates a criterion that measures the mean absolute error (MAE)
    between each element in the input: :math:`x` and the target: :math:`y`.

    .. math::
        \text{MAE} = \frac{\sum_{i=1}^n \|y_i - x_i\|}{n}

    Here :math:`y_i` is the prediction and :math:`x_i` is the true value.

    Note:
        The method `update` must be called with the form `update(y_pred, y)`.

    Examples:
        >>> x = Tensor(np.array([0.1, 0.2, 0.6, 0.9]), mindspore.float32)
        >>> y = Tensor(np.array([0.1, 0.25, 0.7, 0.9]), mindspore.float32)
        >>> error = nn.MAE()
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

    def update(self, *inputs):
        """
        Updates the internal evaluation result :math:`y_{pred}` and :math:`y`.

        Args:
            inputs: Input `y_pred` and `y` for calculating mean absolute error where the shape of
                `y_pred` and `y` are both N-D and the shape are the same.

        Raises:
            ValueError: If the number of the input is not 2.
        """
        if len(inputs) != 2:
            raise ValueError('Mean absolute error need 2 inputs (y_pred, y), but got {}'.format(len(inputs)))
        y_pred = self._convert_data(inputs[0])
        y = self._convert_data(inputs[1])
        abs_error_sum = np.abs(y.reshape(y_pred.shape) - y_pred)
        self._abs_error_sum += abs_error_sum.sum()
        self._samples_num += y.shape[0]

    def eval(self):
        """
        Computes the mean absolute error.

        Returns:
            Float, the computed result.

        Raises:
            RuntimeError: If the number of the total samples is 0.
        """
        if self._samples_num == 0:
            raise RuntimeError('Total samples num must not be 0.')
        return self._abs_error_sum / self._samples_num


class MSE(Metric):
    r"""
    Measures the mean squared error.

    Creates a criterion that measures the mean squared error (squared L2
    norm) between each element in the input: :math:`x` and the target: :math:`y`.

    .. math::
        \text{MSE}(x,\ y) = \frac{\sum_{i=1}^n(y_i - x_i)^2}{n}

    where :math:`n` is batch size.

    Examples:
        >>> x = Tensor(np.array([0.1, 0.2, 0.6, 0.9]), mindspore.float32)
        >>> y = Tensor(np.array([0.1, 0.25, 0.5, 0.9]), mindspore.float32)
        >>> error = nn.MSE()
        >>> error.clear()
        >>> error.update(x, y)
        >>> result = error.eval()
    """
    def __init__(self):
        super(MSE, self).__init__()
        self.clear()

    def clear(self):
        """Clear the internal evaluation result."""
        self._squared_error_sum = 0
        self._samples_num = 0

    def update(self, *inputs):
        """
        Updates the internal evaluation result :math:`y_{pred}` and :math:`y`.

        Args:
            inputs: Input `y_pred` and `y` for calculating mean square error where the shape of
                `y_pred` and `y` are both N-D and the shape are the same.

        Raises:
            ValueError: If the number of input is not 2.
        """
        if len(inputs) != 2:
            raise ValueError('Mean squared error need 2 inputs (y_pred, y), but got {}'.format(len(inputs)))

        y_pred = self._convert_data(inputs[0])
        y = self._convert_data(inputs[1])
        squared_error_sum = np.power(y.reshape(y_pred.shape) - y_pred, 2)
        self._squared_error_sum += squared_error_sum.sum()
        self._samples_num += y.shape[0]

    def eval(self):
        """
        Compute the mean squared error.

        Returns:
            Float, the computed result.

        Raises:
            RuntimeError: If the number of samples is 0.
        """
        if self._samples_num == 0:
            raise RuntimeError('The number of input samples must not be 0.')
        return self._squared_error_sum / self._samples_num
