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
"""Topk."""
import numpy as np
from .metric import Metric


class TopKCategoricalAccuracy(Metric):
    """
    Calculates the top-k categorical accuracy.

    Note:
        The method `update` must receive input of the form :math:`(y_{pred}, y)`. If some samples have
        the same accuracy, the first sample will be chosen.

    Args:
        k (int): Specifies the top-k categorical accuracy to compute.

    Raises:
        TypeError: If `k` is not int.
        ValueError: If `k` is less than 1.

    Examples:
        >>> x = Tensor(np.array([[0.2, 0.5, 0.3, 0.6, 0.2], [0.1, 0.35, 0.5, 0.2, 0.],
        ...         [0.9, 0.6, 0.2, 0.01, 0.3]]), mindspore.float32)
        >>> y = Tensor(np.array([2, 0, 1]), mindspore.float32)
        >>> topk = nn.TopKCategoricalAccuracy(3)
        >>> topk.clear()
        >>> topk.update(x, y)
        >>> output = topk.eval()
        >>> print(output)
        0.6666666666666666
    """
    def __init__(self, k):
        super(TopKCategoricalAccuracy, self).__init__()
        if not isinstance(k, int):
            raise TypeError('k should be integer type, but got {}'.format(type(k)))
        if k < 1:
            raise ValueError('k must be at least 1, but got {}'.format(k))
        self.k = k
        self.clear()

    def clear(self):
        """Clear the internal evaluation result."""
        self._correct_num = 0
        self._samples_num = 0

    def update(self, *inputs):
        """
        Updates the internal evaluation result y_pred and y.

        Args:
            inputs: Input y_pred and y. y_pred and y are Tensor, list or numpy.ndarray.
                y_pred is in most cases (not strictly) a list of floating numbers in range :math:`[0, 1]`
                and the shape is :math:`(N, C)`, where :math:`N` is the number of cases and :math:`C`
                is the number of categories. y contains values of integers. The shape is :math:`(N, C)`
                if one-hot encoding is used. Shape can also be :math:`(N,)` if category index is used.
        """
        if len(inputs) != 2:
            raise ValueError('Topk need 2 inputs (y_pred, y), but got {}'.format(len(inputs)))

        y_pred = self._convert_data(inputs[0])
        y = self._convert_data(inputs[1])
        if y_pred.ndim == y.ndim and self._check_onehot_data(y):
            y = y.argmax(axis=1)
        indices = np.argsort(-y_pred, axis=1)[:, :self.k]
        repeated_y = y.reshape(-1, 1).repeat(self.k, axis=1)
        correct = np.equal(indices, repeated_y).sum(axis=1)
        self._correct_num += correct.sum()
        self._samples_num += repeated_y.shape[0]

    def eval(self):
        """
        Computes the top-k categorical accuracy.

        Returns:
            Float, computed result.
        """
        if self._samples_num == 0:
            raise RuntimeError('Total samples num must not be 0.')
        return self._correct_num / self._samples_num


class Top1CategoricalAccuracy(TopKCategoricalAccuracy):
    """
    Calculates the top-1 categorical accuracy. This class is a specialized class for TopKCategoricalAccuracy.
    Refer to class 'TopKCategoricalAccuracy' for more details.

    Examples:
        >>> x = Tensor(np.array([[0.2, 0.5, 0.3, 0.6, 0.2], [0.1, 0.35, 0.5, 0.2, 0.],
        ...         [0.9, 0.6, 0.2, 0.01, 0.3]]), mindspore.float32)
        >>> y = Tensor(np.array([2, 0, 1]), mindspore.float32)
        >>> topk = nn.Top1CategoricalAccuracy()
        >>> topk.clear()
        >>> topk.update(x, y)
        >>> output = topk.eval()
        >>> print(output)
        0.0
    """
    def __init__(self):
        super(Top1CategoricalAccuracy, self).__init__(1)


class Top5CategoricalAccuracy(TopKCategoricalAccuracy):
    """
    Calculates the top-5 categorical accuracy. This class is a specialized class for TopKCategoricalAccuracy.
    Refer to class 'TopKCategoricalAccuracy' for more details.

    Examples:
        >>> x = Tensor(np.array([[0.2, 0.5, 0.3, 0.6, 0.2], [0.1, 0.35, 0.5, 0.2, 0.],
        ...            [0.9, 0.6, 0.2, 0.01, 0.3]]), mindspore.float32)
        >>> y = Tensor(np.array([2, 0, 1]), mindspore.float32)
        >>> topk = nn.Top5CategoricalAccuracy()
        >>> topk.clear()
        >>> topk.update(x, y)
        >>> output = topk.eval()
        >>> print(output)
        1.0
    """
    def __init__(self):
        super(Top5CategoricalAccuracy, self).__init__(5)
