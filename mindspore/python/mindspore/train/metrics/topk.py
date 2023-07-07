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
"""Topk."""
from __future__ import absolute_import

import numpy as np

from mindspore.train.metrics.metric import Metric, rearrange_inputs, _check_onehot_data


class TopKCategoricalAccuracy(Metric):
    """
    Calculates the top-k categorical accuracy.

    Args:
        k (int): Specifies the top-k categorical accuracy to compute.

    Raises:
        TypeError: If `k` is not int.
        ValueError: If `k` is less than 1.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> from mindspore.train import TopKCategoricalAccuracy
        >>>
        >>> x = Tensor(np.array([[0.2, 0.5, 0.3, 0.6, 0.2], [0.1, 0.35, 0.5, 0.2, 0.],
        ...         [0.9, 0.6, 0.2, 0.01, 0.3]]), mindspore.float32)
        >>> y = Tensor(np.array([2, 0, 1]), mindspore.float32)
        >>> topk = TopKCategoricalAccuracy(3)
        >>> topk.clear()
        >>> topk.update(x, y)
        >>> output = topk.eval()
        >>> print(output)
        0.6666666666666666
    """
    def __init__(self, k):
        super(TopKCategoricalAccuracy, self).__init__()
        if not isinstance(k, int):
            raise TypeError("For 'TopKCategoricalAccuracy', the type of "
                            "the argument 'k' should be int, but got 'k' type: {}.".format(type(k)))
        if k < 1:
            raise ValueError("For 'TopKCategoricalAccuracy', "
                             "the argument 'k' must be at least 1, but got 'k' value: {}.".format(k))
        self.k = k
        self.clear()

    def clear(self):
        """Clear the internal evaluation result."""
        self._correct_num = 0
        self._samples_num = 0

    @rearrange_inputs
    def update(self, *inputs):
        """
        Updates the internal evaluation result `y_pred` and `y`.

        Args:
            inputs: Input `y_pred` and `y`. `y_pred` and `y` are Tensor, list or numpy.ndarray.
                `y_pred` is in most cases (not strictly) a list of floating numbers in range :math:`[0, 1]`
                and the shape is :math:`(N, C)`, where :math:`N` is the number of cases and :math:`C`
                is the number of categories. `y` contains values of integers. The shape is :math:`(N, C)`
                if one-hot encoding is used. Shape can also be :math:`(N,)` if category index is used.

        Note:
            The method `update` must receive input of the form :math:`(y_{pred}, y)`. If some samples have
            the same accuracy, the first sample will be chosen.
        """
        if len(inputs) != 2:
            raise ValueError("For 'TopKCategoricalAccuracy.update', "
                             "it needs 2 inputs (predicted value, true value), "
                             "but got 'inputs' size: {}.".format(len(inputs)))
        y_pred = self._convert_data(inputs[0])
        y = self._convert_data(inputs[1])
        if y_pred.ndim == y.ndim and _check_onehot_data(y):
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
            numpy.float64, computed result.
        """
        if self._samples_num == 0:
            raise RuntimeError("The 'TopKCategoricalAccuracy' "
                               "can not be calculated, because the number of samples is 0, "
                               "please check whether your inputs (predicted value, true value) are empty, "
                               "or has called update method before calling eval method.")
        return self._correct_num / self._samples_num


class Top1CategoricalAccuracy(TopKCategoricalAccuracy):
    """
    Calculates the top-1 categorical accuracy. This class is a specialized class for TopKCategoricalAccuracy.
    Refer to :class:`mindspore.train.TopKCategoricalAccuracy` for more details.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore
        >>> from mindspore import Tensor
        >>> from mindspore.train import Top1CategoricalAccuracy
        >>>
        >>> x = Tensor(np.array([[0.2, 0.5, 0.3, 0.6, 0.2], [0.1, 0.35, 0.5, 0.2, 0.],
        ...         [0.9, 0.6, 0.2, 0.01, 0.3]]), mindspore.float32)
        >>> y = Tensor(np.array([2, 0, 1]), mindspore.float32)
        >>> topk = Top1CategoricalAccuracy()
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
    Refer to :class:`mindspore.train.TopKCategoricalAccuracy` for more details.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore
        >>> from mindspore import Tensor
        >>> from mindspore.train import Top5CategoricalAccuracy
        >>>
        >>> x = Tensor(np.array([[0.2, 0.5, 0.3, 0.6, 0.2], [0.1, 0.35, 0.5, 0.2, 0.],
        ...            [0.9, 0.6, 0.2, 0.01, 0.3]]), mindspore.float32)
        >>> y = Tensor(np.array([2, 0, 1]), mindspore.float32)
        >>> topk = Top5CategoricalAccuracy()
        >>> topk.clear()
        >>> topk.update(x, y)
        >>> output = topk.eval()
        >>> print(output)
        1.0
    """
    def __init__(self):
        super(Top5CategoricalAccuracy, self).__init__(5)
