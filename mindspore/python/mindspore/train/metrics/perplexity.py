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
"""Perplexity"""
from __future__ import absolute_import

import math
import numpy as np

from mindspore import _checkparam as validator
from mindspore.train.metrics.metric import Metric, rearrange_inputs


class Perplexity(Metric):
    r"""
    Computes perplexity. Perplexity is a measurement about how well a probability distribution or a model predicts a
    sample. A low perplexity indicates the model can predict the sample well. The function is shown as follows:

    .. math::
        PP(W)=P(w_{1}w_{2}...w_{N})^{-\frac{1}{N}}=\sqrt[N]{\frac{1}{P(w_{1}w_{2}...w_{N})}}

    Where :math:`w` represents words in corpus.

    Args:
        ignore_label (Union[int, None]): Index of an invalid label to be ignored when counting. If set to `None`,
                it will include all entries. Default: ``None`` .

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> from mindspore.train import Perplexity
        >>> x = Tensor(np.array([[0.2, 0.5], [0.3, 0.1], [0.9, 0.6]]))
        >>> y = Tensor(np.array([1, 0, 1]))
        >>> metric = Perplexity(ignore_label=None)
        >>> metric.clear()
        >>> metric.update(x, y)
        >>> perplexity = metric.eval()
        >>> print(perplexity)
        2.231443166940565
    """

    def __init__(self, ignore_label=None):
        super(Perplexity, self).__init__()

        if ignore_label is None:
            self.ignore_label = ignore_label
        else:
            self.ignore_label = validator.check_value_type("ignore_label", ignore_label, [int])
        self.clear()

    def clear(self):
        """Clears the internal evaluation result."""
        self._sum_metric = 0.0
        self._num_inst = 0

    @rearrange_inputs
    def update(self, *inputs):
        """
        Updates the internal evaluation result `preds` and `labels`.

        Args:
            inputs: Input `preds` and `labels`. `preds` and `labels` are a `Tensor`, list or numpy.ndarray.
                    `preds` is the predicted values, `labels` is the labels of the data.
                    The shape of `preds` and `labels` are both :math:`(N, C)`.

        Raises:
            ValueError: If the number of the inputs is not 2.
            RuntimeError: If preds and labels have different lengths.
            RuntimeError: If label shape is not equal to pred shape.
        """
        if len(inputs) != 2:
            raise ValueError("For 'Perplexity.update', it needs 2 inputs (predicted value, label), but got {}."
                             .format(len(inputs)))

        preds = [self._convert_data(inputs[0])]
        labels = [self._convert_data(inputs[1])]

        if len(preds) != len(labels):
            raise RuntimeError("For 'Perplexity.update', predicted value (input[0]) and label (input[1]) should have "
                               "the same length, but got predicted value length {}, label length {}."
                               .format(len(preds), len(labels)))

        loss = 0.
        num = 0
        for label, pred in zip(labels, preds):
            if label.size != pred.size / pred.shape[-1]:
                raise RuntimeError("For 'Perplexity.update', predicted value (input[0]) and label (input[1]) should "
                                   "have the same shape, but got predicted value shape {}, label shape {}."
                                   .format(pred.shape, label.shape))
            label = label.reshape((label.size,))
            label_expand = label.astype(int)
            label_expand = np.expand_dims(label_expand, axis=1)
            first_indices = np.arange(label_expand.shape[0])[:, None]
            pred = np.squeeze(pred[first_indices, label_expand])
            if self.ignore_label is not None:
                ignore = (label == self.ignore_label).astype(pred.dtype)
                num -= np.sum(ignore)
                pred = pred * (1 - ignore) + ignore
            loss -= np.sum(np.log(np.maximum(1e-10, pred)))
            num += pred.size
        self._sum_metric += loss
        self._num_inst += num

    def eval(self):
        r"""
        Returns the current evaluation result.

        Returns:
            numpy.float64. The computed result.

        Raises:
            RuntimeError: If the sample size is 0.
        """
        if self._num_inst == 0:
            raise RuntimeError("The 'Perplexity' can not be calculated, because the number of samples is 0, please "
                               "check whether has called update method before calling eval method.")

        return math.exp(self._sum_metric / self._num_inst)
