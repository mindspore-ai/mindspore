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
import math
import numpy as np
from mindspore._checkparam import Validator as validator
from .metric import Metric


class Perplexity(Metric):
    r"""
    Computes perplexity. Perplexity is a measurement about how well a probability distribution or a model predicts a
    sample. A low perplexity indicates the model can predict the sample well. The function is shown as follows:

    .. math::
        b^{\\big(-\\frac{1}{N} \\sum_{i=1}^N \\log_b q(x_i) \\big)}
        = \\exp \\big(-\\frac{1}{N} \\sum_{i=1}^N \\log q(x_i)\\big)

    Args:
        ignore_label (int): Index of an invalid label to be ignored when counting. If set to `None`, it will include all
                            entries. Default: -1.

    Examples:
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

    def update(self, *inputs):
        """
        Updates the internal evaluation result: math:preds and :math:labels.

        Args:
            inputs: Input `preds` and `labels`. `preds` and `labels` are Tensor, list or numpy.ndarray.
                    `preds` is the predicted values, `labels` is the label of the data.
                    The shape of `preds` and `labels` are both :math:`(N, C)`.

        Raises:
            ValueError: If the number of the inputs is not 2.
        """
        if len(inputs) != 2:
            raise ValueError('Perplexity needs 2 inputs (preds, labels), but got {}.'.format(len(inputs)))

        preds = [self._convert_data(inputs[0])]
        labels = [self._convert_data(inputs[1])]

        if len(preds) != len(labels):
            raise RuntimeError('preds and labels should have the same length, but the length of preds is{}, '
                               'the length of labels is {}.'.format(len(preds), len(labels)))

        loss = 0.
        num = 0
        for label, pred in zip(labels, preds):
            if label.size != pred.size / pred.shape[-1]:
                raise RuntimeError("shape mismatch: label shape should be equal to pred shape, but got label shape "
                                   "is {}, pred shape is {}.".format(label.shape, pred.shape))
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
            float, the computed result.

        Raises:
            RuntimeError: If the sample size is 0.
        """
        if self._num_inst == 0:
            raise RuntimeError('Perplexity can not be calculated, because the number of samples is 0.')

        return math.exp(self._sum_metric / self._num_inst)
