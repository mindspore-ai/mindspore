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
"""ROC"""
import numpy as np
from mindspore._checkparam import Validator as validator
from .metric import Metric


class ROC(Metric):
    """
    Calculates the ROC curve. It is suitable for solving binary classification and multi classification problems.
    In the case of multiclass, the values will be calculated based on a one-vs-the-rest approach.

    Args:
        class_num (int): Integer with the number of classes. For the problem of binary classification, it is not
            necessary to provide this argument. Default: None.
        pos_label (int): Determine the integer of positive class. Default: None. For binary problems, it is translated
            to 1. For multiclass problems, this argument should not be set, as it is iteratively changed in the
            range [0,num_classes-1]. Default: None.

    Examples:
        >>> # 1) binary classification example
        >>> x = Tensor(np.array([3, 1, 4, 2]))
        >>> y = Tensor(np.array([0, 1, 2, 3]))
        >>> metric = ROC(pos_label=2)
        >>> metric.clear()
        >>> metric.update(x, y)
        >>> fpr, tpr, thresholds = metric.eval()
        >>> print(fpr)
        [0. 0. 0.33333333 0.6666667 1.]
        >>> print(tpr)
        [0. 1. 1. 1. 1.]
        >>> print(thresholds)
        [5 4 3 2 1]
        >>>
        >>> # 2) multiclass classification example
        >>> x = Tensor(np.array([[0.28, 0.55, 0.15, 0.05], [0.10, 0.20, 0.05, 0.05], [0.20, 0.05, 0.15, 0.05],
        ...                     [0.05, 0.05, 0.05, 0.75]]))
        >>> y = Tensor(np.array([0, 1, 2, 3]))
        >>> metric = ROC(class_num=4)
        >>> metric.clear()
        >>> metric.update(x, y)
        >>> fpr, tpr, thresholds = metric.eval()
        >>> print(fpr)
        [array([0., 0., 0.33333333, 0.66666667, 1.]), array([0., 0.33333333, 0.33333333, 1.]),
        array([0., 0.33333333, 1.]), array([0., 0., 1.])]
        >>> print(tpr)
        [array([0., 1., 1., 1., 1.]), array([0., 0., 1., 1.]), array([0., 1., 1.]), array([0., 1., 1.])]
        print(thresholds)
        [array([1.28, 0.28, 0.2, 0.1, 0.05]), array([1.55, 0.55, 0.2, 0.05]), array([1.15, 0.15, 0.05]),
        array([1.75, 0.75, 0.05])]
    """
    def __init__(self, class_num=None, pos_label=None):
        super().__init__()
        self.class_num = class_num if class_num is None else validator.check_value_type("class_num", class_num, [int])
        self.pos_label = pos_label if pos_label is None else validator.check_value_type("pos_label", pos_label, [int])
        self.clear()

    def clear(self):
        """Clear the internal evaluation result."""
        self.y_pred = 0
        self.y = 0
        self.sample_weights = None
        self._is_update = False

    def _precision_recall_curve_update(self, y_pred, y, class_num, pos_label):
        """update curve"""
        if not (len(y_pred.shape) == len(y.shape) or len(y_pred.shape) == len(y.shape) + 1):
            raise ValueError("y_pred and y must have the same number of dimensions, or one additional dimension for"
                             " y_pred.")

        # single class evaluation
        if len(y_pred.shape) == len(y.shape):
            if class_num is not None and class_num != 1:
                raise ValueError('y_pred and y should have the same shape, but number of classes is different from 1.')
            class_num = 1
            if pos_label is None:
                pos_label = 1
            y_pred = y_pred.flatten()
            y = y.flatten()

        # multi class evaluation
        elif len(y_pred.shape) == len(y.shape) + 1:
            if pos_label is not None:
                raise ValueError('Argument `pos_label` should be `None` when running multiclass precision recall '
                                 'curve, but got {}.'.format(pos_label))
            if class_num != y_pred.shape[1]:
                raise ValueError('Argument `class_num` was set to {}, but detected {} number of classes from '
                                 'predictions.'.format(class_num, y_pred.shape[1]))
            y_pred = y_pred.transpose(0, 1).reshape(class_num, -1).transpose(0, 1)
            y = y.flatten()

        return y_pred, y, class_num, pos_label

    def update(self, *inputs):
        """
        Update state with predictions and targets.

        Args:
            inputs: Input `y_pred` and `y`. `y_pred` and `y` are Tensor, list or numpy.ndarray.
                In most cases (not strictly), y_pred is a list of floating numbers in range :math:`[0, 1]`
                and the shape is :math:`(N, C)`, where :math:`N` is the number of cases and :math:`C`
                is the number of categories. y contains values of integers.
        """
        if len(inputs) != 2:
            raise ValueError('ROC need 2 inputs (y_pred, y), but got {}'.format(len(inputs)))
        y_pred = self._convert_data(inputs[0])
        y = self._convert_data(inputs[1])

        y_pred, y, class_num, pos_label = self._precision_recall_curve_update(y_pred, y, self.class_num, self.pos_label)

        self.y_pred = y_pred
        self.y = y
        self.class_num = class_num
        self.pos_label = pos_label
        self._is_update = True

    def _roc_eval(self, y_pred, y, class_num, pos_label, sample_weights=None):
        """Computes the ROC curve."""
        if class_num == 1:
            fps, tps, thresholds = self._binary_clf_curve(y_pred, y, sample_weights=sample_weights,
                                                          pos_label=pos_label)
            tps = np.squeeze(np.hstack([np.zeros(1, dtype=tps.dtype), tps]))
            fps = np.squeeze(np.hstack([np.zeros(1, dtype=fps.dtype), fps]))
            thresholds = np.hstack([thresholds[0][None] + 1, thresholds])

            if fps[-1] <= 0:
                raise ValueError("No negative samples in y, false positive value should be meaningless.")
            fpr = fps / fps[-1]

            if tps[-1] <= 0:
                raise ValueError("No positive samples in y, true positive value should be meaningless.")
            tpr = tps / tps[-1]

            return fpr, tpr, thresholds

        fpr, tpr, thresholds = [], [], []
        for c in range(class_num):
            preds_c = y_pred[:, c]
            res = self.roc(preds_c, y, class_num=1, pos_label=c, sample_weights=sample_weights)
            fpr.append(res[0])
            tpr.append(res[1])
            thresholds.append(res[2])

        return fpr, tpr, thresholds

    def roc(self, y_pred, y, class_num=None, pos_label=None, sample_weights=None):
        """roc"""
        y_pred, y, class_num, pos_label = self._precision_recall_curve_update(y_pred, y, class_num, pos_label)

        return self._roc_eval(y_pred, y, class_num, pos_label, sample_weights)

    def eval(self):
        """
        Computes the ROC curve.

        Returns:
            A tuple, composed of `fpr`, `tpr`, and `thresholds`.

            - **fpr** (np.array) - np.array with false positive rates. If multiclass, this is a list of such np.array,
              one for each class.
            - **tps** (np.array) - np.array with true positive rates. If multiclass, this is a list of such np.array,
              one for each class.
            - **thresholds** (np.array) - thresholds used for computing false- and true positive rates.
        """
        if self._is_update is False:
            raise RuntimeError('Call the update method before calling eval.')

        y_pred = np.squeeze(np.vstack(self.y_pred))
        y = np.squeeze(np.vstack(self.y))

        return self._roc_eval(y_pred, y, self.class_num, self.pos_label)
