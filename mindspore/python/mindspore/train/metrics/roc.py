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
from __future__ import absolute_import

import numpy as np

from mindspore._checkparam import Validator as validator
from mindspore.train.metrics.metric import Metric, rearrange_inputs, _binary_clf_curve


class ROC(Metric):
    """
    Calculates the ROC curve. It is suitable for solving binary classification and multi classification problems.
    In the case of multiclass, the values will be calculated based on a one-vs-the-rest approach.

    Args:
        class_num (int): The number of classes. It is not necessary to provide this argument under the binary
                            classification scenario. Default: None.
        pos_label (int): Determine the integer of positive class. For binary problems, it is translated to 1 by default.
                            For multiclass problems, this argument should not be set, as it will
                            iteratively changed in the range [0,num_classes-1]. Default: None.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> from mindspore.train import ROC
        >>>
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
        >>> print(thresholds)
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

    @rearrange_inputs
    def update(self, *inputs):
        """
        Update state with predictions and targets.

        Args:
            inputs: Input `y_pred` and `y`. `y_pred` and `y` are `Tensor`, list or numpy.ndarray.
                In most cases (not strictly), y_pred is a list of floating numbers in range :math:`[0, 1]`
                and the shape is :math:`(N, C)`, where :math:`N` is the number of cases and :math:`C`
                is the number of categories. y contains values of integers. The shape is :math:`(N, C)` if one-hot
                encoding is used. Shape can also be :math:`(N,)` if category index is used.
        """
        if len(inputs) != 2:
            raise ValueError("For 'ROC.update', it needs 2 inputs (predicted value, true value), but got {}"
                             .format(len(inputs)))
        y_pred = self._convert_data(inputs[0])
        y = self._convert_data(inputs[1])

        y_pred, y, class_num, pos_label = _precision_recall_curve_update(y_pred, y, self.class_num, self.pos_label)

        self.y_pred = y_pred
        self.y = y
        self.class_num = class_num
        self.pos_label = pos_label
        self._is_update = True

    def _roc_eval(self, y_pred, y, class_num, pos_label, sample_weights=None):
        """Computes the ROC curve."""
        if class_num == 1:
            fps, tps, thresholds = _binary_clf_curve(y_pred, y, sample_weights=sample_weights, pos_label=pos_label)
            tps = np.squeeze(np.hstack([np.zeros(1, dtype=tps.dtype), tps]))
            fps = np.squeeze(np.hstack([np.zeros(1, dtype=fps.dtype), fps]))
            thresholds = np.hstack([thresholds[0][None] + 1, thresholds])

            if fps[-1] <= 0:
                raise ValueError("For 'ROC.eval', there is no negative samples in true value, "
                                 "false positive value is meaningless.")
            fpr = fps / fps[-1]

            if tps[-1] <= 0:
                raise ValueError("For 'ROC.eval', there is no positive samples in true value, "
                                 "true positive value is meaningless.")
            tpr = tps / tps[-1]

            return fpr, tpr, thresholds

        fpr, tpr, thresholds = [], [], []
        for c in range(class_num):
            preds_c = y_pred[:, c]
            res = self._roc(preds_c, y, class_num=1, pos_label=c, sample_weights=sample_weights)
            fpr.append(res[0])
            tpr.append(res[1])
            thresholds.append(res[2])

        return fpr, tpr, thresholds

    def _roc(self, y_pred, y, class_num=None, pos_label=None, sample_weights=None):
        """
        Update curve and return the result of the ROC curve.

        Args:
            y_pred (Union[Tensor, list, np.ndarray]): In most cases (not strictly), y_pred is a list of floating numbers
                in range :math:`[0, 1]` and the shape is :math:`(N, C)`, where :math:`N` is the number of cases
                and :math:`C` is the number of categories.
            y (Union[Tensor, list, np.ndarray]): values of integers.
            class_num (int): Integer with the number of classes. For the problem of binary classification, it is not
                necessary to provide this argument. Default: None.
            pos_label (int): Determine the integer of positive class. Default: None. For binary problems, it is
                translated to 1. For multiclass problems, this argument should not be set, as it is iteratively changed
                in the range [0,num_classes-1]. Default: None.
            sample_weights (Union[None, np.ndarray]): If sample_weights is None, the weight value is 1.
                If sample_weights is ndarray, the weight value is the ndarray value.
        """
        y_pred, y, class_num, pos_label = _precision_recall_curve_update(y_pred, y, class_num, pos_label)

        return self._roc_eval(y_pred, y, class_num, pos_label, sample_weights)

    def eval(self):
        """
        Computes the ROC curve.

        Returns:
            A tuple, composed of `fpr`, `tpr`, and `thresholds`.

            - **fpr** (np.array) - False positive rate. In binary classification case, a fpr numpy array under different
              thresholds will be returned, otherwise in multiclass case, a list of
              fpr numpy arrays will be returned and each element represents one category.
            - **tpr** (np.array) - True positive rates. n binary classification case, a tps numpy array under different
              thresholds will be returned, otherwise in multiclass case, a list of tps numpy arrays
              will be returned and each element represents one category.
            - **thresholds** (np.array) - Thresholds used for computing fpr and tpr.

        Raises:
            RuntimeError: If the update method is not called first, an error will be reported.

        """
        if self._is_update is False:
            raise RuntimeError("Please call the 'update' method before calling 'eval' method.")

        y_pred = np.squeeze(self.y_pred)
        y = np.squeeze(self.y)
        return self._roc_eval(y_pred, y, self.class_num, self.pos_label)


def _precision_recall_curve_update(y_pred, y, class_num, pos_label):
    """update curve"""
    if not (len(y_pred.shape) == len(y.shape) or len(y_pred.shape) == len(y.shape) + 1):
        raise ValueError(f"For 'ROC', predicted value (input[0]) and true value (input[1]) must have same "
                         f"dimensions, or the dimension of predicted value equal the dimension of true value add "
                         f"1, but got predicted value ndim: {len(y_pred.shape)}, true value ndim: {len(y.shape)}.")

    # single class evaluation
    if len(y_pred.shape) == len(y.shape):
        if class_num is not None and class_num != 1:
            raise ValueError(f"For 'ROC', when predicted value (input[0]) and true value (input[1]) have the same "
                             f"shape, the 'class_num' must be 1, but got {class_num}.")
        class_num = 1
        if pos_label is None:
            pos_label = 1
        y_pred = y_pred.flatten()
        y = y.flatten()

    # multi class evaluation
    elif len(y_pred.shape) == len(y.shape) + 1:
        if pos_label is not None:
            raise ValueError(f"For 'ROC', when the dimension of predicted value (input[0]) equals the dimension "
                             f"of true value (input[1]) add 1, the 'pos_label' must be None, "
                             f"but got {pos_label}.")
        if class_num != y_pred.shape[1]:
            raise ValueError("For 'ROC', the 'class_num' must equal the number of classes from predicted value "
                             "(input[0]), but got 'class_num' {}, the number of classes from predicted value {}."
                             .format(class_num, y_pred.shape[1]))
        y_pred = y_pred.transpose(0, 1).reshape(class_num, -1).transpose(0, 1)
        y = y.flatten()

    return y_pred, y, class_num, pos_label
