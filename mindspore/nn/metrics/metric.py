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
"""Metric base class."""
from abc import ABCMeta, abstractmethod
import numpy as np
from mindspore.common.tensor import Tensor

_eval_types = {'classification', 'multilabel'}


class Metric(metaclass=ABCMeta):
    """
    Base class of metric.


    Note:
        For examples of subclasses, please refer to the definition of class `MAE`, 'Recall' etc.
    """
    def __init__(self):
        pass

    def _convert_data(self, data):
        """
        Convert data type to numpy array.

        Args:
            data (Object): Input data.

        Returns:
            Ndarray, data with `np.ndarray` type.
        """
        if isinstance(data, Tensor):
            data = data.asnumpy()
        elif isinstance(data, list):
            data = np.array(data)
        elif isinstance(data, np.ndarray):
            pass
        else:
            raise TypeError('Input data type must be tensor, list or numpy.ndarray')
        return data

    def _check_onehot_data(self, data):
        """
        Whether input data are one-hot encoding.

        Args:
            data (numpy.array): Input data.

        Returns:
            bool, return true, if input data are one-hot encoding.
        """
        if data.ndim > 1 and np.equal(data ** 2, data).all():
            shp = (data.shape[0],) + data.shape[2:]
            if np.equal(np.ones(shp), data.sum(axis=1)).all():
                return True
        return False

    def _binary_clf_curve(self, preds, target, sample_weights=None, pos_label=1):
        """Calculate True Positives and False Positives per binary classification threshold."""
        if sample_weights is not None and not isinstance(sample_weights, np.ndarray):
            sample_weights = np.array(sample_weights)

        if preds.ndim > target.ndim:
            preds = preds[:, 0]
        desc_score_indices = np.argsort(-preds)

        preds = preds[desc_score_indices]
        target = target[desc_score_indices]

        if sample_weights is not None:
            weight = sample_weights[desc_score_indices]
        else:
            weight = 1.

        distinct_value_indices = np.where(preds[1:] - preds[:-1])[0]
        threshold_idxs = np.pad(distinct_value_indices, (0, 1), constant_values=target.shape[0] - 1)
        target = np.array(target == pos_label).astype(np.int64)
        tps = np.cumsum(target * weight, axis=0)[threshold_idxs]

        if sample_weights is not None:
            fps = np.cumsum((1 - target) * weight, axis=0)[threshold_idxs]
        else:
            fps = 1 + threshold_idxs - tps

        return fps, tps, preds[threshold_idxs]

    def __call__(self, *inputs):
        """
        Evaluate input data once.

        Args:
            inputs (tuple): The first item is predict array, the second item is target array.

        Returns:
            Float, compute result.
        """
        self.clear()
        self.update(*inputs)
        return self.eval()

    @abstractmethod
    def clear(self):
        """
        An interface describes the behavior of clearing the internal evaluation result.

        Note:
            All subclasses must override this interface.
        """
        raise NotImplementedError('Must define clear function to use this base class')

    @abstractmethod
    def eval(self):
        """
        An interface describes the behavior of computing the evaluation result.

        Note:
            All subclasses must override this interface.
        """
        raise NotImplementedError('Must define eval function to use this base class')

    @abstractmethod
    def update(self, *inputs):
        """
        An interface describes the behavior of updating the internal evaluation result.

        Note:
            All subclasses must override this interface.

        Args:
            inputs: A variable-length input argument list.
        """
        raise NotImplementedError('Must define update function to use this base class')


class EvaluationBase(Metric):
    """
    Base class of evaluation.

    Note:
        Please refer to the definition of class `Accuracy`.

    Args:
        eval_type (str): Type of evaluation must be in {'classification', 'multilabel'}.

    Raises:
        TypeError: If the input type is not classification or multilabel.
    """
    def __init__(self, eval_type):
        super(EvaluationBase, self).__init__()
        if eval_type not in _eval_types:
            raise TypeError('Type must be in {}, but got {}'.format(_eval_types, eval_type))
        self._type = eval_type

    def _check_shape(self, y_pred, y):
        """
        Checks the shapes of y_pred and y.

        Args:
            y_pred (Tensor): Predict array.
            y (Tensor): Target array.
        """
        if self._type == 'classification':
            if y_pred.ndim != y.ndim + 1:
                raise ValueError('Classification case, dims of y_pred equal dims of y add 1, '
                                 'but got y_pred: {} dims and y: {} dims'.format(y_pred.ndim, y.ndim))
            if y.shape != (y_pred.shape[0],) + y_pred.shape[2:]:
                raise ValueError('Classification case, y_pred shape and y shape can not match. '
                                 'got y_pred shape is {} and y shape is {}'.format(y_pred.shape, y.shape))
        else:
            if y_pred.ndim != y.ndim:
                raise ValueError('{} case, dims of y_pred need equal with dims of y, but got y_pred: {} '
                                 'dims and y: {} dims.'.format(self._type, y_pred.ndim, y.ndim))
            if y_pred.shape != y.shape:
                raise ValueError('{} case, y_pred shape need equal with y shape, but got y_pred: {} and y: {}'.
                                 format(self._type, y_pred.shape, y.shape))

    def _check_value(self, y_pred, y):
        """
        Checks the values of y_pred and y.

        Args:
            y_pred (Tensor): Predict array.
            y (Tensor): Target array.
        """
        if self._type != 'classification' and not (np.equal(y_pred ** 2, y_pred).all() and np.equal(y ** 2, y).all()):
            raise ValueError('For multilabel case, input value must be 1 or 0.')

    def clear(self):
        """
        A interface describes the behavior of clearing the internal evaluation result.

        Note:
            All subclasses must override this interface.
        """
        raise NotImplementedError

    def update(self, *inputs):
        """
        A interface describes the behavior of updating the internal evaluation result.

        Note:
            All subclasses must override this interface.

        Args:
            inputs: The first item is predicted array and the second item is target array.
        """
        raise NotImplementedError

    def eval(self):
        """
        A interface describes the behavior of computing the evaluation result.

        Note:
            All subclasses must override this interface.
        """
        raise NotImplementedError
