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
"""Evaluation."""
import numpy as np
from .metric import Metric

_eval_types = {'classification', 'multilabel'}


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
