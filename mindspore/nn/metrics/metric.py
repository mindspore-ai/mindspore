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
from scipy.ndimage import morphology
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

    @staticmethod
    def _check_onehot_data(data):
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

    @staticmethod
    def _get_surface_distance(y_pred_edges, y_edges, distance_metric):
        """
        Calculate the surface distances from `y_pred_edges` to `y_edges`.

         Args:
            y_pred_edges (np.ndarray): the edge of the predictions.
            y_edges (np.ndarray): the edge of the ground truth.
            distance_metric (string): The parameter of calculating Hausdorff distance supports three
                                      measurement methods, "euclidean", "chessboard" or "taxicab".
                                      Default: "euclidean".
        """

        if not np.any(y_pred_edges):
            return np.array([])

        if not np.any(y_edges):
            dis = np.full(y_edges.shape, np.inf)
        else:
            if distance_metric == "euclidean":
                dis = morphology.distance_transform_edt(~y_edges)
            else:
                dis = morphology.distance_transform_cdt(~y_edges, metric=distance_metric)

        surface_distance = dis[y_pred_edges]

        return surface_distance

    def _check_surface_distance_inputs(self, inputs):
        """
        Checks the values of y_pred and y.

        Args:
            y_pred (Tensor): Predict array.
            y (Tensor): Target array.
        """
        y_pred = self._convert_data(inputs[0])
        y = self._convert_data(inputs[1])
        label_idx = inputs[2]

        if not isinstance(label_idx, (int, float)):
            raise TypeError("The data type of label_idx must be int or float, but got {}.".format(type(label_idx)))

        if label_idx not in y_pred and label_idx not in y:
            raise ValueError("The label_idx should be in y_pred or y, but {} is not.".format(label_idx))

        if y_pred.size == 0 or y_pred.shape != y.shape:
            raise ValueError("y_pred and y should have same shape, but got {}, {}.".format(y_pred.shape, y.shape))

        if y_pred.dtype != bool:
            y_pred = y_pred == label_idx
        if y.dtype != bool:
            y = y == label_idx

        y_pred_edges = morphology.binary_erosion(y_pred) ^ y_pred
        y_edges = morphology.binary_erosion(y) ^ y

        return y_pred_edges, y_edges

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

    def _check_inputs_shape(self, inputs):
        """
        Checks the values of y_pred and y.

        Args:
            y_pred (Tensor): Predict array.
            y (Tensor): Target array.
        """
        y_pred = self._convert_data(inputs[0])
        y = self._convert_data(inputs[1])
        if self._type == 'classification' and y_pred.ndim == y.ndim and Metric._check_onehot_data(y):
            y = y.argmax(axis=1)
        self._check_shape(y_pred, y)
        self._check_value(y_pred, y)

        return y_pred, y

    def _check_inputs(self, y_pred, y, class_nums):
        """
        Checks the values of y_pred, y and class_nums.

        Args:
            y_pred (Tensor): Predict array.
            y (Tensor): Target array.
            class_nums(int): Class number.
        """
        if class_nums == 0:
            class_nums = y_pred.shape[1]
        elif y_pred.shape[1] != class_nums:
            raise ValueError('Class number not match, last input data contain {} classes, but current data contain {} '
                             'classes'.format(class_nums, y_pred.shape[1]))

        class_num = class_nums
        if self._type == "classification":
            if y.max() + 1 > class_num:
                raise ValueError('y_pred contains {} classes less than y contains {} classes.'.
                                 format(class_num, y.max() + 1))
            y = np.eye(class_num)[y.reshape(-1)]
            indices = y_pred.argmax(axis=1).reshape(-1)
            y_pred = np.eye(class_num)[indices]
        elif self._type == "multilabel":
            y_pred = y_pred.swapaxes(1, 0).reshape(class_num, -1)
            y = y.swapaxes(1, 0).reshape(class_num, -1)

        return y_pred, y, class_nums

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

    @classmethod
    def update(cls, *inputs):
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
