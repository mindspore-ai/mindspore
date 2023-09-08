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
"""Metric base class."""
from __future__ import absolute_import

from abc import ABCMeta, abstractmethod
import functools
import numpy as np

from mindspore.common.tensor import Tensor

_eval_types = {'classification', 'multilabel'}


def rearrange_inputs(func):
    """
    This decorator is used to rearrange the inputs according to its `indexes` attribute of the class.

    This decorator is currently applied on the `update` of :class:`mindspore.train.Metric`.

    Args:
        func (Callable): A candidate function to be wrapped whose input will be rearranged.

    Returns:
        Callable, used to exchange metadata between functions.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> from mindspore.train import rearrange_inputs
        >>> class RearrangeInputsExample:
        ...     def __init__(self):
        ...         self._indexes = None
        ...
        ...     @property
        ...     def indexes(self):
        ...         return getattr(self, '_indexes', None)
        ...
        ...     def set_indexes(self, indexes):
        ...         self._indexes = indexes
        ...         return self
        ...
        ...     @rearrange_inputs
        ...     def update(self, *inputs):
        ...         return inputs
        >>>
        >>> rearrange_inputs_example = RearrangeInputsExample().set_indexes([1, 0])
        >>> outs = rearrange_inputs_example.update(5, 9)
        >>> print(outs)
        (9, 5)
    """
    @functools.wraps(func)
    def wrapper(self, *inputs):
        indexes = self.indexes
        inputs = inputs if not indexes else [inputs[i] for i in indexes]
        return func(self, *inputs)
    return wrapper


class Metric(metaclass=ABCMeta):
    """
    Base class of metric, which is used to evaluate metrics.

    The `clear`, `update`, and `eval` should be called when evaluating metric, and they should be overridden by
    subclasse. `update` will accumulate intermediate results in the evaluation process, `eval` will evaluate the final
    result, and `clear` will reinitialize the intermediate results.

    Never use this class directly, but instantiate one of its subclasses instead, for examples,
    :class:`mindspore.train.MAE`, :class:`mindspore.train.Recall` etc.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore as ms
        >>>
        >>> class MyMAE(ms.train.Metric):
        ...     def __init__(self):
        ...         super(MyMAE, self).__init__()
        ...         self.clear()
        ...
        ...     def clear(self):
        ...         self._abs_error_sum = 0
        ...         self._samples_num = 0
        ...
        ...     def update(self, *inputs):
        ...         y_pred = inputs[0].asnumpy()
        ...         y = inputs[1].asnumpy()
        ...         abs_error_sum = np.abs(y - y_pred)
        ...         self._abs_error_sum += abs_error_sum.sum()
        ...         self._samples_num += y.shape[0]
        ...
        ...     def eval(self):
        ...         return self._abs_error_sum / self._samples_num
        >>>
        >>> x = ms.Tensor(np.array([[0.1, 0.2, 0.6, 0.9], [0.1, 0.2, 0.6, 0.9]]), ms.float32)
        >>> y = ms.Tensor(np.array([[0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1]]), ms.float32)
        >>> y2 = ms.Tensor(np.array([[0.1, 0.25, 0.7, 0.9], [0.1, 0.25, 0.7, 0.9]]), ms.float32)
        >>> metric = MyMAE().set_indexes([0, 2])
        >>> metric.clear()
        >>> # indexes is [0, 2], using x as logits, y2 as label.
        >>> metric.update(x, y, y2)
        >>> accuracy = metric.eval()
        >>> print(accuracy)
        1.399999976158142
        >>> print(metric.indexes)
        [0, 2]
    """
    def __init__(self):
        self._indexes = None

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
            raise TypeError(f"For 'Metric' and its derived classes, the input data type must be tensor, list or "
                            f"numpy.ndarray, but got {type(data)}.")
        return data

    @property
    def indexes(self):
        """Get the current indexes value. The default value is None and can be changed by `set_indexes`.
        """
        return getattr(self, '_indexes', None)

    def set_indexes(self, indexes):
        """
        This interface is to rearrange the inputs of `update`.

        Given (label0, label1, logits), set the `indexes` to [2, 1] then the (logits, label1) will be the actually
        inputs of `update`.

        Note:
            When customize a metric, decorate the `update` function with the decorator
            :func:`mindspore.train.rearrange_inputs` for the `indexes` to take effect.

        Args:
            indexes (List(int)): The order of logits and labels to be rearranged.

        Outputs:
            :class:`Metric`, its original Class instance.

        Raises:
            ValueError: If the type of input 'indexes'  is not a list or its elements are not all int.
        """
        if not isinstance(indexes, list) or not all(isinstance(i, int) for i in indexes):
            raise ValueError("For 'set_indexes', the argument 'indexes' must be a list and all its elements must "
                             "be int, please check whether it is correct.")
        self._indexes = indexes
        return self

    def __call__(self, *inputs):
        """
        Evaluate input data once.

        Args:
            inputs (tuple): The first item is a predict array, the second item is a target array.

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

        Tutorial Examples:
            - `Evaluation Metrics - Customized Metrics
              <https://mindspore.cn/tutorials/en/master/advanced/model/metric.html#customized-metrics>`_
        """
        raise NotImplementedError('Must define clear function to use this base class')

    @abstractmethod
    def eval(self):
        """
        An interface describes the behavior of computing the evaluation result.

        Note:
            All subclasses must override this interface.

        Tutorial Examples:
            - `Evaluation Metrics - Customized Metrics
              <https://mindspore.cn/tutorials/en/master/advanced/model/metric.html#customized-metrics>`_
        """
        raise NotImplementedError('Must define eval function to use this base class')

    @abstractmethod
    def update(self, *inputs):
        """
        An interface describes the behavior of updating the internal evaluation result.

        Note:
            All subclasses must override this interface.

        Args:
            inputs: A variable-length input argument list, usually are the logits and the corresponding labels.

        Tutorial Examples:
            - `Evaluation Metrics - Customized Metrics
              <https://mindspore.cn/tutorials/en/master/advanced/model/metric.html#customized-metrics>`_
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
            raise TypeError("The argument 'eval_type' must be in {}, but got {}".format(_eval_types, eval_type))
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
                raise ValueError("In classification case, the dimension of y_pred (predicted value) should equal to "
                                 "the dimension of y (true value) add 1, but got y_pred dimension: {} and y "
                                 "dimension: {}.".format(y_pred.ndim, y.ndim))
            if y.shape != (y_pred.shape[0],) + y_pred.shape[2:]:
                raise ValueError("In classification case, y_pred (predicted value) shape and y (true value) shape "
                                 "can not match, y shape should be equal to y_pred shape that the value at index 1 "
                                 "is deleted. Such as y_pred shape (1, 2, 3), then y shape should be (1, 3). "
                                 "But got y_pred shape {} and y shape {}".format(y_pred.shape, y.shape))
        else:
            if y_pred.ndim != y.ndim:
                raise ValueError("In {} case, the dimension of y_pred (predicted value) should equal to the dimension"
                                 " of y (true value), but got y_pred dimension: {} and y dimension: {}."
                                 .format(self._type, y_pred.ndim, y.ndim))
            if y_pred.shape != y.shape:
                raise ValueError("In {} case, the shape of y_pred (predicted value) should equal to the shape of y "
                                 "(true value), but got y_pred shape: {} and y shape: {}."
                                 .format(self._type, y_pred.shape, y.shape))

    def _check_value(self, y_pred, y):
        """
        Checks the values of y_pred and y.

        Args:
            y_pred (Tensor): Predict array.
            y (Tensor): Target array.
        """
        if self._type != 'classification' and not (np.equal(y_pred ** 2, y_pred).all() and np.equal(y ** 2, y).all()):
            raise ValueError("In multilabel case, all elements in y_pred (predicted value) and y (true value) should "
                             "be 0 or 1.Please check whether your inputs y_pred and y are correct.")

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
            inputs: The first item is a predicted array and the second item is a target array.
        """
        raise NotImplementedError

    def eval(self):
        """
        A interface describes the behavior of computing the evaluation result.

        Note:
            All subclasses must override this interface.
        """
        raise NotImplementedError


def _check_onehot_data(data):
    """
    Whether input data is one-hot encoding.

    Args:
        data (numpy.array): Input data.

    Returns:
        bool, return true, if input data is one-hot encoding.
    """
    if data.ndim > 1 and np.equal(data ** 2, data).all():
        shp = (data.shape[0],) + data.shape[2:]
        if np.equal(np.ones(shp), data.sum(axis=1)).all():
            return True
    return False


def _binary_clf_curve(preds, target, sample_weights=None, pos_label=1):
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
