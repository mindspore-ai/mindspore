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
"""auc"""
from __future__ import absolute_import

import numpy as np


def auc(x, y, reorder=False):
    """
    Computes the AUC(Area Under the Curve) using the trapezoidal rule. This is a general function, given points on a
    curve, for computing the area under the ROC-curve.

    Args:
        x (Union[np.array, list]): From the ROC curve(fpr), np.array with false positive rates. If multiclass,
                                   this is a list of such np.array, one for each class. The shape :math:`(N)`.
        y (Union[np.array, list]): From the ROC curve(tpr), np.array with true positive rates. If multiclass,
                                   this is a list of such np.array, one for each class. The shape :math:`(N)`.
        reorder (bool): If False, x must rise or fall monotonously. If True, x will be sorted in ascending order.
                           Default: ``False`` .

    Returns:
        float, the area under the ROC-curve.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore.train import ROC, auc
        >>>
        >>> y_pred = np.array([[3, 0, 1], [1, 3, 0], [1, 0, 2]])
        >>> y = np.array([[0, 2, 1], [1, 2, 1], [0, 0, 1]])
        >>> metric = ROC(pos_label=2)
        >>> metric.clear()
        >>> metric.update(y_pred, y)
        >>> fpr, tpr, thre = metric.eval()
        >>> output = auc(fpr, tpr)
        >>> print(output)
        0.5357142857142857
    """
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("For 'auc', the argument 'x' and 'y' must all be np.ndarray, but got {}, {}"
                        .format(type(x), type(y)))
    _check_consistent_length(x, y)
    x = _column_or_1d(x)
    y = _column_or_1d(y)

    if x.shape[0] < 2:
        raise ValueError("For 'auc', the shape of the argument 'x' in axis 0 must be greater than 2, "
                         "but got {}.".format(x.shape[0]))

    direction = 1
    if reorder:
        order = np.lexsort((y, x))
        x, y = x[order], y[order]
    else:
        dx = np.diff(x)
        if np.any(dx < 0):
            if np.all(dx <= 0):
                direction = -1
            else:
                raise ValueError("For 'auc', if the argument is False, the argument 'x' array must be increasing "
                                 "or decreasing, but got 'x': {}".format(x))

    area = direction * np.trapz(y, x)
    if isinstance(area, np.memmap):
        area = area.dtype.type(area)
    return area


def _column_or_1d(y):
    """
     Ravel column or 1D numpy array, otherwise raise a ValueError.
    """
    shape = np.shape(y)
    if len(shape) == 1 or (len(shape) == 2 and shape[1] == 1):
        return np.ravel(y)

    raise ValueError("For 'auc', the input must be a 1-dimensional array, or a 2-dimensional array with the second "
                     "dimension of 1, but got shape {}.".format(shape))


def _num_samples(x):
    """Return the number of samples in array-like x."""
    if hasattr(x, 'fit') and callable(x.fit):
        raise TypeError('Expected sequence or array-like, got estimator {}.'.format(x))
    if not hasattr(x, '__len__') and not hasattr(x, 'shape'):
        if hasattr(x, '__array__'):
            x = np.asarray(x)
        else:
            raise TypeError("Expected sequence or array-like, got {}." .format(type(x)))
    if hasattr(x, 'shape'):
        if x.ndim == 0:
            raise TypeError("Singleton array {} cannot be considered as a valid collection.".format(x))
        res = x.shape[0]
    else:
        res = x.size

    return res


def _check_consistent_length(*arrays):
    r"""
    Check that all arrays have consistent first dimensions. Check whether all objects in arrays have the same shape
    or length.

    Args:
        - **(*arrays)** - (Union[tuple, list]): list or tuple of input objects. Objects that will be checked for
            consistent length.
    """

    lengths = [_num_samples(array) for array in arrays if array is not None]
    uniques = np.unique(lengths)
    if len(uniques) > 1:
        raise ValueError("Found input variables with inconsistent numbers of samples: {}."
                         .format([int(length) for length in lengths]))
