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
