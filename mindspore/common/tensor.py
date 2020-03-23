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
"""Tensor implementation."""
import numpy as np

from .._c_expression import Tensor as Tensor_
from .._c_expression import MetaTensor
from .._checkparam import check_type, check_typename
from . import dtype as mstype
from ._register_for_tensor import tensor_operator_registry

__all__ = ['Tensor', 'MetaTensor']


class Tensor(Tensor_):
    """
    Tensor for data storage.

    Tensor inherits tensor object in C++ side, some functions are implemented
    in C++ side and some functions are implemented in Python layer.

    Args:
        input_data (Tensor, float, int, bool, tuple, list, numpy.ndarray): Input data of the tensor.
        dtype (:class:`mindspore.dtype`): Should be None, bool or numeric type defined in `mindspore.dtype`.
            The argument is used to define the data type of the output tensor. If it is None, the data type of the
            output tensor will be as same as the `input_data`. Default: None.

    Outputs:
        Tensor, with the same shape as `input_data`.

    Examples:
        >>> # init a tensor with input data
        >>> t1 = mindspore.Tensor(np.zeros([1, 2, 3]), mindspore.float32)
        >>> assert isinstance(t1, mindspore.Tensor)
        >>> assert t1.shape() == (1, 2, 3)
        >>> assert t1.dtype() == mindspore.float32
        >>>
        >>> # init a tensor with a float scalar
        >>> t2 = mindspore.Tensor(0.1)
        >>> assert isinstance(t2, mindspore.Tensor)
        >>> assert t2.dtype() == mindspore.float64
    """

    def __init__(self, input_data, dtype=None):
        # If input_data is tuple/list/numpy.ndarray, it's support in check_type method.
        check_type('tensor input_data', input_data, (Tensor_, float, int))
        if dtype is not None:
            check_typename('dtype', dtype, mstype.number_type + (mstype.bool_,))
        if isinstance(input_data, np.ndarray) and (not input_data.flags['FORC']):
            input_data = np.ascontiguousarray(input_data)
        if dtype is None:
            super(Tensor, self).__init__(input_data)
        else:
            super(Tensor, self).__init__(input_data, dtype)
        self._virtual_flag = False

    def __repr__(self):
        return str(self.__str__())

    def __add__(self, other):
        if not isinstance(other, Tensor):
            raise TypeError("input_data must be a tensor")
        out = tensor_operator_registry.get('__add__')(self, other)
        return out

    def __mul__(self, other):
        if not isinstance(other, Tensor):
            raise TypeError("input_data must be a tensor")
        out = tensor_operator_registry.get('__mul__')(self, other)
        return out

    def __iadd__(self, other):
        out = self.__add__(other)
        return out

    def __imul__(self, other):
        out = self.__mul__(other)
        return out

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            other_tensor = Tensor(other, self.dtype())
        elif isinstance(other, Tensor):
            other_tensor = other
        else:
            raise TypeError("unsupported type for div operation")
        out = tensor_operator_registry.get('__div__')(self, other_tensor)
        return out

    def __sub__(self, other):
        if not isinstance(other, Tensor):
            raise TypeError("input_data must be a tensor")
        out = self.__add__(Tensor(-other.asnumpy()))
        return out

    def __isub__(self, other):
        out = self.__sub__(other)
        return out

    def __str__(self):
        if self.dtype() == mstype.type_none:
            return "Unknown Tensor type!"
        return str(self.asnumpy())

    @property
    def virtual_flag(self):
        """Mark tensor is virtual."""
        return self._virtual_flag

    @virtual_flag.setter
    def virtual_flag(self, value):
        """The setter of virtual_flag."""
        if not isinstance(value, bool):
            raise TypeError("virtual_flag must be bool.")
        self._virtual_flag = value
