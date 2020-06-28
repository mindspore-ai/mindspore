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
np_types = (np.int8, np.int16, np.int32, np.int64,
            np.uint8, np.uint16, np.uint32, np.uint64, np.float16,
            np.float32, np.float64, np.bool_)



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
        >>> t1 = Tensor(np.zeros([1, 2, 3]), mindspore.float32)
        >>> assert isinstance(t1, Tensor)
        >>> assert t1.shape == (1, 2, 3)
        >>> assert t1.dtype == mindspore.float32
        >>>
        >>> # init a tensor with a float scalar
        >>> t2 = Tensor(0.1)
        >>> assert isinstance(t2, Tensor)
        >>> assert t2.dtype == mindspore.float64
    """

    def __init__(self, input_data, dtype=None):
        # If input data is numpy number, convert it to np array
        if isinstance(input_data, np_types):
            input_data = np.array(input_data)

        # If input_data is tuple/list/numpy.ndarray, it's support in check_type method.
        check_type('tensor input_data', input_data, (Tensor_, float, int))
        if dtype is not None:
            check_typename('dtype', dtype, mstype.number_type + (mstype.bool_,))
        if isinstance(input_data, np.ndarray) and (not input_data.flags['FORC']):
            input_data = np.ascontiguousarray(input_data)
        if dtype is None:
            Tensor_.__init__(self, input_data)
        else:
            Tensor_.__init__(self, input_data, dtype)
        self._virtual_flag = False
        self._init_flag = False

    def __repr__(self):
        return str(self.__str__())

    def __add__(self, other):
        out = tensor_operator_registry.get('__add__')(self, other)
        return out

    def __eq__(self, other):
        if not isinstance(other, (int, float, Tensor)):
            return False
        #  bool type is not supported for `Equal` operator in backend.
        if self.dtype == mstype.bool_ or (isinstance(other, Tensor) and other.dtype == mstype.bool_):
            return Tensor(np.array(self.asnumpy() == other.asnumpy()))
        return tensor_operator_registry.get('__eq__')(self, other)

    def __ne__(self, other):
        if not isinstance(other, (int, float, Tensor)):
            return True
        #  bool type is not supported for `NotEqual` operator in backend.
        if self.dtype == mstype.bool_ or (isinstance(other, Tensor) and other.dtype == mstype.bool_):
            return Tensor(np.array(self.asnumpy() != other.asnumpy()))
        return tensor_operator_registry.get('__ne__')(self, other)

    def __hash__(self):
        return hash(id(self))

    def __mul__(self, other):
        out = tensor_operator_registry.get('__mul__')(self, other)
        return out

    def __neg__(self):
        out = tensor_operator_registry.get('__neg__')(self)
        return out

    def __iadd__(self, other):
        return self.__add__(other)

    def __radd__(self, other):
        out = tensor_operator_registry.get('__add__')(self, other)
        return out

    def __imul__(self, other):
        return self.__mul__(other)

    def __rmul__(self, other):
        out = tensor_operator_registry.get('__mul__')(self, other)
        return out

    def __truediv__(self, other):
        out = tensor_operator_registry.get('__truediv__')(self, other)
        return out

    def __rtruediv__(self, other):
        out = tensor_operator_registry.get('__truediv__')(other, self)
        return out

    def __sub__(self, other):
        out = tensor_operator_registry.get('__sub__')(self, other)
        return out

    def __isub__(self, other):
        return self.__sub__(other)

    def __rsub__(self, other):
        out = tensor_operator_registry.get('__sub__')(other, self)
        return out

    def __lt__(self, other):
        out = tensor_operator_registry.get('__lt__')(self, other)
        return out

    def __le__(self, other):
        out = tensor_operator_registry.get('__le__')(self, other)
        return out

    def __getitem__(self, index):
        out = tensor_operator_registry.get('__getitem__')(self, index)
        return out

    def __setitem__(self, index, value):
        out = tensor_operator_registry.get('__setitem__')(self, index, value)
        self.assign_value(out)
        return self

    def __gt__(self, other):
        out = tensor_operator_registry.get('__gt__')(self, other)
        return out

    def __ge__(self, other):
        out = tensor_operator_registry.get('__ge__')(self, other)
        return out

    def __len__(self):
        out = tensor_operator_registry.get('shape')(self)
        if not out:
            return 1
        return out[0]

    def __mod__(self, other):
        return tensor_operator_registry.get('__mod__')(self, other)

    def __imod__(self, other):
        return self.__mod__(other)

    def __floordiv__(self, other):
        return tensor_operator_registry.get('__floordiv__')(self, other)

    def __ifloordiv__(self, other):
        return self.__floordiv__(other)

    def __str__(self):
        if self.dtype == mstype.type_none:
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

    @property
    def init_flag(self):
        """whether the tensor is init."""
        return self._init_flag

    @init_flag.setter
    def init_flag(self, value):
        """Set the tensor is init_flag."""
        if not isinstance(value, bool):
            raise TypeError("init_flag must be bool.")
        self.set_init_flag(value)
        self._init_flag = value
