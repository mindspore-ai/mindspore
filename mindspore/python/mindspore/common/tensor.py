# Copyright 2020-2022 Huawei Technologies Co., Ltd
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

__all__ = ['Tensor']

import abc
import math
import numbers
import numpy as np

from mindspore.communication.management import get_group_size
from mindspore.common._utils import is_shape_unknown
from mindspore.common.seed import get_seed
from mindspore import context
from mindspore import log as logger
from mindspore.common import dtype as mstype

from mindspore.common._utils import get_slice_num
from mindspore.common._register_for_tensor import tensor_operator_registry
from mindspore._c_expression import Tensor as Tensor_
from mindspore._checkparam import Rel, check_is_number, is_stub_tensor
from mindspore._checkparam import Validator as validator
from mindspore._check_jit_forbidden_api import jit_forbidden_register

np_types = (np.int8, np.int16, np.int32, np.int64,
            np.uint8, np.uint16, np.uint32, np.uint64, np.float16,
            np.float32, np.float64, np.bool_, np.complex64, np.complex128)


def _check_input_data_type(input_data):
    """Check the type of input_data for Tensor"""
    validator.check_value_type('input_data', input_data,
                               (Tensor_, Tensor, np.ndarray, np.str_, list, tuple, float, int, bool, complex),
                               'Tensor')
    valid_dtypes = (np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64,
                    np.float16, np.float32, np.float64, np.bool_, np.str_, np.complex64, np.complex128)
    if isinstance(input_data, np.ndarray) and input_data.dtype not in valid_dtypes and \
            input_data.dtype.kind != 'U' and input_data.dtype.kind != 'S':  # Support dtype np.str_
        new_line = '\n'
        for index, x in np.ndenumerate(input_data):
            if np.array(x).dtype not in valid_dtypes:
                raise TypeError(f"initializing tensor by numpy array failed, because the "
                                f"element type '{type(x)}' of array is not supported.\n"
                                f"The element index in array: {index}, numpy array: {input_data}.\n"
                                f"The supported element type of ndarray as follow: "
                                f"{new_line}{new_line.join(map(str, valid_dtypes))}")
        raise TypeError(f"initializing tensor by numpy array failed, numpy array: {input_data}, "
                        f"data type: {input_data.dtype}.\nThe supported element type of ndarray "
                        f"as follow: {new_line}{new_line.join(map(str, valid_dtypes))}")
    if isinstance(input_data, np.ndarray) and input_data.dtype.kind == "S" and \
            input_data.shape and context.get_context("enable_ge"):
        raise TypeError("For binary string input in GE mode, the shape of the data must be ()")
    if isinstance(input_data, (tuple, list)) and np.array(input_data).dtype not in valid_dtypes:
        raise TypeError(
            f"For Tensor, the input_data is {input_data} that contain unsupported element.")


class _TensorMeta(type(Tensor_), abc.ABCMeta):
    """
    Meta class for Tensor. Used internally.
    """


class Tensor(Tensor_, metaclass=_TensorMeta):
    """
    Tensor is a data structure that stores an n-dimensional array.

    Args:
        input_data (Union[Tensor, float, int, bool, tuple, list, numpy.ndarray]): The data to be stored. It can be
            another Tensor, Python number or NumPy ndarray. Default: None.
        dtype (:class:`mindspore.dtype`): Used to indicate the data type of the output Tensor. The argument should
            be defined in `mindspore.dtype`. If it is None, the data type of the output Tensor will be the same
            as the `input_data`. Default: None.
        shape (Union[tuple, list, int]): Used to indicate the shape of the output Tensor. The argument should be
            a list of integers, a tuple of integers or an integer. If `input_data` is available,
            `shape` doesn't need to be set. If None in shape, a tensor of dynamic shape is created, `input_data`
            doesn't need to be set; if None not in shape, a tensor of static shape is created, `input_data` or `init`
            must be set. Default: None.
        init (Initializer): The information of init data.
            'init' is used for delayed initialization in parallel mode. Usually, it is not recommended to use
            'init' interface to initialize Tensor in the other conditions. If 'init' interface is used to initialize
            Tensor, the `Tensor.init_data` API needs to be called to convert `Tensor` to the actual data.
            Default: None.
        internal (bool): Whether it is created by the framework.
            'True' means that the tensor is created by framework.
            'False' means that the tensor is created by user.
            Default: False
        const_arg (bool): Whether the tensor is a constant when it is used for the argument of a network.
            Default: False.

    Outputs:
        Tensor.

    Note:
        The default value None of `input_data` works as a placeholder, it does not mean that we can create a NoneType
        Tensor.

    Examples:
        >>> import numpy as np
        >>> import mindspore as ms
        >>> from mindspore import Tensor
        >>> from mindspore.common.initializer import One
        >>> # initialize a tensor with numpy.ndarray
        >>> t1 = Tensor(np.zeros([1, 2, 3]), ms.float32)
        >>> print(t1)
        [[[0. 0. 0.]
        [0. 0. 0.]]]
        >>> print(type(t1))
        <class 'mindspore.common.tensor.Tensor'>
        >>> print(t1.shape)
        (1, 2, 3)
        >>> print(t1.dtype)
        Float32
        >>>
        >>> # initialize a tensor with a float scalar
        >>> t2 = Tensor(0.1)
        >>> print(t2)
        0.1
        >>> print(type(t2))
        <class 'mindspore.common.tensor.Tensor'>
        >>> print(t2.shape)
        ()
        >>> print(t2.dtype)
        Float32
        >>>
        >>> # initialize a tensor with a tuple
        >>> t3 = Tensor((1, 2))
        >>> print(t3)
        [1 2]
        >>> print(type(t3))
        <class 'mindspore.common.tensor.Tensor'>
        >>> print(t3.shape)
        (2,)
        >>> print(t3.dtype)
        Int64
        ...
        >>> # initialize a tensor with init
        >>> t4 = Tensor(shape = (1, 3), dtype=ms.float32, init=One())
        >>> print(t4)
        [[1. 1. 1.]]
        >>> print(type(t4))
        <class 'mindspore.common.tensor.Tensor'>
        >>> print(t4.shape)
        (1, 3)
        >>> print(t4.dtype)
        Float32
    """
    delta_seed = 0

    def __init__(self, input_data=None, dtype=None, shape=None, init=None, internal=False, const_arg=False):
        self.init_finished = False

        if internal:
            if input_data is not None:
                Tensor_.__init__(self, input_data)
        else:
            if is_stub_tensor(input_data):
                input_data = input_data.stub_sync()

            # If input data is numpy number, convert it to np array
            if isinstance(input_data, np_types):
                input_data = np.array(input_data)

            if isinstance(shape, numbers.Number):
                shape = (shape,)

            _check_tensor_input(input_data, dtype, shape, init)

            # If input_data is tuple/list/numpy.ndarray, it's support in check_type method.
            if (isinstance(shape, (list, tuple)) and None in shape) or init is not None:
                shape = _check_tensor_dynamic_shape(dtype, shape, init)
                Tensor_.__init__(self, dtype, shape)
            else:
                _check_input_data_type(input_data)
                if dtype is not None:
                    validator.check_type_name('dtype', dtype, mstype.number_type +
                                              (mstype.bool_, mstype.string), "Tensor")
                else:
                    dtype = self._set_default_dtype(input_data, dtype)

                if isinstance(input_data, np.ndarray) and (not input_data.flags['FORC']):
                    input_data = np.ascontiguousarray(input_data)

                if dtype is not None:
                    Tensor_.__init__(self, input_data, dtype)
                else:
                    Tensor_.__init__(self, input_data)
                validator.check_value_type('const_arg', const_arg, bool, 'Tensor')

        self.const_arg = const_arg
        self.virtual_flag = False
        self.init = init
        self.init_finished = True

        # if cur Tensor is a index value of another Tensor,
        # parent_tensor_ set to another Tensor
        # index_of_parent_ will set to the index
        self.parent_tensor_ = None
        self.index_of_parent_ = None

        self.slice_num_of_persistent_data_ = None
        self.slice_shape_of_persistent_data_ = None

    @classmethod
    def __subclasshook__(cls, sub):
        """
        Subclass with stub_sync attr will be instance of Tensor
        """
        if cls is Tensor:
            if any("stub_sync" in s.__dict__ for s in sub.__mro__):
                return True
        return NotImplemented

    @staticmethod
    def _set_default_dtype(input_data, dtype):
        """Set tensor default dtype"""
        if isinstance(input_data, (float, list, tuple)):
            if np.array(input_data).dtype == np.float64:
                return mstype.float32
        if isinstance(input_data, (int, list, tuple)):
            if np.array(input_data).dtype == np.int32 or np.array(input_data).dtype == np.int64:
                return mstype.int64
        return dtype

    def __deepcopy__(self, memodict):
        new_obj = Tensor(self)
        new_obj.init = self.init
        new_obj.virtual_flag = self.virtual_flag
        new_obj.const_arg = self.const_arg
        return new_obj

    def __repr__(self):
        if self.init_finished:
            Tensor_.data_sync(self, True)
            return Tensor_.__repr__(self)
        return ''

    def __eq__(self, other):
        if not isinstance(other, (int, float, Tensor)):
            return False
        # bool type is not supported for `Equal` operator in backend.
        if self.dtype == mstype.bool_ or (isinstance(other, Tensor) and other.dtype == mstype.bool_):
            if isinstance(other, Tensor):
                return Tensor(np.array(self.asnumpy() == other.asnumpy()))
            return Tensor(np.array(self.asnumpy() == other))
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

    def __neg__(self):
        out = tensor_operator_registry.get('__neg__')(self)
        return out

    def __invert__(self):
        out = tensor_operator_registry.get('__logical_not__')(self)
        return out

    def __round__(self):
        out = tensor_operator_registry.get('round')()(self)
        return out

    def __bool__(self):
        data = self.asnumpy()
        if data.shape == ():
            return bool(data)
        if data.shape == (1,):
            return bool(data[0])
        raise ValueError("The truth value of an array with more than one element is ambiguous.")

    @staticmethod
    def _convert_scalar_(data, func, message):
        if data.shape == ():
            return func(data)
        if data.shape == (1,):
            return func(data[0])
        raise ValueError(message)

    def __int__(self):
        data = self.asnumpy()
        return self._convert_scalar_(data, int, "Only one element tensors can be converted to Python scalars")

    def __float__(self):
        data = self.asnumpy()
        return self._convert_scalar_(data, float, "Only one element tensors can be converted to Python scalars")

    def __index__(self):
        data = self.asnumpy()
        if not (data.dtype == "int8"
                or data.dtype == "int16"
                or data.dtype == "int32"
                or data.dtype == "int64"
                or data.dtype == "bool"):
            raise ValueError("Only integer tensors of a single element can be converted to an index.")
        return self._convert_scalar_(data, int,
                                     "Only integer tensors of a single element can be converted to an index.")

    def __pos__(self):
        return self

    def __abs__(self):
        data = abs(self.asnumpy())
        if isinstance(data, np.number):
            data = np.array(data)
        return Tensor(data)

    def __add__(self, other):
        return tensor_operator_registry.get('__add__')(self, other)

    def __and__(self, other):
        if isinstance(other, (int, bool, float, Tensor)):
            return tensor_operator_registry.get('bitwise_and')(self, other)
        raise TypeError("Unsupported operand type(s) for &: 'Tensor' and '{}'".format(type(other)))

    def __xor__(self, other):
        if isinstance(other, (int, bool, float, Tensor)):
            return tensor_operator_registry.get('bitwise_xor')(self, other)
        raise TypeError("Unsupported operand type(s) for ^: 'Tensor' and '{}'".format(type(other)))

    def __or__(self, other):
        if isinstance(other, (int, bool, float, Tensor)):
            return tensor_operator_registry.get('bitwise_or')(self, other)
        raise TypeError("Unsupported operand type(s) for |: 'Tensor' and '{}'".format(type(other)))

    def __radd__(self, other):
        return self.__add__(other)

    def __iadd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return tensor_operator_registry.get('__sub__')(self, other)

    def __rsub__(self, other):
        return tensor_operator_registry.get('__sub__')(other, self)

    def __isub__(self, other):
        return self.__sub__(other)

    def __mul__(self, other):
        return tensor_operator_registry.get('__mul__')(self, other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __imul__(self, other):
        return self.__mul__(other)

    def __matmul__(self, other):
        return tensor_operator_registry.get('__matmul__')(self, other)

    def __rmatmul__(self, other):
        return tensor_operator_registry.get('__matmul__')(other, self)

    def __imatmul__(self, other):
        return self.__matmul__(other)

    def __truediv__(self, other):
        return tensor_operator_registry.get('__truediv__')(self, other)

    def __rtruediv__(self, other):
        return tensor_operator_registry.get('__truediv__')(other, self)

    def __mod__(self, other):
        return tensor_operator_registry.get('__mod__')(self, other)

    def __rmod__(self, other):
        return tensor_operator_registry.get('__mod__')(other, self)

    def __imod__(self, other):
        return self.__mod__(other)

    def __pow__(self, other):
        return tensor_operator_registry.get('__pow__')(self, other)

    def __rpow__(self, other):
        return tensor_operator_registry.get('__rpow__')(self, other)

    def __floordiv__(self, other):
        return tensor_operator_registry.get('__floordiv__')(self, other)

    def __rfloordiv__(self, other):
        return tensor_operator_registry.get('__floordiv__')(other, self)

    def __ifloordiv__(self, other):
        return self.__floordiv__(other)

    def __lt__(self, other):
        out = tensor_operator_registry.get('__lt__')(self, other)
        return out

    def __le__(self, other):
        out = tensor_operator_registry.get('__le__')(self, other)
        return out

    def __getitem__(self, index):
        out = tensor_operator_registry.get('__getitem__')(self, index)
        if out is not self:
            out.parent_tensor_ = self
            out.index_of_parent_ = index
        return out

    def __setitem__(self, index, value):
        out = tensor_operator_registry.get('__setitem__')(self, index, value)
        self.assign_value(out)
        if self.parent_tensor_ is not None and self.index_of_parent_ is not None:
            self.parent_tensor_.__setitem__(self.index_of_parent_, self)
        return self

    def __gt__(self, other):
        out = tensor_operator_registry.get('__gt__')(self, other)
        return out

    def __ge__(self, other):
        out = tensor_operator_registry.get('__ge__')(self, other)
        return out

    def __len__(self):
        out = tensor_operator_registry.get('shape')(self)
        if out:
            return out[0]
        raise TypeError("Not support len of a 0-D tensor")

    def __str__(self):
        if self.dtype == mstype.type_none:
            return "Unknown Tensor type!"
        return str(self.asnumpy())

    @property
    def shape(self):
        """
        For details, please refer to :func:`mindspore.ops.shape`.
        """
        return self._shape

    @property
    def dtype(self):
        """Return the dtype of the tensor (:class:`mindspore.dtype`)."""
        return self._dtype

    @property
    def size(self):
        """
        For details, please refer to :func:`mindspore.ops.size`.
        """
        return self._size

    @property
    def ndim(self):
        """Return the number of tensor dimensions."""
        return len(self._shape)

    @property
    def H(self):
        """
        Returns a view of a matrix (2-D tensor) conjugated and transposed.
        x.H is equivalent to x.swapaxes(0, 1).conj() for complex matrices and x.swapaxes(0, 1) for real matrices.
        """
        if self.ndim != 2:
            raise ValueError(f"For tensor.H only support 2-D Tensor, but got {self.ndim}-D.")
        output = self.swapaxes(0, 1)
        if self.dtype in (mstype.complex64, mstype.complex128):
            return output.conj()
        return output

    @property
    def has_init(self):
        """Whether tensor is initialized."""
        return self.init is not None

    @property
    def itemsize(self):
        """Return the length of one tensor element in bytes."""
        return self._itemsize

    @property
    def strides(self):
        """Return the tuple of bytes to step in each dimension when traversing a tensor."""
        return self._strides

    @property
    def nbytes(self):
        """Return the total number of bytes taken by the tensor."""
        return self._nbytes

    @property
    def T(self):
        """Return the transposed tensor."""
        return self.transpose()

    @staticmethod
    def from_numpy(array):
        """
        Convert numpy array to Tensor.
        If the data is not C contiguous, the data will be copied to C contiguous to construct the tensor.
        Otherwise, The tensor will be constructed using this numpy array without copy.

        Args:
            array (numpy.array): The input array.

        Returns:
            Tensor, has the same data type as input array.

        Examples:
            >>> import numpy as np
            >>> from mindspore import Tensor
            >>> x = np.array([1, 2])
            >>> output = Tensor.from_numpy(x)
            >>> print(output)
            [1 2]
        """
        if isinstance(array, np.ndarray) and not array.flags['C_CONTIGUOUS']:
            array = np.ascontiguousarray(array)

        return Tensor(Tensor_.from_numpy(array))

    def ndimension(self):
        r"""
        Alias for :func:`mindspore.Tensor.ndim`.
        """
        return len(self._shape)

    @jit_forbidden_register
    def set_const_arg(self, const_arg=True):
        """
        Specify whether the tensor is a constant when it is used for the argument of a network.

        Args:
            const_arg (bool): Whether the tensor is a constant when it is used for the argument of a network.
                Default: True.

        Returns:
            Tensor, has been specified whether to be a const network argument.

        Raises:
            TypeError: If `const_arg` is not a bool.

        Supported Platforms:
            ``Ascend`` ``GPU`` ``CPU``

        Examples:
            >>> import numpy as np
            >>> from mindspore import Tensor
            >>> x = Tensor(np.array([[1,2,3],[4,5,6]], dtype=np.float32))
            >>> x.set_const_arg(True)
        """
        validator.check_value_type('const_arg', const_arg, bool, 'set_const_arg')
        self.const_arg = const_arg
        return self

    def arccosh(self):
        r"""
        For details, please refer to :func:`mindspore.ops.arccosh`.
        """
        self._init_check()
        return tensor_operator_registry.get('acosh')(self)

    def arcsin(self):
        r"""
        For details, please refer to :func:`mindspore.ops.arcsin`.
        """
        self._init_check()
        return tensor_operator_registry.get('asin')(self)

    def arctan(self):
        r"""
        For details, please refer to :func:`mindspore.ops.arctan`.
        """
        self._init_check()
        return tensor_operator_registry.get('atan')(self)

    def arctan2(self, other):
        r"""
        For details, please refer to :func:`mindspore.ops.arctan2`.
        """
        self._init_check()
        return tensor_operator_registry.get('atan2')(self, other)

    def assign_value(self, value):
        """
        Assign another tensor value to this tensor.

        Args:
            value (Tensor): Tensor for assignment.

        Returns:
            Tensor, Tensor that's been assigned.
        """
        if is_stub_tensor(value):
            value = value.stub_sync()
        self.assign_value_cpp(value)
        return self

    def bincount(self, weights=None, minlength=0):
        r"""
        For details, please refer to :func:`mindspore.ops.bincount`.
        """
        self._init_check()
        return tensor_operator_registry.get('bincount')(self, weights, minlength)

    def chunk(self, chunks, axis=0):
        r"""
        For details, please refer to :func:`mindspore.ops.chunk`.
        """
        self._init_check()
        return tensor_operator_registry.get('chunk')(self, chunks, axis)

    def item(self, index=None):
        """
        Get the item at the specified index of the tensor.

        Note:
            Tensor.item returns a Tensor scalar instead of a Python scalar.

        Args:
            index (Union[None, int, tuple(int)]): The index in Tensor. Default: None.

        Returns:
            A Tensor scalar, dtype is the same with the original Tensor.

        Raises:
            ValueError: If the length of the `index` is not equal to self.ndim.

        Supported Platforms:
            ``Ascend`` ``GPU`` ``CPU``

        Examples:
            >>> import numpy as np
            >>> from mindspore import Tensor
            >>> x = Tensor(np.array([[1,2,3],[4,5,6]], dtype=np.float32))
            >>> x = x.item((0,1))
            >>> print(x)
            2.0
        """
        output = tensor_operator_registry.get('item')(self, index)
        return output

    def itemset(self, *args):
        r"""
        Insert scalar into a tensor (scalar is cast to tensor's dtype, if possible).

        There must be at least 1 argument, and define the last argument as item.
        Then, tensor.itemset(\*args) is equivalent to :math:`tensor[args] = item`.

        Args:
            args (Union[(numbers.Number), (int/tuple(int), numbers.Number)]): The arguments that
                specify the index and value. If `args` contain one argument (a scalar),
                it is only used in case tensor is of size 1. If `args` contain two
                arguments, the last argument is the value to be set and must be a
                scalar, the first argument specifies a single tensor element location.
                It is either an int or a tuple.

        Returns:
            A new tensor that doesn't affect the original tensor, with value set by :math:`tensor[args] = item`.

        Raises:
            ValueError: If the length of the first argument is not equal to self.ndim.
            IndexError: If only one argument is provided, and the original Tensor is not scalar.

        Supported Platforms:
            ``Ascend`` ``GPU`` ``CPU``

        Examples:
            >>> import numpy as np
            >>> from mindspore import Tensor
            >>> x = Tensor(np.array([[1,2,3],[4,5,6]], dtype=np.float32))
            >>> print(x.itemset((0,1), 4))
            [[1. 4. 3.]
            [4. 5. 6.]]
            >>> print(x)
            [[1. 2. 3.]
            [4. 5. 6.]]
        """
        output = tensor_operator_registry.get('itemset')(self, *args)
        return output

    def asnumpy(self):
        """
        Convert tensor to numpy array. Returns self tensor as a NumPy ndarray. This tensor and the returned ndarray
        share the same underlying storage. Changes to self tensor will be reflected in the ndarray.

        Returns:
            A numpy ndarray which shares the same underlying storage with the tensor.

        Examples:
            >>> from mindspore import Tensor
            >>> import numpy as np
            >>> x = Tensor(np.array([1, 2], dtype=np.float32))
            >>> y = x.asnumpy()
            >>> y[0] = 11
            >>> print(x)
            [11.  2.]
            >>> print(y)
            [11.  2.]
        """
        self._init_check()
        return Tensor_.asnumpy(self)

    def numpy(self):
        """
        Alias for :func:`mindspore.Tensor.asnumpy`.
        """
        return self.asnumpy()

    def is_persistent_data(self):
        """
        Check if size of tensor is huge, and need save data to persistent storage.
        If size of tensor is bigger then MS_EMBEDDING_REMOTE_CACHE_MEMORY_SIZE, it will
        use persistent storage to save tensor data. And will spilt data to some slice.

        Returns:
            True or False
        """
        return Tensor_.is_persistent_data(self)

    def asnumpy_of_slice_persistent_data(self, param_key, slice_index):
        """
        Convert a slice of tensor data to numpy array. A slice is part of tensor data.
        Returns as a NumPy ndarray. This slice tensor data and the returned ndarray
        share the same underlying storage. Changes to self tensor will be reflected in the ndarray.

        Returns:
            A numpy ndarray which shares the same underlying storage with the slice of tensor data.
        """
        return Tensor_.asnumpy_of_slice_persistent_data(self, param_key, slice_index)

    def slice_num_of_persistent_data(self):
        """
        Get slice num of a tensor which use persistent storage.

        Returns:
            Num of slice.
        """
        return self.slice_num_of_persistent_data_

    def slice_shape_of_persistent_data(self):
        """
        Get slice shape of tensor after cut to slice size.

        Returns:
            The slice shape of tensor.
        """
        return self.slice_shape_of_persistent_data_

    def value(self):
        """
        Get the value of the tensor or the parameter.

        Returns:
            The value of the tensor or the parameter.

        Examples:
            >>> from mindspore import Tensor
            >>> import numpy as np
            >>> x = Tensor(np.array([1, 2], dtype=np.float32))
            >>> x_value = x.value()
            >>> print(x_value)
            [1.  2.]
        """
        return self

    def flush_from_cache(self):
        """
        Flush cache data to host if tensor is cache enable.

        Examples:
            >>> from mindspore import Tensor
            >>> import numpy as np
            >>> x = Tensor(np.array([1, 2], dtype=np.float32))
            >>> y = x.flush_from_cache()
            >>> print(y)
            None
        """
        self._init_check()
        Tensor_._flush_from_cache(self)

    def addcdiv(self, tensor1, tensor2, value=1):
        r"""
        For details, please refer to :func:`mindspore.ops.addcdiv`.
        """
        self._init_check()
        return tensor_operator_registry.get('addcdiv')()(self, tensor1, tensor2, value)

    def addcmul(self, tensor1, tensor2, value=1):
        r"""
        For details, please refer to :func:`mindspore.ops.addcmul`.
        """
        self._init_check()
        return tensor_operator_registry.get('addcmul')()(self, tensor1, tensor2, value)

    def add(self, other):
        r"""
        For details, please refer to :func:`mindspore.ops.add`.
        """
        self._init_check()
        return tensor_operator_registry.get('add')()(self, other)

    def subtract(self, other, *, alpha=1):
        r"""
        For details, please refer to :func:`mindspore.ops.subtract`.
        """
        self._init_check()
        return tensor_operator_registry.get('sub')(self, alpha * other)

    def true_divide(self, value):
        r"""
        Alias for Tensor.div() with :math:`rounding\_mode=None`.
        For details, please refer to :func:`mindspore.ops.div`.
        """
        self._init_check()
        return tensor_operator_registry.get('div')(self, value, rounding_mode=None)

    def triu(self, diagonal=0):
        r"""
        Returns a triangular matrix based on the diagonal. Default is the main diagonal.

        Args:
            diagonal (int): The index of diagonal. Default: 0.

        Returns:
            Tensor, a tensor has the same shape and data type as input.

        Raises:
            TypeError: If `diagonal` is not an int.
            TypeError: If `x` is not an Tensor.
            ValueError: If length of shape of x is less than 1.

        Supported Platforms:
            ``GPU`` ``CPU``
        """
        self._init_check()
        validator.check_value_type('diagonal', diagonal, [int], 'triu')
        return tensor_operator_registry.get('triu')(diagonal)(self)

    def addbmm(self, batch1, batch2, *, beta=1, alpha=1):
        r"""
        For details, please refer to :func:`mindspore.ops.addbmm`.
        """
        self._init_check()
        return tensor_operator_registry.get('addbmm')(self, batch1, batch2, beta=beta, alpha=alpha)

    def addmm(self, mat1, mat2, *, beta=1, alpha=1):
        r"""
        For details, please refer to :func:`mindspore.ops.addmm`.
        """
        self._init_check()
        return tensor_operator_registry.get('addmm')(self, mat1, mat2, beta=beta, alpha=alpha)

    def addr(self, vec1, vec2, beta=1, alpha=1):
        r"""
        For details, please refer to :func:`mindspore.ops.addr`.
        """
        self._init_check()
        return tensor_operator_registry.get('addr')(self, vec1, vec2, beta=beta, alpha=alpha)

    def adjoint(self):
        r"""
        For details, please refer to :func:`mindspore.ops.adjoint`.
        """
        self._init_check()
        return tensor_operator_registry.get('adjoint')(self)

    def all(self, axis=None, keep_dims=False):
        r"""
        For details, please refer to :func:`mindspore.ops.all`.
        """
        self._init_check()
        if axis is None:
            axis = ()
        return tensor_operator_registry.get('all')(keep_dims)(self, axis)

    def angle(self):
        r"""
        For details, please refer to :func:`mindspore.ops.angle`.
        """
        self._init_check()
        return tensor_operator_registry.get('angle')(self)

    def any(self, axis=None, keep_dims=False):
        r"""
        For details, please refer to :func:`mindspore.ops.any`.
        """
        self._init_check()
        if axis is None:
            axis = ()
        return tensor_operator_registry.get('any')(keep_dims)(self, axis)

    def atan2(self, other):
        r"""
        For details, please refer to :func:`mindspore.ops.atan2`.
        """
        self._init_check()
        return tensor_operator_registry.get('atan2')(self, other)

    def baddbmm(self, batch1, batch2, beta=1, alpha=1):
        r"""
        For details, please refer to :func:`mindspore.ops.baddbmm`.
        """
        self._init_check()
        return tensor_operator_registry.get('baddbmm')(self, batch1, batch2, beta=beta, alpha=alpha)

    def view(self, *shape):
        """
        Reshape the tensor according to the input shape. It's the same as :func:`mindspore.Tensor.reshape`,
        implemented by the underlying reshape operator.

        Args:
            shape (Union[tuple(int), int]): Dimension of the output tensor.

        Returns:
            Tensor, which dimension is the input shape's value.

        Examples:
            >>> from mindspore import Tensor
            >>> import numpy as np
            >>> a = Tensor(np.array([[1, 2, 3], [2, 3, 4]], dtype=np.float32))
            >>> output = a.view((3, 2))
            >>> print(output)
            [[1. 2.]
            [3. 2.]
            [3. 4.]]
        """
        self._init_check()
        if not shape:
            raise ValueError("The shape variable should not be empty")
        if isinstance(shape[0], tuple):
            if len(shape) != 1:
                raise ValueError(f"Only one tuple is needed, but got {shape}")
            shape = shape[0]
        return tensor_operator_registry.get('reshape')()(self, shape)

    def view_as(self, other):
        r"""
        View self Tensor as the same shape as `other` .

        Args:
            other(Tensor): The returned Tensor has the same shape as `other`.

        Returns:
            Tensor, has the same shape as `other`.

        Raises:
            TypeError: If `other` is not a Tensor.

        Supported Platforms:
            ``Ascend`` ``GPU`` ``CPU``

        Examples:
            >>> a = Tensor([[1, 2, 3], [2, 3, 4]], mstype.float32)
            >>> b = Tensor([1, 1, 1, 1, 1, 1], mstype.float32)
            >>> output = a.view_as(b)
            >>> print(output)
            [1. 2. 3. 2. 3. 4.]
        """
        self._init_check()
        if not isinstance(other, (Tensor, Tensor_)):
            raise TypeError(f"For view_as, the input other must be a Tensor, but got {type(other)}")
        return self.view(other.shape)

    def t(self):
        r"""
        For details, please refer to :func:`mindspore.ops.t`.
        """
        self._init_check()
        return tensor_operator_registry.get("t")(self)

    def bitwise_and(self, other):
        """
        For details, please refer to :func:`mindspore.ops.bitwise_and`.
        """
        self._init_check()
        return tensor_operator_registry.get('bitwise_and')(self, other)

    def bitwise_or(self, other):
        """
        For details, please refer to :func:`mindspore.ops.bitwise_or`.
        """
        self._init_check()
        return tensor_operator_registry.get('bitwise_or')(self, other)

    def bitwise_xor(self, other):
        """
        For details, please refer to :func:`mindspore.ops.bitwise_xor`.
        """
        self._init_check()
        return tensor_operator_registry.get('bitwise_xor')(self, other)

    def bitwise_left_shift(self, other):
        """
        For details, please refer to :func:`mindspore.ops.bitwise_left_shift`.
        """
        self._init_check()
        return tensor_operator_registry.get('bitwise_left_shift')(self, other)

    def bitwise_right_shift(self, other):
        """
        For details, please refer to :func:`mindspore.ops.bitwise_right_shift`.
        """
        self._init_check()
        _cast = tensor_operator_registry.get('cast')
        other = _cast(other, self.dtype)
        return tensor_operator_registry.get('bitwise_right_shift')(self, other)

    def scatter(self, axis, index, src):
        """
        For details, please refer to :func:`mindspore.ops.scatter`.
        """
        self._init_check()
        return tensor_operator_registry.get('scatter')(self, axis, index, src)

    def scatter_mul(self, indices, updates):
        """
        For details, please refer to :func:`mindspore.ops.scatter_mul`.
        """
        self._init_check()
        return tensor_operator_registry.get('tensor_scatter_mul')(self, indices, updates)

    def scatter_div(self, indices, updates):
        """
        For details, please refer to :func:`mindspore.ops.scatter_div`.
        """
        self._init_check()
        return tensor_operator_registry.get('tensor_scatter_div')(self, indices, updates)

    def ger(self, x):
        """
        For details, please refer to :func:`mindspore.ops.ger`.
        """
        self._init_check()
        return tensor_operator_registry.get('ger')(self, x)

    def gt(self, x):
        """
        For details, please refer to :func:`mindspore.ops.gt`.
        """
        self._init_check()
        return tensor_operator_registry.get('gt')()(self, x)

    def ge(self, x):
        """
        For details, please refer to :func:`mindspore.ops.ge`.
        """
        self._init_check()
        return tensor_operator_registry.get('ge')()(self, x)

    def broadcast_to(self, shape):
        """
        For details, please refer to :func:`mindspore.ops.broadcast_to`.
        """
        self._init_check()
        return tensor_operator_registry.get('broadcast_to')(shape)(self)

    def expand_as(self, x):
        """
        Expand the dimension of target tensor to the dimension of input tensor.

        Args:
            x (Tensor): The input tensor. The shape of the input tensor must obey
                the broadcasting rule.

        Returns:
            Tensor, has the same dimension as input tensor.

        Examples:
            >>> import numpy as np
            >>> from mindspore import Tensor
            >>> from mindspore import dtype as mstype
            >>> x = Tensor([1, 2, 3], dtype=mstype.float32)
            >>> y = Tensor(np.ones((2, 3)), dtype=mstype.float32)
            >>> output = x.expand_as(y)
            >>> print(output)
            [[1. 2. 3.]
            [1. 2. 3.]]
        """
        self._init_check()
        return tensor_operator_registry.get('broadcast_to')(x.shape)(self)

    def exp(self):
        """
        For details, please refer to :func:`mindspore.ops.exp`.
        """
        self._init_check()
        return tensor_operator_registry.get('exp')()(self)

    def real(self):
        r"""
        For details, please refer to :func:`mindspore.ops.real`.
        """
        self._init_check()
        return tensor_operator_registry.get('real')(self)

    def rsqrt(self):
        r"""
        For details, please refer to :func:`mindspore.ops.rsqrt`.
        """
        self._init_check()
        return tensor_operator_registry.get('rsqrt')(self)

    def reciprocal(self):
        r"""
        For details, please refer to :func:`mindspore.ops.reciprocal`.
        """
        self._init_check()
        return tensor_operator_registry.get('reciprocal')(self)

    def sqrt(self):
        """
        For details, please refer to :func:`mindspore.ops.sqrt`.
        """
        self._init_check()
        return tensor_operator_registry.get('sqrt')(self)

    def square(self):
        """
        For details, please refer to :func:`mindspore.ops.square`.
        """
        self._init_check()
        return tensor_operator_registry.get('square')(self)

    def sub(self, y):
        r"""
        For details, please refer to :func:`mindspore.ops.sub`.
        """
        self._init_check()
        return tensor_operator_registry.get('sub')(self, y)

    def tan(self):
        """
        For details, please refer to :func:`mindspore.ops.tan`.
        """
        self._init_check()
        return tensor_operator_registry.get('tan')()(self)

    def tanh(self):
        r"""
        For details, please refer to :func:`mindspore.ops.tanh`.
        """
        self._init_check()
        return tensor_operator_registry.get('tanh')(self)

    def cosh(self):
        r"""
        For details, please refer to :func:`mindspore.ops.cosh`.
        """
        self._init_check()
        return tensor_operator_registry.get('cosh')()(self)

    def acos(self):
        r"""
        For details, please refer to :func:`mindspore.ops.acos`.
        """
        self._init_check()
        return tensor_operator_registry.get('acos')(self)

    def arccos(self):
        r"""
        Alias for :func:`mindspore.Tensor.acos`.
        """
        return self.acos()

    def cos(self):
        r"""
        For details, please refer to :func:`mindspore.ops.cos`.
        """
        self._init_check()
        return tensor_operator_registry.get('cos')(self)

    def cov(self, *, correction=1, fweights=None, aweights=None):
        r"""
        For details, please refer to :func:`mindspore.ops.cov`.
        """
        self._init_check()
        return tensor_operator_registry.get('cov')(self, correction=correction, fweights=fweights, aweights=aweights)

    def acosh(self):
        """
        For details, please refer to :func:`mindspore.ops.acosh`.
        """
        self._init_check()
        return tensor_operator_registry.get('acosh')(self)

    def asin(self):
        r"""
        For details, please refer to :func:`mindspore.ops.asin`.
        """
        self._init_check()
        return tensor_operator_registry.get('asin')(self)

    def abs(self):
        """
        For details, please refer to :func:`mindspore.ops.abs`.
        """
        self._init_check()
        return tensor_operator_registry.get('abs')()(self)

    def absolute(self):
        """
        Alias for :func:`mindspore.Tensor.abs`.
        """
        return self.abs()

    def ceil(self):
        """
        For details, please refer to :func:`mindspore.ops.ceil`.
        """
        self._init_check()
        return tensor_operator_registry.get('ceil')()(self)

    def floor(self):
        """
        For details, please refer to :func:`mindspore.ops.floor`.
        """
        self._init_check()
        return tensor_operator_registry.get('floor')(self)

    def lerp(self, end, weight):
        """
        For details, please refer to :func:`mindspore.ops.lerp`.
        """
        self._init_check()
        return tensor_operator_registry.get('lerp')(self, end, weight)

    def negative(self):
        r"""
        For details, please refer to :func:`mindspore.ops.negative`.
        """
        self._init_check()
        return tensor_operator_registry.get("negative")(self)

    # pylint: disable=redefined-builtin
    # pylint: disable=invalid-name
    def norm(self, ord=None, dim=None, keepdim=False, *, dtype=None):
        """
        For details, please refer to :func:`mindspore.ops.norm`.
        """
        self._init_check()
        return tensor_operator_registry.get('norm')(self, ord, dim, keepdim, dtype=dtype)

    def renorm(self, p, axis, maxnorm):
        """
        For details, please refer to :func:`mindspore.ops.renorm`.
        """
        self._init_check()
        return tensor_operator_registry.get("renorm")(self, p, axis, maxnorm)

    def approximate_equal(self, other, tolerance=1e-5):
        r"""
        For details, please refer to :func:`mindspore.ops.approximate_equal`.
        """
        validator.check_isinstance("x", self, Tensor)
        validator.check_isinstance("y", other, Tensor)
        validator.check_isinstance("tolerance", tolerance, float)
        self._init_check()
        input_x = self.copy() if self.dtype == mstype.float32 else self.astype(mstype.float16)
        input_y = other.copy() if other.dtype == mstype.float32 else other.astype(mstype.float16)
        return tensor_operator_registry.get('__lt__')(tensor_operator_registry.get('abs')()(
            tensor_operator_registry.get('__sub__')(input_x, input_y)
        ), tolerance)

    def log1p(self):
        r"""
        For details, please refer to :func:`mindspore.ops.log1p`.
        """
        self._init_check()
        return tensor_operator_registry.get('log1p')(self)

    def logit(self, eps=None):
        r"""
        For details, please refer to :func:`mindspore.ops.logit`.
        """
        self._init_check()
        if eps is None:
            eps = -1.0
        validator.check_value_type('eps', eps, (float,), 'Tensor.logit')
        return tensor_operator_registry.get('logit')(self, eps)

    def logaddexp(self, other):
        r"""
        For details, please refer to :func:`mindspore.ops.logaddexp`.
        """
        self._init_check()
        return tensor_operator_registry.get('logaddexp')(self, other)

    def logaddexp2(self, other):
        r"""
        For details, please refer to :func:`mindspore.ops.logaddexp2`.
        """
        self._init_check()
        return tensor_operator_registry.get('logaddexp2')(self, other)

    def logsumexp(self, dim, keepdim=False):
        r"""
        For details, please refer to :func:`mindspore.ops.logsumexp`.
        """
        self._init_check()
        return tensor_operator_registry.get('logsumexp')(self, dim, keepdim)

    def logdet(self):
        r"""
        For details, please refer to :func:`mindspore.ops.logdet`.
        """
        self._init_check()
        return tensor_operator_registry.get('logdet')(self)

    def i0(self):
        r"""
        For details, please refer to :func:`mindspore.ops.i0`.
        """
        self._init_check()
        return tensor_operator_registry.get('i0')(self)

    def isclose(self, x2, rtol=1e-05, atol=1e-08, equal_nan=False):
        """
        For details, please refer to :func:`mindspore.ops.isclose`.
        """
        self._init_check()
        return tensor_operator_registry.get('isclose')(self, x2, rtol, atol, equal_nan)

    def isneginf(self):
        r"""
        For details, please refer to :func:`mindspore.ops.isneginf`.
        """
        self._init_check()
        return tensor_operator_registry.get('isneginf')(self)

    def isposinf(self):
        r"""
        For details, please refer to :func:`mindspore.ops.isposinf`.
        """
        self._init_check()
        return tensor_operator_registry.get('isposinf')(self)

    def isreal(self):
        r"""
        For details, please refer to :func:`mindspore.ops.isreal`.
        """
        self._init_check()
        return tensor_operator_registry.get('isreal')(self)

    def isfinite(self):
        r"""
        For details, please refer to :func:`mindspore.ops.isfinite`.
        """
        self._init_check()
        return tensor_operator_registry.get('isfinite')()(self)

    def is_complex(self):
        r"""
        For details, please refer to :func:`mindspore.ops.is_complex`.
        """
        self._init_check()
        return tensor_operator_registry.get('is_complex')(self)

    def inv(self):
        r"""
        For details, please refer to :func:`mindspore.ops.inv`.
        """
        self._init_check()
        return tensor_operator_registry.get('inv')(self)

    def inverse(self):
        r"""
        For details, please refer to :func:`mindspore.ops.inverse`.
        """
        self._init_check()
        return tensor_operator_registry.get('inverse')(self)

    def invert(self):
        r"""
        For details, please refer to :func:`mindspore.ops.invert`.
        """
        self._init_check()
        return tensor_operator_registry.get('invert')(self)

    def pow(self, power):
        r"""
        For details, please refer to :func:`mindspore.ops.pow`.
        """
        self._init_check()
        return tensor_operator_registry.get('pow')()(self, power)

    def log(self):
        """
        For details, please refer to :func:`mindspore.ops.log`.
        """
        self._init_check()
        return tensor_operator_registry.get('log')(self)

    def log10(self):
        r"""
        For details, please refer to :func:`mindspore.ops.log10`.
        """
        self._init_check()
        return tensor_operator_registry.get('log10')(self)

    def log2(self):
        r"""
        For details, please refer to :func:`mindspore.ops.log2`.
        """
        self._init_check()
        return tensor_operator_registry.get('log2')(self)

    def mean(self, axis=None, keep_dims=False):
        """
        For details, please refer to :func:`mindspore.ops.mean`.
        """
        self._init_check()
        if axis is None:
            axis = ()
        return tensor_operator_registry.get('mean')(keep_dims)(self, axis)

    def amin(self, axis=None, keep_dims=False):
        """
        For details, please refer to :func:`mindspore.ops.amin`.
        """
        self._init_check()
        return tensor_operator_registry.get('amin')(self, axis, keep_dims)

    def reverse(self, axis):
        """
        For details, please refer to :func:`mindspore.ops.reverse`.
        """
        self._init_check()
        return tensor_operator_registry.get('reverse')(axis)(self)

    def amax(self, axis=None, keep_dims=False):
        """
        For details, please refer to :func:`mindspore.ops.amax`.
        """
        self._init_check()
        return tensor_operator_registry.get('amax')(self, axis, keep_dims)

    def reverse_sequence(self, seq_lengths, seq_dim=0, batch_dim=0):
        """
        For details, please refer to :func:`mindspore.ops.reverse_sequence`.
        """
        self._init_check()
        return tensor_operator_registry.get("reverse_sequence")(seq_dim, batch_dim)(self, seq_lengths)

    def prod(self, axis=None, keep_dims=False):
        """
        For details, please refer to :func:`mindspore.ops.prod`.
        """
        self._init_check()
        return tensor_operator_registry.get('prod')(self, axis, keep_dims)

    def select(self, condition, y):
        r"""
        For details, please refer to :func:`mindspore.ops.select`.
        """
        self._init_check()
        if not isinstance(condition, Tensor):
            raise TypeError(f"For 'Tensor.select', the argument 'condition' should be Tensor,"
                            f" but got {type(condition)}.")
        if not isinstance(y, (Tensor, int, float)):
            raise TypeError(f"For 'Tensor.select', the argument 'y' should be Tensor, int or float,"
                            f" but got {type(y)}.")
        if isinstance(y, int) and self.dtype != mstype.int32:
            raise TypeError(f"For 'Tensor.select', if the argument 'y' is int,"
                            f" then the tensor type should be int32 but got {self.dtype}")
        if isinstance(y, float) and self.dtype != mstype.float32:
            raise TypeError(f"For 'Tensor.select', if the argument 'y' is float,"
                            f" then the tensor type should be float32 but got {self.dtype}")
        input_y = y
        if isinstance(y, (int, float)):
            input_y = tensor_operator_registry.get('zeros_like')()(self) + y
            if isinstance(y, int):
                input_y = tensor_operator_registry.get('cast')(input_y, mstype.int32)
            else:
                input_y = tensor_operator_registry.get('cast')(input_y, mstype.float32)
        return tensor_operator_registry.get('select')(condition, self, input_y)

    def transpose(self, *axes):
        r"""
        For details, please refer to :func:`mindspore.ops.transpose`.
        """
        self._init_check()
        perm = validator.check_transpose_axis(axes, self.ndim)
        return tensor_operator_registry.get('transpose')()(self, perm)

    def col2im(self, output_size, kernel_size, dilation, padding_value, stride):
        """
        For details, please refer to :func:`mindspore.ops.col2im`.
        """
        self._init_check()
        return tensor_operator_registry.get('col2im')(self, output_size, kernel_size, dilation, padding_value, stride)

    def reshape(self, *shape):
        """
        For details, please refer to :func:`mindspore.ops.reshape`.
        """
        self._init_check()
        new_shape = validator.check_reshape_shp(shape)
        return tensor_operator_registry.get('reshape')()(self, new_shape)

    def reshape_as(self, other):
        """
        Change the shape of the Tensor to the shape of `other` without changing the data.

        Args:
            other(Tensor): The result tensor has the same shape as `other`.

        Returns:
            Tensor, has the same shape as `other`.

        Supported Platforms:
            ``Ascend`` ``GPU`` ``CPU``

        Examples:
            >>> import mindspore as ms
            >>> from mindspore import Tensor
            >>> import numpy as np
            >>> x = Tensor([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]], dtype=ms.float32)
            >>> y = Tensor(np.arange(6).reshape(3,2))
            >>> output = x.reshape_as(y)
            >>> print(output)
            [[-0.1  0.3]
             [ 3.6  0.4]
             [ 0.5 -3.2]]
        """
        self._init_check()
        return tensor_operator_registry.get('reshape')()(self, other.shape)

    def ravel(self):
        """
        Return a contiguous flattened tensor.

        Returns:
            Tensor, a 1-D tensor, containing the same elements of the input.

        Supported Platforms:
            ``Ascend`` ``GPU`` ``CPU``

        See also:
            :func:`mindspore.Tensor.reshape`: Give a new shape to a tensor without changing its data.

            :func:`mindspore.Tensor.flatten`: Return a copy of the tensor collapsed into one dimension.

        Examples:
            >>> import numpy as np
            >>> from mindspore import Tensor
            >>> x = Tensor(np.ones((2,3,4), dtype=np.float32))
            >>> output = x.ravel()
            >>> print(output.shape)
            (24,)
        """
        self._init_check()
        reshape_op = tensor_operator_registry.get('reshape')()
        return reshape_op(self, (-1,))

    def round(self):
        """
        For details, please refer to :func:`mindspore.ops.round`.
        """
        self._init_check()
        return tensor_operator_registry.get('round')()(self)

    def roll(self, shifts, dims):
        """
        For details, please refer to :func:`mindspore.ops.roll`.
        """
        self._init_check()
        return tensor_operator_registry.get('roll')(shifts, dims)(self)

    def rot90(self, k, dims):
        r"""
        For details, please refer to :func:`mindspore.ops.rot90`.
        """
        self._init_check()
        return tensor_operator_registry.get('rot90')(self, k, dims)

    def deg2rad(self):
        r"""
        For details, please refer to :func:`mindspore.ops.deg2rad`.
        """
        self._init_check()
        return tensor_operator_registry.get('deg2rad')(self)

    def dot(self, other):
        r"""
        For details, please refer to :func:`mindspore.ops.dot`.
        """
        self._init_check()
        return tensor_operator_registry.get('dot')(self, other)

    def rad2deg(self):
        r"""
        For details, please refer to :func:`mindspore.ops.rad2deg`.
        """
        self._init_check()
        return tensor_operator_registry.get('rad2deg')(self)

    def copysign(self, other):
        r"""
        For details, please refer to :func:`mindspore.ops.copysign`.
        """
        self._init_check()
        return tensor_operator_registry.get('copysign')(self, other)

    def nelement(self):
        r"""
        Alias for :func:`mindspore.Tensor.numel`.
        """
        self._init_check()
        return tensor_operator_registry.get('nelement')(self)

    def numel(self):
        r"""
        For details, please refer to :func:`mindspore.ops.numel`.
        """
        self._init_check()
        return tensor_operator_registry.get('numel')(self)

    def permute(self, *dims):
        """
        For details, please refer to :func:`mindspore.ops.permute`.
        """
        self._init_check()
        if not dims:
            raise ValueError(f"For Tensor.permute, the dims must not be none.")
        if len(dims) == 1 and isinstance(dims[0], tuple):
            return tensor_operator_registry.get("permute")(self, dims[0])
        return tensor_operator_registry.get("permute")(self, dims)

    def positive(self):
        """
        For details, please refer to :func:`mindspore.ops.positive`.
        """
        self._init_check()
        return tensor_operator_registry.get("positive")(self)

    def remainder(self, divisor):
        r"""
        For details, please refer to :func:`mindspore.ops.remainder`.
        """
        self._init_check()
        return tensor_operator_registry.get('remainder')(self, divisor)

    def flatten(self, order='C'):
        r"""
        For details, please refer to :func:`mindspore.ops.flatten`.
        """
        self._init_check()
        reshape_op = tensor_operator_registry.get('reshape')()
        trans_op = tensor_operator_registry.get('transpose')()

        order = validator.check_flatten_order(order)
        if order == 'C':
            return reshape_op(self, (-1,))

        perm = tuple(range(self.ndim - 1, -1, -1))
        return reshape_op(trans_op(self, perm), (-1,))

    def float_power(self, other):
        r"""
        For details, please refer to :func:`mindspore.ops.float_power`.
        """
        self._init_check()
        return tensor_operator_registry.get('float_power')(self, other)

    def fmod(self, other):
        r"""
        For details, please refer to :func:`mindspore.ops.fmod`.
        """
        self._init_check()
        return tensor_operator_registry.get('fmod')(self, other)

    def narrow(self, axis, start, length):
        """
        For details, please refer to :func:`mindspore.ops.narrow`.
        """
        self._init_check()
        return tensor_operator_registry.get('narrow')(self, axis, start, length)

    def swapaxes(self, axis0, axis1):
        """
        For details, please refer to :func:`mindspore.ops.swapaxes`.
        """
        self._init_check()
        return tensor_operator_registry.get('swapaxes')(self, axis0, axis1)

    def swapdims(self, dim0, dim1):
        """
        For details, please refer to :func:`mindspore.ops.swapdims`.
        """
        self._init_check()
        return tensor_operator_registry.get('swapdims')(self, dim0, dim1)

    def squeeze(self, axis=None):
        """
        For details, please refer to :func:`mindspore.ops.squeeze`.
        """
        self._init_check()
        if axis is None:
            return tensor_operator_registry.get('squeeze')(self)
        new_shape = validator.prepare_shape_for_squeeze(self.shape, axis)
        return tensor_operator_registry.get('reshape')()(self, new_shape)

    def slogdet(self):
        """
        For details, please refer to :func:`mindspore.ops.slogdet`.
        """
        self._init_check()
        return tensor_operator_registry.get('slogdet')(self)

    def tril(self, diagonal=0):
        """
        For details, please refer to :func:`mindspore.ops.tril`.
        """
        self._init_check()
        return tensor_operator_registry.get('tril')(self, diagonal)

    def unsqueeze(self, dim):
        """
        For details, please refer to :func:`mindspore.ops.unsqueeze`.
        """
        self._init_check()
        validator.check_is_int(dim, 'dim')
        validator.check_int_range(dim, -self.ndim - 1, self.ndim + 1, Rel.INC_LEFT, 'dim')
        return tensor_operator_registry.get('unsqueeze')(self, dim)

    def expand_dims(self, axis):
        """
        For details, please refer to :func:`mindspore.ops.expand_dims`.
        """
        self._init_check()
        validator.check_is_int(axis, 'axis')
        validator.check_int_range(axis, -self.ndim - 1, self.ndim + 1, Rel.INC_LEFT, 'axis')
        return tensor_operator_registry.get('expand_dims')(self, axis)

    def astype(self, dtype, copy=True):
        """
        Return a copy of the tensor, cast to a specified type.

        Args:
            dtype (Union[:class:`mindspore.dtype`, numpy.dtype, str]): Designated tensor dtype, can be in
                format of `mindspore.dtype.float32` or `numpy.float32` or `float32`.
            copy (bool, optional): By default, astype always returns a newly allocated
                tensor. If this is set to false, the input tensor is returned instead
                of a copy. Default: True.

        Returns:
            Tensor, with the designated dtype.

        Raises:
            TypeError: If the specified dtype cannot be understood.

        Supported Platforms:
            ``Ascend`` ``GPU`` ``CPU``

        Examples:
            >>> import numpy as np
            >>> from mindspore import Tensor
            >>> x = Tensor(np.ones((1,2,2,1), dtype=np.float32))
            >>> x = x.astype("int32")
            >>> print(x.dtype)
            Int32
        """
        self._init_check()
        dtype = _check_astype_and_convert(dtype)
        if not copy and dtype == self.dtype:
            return self
        return tensor_operator_registry.get('cast')(self, dtype)

    def argmax(self, axis=None, keepdims=False):
        """
        For details, please refer to :func:`mindspore.ops.argmax`.
        """
        if self.shape == ():
            return Tensor(0)
        a = self
        is_axis_none = False
        if axis is None:
            a = a.ravel()
            axis = 0
            is_axis_none = True
        out = tensor_operator_registry.get('argmax')(axis, mstype.int64)(a)
        if keepdims and not is_axis_none:
            out = out.expand_dims(axis)
        return out

    def argmin(self, axis=None, keepdims=False):
        """
        For details, please refer to :func:`mindspore.ops.argmin`.
        """
        if self.shape == ():
            return Tensor(0)
        # P.Argmin only supports float
        is_axis_none = False
        a = self.astype(mstype.float32)
        if axis is None:
            a = a.ravel()
            axis = 0
        else:
            axis = validator.check_axis_in_range(axis, a.ndim)
        # P.Argmin is currently not supported
        out = tensor_operator_registry.get('argmin')(axis)(a)
        if keepdims and not is_axis_none:
            out = out.expand_dims(axis)
        return out

    def argmax_with_value(self, axis=0, keep_dims=False):
        """
        Returns the maximum value with corresponding index.

        Compute the max value of input Tensor on the specified axis, and return the max value and index.

        Note:
            In auto_parallel and semi_auto_parallel mode, the first output index can not be used.

        .. warning::
            - If there are multiple maximum values, the index of the first maximum value is used.
            - The value range of `axis` is [-dims, dims - 1]. `dims` is the dimension length of this tensor.

        Args:
            axis (int): The dimension to reduce. Default: 0.
            keep_dims (bool): Whether to reduce dimension, if true the output will keep the same dimension as the input,
                            the output will reduce dimension if false. Default: False.

        Returns:
            tuple (Tensor), tuple of 2 tensors, containing the corresponding index and the maximum value of the input
            tensor.

            - **index** (Tensor) - The index for the maximum value of the input tensor.
              If `keep_dims` is true, the shape of
              output tensors is :math:`(x_1, x_2, ..., x_{axis-1}, 1, x_{axis+1}, ..., x_N)`. Otherwise, the shape is
              :math:`(x_1, x_2, ..., x_{axis-1}, x_{axis+1}, ..., x_N)` .
            - **value** (Tensor) - The maximum value of input tensor, with the same shape as index.

        Raises:
            TypeError: If `keep_dims` is not a bool.
            TypeError: If `axis` is not an int.

        Supported Platforms:
            ``Ascend`` ``GPU`` ``CPU``

        Examples:
            >>> x = Tensor(np.array([0.0, 0.4, 0.6, 0.7, 0.1]), mindspore.float32)
            >>> index, output = x.argmax_with_value()
            >>> print(index, output)
            3 0.7
            >>> index, output = x.argmax_with_value(keep_dims=True)
            >>> print(index, output)
            [3] [0.7]
        """
        if self.shape == ():
            return (Tensor(0), self)
        self._init_check()
        return tensor_operator_registry.get('argmax_with_value')(self, axis, keep_dims)

    def argmin_with_value(self, axis=0, keep_dims=False):
        """
        Returns the minimum value with corresponding index.

        Note:
            In auto_parallel and semi_auto_parallel mode, the first output index can not be used.

        .. warning::
            - If there are multiple minimum values, the index of the first minimum value is used.
            - The value range of `axis` is [-dims, dims - 1]. `dims` is the dimension length of this tensor.

        Args:
            axis (int): The dimension to reduce. Default: 0.
            keep_dims (bool): Whether to reduce dimension, if true the output will keep the same dimension as the input,
                            the output will reduce dimension if false. Default: False.

        Returns:
            tuple (Tensor), tuple of 2 tensors, containing the corresponding index and the minimum value of the input
            tensor.

            - **index** (Tensor) - The index for the minimum value of the input tensor.
              If `keep_dims` is true, the shape of
              output tensors is :math:`(x_1, x_2, ..., x_{axis-1}, 1, x_{axis+1}, ..., x_N)`. Otherwise, the shape is
              :math:`(x_1, x_2, ..., x_{axis-1}, x_{axis+1}, ..., x_N)` .
            - **value** (Tensor) - The minimum value of input tensor, with the same shape as index.

        Raises:
            TypeError: If `keep_dims` is not a bool.
            TypeError: If `axis` is not an int.

        Supported Platforms:
            ``Ascend`` ``GPU`` ``CPU``

        Examples:
            >>> x = Tensor(np.array([0.0, 0.4, 0.6, 0.7, 0.1]), mindspore.float32)
            >>> index, output = x.argmin_with_value()
            >>> print(index, output)
            0 0.0
            >>> index, output = x.argmin_with_value(keep_dims=True)
            >>> print(index, output)
            [0] [0.0]
        """
        if self.shape == ():
            return (Tensor(0), self)
        self._init_check()
        return tensor_operator_registry.get('argmin_with_value')(self, axis, keep_dims)

    def cumsum(self, axis=None, dtype=None):
        """
        For details, please refer to :func:`mindspore.ops.cumsum`.
        """
        x = self
        original_dtype = x.dtype
        # If original tensor is int, and has precision less then int32, convert to int32
        if x.dtype in (mstype.bool_, mstype.int8, mstype.int16, mstype.uint8, mstype.int16):
            x = x.astype(mstype.int32)
        if axis is None:
            x = x.ravel()
            axis = 0
        validator.check_axis_in_range(axis, x.ndim)
        if dtype is not None and original_dtype != dtype:
            return tensor_operator_registry.get('cumsum')()(x, axis).astype(dtype, copy=False)
        return tensor_operator_registry.get('cumsum')()(x, axis)

    def cummin(self, axis):
        r"""
        For details, please refer to :func:`mindspore.ops.cummin`.
        """
        return tensor_operator_registry.get('cummin')(self, axis)

    def cummax(self, axis):
        r"""
        For details, please refer to :func:`mindspore.ops.cummax`.
        """
        return tensor_operator_registry.get('cummax')(self, axis)

    def index_fill(self, axis, index, value):
        """
        For details, please refer to :func:`mindspore.ops.index_fill`.
        """
        return tensor_operator_registry.get('index_fill')(self, axis, index, value)

    def index_select(self, axis, index):
        """
        For details, please refer to :func:`mindspore.ops.index_select`.
        """
        self._init_check()
        return tensor_operator_registry.get('index_select')(self, axis, index)

    def inplace_update(self, v, indices):
        """
        For details, please refer to :func:`mindspore.ops.inplace_update`.
        """
        self._init_check()
        return tensor_operator_registry.get('inplace_update')(indices)(self, v)

    def copy(self):
        """
        Return a copy of the tensor.

        Note:
            The current implementation does not support `order` argument.

        Returns:
            Copied tensor.

        Supported Platforms:
            ``Ascend`` ``GPU`` ``CPU``

        Examples:
            >>> import numpy as np
            >>> from mindspore import Tensor
            >>> a = Tensor(np.ones((3,3)).astype("float32"))
            >>> output = a.copy()
            >>> print(output)
            [[1. 1. 1.]
            [1. 1. 1.]
            [1. 1. 1.]]
        """
        if self.size == 0:
            return self
        origin_dtype = self.dtype
        x = self
        logical_not_op = tensor_operator_registry.get('logical_not')
        if origin_dtype == mstype.bool_:
            return logical_not_op(logical_not_op(x))
        if origin_dtype != mstype.float64:
            x = x.astype("float32")
        x = x / 1.0
        x = x.astype(origin_dtype)
        return x

    def max(self, axis=None, keepdims=False, initial=None, where=True):
        """
        Return the maximum of a tensor or maximum along an axis.

        Args:
            axis (Union[None, int, list, tuple of ints], optional): Axis or
                axes along which to operate. By default, flattened input is used. If
                this is a tuple of ints, the maximum is selected over multiple axes,
                instead of a single axis or all the axes as before. Default: None.
            keepdims (bool, optional):
                If this is set to True, the axes which are reduced are left in the
                result as dimensions with size one. With this option, the result will
                broadcast correctly against the input array. Default: False.
            initial (scalar, optional):
                The minimum value of an output element. Must be present to allow
                computation on empty slice. Default: None.
            where (bool Tensor, optional):
                A boolean tensor which is broadcasted to match the dimensions of array,
                and selects elements to include in the reduction. If non-default value
                is passed, initial must also be provided. Default: True.

        Returns:
            Tensor or scalar, maximum of input tensor. If `axis` is None, the result is a scalar
            value. If `axis` is given, the result is a tensor of dimension ``self.ndim - 1``.

        Raises:
            TypeError: If arguments have types not specified above.

        Supported Platforms:
            ``Ascend`` ``GPU`` ``CPU``

        See also:
            :func:`mindspore.Tensor.argmin`: Return the indices of the minimum values along an axis.

            :func:`mindspore.Tensor.argmax`: Return the indices of the maximum values along an axis.

            :func:`mindspore.Tensor.min`: Return the minimum of a tensor or minimum along an axis.

        Examples:
            >>> import numpy as np
            >>> from mindspore import Tensor
            >>> a = Tensor(np.arange(4).reshape((2, 2)).astype('float32'))
            >>> output = a.max()
            >>> print(output)
            3.0
        """
        reduce_ = tensor_operator_registry.get("reduce")
        reduce_max = tensor_operator_registry.get("reduce_max")
        maximum = tensor_operator_registry.get("maximum")
        return reduce_(self, reduce_max(keepdims), cmp_fn=maximum, axis=axis, keepdims=keepdims,
                       initial=initial, where=where)

    def min(self, axis=None, keepdims=False, initial=None, where=True):
        """
        Return the minimum of a tensor or minimum along an axis.

        Args:
            axis (Union[None, int, list, tuple of ints], optional): An axis or
                axes along which to operate. By default, flattened input is used. If
                `axis` is a tuple of ints, the minimum is selected over multiple axes,
                instead of a single axis or all the axes as before. Default: None.
            keepdims (bool, optional):
                If True, the axes which are reduced are left in the
                result as dimensions with size one. With this option, the result will
                broadcast correctly against the input array. Default: False.
            initial (scalar, optional):
                The minimum value of an output element. Must be present to allow
                computation on empty slice. Default: None.
            where (bool Tensor, optional):
                A boolean tensor which is broadcasted to match the dimensions of array,
                and selects elements to include in the reduction. If non-default value
                is passed, initial must also be provided. Default: True.

        Returns:
            Tensor or scalar, minimum of input tensor. If `axis` is None, the result is a scalar
            value. If `axis` is given, the result is a tensor of dimension ``self.ndim - 1``.

        Raises:
            TypeError: If arguments have types not specified above.

        Supported Platforms:
            ``Ascend`` ``GPU`` ``CPU``

        See also:
            :func:`mindspore.Tensor.argmin`: Return the indices of the minimum values along an axis.

            :func:`mindspore.Tensor.argmax`: Return the indices of the maximum values along an axis.

            :func:`mindspore.Tensor.max`: Return the minimum of a tensor or minimum along an axis.

        Examples:
            >>> import numpy as np
            >>> from mindspore import Tensor
            >>> a = Tensor(np.arange(4).reshape((2, 2)).astype('float32'))
            >>> output = a.min()
            >>> print(output)
            0.0
            >>> output = a.min(axis=0)
            >>> print(output)
            [0. 1.]
            >>> output = a.min(axis=0, initial=9, where=Tensor([False]))
            >>> print(output)
            [9. 9.]
            >>> output = a.min(axis=0, initial=9, where=Tensor([False, True]))
            >>> print(output)
            [9. 1.]
        """
        reduce_ = tensor_operator_registry.get("reduce")
        reduce_min = tensor_operator_registry.get("reduce_min")
        minimum = tensor_operator_registry.get("minimum")
        return reduce_(self, reduce_min(keepdims), cmp_fn=minimum(), axis=axis, keepdims=keepdims,
                       initial=initial, where=where)

    def scatter_add(self, indices, updates):
        """
        For details, please refer to :func:`mindspore.ops.scatter_add`.
        """
        self._init_check()
        return tensor_operator_registry.get("tensor_scatter_add")(self, indices, updates)

    def scatter_sub(self, indices, updates):
        """
        Creates a new tensor by subtracting the values from the positions in self tensor indicated by
        `indices`, with values from `updates`. When multiple values are provided for the same
        index, the result of the update will be to subtract these values respectively. This operation is almost
        equivalent to using :class:`mindspore.ops.ScatterNdSub` , except that the updates are applied on output `Tensor`
        instead of input `Parameter`.

        The last axis of `indices` is the depth of each index vectors. For each index vector,
        there must be a corresponding value in `updates`. The shape of `updates` should be
        equal to the shape of `self[indices]`. For more details, see use cases.

        Note:
            On GPU, if some values of the `indices` are out of bound, instead of raising an index error,
            the corresponding `updates` will not be updated to self tensor. On CPU, if some values of
            the `indices` are out of bound, raising an index error. On Ascend, out of bound checking is
            not supported, if some values of the `indices` are out of bound, unknown errors may be caused.

        Args:
            indices (Tensor): The index of input tensor whose data type is int32 or int64.
                The rank must be at least 2.
            updates (Tensor): The tensor to update the input tensor, has the same type as input,
                and updates.shape should be equal to indices.shape[:-1] + self.shape[indices.shape[-1]:].

        Returns:
            Tensor, has the same shape and type as self tensor.

        Raises:
            TypeError: If dtype of `indices` is neither int32 nor int64.
            ValueError: If length of shape of self tensor is less than the last dimension of shape of `indices`.

        Supported Platforms:
            ``Ascend`` ``GPU`` ``CPU``

        Examples:
            >>> import numpy as np
            >>> from mindspore import Tensor
            >>> x = Tensor(np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]).astype('float32'))
            >>> indices = Tensor(np.array([[0, 0], [0, 0]]).astype('int32'))
            >>> updates = Tensor(np.array([1.0, 2.2]).astype('float32'))
            >>> output = x.scatter_sub(indices, updates)
            >>> print(output)
            [[-3.3000002  0.3        3.6      ]
            [ 0.4        0.5       -3.2      ]]
        """
        self._init_check()
        return tensor_operator_registry.get('tensor_scatter_sub')(self, indices, updates)

    def scatter_min(self, indices, updates):
        """
        For details, please refer to :func:`mindspore.ops.scatter_min`.
        """
        self._init_check()
        return tensor_operator_registry.get('tensor_scatter_min')()(self, indices, updates)

    def scatter_max(self, indices, updates):
        """
        For details, please refer to :func:`mindspore.ops.scatter_max`.
        """
        self._init_check()
        return tensor_operator_registry.get('tensor_scatter_max')()(self, indices, updates)

    def masked_fill(self, mask, value):
        """
        For details, please refer to :func:`mindspore.ops.masked_fill`.
        """
        self._init_check()
        if isinstance(value, (float, int)):
            value = tensor_operator_registry.get("scalar_to_tensor")(value, self.dtype)
        if not isinstance(mask, Tensor):
            raise TypeError("For 'Tensor.masked_fill', the type of the argument 'mask' must be Tensor, but "
                            "got {}.".format(type(mask)))
        validator.check_type_name('mask', mask.dtype, [mstype.bool_], "Tensor")
        return tensor_operator_registry.get("masked_fill")(self, mask, value)

    def ptp(self, axis=None, keepdims=False):
        """
        The name of the function comes from the acronym for "peak to peak". Calculate the difference between the
        maximum value and the minimum value along the axis.

        Note:
            Numpy argument `out` is not supported.

        Args:
            axis (Union[None, int, tuple(int)]): Axis or axes along which the range is computed.
                The default is to compute the variance of the flattened tensor. Default: None.
            keepdims (bool): If this is set to True, the axes which are reduced are left in the result as
                dimensions with size one. With this option, the result will broadcast correctly against the tensor.
                Default is False.

        Returns:
            Tensor.

        Raises:
            TypeError: If `self` is not a tensor, or `axis` and `keepdims` have types not specified above.

        Supported Platforms:
            ``Ascend`` ``GPU`` ``CPU``

        Examples:
            >>> from mindspore import Tensor
            >>> x = Tensor([[4.0, 9.0, 2.0, 10.0], [6.0, 9.0, 7.0, 12.0]]).astype("float32")
            >>> print(x.ptp(axis=1))
            [8. 6.]
            >>> print(x.ptp(axis=0))
            [2. 0. 5. 2.]
        """
        if not isinstance(keepdims, bool):
            raise TypeError("For 'Tensor.ptp', the type of the argument 'keepdims' must be bool, "
                            "but got {}.".format(type(keepdims)))
        if axis is None:
            axis = ()
        else:
            validator.check_axis_type(axis, True, True, False)
            axis = validator.check_axis_valid(axis, self.ndim)

        return self.max(axis, keepdims) - self.min(axis, keepdims)

    def minimum(self, other):
        r"""
        For details, please refer to :func:`mindspore.ops.minimum`.
        """
        return tensor_operator_registry.get('minimum')()(self, other)

    def clamp(self, min=None, max=None):
        r"""
        For details, please refer to :func:`mindspore.ops.clamp`.
        """
        self._init_check()
        return tensor_operator_registry.get('clamp')(self, min, max)

    def clip(self, min=None, max=None):
        r"""
        Alias for :func:`mindspore.Tensor.clamp`.
        """
        return self.clamp(min, max)

    def _init_check(self):
        if self.has_init:
            self.init_data()
        return self

    def init_data(self, slice_index=None, shape=None, opt_shard_group=None):
        """
        Get the tensor format data of this Tensor.

        Note:
            The init_data function can be called once for the same tensor.

        Args:
            slice_index (int): Slice index of a parameter's slices.
                It is used when initialize a slice of a parameter, it guarantees that devices
                using the same slice can generate the same tensor. Default: None.
            shape (list[int]): Shape of the slice, it is used when initialize a slice of the parameter. Default: None.
            opt_shard_group(str): Optimizer shard group which is used in auto or semi auto parallel mode
                to get one shard of a parameter's slice. Default: None.

        Returns:
            Initialized Tensor.

        Supported Platforms:
            ``Ascend`` ``GPU`` ``CPU``

        Examples:
            >>> import mindspore as ms
            >>> from mindspore.common.initializer import initializer, Constant
            >>> x = initializer(Constant(1), [2, 2], ms.float32)
            >>> out = x.init_data()
            >>> print(out)
            [[1. 1.]
             [1. 1.]]
        """
        if self.init is None:
            raise TypeError("init_data must be set Tensor.init, init can't be None")

        if shape is None:
            shape = self.shape
        # At embedding cache scenes, we need limit the size of memory for tensor.
        # And save out of range data to persistent storage to support TB-Level size of tensor.
        data_shape = list(shape)
        slice_num_of_persistent_data = get_slice_num(self.dtype, shape)
        if slice_num_of_persistent_data > 1:
            slice_first_dim = math.ceil(shape[0] / slice_num_of_persistent_data)
            data_shape[0] = slice_first_dim
            self.slice_shape_of_persistent_data_ = data_shape
            self.slice_num_of_persistent_data_ = slice_num_of_persistent_data

        try:
            data = np.ndarray(data_shape, dtype=mstype.dtype_to_nptype(self.dtype))
        except ValueError as e:
            msg = "Error shape={}".format(shape)
            logger.critical(msg)
            raise ValueError(msg) from e

        class seed_context:
            """Set and restore seed."""

            def __init__(self, init):
                self.init = init
                global_seed = get_seed()
                self._np_seed = np.random.get_state()[1][0]
                self.need_set_seed = (slice_index is not None)
                self._global_seed = global_seed
                self._seed_offset = 1
                if self.need_set_seed:
                    self._seed_offset = get_group_size() * 2

            def __enter__(self):
                if self.need_set_seed:
                    self.seed = self.init.seed
                    if self._global_seed is not None:
                        np.random.seed(slice_index + self._global_seed)
                        self.init.seed = slice_index + self._global_seed
                    else:
                        np.random.seed(slice_index + Tensor.delta_seed)
                        self.init.seed = slice_index + Tensor.delta_seed
                        Tensor.delta_seed += self._seed_offset

            def __exit__(self, ptype, value, trace):
                if self.need_set_seed:
                    np.random.seed(self._np_seed)
                    self.init.seed, _ = self.seed

        with seed_context(self.init):
            self.init(data)
        self.init = None

        # At embedding cache scenes. When size of tensor is out of range, we store data to persistent storage
        if slice_num_of_persistent_data > 1:
            self.assign_value(Tensor_.persistent_data_from_numpy(data, slice_num_of_persistent_data))
        else:
            self.assign_value(Tensor_.from_numpy(data))
        return self

    def to_tensor(self, slice_index=None, shape=None, opt_shard_group=None):
        """
        Return init_data() and get the tensor format data of this Tensor.

        Note:
            The usage of `to_tensor` is deprecated. Please use `init_data`.

        Args:
            slice_index (int): Slice index of a parameter's slices.
                It is used when initialize a slice of a parameter, it guarantees that devices
                using the same slice can generate the same tensor. Default: None.
            shape (list[int]): Shape of the slice, it is used when initialize a slice of the parameter. Default: None.
            opt_shard_group(str): Optimizer shard group which is used in auto or semi auto parallel mode
                to get one shard of a parameter's slice. Default: None.

        Returns:
            Initialized Tensor.

        Raises:
            TypeError: `indices` is neither int32 nor int64.
            ValueError: The length of the shape of the tensor is less than the last dimension of `indices`.

        Supported Platforms:
            ``Ascend`` ``GPU`` ``CPU``

        Examples:
            >>> import mindspore as ms
            >>> from mindspore.common.initializer import initializer, Constant
            >>> x = initializer(Constant(1), [2, 2], ms.float32)
            >>> out = x.to_tensor()
            >>> print(out)
            [[1. 1.]
             [1. 1.]]
        """
        logger.warning("WARN_DEPRECATED: The usage of to_tensor is deprecated."
                       " Please use init_data")
        return self.init_data(slice_index, shape, opt_shard_group)

    def resize(self, *new_shape):
        """
        Changes shape and size of tensor in-place.

        If the shape of the new tensor is larger than the shape of the original tensor, the new tensor will be filled
        with 0. And if the shape of the new tensor is smaller than the shape of the original tensor, the new tensor is
        filled with the elements of the original tensor in order.

        Note:
            Instead of changing the size of the input tensor and returns nothing as in numpy,
            this method returns a new Tensor with the input size.
            Numpy argument `refcheck` is not supported.

        Args:
            new_shape (Union[ints, tuple of ints]): Shape of resized tensor.

        Returns:
            Tensor.

        Supported Platforms:
            ``Ascend`` ``GPU`` ``CPU``

        See also:
            :func:`mindspore.Tensor.reshape`: Give a new shape to a tensor without changing its data.

            :func:`mindspore.Tensor.repeat`: Repeat elements of a tensor.

        Examples:
            >>> import numpy as np
            >>> from mindspore import Tensor
            >>> x = Tensor(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32))
            >>> y = x.resize(3, 3)
            >>> print(y)
            [[1. 2. 3.]
            [4. 5. 6.]
            [0. 0. 0.]]
            >>> y = x.resize(2, 2)
            >>> print(y)
            [[1. 2.]
            [3. 4.]]
        """
        if not new_shape:
            return self
        if len(new_shape) == 1:
            if isinstance(new_shape[0], tuple):
                new_shape = new_shape[0]
        flattened = self.ravel()
        cur_size = flattened.size
        new_size = tensor_operator_registry.get('shape_mul')(new_shape)
        diff_size = new_size - cur_size
        if diff_size > 0:
            pad_val = tensor_operator_registry.get('fill')(self.dtype, (diff_size,), 0)
            res = tensor_operator_registry.get('concatenate')(0)((flattened, pad_val))
        else:
            res = flattened[:new_size]
        return res.reshape(new_shape)

    def det(self):
        r"""
        For details, please refer to :func:`mindspore.ops.det`.
        """
        self._init_check()
        return tensor_operator_registry.get('det')(self)

    def diff(self, n=1, axis=-1, prepend=None, append=None):
        r"""
        For details, please refer to :func:`mindspore.ops.diff`.
        """
        self._init_check()
        return tensor_operator_registry.get('diff')(self, n, axis, prepend, append)

    def frac(self):
        r"""
        For details, please refer to :func:`mindspore.ops.frac`.
        """
        self._init_check()
        return tensor_operator_registry.get('frac')(self)

    def argwhere(self):
        r"""
        For details, please refer to :func:`mindspore.ops.argwhere`.
        """
        self._init_check()
        return tensor_operator_registry.get('argwhere')(self)

    def moveaxis(self, source, destination):
        r"""
        For details, please refer to :func:`mindspore.ops.moveaxis`.
        """
        self._init_check()
        return tensor_operator_registry.get('moveaxis')(self, source, destination)

    def movedim(self, source, destination):
        r"""
        For details, please refer to :func:`mindspore.ops.movedim`.
        """
        self._init_check()
        return tensor_operator_registry.get('movedim')(self, source, destination)

    def digamma(self):
        r"""
        For details, please refer to :func:`mindspore.ops.digamma`.
        """
        self._init_check()
        return tensor_operator_registry.get('digamma')(self)

    def lgamma(self):
        r"""
        For details, please refer to :func:`mindspore.ops.lgamma`.
        """
        self._init_check()
        return tensor_operator_registry.get('lgamma')(self)

    def diagonal(self, offset=0, axis1=0, axis2=1):
        """
        For details, please refer to :func:`mindspore.ops.diagonal`.
        """
        ndim = self.ndim
        if ndim < 2:
            raise ValueError("For 'Tensor.diagonal', the original tensor requires at least two dimensions, "
                             "but got {}.".format(ndim))
        dtype = self.dtype

        axes = validator.check_axis_valid((axis1, axis2), ndim)
        perm = ()
        for i in range(ndim):
            if i not in axes:
                perm += (i,)
        perm += axes
        a = self.transpose(perm)

        shape = a.shape
        n, m = shape[-2:]

        e = tensor_operator_registry.get('eye')(n, m, dtype)
        if offset >= m or offset <= -n:
            e = tensor_operator_registry.get('fill')(dtype, (n, m), 0)
        elif offset != 0:
            e = e.astype(mstype.float32)
            if offset > 0:
                e_left = tensor_operator_registry.get('fill')(mstype.float32, (n, offset), 0)
                e_right = e[..., 0:m - offset:1]
                e = tensor_operator_registry.get('concatenate')(1)((e_left, e_right)).astype(dtype)
            elif offset < 0:
                e_upper = tensor_operator_registry.get('fill')(mstype.float32, (-offset, m), 0)
                e_lower = e[0:n + offset:1, ...]
                e = tensor_operator_registry.get('concatenate')(0)((e_upper, e_lower)).astype(dtype)
        e = tensor_operator_registry.get('broadcast_to')(shape)(e)

        prod = tensor_operator_registry.get('__mul__')(a, e)
        res = tensor_operator_registry.get('reduce_sum')(prod.astype(mstype.float32), -1)

        begin = ()
        for _ in range(ndim - 2):
            begin += (0,)
        last_dim_begin = max(0, -offset)
        begin += (last_dim_begin,)
        size = res.shape[:-1]
        last_dim_end = min(
            shape[-2], max(0, shape[-1] - offset)) - last_dim_begin
        if last_dim_end <= 0:
            return Tensor([])
        size += (last_dim_end,)
        res = tensor_operator_registry.get('tensor_slice')(res, begin, size)
        return res.astype(dtype)

    def trace(self, offset=0, axis1=0, axis2=1, dtype=None):
        """
        Return the sum along diagonals of the tensor.

        Args:
            offset (int, optional): Offset of the diagonal from the main diagonal.
                Can be positive or negative. Defaults to main diagonal.
            axis1 (int, optional): Axis to be used as the first axis of the 2-D
                sub-arrays from which the diagonals should be taken. Defaults to
                first axis (0).
            axis2 (int, optional): Axis to be used as the second axis of the 2-D
                sub-arrays from which the diagonals should be taken. Defaults to
                second axis.
            dtype (:class:`mindspore.dtype`, optional): defaults to None. Overrides the dtype of the
                output Tensor.

        Returns:
            Tensor, the sum along diagonals.

        Raises:
            ValueError: If the input tensor has less than two dimensions.

        Supported Platforms:
            ``Ascend`` ``GPU`` ``CPU``

        See also:
            :func:`mindspore.Tensor.diagonal`: Return specified diagonals.

        Examples:
            >>> import numpy as np
            >>> from mindspore import Tensor
            >>> x = Tensor(np.eye(3, dtype=np.float32))
            >>> print(x.trace())
            3.0
        """
        d = self.diagonal(offset, axis1=axis1, axis2=axis2)
        shape = d.shape
        if dtype is None:
            dtype = d.dtype
        if shape[-1] == 0:
            return tensor_operator_registry.get('fill')(dtype, shape[:-1], 0)
        res = tensor_operator_registry.get('reduce_sum')(d.astype(mstype.float32), -1)
        return res.astype(dtype)

    def take(self, indices, axis=None, mode='clip'):
        """
        Takes elements from a tensor along an axis.

        Args:
            indices (Tensor): The indices with shape `(Nj...)` of the values to extract.
            axis (int, optional): The axis over which to select values. By default,
                the flattened input tensor is used. Default: `None`.
            mode ('raise', 'wrap', 'clip', optional):

                - raise: Raises an error;

                - wrap: Wraps around;

                - clip: Clips to the range. 'clip' mode means that all indices that are
                  too large are replaced by the index that addresses the last element
                  along that axis. Note that this disables indexing with negative numbers.

                Default: 'clip'.

        Returns:
            Tensor, the indexed result.

        Raises:
            ValueError: If `axis` is out of range, or `mode` has values other than ('raise', 'wrap', 'clip')

        Supported Platforms:
            ``Ascend`` ``GPU`` ``CPU``

        Examples:
            >>> import numpy as np
            >>> from mindspore import Tensor
            >>> a = Tensor(np.array([4, 3, 5, 7, 6, 8]))
            >>> indices = Tensor(np.array([0, 1, 4]))
            >>> output = a.take(indices)
            >>> print(output)
            [4 3 6]
        """
        if mode not in ('raise', 'wrap', 'clip'):
            raise ValueError(f"For 'Tensor.take', the argument 'mode' should be one of in ['raise', 'wrap', 'clip'],"
                             f" but got {mode}.")
        if axis is None:
            a = self.ravel()
            axis = 0
        else:
            a = self
        ndim = a.ndim
        validator.check_axis_in_range(axis, ndim)
        axis = axis + ndim if axis < 0 else axis

        shape_a = a.shape
        shape_indices = indices.shape
        size_indices = indices.size
        indices = tensor_operator_registry.get('check_indices')(shape_a[axis], indices, mode)

        # reshapes indices to shape (Ni..., Nj..., Nk)
        shape_ni = shape_a[:axis]
        shape_nk = shape_a[axis + 1:]
        shape_out = shape_ni + shape_indices + shape_nk
        shape_indices = tuple(size_indices if i == axis else 1 for i in range(ndim))
        indices = indices.reshape(shape_indices)
        shape_indices = shape_ni + (indices.size,) + shape_nk
        indices = tensor_operator_registry.get('broadcast_to')(shape_indices)(indices)

        res = tensor_operator_registry.get('gather_d')(a, axis, indices)
        return res.reshape(shape_out)

    def choose(self, choices, mode='clip'):
        """
        Construct a tensor from an index tensor and a list of tensors to choose from.

        Args:
            choices (Union[tuple, list, Tensor]): Choice tensors. The input tensor and all of the
                `choices` must be broadcasted to the same shape. If `choices` is itself a tensor,
                then its outermost dimension (i.e., the one corresponding to ``choices.shape[0]``)
                is taken as defining the "sequence".
            mode ('raise', 'wrap', 'clip', optional): Specifies how indices outside
                ``[0, n-1]`` will be treated:

                - raise: Raises an error;

                - wrap: Wraps around;

                - clip: Clips to the range. 'clip' mode means that values greater than n-1 are mapped to n-1.
                  Note that this disables indexing with negative numbers.

                Default: 'clip'.

        Returns:
            Tensor, the merged result.

        Raises:
            ValueError: If the input tensor and any of the `choices` cannot be broadcast.

        Supported Platforms:
            ``Ascend`` ``GPU`` ``CPU``

        Examples:
            >>> import numpy as np
            >>> from mindspore import Tensor
            >>> choices = [[0, 1, 2, 3], [10, 11, 12, 13], [20, 21, 22, 23], [30, 31, 32, 33]]
            >>> x = Tensor(np.array([2, 3, 1, 0]))
            >>> print(x.choose(choices))
            [20 31 12  3]
        """
        if isinstance(choices, Tensor):
            shape_choice = validator.infer_out_shape(self.shape, choices.shape[1:])
            choices = tensor_operator_registry.get('broadcast_to')((choices.shape[0],) + shape_choice)(choices)
        else:
            # broadcasts choices to the same shape if choices is a sequence
            choicelist = []
            shapes = ()
            for choice in choices:
                if not isinstance(choice, Tensor):
                    choice = tensor_operator_registry.get('make_tensor')(choice)
                shapes += (choice.shape,)
                choicelist.append(choice)
            shape_choice = validator.infer_out_shape(self.shape, *shapes)
            tmp = []
            for choice in choicelist:
                tmp.append(tensor_operator_registry.get('broadcast_to')(shape_choice)(choice))
            choices = tensor_operator_registry.get('stack')(tmp, 0)

        if self.ndim == 0 or choices.ndim == 0:
            raise ValueError(f"For 'Tensor.choose', the original tensor and the argument 'choices' cannot be scalars."
                             f" Their dimensions should all be > 0, but got the original tensor's dimension "
                             f"{self.ndim}, 'choices' dimension {choices.ndim}.")
        a = tensor_operator_registry.get('broadcast_to')(shape_choice)(self)
        dtype = choices.dtype
        # adjusts dtype for F.tensor_mul and F.gather_nd
        a = a.astype(mstype.int32)
        choices = choices.astype(mstype.int32)
        a = tensor_operator_registry.get('check_indices')(choices.shape[0], a, mode, allow_negative_index=False)

        grids = []
        ndim = len(a.shape)
        for i in range(ndim):
            dim_grid = Tensor(list(range(a.shape[i])), mstype.int32)
            dim_shape = validator.expanded_shape(ndim, a.shape[i], i)
            dim_grid = tensor_operator_registry.get('broadcast_to')(a.shape)(dim_grid.reshape(dim_shape))
            grids.append(dim_grid)
        grid = tensor_operator_registry.get('stack')(grids, -1)
        indices = tensor_operator_registry.get('concatenate')(-1)((a.reshape(a.shape + (1,)), grid))
        return tensor_operator_registry.get('gather_nd')(choices, indices).astype(dtype)

    def searchsorted(self, v, side='left', sorter=None):
        """
        Finds indices where elements should be inserted to maintain order.

        Args:
            v (Union[int, float, bool, list, tuple, Tensor]): Values to insert into the tensor.
            side (str, optional): If 'left', the index of the first suitable
                location found is given. If 'right', return the last such index. If there is
                no suitable index, return either 0 or N (where N is the length of the tensor).
                Default: 'left'.
            sorter (Union[int, float, bool, list, tuple, Tensor]): 1-D optional tensor of
                integer indices that sort the tensor into ascending order. They are typically
                the result of argsort. Default: None.

        Returns:
            Tensor, array of insertion points with the same shape as `v`.

        Raises:
            ValueError: If argument for `side` or `sorter` is invalid.

        Supported Platforms:
            ``Ascend`` ``GPU`` ``CPU``

        Examples:
            >>> import numpy as np
            >>> from mindspore import Tensor
            >>> x = Tensor(np.array([1, 2, 3, 4, 5]))
            >>> print(x.searchsorted(3))
            2
        """
        if side not in ('left', 'right'):
            raise ValueError(f"For 'Tensor.searchsorted', the argument 'side' should be one of in "
                             f"['left', 'right'], but got {side}.")
        a = self.astype(mstype.float32)
        if not isinstance(v, Tensor):
            v = tensor_operator_registry.get('make_tensor')(v)
        shape = v.shape
        if sorter is not None:
            if not isinstance(sorter, (int, float, bool, list, tuple, Tensor)):
                raise TypeError("For Tensor.searchsorted, the type of the argument 'sorter' must be one of 'int', "
                                "'float', 'bool', 'list', 'tuple', 'Tensor', but got {}.".format(type(sorter)))
            if not isinstance(sorter, Tensor):
                sorter = tensor_operator_registry.get('make_tensor')(sorter)
            if sorter.ndim != 1 or sorter.size != a.size:
                raise ValueError('sorter must be 1-D array with the same size as the Tensor')
            sorter = sorter.reshape(sorter.shape + (1,))
            a = tensor_operator_registry.get('gather_nd')(a, sorter)
        less_op = tensor_operator_registry.get('__le__') if side == 'left' else tensor_operator_registry.get('__lt__')
        i = tensor_operator_registry.get('fill')(mstype.int32, shape, 0)
        j = tensor_operator_registry.get('fill')(mstype.int32, shape, a.size)

        sort_range = tuple(range(validator.get_log2_size(tensor_operator_registry.get('shape_mul')(a.shape) + 1)))
        for _ in sort_range:
            mid = (i - -j) // 2
            mask = less_op(v, tensor_operator_registry.get('gather_nd')(a, mid.reshape(mid.shape + (1,))))
            i = tensor_operator_registry.get('select')(mask, i, mid)
            j = tensor_operator_registry.get('select')(mask, mid, j)
        return j

    def gather_nd(self, indices):
        r"""
        For details, please refer to :func:`mindspore.ops.gather_nd`.
        """
        self._init_check()
        validator.check_value_type('indices', indices, (Tensor, Tensor_,), 'Tensor.gather_nd')
        return tensor_operator_registry.get('gather_nd')(self, indices)

    def gather(self, input_indices, axis, batch_dims=0):
        r"""
        For details, please refer to :func:`mindspore.ops.gather`.
        """
        self._init_check()
        validator.check_is_int(axis, 'axis')
        validator.check_is_int(batch_dims, "batch_dims")
        return tensor_operator_registry.get('gather')(self, input_indices, axis, batch_dims)

    def var(self, axis=None, ddof=0, keepdims=False):
        """
        Compute the variance along the specified axis.

        The variance is the average of the squared deviations from the mean, i.e.,
        :math:`var = mean(abs(x - x.mean())**2)`.

        Return the variance, which is computed for the flattened array by default,
        otherwise over the specified axis.

        Note:
            Numpy arguments `dtype`, `out` and `where` are not supported.

        Args:
            axis (Union[None, int, tuple(int)]): Axis or axes along which the variance is computed.
                The default is to compute the variance of the flattened array. Default: `None`.
            ddof (int): Means Delta Degrees of Freedom. Default: 0.
                The divisor used in calculations is :math:`N - ddof`, where :math:`N` represents the number of elements.
            keepdims (bool): Default: `False`.

        Returns:
            Variance tensor.

        Supported Platforms:
            ``Ascend`` ``GPU`` ``CPU``

        See also:
            :func:`mindspore.Tensor.mean`: Reduce a dimension of a tensor by averaging all elements in the dimension.

            :func:`mindspore.Tensor.std`: Compute the standard deviation along the specified axis.

        Examples:
            >>> import numpy as np
            >>> from mindspore import Tensor
            >>> input_x = Tensor(np.array([1., 2., 3., 4.], np.float32))
            >>> output = input_x.var()
            >>> print(output)
            1.25
        """
        if 0 in self.shape:
            return Tensor(float('nan'), self.dtype)
        if not isinstance(ddof, int):
            raise TypeError("For 'Tensor.var', the type of the argument 'ddof' must be int, but got "
                            "{}.".format(type(ddof)))
        if not isinstance(keepdims, bool):
            raise TypeError("For 'Tensor.var', the type of the argument 'keepdims' must be bool, but "
                            "got {}.".format(type(keepdims)))

        if axis is None:
            axis = ()
        else:
            axis = validator.check_and_canonicalize_axes(axis, self.ndim)
        x_mean = tensor_operator_registry.get('mean')(True)(self, axis)
        x_sub = tensor_operator_registry.get('__sub__')(self, x_mean)
        x_pow = tensor_operator_registry.get('__pow__')(x_sub, 2)
        x_sum = tensor_operator_registry.get('sum')(bool(keepdims))(x_pow, axis)
        nums = 1
        if axis == ():
            nums = self.size
        else:
            for ax in axis:
                nums *= self.shape[ax]
        return tensor_operator_registry.get('__truediv__')(x_sum, nums - ddof)

    def std(self, axis=None, ddof=0, keepdims=False):
        """
        For details, please refer to :func:`mindspore.ops.std`.
        """
        x_var = self.var(axis, ddof, keepdims)
        return tensor_operator_registry.get('__pow__')(x_var, 0.5)

    def sum(self, axis=None, dtype=None, keepdims=False, initial=None):
        """
        Return sum of tensor elements over a given axis.

        Note:
            Numpy arguments `out`, `where`, `casting`, `order`, `subok`, `signature`, and
            `extobj` are not supported.

        Args:
            axis (Union[None, int, tuple(int)]): Axis or axes along which a sum is performed. Default: None.
                If None, sum all the elements of the input tensor.
                If the axis is negative, it counts from the last to the first axis.
                If the axis is a tuple of ints, a sum is performed on all the axes specified in the tuple
                instead of a single axis or all the axes as before.
            dtype (:class:`mindspore.dtype`, optional): defaults to None. Overrides the dtype of the
                output Tensor.
            keepdims (bool): If this is set to True, the axes which are reduced are left in the result as
                dimensions with size one. With this option, the result will broadcast correctly against the input array.
                If the default value is passed, then keepdims will not be passed through to the sum method of
                sub-classes of ndarray, however any non-default value will be. If the sub-class method does not
                implement keepdims any exceptions will be raised. Default: `False`.
            initial (scalar): Starting value for the sum. Default: `None`.

        Returns:
            Tensor. A tensor with the same shape as input, with the specified axis removed.
            If the input tensor is a 0-d array, or if the axis is None, a scalar is returned.

        Raises:
            TypeError: If input is not array_like, or `axis` is not int or tuple of ints,
                or `keepdims` is not integer, or `initial` is not scalar.
            ValueError: If any axis is out of range or duplicate axes exist.

        Supported Platforms:
            ``Ascend`` ``GPU`` ``CPU``

        See also:
            :func:`mindspore.Tensor.cumsum`: Return the cumulative sum of the elements along a given axis.

        Examples:
            >>> import numpy as np
            >>> from mindspore import Tensor
            >>> input_x = Tensor(np.array([-1, 0, 1]).astype(np.float32))
            >>> print(input_x.sum())
            0.0
            >>> input_x = Tensor(np.arange(10).reshape(2, 5).astype(np.float32))
            >>> print(input_x.sum(axis=1))
            [10. 35.]
        """
        input_x = self.astype(mstype.int32) if self.dtype == mstype.bool_ else self
        dtype = input_x.dtype if dtype is None else dtype
        if not isinstance(keepdims, int):
            raise TypeError("For 'Tensor.sum', the type of the argument 'keepdims' must be int, but "
                            "got {}.".format(type(keepdims)))
        if initial is not None and not isinstance(initial, (int, float, bool)):
            raise TypeError("For 'Tensor.sum', when the argument 'initial' is not None, it must be int, "
                            "float or bool, but got {}.".format(type(initial)))
        if axis is None:
            axis = ()
        else:
            axis = validator.check_and_canonicalize_axes(axis, self.ndim)

        if not validator.check_type_support(input_x.dtype, 'GPU', (mstype.float64, mstype.float32, mstype.float16)):
            input_x = input_x.astype(mstype.float32)
        if 0 in self.shape:
            input_x = tensor_operator_registry.get('make_tensor')([0], self.dtype)
        res = tensor_operator_registry.get('sum')(bool(keepdims))(input_x, axis)
        if initial is not None:
            res += initial
        return res.astype(dtype)

    def sum_to_size(self, *size):
        r"""
        Sum self Tensor to the `size`. `size` must be expandable to the Tensor size.

        Args:
            size (Union[tuple(int), int]): The expected shape of output Tensor.

        Returns:
            Tensor, the sum result of self Tensor according to the `size`.

        Raises:
            ValueError: If `size` is not expandable to the size of self Tensor.

        Supported Platforms:
            ``Ascend`` ``GPU`` ``CPU``

        Examples:
            >>> x = Tensor(np.random.randn(3, 3, 3, 3, 3, 3), mindspore.float32)
            >>> output = x.sum_to_size((1, 3, 1, 3))
            >>> print(output.shape)
            (1, 3, 1, 3)
        """
        self._init_check()
        x = self
        if len(size) == 1 and isinstance(size[0], tuple):
            size = size[0]
        shape_x = x.shape
        if len(size) > x.ndim:
            raise ValueError(f"For sum_to_size, size {size} is not expandable to the tensor size {shape_x}.")
        if len(size) < x.ndim:
            pre_axis = tuple([axis for axis in range(x.ndim - len(size))])
            x = x.sum(pre_axis)
        axes = []
        for i, element in enumerate(size):
            if element != x.shape[i] and element == 1:
                axes.append(i)
            elif element != x.shape[i]:
                raise ValueError(f"For sum_to_size, size {size} is not expandable to the tensor size {shape_x}.")
        if axes:
            return x.sum(tuple(axes), keepdims=True)
        return x

    def nansum(self, axis=None, keepdims=False, dtype=None):
        """
        For details, please refer to :func:`mindspore.ops.nansum`.
        """
        self._init_check()
        return tensor_operator_registry.get('nansum')(self, axis=axis, keepdims=keepdims, dtype=dtype)

    def repeat(self, repeats, axis=None):
        """
        Repeat elements of a tensor.

        Args:
            repeats (Union[int, tuple, list]): The number of repetitions for each element.
                `repeats` is broadcasted to fit the shape of the given axis.
            axis (int, optional): The axis along which to repeat values. By default,
                use the flattened input tensor, and return a flat output tensor. Default: None.

        Returns:
            Tensor, has the same shape as input tensor except along the given axis.

        Raises:
            ValueError: If the axis is out of range.
            TypeError: If arguments have types not specified above.

        Supported Platforms:
            ``Ascend`` ``GPU`` ``CPU``

        See also:
            :func:`mindspore.Tensor.reshape`: Give a new shape to a tensor without changing its data.

            :func:`mindspore.Tensor.resize`: Changes shape and size of tensor in-place.

        Examples:
            >>> import numpy as np
            >>> from mindspore import Tensor
            >>> x = Tensor(np.array(3))
            >>> print(x.repeat(4))
            [3 3 3 3]
            >>> x = Tensor(np.array([[1, 2],[3, 4]]))
            >>> print(x.repeat(2))
            [1 1 2 2 3 3 4 4]
            >>> print(x.repeat(3, axis=1))
            [[1 1 1 2 2 2]
            [3 3 3 4 4 4]]
            >>> print(x.repeat([1,2], axis=0))
            [[1 2]
            [3 4]
            [3 4]]
        """
        if not isinstance(repeats, (tuple, list)):
            repeats = (repeats,)
        for index, element in enumerate(repeats):
            if not isinstance(element, int):
                raise TypeError(f"For 'Tensor.repeat', each element in {repeats} should be int, but got "
                                f"{type(element)} at index {index}.")
        input_x = self
        if axis is None:
            input_x = self.ravel()
            axis = 0
        if axis is not None and not isinstance(axis, int):
            raise TypeError(f"For 'Tensor.repeat', the argument 'axis' should be int, but got {type(axis)}.")
        validator.check_axis_in_range(axis, input_x.ndim)
        axis = axis + input_x.ndim if axis < 0 else axis

        if len(repeats) == 1:
            repeats = repeats[0]
            if repeats == 0:
                return Tensor_(input_x.dtype, (0,))
            return tensor_operator_registry.get('repeat_elements')(input_x, repeats, axis)
        size = input_x.shape[axis]
        if len(repeats) != size:
            raise ValueError(f"For 'Tensor.repeat', the length of 'repeats' must be the same as the shape of the "
                             f"original tensor in the 'axis' dimension, but got the length of 'repeats' "
                             f"{len(repeats)}, the shape of the original tensor in the 'axis' dimension {size}.")
        subs = tensor_operator_registry.get('tensor_split')(input_x, size, axis)
        repeated_subs = []
        for sub, rep in zip(subs, repeats):
            if rep != 0:
                repeated_subs.append(tensor_operator_registry.get('repeat_elements')(sub, rep, axis))
        return tensor_operator_registry.get('concatenate')(axis)(repeated_subs)

    def repeat_interleave(self, repeats, dim=None):
        """
        For details, please refer to :func:`mindspore.ops.repeat_interleave`.
        """
        self._init_check()
        return tensor_operator_registry.get('repeat_interleave')(self, repeats, dim)

    def bernoulli(self, p=0.5, seed=-1):
        r"""
        For details, please refer to :func:`mindspore.ops.bernoulli`.
        """
        self._init_check()
        validator.check_is_int(seed, 'seed')
        return tensor_operator_registry.get('bernoulli')(self, p, seed)

    def random_categorical(self, num_sample, seed=0, dtype=mstype.int64):
        r"""
        For details, please refer to :func:`mindspore.ops.random_categorical`.
        """
        self._init_check()
        validator.check_is_int(num_sample, 'num_sample')
        validator.check_is_int(seed, 'seed')
        return tensor_operator_registry.get('random_categorical')(self, num_sample, seed, dtype)

    def masked_select(self, mask):
        """
        For details, please refer to :func:`mindspore.ops.masked_select`.
        """
        self._init_check()
        return tensor_operator_registry.get('masked_select')(self, mask)

    def gather_elements(self, dim, index):
        """
        For details, please refer to :func:`mindspore.ops.gather_elements`.
        """
        self._init_check()
        validator.check_value_type('index', index, (Tensor, Tensor_,), 'Tensor.gather_elements')
        return tensor_operator_registry.get('gather_elements')(self, dim, index)

    def nonzero(self):
        """
        For details, please refer to :func:`mindspore.ops.nonzero`.
        """
        self._init_check()
        return tensor_operator_registry.get('nonzero')(self)

    def svd(self, full_matrices=False, compute_uv=True):
        """
        For details, please refer to :func:`mindspore.ops.svd`.
        """
        svd_op = tensor_operator_registry.get("svd")
        if compute_uv:
            return svd_op(full_matrices, compute_uv)(self)

        s, _, _ = svd_op(full_matrices, compute_uv)(self)
        return s

    def hardshrink(self, lambd=0.5):
        r"""
        For details, please refer to :func:`mindspore.ops.hardshrink`.
        """
        self._init_check()
        return tensor_operator_registry.get('hardshrink')(lambd)(self)

    def heaviside(self, values):
        r"""
        For details, please refer to :func:`mindspore.ops.heaviside`.
        """
        self._init_check()
        return tensor_operator_registry.get('heaviside')(self, values)

    def hypot(self, other):
        r"""
        For details, please refer to :func:`mindspore.ops.hypot`.
        """
        self._init_check()
        return tensor_operator_registry.get('hypot')(self, other)

    def to_coo(self):
        """
        Convert a Tensor to COOTensor.

        Note:
            Only 2-D tensor is supported for now.

        Returns:
            COOTensor, a sparse representation of the original dense tensor, containing the following parts.

            - indices (Tensor): 2-D integer tensor, indicates the positions of `values` of the dense tensor.
            - values (Tensor): 1-D tensor, indicates the non-zero values of the dense tensor.
            - shape (tuple(int)): the shape of the COOTensor, is the same as the original dense tensor.

        Raises:
            ValueError: If input tensor is not 2-D.

        Supported Platforms:
            ``GPU``

        Examples:
            >>> import numpy as np
            >>> import mindspore
            >>> from mindspore import Tensor
            >>> x = Tensor(np.array([[1,  0], [-5, 0]]), mindspore.float32)
            >>> output = x.to_coo()
            >>> print(output.indices, output.values, output.shape)
            [[0 0]
             [1 0]] [ 1. -5.] (2, 2)

        """
        self._init_check()
        return tensor_operator_registry.get('dense_to_sparse_coo')(self)

    def to_csr(self):
        """
        Convert a Tensor to CSRTensor.

        Note:
            Only 2-D tensor is supported for now.

        Returns:
            CSRTensor, a sparse representation of the original dense tensor, containing the following parts.

            - indptr (Tensor): 1-D integer tensor, indicates the start and end point for `values` in each row.
            - indices (Tensor): 1-D integer tensor, indicates the column positions of all non-zero values of the input.
            - values (Tensor): 1-D tensor, indicates the non-zero values of the dense tensor.
            - shape (tuple(int)): the shape of the CSRTensor, is the same as the original dense tensor.

        Raises:
            ValueError: If input tensor is not 2-D.

        Supported Platforms:
            ``GPU``

        Examples:
            >>> import numpy as np
            >>> import mindspore
            >>> from mindspore import Tensor
            >>> x = Tensor(np.array([[1,  0], [-5, 0]]), mindspore.float32)
            >>> output = x.to_csr()
            >>> print(output.indptr, output.indices, output.values, output.shape)
            [0 1 2] [0 0] [ 1. -5.] (2, 2)
        """
        self._init_check()
        return tensor_operator_registry.get('dense_to_sparse_csr')(self)

    def unbind(self, dim=0):
        r"""
        For details, please refer to :func:`mindspore.ops.unbind`.
        """
        self._init_check()
        return tensor_operator_registry.get('unbind')(dim)(self)

    def unsorted_segment_min(self, segment_ids, num_segments):
        r"""
        For details, please refer to :func:`mindspore.ops.unsorted_segment_min`.
        """
        self._init_check()
        return tensor_operator_registry.get('unsorted_segment_min')(self, segment_ids, num_segments)

    def unsorted_segment_max(self, segment_ids, num_segments):
        r"""
        For details, please refer to :func:`mindspore.ops.unsorted_segment_max`.
        """
        self._init_check()
        return tensor_operator_registry.get('unsorted_segment_max')(self, segment_ids, num_segments)

    def unsorted_segment_prod(self, segment_ids, num_segments):
        r"""
        For details, please refer to :func:`mindspore.ops.unsorted_segment_prod`.
        """
        self._init_check()
        return tensor_operator_registry.get('unsorted_segment_prod')(self, segment_ids, num_segments)

    def unique_consecutive(self, return_idx=False, return_counts=False, axis=None):
        """
        For details, please refer to :func:`mindspore.ops.unique_consecutive`.
        """
        self._init_check()
        output, idx, counts = tensor_operator_registry.get("unique_consecutive")(return_idx, return_counts, axis)(self)
        if return_idx and return_counts:
            return output, idx, counts
        if return_idx:
            return output, idx
        if return_counts:
            return output, counts
        return output

    def unique_with_pad(self, pad_num):
        """
        For details, please refer to :func:`mindspore.ops.unique_with_pad`.
        """
        self._init_check()
        return tensor_operator_registry.get("unique_with_pad")()(self, pad_num)

    def diag(self):
        r"""
        For details, please refer to :func:`mindspore.ops.diag`.
        """
        self._init_check()
        return tensor_operator_registry.get('diag')()(self)

    def diagflat(self, offset=0):
        r"""
        For details, please refer to :func:`mindspore.ops.diagflat`.
        """
        self._init_check()
        return tensor_operator_registry.get('diagflat')(self, offset)

    def xdivy(self, y):
        r"""
        For details, please refer to :func:`mindspore.ops.xdivy`.
        """
        self._init_check()
        return tensor_operator_registry.get("xdivy")()(self, y)

    def split(self, split_size_or_sections, axis=0):
        """
        For details, please refer to :func:`mindspore.ops.split`.
        """
        return tensor_operator_registry.get('split')(self, split_size_or_sections, axis)

    def tensor_split(self, indices_or_sections, axis=0):
        """
        For details, please refer to :func:`mindspore.ops.tensor_split`.
        """
        self._init_check()
        return tensor_operator_registry.get('tensor_split')(self, indices_or_sections, axis)

    def vsplit(self, indices_or_sections):
        """
        For details, please refer to :func:`mindspore.ops.vsplit`.
        """

        self._init_check()
        return tensor_operator_registry.get('vsplit')(self, indices_or_sections)

    def hsplit(self, indices_or_sections):
        """
        For details, please refer to :func:`mindspore.ops.hsplit`.
        """
        self._init_check()
        return tensor_operator_registry.get('hsplit')(self, indices_or_sections)

    def dsplit(self, indices_or_sections):
        """
        For details, please refer to :func:`mindspore.ops.dsplit`.
        """
        self._init_check()
        return tensor_operator_registry.get('dsplit')(self, indices_or_sections)

    def xlogy(self, y):
        r"""
        For details, please refer to :func:`mindspore.ops.xlogy`.
        """
        return tensor_operator_registry.get("xlogy")()(self, y)

    def erf(self):
        r"""
        For details, please refer to :func:`mindspore.ops.erf`.
        """
        return tensor_operator_registry.get("erf")()(self)

    def erfc(self):
        r"""
        For details, please refer to :func:`mindspore.ops.erfc`.
        """
        return tensor_operator_registry.get("erfc")()(self)

    def tile(self, multiples):
        r"""
        For details, please refer to :func:`mindspore.ops.tile`.
        """
        return tensor_operator_registry.get('tile')()(self, multiples)

    def topk(self, k, dim=None, largest=True, sorted=True):
        r"""
        For details, please refer to :func:`mindspore.ops.topk`.
        """
        self._init_check()
        return tensor_operator_registry.get("topk")(self, k, dim, largest, sorted)

    def sigmoid(self):
        r"""
        For details, please refer to :func:`mindspore.ops.sigmoid`.
        """
        return tensor_operator_registry.get("sigmoid")()(self)

    def median(self, axis=-1, keepdims=False):
        r"""
        For details, please refer to :func:`mindspore.ops.median`.
        """
        self._init_check()
        validator.check_axis_in_range(axis, self.ndim)
        return tensor_operator_registry.get('median')(False, axis, keepdims)(self)

    def addmv(self, mat, vec, beta=1, alpha=1):
        r"""
        For details, please refer to :func:`mindspore.ops.addmv`.
        """
        self._init_check()
        return tensor_operator_registry.get('addmv')(self, mat, vec, beta=beta, alpha=alpha)

    def asinh(self):
        r"""
        For details, please refer to :func:`mindspore.ops.asinh`.
        """
        self._init_check()
        return tensor_operator_registry.get('asinh')(self)

    def arcsinh(self):
        r"""
        Alias for :func:`mindspore.Tensor.asinh`.
        """
        self._init_check()
        return tensor_operator_registry.get('arcsinh')(self)

    def atan(self):
        r"""
        For details, please refer to :func:`mindspore.ops.atan`.
        """
        self._init_check()
        return tensor_operator_registry.get('atan')(self)

    def atanh(self):
        r"""
        For details, please refer to :func:`mindspore.ops.atanh`.
        """
        self._init_check()
        return tensor_operator_registry.get('atanh')(self)

    def arctanh(self):
        r"""
        Alias for :func:`mindspore.Tensor.atanh`.
        """
        self._init_check()
        return tensor_operator_registry.get('arctanh')(self)

    def bmm(self, mat2):
        r"""
        For details, please refer to :func:`mindspore.ops.bmm`.
        """
        self._init_check()
        return tensor_operator_registry.get('bmm')(self, mat2)

    def to(self, dtype):
        r"""
        Performs tensor dtype conversion.

        Args:
            dtype (dtype.Number): The valid data type of the output tensor. Only constant value is allowed.

        Returns:
            Tensor, converted to the specified `dtype`.

        Raises:
            TypeError: If `dtype` is not a Number.

        Supported Platforms:
            ``Ascend`` ``GPU`` ``CPU``

        Examples:
            >>> input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
            >>> input_x = Tensor(input_np)
            >>> dtype = mindspore.int32
            >>> output = input_x.to(dtype)
            >>> print(output.dtype)
            Int32
        """
        self._init_check()
        return tensor_operator_registry.get('to')()(self, dtype)

    def bool(self):
        r"""
        Converts input tensor dtype to `bool`.
        If the value in tensor is zero, it will be `False`, otherwise it will be `True`.

        Returns:
            Tensor, converted to the `bool` dtype.

        Supported Platforms:
            ``Ascend`` ``GPU`` ``CPU``

        Examples:
            >>> input_x = Tensor(np.ones([2,2]), mindspore.float32)
            >>> output = input_x.bool()
            >>> print(output.dtype)
            Bool
        """
        self._init_check()
        return tensor_operator_registry.get('bool')()(self, mstype.bool_)

    def float(self):
        r"""
        Converts input tensor dtype to `float32`.

        Returns:
            Tensor, converted to the `float32` dtype.

        Supported Platforms:
            ``Ascend`` ``GPU`` ``CPU``

        Examples:
            >>> input_x = Tensor(np.ones([2,2]), mindspore.int32)
            >>> output = input_x.float()
            >>> print(output.dtype)
            Float32
        """
        self._init_check()
        return tensor_operator_registry.get('float')()(self, mstype.float32)

    def half(self):
        r"""
        Converts input tensor dtype to `float16`.

        Returns:
            Tensor, converted to the `float16` dtype.

        Supported Platforms:
            ``Ascend`` ``GPU`` ``CPU``

        Examples:
            >>> input_x = Tensor(np.ones([2,2]), mindspore.int32)
            >>> output = input_x.half()
            >>> print(output.dtype)
            Float16
        """
        self._init_check()
        return tensor_operator_registry.get('half')()(self, mstype.float16)

    def int(self):
        r"""
        Converts input tensor dtype to `int32`. If the value in tensor is float or half, the decimal will be discarded.

        Returns:
            Tensor, converted to the `int32` dtype.

        Supported Platforms:
            ``Ascend`` ``GPU`` ``CPU``

        Examples:
            >>> input_x = Tensor(np.ones([2,2]), mindspore.float32)
            >>> output = input_x.int()
            >>> print(output.dtype)
            Int32
        """
        self._init_check()
        return tensor_operator_registry.get('int')()(self, mstype.int32)

    def long(self):
        r"""
        Converts input tensor dtype to `int64`. If the value in tensor is float or half, the decimal will be discarded.

        Returns:
            Tensor, converted to the `int64` dtype.

        Supported Platforms:
            ``Ascend`` ``GPU`` ``CPU``

        Examples:
            >>> input_x = Tensor(np.ones([2,2]), mindspore.int32)
            >>> output = input_x.long()
            >>> print(output.dtype)
            Int64
        """
        self._init_check()
        return tensor_operator_registry.get('long')()(self, mstype.int64)

    def short(self):
        r"""
        Return a copy of the tensor, cast to int16 type, equivalent to self.astype(mstype.int16).
        If the value in tensor is float or half, the decimal will be discarded.
        For details, please refer to :func:`mindspore.Tensor.astype`.

        Returns:
            Tensor, converted to the `int16` dtype.

        Supported Platforms:
            ``Ascend`` ``GPU`` ``CPU``

        Examples:
            >>> import mindspore as ms
            >>> import numpy as np
            >>> x = ms.Tensor(np.array([1,2,3,4,5]), ms.int32)
            >>> output = x.short()
            >>> output
            Tensor(shape=[5], dtype=Int16, value= [1, 2, 3, 4, 5])
        """
        self._init_check()
        return tensor_operator_registry.get('cast')(self, mstype.int16)

    def cholesky(self, upper=False):
        r"""
        For details, please refer to :func:`mindspore.ops.cholesky`.
        """
        self._init_check()
        return tensor_operator_registry.get('cholesky')(upper=upper)(self)

    def cholesky_inverse(self, upper=False):
        r"""
        For details, please refer to :func:`mindspore.ops.cholesky_inverse`.
        """
        self._init_check()
        return tensor_operator_registry.get('cholesky_inverse')(upper=upper)(self)

    def conj(self):
        r"""
        For details, please refer to :func:`mindspore.ops.conj`.
        """
        self._init_check()
        return tensor_operator_registry.get('conj')(self)

    def cross(self, other, dim=None):
        r"""
        For details, please refer to :func:`mindspore.ops.cross`.
        """
        self._init_check()
        return tensor_operator_registry.get('cross')(self, other, dim)

    def erfinv(self):
        r"""
        For details, please refer to :func:`mindspore.ops.erfinv`.
        """
        self._init_check()
        return tensor_operator_registry.get('erfinv')(self)

    def less_equal(self, other):
        r"""
        For details, please refer to :func:`mindspore.ops.less_equal`.
        """
        self._init_check()
        return tensor_operator_registry.get('less_equal')(self, other)

    def lcm(self, other):
        r"""
        For details, please refer to :func:`mindspore.ops.lcm`.
        """
        self._init_check()
        return tensor_operator_registry.get('lcm')(self, other)

    def ldexp(self, other):
        r"""
        For details, please refer to :func:`mindspore.ops.ldexp`.
        """
        self._init_check()
        return tensor_operator_registry.get('ldexp')(self, other)

    def fold(self, output_size, kernel_size, dilation=1, padding=0, stride=1):
        r"""
        For details, please refer to :func:`mindspore.ops.fold`.
        """
        self._init_check()
        return tensor_operator_registry.get('fold')(self, output_size, kernel_size, dilation, padding, stride)

    def unfold(self, kernel_size, dilation=1, padding=0, stride=1):
        r"""
        For details, please refer to :func:`mindspore.ops.unfold`.
        """
        self._init_check()
        return tensor_operator_registry.get('unfold')(self, kernel_size, dilation, padding, stride)

    def expand(self, size):
        r"""
        For details, please refer to :func:`mindspore.ops.expand`.
        """
        self._init_check()
        return tensor_operator_registry.get('expand')(self, size)

    def cumprod(self, dim, dtype=None):
        r"""
        For details, please refer to :func:`mindspore.ops.cumprod`.
        """
        self._init_check()
        return tensor_operator_registry.get('cumprod')(self, dim, dtype)

    def multiply(self, value):
        r"""
        For details, please refer to :func:`mindspore.ops.multiply`.
        """
        self._init_check()
        return tensor_operator_registry.get('multiply')(self, value)

    def div(self, value, *, rounding_mode=None):
        r"""
        For details, please refer to :func:`mindspore.ops.div`.
        """
        self._init_check()
        return tensor_operator_registry.get('div')(self, value, rounding_mode=rounding_mode)

    def divide(self, value, *, rounding_mode=None):
        r"""
        Alias for :func:`mindspore.Tensor.div`.
        """
        self._init_check()
        return tensor_operator_registry.get('div')(self, value, rounding_mode=rounding_mode)

    def equal(self, other):
        r"""
        For details, please refer to :func:`mindspore.ops.equal`.
        """
        self._init_check()
        return tensor_operator_registry.get('equal')(self, other)

    def expm1(self):
        r"""
        For details, please refer to :func:`mindspore.ops.expm1`.
        """
        self._init_check()
        return tensor_operator_registry.get('expm1')(self)

    def index_add(self, dim, index, source, *, alpha=1):
        r"""
        For details, please refer to :func:`mindspore.ops.index_add`.
        """
        self._init_check()
        check_is_number(alpha, (int, float))
        source = tensor_operator_registry.get('__mul__')(source, alpha)
        return tensor_operator_registry.get('index_add')(self, indices=index, y=source, axis=dim)

    def greater(self, other):
        r"""
        For details, please refer to :func:`mindspore.ops.greater`.
        """
        self._init_check()
        return tensor_operator_registry.get('greater')(self, other)

    def greater_equal(self, other):
        r"""
        For details, please refer to :func:`mindspore.ops.greater_equal`.
        """
        self._init_check()
        return tensor_operator_registry.get('greater_equal')(self, other)

    def igamma(self, other):
        r"""
        For details, please refer to :func:`mindspore.ops.igamma`.
        """
        self._init_check()
        return tensor_operator_registry.get('igamma')(self, other)

    def igammac(self, other):
        r"""
        For details, please refer to :func:`mindspore.ops.igammac`.
        """
        self._init_check()
        return tensor_operator_registry.get('igammac')(self, other)

    def isinf(self):
        r"""
        For details, please refer to :func:`mindspore.ops.isinf`.
        """
        self._init_check()
        return tensor_operator_registry.get('isinf')(self)

    def isnan(self):
        r"""
        For details, please refer to :func:`mindspore.ops.isnan`.
        """
        self._init_check()
        return tensor_operator_registry.get('isnan')(self)

    def flip(self, dims):
        """
        For details, please refer to :func:`mindspore.ops.flip`.
        """
        return tensor_operator_registry.get('flip')(self, dims)

    def fliplr(self):
        """
        For details, please refer to :func:`mindspore.ops.fliplr`.
        """
        return tensor_operator_registry.get('fliplr')(self)

    def flipud(self):
        """
        For details, please refer to :func:`mindspore.ops.flipud`.
        """
        return tensor_operator_registry.get('flipud')(self)

    def is_floating_point(self):
        """
        For details, please refer to :func:`mindspore.ops.is_floating_point`.
        """
        return tensor_operator_registry.get('is_floating_point')(self)

    def is_signed(self):
        """
        Judge whether the data type of tensor is a signed data type.

        Returns:
            Bool. If the dtype of `self` is a signed data type, return True. Otherwise, return False.

        Supported Platforms:
            ``Ascend`` ``GPU`` ``CPU``

        Examples:
            >>> import mindspore as ms
            >>> from mindspore import Tensor
            >>> x = ms.Tensor([1, 2, 3], ms.int64)
            >>> y = ms.Tensor([1, 2, 3], ms.uint64)
            >>> output = x.is_signed()
            >>> output2 = y.is_signed()
            >>> print(output)
            True
            >>> print(output2)
            False
        """
        return self.dtype in mstype.signed_type

    def le(self, other):
        r"""
        For details, please refer to :func:`mindspore.ops.le`.
        """
        self._init_check()
        return tensor_operator_registry.get('le')(self, other)

    def less(self, other):
        r"""
        For details, please refer to :func:`mindspore.ops.less`.
        """
        self._init_check()
        return tensor_operator_registry.get('less')(self, other)

    def lt(self, other):
        """
        Alias for :func:`mindspore.Tensor.less`.
        """
        return self.less(other)

    def logical_and(self, other):
        r"""
        For details, please refer to :func:`mindspore.ops.logical_and`.
        """
        self._init_check()
        return tensor_operator_registry.get('logical_and')(self, other)

    def logical_not(self):
        r"""
        For details, please refer to :func:`mindspore.ops.logical_not`.
        """
        self._init_check()
        return tensor_operator_registry.get('logical_not')(self)

    def logical_or(self, other):
        r"""
        For details, please refer to :func:`mindspore.ops.logical_or`.
        """
        self._init_check()
        return tensor_operator_registry.get('logical_or')(self, other)

    def logical_xor(self, other):
        r"""
        For details, please refer to :func:`mindspore.ops.logical_xor`.
        """
        self._init_check()
        return tensor_operator_registry.get('logical_xor')(self, other)

    def lstsq(self, A):
        r"""
        Computes the solutions of the least squares and minimum norm problems of full-rank
        matrix `x` of size :math:`(m \times n)` and matrix `a` of size :math:`(m \times k)`.

        If :math:`m \geq n`, `lstsq` solves the least-squares problem:

        .. math::

           \begin{array}{ll}
           \min_y & \|xy-a\|_2.
           \end{array}

        If :math:`m < n`, `lstsq` solves the least-norm problem:

        .. math::

           \begin{array}{llll}
           \min_y & \|y\|_2 & \text{subject to} & xy = a.
           \end{array}

        Args:
            A (Tensor) - The m by k matrix equivalent to `a` in above.
                The input tensor whose data type is float16, float32 or float64.

        Returns:
            Tensor, the least squares or minimum norm problems solution, which has shape :math:`(n \times k)`.
            The data type is the same with `input`.

        Raises:
            TypeError: If `A` is not a Tensor.
            TypeError: If dtype of input tensor or `A` is not one of: float16, float32, float64.
            TypeError: If the dtypes of input tensor and `A` are not the same.
            ValueError: If the dimension of input tensor is not equal to 2.
            ValueError: If the dimension of `A` is not equal to 2 or 1.
            ValueError: If the length of input_dims[0] is not equal to the length of A_dims[0].

        Supported Platforms:
            ``CPU``

        Examples:
            >>> x = Tensor(np.array([[2,1,5],[3,5,1],[1,1,1]]),mindspore.float32)
            >>> a = Tensor(np.array([[10,5],[15,8],[7,4]]),mindspore.float32)
            >>> output = x.lstsq(a)
            >>> print(output)
            [[17.000002  11.000002 ]
             [-6.5000005 -4.500001 ]
             [-3.500002  -2.5000017]]
        """
        self._init_check()
        return tensor_operator_registry.get('lstsq')(self, A)

    @property
    def mH(self):
        r"""
        Accessing this property is equivalent to Calling self.adjoint().
        For details, please refer to :func:`mindspore.ops.adjoint`.
        """
        return self.adjoint()

    @property
    def mT(self):
        r"""
        Returns the Tensor that exchanges the last two dimensions.
        Accessing the attribute, x.mT, is equal to calling the method, x.swapaxes(-2, -1).
        For details, please refer to :func:`mindspore.Tensor.swapaxes`.
        """
        return self.swapaxes(-2, -1)

    def mvlgamma(self, p):
        r"""
        Computes the multivariate log-gamma function with dimension p element-wise.

        The following tex shows the mathematical calculation process of Mvlgamma:

        .. math::

            \log (\Gamma_{p}(a))=C+\sum_{i=1}^{p} \log (\Gamma(a-\frac{i-1}{2}))

        where :math:`C = \log(\pi) \times \frac{p(p-1)}{4}` and :math:`\Gamma(\cdot)` is the Gamma function.

        Args:
            p (int): The number of dimensions. And the value of `p` must be greater than or equal to 1.

        Returns:
            Tensor, has the same shape and type as input tensor.

        Raises:
            TypeError: If dtype of input tensor is neither float32 nor float64.
            TypeError: If `p` is not an int.
            ValueError: If `p` is not greater than or equal to 1.
            ValueError: If all elements of input tensor are not greater than (p-1)/2.

        Supported Platforms:
            ``GPU`` ``CPU``

        Examples:
            >>> x = Tensor(np.array([[3, 4, 5], [4, 2, 6]]), mindspore.float32)
            >>> y = x.mvlgamma(p=3)
            >>> print(y)
            [[2.694925 5.402975 9.140645]
             [5.402975 1.596312 13.64045]]
        """
        self._init_check()
        return tensor_operator_registry.get('mvlgamma')(self, p)

    def matmul(self, tensor2):
        r"""
        Returns the matrix product of two tensors.

        Note:
            Numpy arguments `out`, `casting`, `order`, `subok`, `signature`, and `extobj` are
            not supported.
            On CPU, the supported dtypes are np.float16 and np.float32.
            On GPU, the supported dtypes are np.float16 and np.float32.

        Args:
            tensor2 (Tensor): Second input tensor, scalar not allowed.
              The last dimension of input tensor must be the same size as the second last dimension of `tensor2`.
              And the shape of input tensor and tensor2 could be broadcast.

        Returns:
            Tensor or scalar, the matrix product of the inputs. This is a scalar only
            when both input tensor, `tensor2` are 1-d vectors.

        Raises:
            ValueError: If the last dimension of input tensor is not the same size as the
                second-to-last dimension of `tensor2`, or if a scalar value is passed in.
            ValueError: If the shape of input tensor and `tensor2` could not broadcast together.

        Supported Platforms:
            ``Ascend`` ``GPU`` ``CPU``

        Examples:
            >>> x = Tensor(np.arange(2*3*4).reshape(2, 3, 4), mindspore.float32)
            >>> y = Tensor(np.arange(4*5).reshape(4, 5), mindspore.float32)
            >>> output = x.matmul(y)
            >>> print(output)
            [[[  70.   76.   82.   88.   94.]
            [ 190.  212.  234.  256.  278.]
            [ 310.  348.  386.  424.  462.]]
            [[ 430.  484.  538.  592.  646.]
            [ 550.  620.  690.  760.  830.]
            [ 670.  756.  842.  928. 1014.]]]
        """
        self._init_check()
        return tensor_operator_registry.get('matmul')(self, tensor2)

    def inner(self, other):
        r"""
        For details, please refer to :func:`mindspore.ops.inner`.
        """
        self._init_check()
        return tensor_operator_registry.get('inner')(self, other)

    def matrix_power(self, n):
        r"""
        For details, please refer to :func:`mindspore.ops.matrix_power`.
        """
        self._init_check()
        return tensor_operator_registry.get('matrix_power')(self, n)

    def maximum(self, other):
        r"""
        For details, please refer to :func:`mindspore.ops.maximum`.
        """
        self._init_check()
        return tensor_operator_registry.get('maximum')(self, other)

    def mm(self, mat2):
        r"""
        For details, please refer to :func:`mindspore.ops.mm`.
        """
        self._init_check()
        return tensor_operator_registry.get('mm')(self, mat2)

    def msort(self):
        r"""
        For details, please refer to :func:`mindspore.ops.msort`.
        """
        self._init_check()
        return tensor_operator_registry.get('msort')(self)

    def mul(self, value):
        r"""
        For details, please refer to :func:`mindspore.ops.mul`.
        """
        self._init_check()
        return tensor_operator_registry.get('mul')(self, value)

    def nan_to_num(self, nan=0.0, posinf=None, neginf=None):
        """
        For details, please refer to :func:`mindspore.ops.nan_to_num`.
        """
        return tensor_operator_registry.get('nan_to_num')(self, nan, posinf, neginf)

    def neg(self):
        r"""
        Returns a tensor with negative values of the input tensor element-wise.

        .. math::

            out_{i} = - x_{i}

        Returns:
            Tensor, has the same shape and dtype as input.

        Supported Platforms:
            ``Ascend`` ``GPU`` ``CPU``

        Examples:
            >>> x = Tensor(np.array([1, 2, -1, 2, 0, -3.5]), mindspore.float32)
            >>> output = x.neg()
            >>> print(output)
            [-1.  -2.   1.  -2.   0.   3.5]
        """
        self._init_check()
        return tensor_operator_registry.get('neg')(self)

    def ne(self, other):
        r"""
        For details, please refer to :func:`mindspore.ops.ne`.
        """
        self._init_check()
        return tensor_operator_registry.get('ne')(self, other)

    def not_equal(self, other):
        r"""
        For details, please refer to :func:`mindspore.ops.not_equal`.
        """
        self._init_check()
        return tensor_operator_registry.get('not_equal')(self, other)

    def new_zeros(self, size, *, dtype=None):
        r"""
        Return a tensor of `size` filled with zeros.

        Args:
            size (Union[int, tuple, list]): An int, list or tuple of integers defining the output shape.

        Keyword Args:
            dtype (mindspore.dtype, optional): The desired dtype of the output tensor. If None, the returned tensor has
                thesame dtype as `self`. Default: None.

        Returns:
            Tensor, the shape and dtype is defined above and filled with zeros.

        Raises:
            TypeError: If `size` is not an int, list or tuple of integers.

        Supported Platforms:
            ``Ascend`` ``GPU`` ``CPU``

        Examples:
            >>> x = Tensor(np.array([1, 2, 3]), mindspore.float32)
            >>> output = x.new_zeros((2, 2))
            >>> print(output)
            [[0. 0.]
             [0. 0.]]
        """
        validator.check_value_type('size', size, [list, int, tuple], 'Tensor.new_zeros')
        if isinstance(size, list):
            size = tuple(size)
        self._init_check()
        _dtype = self.dtype if dtype is None else dtype
        return tensor_operator_registry.get('zeros')(size, _dtype)

    def new_ones(self, size, *, dtype=None):
        r"""
        Return a tensor of `size` filled with ones.

        Args:
            size (Union[int, tuple, list]): An int, list or tuple of integers defining the output shape.

        Keyword Args:
            dtype (mindspore.dtype, optional): The desired dtype of the output tensor. If None, the returned
                tensor has the same dtype as `self`. Default: None.

        Returns:
            Tensor, the shape and dtype is defined above and filled with ones.

        Raises:
            TypeError: If `size` is not an int, list or tuple of integers.

        Supported Platforms:
            ``Ascend`` ``GPU`` ``CPU``

        Examples:
            >>> x = Tensor(np.array([1, 2, 3]), mindspore.float32)
            >>> output = x.new_ones((2, 2))
            >>> print(output)
            [[1. 1.]
             [1. 1.]]
        """
        validator.check_value_type('size', size, [list, int, tuple], 'Tensor.new_zeros')
        if isinstance(size, list):
            size = tuple(size)
        self._init_check()
        _dtype = self.dtype if dtype is None else dtype
        return tensor_operator_registry.get('ones')(size, _dtype)

    def sign(self):
        r"""
        For details, please refer to :func:`mindspore.ops.sign`.
        """
        self._init_check()
        return tensor_operator_registry.get('sign')(self)

    def signbit(self):
        """
        For details, please refer to :func:`mindspore.ops.signbit`.
        """
        self._init_check()
        return tensor_operator_registry.get('signbit')(self)

    def sgn(self):
        """
        For details, please refer to :func:`mindspore.ops.sgn`.
        """
        self._init_check()
        return tensor_operator_registry.get('sgn')(self)

    def sin(self):
        r"""
        For details, please refer to :func:`mindspore.ops.sin`.
        """
        self._init_check()
        return tensor_operator_registry.get('sin')(self)

    def sinc(self):
        r"""
        For details, please refer to :func:`mindspore.ops.sinc`.
        """
        self._init_check()
        return tensor_operator_registry.get('sinc')(self)

    def sinh(self):
        r"""
        For details, please refer to :func:`mindspore.ops.sinh`.
        """
        self._init_check()
        return tensor_operator_registry.get('sinh')(self)

    def sort(self, dim=-1, descending=False):
        r"""
        Sorts the elements of the input tensor along a given dimension in ascending order by value.

        Args:
            dim (int, optional): The dimension to sort along. Default: -1.
            descending (bool, optional): Controls the sorting order. If descending is True, then the elements
                are sorted in descending order by value. Default: False.

        Returns:
            y1 (Tensor): A tensor whose values are the sorted values, with the same shape and dtype as input.
            y2 (Tensor): The indices of the elements in the original input tensor. Tensor dtype is int32.

        Raises:
            TypeError: If dtype of `dim` is not int.
            TypeError: If dtype of `descending` is not bool.
            TypeError: If dtype of input tensor is neither float16 nor float32.
            ValueError: If `dim` is not in range of [-len(x.shape), len(x.shape)).

        Supported Platforms:
            ``Ascend`` ``GPU`` ``CPU``

        Examples:
            >>> x = Tensor(np.array([[8, 2, 1], [5, 9, 3], [4, 6, 7]]), mindspore.float16)
            >>> output = x.sort()
            >>> print(output)
            (Tensor(shape=[3, 3], dtype=Float16, value=
            [[ 1.0000e+00,  2.0000e+00,  8.0000e+00],
             [ 3.0000e+00,  5.0000e+00,  9.0000e+00],
             [ 4.0000e+00,  6.0000e+00,  7.0000e+00]]), Tensor(shape=[3, 3], dtype=Int32, value=
            [[2, 1, 0],
             [2, 0, 1],
             [0, 1, 2]]))
        """
        self._init_check()
        return tensor_operator_registry.get('sort')(self, axis=dim, descending=descending)

    def argsort(self, axis=-1, descending=False):
        """
        For details, please refer to :func:`mindspore.ops.argsort`.
        """
        self._init_check()
        return tensor_operator_registry.get('argsort')(self, axis, descending)

    def trunc(self):
        r"""
        For details, please refer to :func:`mindspore.ops.trunc`.
        """
        self._init_check()
        return tensor_operator_registry.get('trunc')(self)

    def where(self, condition, y):
        r"""
        For details, please refer to :func:`mindspore.ops.where`.
        """
        self._init_check()
        return tensor_operator_registry.get('where')(condition, self, y)

    def imag(self):
        r"""
        Returns a new tensor containing imaginary value of the input tensor.
        If input tensor is real, it will return zeros.

        Returns:
            Tensor, the shape is the same as the input tensor.

        Supported Platforms:
            ``GPU`` ``CPU``

        Examples:
            >>> x = Tensor(np.asarray(np.complex(1.3 + 0.4j)), mindspore.complex64)
            >>> output = x.imag()
            >>> print(output)
            0.4
        """
        self._init_check()
        return tensor_operator_registry.get('imag')(self)


def _vm_compare(*args):
    """Implement `vm_compare` for tensor."""
    obj_str = args[-1]
    if obj_str == "shape":
        fn = getattr(args[0].asnumpy(), obj_str)
        return fn
    if len(args) == 2:
        fn = getattr(args[0].asnumpy(), obj_str)
        return Tensor(fn())
    if isinstance(args[0], Tensor):
        fn = getattr(args[0].asnumpy(), obj_str)
        y = args[1].asnumpy() if isinstance(args[1], Tensor) else args[1]
    else:
        obj_str = "__r" + obj_str[2:]
        fn = getattr(args[1].asnumpy(), obj_str)
        y = args[0]
    return Tensor(np.array(fn(y)))


def _check_tensor_input(input_data=None, dtype=None, shape=None, init=None):
    """Check the tensor input."""
    if input_data is not None and shape is not None:
        raise ValueError(f"When initializing a tensor with 'input_data', 'shape' should be set to None."
                         f"But got shape: {shape}.")

    if init is not None and (shape is None or dtype is None):
        raise ValueError("init, dtype and shape must have values at the same time.")

    if input_data is not None:
        if isinstance(input_data, np.ndarray) and input_data.ndim > 1 and input_data.size == 0:
            raise ValueError("input_data can not contain zero dimension.")
        if isinstance(input_data, (tuple, list)) and np.array(input_data).ndim > 1 \
                and np.array(input_data).size == 0:
            raise ValueError("input_data can not contain zero dimension.")

    if shape is not None and not (hasattr(init, "__enable_zero_dim__") and init.__enable_zero_dim__) and 0 in shape:
        raise ValueError("Shape can not contain zero value.")


def _check_tensor_dynamic_shape(dtype=None, shape=None, init=None):
    """Check if the tensor has dynamic shape."""
    shape_list = list(shape)
    if len(shape_list) >= 1:
        shape_replaced_list = [-1 if i is None else i for i in shape_list]
        if isinstance(shape, tuple):
            shape = tuple(shape_replaced_list)
        if isinstance(shape, list):
            shape = shape_replaced_list
    if is_shape_unknown(shape) and (dtype is None or init is not None):
        raise ValueError("If setting dynamic shape, dtype must not be None, init must be None")
    return shape


def _check_astype_and_convert(dtype):
    """Check whether dtype is a valid input, and convert to mstype"""
    all_types = mstype.__dtype__ + ["int", "float", "bool"]
    if isinstance(dtype, str):
        if dtype.lower() not in all_types:
            raise TypeError(f"For Tensor.astype, the string input type must be one of {all_types}, "
                            f"but got '{dtype}'.")
        dtype = mstype.pytype_to_dtype(np.dtype(dtype.lower()))
    elif isinstance(dtype, type):
        dtype = mstype.pytype_to_dtype(dtype)
    elif dtype not in mstype.number_type + (mstype.bool_,):
        raise TypeError(
            f"For Tensor.astype, the input type must be one of {list(mstype.number_type + (mstype.bool_,) + np_types)},"
            f" but got '{dtype}'.")
    return dtype


tensor_operator_registry.register('vm_compare', _vm_compare)
