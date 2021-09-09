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
"""Tensor implementation."""
import numbers
import numpy as np

from mindspore import log as logger
from mindspore.communication.management import get_rank, get_group_size
from . import dtype as mstype
from ._register_for_tensor import tensor_operator_registry
from .._c_expression import Tensor as Tensor_
from .._c_expression import PynativeExecutor_
from .._checkparam import Validator as validator

__all__ = ['Tensor', 'RowTensor', 'SparseTensor']
np_types = (np.int8, np.int16, np.int32, np.int64,
            np.uint8, np.uint16, np.uint32, np.uint64, np.float16,
            np.float32, np.float64, np.bool_, np.complex64, np.complex128)


class Tensor(Tensor_):
    """
    Tensor is used for data storage.

    Tensor inherits tensor object in C++.
    Some functions are implemented in C++ and some functions are implemented in Python.

    Args:
        input_data (Union[Tensor, float, int, bool, tuple, list, numpy.ndarray]): Input data of the tensor.
        dtype (:class:`mindspore.dtype`): Input data should be None, bool or numeric type defined in `mindspore.dtype`.
            The argument is used to define the data type of the output tensor. If it is None, the data type of the
            output tensor will be the same as the `input_data`. Default: None.
        shape (Union[tuple, list, int]): A list of integers, a tuple of integers or an integer as the shape of
            output. If `input_data` is available, `shape` doesn't need to be set. Default: None.
        init (Initializer): the information of init data.
            'init' is used for delayed initialization in parallel mode. Usually, it is not recommended to use
            'init' interface to initialize parameters in other conditions. If 'init' interface is used to initialize
            parameters, the `Tensor.init_data` API needs to be called to convert `Tensor` to the actual data.

    Outputs:
        Tensor. If `dtype` and `shape` are not set, return a tensor with the same dtype and shape as `input_data`.
        If `dtype` or `shape` is set, the dtype or shape of the output Tensor is consistent with the setting.

    Examples:
        >>> import numpy as np
        >>> import mindspore as ms
        >>> from mindspore import Tensor
        >>> from mindspore.common.initializer import One
        >>> # initialize a tensor with input data
        >>> t1 = Tensor(np.zeros([1, 2, 3]), ms.float32)
        >>> assert isinstance(t1, Tensor)
        >>> assert t1.shape == (1, 2, 3)
        >>> assert t1.dtype == ms.float32
        >>>
        >>> # initialize a tensor with a float scalar
        >>> t2 = Tensor(0.1)
        >>> assert isinstance(t2, Tensor)
        >>> assert t2.dtype == ms.float64
        ...
        >>> # initialize a tensor with init
        >>> t3 = Tensor(shape = (1, 3), dtype=ms.float32, init=One())
        >>> assert isinstance(t3, Tensor)
        >>> assert t3.shape == (1, 3)
        >>> assert t3.dtype == ms.float32
    """

    def __init__(self, input_data=None, dtype=None, shape=None, init=None):
        self.init_finished = False
        # If input data is numpy number, convert it to np array
        if isinstance(input_data, np_types):
            input_data = np.array(input_data)

        if isinstance(shape, numbers.Number):
            shape = (shape,)

        _check_tensor_input(input_data, dtype, shape, init)

        # If input_data is tuple/list/numpy.ndarray, it's support in check_type method.
        if init is None:
            validator.check_value_type('input_data', input_data,
                                       (Tensor_, np.ndarray, list, tuple, float, int, bool, complex), 'Tensor')
            valid_dtypes = (np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64,
                            np.float16, np.float32, np.float64, np.bool_, np.str_, np.complex64, np.complex128)
            if isinstance(input_data, np.ndarray) and input_data.dtype not in valid_dtypes and \
                input_data.dtype.kind != 'U':  # Support dtype np.str_
                raise TypeError(f"For Tensor, the input_data is a numpy array, "
                                f"but it's data type: {input_data.dtype} is not in supported list:\
                                {list(i.__name__ for i in valid_dtypes)}.")
            if isinstance(input_data, (tuple, list)):
                if np.array(input_data).dtype not in valid_dtypes:
                    raise TypeError(f"For Tensor, the input_data is {input_data} that contain unsupported element.")
            if dtype is not None:
                validator.check_type_name('dtype', dtype, mstype.number_type + (mstype.bool_, mstype.string), "Tensor")

            if isinstance(input_data, np.ndarray) and (not input_data.flags['FORC']):
                input_data = np.ascontiguousarray(input_data)
            if dtype is None:
                Tensor_.__init__(self, input_data)
            else:
                Tensor_.__init__(self, input_data, dtype)
        else:
            Tensor_.__init__(self, dtype, shape)
        self._virtual_flag = False
        self.init = init
        self.init_finished = True

    def __deepcopy__(self, memodict):
        new_obj = Tensor(self)
        new_obj.init = self.init
        new_obj._virtual_flag = self._virtual_flag  # pylint:disable=w0212
        return new_obj

    def __repr__(self):
        if self.init_finished:
            Tensor_.data_sync(self, False)
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

    def __bool__(self):
        data = self.asnumpy()
        if data.shape == ():
            return bool(data)
        if data.shape == (1,):
            return bool(data[0])
        raise ValueError("The truth value of an array with several elements is ambiguous.")

    def __index__(self):
        data = self.asnumpy()
        if not (data.dtype == "int8"
                or data.dtype == "int16"
                or data.dtype == "int32"
                or data.dtype == "int64"
                or data.dtype == "bool"):
            raise ValueError("Only integer tensors of a single element can be converted to an index.")
        if data.shape == ():
            return int(data)
        if data.shape == (1,):
            return int(data[0])
        raise ValueError("Only integer tensors of a single element can be converted to an index.")

    def __pos__(self):
        return self

    def __add__(self, other):
        return tensor_operator_registry.get('__add__')(self, other)

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
        if out:
            return out[0]
        raise TypeError("Not support len of a 0-D tensor")

    def __str__(self):
        if self.dtype == mstype.type_none:
            return "Unknown Tensor type!"
        return str(self.asnumpy())

    @property
    def shape(self):
        """Returns the shape of the tensor as a tuple."""
        return self._shape

    @property
    def dtype(self):
        """Return the dtype of the tensor (:class:`mindspore.dtype`)."""
        return self._dtype

    @property
    def size(self):
        """Returns the total number of elements in tensor."""
        return self._size

    @property
    def ndim(self):
        """Return the number of tensor dimensions."""
        return len(self._shape)

    @property
    def has_init(self):
        """tensor is inited."""
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

    @property
    def virtual_flag(self):
        """Used to mark whether the tensor is virtual. If the tensor is virtual, return True."""
        return self._virtual_flag

    @virtual_flag.setter
    def virtual_flag(self, value):
        """The setter of virtual_flag."""
        if not isinstance(value, bool):
            raise TypeError("virtual_flag must be bool.")
        self._virtual_flag = value

    @staticmethod
    def from_numpy(array):
        """
        Convert numpy array to Tensor without copy data.

        Args:
            array (numpy.array): The input array.

        Returns:
            Tensor, has the same data type as input array.
        """
        return Tensor(Tensor_.from_numpy(array))

    def assign_value(self, value):
        PynativeExecutor_.get_instance().execute_all_task()
        self.assign_value_cpp(value)
        return self

    def item(self, index=None):
        """
        Getitem from the Tensor with the index.

        Note:
            Tensor.item returns a Tensor scalar instead of a Python scalar.

        Args:
            index (Union[None, int, tuple(int)]): The index in Tensor. Default: None.

        Returns:
            A Tensor scalar, dtype is the same with the original Tensor.

        Raises:
            ValueError: If the length of the `index` is not euqal to self.ndim.

        Supported Platforms:
            ``Ascend`` ``GPU``

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
        Insert scalar into a tensor (scalar is cast to tensor’s dtype, if possible).

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
            A new Tensor, with value set by :math:`tensor[args] = item`.

        Raises:
            ValueError: If the length of the first argument is not euqal to self.ndim.
            IndexError: If only one argument is provided, and the original Tensor is not scalar.

        Supported Platforms:
            ``Ascend`` ``GPU``

        Examples:
            >>> import numpy as np
            >>> from mindspore import Tensor
            >>> x = Tensor(np.array([[1,2,3],[4,5,6]], dtype=np.float32))
            >>> x = x.itemset((0,1), 4)
            >>> print(x)
            [[1. 4. 3.]
            [4. 5. 6.]]
        """
        output = tensor_operator_registry.get('itemset')(self, *args)
        return output

    def asnumpy(self):
        """Convert tensor to numpy array."""
        self._init_check()
        PynativeExecutor_.get_instance().execute_all_task()
        return Tensor_.asnumpy(self)

    def flush_from_cache(self):
        """Flush cache data to host if tensor is cache enable."""
        self._init_check()
        Tensor_._flush_from_cache(self)

    def all(self, axis=(), keep_dims=False):
        """
        Check all array elements along a given axis evaluate to True.

        Args:
            axis (Union[None, int, tuple(int)): Dimensions of reduction,
                when the axis is None or empty tuple, reduce all dimensions. Default: ().
            keep_dims (bool): Whether to keep the reduced dimensions. Default: False.

        Returns:
            Tensor, if all array elements along the given axis evaluate to True, its value is True,
            otherwise its value is False. If the axis is None or empty tuple, reduce all dimensions.

        Supported Platforms:
            ``Ascend`` ``GPU`` ``CPU``

        Examples:
            >>> from mindspore import Tensor
            >>> a = Tensor([True, True, False])
            >>> output = a.all()
            >>> print(output)
            False
        """

        self._init_check()
        if axis is None:
            axis = ()
        return tensor_operator_registry.get('all')(keep_dims)(self, axis)

    def any(self, axis=(), keep_dims=False):
        """
        Check any array element along a given axis evaluate to True.

        Args:
            axis (Union[None, int, tuple(int)): Dimensions of reduction,
                when the axis is None or empty tuple, reduce all dimensions. Default: ().
            keep_dims (bool): Whether to keep the reduced dimensions. Default: False.

        Returns:
            Tensor, if any array element along the given axis evaluates to True, its value is True,
            otherwise its value is False. If the axis is None or empty tuple, reduce all dimensions.

        Supported Platforms:
            ``Ascend`` ``GPU`` ``CPU``

        Examples:
            >>> from mindspore import Tensor
            >>> a = Tensor([True, True, False])
            >>> output = a.any()
            >>> print(output)
            True
        """

        self._init_check()
        if axis is None:
            axis = ()
        return tensor_operator_registry.get('any')(keep_dims)(self, axis)

    def view(self, *shape):
        r"""
        Reshape the tensor according to the input shape.

        Args:
            shape (Union[tuple(int), int]): Dimension of the output tensor.

        Returns:
            Tensor, has the same dimension as the input shape.
        """
        self._init_check()
        if not shape:
            raise ValueError("The shape variable should not be empty")
        if isinstance(shape[0], tuple):
            if len(shape) != 1:
                raise ValueError(f"Only one tuple is needed, but got {shape}")
            shape = shape[0]
        return tensor_operator_registry.get('reshape')()(self, shape)

    def expand_as(self, x):
        """
        Expand the dimension of target tensor to the dimension of input tensor.

        Args:
            x (Tensor): The input tensor. The shape of input tensor must obey
                the broadcasting rule.

        Returns:
            Tensor, has the same dimension as input tensor.
        """
        self._init_check()
        return tensor_operator_registry.get('broadcast_to')(x.shape)(self)

    def abs(self):
        """
        Return absolute value element-wisely.

        Returns:
            Tensor, with absolute value element-wisely.

        Supported Platforms:
            ``Ascend`` ``GPU`` ``CPU``

        Examples:
            >>> from mindspore import Tensor
            >>> a = Tensor([1.1, -2.1]).astype("float32")
            >>> output = a.abs()
            >>> print(output)
            [1.1 2.1]
        """
        self._init_check()
        return tensor_operator_registry.get('abs')()(self)

    def mean(self, axis=(), keep_dims=False):
        """
        Reduce a dimension of a tensor by averaging all elements in the dimension.

        Args:
            axis (Union[None, int, tuple(int), list(int)]): Dimensions of reduction,
                when the axis is None or empty tuple, reduce all dimensions. Default: ().
            keep_dims (bool): Whether to keep the reduced dimensions. Default: False.

        Returns:
            Tensor, has the same data type as input tensor.

        Supported Platforms:
            ``Ascend`` ``GPU`` ``CPU``

        Examples:
            >>> import numpy as np
            >>> from mindspore import Tensor
            >>> input_x = Tensor(np.array([1, 2, 3], dtype=np.float32))
            >>> output = input_x.mean()
            >>> print(output)
            2.0
        """
        self._init_check()
        if axis is None:
            axis = ()
        return tensor_operator_registry.get('mean')(keep_dims)(self, axis)

    def transpose(self, *axes):
        r"""
        Return a view of the tensor with axes transposed.

        - For a 1-D tensor this has no effect, as a transposed vector is simply the same vector.
        - For a 2-D tensor, this is a standard matrix transpose.
        - For an n-D tensor, if axes are given, their order indicates how the axes are permuted.

        If axes are not provided and ``tensor.shape = (i[0], i[1],...i[n-2], i[n-1])``,
        then ``tensor.transpose().shape = (i[n-1], i[n-2], ... i[1], i[0])``.

        Args:
            axes(Union[None, tuple(int), list(int), int], optional): If axes is None or
                blank, the method will reverse the order of the axes. If axes is tuple(int)
                or list(int), tensor.transpose() will transpose the tensor to the new axes order.
                If axes is int, this form is simply intended as a convenience alternative to the
                tuple/list form.

        Returns:
            Tensor, has the same dimension as input tensor, with axes suitably permuted.

        Raises:
            TypeError: If input arguments have types not specified above.
            ValueError: If the number of `axes` is not euqal to a.ndim.

        Supported Platforms:
            ``Ascend`` ``GPU`` ``CPU``

        Examples:
            >>> import numpy as np
            >>> from mindspore import Tensor
            >>> x = Tensor(np.ones((1,2,3), dtype=np.float32))
            >>> x = x.transpose()
            >>> print(x.shape)
            (3, 2, 1)
        """
        self._init_check()
        perm = validator.check_transpose_axis(axes, self.ndim)
        return tensor_operator_registry.get('transpose')()(self, perm)

    def reshape(self, *shape):
        """
        Give a new shape to a tensor without changing its data.

        Args:
            shape(Union[int, tuple(int), list(int)]): The new shape should be compatible
                with the original shape. If an integer, then the result will be a 1-D
                array of that length. One shape dimension can be -1. In this case, the
                value is inferred from the length of the array and remaining dimensions.

        Returns:
            Tensor, with new specified shape.

        Raises:
            TypeError: If new_shape is not integer, list or tuple, or `x` is not tensor.
            ValueError: If new_shape is not compatible with the original shape.

        Supported Platforms:
            ``Ascend`` ``GPU`` ``CPU``

        Examples:
            >>> from mindspore import Tensor
            >>> from mindspore import dtype as mstype
            >>> x = Tensor([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]], dtype=mstype.float32)
            >>> output = x.reshape((3, 2))
            >>> print(output)
            [[-0.1  0.3]
            [ 3.6  0.4]
            [ 0.5 -3.2]]
        """
        self._init_check()
        new_shape = validator.check_reshape_shp(shape)
        return tensor_operator_registry.get('reshape')()(self, new_shape)

    def ravel(self):
        """
        Return a contiguous flattened tensor.

        Returns:
            Tensor, a 1-D tensor, containing the same elements of the input.

        Supported Platforms:
            ``Ascend`` ``GPU`` ``CPU``

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

    def flatten(self, order='C'):
        r"""
        Return a copy of the tensor collapsed into one dimension.

        Args:
            order (str, optional): Can choose between 'C' and 'F'. 'C' means to
                flatten in row-major (C-style) order. 'F' means to flatten in column-major
                (Fortran-style) order. Only 'C' and 'F' are supported. Default: 'C'.

        Returns:
            Tensor, has the same data type as input.

        Supported Platforms:
            ``Ascend`` ``GPU`` ``CPU``

        Raises:
            TypeError: If `order` is not string type.
            ValueError: If `order` is string type, but not 'C' or 'F'.

        Examples:
            >>> import numpy as np
            >>> from mindspore import Tensor
            >>> x = Tensor(np.ones((2,3,4), dtype=np.float32))
            >>> output = x.flatten()
            >>> print(output.shape)
            (24,)
        """
        self._init_check()
        reshape_op = tensor_operator_registry.get('reshape')()
        trans_op = tensor_operator_registry.get('transpose')()

        order = validator.check_flatten_order(order)
        if order == 'C':
            return reshape_op(self, (-1,))

        perm = tuple(range(self.ndim-1, -1, -1))
        return reshape_op(trans_op(self, perm), (-1,))

    def swapaxes(self, axis1, axis2):
        """
        Interchange two axes of a tensor.

        Args:
            axis1 (int): First axis.
            axis2 (int): Second axis.

        Returns:
            Transposed tensor, has the same data type as the input.

        Raises:
            TypeError: If `axis1` or `axis2` is not integer.
            ValueError: If `axis1` or `axis2` is not in the range of :math:`[-ndim, ndim-1]`.

        Supported Platforms:
            ``Ascend`` ``GPU`` ``CPU``

        Examples:
            >>> import numpy as np
            >>> from mindspore import Tensor
            >>> x = Tensor(np.ones((2,3,4), dtype=np.float32))
            >>> output = x.swapaxes(0, 2)
            >>> print(output.shape)
            (4,3,2)
        """
        self._init_check()
        axis1, axis2 = validator.check_swapaxes_axis((axis1, axis2), self.ndim)

        if axis1 == axis2:
            return self
        if axis1 > axis2:
            axis1, axis2 = axis2, axis1

        perm = tuple(range(0, self.ndim))
        if axis2 + 1 < self.ndim:
            new_perm = perm[0:axis1] + perm[axis2:axis2+1] + \
                perm[axis1+1:axis2] + perm[axis1:axis1+1] + perm[axis2+1:]
        else:
            new_perm = perm[0:axis1] + perm[axis2:axis2+1] + \
                perm[axis1+1:axis2] + perm[axis1:axis1+1]

        return tensor_operator_registry.get('transpose')()(self, new_perm)

    def squeeze(self, axis=None):
        """
        Remove single-dimensional entries from the shape of a tensor.

        Args:
            axis (Union[None, int, list(int), tuple(int)], optional): Selects a subset of the entries of
                length one in the shape. If an axis is selected with shape entry greater than one,
                an error is raised. Default is None.

        Returns:
            Tensor, with all or a subset of the dimensions of length 1 removed.

        Raises:
            TypeError: If input arguments have types not specified above.
            ValueError: If specified axis has shape entry :math:`> 1`.

        Supported Platforms:
            ``Ascend`` ``GPU`` ``CPU``

        Examples:
            >>> import numpy as np
            >>> from mindspore import Tensor
            >>> x = Tensor(np.ones((1,2,2,1), dtype=np.float32))
            >>> x = x.squeeze()
            >>> print(x.shape)
            (2, 2)
        """
        self._init_check()
        if axis is None:
            return tensor_operator_registry.get('squeeze')(self)
        new_shape = validator.prepare_shape_for_squeeze(self.shape, axis)
        return tensor_operator_registry.get('reshape')()(self, new_shape)

    def astype(self, dtype, copy=True):
        """
        Return a copy of the tensor, cast to a specified type.

        Args:
            dtype (Union[:class:`mindspore.dtype`, str]): Designated tensor dtype, can be in format
                of :class:`mindspore.dtype.float32` or `float32`.
                Default: :class:`mindspore.dtype.float32`.
            copy (bool, optional): By default, astype always returns a newly allocated
                tensor. If this is set to false, the input tensor is returned instead
                of a copy if possible. Default: True.

        Returns:
            Tensor, with the designated dtype.

        Raises:
            TypeError: If `dtype` has types not specified above, or values cannot be understood.

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
        dtype = validator.check_astype_dtype(dtype)
        if not copy and dtype == self.dtype:
            return self
        return tensor_operator_registry.get('cast')(self, dtype)

    def argmax(self, axis=None):
        """
        Return the indices of the maximum values along an axis.

        Args:
            axis (int, optional): By default, the index is into
                the flattened tensor, otherwise along the specified axis.

        Returns:
            Tensor, indices into the input tensor. It has the same
            shape as self.shape with the dimension along axis removed.

        Raises:
            ValueError: if the axis is out of range.

        Supported Platforms:
            ``Ascend`` ``GPU`` ``CPU``

        Examples:
            >>> import numpy as np
            >>> from mindspore import Tensor
            >>> a = Tensor(np.arange(10, 16).reshape(2, 3).astype("float32"))
            >>> print(a.argmax())
            5
        """
        # P.Argmax only supports float
        a = self.astype(mstype.float32)
        if axis is None:
            a = a.ravel()
            axis = 0
        else:
            axis = validator.check_axis_in_range(axis, a.ndim)
        return tensor_operator_registry.get('argmax')(axis)(a)

    def argmin(self, axis=None):
        """
        Return the indices of the minimum values along an axis.

        Args:
            axis (int, optional): By default, the index is into
                the flattened tensor, otherwise along the specified axis.

        Returns:
            Tensor, indices into the input tensor. It has the same
            shape as self.shape with the dimension along axis removed.

        Raises:
            ValueError: if the axis is out of range.

        Supported Platforms:
            ``Ascend`` ``GPU`` ``CPU``

        Examples:
            >>> import numpy as np
            >>> from mindspore import Tensor
            >>> a = Tensor(np.arange(10, 16).reshape(2, 3).astype("float32"))
            >>> print(a.argmin())
            0
        """
        # P.Argmax only supports float
        a = self.astype(mstype.float32)
        if axis is None:
            a = a.ravel()
            axis = 0
        else:
            axis = validator.check_axis_in_range(axis, a.ndim)
        # P.Argmin is currently not supported
        return tensor_operator_registry.get('argmax')(axis)(tensor_operator_registry.get('__neg__')(a))

    def cumsum(self, axis=None, dtype=None):
        """
        Return the cumulative sum of the elements along a given axis.

        Note:
            If ``self.dtype`` is :class:`int8`, :class:`int16` or :class:`bool`, the result
            `dtype` will be elevated to :class:`int32`, :class:`int64` is not supported.

        Args:
            axis (int, optional): Axis along which the cumulative sum is computed. The
                default (None) is to compute the cumsum over the flattened array.
            dtype (:class:`mindspore.dtype`, optional): If not specified, stay the same as original,
                tensor, unless it has an integer dtype with a precision less than :class:`float32`.
                In that case, :class:`float32` is used. Default: None.

        Raises:
            ValueError: if the axis is out of range.

        Returns:
            Tensor.

        Supported Platforms:
            ``Ascend`` ``GPU`` ``CPU``

        Examples:
            >>> import numpy as np
            >>> from mindspore import Tensor
            >>> a = Tensor(np.ones((3,3)).astype("float32"))
            >>> output = a.cumsum(axis=0)
            >>> print(output)
            [[1. 1. 1.]
            [2. 2. 2.]
            [3. 3. 3.]]
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
        logical_not_op = tensor_operator_registry.get('logical_not')()
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
            axis (Union[None, int, tuple of ints], optional): Axis or
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
                A boolean array which is broadcasted to match the dimensions of array,
                and selects elements to include in the reduction. If non-default value
                is passed, initial must also be provided. Default: True.

        Returns:
            Tensor or scalar, maximum of input tensor. If `axis` is None, the result is a scalar
            value. If `axis` is given, the result is an array of dimension ``self.ndim - 1``.

        Raises:
            TypeError: if arguments have types not specified above.

        Supported Platforms:
            ``Ascend`` ``GPU`` ``CPU``

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
        return reduce_(self, reduce_max(keepdims), cmp_fn=maximum(), axis=axis, keepdims=keepdims,
                       initial=initial, where=where)

    def min(self, axis=None, keepdims=False, initial=None, where=True):
        """
        Return the minimum of a tensor or minimum along an axis.

        Args:
            axis (Union[None, int, tuple of ints], optional): Axis or
                axes along which to operate. By default, flattened input is used. If
                this is a tuple of ints, the minimum is selected over multiple axes,
                instead of a single axis or all the axes as before. Default: None.
            keepdims (bool, optional):
                If this is set to True, the axes which are reduced are left in the
                result as dimensions with size one. With this option, the result will
                broadcast correctly against the input array. Default: False.
            initial (scalar, optional):
                The maximum value of an output element. Must be present to allow
                computation on empty slice. Default: None.
            where (bool Tensor, optional):
                A boolean array which is broadcasted to match the dimensions of array,
                and selects elements to include in the reduction. If non-default value
                is passed, initial must also be provided. Default: True.

        Returns:
            Tensor or scalar, minimum of input tensor. If the axis is None, the result is a scalar
            value. If `axis` is given, the result is an array of dimension ``self.ndim - 1``.

        Raises:
            TypeError: if arguments have types not specified above.

        Supported Platforms:
            ``Ascend`` ``GPU`` ``CPU``

        Examples:
            >>> import numpy as np
            >>> from mindspore import Tensor
            >>> import mindspore.numpy as np
            >>> a = Tensor(np.arange(4).reshape((2,2)).astype('float32'))
            >>> output = a.min()
            >>> print(output)
            0.0
        """
        reduce_ = tensor_operator_registry.get("reduce")
        reduce_min = tensor_operator_registry.get("reduce_min")
        minimum = tensor_operator_registry.get("minimum")
        return reduce_(self, reduce_min(keepdims), cmp_fn=minimum(), axis=axis, keepdims=keepdims,
                       initial=initial, where=where)

    def fill(self, value):
        """
        Fill the array with a scalar value.

        Note:
            Unlike Numpy, tensor.fill() will always returns a new tensor, instead of
            filling the original tensor.

        Args:
            value (Union[None, int, float, bool]): All elements of a will be assigned this value.

        Returns:
            Tensor, with the original dtype and shape as input tensor.

        Raises:
            TypeError: If input arguments have types not specified above.

        Supported Platforms:
            ``Ascend`` ``GPU`` ``CPU``

        Examples:
            >>> import numpy as np
            >>> from mindspore import Tensor
            >>> a = Tensor(np.arange(4).reshape((2,2)).astype('float32'))
            >>> print(a.fill(1.0))
            [[1. 1.]
            [1. 1.]]
        """
        if value is None:
            if self.dtype not in (mstype.float16, mstype.float32, mstype.float64):
                raise TypeError("If None is used as value, the original Tensor's dtype must be float.")
            value = Tensor(float('nan')).astype("float32")
            return tensor_operator_registry.get("tile")()(value, self.shape).astype(self.dtype)
        if not isinstance(value, (int, float, bool)):
            raise TypeError("input value must be a scalar.")
        return tensor_operator_registry.get("fill")(self.dtype, self.shape, value)

    def ptp(self, axis=None, keepdims=False):
        """
        The name of the function comes from the acronym for ‘peak to peak’.

        Note:
            Numpy arguments `dtype` and `out` are not supported.

        Args:
            axis (Union[None, int, tuple(int)]): Axis or axes along which the range is computed.
                The default is to compute the variance of the flattened array. Default: None.
            keepdims (bool): If this is set to True, the axes which are reduced are left in the result as
                dimensions with size one. With this option, the result will broadcast correctly against the array.
                Default is False.

        Returns:
            Tensor.

        Raises:
            TypeError: if `self` is not a tensor, or `axis` and `keepdims` have types not specified above.

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
            raise TypeError('keepdims should be boolean')
        if axis is None:
            axis = ()
        else:
            validator.check_axis_type(axis, True, True, False)
            axis = validator.check_axis_valid(axis, self.ndim)

        return self.max(axis, keepdims) - self.min(axis, keepdims)

    def clip(self, xmin, xmax, dtype=None):
        """
        Clips (limits) the values in a Tensor.

        Given an interval, values outside the interval are clipped to the interval edges.
        For example, if an interval of :math:`[0, 1]` is specified, values smaller than 0 become 0,
        and values larger than 1 become 1.

        Note:
            Currently, clip with `xmin=nan` or `xmax=nan` is not supported.

        Args:
            xmin (Tensor, scalar, None): Minimum value. If None, clipping is not performed
                on lower interval edge. Not more than one of `xmin` and `xmax` may be None.
            xmax (Tensor, scalar, None): Maximum value. If None, clipping is not performed
                on upper interval edge. Not more than one of `xmin` and `xmax` may be None.
                If `xmin` or `xmax` are tensors, then the three tensors will be broadcasted
                to match their shapes.
            dtype (:class:`mindspore.dtype`, optional): Overrides the dtype of the
                output Tensor. Default is None.

        Returns:
            Tensor, a tensor with the elements of input tensor, but where values
            < `xmin` are replaced with `xmin`, and those > `xmax` with `xmax`.

        Raises:
            TypeError: If inputs have types not specified above.
            ValueError: If the shapes of `x1` and `x2` cannot broadcast, or both `xmin` and `xmax` are `None`.

        Supported Platforms:
            ``Ascend`` ``GPU`` ``CPU``

        Examples:
            >>> from mindspore import Tensor
            >>> x = Tensor([1, 2, 3, -4, 0, 3, 2, 0]).astype("float32")
            >>> output = x.clip(0, 2)
            >>> print(output)
            [1. 2. 2. 0. 0. 2. 2. 0.]
        """
        if xmin is None and xmax is None:
            raise ValueError("One of max or min must be given.")
        x = self
        # F.maximum/minimum does not support when both operands are scalar
        if xmin is not None:
            xmin = Tensor(xmin).astype(x.dtype)
            if x.ndim == 0 and xmin.ndim == 0:
                x = tensor_operator_registry.get("maximum")()(x.reshape((1,)), xmin).squeeze()
            else:
                x = tensor_operator_registry.get("maximum")()(x, xmin)
        if xmax is not None:
            xmax = Tensor(xmax).astype(x.dtype)
            if x.ndim == 0 and xmax.ndim == 0:
                x = tensor_operator_registry.get("minimum")()(x.reshape((1,)), xmax).squeeze()
            else:
                x = tensor_operator_registry.get("minimum")()(x, xmax)
        if dtype is not None and dtype != x.dtype:
            return x.astype(dtype)
        return x

    def _init_check(self):
        if self.has_init:
            self.init_data()
        return self

    def init_data(self, slice_index=None, shape=None, opt_shard_group=None):
        """
        Get the tensor format data of this Tensor.
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
            >>> import mindspore.common.initializer as init
            >>> x = init.initializer(init.Constant(1), [2, 2], ms.float32)
            >>> out = x.init_data()
            >>> print(out)
            [[1. 1.]
             [1. 1.]]
        """
        if self.init is None:
            raise TypeError("init_data must be set Tensor.init, init can't be None")

        if shape is None:
            shape = self.shape

        try:
            arr = np.ndarray(shape, dtype=mstype.dtype_to_nptype(self.dtype))
        except ValueError:
            msg = "Error shape={}".format(shape)
            logger.error(msg)
            raise ValueError(msg)

        class seed_context:
            """Set and restore seed."""

            def __init__(self, init):
                self.init = init
                from .seed import get_seed
                global_seed = get_seed()
                self._np_seed = np.random.get_state()[1][0]
                self.need_set_seed = ((slice_index is not None) and (global_seed is None))

            def __enter__(self):
                if self.need_set_seed:
                    self.seed = self.init.seed
                    np.random.seed(slice_index)
                    self.init.seed = slice_index

            def __exit__(self, ptype, value, trace):
                if self.need_set_seed:
                    np.random.seed(self._np_seed)
                    self.init.seed, _ = self.seed

        with seed_context(self.init):
            self.init(arr)
        data = np.array(arr)
        if opt_shard_group:
            rank = get_rank(opt_shard_group)
            size = get_group_size(opt_shard_group)
            data = np.split(data, size)[rank]
        self.init = None
        self.assign_value(Tensor(data, dtype=self.dtype))
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

        Supported Platforms:
            ``Ascend`` ``GPU`` ``CPU``

        Examples:
            >>> import mindspore as ms
            >>> import mindspore.common.initializer as init
            >>> x = init.initializer(init.Constant(1), [2, 2], ms.float32)
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
        Changes shape and size of array in-place.

        Note:
            Instead of changing the size of the input array and returns nothing as in numpy,
            this method returns a new Tensor with the input size.
            Numpy argument `refcheck` is not supported.

        Args:
            new_shape (Union[ints, tuple of ints]): Shape of resized array.

        Returns:
            Tensor.

        Supported Platforms:
            ``Ascend`` ``GPU`` ``CPU``

        Examples:
            >>> import numpy as np
            >>> from mindspore import Tensor
            >>> x = Tensor(np.array([[0, 1], [2, 3]]))
            >>> x = x.resize(2, 3)
            >>> print(x)
            [[0 1 2]
            [3 0 0]]
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

    def diagonal(self, offset=0, axis1=0, axis2=1):
        """
        Return specified diagonals.

        Args:
            offset (int, optional): Offset of the diagonal from the main diagonal.
                Can be positive or negative. Defaults to main diagonal.
            axis1 (int, optional): Axis to be used as the first axis of the 2-D
                sub-arrays from which the diagonals should be taken. Defaults to
                first axis (0).
            axis2 (int, optional): Axis to be used as the second axis of the 2-D
                sub-arrays from which the diagonals should be taken. Defaults to
                second axis.

        Returns:
            Tensor, if `a` is 2-D, then `a` 1-D array containing the diagonal.

        Raises:
            ValueError: if the input tensor has less than two dimensions.

        Supported Platforms:
            ``Ascend`` ``GPU`` ``CPU``

        Examples:
            >>> import numpy as np
            >>> from mindspore import Tensor
            >>> a = Tensor(np.arange(4).reshape(2, 2))
            >>> print(a)
            [[0 1]
            [2 3]]
            >>> output = a.diagonal()
            >>> print(output)
            [0 3]
        """
        ndim = self.ndim
        if ndim < 2:
            raise ValueError('diagonal requires an array of at least two dimensions')
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
                e_left = tensor_operator_registry.get('fill')(dtype, (n, offset), 0)
                e_right = e[..., 0:m-offset:1]
                e = tensor_operator_registry.get('concatenate')(1)((e_left, e_right)).astype(dtype)
            elif offset < 0:
                e_upper = tensor_operator_registry.get('fill')(dtype, (-offset, m), 0)
                e_lower = e[0:n+offset:1, ...]
                e = tensor_operator_registry.get('concatenate')(0)((e_upper, e_lower)).astype(dtype)
        e = tensor_operator_registry.get('broadcast_to')(shape)(e)

        prod = tensor_operator_registry.get('__mul__')(a, e)
        res = tensor_operator_registry.get('reduce_sum')(prod.astype(mstype.float32), -1)

        begin = ()
        for i in range(ndim-2):
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
        Return the sum along diagonals of the array.

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
            Tensor, sum_along_diagonals.

        Raises:
            ValueError: if the input tensor has less than two dimensions.

        Supported Platforms:
            ``Ascend`` ``GPU`` ``CPU``

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
        Takes elements from an array along an axis.

        Args:
            indices (Tensor): The indices with shape `(Nj...)` of the values to extract.
            axis (int, optional): The axis over which to select values. By default,
                the flattened input array is used. Default: `None`.
            mode (‘raise’, ‘wrap’, ‘clip’, optional):
                - edge: Pads with the edge values of `arr`.
                - raise: Raises an error;
                - wrap: Wraps around;
                - clip: Clips to the range. `clip` mode means that all indices that are
                  too large are replaced by the index that addresses the last element
                  along that axis. Note that this disables indexing with negative numbers.

                Default: `clip`.

        Returns:
            Tensor, the indexed result.

        Raises:
            ValueError: if `axis` is out of range, or `mode` has values other than (‘raise’, ‘wrap’, ‘clip’)

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
            raise ValueError('raise should be one of "raise", "wrap", or "clip"')
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
        Construct an array from an index array and a list of arrays to choose from.

        Args:
            choices (Union[tuple, list, Tensor]): Choice arrays. `a` and all of the `choices` must
                be broadcasted to the same shape. If `choices` is itself an array, then
                its outermost dimension (i.e., the one corresponding to ``choices.shape[0]``)
                is taken as defining the “sequence”.
            mode (‘raise’, ‘wrap’, ‘clip’, optional): Specifies how indices outside
                ``[0, n-1]`` will be treated:

                ‘raise’ – raise an error (default);

                ‘wrap’ – wrap around;

                ‘clip’ – clip to the range. ‘clip’ mode means that all indices that are
                too large are replaced by the index that addresses the last element
                along that axis. Note that this disables indexing with negative numbers.

        Returns:
            Tensor, the merged result.

        Supported Platforms:
            ``Ascend`` ``GPU`` ``CPU``

        Raises:
            ValueError: if the input tensor and any of the `choices` cannot be broadcast.

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
            choices = tensor_operator_registry.get('stack')(0)(tmp)

        if self.ndim == 0 or choices.ndim == 0:
            raise ValueError('input cannot be scalars')
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
        grid = tensor_operator_registry.get('stack')(-1)(grids)
        indices = tensor_operator_registry.get('concatenate')(-1)((a.reshape(a.shape + (1,)), grid))
        return tensor_operator_registry.get('gather_nd')(choices, indices).astype(dtype)

    def searchsorted(self, v, side='left', sorter=None):
        """
        Finds indices where elements should be inserted to maintain order.

        Args:
            v (Union[int, float, bool, list, tuple, Tensor]): Values to insert into `a`.
            side ('left', 'right', optional): If ‘left’, the index of the first suitable
                location found is given. If ‘right’, return the last such index. If there is
                no suitable index, return either 0 or N (where N is the length of `a`).
                Default: `left`.
            sorter (Union[int, float, bool, list, tuple, Tensor]): 1-D optional array of
                integer indices that sort array `a` into ascending order. They are typically
                the result of argsort.

        Returns:
            Tensor, array of insertion points with the same shape as `v`.

        Raises:
            ValueError: if argument for `side` or `sorter` is invalid.

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
            raise ValueError(f'{side} is an invalid value for keyword "side"')
        a = self.astype(mstype.float32)
        if not isinstance(v, Tensor):
            v = tensor_operator_registry.get('make_tensor')(v)
        shape = v.shape
        if sorter is not None:
            if sorter.ndim != 1 or sorter.size != a.size:
                raise ValueError('sorter must be 1-D array with the same size as `a`')
            sorter = tensor_operator_registry.get('make_tensor')(sorter)
            sorter = sorter.reshape(sorter.shape + (1,))
            a = tensor_operator_registry.get('gather_nd')(a, sorter)
        less_op = tensor_operator_registry.get('__le__') if side == 'left' else tensor_operator_registry.get('__lt__')
        i = tensor_operator_registry.get('fill')(mstype.int32, shape, 0)
        j = tensor_operator_registry.get('fill')(mstype.int32, shape, a.size)

        sort_range = tuple(range(validator.get_log2_size(
            tensor_operator_registry.get('shape_mul')(shape) + 1)))
        for _ in sort_range:
            mid = (i - -j)//2
            mask = less_op(v, tensor_operator_registry.get('gather_nd')(a, mid.reshape(mid.shape + (1,))))
            i = tensor_operator_registry.get('select')(mask, i, mid)
            j = tensor_operator_registry.get('select')(mask, mid, j)
        return j

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

        Supported Platforms:
            ``Ascend`` ``GPU`` ``CPU``

        Returns:
            Standard deviation tensor.

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
            raise TypeError(f"integer argument expected, but got {type(ddof)}")
        if not isinstance(keepdims, int):
            raise TypeError(f"integer argument expected, but got {type(keepdims)}")

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
        Compute the standard deviation along the specified axis.
        The standard deviation is the square root of the average of the squared deviations
        from the mean, i.e., :math:`std = sqrt(mean(abs(x - x.mean())**2))`.

        Return the standard deviation, which is computed for the flattened array by default,
        otherwise over the specified axis.

        Note:
            Numpy arguments `dtype`, `out` and `where` are not supported.

        Args:
            axis (Union[None, int, tuple(int)]): Axis or axes along which the standard
                deviation is computed. Default: `None`.

                If `None`, compute the standard deviation of the flattened array.
            ddof (int): Means Delta Degrees of Freedom. The divisor used in calculations is :math:`N - ddof`,
                where :math:`N` represents the number of elements. Default: 0.
            keepdims: Default: `False`.

        Returns:
            Standard deviation tensor.

        Supported Platforms:
            ``Ascend`` ``GPU`` ``CPU``

        Examples:
            >>> import numpy as np
            >>> from mindspore import Tensor
            >>> input_x = Tensor(np.array([1, 2, 3, 4], dtype=np.float32))
            >>> output = input_x.std()
            >>> print(output)
            1.118034
        """
        x_var = self.var(axis, ddof, keepdims)
        return tensor_operator_registry.get('__pow__')(x_var, 0.5)

    def sum(self, axis=None, dtype=None, keepdims=False, initial=None):
        """
        Return sum of array elements over a given axis.

        Note:
            Numpy arguments `out`, `where`, `casting`, `order`, `subok`, `signature`, and
            `extobj` are not supported.

        Args:
            axis (Union[None, int, tuple(int)]): Axis or axes along which a sum is performed. Default: None.
                If None, sum all of the elements of the input array.
                If the axis is negative, it counts from the last to the first axis.
                If the axis is a tuple of ints, a sum is performed on all of the axes specified in the tuple
                instead of a single axis or all the axes as before.
            dtype (:class:`mindspore.dtype`, optional): defaults to None. Overrides the dtype of the
                output Tensor.
            keepdims (bool): If this is set to True, the axes which are reduced are left in the result as
                dimensions with size one. With this option, the result will broadcast correctly against the input array.
                If the default value is passed, then keepdims will not be passed through to the sum method of
                sub-classes of ndarray, however any non-default value will be. If the sub-class’ method does not
                implement keepdims any exceptions will be raised. Default: `False`.
            initial (scalar): Starting value for the sum. Default: `None`.

        Returns:
            Tensor. A tensor with the same shape as input, with the specified axis removed.
            If input tensor is a 0-d array, or if the axis is None, a scalar is returned.

        Raises:
            TypeError: If input is not array_like, or `axis` is not int or tuple of ints,
                or `keepdims` is not integer, or `initial` is not scalar.
            ValueError: If any axis is out of range or duplicate axes exist.

        Supported Platforms:
            ``Ascend`` ``GPU`` ``CPU``

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
            raise TypeError(f"integer argument expected, but got {type(keepdims)}")
        if initial is not None and not isinstance(initial, (int, float, bool)):
            raise TypeError("initial argument should be a scalar.")
        if axis is None:
            axis = ()
        else:
            axis = validator.check_and_canonicalize_axes(axis, self.ndim)

        if not validator.check_type_support(input_x.dtype, 'GPU',
                                            (mstype.float64, mstype.float32, mstype.float16)):
            input_x = input_x.astype(mstype.float32)
        if 0 in self.shape:
            input_x = tensor_operator_registry.get('make_tensor')([0], self.dtype)
        res = tensor_operator_registry.get('sum')(bool(keepdims))(input_x, axis)
        if initial is not None:
            res += initial
        return res.astype(dtype)

    def repeat(self, repeats, axis=None):
        """
        Repeat elements of an array.

        Args:
            repeats (Union[int, tuple, list]): The number of repetitions for each element.
                `repeats` is broadcasted to fit the shape of the given axis.
            axis (int, optional): The axis along which to repeat values. By default,
                use the flattened input tensor, and return a flat output tensor.

        Returns:
            Tensor, has the same shape as input tensor except along the given axis.

        Raises:
            ValueError: if the axis is out of range.
            TypeError: if arguments have types not specified above.

        Supported Platforms:
            ``Ascend`` ``GPU`` ``CPU``

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
        for element in repeats:
            if not isinstance(element, int):
                raise TypeError(f"Each element in {repeats} should be integer, but got {type(element)}.")
        input_x = self
        if axis is None:
            input_x = self.ravel()
            axis = 0
        if axis is not None and not isinstance(axis, int):
            raise TypeError(f'axes should be integers, not {type(axis)}')
        validator.check_axis_in_range(axis, input_x.ndim)
        axis = axis + input_x.ndim if axis < 0 else axis

        if len(repeats) == 1:
            repeats = repeats[0]
            if repeats == 0:
                return Tensor_(input_x.dtype, (0,))
            return tensor_operator_registry.get('repeat_elements')(input_x, repeats, axis)
        size = input_x.shape[axis]
        if len(repeats) != size:
            raise ValueError('operands could not be broadcast together')
        subs = tensor_operator_registry.get('split')(axis, size)(input_x)
        repeated_subs = []
        for sub, rep in zip(subs, repeats):
            if rep != 0:
                repeated_subs.append(tensor_operator_registry.get('repeat_elements')(sub, rep, axis))
        return tensor_operator_registry.get('concatenate')(axis)(repeated_subs)


class RowTensor:
    """
    A sparse representation of a set of tensor slices at given indices.

    An RowTensor is typically used to represent a subset of a larger
    tensor dense of shape [L0, D1, .. , DN] where L0 >> D0.

    The values in indices are the indices in the first dimension of the slices
    that have been extracted from the larger tensor.

    The dense tensor dense represented by an RowTensor slices has
    `dense[slices.indices[i], :, :, :, ...] = slices.values[i, :, :, :, ...]`.

    RowTensor can only be used in the `Cell`'s construct method.

    It is not supported in pynative mode at the moment.

    Args:
        indices (Tensor): A 1-D integer Tensor of shape [D0].
        values (Tensor): A Tensor of any dtype of shape [D0, D1, ..., Dn].
        dense_shape (tuple(int)): An integer tuple which contains the shape
            of the corresponding dense tensor.

    Returns:
        RowTensor, composed of `indices`, `values`, and `dense_shape`.

    Examples:
        >>> import mindspore as ms
        >>> import mindspore.nn as nn
        >>> from mindspore import RowTensor
        >>> class Net(nn.Cell):
        ...     def __init__(self, dense_shape):
        ...         super(Net, self).__init__()
        ...         self.dense_shape = dense_shape
        ...     def construct(self, indices, values):
        ...         x = RowTensor(indices, values, self.dense_shape)
        ...         return x.values, x.indices, x.dense_shape
        >>>
        >>> indices = Tensor([0])
        >>> values = Tensor([[1, 2]], dtype=ms.float32)
        >>> out = Net((3, 2))(indices, values)
        >>> print(out[0])
        [[1. 2.]]
        >>> print(out[1])
        [0]
        >>> print(out[2])
        (3, 2)
    """

    def __init__(self, indices, values, dense_shape):
        "Init RowTensor"
        self.__indices = indices
        self.__values = values
        self.__dense_shape = dense_shape

    @property
    def indices(self):
        return self.__indices

    @property
    def values(self):
        return self.__values

    @property
    def dense_shape(self):
        return self.__dense_shape


class SparseTensor:
    """
    A sparse representation of a set of nonzero elememts from a tensor at given indices.

    SparseTensor can only be used in the `Cell`'s construct method.

    Pynative mode not supported at the moment.

    For a tensor dense, its SparseTensor(indices, values, dense_shape) has
    `dense[indices[i]] = values[i]`.

    Args:
        indices (Tensor): A 2-D integer Tensor of shape `[N, ndims]`,
            where N and ndims are the number of `values` and number of dimensions in
            the SparseTensor, respectively.
        values (Tensor): A 1-D tensor of any type and shape `[N]`, which
            supplies the values for each element in `indices`.
        dense_shape (tuple(int)): A integer tuple of size `ndims`,
            which specifies the dense_shape of the sparse tensor.

    Returns:
        SparseTensor, composed of `indices`, `values`, and `dense_shape`.

    Examples:
        >>> import mindspore as ms
        >>> import mindspore.nn as nn
        >>> from mindspore import SparseTensor
        >>> class Net(nn.Cell):
        ...     def __init__(self, dense_shape):
        ...         super(Net, self).__init__()
        ...         self.dense_shape = dense_shape
        ...     def construct(self, indices, values):
        ...         x = SparseTensor(indices, values, self.dense_shape)
        ...         return x.values, x.indices, x.dense_shape
        >>>
        >>> indices = Tensor([[0, 1], [1, 2]])
        >>> values = Tensor([1, 2], dtype=ms.float32)
        >>> out = Net((3, 4))(indices, values)
        >>> print(out[0])
        [1. 2.]
        >>> print(out[1])
        [[0 1]
         [1 2]]
        >>> print(out[2])
        (3, 4)
    """

    def __init__(self, indices, values, dense_shape):
        "Init SparseTensor"
        self.__indices = indices
        self.__values = values
        self.__dense_shape = dense_shape

    @property
    def indices(self):
        return self.__indices

    @property
    def values(self):
        return self.__values

    @property
    def dense_shape(self):
        return self.__dense_shape


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
        raise ValueError("If input_data is available, shape doesn't need to be set")

    if init is not None and (shape is None or dtype is None):
        raise ValueError("init, dtype and shape must have values at the same time.")

    if (int(input_data is None) + int(init is None)) != 1:
        raise TypeError("input_data and init can not be None at the same time.")

    if input_data is not None:
        if isinstance(input_data, np.ndarray) and input_data.ndim > 1 and input_data.size == 0:
            raise ValueError("input_data can not contain zero dimension.")
        if isinstance(input_data, (tuple, list)) and np.array(input_data).ndim > 1 \
                and np.array(input_data).size == 0:
            raise ValueError("input_data can not contain zero dimension.")

    if shape is not None and not (hasattr(init, "__enable_zero_dim__") and init.__enable_zero_dim__) and 0 in shape:
        raise ValueError("Shape can not contain zero value.")


tensor_operator_registry.register('vm_compare', _vm_compare)
