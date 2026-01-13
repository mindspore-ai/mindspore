# Copyright 2020-2025 Huawei Technologies Co., Ltd
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
import numbers
import numpy as np

from mindspore.communication.management import get_group_size
from mindspore.common._utils import is_shape_unknown
from mindspore.common.seed import get_seed
from mindspore import context
from mindspore import log as logger
from mindspore.common import dtype as mstype
from mindspore.common._decorator import deprecated
from mindspore.common.hook_handle import _TensorHookHandle

from mindspore.common._register_for_tensor import tensor_operator_registry
from mindspore._c_expression import TensorPy as TensorPy_
from mindspore._c_expression import _rmod_instance
from mindspore import _checkparam as validator
from mindspore._checkparam import is_stub_tensor, check_hook_fn
from mindspore._check_jit_forbidden_api import jit_forbidden_register
from mindspore.common.symbol import Symbol
from mindspore._c_expression import is_reboot_node

np_types = (np.int8, np.int16, np.int32, np.int64,
            np.uint8, np.uint16, np.uint32, np.uint64, np.float16,
            np.float32, np.float64, np.bool_, np.complex64, np.complex128)


def _check_input_data_type(input_data):
    """Check the type of input_data for Tensor"""
    validator.check_value_type('input_data', input_data, (TensorPy_, Tensor, np.ndarray, np.str_, list, tuple, float,
                                                          int, bool, complex, bytes),
                               'Tensor')
    valid_dtypes = (np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64,
                    np.float16, np.float32, np.float64, np.bool_, np.str_, np.complex64, np.complex128)
    if isinstance(input_data, np.ndarray) and input_data.dtype not in valid_dtypes and \
            input_data.dtype.kind != 'U' and input_data.dtype.kind != 'S' and \
            not (input_data.dtype.kind == 'V' and input_data.dtype.char == 'E'):  # Support np.str_ and np.bfloat16
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


def _set_symbolic_shape(shape):
    """Set symbolic_shape"""
    symbolic_shape = None
    if shape is None:
        return None, None
    if isinstance(shape, numbers.Number):
        shape = (shape,)
        symbolic_shape = None
        return shape, symbolic_shape
    if isinstance(shape, Symbol):
        symbolic_shape = [shape]
        shape = (None,)
        return shape, symbolic_shape
    if isinstance(shape, (list, tuple)) and any(isinstance(s, Symbol) for s in shape):
        symbolic_shape = [item.to_dict() if isinstance(item, Symbol) else item for item in shape]
        shape_without_symbol = (None if isinstance(item, Symbol) else item for item in shape)
        shape = list(shape_without_symbol) if isinstance(shape, list) else tuple(shape_without_symbol)
        return shape, symbolic_shape
    return shape, symbolic_shape


def _convert_numpy_array(input_data):
    """Convert inpyt to numpy array"""
    if not isinstance(input_data, np_types):
        return input_data
    return np.array(input_data)


def _check_device(device):
    """Check device"""
    if device is not None and device != "CPU":
        raise ValueError(f"Only 'CPU' is supported for device, but got {device}.")


def _set_default_dtype(input_data, dtype):
    """Set tensor default dtype"""
    if isinstance(input_data, (float, list, tuple)):
        if np.array(input_data).dtype == np.float64:
            return mstype.float32
    if isinstance(input_data, (int, list, tuple)):
        if np.array(input_data).dtype in (np.int32, np.int64):
            return mstype.int64
    return dtype


def _set_dtype(input_data, dtype):
    """Set and check dtype"""
    if dtype is not None:
        validator.check_type_name('dtype', dtype, mstype.number_type + (mstype.bool_, mstype.string), "Tensor")
        return dtype
    return _set_default_dtype(input_data, dtype)


def _init(input_data=None, dtype=None, shape=None, init=None, const_arg=False, device=None):
    """
    Verifying parameters. Will sink to C++
    """
    validator.check_value_type('const_arg', const_arg, bool, 'Tensor')
    _check_device(device)

    if isinstance(input_data, (Tensor, TensorPy_)) and dtype is not None:
        logger.info("It is suggested to use 'Tensor.astype()' to convert the dtype of a Tensor.")
        _cast = tensor_operator_registry.get("cast")
        input_data = _cast(input_data, dtype)

    if input_data is None and shape is None and init is None and dtype is not None:
        validator.check_type_name('dtype', dtype, mstype.number_type + (mstype.bool_, mstype.string), "Tensor")
        logger.warning("For 'Tensor', if 'dtype' is not None, 'input_data', 'shape' or 'init' must not be None.")
        return {"dtype": dtype, "shape": [-2], "init": init, "const_arg": const_arg, "device": device}

    # If input data is numpy number, convert it to np array
    input_data = _convert_numpy_array(input_data)
    shape, symbolic_shape = _set_symbolic_shape(shape)
    _check_tensor_input(input_data, dtype, shape, init)

    # If input_data is tuple/list/numpy.ndarray, it's support in check_type method.
    if (isinstance(shape, (list, tuple)) and None in shape) or init is not None:
        shape = _check_tensor_dynamic_shape(dtype, shape, init)
        return {"dtype": dtype, "shape": shape, "init": init, "const_arg": const_arg, "device": device,
                "symbolic_shape": symbolic_shape}

    if input_data is None and dtype is not None and shape is not None:
        validator.check_type_name('dtype', dtype, mstype.number_type + (mstype.bool_, mstype.string), "Tensor")
        return {"dtype": dtype, "shape": shape, "init": init, "const_arg": const_arg, "device": device,
                "symbolic_shape": symbolic_shape}

    _check_input_data_type(input_data)
    dtype = _set_dtype(input_data, dtype)

    if isinstance(input_data, np.ndarray) and (not input_data.flags['FORC']):
        input_data = np.ascontiguousarray(input_data)

    if dtype is not None:
        return {"input_data": input_data, "dtype": dtype, "init": init, "const_arg": const_arg, "device": device,
                "symbolic_shape": symbolic_shape}

    return {"input_data": input_data, "init": init, "const_arg": const_arg, "device": device,
            "symbolic_shape": symbolic_shape}


def tensor(input_data=None, dtype=None, shape=None, init=None, const_arg=False):
    """
    Create a new Tensor in Cell.construct() or function decorated by @jit.

    In graph mode, MindSpore would create a new Tensor object at runtime dynamically,
    based on the `dtype` argument.

    Please refer to `Creating and Using Tensor
    <https://www.mindspore.cn/tutorials/en/master/compile/static_graph.html#mindspore-user-defined-data-types>`_ .

    The difference between it and the Tensor class is that it adds
    `Annotation
    <https://www.mindspore.cn/tutorials/en/master/compile/static_graph.html#annotation-type>`_
    which can prevent the generation of AnyType compared to the Tensor class.

    The arguments and return values are the same as the Tensor class. Also see: :class:`mindspore.Tensor`.
    internally to indicate the type of the Tensor currently being created,

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore as ms
        >>> from mindspore import jit, tensor
        >>> @jit
        ... def func(x):
        ...    return tensor(x.asnumpy(), dtype=ms.float32)
        >>> x = tensor([1, 2, 3])
        >>> y = func(x)
        >>> print(y)
        [1. 2. 3.]
    """
    return Tensor(input_data, dtype, shape, init, const_arg)  # @jit.typing: () -> tensor_type[{dtype}]


class _TensorMeta(abc.ABCMeta, type(TensorPy_)):
    """
    Meta class for Tensor. Used internally.
    """


class Tensor(TensorPy_, metaclass=_TensorMeta):
    """
    Tensor is a data structure that stores an n-dimensional array.

    Note:
        - If `init` interface is used to initialize `Tensor`, the `Tensor.init_data` API needs to be called to load the
          actual data to `Tensor`.
        - All modes of CPU and GPU, and Atlas training series with `graph mode (mode=mindspore.GRAPH_MODE)
          <https://www.mindspore.cn/tutorials/en/master/compile/static_graph.html>`_  do not supported
          in-place operations yet.

    Warning:
        To convert dtype of a `Tensor`, it is recommended to use `Tensor.astype()` rather than
        `Tensor(sourceTensor, dtype=newDtype)`.

    Args:
        input_data (Union[Tensor, float, int, bool, tuple, list, numpy.ndarray]): The data to be stored. It can be
            another Tensor, Python number or NumPy ndarray. Default: ``None`` .
        dtype (:class:`mindspore.dtype`): Used to indicate the data type of the output Tensor. The argument should
            be defined in `mindspore.dtype`. If it is ``None`` , the data type of the output Tensor will be the same
            as the `input_data`. Default: ``None`` .
        shape (Union[tuple, list, int, :class:`mindspore.Symbol`]): Used to indicate the shape of the output Tensor.
            If `input_data` is available, `shape` doesn't need to be set. If ``None`` or `Symbol` exists in `shape` ,
            a tensor of dynamic shape is created, `input_data` doesn't need to be set; if only integers exist in
            `shape`, a tensor of static shape is created, `input_data` or `init` must be set. Default: ``None`` .
        init (Initializer): The information of init data.
            `init` is used for delayed initialization in parallel mode, when using init, `dtype` and `shape` must be
            set. Default: ``None`` .
        const_arg (bool): Whether the tensor is a constant when it is used for the argument of a network.
            Default: ``False`` .
        device(str): This parameter is reserved and does not need to be configured.
            Default: ``None`` .

    Outputs:
        Tensor.

    Note:
        The default value ``None`` of `input_data` works as a placeholder,
        it does not mean that we can create a NoneType Tensor.
        Tensor with `shape` contains 0 is not fully tested and supported.

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
        >>> t4.init_data()
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

    @classmethod
    def __subclasshook__(cls, sub):
        """
        Subclass with stub_sync attr will be instance of Tensor
        """
        if cls is Tensor:
            if any("stub_sync" in s.__dict__ for s in sub.__mro__):
                return True
        return NotImplemented

    def __deepcopy__(self, memodict):
        new_obj = Tensor(self)
        new_obj.init = self.init
        new_obj.virtual_flag = self.virtual_flag
        new_obj.const_arg = self.const_arg
        return new_obj

    def __repr__(self):
        if self.init_finished:
            return TensorPy_.__repr__(self)
        return ''

    def __eq__(self, other):
        if not isinstance(other, (int, float, Tensor)):
            return False
        return tensor_operator_registry.get('__eq__')(self, other)

    def __ne__(self, other):
        if not isinstance(other, (int, float, Tensor)):
            return True
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
        out = tensor_operator_registry.get('round')(self)
        return out

    def __bool__(self):
        return bool(self._item())

    @staticmethod
    def _convert_scalar_(data, func, message):
        if data.shape == ():
            return func(data)
        if data.shape == (1,):
            return func(data[0])
        raise ValueError(message)

    def __int__(self):
        try:
            data = self._item()
            return int(data)
        except ValueError as e:
            raise ValueError("Only one element tensors can be converted to Python scalars") from e

    def __float__(self):
        try:
            data = self._item()
            return float(data)
        except ValueError as e:
            raise ValueError("Only one element tensors can be converted to Python scalars") from e

    def __index__(self):
        try:
            data = self._item()
            if not isinstance(data, (int, bool)):
                raise ValueError
            return int(data)
        except ValueError as e:
            raise ValueError("Only integer tensors of a single element can be converted to an index.") from e

    def __pos__(self):
        return self

    def __radd__(self, other):
        return self.__add__(other)

    def __rsub__(self, other):
        return tensor_operator_registry.get('__sub__')(other, self)

    def __rmul__(self, other):
        return self.mul(other)

    def __matmul__(self, other):
        return tensor_operator_registry.get('__matmul__')(self, other)

    def __rmatmul__(self, other):
        return tensor_operator_registry.get('__matmul__')(other, self)

    def __truediv__(self, other):
        return tensor_operator_registry.get('__truediv__')(self, other)

    def __rtruediv__(self, other):
        return tensor_operator_registry.get('__truediv__')(other, self)

    def __rmod__(self, other):
        return _rmod_instance(other, self)

    def __rpow__(self, other):
        return tensor_operator_registry.get('__rpow__')(self, other)

    def __floordiv__(self, other):
        return tensor_operator_registry.get('__floordiv__')(self, other)

    def __rfloordiv__(self, other):
        return tensor_operator_registry.get('__floordiv__')(other, self)

    def __lt__(self, other):
        out = tensor_operator_registry.get('__lt__')(self, other)
        return out

    def __le__(self, other):
        out = tensor_operator_registry.get('__le__')(self, other)
        return out

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
        if not self._data_ptr():
            return TensorPy_.__str__(self)
        return str(self.asnumpy())

    def __getstate__(self):
        state = self.__dict__.copy()
        state["value"] = TensorPy_.__getstate__(self)
        return state

    def __setstate__(self, state):
        if isinstance(state, tuple):
            value = state
        else:
            value = state.pop("value")
            self.__dict__.update(state)
        TensorPy_.__setstate__(self, value)

    def __array__(self, dtype=None):
        """support create numpy array from tensor."""
        if dtype is None:
            return self.asnumpy()
        return self.asnumpy().astype(dtype, copy=False)

    def __contains__(self, element):
        """support 'in' operator."""
        if isinstance(element, (Tensor, numbers.Number)):
            return (element == self).any().item()
        return False

    def _getitem_origin(self, index):
        """__getitem__ origin process, called by TensorPy::TensorGetItem"""
        out = tensor_operator_registry.get('_tensor_getitem_origin')(self, index)
        if out is not self:
            out.parent_tensor_ = self
            out.index_of_parent_ = index
        return out

    def _setitem_origin(self, index, value):
        """__setitem__ origin process, called by TensorPy::TensorSetItem"""
        out = tensor_operator_registry.get('_tensor_setitem_origin')(self, index, value)
        if isinstance(out, tuple):
            if self.parent_tensor_ is not None and self.index_of_parent_ is not None:
                self.parent_tensor_.__setitem__(self.index_of_parent_, out[0])
                return self
            return self
        self.assign_value(out)
        if self.parent_tensor_ is not None and self.index_of_parent_ is not None:
            self.parent_tensor_.__setitem__(self.index_of_parent_, self)
        return self

    def _getitem(self, index):
        """__getitem__ process, called by TensorPy::TensorGetItem"""
        return tensor_operator_registry.get('_tensor_getitem')(self, index)

    def _setitem(self, index, value):
        """__setitem__ process, called by TensorPy::TensorSetItem"""
        return tensor_operator_registry.get('_tensor_setitem')(self, index, value)

    @property
    def _dtensor_info(self):
        """
        Return the distributed tensor information. For details,
        please refer to :class:`mindspore.parallel.DistributedTensorInfo`.

        Examples:
            >>> from mindspore import Tensor
            >>> import numpy as np
            >>> x = Tensor(np.array([[1, 2], [3, 4]]))
            >>> print(x._dtensor_info)
            None
        """
        if not hasattr(self, '_dist_tensor_info'):
            self._dist_tensor_info = None
        return self._dist_tensor_info

    @_dtensor_info.setter
    def _dtensor_info(self, input_dtensor_info):
        """
        Set the distributed tensor information to current tensor.

        Args:
            input_dtensor_info (DistributedTensorInfo): The distributed tensor information.

        Examples:
            >>> from mindspore import Tensor, Layout, _DistributedTensorInfo
            >>> import numpy as np
            >>> layout = Layout((2, 2), ("dp", "mp"))
            >>> src_layout = layout("dp", "mp")
            >>> distributed_info = _DistributedTensorInfo(src_layout)
            >>> x = Tensor(np.array([[1, 2], [3, 4]]))
            >>> x._dtensor_info = distributed_info
        """
        self._dist_tensor_info = input_dtensor_info

    @property
    def shape(self):
        """
        For details, please refer to :func:`mindspore.ops.shape`.

        Examples:
            >>> from mindspore import Tensor
            >>> import numpy as np
            >>> x = Tensor(np.array([[1, 2], [3, 4]]))
            >>> print(x.shape)
            (2, 2)
        """
        return self._shape

    @shape.setter
    def shape(self, shape_value):
        r"""
        Set the shape value.
        """
        self._shape = shape_value

    @property
    def dtype(self):
        """
        Return the dtype of the tensor (:class:`mindspore.dtype`).

        Examples:
            >>> from mindspore import Tensor
            >>> import numpy as np
            >>> x = Tensor(np.array([1, 2], dtype=np.float32))
            >>> print(x.dtype)
            Float32
        """
        return self._dtype

    @property
    def size(self):
        """
        For details, please refer to :func:`mindspore.ops.size`.

        Examples:
            >>> from mindspore import Tensor
            >>> import numpy as np
            >>> x = Tensor(np.array([[1, 2], [3, 4]]))
            >>> output = x.size
            >>> print(output)
            4
        """
        return self._size

    @property
    def ndim(self):
        """
        Return the number of tensor dimensions.

        Examples:
            >>> from mindspore import Tensor
            >>> import numpy as np
            >>> x = Tensor(np.array([[1, 2], [3, 4]]))
            >>> output = x.ndim
            >>> print(output)
            2
        """
        return len(self._shape)

    @property
    def H(self):
        """
        Returns a view of a matrix (2-D tensor) conjugated and transposed.
        x.H is equivalent to `mindspore.Tensor.swapaxes(0, 1).conj()` for complex matrices and
        `mindspore.Tensor.swapaxes(0, 1)` for real matrices.

        Supported Platforms:
            ``Ascend`` ``GPU`` ``CPU``

        Examples:
            >>> from mindspore import Tensor
            >>> import numpy as np
            >>> x = Tensor(np.array([[1, 2], [3, 4]]))
            >>> output = x.H
            >>> print(output)
            [[1 3]
            [2 4]]
        """
        if self.ndim != 2:
            raise ValueError(f"For tensor.H only support 2-D Tensor, but got {self.ndim}-D.")
        output = self.swapaxes(0, 1)
        if self.dtype in (mstype.complex64, mstype.complex128):
            return output.conj()
        return output

    @property
    def has_init(self):
        """
        Whether tensor is initialized.

        Examples:
            >>> from mindspore import Tensor
            >>> import numpy as np
            >>> x = Tensor(np.array([[1, 2], [3, 4]]))
            >>> output = x.has_init
            >>> print(output)
            False
        """
        return self.init is not None

    @property
    def itemsize(self):
        """
        Return the length of one tensor element in bytes.

        Examples:
            >>> from mindspore import Tensor
            >>> import numpy as np
            >>> x = Tensor(np.array([[1, 2], [3, 4]]))
            >>> output = x.itemsize
            >>> print(output)
            8
        """
        return self._itemsize

    @property
    def strides(self):
        """
        Return the tuple of bytes to step in each dimension when traversing a tensor.

        Examples:
            >>> from mindspore import Tensor
            >>> from mindspore import dtype as mstype
            >>> import numpy as np
            >>> x = Tensor(np.array([[1, 2], [3, 4]]), dtype=mstype.int64)
            >>> output = x.strides
            >>> print(output)
            (16, 8)
        """
        return self._strides

    @property
    def nbytes(self):
        """
        Return the total number of bytes taken by the tensor.

        Examples:
            >>> from mindspore import Tensor
            >>> import numpy as np
            >>> x = Tensor(np.array([[1, 2], [3, 4]]))
            >>> output = x.nbytes
            >>> print(output)
            32
        """
        return self._nbytes

    @property
    def T(self):
        """
        Return the transposed tensor.

        Examples:
            >>> from mindspore import Tensor
            >>> import numpy as np
            >>> x = Tensor(np.array([[1, 2], [3, 4]]))
            >>> output = x.T
            >>> print(output)
            [[1 3]
            [2 4]]
        """
        rank = self.ndim
        if rank <= 1:
            return self
        dims = list(range(rank - 1, -1, -1))
        return self.permute(dims)

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

        return TensorPy_.from_numpy(array)

    def ndimension(self):
        r"""
        Alias for :attr:`mindspore.Tensor.ndim`.
        """
        return len(self._shape)

    @jit_forbidden_register
    def set_const_arg(self, const_arg=True):
        """
        Specify whether the tensor is a constant when it is used for the argument of a network.

        Args:
            const_arg (bool, optional): Whether the tensor is a constant when it is used for the argument of a network.
                Default: ``True`` .

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

    def cauchy(self, median=0.0, sigma=1.0):
        r"""
        Fills the tensor with numbers drawn from the Cauchy distribution. It is
        defined as follows:

        .. math::
            f(x)= \frac{1}{\pi} \frac{\sigma}{(x-median)^2 +\sigma^2}

        .. warning::
            This is an experimental API that is subject to change or deletion.

        Args:
            median (float, optional): the location parameter, specifying the location
                of the peak of the distribution. Default: ``0.0``.
            sigma (float, optional): the scale parameter which specifies the half-width
                at half-maximum. Default: ``1.0``.

        Returns:
            Tensor. A Tensor with the same type and shape of input.

        Supported Platforms:
            ``Ascend`` ``CPU``

        Examples:
            >>> import mindspore
            >>> import numpy as np
            >>> x = mindspore.Tensor(np.zeros((1, 2)), dtype=mindspore.float32)
            >>> x.cauchy()
            Tensor(shape=[1, 2], dtype=Float32, value=
            [[8.79836142e-01, 9.37541723e-01]])

        """
        out = tensor_operator_registry.get('cauchy')(list(self.shape), median, sigma)()
        return out.astype(self.dtype)

    def log_normal(self, mean=1.0, std=2.0):
        r"""
        Fills the elements of the input tensor with log normal values initialized by
        given mean and std:

        .. math::
            \text{f}(x;1.0,2.0)=\frac{1}{x\delta \sqrt[]{2\pi} }e^{-\frac{(\ln x-\mu )^2}{2\delta ^2} }

        where :math:`\mu`, :math:`\delta` is mean and standard deviation of  lognormal distribution respectively.

        .. warning::
            This is an experimental API that is subject to change or deletion.

        Args:
            mean (float, optional): the mean of normal distribution. With float data type.
                Default: ``1.0``.
            std (float, optional): the std of normal distribution. With float data type.
                Default: ``2.0``.

        Returns:
            Tensor. A Tensor with the same type and shape of input.

        Supported Platforms:
            ``Ascend`` ``GPU`` ``CPU``

        Examples:
            >>> import mindspore
            >>> import numpy as np
            >>> x = mindspore.Tensor(np.array([[1, 2], [3, 4]]), dtype=mindspore.float32)
            >>> output = x.log_normal()
            >>> print(output)
            [[1.2788825 2.3305743]
            [14.944194 0.16303174]]
        """
        return tensor_operator_registry.get('log_normal')(mean, std)(self)

    @jit_forbidden_register
    def assign_value(self, value):
        """
        Assign another tensor value to this tensor.

        Args:
            value (Tensor): Tensor for assignment.

        Returns:
            Tensor, Tensor that's been assigned.

        Examples:
            >>> from mindspore import Tensor
            >>> import numpy as np
            >>> x = Tensor([1, 2, 3, 4])
            >>> y = Tensor(np.array([[1, 2], [3, 4]]))
            >>> output = x.assign_value(y)
            >>> print(x)
            [[1 2]
            [3 4]]
        """
        if is_stub_tensor(value):
            value = value.stub_sync()
        self.assign_value_cpp(value)
        return self

    def item(self):
        """
        Return the value of this tensor as standard Python number.
        This only works for tensors with one element.

        Returns:
            A scalar, type is defined by the dtype of the Tensor.

        Raises:
            ValueError: If the count of value in tensor is more than one.
            TypeError: The type of element in tensor is not supported.

        Supported Platforms:
            ``Ascend`` ``GPU`` ``CPU``

        Examples:
            >>> import mindspore as ms
            >>> from mindspore import Tensor
            >>> x = Tensor(1.2, ms.float32)
            >>> print(x.item())
            1.2
        """
        return self._item()

    def itemset(self, *args):
        r"""
        Insert scalar into a tensor (scalar is cast to tensor's dtype, if possible).

        There must be at least 1 argument, and define the last argument as item.
        Then, tensor.itemset(\*args) is equivalent to :math:`Tensor[args] = item`.

        Args:
            args (Union[(numbers.Number), (int/tuple(int), numbers.Number)]): The arguments that
                specify the index and value. If `args` contain one argument (a scalar),
                it is only used in case tensor is of size 1. If `args` contains two
                arguments, the last argument is the value to be set and must be a
                scalar, the first argument specifies a single tensor element location.
                It is either an int or a tuple.

        Returns:
            A new tensor that doesn't affect the original tensor, with value set by :math:`Tensor[args] = item`.

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

    def get_bytes(self):
        r"""
        Get raw data of tensor with type of bytes.

        Supported Platforms:
            ``CPU`` ``GPU`` ``Ascend``

        Returns:
            Bytes of tensor.

        Examples:
            >>> import mindspore as ms
            >>> from mindspore import Tensor
            >>> x = ms.Tensor([1, 2, 3], ms.int16)
            >>> print(x.get_bytes())
            b'\x01\x00\x02\x00\x03\x00'
        """
        return TensorPy_.get_bytes(self)

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
        if self.has_init:
            self.init_data()
        return TensorPy_.asnumpy(self)

    def numpy(self):
        """
        Alias for :func:`mindspore.Tensor.asnumpy`.
        """
        return self.asnumpy()

    def slice_scatter(self, src, axis=0, start=None, end=None, step=1):
        """
        For details, please refer to :func:`mindspore.ops.slice_scatter`.
        """
        return tensor_operator_registry.get('slice_scatter')(self, src, axis, start, end, step)

    def select_scatter(self, src, axis, index):
        """
        For details, please refer to :func:`mindspore.ops.select_scatter`.
        """
        return tensor_operator_registry.get('select_scatter')(self, src, axis, index)

    def geqrf(self):
        """
        For details, please refer to :func:`mindspore.ops.geqrf`.
        """
        return tensor_operator_registry.get('geqrf')(self)

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

    def contiguous(self):
        """
        Converts a Tensor into a continuous-memory Tensor that contains the same data as the original Tensor.

        Returns:
            A contiguous in memory tensor containing the same data as self tensor.

        Examples:
            >>> import mindspore as ms
            >>> import numpy as np
            >>> from mindspore import Tensor, ops
            >>> x = Tensor([[1, 2, 3], [4, 5, 6]], dtype=ms.float32)
            >>> y = ops.transpose(x, (1, 0))
            >>> z = y.contiguous()
            >>> print(z.is_contiguous())
            True
        """
        if not self._need_contiguous():
            return self
        return tensor_operator_registry.get('contiguous')(self)

    def is_contiguous(self):
        """
        Determines whether the memory of tensor is contiguous.

        Returns:
            Bool, True if tensor memory is contiguous, False otherwise.

        Examples:
            >>> import mindspore as ms
            >>> import numpy as np
            >>> from mindspore import Tensor, ops
            >>> x = Tensor([[1, 2, 3], [4, 5, 6]], dtype=ms.float32)
            >>> y = ops.transpose(x, (1, 0))
            >>> print(y.is_contiguous())
            False
        """
        return TensorPy_.is_contiguous(self)

    def stride(self, dim=None):
        """
        The stride to jump from one element to the next in the input dim.
        When no parameters are passed in, a list of stride for all dimensions is returned.

        Args:
            dim (int, optional): The dim of stride from one element to the next. Default: ``None``.

        Returns:
            Int, returns the step size necessary to jump from one element to the next in the specified dimension.

        Raises:
            TypeError: `dim` is not an int.

        Examples:
            >>> import mindspore as ms
            >>> x = ms.Tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], dtype=ms.float32)
            >>> x.stride()
            [5, 1]
        """
        stride = TensorPy_.stride(self)
        if dim is None:
            return stride
        return stride[dim]

    def storage_offset(self):
        """
        Tensor's offset in the underlying storage in terms of the number of storage elements.

        Returns:
            int, tensor's offset in the underlying storage in terms of number of storage elements.

        Examples:
            >>> import mindspore as ms
            >>> x = ms.Tensor([1, 2, 3, 4, 5], dtype=ms.float32)
            >>> ret = x.storage_offset()
            >>> print(ret)
            0
        """
        return TensorPy_.storage_offset(self)

    def register_hook(self, hook):
        """
        Registers a backward hook for tensor.

        Note:
            - The `hook` must be defined as the following code. `grad` is the gradient passed to the tensor,
              which may be modified by returning a new output gradient.
            - The `hook` should have the following signature:
              hook(grad) -> New output gradient, but can not return None or not set return value.
            - The following constraints must be met under graph mode:

              - The `hook` must satisfy the syntax constraints of the graph mode.
              - It is not supported to delete `hook` inside graph.
              - It is not supported to register `hook` after the `Tensor` is used before.
              - It is not supported to register multiple `hooks` for a `Tensor` inside graph.
              - Register `hook` in the graph will return then `Tensor` it self.

        Args:
            hook (function): Python function. Tensor backward hook function.

        Returns:
            A handle corresponding to the `hook` . The handle can be used to remove the added `hook` by calling
            `handle.remove()` .

        Raises:
            TypeError: If the `hook` is not a function of python.

        Supported Platforms:
            ``Ascend`` ``GPU`` ``CPU``

        Examples:
            >>> import mindspore as ms
            >>> from mindspore import Tensor
            >>> ms.set_context(mode=ms.PYNATIVE_MODE)
            >>> def hook_fn(grad):
            ...     return grad * 2
            ...
            >>> def hook_test(x, y):
            ...     z = x * y
            ...     z.register_hook(hook_fn)
            ...     z = z * y
            ...     return z
            ...
            >>> ms_grad = ms.grad(hook_test, grad_position=(0,1))
            >>> output = ms_grad(Tensor(1, ms.float32), Tensor(2, ms.float32))
            >>> print(output)
            (Tensor(shape=[], dtype=Float32, value=8), Tensor(shape=[], dtype=Float32, value=6))
        """
        check_hook_fn(hook)
        handle = _TensorHookHandle(self)
        handle.id = TensorPy_.register_hook(self, hook)
        return handle

    def _remove_hook(self):
        pass

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
        TensorPy_._flush_from_cache(self)

    def addcmul(self, tensor1, tensor2, value=1):
        r"""
        For details, please refer to :func:`mindspore.ops.addcmul`.
        """
        return tensor_operator_registry.get('addcmul')(self, tensor1, tensor2, value)

    def addmm_(self, mat1, mat2, *, beta=1, alpha=1):
        r"""
        In-place version of :func:`mindspore.Tensor.addmm`.

        .. warning::
            This is an experimental API that is subject to change or deletion.
        """
        return tensor_operator_registry.get('addmm_')(self, mat1, mat2, beta=beta, alpha=alpha)

    def addr(self, vec1, vec2, beta=1, alpha=1):
        r"""
        For details, please refer to :func:`mindspore.ops.addr`.
        """
        return tensor_operator_registry.get('addr')(self, vec1, vec2, beta=beta, alpha=alpha)

    def adjoint(self):
        r"""
        For details, please refer to :func:`mindspore.ops.adjoint`.
        """
        return tensor_operator_registry.get('adjoint')(self)

    def angle(self):
        r"""
        For details, please refer to :func:`mindspore.ops.angle`.
        """
        return tensor_operator_registry.get('angle')(self)

    def bitwise_left_shift(self, other):
        """
        For details, please refer to :func:`mindspore.ops.bitwise_left_shift`.
        """
        return tensor_operator_registry.get('bitwise_left_shift')(self, other)

    def bitwise_right_shift(self, other):
        """
        For details, please refer to :func:`mindspore.ops.bitwise_right_shift`.
        """
        _cast = tensor_operator_registry.get('cast')
        other = _cast(other, self.dtype)
        return tensor_operator_registry.get('bitwise_right_shift')(self, other)

    def scatter_mul(self, indices, updates):
        """
        For details, please refer to :func:`mindspore.ops.scatter_mul`.
        """
        return tensor_operator_registry.get('tensor_scatter_mul')(self, indices, updates)

    def scatter_div(self, indices, updates):
        """
        For details, please refer to :func:`mindspore.ops.scatter_div`.
        """
        return tensor_operator_registry.get('tensor_scatter_div')(self, indices, updates)

    def ger(self, vec2):
        """
        For details, please refer to :func:`mindspore.ops.ger`.
        """
        return tensor_operator_registry.get('ger')(self, vec2)

    def tanh_(self):
        r"""
        Computes hyperbolic tangent of self inplace element-wise. The Tanh function is defined as:

        .. math::

            tanh(x_i) = \frac{\exp(x_i) - \exp(-x_i)}{\exp(x_i) + \exp(-x_i)} = \frac{\exp(2x_i) - 1}{\exp(2x_i) + 1},

        where :math:`x_i` is an element of the input Tensor.

        Tanh Activation Function Graph:

        .. image:: ../../images/Tanh.png
            :align: center

        .. warning::
            This is an experimental API that is subject ot change or deletion.

        Returns:
            Tensor, with the same type and shape as the `self`.

        Raises:
            TypeError: If `self` is not a Tensor.

        Supported Platforms:
            ``Ascend``

        Examples:
            >>> import mindspore
            >>> import numpy as np
            >>> from mindspore import Tensor
            >>> x = Tensor(np.array([1, 2, 3, 4, 5]), mindspore.float32)
            >>> output = x.tanh_()
            >>> print(output)
            [0.7615941 0.9640276 0.9950547 0.9993293 0.9999092]
        """
        return tensor_operator_registry.get('tanh_')(self)

    def cov(self, *, correction=1, fweights=None, aweights=None):
        r"""
        For details, please refer to :func:`mindspore.ops.cov`.
        """
        return tensor_operator_registry.get('cov')(self, correction=correction, fweights=fweights, aweights=aweights)

    def floor_(self):
        r"""
        In-place version of :func:`mindspore.Tensor.floor`.

        .. warning::
            This is an experimental API that is subject to change or deletion.
        """
        return tensor_operator_registry.get('floor_')(self)

    # pylint: disable=redefined-builtin
    def norm(self, ord=None, dim=None, keepdim=False, *, dtype=None):
        """
        For details, please refer to :func:`mindspore.ops.norm`.
        """
        return tensor_operator_registry.get('norm')(self, ord, dim, keepdim, dtype=dtype)

    def renorm(self, p, axis, maxnorm):
        """
        For details, please refer to :func:`mindspore.ops.renorm`.
        """
        return tensor_operator_registry.get("renorm")(self, p, axis, maxnorm)

    @deprecated("2.8.0", "Tensor.isclose", False)
    def approximate_equal(self, other, tolerance=1e-5):
        r"""
        `Tensor.approximate_equal` is deprecated from version 2.8.0 and will be removed in a future version,
        please use :func:`mindspore.Tensor.isclose` instead.
        """
        validator.check_isinstance("x", self, Tensor)
        validator.check_isinstance("y", other, Tensor)
        validator.check_isinstance("tolerance", tolerance, float)
        input_x = self.copy() if self.dtype == mstype.float32 else self.astype(mstype.float16)
        input_y = other.copy() if other.dtype == mstype.float32 else other.astype(mstype.float16)
        return tensor_operator_registry.get('__lt__')(tensor_operator_registry.get('abs')(
            tensor_operator_registry.get('__sub__')(input_x, input_y)
        ), tolerance)

    def logit(self, eps=None):
        r"""
        For details, please refer to :func:`mindspore.ops.logit`.
        """
        if eps is None:
            eps = -1.0
        validator.check_value_type('eps', eps, (float,), 'Tensor.logit')
        return tensor_operator_registry.get('logit')(self, eps)

    def logcumsumexp(self, axis):
        r"""
        For details, please refer to :func:`mindspore.ops.logcumsumexp`.

        .. warning::
            This is an experimental API that is subject to change or deletion.
        """
        return tensor_operator_registry.get('logcumsumexp')(self, axis)

    def logdet(self):
        r"""
        For details, please refer to :func:`mindspore.ops.logdet`.
        """
        return tensor_operator_registry.get('logdet')(self)

    def i0(self):
        r"""
        For details, please refer to :func:`mindspore.ops.bessel_i0`.
        """
        return tensor_operator_registry.get('i0')(self)

    def isposinf(self):
        r"""
        For details, please refer to :func:`mindspore.ops.isposinf`.
        """
        return tensor_operator_registry.get('isposinf')(self)

    def isreal(self):
        r"""
        For details, please refer to :func:`mindspore.ops.isreal`.
        """
        return tensor_operator_registry.get('isreal')(self)

    def inv(self):
        r"""
        For details, please refer to :func:`mindspore.ops.inv`.
        """
        return tensor_operator_registry.get('inv')(self)

    def invert(self):
        r"""
        For details, please refer to :func:`mindspore.ops.invert`.
        """
        return tensor_operator_registry.get('invert')(self)

    def amin(self, axis=None, keepdims=False, *, initial=None, where=None):
        """
        For details, please refer to :func:`mindspore.ops.amin`.
        """
        if axis is None:
            axis = ()
        return tensor_operator_registry.get('amin')(self, axis, keepdims, initial=initial, where=where)

    def reverse(self, axis):
        """
        For details, please refer to :func:`mindspore.ops.flip`.
        The `axis` parameter in `Tensor.reverse` is equivalent to the `dims` parameter in :func:`mindspore.ops.flip`.
        """
        return tensor_operator_registry.get('flip')(self, axis)

    def amax(self, axis=None, keepdims=False, *, initial=None, where=None):
        """
        For details, please refer to :func:`mindspore.ops.amax`.
        """
        if axis is None:
            axis = ()
        return tensor_operator_registry.get('amax')(self, axis, keepdims, initial=initial, where=where)

    def aminmax(self, *, axis=0, keepdims=False):
        r"""
        For details, please refer to :func:`mindspore.ops.aminmax`.
        """
        return tensor_operator_registry.get('aminmax')(self, axis=axis, keepdims=keepdims)

    def reverse_sequence(self, seq_lengths, seq_dim=0, batch_dim=0):
        """
        For details, please refer to :func:`mindspore.ops.reverse_sequence`.
        """
        return tensor_operator_registry.get("reverse_sequence")(self, seq_lengths, seq_dim, batch_dim)

    def col2im(self, output_size, kernel_size, dilation, padding_value, stride):
        """
        For details, please refer to :func:`mindspore.ops.col2im`.
        """
        return tensor_operator_registry.get('col2im')(self, output_size, kernel_size, dilation, padding_value, stride)

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
        return tensor_operator_registry.get('reshape')(self, other.shape)

    def ravel(self):
        """
        Return a contiguous flattened tensor.

        Returns:
            Tensor, a 1-D tensor, containing the same elements of the input.

        See also:
            - :func:`mindspore.Tensor.reshape`: Give a new shape to a tensor without changing its data.
            - :func:`mindspore.Tensor.flatten`: Return a copy of the tensor collapsed into one dimension.

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
        return self.reshape((-1,))

    def rot90(self, k, dims):
        r"""
        For details, please refer to :func:`mindspore.ops.rot90`.
        """
        return tensor_operator_registry.get('rot90')(self, k, dims)

    def deg2rad(self):
        r"""
        For details, please refer to :func:`mindspore.ops.deg2rad`.
        """
        return tensor_operator_registry.get('deg2rad')(self)

    def rad2deg(self):
        r"""
        For details, please refer to :func:`mindspore.ops.rad2deg`.
        """
        return tensor_operator_registry.get('rad2deg')(self)

    def copysign(self, other):
        r"""
        For details, please refer to :func:`mindspore.ops.copysign`.
        """
        return tensor_operator_registry.get('copysign')(self, other)

    def nelement(self):
        r"""
        Alias for :func:`mindspore.Tensor.numel`.
        """
        return self.size

    def numel(self):
        r"""
        For details, please refer to :func:`mindspore.ops.numel`.
        """
        return self._size

    def positive(self):
        """
        For details, please refer to :func:`mindspore.ops.positive`.
        """
        return tensor_operator_registry.get("positive")(self)

    def float_power(self, other):
        r"""
        For details, please refer to :func:`mindspore.ops.float_power`.
        """
        return tensor_operator_registry.get('float_power')(self, other)

    def fmax(self, other):
        r"""
        For details, please refer to :func:`mindspore.ops.fmax`.
        """
        return tensor_operator_registry.get('fmax')(self, other)

    def fmin(self, other):
        r"""
        This interface is deprecated from version 2.4 and will be removed in a future version.
        """
        return tensor_operator_registry.get('fmin')(self, other)

    def swapaxes(self, axis0, axis1):
        """
        For details, please refer to :func:`mindspore.ops.swapaxes`.
        """
        return tensor_operator_registry.get('swapaxes')(self, axis0, axis1)

    def swapdims(self, dim0, dim1):
        """
        For details, please refer to :func:`mindspore.ops.swapdims`.
        """
        return tensor_operator_registry.get('swapdims')(self, dim0, dim1)

    def slogdet(self):
        """
        For details, please refer to :func:`mindspore.ops.slogdet`.
        """
        return tensor_operator_registry.get('slogdet')(self)

    def expand_dims(self, axis):
        """
        For details, please refer to :func:`mindspore.ops.expand_dims`.
        """
        validator.check_is_int(axis, 'axis')
        validator.check_int_range(axis, -self.ndim - 1, self.ndim + 1, validator.INC_LEFT, 'axis')
        return tensor_operator_registry.get('expand_dims')(self, axis)

    def astype(self, dtype, copy=True):
        """
        Return a copy of the tensor, cast to a specified type.

        Args:
            dtype (Union[:class:`mindspore.dtype`, numpy.dtype, str]): Designated tensor dtype, can be in
                format of `mindspore.dtype.float32` or `numpy.float32` or `float32`.
            copy (bool, optional): By default, astype always returns a newly allocated
                tensor. If this is set to ``false`` , the input tensor is returned instead
                of a copy. Default:  ``True`` .

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
        dtype = _check_astype_and_convert(dtype)
        if not copy and dtype == self.dtype:
            return self
        return self.to(dtype)

    def argmax_with_value(self, axis=0, keep_dims=False):
        """
        Return the maximum values and their indices along the given axis of the tensor.

        Args:
            axis (Union[int, None], optional): Specify the axis for computation. If ``None`` , compute all elements in
                the tensor. Default ``0`` .
            keep_dims (bool, optional): Whether the output tensor has dim retained. Default ``False`` .

        Returns:
            Tuple(max, max_indices) of 2 tensors.

        Supported Platforms:
            ``Ascend`` ``GPU`` ``CPU``

        Examples:
            >>> import mindspore
            >>> x = mindspore.tensor([[9, 3, 4, 5],
            ...                       [5, 2, 7, 4],
            ...                       [8, 1, 3, 6]])
            >>> # case 1: By default, compute the maximum along axis 0.
            >>> x.argmax_with_value()
            (Tensor(shape=[4], dtype=Int64, value= [9, 3, 7, 6]),
             Tensor(shape=[4], dtype=Int64, value= [0, 0, 1, 2]))
            >>>
            >>> # case 2: Compute the maximum along axis 1.
            >>> x.argmax_with_value(axis=1)
            (Tensor(shape=[3], dtype=Int64, value= [9, 7, 8]),
             Tensor(shape=[3], dtype=Int64, value= [0, 2, 0]))
            >>>
            >>> # case 3: If keep_dims=True, the output shape will be same of that of the input.
            >>> x.argmax_with_value(axis=1, keep_dims=True)
            (Tensor(shape=[3, 1], dtype=Int64, value=
             [[9],
              [7],
              [8]]),
             Tensor(shape=[3, 1], dtype=Int64, value=
             [[0],
              [2],
              [0]]))
            >>>
            >>> # case 4: If axis=None, compute the maximum of all elements.
            >>> x.argmax_with_value(axis=None, keep_dims=True)
            (Tensor(shape=[], dtype=Int64, value= 9),
             Tensor(shape=[], dtype=Int64, value= 0))
        """
        if self.shape == ():
            return (self, Tensor(0))
        return tensor_operator_registry.get('argmax_with_value')(self, axis, keep_dims)

    def argmin_with_value(self, axis=0, keep_dims=False):
        """
        Return the minimum values and their indices along the given axis of the tensor.

        Args:
            axis (Union[int, None], optional): Specify the axis for computation. If ``None`` , compute all elements in
                the tensor. Default ``0`` .
            keep_dims (bool, optional): Whether the output tensor has dim retained. Default ``False`` .

        Returns:
            Tuple(min, min_indices) of 2 tensors.

        Supported Platforms:
            ``Ascend`` ``GPU`` ``CPU``

        Examples:
            >>> import mindspore
            >>> x = mindspore.tensor([[2, 5, 1, 6],
            ...                       [3, -7, -2, 4],
            ...                       [8, -4, 1, -3]])
            >>> # case 1: By default, compute the minimum along axis 0.
            >>> x.argmin_with_value()
            (Tensor(shape=[4], dtype=Int64, value= [ 2, -7, -2, -3]),
             Tensor(shape=[4], dtype=Int64, value= [0, 1, 1, 2]))
            >>>
            >>> # case 2: Compute the minimum along axis 1.
            >>> x.argmin_with_value(axis=1)
            (Tensor(shape=[3], dtype=Int64, value= [ 1, -7, -4]),
             Tensor(shape=[3], dtype=Int64, value= [2, 1, 1]))
            >>>
            >>> # case 3: If keep_dims=True, the output shape will be same of that of the input.
            >>> x.argmin_with_value(axis=1, keep_dims=True)
            (Tensor(shape=[3, 1], dtype=Int64, value=
             [[ 1],
              [-7],
              [-4]]),
             Tensor(shape=[3, 1], dtype=Int64, value=
             [[2],
              [1],
              [1]]))
            >>>
            >>> # case 4: If axis=None, compute the minimum of all elements.
            >>> x.argmin_with_value(axis=None, keep_dims=True)
            (Tensor(shape=[], dtype=Int64, value= -7),
             Tensor(shape=[], dtype=Int64, value= 0))
        """
        if self.shape == ():
            return (self, Tensor(0))
        return tensor_operator_registry.get('argmin_with_value')(self, axis, keep_dims)

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

    def inplace_update(self, v, indices):
        """
        For details, please refer to :func:`mindspore.ops.inplace_update`.
        """
        return tensor_operator_registry.get('inplace_update')(self, v, indices)

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
            x = logical_not_op(logical_not_op(x))
        elif origin_dtype in (mstype.int32, mstype.int64, mstype.uint32, mstype.uint64):
            tensor_move_op = tensor_operator_registry.get('_tensor_move')
            x = tensor_move_op(x)
        elif origin_dtype in mstype.complex_type:
            x = x + 0.0
        else:
            if origin_dtype != mstype.float64:
                x = x.astype("float32")
            x = x / 1.0

        x = x.astype(origin_dtype)
        return x

    def scatter_add_(self, dim, index, src):
        """
        Add all elements in `src` to the index specified by `index` to `self` along dimension specified by `dim`,
        `scatter_add` is an in-place operation.
        The ranks of `self`, `index` and `src` must be greater or equal to 1.

        For a 3-D tensor, the operation updates `self` as follows:

        .. code-block::

            self[index[i][j][k]][j][k] += src[i][j][k]  # if dim == 0

            self[i][index[i][j][k]][k] += src[i][j][k]  # if dim == 1

            self[i][j][index[i][j][k]] += src[i][j][k]  # if dim == 2

        .. warning::
            When deterministic computation is enabled, `index` can not be a non-contiguous Tensor; otherwise,
            deterministic results can not be guaranteed.

        Args:
            dim (int): Which dim to scatter. Accepted range is [-r, r) where r = rank(`self`).
            index (Tensor): The index of `self` to do scatter operation whose data type must
                be int32 or int64. Same rank as `self`. Except for the dimension
                specified by `dim`, size of each dimension of `index` must be less than or equal to the size of
                the corresponding dimension of `self`.
            src (Tensor): The tensor doing the scatter operation with `self`, has the same type as `self` and
                the size of each dimension must be greater than or equal to that of `index`.

        Returns:
            Tensor, has the same shape and type as `self`.

        Raises:
            TypeError: If `index` is neither int32 nor int64.
            ValueError: If anyone of the rank among `self`, `index` and `src` is less than 1.
            ValueError: If the ranks of `self`, `index` and `src` are not the same.
            ValueError: The size of any dimension of `index` except the dimension specified by `dim` is
                greater than the size of the corresponding dimension of `self`.
            ValueError: If the size of any dimension of `src` is less than that of `index`.

        Supported Platforms:
            ``Ascend``

        Examples:
            >>> import numpy as np
            >>> import mindspore as ms
            >>> from mindspore import Tensor
            >>> input = Tensor(np.array([[1, 2, 3, 4, 5]]), dtype=ms.float32)
            >>> src = Tensor(np.array([[8, 8]]), dtype=ms.float32)
            >>> index = Tensor(np.array([[2, 4]]), dtype=ms.int64)
            >>> out = input.scatter_add_(1, index, src)
            >>> print(out)
            [[1. 2. 11. 4. 13.]]
            >>> input = Tensor(np.zeros((5, 5)), dtype=ms.float32)
            >>> src = Tensor(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), dtype=ms.float32)
            >>> index = Tensor(np.array([[0, 0, 0], [2, 2, 2], [4, 4, 4]]), dtype=ms.int64)
            >>> out = input.scatter_add_(0, index, src)
            >>> print(out)
            [[1. 2. 3. 0. 0.]
            [0. 0. 0. 0. 0.]
            [4. 5. 6. 0. 0.]
            [0. 0. 0. 0. 0.]
            [7. 8. 9. 0. 0.]]
            >>> input = Tensor(np.zeros((5, 5)), dtype=ms.float32)
            >>> src = Tensor(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), dtype=ms.float32)
            >>> index = Tensor(np.array([[0, 2, 4], [0, 2, 4], [0, 2, 4]]), dtype=ms.int64)
            >>> out = input.scatter_add_(1, index, src)
            >>> print(out)
            [[1. 0. 2. 0. 3.]
            [4. 0. 5. 0. 6.]
            [7. 0. 8. 0. 9.]
            [0. 0. 0. 0. 0.]
            [0. 0. 0. 0. 0.]]
        """
        return tensor_operator_registry.get("inplace_scatter_add")(self, dim, index, src)

    def scatter_sub(self, indices, updates):
        """
        For details, please refer to :func:`mindspore.ops.tensor_scatter_sub`.
        """
        return tensor_operator_registry.get('tensor_scatter_sub')(self, indices, updates)

    def scatter_min(self, indices, updates):
        """
        For details, please refer to :func:`mindspore.ops.scatter_min`.
        """
        return tensor_operator_registry.get('tensor_scatter_min')(self, indices, updates)

    def scatter_max(self, indices, updates):
        """
        For details, please refer to :func:`mindspore.ops.scatter_max`.
        """
        return tensor_operator_registry.get('tensor_scatter_max')(self, indices, updates)

    def softmax(self, axis, dtype=None):
        """
        For details, please refer to :func:`mindspore.ops.softmax`.
        """
        return tensor_operator_registry.get('softmax')(self, axis, dtype=dtype)

    def fill(self, value):
        """
        `Tensor.fill` is deprecated, please use `ops.fill` instead.
        """
        if value is None:
            if self.dtype not in (mstype.float16, mstype.float32, mstype.float64):
                raise TypeError("For 'Tensor.fill', if the argument 'value' is None, the type of the original "
                                "tensor must be float, but got {}.".format(self.dtype))
            value = Tensor(float('nan')).astype("float32")
            return tensor_operator_registry.get("tile")()(value, self.shape).astype(self.dtype)
        return tensor_operator_registry.get("fill")(self.dtype, self.shape, value)

    def fills(self, value):
        """
        `Tensor.fills` is deprecated, please use `ops.fill` instead.
        """
        return tensor_operator_registry.get('fills')(self, value)

    def fill_diagonal(self, fill_value, wrap=False):
        """
        Fills the main diagonal of a Tensor with a specified value and returns the result.
        The input has at least 2 dimensions, and all dimensions of input must be equal in length
        when the dimension of input is greater than 2.

        .. warning::
            This is an experimental API that is subject to change or deletion.

        Args:
            fill_value (float): The value to fill with the diagonal of `self`.
            wrap (bool, optional): Controls whether the diagonal elements continue onto the
                remaining rows in case of a tall matrix(a matrix has more rows than columns). Default: ``False``.

        Returns:
            - **y** (Tensor) - Tensor, has the same shape and data type as `self`.

        Raises:
            TypeError: If data type of `self` is not one of the following: float32, int32, int64.
            ValueError: If the dimension of `self` is not greater than 1.
            ValueError: If the size of each dimension is not equal, when the dimension is greater than 2.

        Supported Platforms:
            ``Ascend`` ``GPU`` ``CPU``

        Examples:
            >>> import numpy as np
            >>> import mindspore
            >>> from mindspore import Tensor
            >>> x = Tensor(np.ones((6, 3)), mindspore.float32)
            >>> output = x.fill_diagonal(5.0, wrap=True)
            >>> print(output)
            [[5. 1. 1.]
             [1. 5. 1.]
             [1. 1. 5.]
             [1. 1. 1.]
             [5. 1. 1.]
             [1. 5. 1.]]
        """
        return tensor_operator_registry.get('fill_diagonal')(fill_value, wrap)(self)

    @deprecated("2.8.0", "Tensor.max() - Tensor.min()", False)
    def ptp(self, axis=None, keepdims=False):
        """
        `Tensor.ptp` is deprecated from version 2.8.0 and will be removed in a future version,
        please use `Tensor.max() - Tensor.min()` instead.

        The name of the function comes from the acronym for "peak to peak". Calculate the difference between the
        maximum value and the minimum value along the axis.

        Note:
            Numpy argument `out` is not supported.

        Args:
            axis (Union[None, int, tuple(int)], optional): Axis or axes along which the range is computed.
                The default is to compute the difference between
                the maximum value and the minimum value of the flattened tensor. Default: ``None`` .
            keepdims (bool, optional): If this is set to ``True`` , the axes which are reduced are left in the result as
                dimensions with size one. With this option, the result will broadcast correctly against the tensor.
                Default is ``False`` .

        Returns:
            Tensor.

        Raises:
            TypeError: If `self` is not a tensor, or `axis` and `keepdims` have types not specified above.

        Supported Platforms:
            Deprecated

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

    def clamp_(self, min=None, max=None):
        r"""
        In-place version of :func:`mindspore.Tensor.clamp`.

        .. warning::
            This is an experimental API that is subject to change or deletion.
        """
        return tensor_operator_registry.get('clamp_')(self, min, max)

    def init_data(self, slice_index=None, shape=None, opt_shard_group=None):
        """
        Get the tensor format data of this Tensor.

        Note:
            The init_data function can be called once for the same tensor.

        Args:
            slice_index (int): Slice index of a parameter's slices.
                It is used when initialize a slice of a parameter, it guarantees that devices
                using the same slice can generate the same tensor. Default: ``None``.
            shape (list[int]): Shape of the slice, it is used when initialize a slice of the parameter.
                Default: ``None``.
            opt_shard_group(str): Optimizer shard group which is used in auto or semi auto parallel mode
                to get one shard of a parameter's slice. For more information about optimizer parallel, please refer to:
                `Optimizer Parallel
                <https://www.mindspore.cn/tutorials/en/master/parallel/optimizer_parallel.html>`_.
                Default: ``None``.

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

        from mindspore.common.initializer import Zero as ZeroInitializer

        is_qint4x2 = self.dtype == mstype.qint4x2
        try:
            dtype_ = mstype.int8 if is_qint4x2 else self.dtype
            if isinstance(self.init, ZeroInitializer):
                data = np.zeros(shape, dtype=mstype._dtype_to_nptype(dtype_))  # pylint:disable=protected-access
            else:
                data = np.ndarray(shape, dtype=mstype._dtype_to_nptype(dtype_))  # pylint:disable=protected-access
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
                self.need_set_seed = slice_index is not None
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
            if (not isinstance(self.init, ZeroInitializer)) \
                    and not is_reboot_node():
                self.init(data)
        self.init = None

        self.assign_value(TensorPy_.from_numpy(data))

        if is_qint4x2:
            self.set_dtype(mstype.qint4x2)

        return self

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
            new_shape (Union[int, tuple(int)]): Shape of resized tensor.

        Returns:
            Tensor.

        See also:
            - :func:`mindspore.Tensor.reshape`: Give a new shape to a tensor without changing its data.
            - :func:`mindspore.Tensor.repeat`: Repeat elements of a tensor.

        Supported Platforms:
            ``Ascend`` ``GPU`` ``CPU``

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
            res = tensor_operator_registry.get('concatenate')((flattened, pad_val), 0)
        else:
            res = flattened[:new_size]
        return res.reshape(new_shape)

    def det(self):
        r"""
        This interface is deprecated from version 2.4 and will be removed in a future version.
        """
        return tensor_operator_registry.get('det')(self)

    def diff(self, n=1, axis=-1, prepend=None, append=None):
        r"""
        For details, please refer to :func:`mindspore.ops.diff`.
        """
        return tensor_operator_registry.get('diff')(self, n, axis, prepend, append)

    def argwhere(self):
        r"""
        For details, please refer to :func:`mindspore.ops.argwhere`.
        """
        return tensor_operator_registry.get('argwhere')(self)

    def moveaxis(self, source, destination):
        r"""
        For details, please refer to :func:`mindspore.ops.moveaxis`.
        """
        return tensor_operator_registry.get('moveaxis')(self, source, destination)

    def movedim(self, source, destination):
        r"""
        For details, please refer to :func:`mindspore.ops.movedim`.
        """
        return tensor_operator_registry.get('movedim')(self, source, destination)

    def digamma(self):
        r"""
        For details, please refer to :func:`mindspore.ops.digamma`.
        """
        return tensor_operator_registry.get('digamma')(self)

    def lgamma(self):
        r"""
        This interface is deprecated from version 2.4 and will be removed in a future version.
        """
        return tensor_operator_registry.get('lgamma')(self)

    def diagonal(self, offset=0, axis1=0, axis2=1):
        """
        For details, please refer to :func:`mindspore.ops.diagonal`.
        The parameter `axis1` of the current interface is the same as the parameter `dim1` of the reference interface,
        the parameter `axis2` of the current interface is the same as the parameter `dim2` of the reference interface.
        """
        return tensor_operator_registry.get('diagonal')(self, offset, axis1, axis2)

    def diagonal_scatter(self, src, offset=0, dim1=0, dim2=1):
        r"""
        For details, please refer to :func:`mindspore.ops.diagonal_scatter`.
        """
        return tensor_operator_registry.get('diagonal_scatter')(self, src, offset, dim1, dim2)

    def trace(self, offset=0, axis1=0, axis2=1, dtype=None):
        """
        Return the sum along diagonals of the tensor.

        Args:
            offset (int, optional): Offset of the diagonal from the main diagonal.
                Can be positive or negative. Defaults to main diagonal. Default: ``0`` .
            axis1 (int, optional): Axis to be used as the first axis of the 2-D
                sub-arrays from which the diagonals should be taken. Defaults to
                first axis (0). Default: ``0`` .
            axis2 (int, optional): Axis to be used as the second axis of the 2-D
                sub-arrays from which the diagonals should be taken. Defaults to
                second axis. Default: ``1`` .
            dtype (:class:`mindspore.dtype`, optional): defaults to None. Overrides the dtype of the
                output Tensor. Default: ``None`` .

        Returns:
            Tensor, the sum along diagonals.

        Raises:
            ValueError: If the input tensor has less than two dimensions.

        See also:
            - :func:`mindspore.Tensor.diagonal`: Return specified diagonals.

        Supported Platforms:
            ``Ascend`` ``GPU`` ``CPU``

        Examples:
            >>> import numpy as np
            >>> from mindspore import Tensor
            >>> x = Tensor(np.eye(3, dtype=np.float32))
            >>> print(x.trace())
            3.0
        """
        return tensor_operator_registry.get('tracev2')(self, offset, axis1, axis2, dtype)

    def choose(self, choices, mode='clip'):
        """
        Construct a tensor from an index tensor and a list of tensors to choose from.

        Args:
            choices (Union[tuple, list, Tensor]): Choice tensors. The input tensor and all of the
                `choices` must be broadcasted to the same shape. If `choices` is itself a tensor,
                then its outermost dimension (i.e., the one corresponding to ``choices.shape[0]``)
                is taken as defining the "sequence".
            mode (str, optional): Specifies how indices outside
                ``[0, n-1]`` will be treated. Support ``'raise'``, ``'wrap'``, ``'clip'``.

                - ``raise``: Raises an error;

                - ``wrap``: Wraps around;

                - ``clip``: Clips to the range. The values greater than n-1 will be mapped to n-1.
                  Note that this mode disables indexing with negative numbers.

                Default: ``'clip'``.

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
            choices = tensor_operator_registry.get('broadcast_to')(choices, (choices.shape[0],) + shape_choice)
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
                tmp.append(tensor_operator_registry.get('broadcast_to')(choice, shape_choice))
            choices = tensor_operator_registry.get('stack')(tmp, 0)

        if self.ndim == 0 or choices.ndim == 0:
            raise ValueError(f"For 'Tensor.choose', the original tensor and the argument 'choices' cannot be scalars."
                             f" Their dimensions should all be > 0, but got the original tensor's dimension "
                             f"{self.ndim}, 'choices' dimension {choices.ndim}.")
        a = tensor_operator_registry.get('broadcast_to')(self, shape_choice)
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
            dim_grid = tensor_operator_registry.get('broadcast_to')(dim_grid.reshape(dim_shape), a.shape)
            grids.append(dim_grid)
        grid = tensor_operator_registry.get('stack')(grids, -1)
        indices = tensor_operator_registry.get('concatenate')((a.reshape(a.shape + (1,)), grid), -1)
        return tensor_operator_registry.get('gather_nd')(choices, indices).astype(dtype)

    def searchsorted(self, v, side='left', sorter=None):
        """
        For details, please refer to :func:`mindspore.ops.searchsorted`.
        """
        if side not in ('left', 'right'):
            raise ValueError(f"For 'Tensor.searchsorted', the argument 'side' should be one of in "
                             f"['left', 'right'], but got {side}.")
        if not isinstance(v, Tensor):
            v = tensor_operator_registry.get('make_tensor')(v)
        if sorter is not None:
            if not isinstance(sorter, (int, list, tuple, Tensor)):
                raise TypeError("For Tensor.searchsorted, the type of the argument 'sorter' must be one of 'int', "
                                "'list', 'tuple', 'Tensor', but got {}.".format(type(sorter)))
            if not isinstance(sorter, Tensor):
                sorter = tensor_operator_registry.get('make_tensor')(sorter)
            if sorter.size != self.size:
                raise ValueError('The size of sorter must be the same as the Tensor')

        dtype = mstype.int32
        right = side == 'right'
        search_sorted_ = tensor_operator_registry.get('searchsorted')(dtype, right)
        return search_sorted_(self, v, sorter)

    def gather_nd(self, indices):
        r"""
        For details, please refer to :func:`mindspore.ops.gather_nd`.
        """
        validator.check_value_type('indices', indices, (Tensor, TensorPy_,), 'Tensor.gather_nd')
        return tensor_operator_registry.get('gather_nd')(self, indices)

    def uniform(self, from_=0., to=1., generator=None):
        r"""
        Generates random numbers that follows a uniform distribution within the half-open interval :math:`[from\_, to)`.

        .. math::
            P(x) = \frac{1}{to - from\_}

        Args:
            from\_ (number, optional): The lower bound of the interval. Default: ``0.`` .
            to (number, optional): The upper bound of the interval. Default: ``1.`` .
            generator (Generator, optional): The random seed. Default: ``None`` .

        Returns:
            Tensor, with the same shape as tensor.

        Raises:
            TypeError: If `from_` is larger than `to`.

        Supported Platforms:
            ``Ascend``

        Examples:
            >>> import mindspore
            >>> x = mindspore.ops.ones((4, 2))
            >>> generator = mindspore.Generator()
            >>> generator.manual_seed(100)
            >>> output = x.uniform(1., 2., generator)
            >>> print(output.shape)
            (4, 2)
        """
        return tensor_operator_registry.get('uniform')(self, from_, to, generator)

    def uniform_(self, from_=0, to=1, *, generator=None):
        r"""
        Update the `self` Tensor in place by generating random numbers sampled from uniform distribution in the
        half-open interval :math:`[from\_, to)`.

        .. math::
            P(x) = \frac{1}{to - from\_}

        .. warning::
            This is an experimental API that is subject to change or deletion.

        Args:
            from\_ (Union[number.Number, Tensor], optional): The lower bound of the uniform distribution, it can be a
                scalar value or a tensor of any dimension with a single element. Default: ``0``.
            to (Union[number.Number, Tensor], optional): The upper bound of the uniform distribution, it can be a
                scalar value or a tensor of any dimension with a single element. Default: ``1``.

        Keyword Args:
            generator (:class:`mindspore.Generator`, optional): a pseudorandom number generator.
                Default: ``None``, uses the default pseudorandom number generator.

        Returns:
            Return `self` Tensor.

        Raises:
            TypeError: If `from_` or `to` is neither a number nor a Tensor.
            TypeError: If dtype of `from` or `to` is not one of: bool, int8, int16, int32, int64, uint8, float32,
                float64.
            ValueError: If `from_` or `to` is Tensor but contains multiple elements.
            RuntimeError: If `from_` is larger than `to`.

        Supported Platforms:
            ``Ascend``

        Examples:
            >>> import mindspore
            >>> x = mindspore.ops.ones((4, 2))
            >>> generator = mindspore.Generator()
            >>> generator.manual_seed(100)
            >>> output = x.uniform_(1., 2., generator=generator)
            >>> print(output.shape)
            (4, 2)
        """
        return tensor_operator_registry.get('uniform_')(self, from_=from_, to=to, generator=generator)

    def exponential_(self, lambd=1, *, generator=None):
        r"""
        Fills `self` tensor with elements drawn from the exponential distribution:

        .. math::
            f(x) = \lambda \exp(-\lambda x)

        .. warning::
            - It is only supported on Atlas A2 Training Series Products.
            - This is an experimental API that is subject to change or deletion.

        Args:
            lambd (float, optional): Parameters of exponential distribution. Default: ``1``.

        Keyword Args:
            generator (Generator, optional): a pseudorandom number generator.
                Default: ``None`` .

        Returns:
            Tensor, with same shape and same data type with input.

        Supported Platforms:
            ``Ascend``

        Examples:
            >>> import mindspore
            >>> x = mindspore.Tensor([1, 2, 3.0])
            >>> out = x.exponential_(2)
            >>> print(out.shape)
            (3,)
        """
        return tensor_operator_registry.get('exponential_')(self, lambd=lambd, generator=generator)

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
            >>> import numpy as np
            >>> import mindspore
            >>> from mindspore import Tensor
            >>> x = Tensor(np.random.randn(3, 3, 3, 3, 3, 3), mindspore.float32)
            >>> output = x.sum_to_size((1, 3, 1, 3))
            >>> print(output.shape)
            (1, 3, 1, 3)
        """
        x = self
        if len(size) == 1 and isinstance(size[0], tuple):
            size = size[0]
        shape_x = x.shape
        if len(size) > x.ndim:
            raise ValueError(f"For sum_to_size, size {size} is not expandable to the tensor size {shape_x}.")
        pre_len = 0
        pre_axis = []
        if len(size) < x.ndim:
            pre_len = x.ndim - len(size)
            pre_axis = list(range(pre_len))
        axes = pre_axis
        for i, element in enumerate(size):
            if element != x.shape[i + pre_len] and element == 1:
                axes.append(i + pre_len)
            elif element != x.shape[i + pre_len]:
                raise ValueError(f"For sum_to_size, size {size} is not expandable to the tensor size {shape_x}.")
        if axes:
            return x.sum(tuple(axes), keepdims=True).reshape(size)
        return x

    def nanmean(self, axis=None, keepdims=False, *, dtype=None):
        r"""
        For details, please refer to :func:`mindspore.ops.nanmean`.
        """
        return tensor_operator_registry.get('nanmean')(self, axis, keepdims, dtype=dtype)

    def nanmedian(self, axis=-1, keepdims=False):
        r"""
        For details, please refer to :func:`mindspore.ops.nanmedian`.
        """
        return tensor_operator_registry.get('nanmedian')(self, axis, keepdims)

    def bernoulli(self, *, generator=None):
        r"""
        For details, please refer to :func:`mindspore.mint.bernoulli`.
        """
        return tensor_operator_registry.get('bernoulli')(self, generator=generator)

    def bernoulli_(self, p=0.5, *, generator=None):
        r"""
        Fills each location of self with an independent sample from Bernoulli(p).

        Args:
            p (Union[number.Number, Tensor], optional): `p` should either be a scalar or tensor containing
                probabilities to be used for drawing the binary random number, between ``0`` and ``1`` .
                If it is a tensor, `p` must be floating point. Default: ``0.5`` .

        Keyword Args:
            generator (:class:`mindspore.Generator`, optional): a pseudorandom number generator.
                Default: ``None`` , uses the default pseudorandom number generator.

        Returns:
            The input tensor.

        Supported Platforms:
            ``Ascend``

        Examples:
            >>> from mindspore import Tensor
            >>> x = Tensor([[2, 3, 4], [1, 2, 3]])
            >>> p = 0.1
            >>> print(x.bernoulli_(p).shape)
            (2, 3)
        """
        return tensor_operator_registry.get('bernoulli_')(self, p, generator=generator)

    def random_(self, from_=0, to=None, *, generator=None):
        r"""
        Fill the tensor with numbers sampled from a discrete uniform distribution over an
        interval :math:`[from\_, to-1]`.

        .. warning::
            This is an experimental API that is subject to change or deletion.

        Args:
            from\_ (Union[numbers.Number, Tensor], optional): the lower bound of the generated random number.
                It can be a scalar value or a Tensor of any dimension with only a single element. Default: 0.
            to (Union[numbers.Number, Tensor], optional): the upper bound of the generated random number.
                By default it's the upper limit of the input data type.
                It can be a scalar value or a Tensor of any dimension with only a single element.
                Default: ``None``.

        Keyword Args:
            generator (:class:`mindspore.Generator`, optional): a pseudorandom number generator.
                Default: ``None``, uses the default pseudorandom number generator.

        Returns:
            The input tensor.

        Raises:
            TypeError: If `from_` or `to` is not integer.
            RuntimeError: If `from_` >= `to`.

        Supported Platforms:
            ``Ascend``

        Examples:
            >>> from mindspore import Tensor
            >>> a = Tensor([[2, 3, 4], [1, 2, 3]])
            >>> from_ = 0
            >>> to = 5
            >>> print(a.random_(from_, to).shape)
            (2, 3)
        """
        return tensor_operator_registry.get('random_')(self, from_=from_, to=to, generator=generator)

    def random_categorical(self, num_sample, seed=0, dtype=mstype.int64):
        r"""
        For details, please refer to :func:`mindspore.ops.random_categorical`.
        """
        validator.check_is_int(num_sample, 'num_sample')
        validator.check_is_int(seed, 'seed')
        return tensor_operator_registry.get('random_categorical')(self, num_sample, seed, dtype)

    def gather_elements(self, dim, index):
        """
        For details, please refer to :func:`mindspore.ops.gather_elements`.
        """
        validator.check_value_type('index', index, (Tensor, TensorPy_,), 'Tensor.gather_elements')
        return tensor_operator_registry.get('gather_elements')(self, dim, index)

    def nonzero(self, *, as_tuple=False):
        r"""
        Return the positions of all non-zero values.

        Note:
           The rank of `self`.

           - Ascend: its rank can be equal to 0 except GE backend.
           - CPU/GPU: its rank should be greater than or eaqual to 1.

        Keyword Args:
            as_tuple (bool, optional): Whether the output is tuple.
                If ``False`` , return Tensor. Default: ``False`` .
                If ``True`` , return Tuple of Tensor, only support ``Ascend`` .

        Returns:
            - If `as_tuple` is ``False``, return the Tensor, a 2-D Tensor whose data type is int64,
              containing the positions of all non-zero values of the `self` .
            - If `as_tuple` is ``True``, return the Tuple of Tensor and data type is int64.
              The Tuple length is the dimension of the `self` tensor,
              and each element is the 1D tensor of the subscript of all non-zero elements of
              the `self` tensor in that dimension.

        Raises:
            TypeError: If `self` is not Tensor.
            TypeError: If `as_tuple` is not bool.
            RuntimeError: On GPU or CPU or Ascend GE backend, if dim of `input` equals to 0.

        Supported Platforms:
            ``Ascend`` ``GPU`` ``CPU``

        Examples:
            >>> import mindspore
            >>> import numpy as np
            >>> from mindspore import Tensor
            >>> x = Tensor(np.array([[[1,  0], [-5, 0]]]), mindspore.int32)
            >>> output = x.nonzero()
            >>> print(output)
            [[0 0 0]
            [0 1 0]]
            >>> x = Tensor(np.array([1, 0, 2, 0, 3]), mindspore.int32)
            >>> output = x.nonzero(as_tuple=False)
            >>> print(output)
            [[0]
            [2]
            [4]]
            >>> x = Tensor(np.array([[[1,  0], [-5, 0]]]), mindspore.int32)
            >>> output = x.nonzero(as_tuple=True)
            >>> print(output)
            (Tensor(shape=[2], dtype=Int64, value=[0, 0]),
            Tensor(shape=[2], dtype=Int64, value=[0, 1]),
            Tensor(shape=[2], dtype=Int64, value=[0, 0]))
            >>> x = Tensor(np.array([1, 0, 2, 0, 3]), mindspore.int32)
            >>> output = x.nonzero(as_tuple=True)
            >>> print(output)
            (Tensor(shape=[3], dtype=Int64, value=[0, 2, 4]), )
        """
        return tensor_operator_registry.get('nonzero')(self, as_tuple=as_tuple)

    def svd(self, full_matrices=False, compute_uv=True):
        """
        For details, please refer to :func:`mindspore.ops.svd`.
        """
        svd_op = tensor_operator_registry.get("svd")
        if compute_uv:
            return svd_op(full_matrices, compute_uv)(self)

        s, _, _ = svd_op(full_matrices, compute_uv)(self)
        return s

    def heaviside(self, values):
        r"""
        For details, please refer to :func:`mindspore.ops.heaviside`.
        """
        return tensor_operator_registry.get('heaviside')(self, values)

    def hypot(self, other):
        r"""
        For details, please refer to :func:`mindspore.ops.hypot`.
        """
        return tensor_operator_registry.get('hypot')(self, other)

    def soft_shrink(self, lambd=0.5):
        r"""
        For details, please refer to :func:`mindspore.ops.soft_shrink`.
        """
        return tensor_operator_registry.get('soft_shrink')(self, lambd)

    def matrix_determinant(self):
        r"""
        For details, please refer to :func:`mindspore.ops.matrix_determinant`.
        """
        return tensor_operator_registry.get('matrix_determinant')(self)

    def log_matrix_determinant(self):
        r"""
        For details, please refer to :func:`mindspore.ops.log_matrix_determinant`.
        """
        return tensor_operator_registry.get('log_matrix_determinant')(self)

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
        return tensor_operator_registry.get('dense_to_sparse_csr')(self)

    def tolist(self):
        r"""
        Convert a Tensor to List. If the input is Tensor scalar, a Python scalar will be returned.

        Returns:
            List or Python scalar.

        Supported Platforms:
            ``Ascend`` ``GPU`` ``CPU``

        Examples:
            >>> import mindspore as ms
            >>> x = ms.Tensor([[1, 2, 3], [4, 5, 6]])
            >>> out1 = x.tolist()
            >>> print(out1)
            [[1, 2, 3], [4, 5, 6]]
            >>> out2 = x[0][0].tolist()
            >>> print(out2)
            1
        """
        if self.ndim == 1 and self.size == 0:
            return []
        return self._tolist()

    def unsorted_segment_min(self, segment_ids, num_segments):
        r"""
        For details, please refer to :func:`mindspore.ops.unsorted_segment_min`.
        """
        return tensor_operator_registry.get('unsorted_segment_min')(self, segment_ids, num_segments)

    def unsorted_segment_max(self, segment_ids, num_segments):
        r"""
        For details, please refer to :func:`mindspore.ops.unsorted_segment_max`.
        """
        return tensor_operator_registry.get('unsorted_segment_max')(self, segment_ids, num_segments)

    def unsorted_segment_prod(self, segment_ids, num_segments):
        r"""
        For details, please refer to :func:`mindspore.ops.unsorted_segment_prod`.
        """
        return tensor_operator_registry.get('unsorted_segment_prod')(self, segment_ids, num_segments)

    def unique_consecutive(self, return_inverse=False, return_counts=False, dim=None):
        """
        For details, please refer to :func:`mindspore.ops.unique_consecutive`.
        """
        output, idx, counts = \
            tensor_operator_registry.get("unique_consecutive")(return_inverse, return_counts, dim)(self)
        if return_inverse and return_counts:
            return output, idx, counts
        if return_inverse:
            return output, idx
        if return_counts:
            return output, counts
        return output

    @deprecated("2.8.0", "ops.unique", False)
    def unique_with_pad(self, pad_num):
        """
        `Tensor.unique_with_pad` is deprecated from version 2.8.0 and will be removed in a future version,
        please use :func:`mindspore.Tensor.unique` combined with :func:`mindspore.Tensor.pad` instead.
        """
        return tensor_operator_registry.get("unique_with_pad")(self, pad_num)

    def diagflat(self, offset=0):
        r"""
        For details, please refer to :func:`mindspore.ops.diagflat`.
        """
        return tensor_operator_registry.get('diagflat')(self, offset)

    def xdivy(self, y):
        r"""
        For details, please refer to :func:`mindspore.ops.xdivy`.
        """
        return tensor_operator_registry.get("xdivy")(self, y)

    def tensor_split(self, indices_or_sections, axis=0):
        """
        For details, please refer to :func:`mindspore.ops.tensor_split`.
        """
        return tensor_operator_registry.get('tensor_split')(self, indices_or_sections, axis)

    def vsplit(self, indices_or_sections):
        """
        For details, please refer to :func:`mindspore.ops.vsplit`.
        """

        return tensor_operator_registry.get('vsplit')(self, indices_or_sections)

    def hsplit(self, indices_or_sections):
        """
        For details, please refer to :func:`mindspore.ops.hsplit`.
        """
        return tensor_operator_registry.get('hsplit')(self, indices_or_sections)

    def dsplit(self, indices_or_sections):
        """
        For details, please refer to :func:`mindspore.ops.dsplit`.
        """
        return tensor_operator_registry.get('dsplit')(self, indices_or_sections)

    def eigvals(self):
        r"""
        For details, please refer to :func:`mindspore.ops.eigvals`.

        .. warning::
            This is an experimental API that is subject to change or deletion.
        """
        return tensor_operator_registry.get("eigvals")()(self)

    def top_k(self, k, sorted=True):
        r"""
        `Tensor.top_k` is deprecated, please use `Tensor.topk` instead.
        """
        validator.check_is_int(k, 'k')
        validator.check_bool(sorted, 'sorted')
        return tensor_operator_registry.get("top_k")(self, k, sorted)

    def bmm(self, mat2):
        r"""
        For details, please refer to :func:`mindspore.ops.bmm`.
        """
        return tensor_operator_registry.get('bmm')(self, mat2)

    def to_(self, device=None, non_blocking=False):
        r"""
        In-place version of :func:`mindspore.Tensor.to`, converts the device of the original tensor to the specified
        `device` and returns the tensor.

        .. warning::
            This is an experimental API that is subject to change or deletion.

        Args:
            device (str, optional): Specifies the hardware device where the output tensor is located.
                Default: ``None``.
            non_blocking (bool, optional): Whether to perform asynchronous data conversion.
                If ``True``, the data conversion is asynchronous; if ``False``, the data conversion is synchronous.
                Default: ``False``.

        Returns:
            Tensor, the modified `self` itself, which is stored on the specified `device`.

        Examples:
            >>> import mindspore as ms
            >>> x = ms.Tensor([1, 2, 3, 4])
            >>> x.to_(device="Ascend", non_blocking=True)
            >>> print(x.device)
            "Ascend:0"
        """
        if not isinstance(non_blocking, bool):
            raise ValueError(f"The type of 'non_blocking' must be bool, but got {non_blocking}")
        if device not in ("Ascend", "CPU"):
            raise ValueError(f"The value of 'to' must be one of ['Ascend', 'CPU'], but got {device}")
        copy_data = self.to(device=device, non_blocking=non_blocking, copy=True)
        self.data.delete_(non_blocking)  # pylint: disable=E0203
        self.data = copy_data
        return self

    def delete_(self, non_blocking=False):
        r"""
        Actively releases the memory of the tensor on the `device` or `host`.

        .. warning::
            This is an experimental API that is subject to change or deletion.

        Args:
            non_blocking (bool, optional): Whether to release memory asynchronously.
                If ``True``, the memory is released asynchronously; if ``False``, the memory is released synchronously.
                Default: ``False``.

        Returns:
            Tensor, the modified `self` itself, with its data memory already released.

        Examples:
            >>> import mindspore as ms
            >>> x = ms.Tensor([1, 2, 3, 4])
            >>> y = x.to(device="Ascend", non_blocking=True)
            >>> x.delete_(non_blocking=True)
            Tensor(shape=[4], dtype=Int64, value= [0, 0, 0, 0])
            >>> x.data = y
            >>> print(x.device)
            "Ascend:0"
        """
        if not isinstance(non_blocking, bool):
            raise ValueError(f"The type of 'non_blocking' must be bool, but got {non_blocking}")
        sync = not non_blocking
        return tensor_operator_registry.get('delete_')()(self, sync)

    def type(self, dtype=None):
        r"""
        Change the dtype of the Tensor to the `dtype` . Return the type if `dtype` is ``None`` .

        Args:
            dtype (mindspore.dtype, optional): The specified dtype of output tensor. Default: ``None``.

        Returns:
            Tensor or str. If `dtype` is ``None`` , return a str, which describes the dtype of Tensor.
            If `dtype` is not ``None`` , then return a Tensor, and the dtype of returned Tensor is `dtype` .

        Supported Platforms:
            ``Ascend`` ``GPU`` ``CPU``

        Examples:
            >>> import mindspore
            >>> from mindspore import Tensor
            >>> x = Tensor([[1.2, 2], [3.4, 4]], dtype=mindspore.float32)
            >>> print(x.type())
            Float32
            >>> print(x.type(dtype=mindspore.int32))
            [[1 2]
             [3 4]]
        """
        if dtype is None:
            return str(self.dtype)
        return self.astype(dtype)

    def type_as(self, other):
        r"""
        Returns self tensor cast to the type of the with the input other tensor.

        Note:
            When converting complex numbers to boolean type, the imaginary part of the complex number is not
            taken into account. As long as the real part is non-zero, it returns True; otherwise, it returns False.

        Args:
            other (Tensor): The tensor whose data type is specified.
                The shape of tensor is :math:`(x_0, x_1, ..., x_R)`.

        Returns:
            Tensor, the shape of tensor is the same as `self`, :math:`(x_0, x_1, ..., x_R)`.

        Raises:
            TypeError: If `other` is not a Tensor.

        Supported Platforms:
            ``Ascend`` ``GPU`` ``CPU``

        Examples:
            >>> import mindspore
            >>> import numpy as np
            >>> from mindspore import Tensor
            >>> input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
            >>> self = Tensor(input_np)
            >>> other_np = np.random.randn(2, 3, 4).astype(np.int32)
            >>> other = Tensor(other_np)
            >>> output = self.type_as(other)
            >>> print(output.dtype)
            Int32
            >>> print(output.shape)
            (2, 3, 4, 5)
        """
        if self.dtype == other.dtype:
            return self
        return TensorPy_.type_as(self, other)

    def bool(self):
        r"""
        Converts input tensor dtype to `bool`.
        If the value in tensor is zero, it will be `False`, otherwise it will be `True`.

        Returns:
            Tensor, converted to the `bool` dtype.

        Supported Platforms:
            ``Ascend`` ``GPU`` ``CPU``

        Examples:
            >>> import numpy as np
            >>> import mindspore
            >>> from mindspore import Tensor
            >>> input_x = Tensor(np.ones([2,2]), mindspore.float32)
            >>> output = input_x.bool()
            >>> print(output.dtype)
            Bool
        """
        return self.to(mstype.bool_)

    def float(self):
        r"""
        Converts input tensor dtype to `float32`.

        Returns:
            Tensor, converted to the `float32` dtype.

        Supported Platforms:
            ``Ascend`` ``GPU`` ``CPU``

        Examples:
            >>> import numpy as np
            >>> import mindspore
            >>> from mindspore import Tensor
            >>> input_x = Tensor(np.ones([2,2]), mindspore.int32)
            >>> output = input_x.float()
            >>> print(output.dtype)
            Float32
        """
        return self.to(mstype.float32)

    def bfloat16(self):
        r"""
        Converts input tensor dtype to `bfloat16`.

        Returns:
            Tensor, converted to the `bfloat16` dtype.

        Supported Platforms:
            ``Ascend`` ``GPU`` ``CPU``

        Examples:
            >>> import numpy as np
            >>> import mindspore
            >>> from mindspore import Tensor
            >>> input_x = Tensor(np.ones([2,2]), mindspore.int32)
            >>> output = input_x.bfloat16()
            >>> print(output.dtype)
            BFloat16
        """
        return self.to(mstype.bfloat16)

    def double(self):
        r"""
        Converts input tensor dtype to `float64`.

        Returns:
            Tensor, converted to the `float64` dtype.

        Supported Platforms:
            ``Ascend`` ``GPU`` ``CPU``

        Examples:
            >>> import numpy as np
            >>> import mindspore
            >>> from mindspore import Tensor
            >>> input_x = Tensor(np.ones([2,2]), mindspore.int32)
            >>> output = input_x.double()
            >>> print(output.dtype)
            Float64
        """
        return self.to(mstype.float64)

    def half(self):
        r"""
        Converts input tensor dtype to `float16`.

        Returns:
            Tensor, converted to the `float16` dtype.

        Supported Platforms:
            ``Ascend`` ``GPU`` ``CPU``

        Examples:
            >>> import numpy as np
            >>> import mindspore
            >>> from mindspore import Tensor
            >>> input_x = Tensor(np.ones([2,2]), mindspore.int32)
            >>> output = input_x.half()
            >>> print(output.dtype)
            Float16
        """
        return self.to(mstype.float16)

    def int(self):
        r"""
        Converts input tensor dtype to `int32`. If the value in tensor is float or half, the decimal will be discarded.

        Returns:
            Tensor, converted to the `int32` dtype.

        Supported Platforms:
            ``Ascend`` ``GPU`` ``CPU``

        Examples:
            >>> import numpy as np
            >>> import mindspore
            >>> from mindspore import Tensor
            >>> input_x = Tensor(np.ones([2,2]), mindspore.float32)
            >>> output = input_x.int()
            >>> print(output.dtype)
            Int32
        """
        return self.to(mstype.int32)

    def byte(self):
        r"""
        Converts input tensor dtype to `uint8`.

        Returns:
            Tensor, converted to the `uint8` dtype.

        Supported Platforms:
            ``Ascend`` ``GPU`` ``CPU``

        Examples:
            >>> import numpy as np
            >>> import mindspore
            >>> from mindspore import Tensor
            >>> input_x = Tensor(np.ones([2,2]), mindspore.float32)
            >>> output = input_x.byte()
            >>> print(output.dtype)
            UInt8
        """
        return self.to(mstype.uint8)

    def long(self):
        r"""
        Converts input tensor dtype to `int64`. If the value in tensor is float or half, the decimal will be discarded.

        Returns:
            Tensor, converted to the `int64` dtype.

        Supported Platforms:
            ``Ascend`` ``GPU`` ``CPU``

        Examples:
            >>> import numpy as np
            >>> import mindspore
            >>> from mindspore import Tensor
            >>> input_x = Tensor(np.ones([2,2]), mindspore.int32)
            >>> output = input_x.long()
            >>> print(output.dtype)
            Int64
        """
        return self.to(mstype.int64)

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
        return self.to(mstype.int16)

    def cholesky(self, upper=False):
        r"""
        For details, please refer to :func:`mindspore.ops.cholesky`.
        """
        return tensor_operator_registry.get('cholesky')(self, upper=upper)

    def cholesky_inverse(self, upper=False):
        r"""
        This interface is deprecated from version 2.4 and will be removed in a future version.
        """
        return tensor_operator_registry.get('cholesky_inverse')(self, upper=upper)

    def cholesky_solve(self, input2, upper=False):
        r"""
        For details, please refer to :func:`mindspore.ops.cholesky_solve`.

        .. warning::
            This is an experimental API that is subject to change or deletion.
        """
        return tensor_operator_registry.get('cholesky_solve')(self, input2, upper)

    def conj(self):
        r"""
        For details, please refer to :func:`mindspore.ops.conj`.
        """
        return tensor_operator_registry.get('conj')(self)

    def cross(self, other, dim=None):
        r"""
        For details, please refer to :func:`mindspore.ops.cross`.
        """
        return tensor_operator_registry.get('cross')(self, other, dim)

    def erfinv(self):
        r"""
        For details, please refer to :func:`mindspore.ops.erfinv`.
        """
        return tensor_operator_registry.get('erfinv')(self)

    def erfinv_(self):
        r"""
        In-place version of erfinv(), for details, please refer to :func:`mindspore.ops.erfinv`.

        .. warning::
            This is an experimental API that is subject to change or deletion.
        """
        return tensor_operator_registry.get('erfinv_')(self)

    def lcm(self, other):
        r"""
        For details, please refer to :func:`mindspore.ops.lcm`.
        """
        return tensor_operator_registry.get('lcm')(self, other)

    def ldexp(self, other):
        r"""
        For details, please refer to :func:`mindspore.ops.ldexp`.
        """
        return tensor_operator_registry.get('ldexp')(self, other)

    def fold(self, output_size, kernel_size, dilation=1, padding=0, stride=1):
        r"""
        For details, please refer to :func:`mindspore.ops.fold`.
        """
        return tensor_operator_registry.get('fold')(self, output_size, kernel_size, dilation, padding, stride)

    def unfold(self, kernel_size, dilation=1, padding=0, stride=1):
        r"""
        For details, please refer to :func:`mindspore.ops.unfold`.

        .. warning::
            This is an experimental API that is subject to change or deletion.

        """
        return tensor_operator_registry.get('unfold')(self, kernel_size, dilation, padding, stride)

    def expand(self, *size):
        r"""
        For details, please refer to :func:`mindspore.ops.broadcast_to`.
        The parameter `size` of the current interface is the same as the parameter `shape` of the reference interface.
        """
        return self.broadcast_to(*size)

    def cumprod(self, dim, dtype=None):
        r"""
        For details, please refer to :func:`mindspore.ops.cumprod`.
        """
        return tensor_operator_registry.get('cumprod')(self, dim, dtype)

    def multiply(self, value):
        r"""
        For details, please refer to :func:`mindspore.ops.mul`.
        The parameter `value` of the current interface is the same as the parameter `other` of the reference interface.
        """
        return tensor_operator_registry.get('multiply')(self, value)

    def equal(self, other):
        r"""
        For details, please refer to :func:`mindspore.ops.equal`.
        """
        return tensor_operator_registry.get('equal')(self, other)

    def index_add_(self, dim, index, source, *, alpha=1):
        r"""
        Accumulate the elements of `alpha` times `source` into the `self` by adding to the index
        in the order given in `index`. For example, if `dim == 0`, `index[i] == j`, and `alpha = -1`,
        then the `i` th row of `source` is subtracted from the `j` th row of `self` .
        The `dim` th dimension of `source` must have the same size as the length of `index` ,
        and all other dimensions must match `self`, or an error will be raised.
        For a 3-D tensor the output is defined as follows:

        .. math::
            \begin{array}{ll}
            self[index[i],\ :,\ :]\ +=\ alpha * src[i,\ :,\ :]  \qquad \#if\ dim == 0 \\
            self[:,\ \ index[i],\ :]\ +=\ alpha * src[:,\ \ i,\ :]  \qquad \#if\ dim == 1 \\
            self[:,\ :,\ \ index[i]]\ +=\ alpha * src[:,\ :,\ \ i]  \qquad\#if\ dim == 2 \\
            \end{array}

        .. warning::
            This is an experimental API that is subject to change or deletion.

        Args:
            dim (int): The dimension along which to index.
            index (Tensor): Add the value of `self` and `source` along the dimension of the `dim` according to
                the specified index value, with data type int32. The `index` must be 1D with the same size as
                the size of `source` in the `dim` dimension. The values of `index` should be in [0, b),
                where the b is the size of `self` in the `dim` dimension.
            source (Tensor): The input tensor with the value to add. Must have same data type as `self`.
                The shape must be the same as `self` except the `dim` th dimension.

        Keyword Args:
            alpha (number, optional): The scalar multiplier for source. Default: ``1``.

        Returns:
            Tensor, has the same shape and dtype as `self`.

        Raises:
            TypeError: If neither `index` nor `source` is a Tensor.
            ValueError: If dim is out of `self` rank's range.
            ValueError: If `self` rank is not the same as `source` rank.
            ValueError: If shape of `index` is not 1D or size of `index` is not equal to dimension
                of `source[dim]`.
            ValueError: If `source`'s shape is not the same as `self` except the `dim` th dimension.

        Supported Platforms:
            ``Ascend``

        Examples:
            >>> import numpy as np
            >>> import mindspore
            >>> from mindspore import Tensor
            >>> x = Tensor(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), mindspore.float32)
            >>> index = Tensor(np.array([0, 2]), mindspore.int32)
            >>> y = Tensor(np.array([[0.5, 1.0], [1.0, 1.5], [2.0, 2.5]]), mindspore.float32)
            >>> output = x.index_add_(1, index, y, alpha=1)
            >>> print(output)
            [[ 1.5  2.   4. ]
             [ 5.   5.   7.5]
             [ 9.   8.  11.5]]
            >>> print(x)
            [[ 1.5  2.   4. ]
             [ 5.   5.   7.5]
             [ 9.   8.  11.5]]
        """
        return tensor_operator_registry.get('index_add_')(self, dim, index, source, alpha)

    def igamma(self, other):
        r"""
        For details, please refer to :func:`mindspore.ops.igamma`.
        """
        return tensor_operator_registry.get('igamma')(self, other)

    def igammac(self, other):
        r"""
        For details, please refer to :func:`mindspore.ops.igammac`.
        """
        return tensor_operator_registry.get('igammac')(self, other)

    def isnan(self):
        r"""
        For details, please refer to :func:`mindspore.ops.ne`.
        """
        return self.ne(self)

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

    def lstsq(self, A):
        r"""
        This interface is deprecated from version 2.4 and will be removed in a future version.
        """
        return tensor_operator_registry.get('lstsq')(self, A)

    @property
    def mH(self):
        r"""
        Accessing this property is equivalent to Calling self.adjoint().
        For details, please refer to :func:`mindspore.ops.adjoint`.

        Examples:
            >>> from mindspore import Tensor
            >>> import numpy as np
            >>> x = Tensor(np.array([[0. + 0.j, 1. + 1.j], [2. + 2.j, 3. + 3.j]]))
            >>> output = x.mH
            >>> print(output)
            [[0.-0.j 2.-2.j]
             [1.-1.j 3.-3.j]]
        """
        return self.adjoint()

    @property
    def mT(self):
        r"""
        Returns the Tensor that exchanges the last two dimensions.
        Accessing the attribute, x.mT, is equal to calling the method, x.swapaxes(-2, -1).
        For details, please refer to :func:`mindspore.Tensor.swapaxes`.

        Examples:
            >>> from mindspore import Tensor
            >>> import numpy as np
            >>> x = Tensor(np.ones((2, 3, 4)))
            >>> output = x.mT
            >>> print(output.shape)
            (2, 4, 3)
        """
        return self.swapaxes(-2, -1)

    def mvlgamma(self, p):
        r"""
        For details, please refer to :func:`mindspore.ops.mvlgamma`.
        """
        return tensor_operator_registry.get('mvlgamma')(self, p)

    def inner(self, other):
        r"""
        For details, please refer to :func:`mindspore.ops.inner`.
        """
        return tensor_operator_registry.get('inner')(self, other)

    def multinomial(self, num_samples, replacement=True, seed=None):
        r"""
        For details, please refer to :func:`mindspore.ops.multinomial`.
        """
        return tensor_operator_registry.get('multinomial')(self, num_samples, replacement, seed)

    def matrix_power(self, n):
        r"""
        This interface is deprecated from version 2.4 and will be removed in a future version.
        """
        return tensor_operator_registry.get('matrix_power')(self, n)

    def msort(self):
        r"""
        For details, please refer to :func:`mindspore.ops.msort`.
        """
        return tensor_operator_registry.get('msort')(self)

    def zero_(self):
        r"""
        Return a tensor filled with zeros.

        Returns:
            Return a tensor. Fill self tensor with zeros.

        Supported Platforms:
            ``Ascend``

        Examples:
            >>> import numpy as np
            >>> import mindspore
            >>> from mindspore import Tensor
            >>> x = Tensor(np.array([2, 2]))
            >>> output = x.zero_()
            >>> print(output)
            [0 0]
        """
        return tensor_operator_registry.get('zero_')(self)

    def sign(self):
        r"""
        For details, please refer to :func:`mindspore.ops.sign`.
        """
        return tensor_operator_registry.get('sign')(self)

    def sign_(self):
        """
        In-place version of :func:`mindspore.mint.sign`.
        """
        return tensor_operator_registry.get('sign_')(self)

    def signbit(self):
        """
        For details, please refer to :func:`mindspore.ops.signbit`.
        """
        return tensor_operator_registry.get('signbit')(self)

    def sgn(self):
        """
        For details, please refer to :func:`mindspore.ops.sgn`.
        """
        return tensor_operator_registry.get('sgn')(self)

    def quantile(self, q, axis=None, keepdims=False):
        r"""
        This interface is deprecated from version 2.4 and will be removed in a future version.
        """
        return tensor_operator_registry.get('quantile')(self, q, axis, keepdims)

    def nanquantile(self, q, axis=None, keepdims=False):
        """
        This interface is deprecated from version 2.4 and will be removed in a future version.
        """
        return tensor_operator_registry.get('nanquantile')(self, q, axis, keepdims)

    def orgqr(self, input2):
        r"""
        For details, please refer to :func:`mindspore.ops.orgqr`.
        """
        return tensor_operator_registry.get('orgqr')(self, input2)

    def lu_solve(self, LU_data, LU_pivots):
        r"""
        For details, please refer to :func:`mindspore.ops.lu_solve`.

        .. warning::
            This is an experimental API that is subject to change or deletion.
        """
        return tensor_operator_registry.get('lu_solve')(self, LU_data, LU_pivots)

    def nextafter(self, other):
        r"""
        For details, please refer to :func:`mindspore.ops.nextafter`.
        """
        return tensor_operator_registry.get('nextafter')(self, other)

    def qr(self, some=True):
        r"""
        This interface is deprecated from version 2.4 and will be removed in a future version.
        """
        validator.check_value_type('some', some, bool, 'Tensor.qr')
        return tensor_operator_registry.get('qr')(self, 'reduced' if some else 'complete')

    def ormqr(self, input2, input3, left=True, transpose=False):
        r"""
        For details, please refer to :func:`mindspore.ops.ormqr`,
        Args `input2` and `input3` correspond to the args `tau` and `other` of :func:`mindspore.ops.ormqr`.
        """
        return tensor_operator_registry.get('ormqr')(self, input2, input3, left, transpose)

    def index_put(self, indices, values, accumulate=False):
        r"""
        Based on the indices in `indices`, replace the corresponding elements in Tensor `self`
        with the values in `values`. Outplace version of  :func:`mindspore.Tensor.index_put_` 。

        .. warning::
            The behavior is unpredictable in the following scenario:

            - If `accumulate` is `False` and `indices` contains duplicate elements.

        Args:
            indices (tuple[Tensor], list[Tensor]): the indices of type int32 or int64, used to index into the `self`.
                The rank of tensors in indices should be 1-D, size of indices should <= `self.rank`
                and the tensors in indices should be broadcastable.
            values (Tensor): 1-D Tensor with the same type as `self`. `values` should be broadcastable with size 1.
            accumulate (bool, optional): If `accumulate` is ``True``, the elements in `values` will be added to `self`,
                otherwise the elements in `values` will replace the corresponding elements in the `self`.
                Default: ``False``.

        Returns:
            Tensor, with the same type and shape as the "self Tensor".

        Raises:
            TypeError: If the dtype of the `self` is not equal to the dtype of `values`.
            TypeError: If the dtype of `indices` is not tuple[Tensor], list[Tensor].
            TypeError: If the dtype of tensors in `indices` are not int32 or int64.
            TypeError: If the dtype of tensors in `indices` are inconsistent.
            TypeError: If the dtype of `accumulate` is not bool.
            ValueError: If rank(`values`) is not 1-D.
            ValueError: If size(`values`) is not 1 or max size of the tensors in `indices` when
                rank(`self`) == size(`indices`).
            ValueError: If size(`values`) is not 1 or `self`.shape[-1] when rank(`self`) > size(`indices`).
            ValueError: If the rank of tensors in `indices` is not 1-D.
            ValueError: If the tensors in `indices` is not be broadcastable.
            ValueError: If size(`indices`) > rank(`self`).

        Supported Platforms:
            ``Ascend`` ``CPU``

        Examples:
            >>> import numpy as np
            >>> import mindspore
            >>> from mindspore import Tensor
            >>> x = Tensor(np.array([[1, 2, 3], [4, 5, 6]]).astype(np.int32))
            >>> values = Tensor(np.array([3]).astype(np.int32))
            >>> indices = [Tensor(np.array([0, 1, 1]).astype(np.int32)), Tensor(np.array([1, 2, 1]).astype(np.int32))]
            >>> accumulate = True
            >>> output = x.index_put(indices, values, accumulate)
            >>> print(output)
            [[1 5 3]
            [4 8 9]]
        """
        validator.check_value_type('accumulate', accumulate, bool, 'Tensor.index_put')
        _index_put = tensor_operator_registry.get('index_put')(0 if accumulate is False else 1)
        return _index_put(self, values, indices)

    def index_put_(self, indices, values, accumulate=False):
        r"""
        Based on the indices in `indices`, replace the corresponding elements in Tensor `self` with the values
        in `values`. The expression `Tensor.index_put_(indices, values)` is equivalent to `tensor[indices] = values`.
        Update and return `self`.

        .. warning::
            The behavior is unpredictable in the following scenario:

            - If `accumulate` is `False` and `indices` contains duplicate elements.

        Args:
            indices (tuple[Tensor], list[Tensor]): the indices of type is bool, uint8, int32 or int64,
                used to index into the `self`. The size of indices should <=  the rank of `self`
                and the tensors in indices should be broadcastable.
            values (Tensor): Tensor with the same type as `self`. If size == 1, it will be broadcastable.
            accumulate (bool, optional): If `accumulate` is `True`, the elements in `values` will be added to `self`,
                otherwise the elements in `values` will replace the corresponding elements in the `self`.
                Default: ``False``.

        Returns:
            Tensor `self`.

        Raises:
            TypeError: If the dtype of the `self` is not equal to the dtype of `values`.
            TypeError: If the dtype of `indices` is not tuple[Tensor], list[Tensor].
            TypeError: If the dtype of tensors in `indices` are not bool, uint8, int32 or int64.
            TypeError: If the dtypes of tensors in `indices` are inconsistent.
            TypeError: If the dtype of `accumulate` is not bool.
            ValueError: If size(`values`) is not 1 or max size of the tensors in `indices` when
                rank(`self`) == size(`indices`).
            ValueError: If size(`values`) is not 1 or `self`.shape[-1] when rank(`self`) > size(`indices`).
            ValueError: If the tensors in `indices` is not be broadcastable.
            ValueError: If size(`indices`) > rank(`self`).

        Supported Platforms:
            ``Ascend``

        Examples:
            >>> import numpy as np
            >>> import mindspore
            >>> from mindspore import Tensor
            >>> x = Tensor(np.array([[1, 2, 3], [4, 5, 6]]).astype(np.int32))
            >>> values = Tensor(np.array([3]).astype(np.int32))
            >>> indices = [Tensor(np.array([0, 1, 1]).astype(np.int32)), Tensor(np.array([1, 2, 1]).astype(np.int32))]
            >>> accumulate = True
            >>> x.index_put_(indices, values, accumulate)
            >>> print(x)
            [[1 5 3]
             [4 8 9]]
        """
        index_put_ = tensor_operator_registry.get('index_put_')
        return index_put_(self, indices, values, accumulate)

    def move_to(self, to, blocking=True):
        r"""
        Copy Tensor to target device synchronously or asynchronously, default synchronously. only support PyNative mode.

        Args:
            to (str): a string type value, one of ``"Ascend"``, ``"GPU"``, ``"CPU"``.
            blocking (bool, optional): a bool type value, using synchronous copy or asynchronous copy.
                Default: ``True`` , synchronous copy.

        Returns:
            New Tensor, storged on target device which with the same type and shape as the "self Tensor".

        Raises:
            ValueError: If the type of `blocking` is not bool type.
            ValueError: If the value of `to` is not one of ``"Ascend"``, ``"GPU"``, ``"CPU"``.
            ValueError: If the run mode is not PyNative mode.

        Supported Platforms:
            ``Ascend`` ``GPU`` ``CPU``

        Examples:
            >>> import mindspore as ms
            >>> from mindspore import Tensor
            >>> x = ms.Tensor([1, 2, 3], ms.int64)
            >>> new_tensor = x.move_to("CPU")
        """
        if not isinstance(blocking, bool):
            raise ValueError(f"The type of 'blocking' must be bool, but got {blocking}")
        if to not in ("Ascend", "GPU", "CPU"):
            raise ValueError(f"The value of 'to' must be one of ['Ascend', 'GPU', 'CPU'], but got {to}")
        return TensorPy_.move_to(self, to, blocking)

    def _offload(self):
        r"""
        Offload tensor parameter to host. Currently, only support for pynative mode.

        Supported Platforms:
            ``Ascend``

        Examples:
            >>> import mindspore as ms
            >>> from mindspore import Tensor
            >>> x = ms.Tensor([1, 2, 3], ms.int64)
            >>> x._offload()
        """
        return TensorPy_._offload(self, False)

    def _data_ptr(self):
        r"""
        Get the data ptr address of tensor, for CPU is host address, GPU/NPU is device address.
        User should know how to use the data ptr address.
        Note: this api is an experimental api, users need understatnd it before use.

        Supported Platforms:
            ``CPU/GPU/Ascend``

        Examples:
            >>> import mindspore as ms
            >>> from mindspore import Tensor
            >>> x = ms.Tensor([1, 2, 3], ms.int64)
            >>> data_ptr = x._data_ptr()
        """
        return TensorPy_._data_ptr(self)

    def data_ptr(self):
        r"""
        Get the data ptr address of tensor, for CPU is host address, GPU/NPU is device address.
        User should know how to use the data ptr address.
        Note: this api is an experimental api, users need understatnd it before use.

        Supported Platforms:
            ``CPU/GPU/Ascend``

        Examples:
            >>> import mindspore as ms
            >>> from mindspore import Tensor
            >>> x = ms.Tensor([1, 2, 3], ms.int64)
            >>> data_ptr = x.data_ptr()
        """
        return TensorPy_._data_ptr(self)

    def normal_(self, mean=0, std=1, *, generator=None):
        r"""
        Update the `self` tensor in place by generating random numbers sampled from the normal
        distribution which constructed by the parameters `mean` and `std`.

        .. warning::
            This is an experimental API that is subject to change or deletion.

        Args:
            mean (number, optional): the mean of normal distribution. With float data type.
                Default: ``0``.
            std (number, optional): the std of normal distribution. With float data type.
                Default: ``1``.

        Keyword Args:
            generator (:class:`mindspore.Generator`, optional): a pseudorandom number generator.
                Default: ``None``, uses the default pseudorandom number generator.

        Returns:
            A tensor that is filled with random numbers that follow a normal distribution and
            that has the same type and shape as the `self` tensor.

        Raises:
            TypeError: If the dtype of `mean` or `std` is not one of: bool, int, float, complex.

        Supported Platforms:
            ``Ascend``

        Examples:
            >>> import mindspore
            >>> import numpy as np
            >>> x = mindspore.Tensor(np.array([[1, 2], [3, 4]]), dtype=mindspore.float32)
            >>> output = x.normal_()
            >>> print(output)
            [[0.2788825 1.3305743]
             [1.244194 1.16303174]]
        """
        return tensor_operator_registry.get('normal_')(self, mean=mean, std=std, generator=generator)

    def triangular_solve(self, A, upper=True, transpose=False, unitriangular=False):
        r"""
        For details, please refer to :func:`mindspore.mint.triangular_solve`.
        """
        return tensor_operator_registry.get('triangular_solve')(self, A, upper, transpose, unitriangular)


def _vm_compare(*args):
    """Implement `vm_compare` for tensor."""
    if args:
        obj_str = args[-1]
    else:
        raise ValueError("_vm_compare does not receive any input.")
    if obj_str == "shape":
        fn = getattr(args[0].asnumpy(), obj_str)
        return fn
    if obj_str == "_tensor_setitem" or obj_str == "_tensor_setitem_origin":
        fn = getattr(args[0].asnumpy(), "__setitem__")
        index = args[1].asnumpy() if isinstance(args[1], Tensor) else args[1]
        value = args[2].asnumpy() if isinstance(args[2], Tensor) else args[2]
        fn(index, value)
        return args[0]
    if obj_str == "_tensor_getitem" or obj_str == "_tensor_getitem_origin":
        fn = getattr(args[0].asnumpy(), "__getitem__")
        index = args[1].asnumpy() if isinstance(args[1], Tensor) else args[1]
        return Tensor(np.array(fn(index)))
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
        if isinstance(input_data, (tuple, list)):
            try:
                _ = np.array(input_data)
            except ValueError as e:
                if "The requested array has an inhomogeneous shape" in str(e):
                    raise TypeError(
                        f"For Tensor, the input_data is {input_data} that contain unsupported element.") from e
                raise


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
        dtype = mstype._pytype_to_dtype(np.dtype(dtype.lower()))  # pylint:disable=protected-access
    elif isinstance(dtype, type):
        dtype = mstype._pytype_to_dtype(dtype)  # pylint:disable=protected-access
    elif dtype not in mstype.number_type + (mstype.bool_,):
        raise TypeError(
            f"For Tensor.astype, the input type must be one of {list(mstype.number_type + (mstype.bool_,) + np_types)},"
            f" but got '{dtype}'.")
    return dtype


setattr(tensor_operator_registry, 'vm_compare', _vm_compare)
