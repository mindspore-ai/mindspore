# Copyright 2022 Huawei Technologies Co., Ltd
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
"""SparseTensor implementation."""
from __future__ import absolute_import, annotations

__all__ = ['RowTensorInner', 'RowTensor', 'SparseTensor', 'COOTensor', 'CSRTensor']

from typing import Tuple

from mindspore import log as logger
from mindspore.common import dtype as mstype
from mindspore.common._register_for_tensor import tensor_operator_registry
from mindspore.common.tensor import Tensor
from mindspore._c_expression import COOTensor as COOTensor_
from mindspore._c_expression import CSRTensor as CSRTensor_
from mindspore._c_expression import RowTensor as RowTensor_
from mindspore._c_expression import Tensor as Tensor_
from mindspore._checkparam import Validator as validator
from mindspore._checkparam import is_stub_tensor


class RowTensorInner(RowTensor_):
    """
    Implementation for RowTensor, for MindSpore developers only.
    """

    def __init__(self, indices=None, values=None, shape=None, row_tensor=None):
        """Init RowTensor"""
        self.init_finished = False
        # Directly init a RowTensor from another RowTensor
        if row_tensor is not None:
            if not isinstance(row_tensor, (RowTensor, RowTensor_)):
                raise TypeError(f"Expect input `row_tensor` to be a RowTensor, but got {type(row_tensor)}")
            if not (indices is None and values is None and shape is None):
                raise TypeError("If input `row_tensor` is provided, `indices`, `values`, `shapes` should all be `None`")
            RowTensor_.__init__(self, row_tensor)
        # Init a RowTensor from indices, values and shape
        else:
            if is_stub_tensor(values):
                values = values.stub_sync()
            RowTensor_.__init__(self, indices, values, shape)
        self.init_finished = True

    def __repr__(self):
        """Avoid PyTest Segfault when RowTensor is not initialized."""
        if self.init_finished:
            return RowTensor_.__repr__(self)
        return ''

    @property
    def indices(self):
        """Return RowTensor's indices."""
        return Tensor(self._indices)

    @property
    def values(self):
        """Return RowTensor's non-zero values."""
        return Tensor(self._values)

    @property
    def dense_shape(self):
        """Return RowTensor's shape."""
        return self._shape


class RowTensor(RowTensorInner):
    """
    A sparse representation of a set of tensor slices at given indices.

    An RowTensor is typically used to represent a subset of a larger
    tensor dense of shape [L0, D1, .. , DN] where L0 >> D0.

    The values in indices are the indices in the first dimension of the slices
    that have been extracted from the larger tensor.

    The dense tensor dense represented by an RowTensor slices has
    `dense[slices.indices[i], :, :, :, ...] = slices.values[i, :, :, :, ...]`.

    For example, if indices is [0], values is [[1, 2]], shape is
    (3, 2), then the dense representation of the row tensor will be:

    .. code-block::

        [[1, 2],
         [0, 0],
         [0, 0]]

    Note:
        RowTensor is deprecated from version 2.0, and will be removed in future version.

    Args:
        indices (Tensor): A 1-D integer Tensor of shape [D0]. Default: None.
        values (Tensor): A Tensor of any dtype of shape [D0, D1, ..., Dn]. Default: None.
        shape (tuple(int)): An integer tuple which contains the shape
            of the corresponding dense tensor. Default: None.
        row_tensor (RowTensor): A RowTensor object. Default: None.

    Returns:
        RowTensor, composed of `indices`, `values`, and `shape`.

    Examples:
        >>> import mindspore as ms
        >>> from mindspore import Tensor, RowTensor
        >>> indices = Tensor([0])
        >>> values = Tensor([[1, 2]], dtype=ms.float32)
        >>> shape = (3, 2)
        >>> x = RowTensor(indices, values, shape)
        >>> print(x.values)
        [[1. 2.]]
        >>> print(x.indices)
        [0]
        >>> print(x.dense_shape)
        (3, 2)
    """

    def __init__(self, indices=None, values=None, shape=None, row_tensor=None):
        """Init RowTensor"""
        logger.warning("'RowTensor' is deprecated from version 1.7 and will be removed in a future version.")
        super().__init__(indices, values, shape, row_tensor)


class SparseTensor(COOTensor_):
    """
    A sparse representation of a set of nonzero elements from a tensor at given indices.

    SparseTensor can only be used in the `Cell`'s construct method.

    For a tensor dense, its SparseTensor(indices, values, dense_shape) has
    `dense[indices[i]] = values[i]`.

    For example, if indices is [[0, 1], [1, 2]], values is [1, 2], dense_shape is
    (3, 4), then the dense representation of the sparse tensor will be:

    .. code-block::

        [[0, 1, 0, 0],
         [0, 0, 2, 0],
         [0, 0, 0, 0]]

    Note:
        The interface is deprecated from version 1.7 and will be removed in a future version.
        Please use 'COOTensor' instead.

    Args:
        indices (Tensor): A 2-D integer Tensor of shape :math:`(N, ndims)`,
            where N and ndims are the number of `values` and number of dimensions in
            the SparseTensor, respectively.
        values (Tensor): A 1-D tensor of any type and shape :math:`(N)`, which
            supplies the values for each element in `indices`.
        shape (tuple(int)): An integer tuple of size `ndims`,
            which specifies the shape of the sparse tensor.

    Returns:
        SparseTensor, composed of `indices`, `values`, and `shape`.

    Examples:
        >>> import mindspore as ms
        >>> import mindspore.nn as nn
        >>> from mindspore import Tensor, SparseTensor
        >>> indices = Tensor([[0, 1], [1, 2]])
        >>> values = Tensor([1, 2], dtype=ms.float32)
        >>> shape = (3, 4)
        >>> x = SparseTensor(indices, values, shape)
        >>> print(x.values)
        [1. 2.]
        >>> print(x.indices)
        [[0 1]
         [1 2]]
        >>> print(x.shape)
        (3, 4)
    """

    def __init__(self, indices, values, shape):
        """Init COOTensor."""
        logger.warning("'SparseTensor' is deprecated from version 1.7 and will be removed in a future version. " +
                       "Please use 'COOTensor' instead.")
        if not (isinstance(indices, Tensor) and isinstance(values, Tensor) and isinstance(shape, tuple)):
            raise TypeError("Inputs must follow: COOTensor(indices, values, shape).")
        if is_stub_tensor(indices):
            indices = indices.stub_sync()
        if is_stub_tensor(values):
            values = values.stub_sync()
        COOTensor_.__init__(self, indices, values, shape)

    @property
    def indices(self):
        """Return SparseTensor's indices."""
        return Tensor(self._indices)

    @property
    def values(self):
        """Return SparseTensor's non-zero values."""
        return Tensor(self._values)

    @property
    def shape(self):
        """Return SparseTensor's shape."""
        return self._shape


class COOTensor(COOTensor_):
    """
    A sparse representation of a set of nonzero elements from a tensor at given indices.

    For a tensor dense, its COOTensor(indices, values, shape) has
    `dense[indices[i]] = values[i]`.

    For example, if indices is [[0, 1], [1, 2]], values is [1, 2], shape is
    (3, 4), then the dense representation of the sparse tensor will be:

    .. code-block::

        [[0, 1, 0, 0],
         [0, 0, 2, 0],
         [0, 0, 0, 0]]

    Common arithmetic operations include: addition (+), subtraction (-), multiplication (*),
    and division (/). For details about operations supported by `COOTensor`, see
    `operators <https://www.mindspore.cn/docs/en/master/note/static_graph_syntax_support.html#operators>`_.

    Note:
        This is an experimental feature and is subjected to change.
        Currently, duplicate coordinates in the indices will not be coalesced.
        If the indices contain out-of-bound values, the result will be undefined.

    Args:
        indices (Tensor): A 2-D integer Tensor of shape `[N, ndims]`,
            where N and ndims are the number of `values` and number of dimensions in
            the COOTensor, respectively. Currently, `ndims` must be 2.
            Please make sure that the indices are in range of the given shape.
        values (Tensor): A 1-D tensor of any type and shape `[N]`, which
            supplies the values for each element in `indices`.
        shape (tuple(int)): An integer tuple of size `ndims`,
            which specifies the dense_shape of the sparse tensor.
        coo_tensor (COOTensor): A COOTensor object.

    Returns:
        COOTensor, composed of `indices`, `values`, and `shape`.

    Examples:
        >>> import mindspore as ms
        >>> import mindspore.nn as nn
        >>> from mindspore import Tensor, COOTensor
        >>> indices = Tensor([[0, 1], [1, 2]], dtype=ms.int32)
        >>> values = Tensor([1, 2], dtype=ms.float32)
        >>> shape = (3, 4)
        >>> x = COOTensor(indices, values, shape)
        >>> print(x.values)
        [1. 2.]
        >>> print(x.indices)
        [[0 1]
         [1 2]]
        >>> print(x.shape)
        (3, 4)
    """

    def __init__(self, indices=None, values=None, shape=None, coo_tensor=None):
        """Init COOTensor"""
        self.init_finished = False
        # Directly init a COOTensor from another COOTensor
        if coo_tensor is not None:
            if not isinstance(coo_tensor, (COOTensor, COOTensor_)):
                raise TypeError(f"Expect input `coo_tensor` to be a COOTensor, but got {type(coo_tensor)}")
            if not (indices is None and values is None and shape is None):
                raise TypeError("If input `coo_tensor` is provided, `indices`, `values`, `shapes` should all be `None`")
            COOTensor_.__init__(self, coo_tensor)
        # Init a COOTensor from indices, values and shape
        else:
            validator.check_coo_tensor_input(indices, values, shape)
            validator.check_coo_tensor_shape(indices.shape, values.shape, shape)
            validator.check_coo_tensor_dtype(indices.dtype)
            indices = tensor_operator_registry.get('stop_gradient')(indices)
            if is_stub_tensor(indices):
                indices = indices.stub_sync()
            if is_stub_tensor(values):
                values = values.stub_sync()
            COOTensor_.__init__(self, indices, values, shape)
        self.init_finished = True

    def __repr__(self):
        """Avoid PyTest Segfault when COOTensor is not initialized."""
        if self.init_finished:
            return COOTensor_.__repr__(self)
        return ''

    def __neg__(self):
        return COOTensor(self.indices, -self.values, self.shape)

    def __add__(self, other):
        if not self.shape == other.shape:
            raise ValueError("Input tensors should have the same shape.")
        if isinstance(other, Tensor):
            return tensor_operator_registry.get("tensor_scatter_add")(other, self.indices, self.values)
        if isinstance(other, COOTensor):
            return tensor_operator_registry.get('coo_add')(self, other, Tensor(0, self.values.dtype))
        raise TypeError("COOTensor add with %s is not supported." % type(other))

    def __sub__(self, other):
        if not self.shape == other.shape:
            raise ValueError("Input tensors should have the same shape.")
        if isinstance(other, Tensor):
            return tensor_operator_registry.get("tensor_scatter_add")(-other, self.indices, self.values)
        if isinstance(other, COOTensor):
            return tensor_operator_registry.get('coo_add')(
                self, -other, Tensor(0, self.values.dtype))
        raise TypeError("COOTensor subtract with %s is not supported." % type(other))

    def __mul__(self, other):
        if not self.shape == other.shape:
            raise ValueError("Input tensors should have the same shape.")
        if isinstance(other, Tensor):
            other_values = tensor_operator_registry.get("gather_nd")(other, self.indices)
            return COOTensor(self.indices, self.values * other_values, self.shape)
        raise TypeError("COOTensor multiply with %s is not supported." % type(other))

    def __div__(self, other):
        if not self.shape == other.shape:
            raise ValueError("Input tensors should have the same shape.")
        if isinstance(other, Tensor):
            logger.warning("For sparse divide, zero values in the dense tensor are ignored.")
            other_values = tensor_operator_registry.get("gather_nd")(other, self.indices)
            return COOTensor(self.indices, self.values / other_values, self.shape)
        raise TypeError("COOTensor divide with %s is not supported." % type(other))

    def __truediv__(self, other):
        return self.__div__(other)

    @property
    def indices(self) -> Tensor:
        """Return COOTensor's indices."""
        return Tensor(self._indices)

    @property
    def values(self) -> Tensor:
        """Return COOTensor's non-zero values."""
        return Tensor(self._values)

    @property
    def shape(self) -> Tuple[int, ...]:
        """Return COOTensor's shape."""
        return self._shape

    @property
    def dtype(self) -> mstype:
        """Return the dtype of the values of COOTensor (:class:`mindspore.dtype`)."""
        return self._dtype

    @property
    def size(self) -> int:
        """Return the number of non-zero values."""
        return self.values.size

    @property
    def itemsize(self) -> int:
        """Return the length of one tensor element in bytes."""
        return self.values.itemsize

    @property
    def ndim(self) -> int:
        """Return the number of tensor dimensions."""
        return len(self.shape)

    def coalesce(self) -> COOTensor:
        """
        Returns a coalesced copy of an uncoalesced sparse tensor.

        Returns:
            A COOTensor.

        Supported Platforms:
            ``GPU``

        Examples:
            >>> import mindspore as ms
            >>> import mindspore.ops as ops
            >>> from mindspore import Tensor, COOTensor
            >>> x_indices = Tensor([[0, 0, 1], [1, 1, 2]], dtype=ms.int64)
            >>> x_values = Tensor([1, 5, 4], dtype=ms.float32)
            >>> x_shape = (3, 3)
            >>> coo_tensor = COOTensor(x_indices.transpose(), x_values, x_shape)
            >>> res = coo_tensor.coalesce()
            >>> print(res)
            COOTensor(shape=[3, 3], dtype=Float32, indices=Tensor(shape=[2, 2], dtype=Int64,
                value=[[0 1] [1 2]]), values=Tensor(shape=[2], dtype=Float32, value=[6.00000000e+00 4.00000000e+00]))
        """
        shape = Tensor(self.shape)
        res_indices, res_values, _ = tensor_operator_registry.get("coalesce")(self.indices.transpose(),
                                                                              self.values, shape)
        return COOTensor(res_indices.transpose(), res_values, self.shape)

    def to_csr(self) -> CSRTensor:
        """
        Converts COOTensor to CSRTensor.

        Note:
            Currently only supports CPU backend with LLVM 12.0.1 installed.

        Returns:
            CSRTensor.

        Supported Platforms:
            ``GPU`` ``CPU``
        """
        row_indices = self.indices[:, 0]
        col_indices = self.indices[:, 1]
        idx_dtype = self.indices.dtype
        row_indices, sort_idx = tensor_operator_registry.get("sort")(
            row_indices.astype(mstype.float32))
        row_indices = row_indices.astype(idx_dtype)
        col_indices = col_indices[sort_idx]
        values = self.values[sort_idx]
        indptr = tensor_operator_registry.get("coo2csr")(row_indices, self.shape[0])
        return CSRTensor(indptr, col_indices, values, self.shape)

    def to_dense(self) -> Tensor:
        """
        Converts COOTensor to Dense Tensor.

        Returns:
            Tensor.

        Supported Platforms:
            ``GPU``
        """
        zeros_tensor = tensor_operator_registry.get("zeros")(self.shape, self.values.dtype)
        return tensor_operator_registry.get("tensor_scatter_add")(
            zeros_tensor, self.indices, self.values)

    def astype(self, dtype: mstype) -> COOTensor:
        """
        Return a copy of the COOTensor, cast its values to a specified type.

        Args:
            dtype (Union[:class:`mindspore.dtype`, numpy.dtype, str]): Designated tensor dtype.

        Returns:
            COOTensor.

        Supported Platforms:
            ``Ascend`` ``GPU`` ``CPU``

        Examples:
            >>> import mindspore as ms
            >>> from mindspore import Tensor, COOTensor
            >>> indices = Tensor([[0, 1], [1, 2]], dtype=ms.int32)
            >>> values = Tensor([1, 2], dtype=ms.float32)
            >>> shape = (3, 4)
            >>> coo_tensor = COOTensor(indices, values, shape)
            >>> print(coo_tensor.astype(ms.float64).dtype)
            Float64
        """
        data = self.values.astype(dtype)
        return COOTensor(self.indices, data, self.shape)

    def to_tuple(self) -> Tuple[Tensor, Tensor, Tuple[int, ...]]:
        """
        Return indices, values and shape as a tuple.

        Returns:
            Tuple.

        Supported Platforms:
            ``Ascend`` ``GPU`` ``CPU``
        """
        return self.indices, self.values, self.shape

    def abs(self) -> COOTensor:
        """
        Return absolute value element-wisely.

        Returns:
            COOTensor.

        Supported Platforms:
            ``Ascend`` ``GPU`` ``CPU``
        """
        data = self.values.abs()
        return COOTensor(self.indices, data, self.shape)

    def add(self, other: COOTensor, thresh: Tensor) -> COOTensor:
        """
        Return the sum with another COOTensor.

        Args:
            other(COOTensor): the second SparseTensor to sum.
            thresh(Tensor): A 0-D Tensor, represents the magnitude threshold that determines
                if an output value/index pair take space, Its dtype
                should match that of the values if they are real. If output's
                value is less than the `thresh`, it will vanish.

        Returns:
            COOTensor, representing the sum.

        Raises:
            ValueError: If any input(self/other)'s indices's dim is not equal to 2.
            ValueError: If any input(self/other)'s values's dim is not equal to 1.
            ValueError: If any input(self/other)'s shape's dim is not equal to 1.
            ValueError: If thresh's dim is not equal to 0.
            TypeError: If any input(self/other)'s indices's type is not equal to int64.
            TypeError: If any input(self/other)'s shape's type is not equal to int64.
            ValueError: If any input(self/other)'s indices's length is not equal to
                its values's length.
            TypeError: If any input(self/other)'s values's type is not equal to anf of
                (int8/int16/int32/int64/float32/float64/complex64/complex128)
            TypeError: If thresh's type is not equal to anf of
                (int8/int16/int32/int64/float32/float64)
            TypeError: If self's indices's type is not equal to other's indices's type
            TypeError: If self's values's type is not equal to other's values's type
            TypeError: If self's shape's type is not equal to other's shape's type
            TypeError: If (self/other)'s value's type is not matched with thresh's type

        Supported Platforms:
            ``GPU`` ``CPU``

        Examples:
            >>> from mindspore import Tensor, COOTensor
            >>> from mindspore import dtype as mstype
            >>> indics0 = Tensor([[0, 1], [1, 2]], dtype=mstype.int64)
            >>> values0 = Tensor([1, 2], dtype=mstype.int32)
            >>> shape0 = (3, 4)
            >>> input0 = COOTensor(indics0, values0, shape0)
            >>> indics1 = Tensor([[0, 0], [1, 1]], dtype=mstype.int64)
            >>> values1 = Tensor([3, 4], dtype=mstype.int32)
            >>> shape1 = (3, 4)
            >>> input1 = COOTensor(indics1, values1, shape1)
            >>> thres = Tensor(0, dtype=mstype.int32)
            >>> out = input0.add(input1, thres)
            >>> print(out)
            COOTensor(shape=[3, 4], dtype=Int32, indices=Tensor(shape=[4, 2], dtype=Int64, value=
            [[0 0]
            [0 1]
            [1 1]
            [1 2]]), values=Tensor(shape[4], dtype=Int32, value=[3 1 4 2]))
        """
        return tensor_operator_registry.get('coo_add')(self, other, thresh)


class CSRTensor(CSRTensor_):
    """
    Constructs a sparse tensor in CSR (Compressed Sparse Row) format, with specified
    values indicated by `values` and row and column positions indicated by `indptr`
    and `indices`.

    For example, if indptr is [0, 1, 2, 2], indices is [1, 2], values is [1., 2.], shape is
    (3, 4), then the dense representation of the sparse tensor will be:

    .. code-block::

        [[0., 1., 0., 0.],
         [0., 0., 2., 0.],
         [0., 0., 0., 0.]]

    Common arithmetic operations include: addition (+), subtraction (-), multiplication (*),
    and division (/). For details about operations supported by `CSRTensor`, see
    `operators <https://www.mindspore.cn/docs/en/master/note/static_graph_syntax_support.html#operators>`_.

    Note:
        This is an experimental feature and is subjected to change.
        If the values given by `indptr` or `indices` are invalid, the results may be undefined. Invalid values include
    when the length of `values` or `indices` exceeds the range indicated by indptr, and when the columns indicated by
    `indices` are repeated on the same row.

    Args:
        indptr (Tensor): 1-D Tensor of shape :math:`(M)`, which equals to `shape[0] + 1`, which indicates the
            start and end point for `values` in each row. Default: None. If provided,
            must be int16, int32 or int64.
        indices (Tensor): 1-D Tensor of shape :math:`(N)`, which has the same length as `values`. `indices`
            indicates the which column `values` should be placed. Default: None. If provided,
            must be int16, int32 or int64.
        values (Tensor): Tensor, which has the same length as `indices` (values.shape[0] == indices.shape[0]).
            `values`  stores the data for CSRTensor. Default: None.
        shape (tuple(int)): A tuple indicates the shape of the CSRTensor, and `shape[0]` must equal to `M - 1`,
            which all equal to number of rows of the CSRTensor. Default: None.
        csr_tensor (CSRTensor): A CSRTensor object.  Values' feature dimension should match with
            CSRTensor's feature dimension (values.shape[1:] == csr_tensor.shape[2:]). Default: None.

    Outputs:
        CSRTensor, with shape defined by `shape`, and dtype inferred from `value`.

    Examples:
        >>> import mindspore as ms
        >>> from mindspore import Tensor, CSRTensor
        >>> # initialize a csr_tensor with indptr, indices, values and shape
        >>> indptr = Tensor([0, 1, 2], dtype=ms.int32)
        >>> indices = Tensor([0, 1], dtype=ms.int32)
        >>> values = Tensor([1, 2], dtype=ms.float32)
        >>> shape = (2, 4)
        >>> csr_tensor = CSRTensor(indptr, indices, values, shape)
        >>> # access a data member of CSRTensor
        >>> print(indptr == csr_tensor.indptr)
        [ True  True  True]
    """

    def __init__(self, indptr=None, indices=None, values=None, shape=None, csr_tensor=None):
        "Init CSRTensor"
        self.init_finished = False
        # Directly init a CSRTensor from another CSRTensor
        if csr_tensor is not None:
            if not isinstance(csr_tensor, (CSRTensor, CSRTensor_)):
                raise TypeError(f"Expect input `csr_tensor` to be a CSRTensor, but got {type(csr_tensor)}")
            if not (indptr is None and indices is None and values is None and shape is None):
                raise TypeError(
                    "If input `csr_tensor` is provided, `indptr`, `indices`, `values`, `shapes` should all be `None`")
            CSRTensor_.__init__(self, csr_tensor)
        # Init a CSRTensor from indptr, indices, values and shape
        else:
            validator.check_csr_tensor_input(indptr, indices, values, shape)
            validator.check_csr_tensor_shape(indptr.shape, indices.shape, values.shape, shape)
            validator.check_csr_tensor_dtype(indptr.dtype, indices.dtype)
            indptr = tensor_operator_registry.get('stop_gradient')(indptr)
            indices = tensor_operator_registry.get('stop_gradient')(indices)
            if is_stub_tensor(indptr):
                indptr = indptr.stub_sync()
            if is_stub_tensor(values):
                values = values.stub_sync()
            if is_stub_tensor(indices):
                indices = indices.stub_sync()
            CSRTensor_.__init__(self, indptr, indices, values, shape)
        self.init_finished = True

    def __repr__(self):
        """Avoid PyTest Segfault when CSRTensor is not initialized."""
        if self.init_finished:
            return CSRTensor_.__repr__(self)
        return ''

    def __mul__(self, other):
        return tensor_operator_registry.get('csr_mul')(self, other)

    def __div__(self, other):
        logger.warning("For CSR divide, zero values in the dense tensor are ignored.")
        return tensor_operator_registry.get('csr_div')(self, other)

    def __truediv__(self, other):
        return self.__div__(other)

    def __neg__(self):
        return CSRTensor(self.indptr, self.indices, -self.values, self.shape)

    def __add__(self, other):
        if not self.shape == other.shape:
            raise ValueError("Input tensors should have the same shape.")
        if isinstance(other, CSRTensor):
            return tensor_operator_registry.get('csr_add')(
                self, other, Tensor(1, self.values.dtype), Tensor(1, self.values.dtype))
        raise TypeError("CSRTensor add with %s is not supported." % type(other))

    def __sub__(self, other):
        if not self.shape == other.shape:
            raise ValueError("Input tensors should have the same shape.")
        if isinstance(other, CSRTensor):
            return tensor_operator_registry.get('csr_add')(
                self, other, Tensor(1, self.values.dtype), Tensor(-1, self.values.dtype))
        raise TypeError("CSRTensor subtract with %s is not supported." % type(other))

    @property
    def indptr(self) -> Tensor:
        """Return CSRTensor's row indices pointers."""
        return Tensor(self._indptr)

    @property
    def indices(self) -> Tensor:
        """Return CSRTensor's column indices."""
        return Tensor(self._indices)

    @property
    def values(self) -> Tensor:
        """Return CSRTensor's non-zero values."""
        return Tensor(self._values)

    @property
    def shape(self) -> Tuple[int, ...]:
        """Return CSRTensor's shape."""
        return self._shape

    @property
    def dtype(self) -> mstype:
        """Return the dtype of the values of CSRTensor (:class:`mindspore.dtype`)."""
        return self._dtype

    @property
    def size(self) -> int:
        """Return the number of non-zero values."""
        return self.values.size

    @property
    def itemsize(self) -> int:
        """Return the length of one tensor element in bytes."""
        return self.values.itemsize

    @property
    def ndim(self) -> int:
        """Return the number of tensor dimensions."""
        return len(self.shape)

    def to_tuple(self) -> Tuple[Tensor, Tensor, Tensor, Tuple[int, ...]]:
        """
        Return indptr, indices, values and shape as a tuple.

        Returns:
            Tuple.

        Supported Platforms:
            ``Ascend`` ``GPU`` ``CPU``
        """
        return self.indptr, self.indices, self.values, self.shape

    def to_coo(self) -> COOTensor:
        """
        Converts CSRTensor to COOTensor.

        Note:
            Currently only supports CPU backend with LLVM 12.0.1 installed.

        Returns:
            COOTensor.

        Supported Platforms:
            ``GPU`` ``CPU``
        """
        if self.ndim != 2:
            raise ValueError("Currently only support 2-D CSRTensor when converting to COOTensor.")
        row_indices = tensor_operator_registry.get("csr2coo")(self.indptr, self.values.shape[0])
        coo_indices = tensor_operator_registry.get("stack")((row_indices, self.indices), 1)
        return COOTensor(coo_indices, self.values, self.shape)

    def to_dense(self) -> Tensor:
        """
        Converts CSRTensor to Dense Tensor.

        Returns:
            Tensor.

        Supported Platforms:
            ``GPU``
        """
        return tensor_operator_registry.get("csr_to_dense")(self)

    def astype(self, dtype: mstype) -> CSRTensor:
        """
        Return a copy of the CSRTensor, cast its values to a specified type.

        Args:
            dtype (Union[:class:`mindspore.dtype`, numpy.dtype, str]): Designated tensor dtype.

        Returns:
            CSRTensor.

        Supported Platforms:
            ``Ascend`` ``GPU`` ``CPU``

        Examples:
            >>> import mindspore as ms
            >>> from mindspore import Tensor, CSRTensor
            >>> indptr = Tensor([0, 1, 2], dtype=ms.int32)
            >>> indices = Tensor([0, 1], dtype=ms.int32)
            >>> values = Tensor([1, 2], dtype=ms.float32)
            >>> shape = (2, 4)
            >>> csr_tensor = CSRTensor(indptr, indices, values, shape)
            >>> print(csr_tensor.astype(ms.float64).dtype)
            Float64
        """
        data = self.values.astype(dtype)
        return CSRTensor(self.indptr, self.indices, data, self.shape)

    def mv(self, dense_vector: Tensor) -> Tensor:
        """
        Return the matrix multiplication result of the right-multiply dense matrix of the CSRTensor.
        The CSRTensor with shape `[M, N]` needs to adapt the dense vector with shape `[N, 1]`
        to get the dense vector with result `[M, 1]`.

        Note:
            Currently only supports CPU backend with LLVM 12.0.1 installed.

        Args:
            dense_vector (Tensor): A dense Tensor, its shape must be (csr_tensor.shape[1], 1)

        Returns:
            Tensor.

        Supported Platforms:
            ``GPU`` ``CPU``

        Examples:
            >>> from mindspore import Tensor, CSRTensor
            >>> from mindspore import dtype as mstype
            >>> indptr = Tensor([0, 1, 2], dtype=mstype.int32)
            >>> indices = Tensor([0, 1], dtype=mstype.int32)
            >>> values = Tensor([2, 1], dtype=mstype.float32)
            >>> dense_shape = (2, 4)
            >>> csr_tensor = CSRTensor(indptr, indices, values, dense_shape)
            >>> dense = Tensor([[1], [1], [1], [1]], dtype=mstype.float32)
            >>> print(csr_tensor.mv(dense))
            [[2.]
            [1.]]
        """
        validator.check_value_type('dense_vector', dense_vector, (Tensor, Tensor_,), 'CSRTensor.mv')
        return tensor_operator_registry.get("csr_mv")(self, dense_vector)

    def mm(self, matrix: Union[Tensor, CSRTensor]) -> Union[Tensor, CSRTensor]:
        """
        Return the matrix multiplication result of the right-multiply matrix（dense or CSRTensor） of the CSRTensor.
        The CSRTensor with shape `[M, N]` needs to adapt the right matrix with shape `[N, K]`
        to get the dense matrix or CSRTensor with result `[M, K]`.

        Note:
            If right matrix is CSRTensor, currently only supports GPU backend.
            if right matrix is Tensor, currently supports CPU backend with LLVM 12.0.1 or GPU backend.

        Args:
            matrix (Tensor or CSRTensor): A dense Tensor or CSRTensor,
                its shape[0] should be equal to csr_tensor.shape[1]

        Returns:
            Tensor or CSRTensor.

        Supported Platforms:
            ``GPU`` ``CPU``

        Examples:
            >>> from mindspore import Tensor, CSRTensor
            >>> from mindspore import dtype as mstype
            >>> indptr = Tensor([0, 1, 2], dtype=mstype.int32)
            >>> indices = Tensor([0, 1], dtype=mstype.int32)
            >>> values = Tensor([2, 1], dtype=mstype.float32)
            >>> dense_shape = (2, 4)
            >>> csr_tensor = CSRTensor(indptr, indices, values, dense_shape)
            >>> dense_matrix = Tensor([[1., 2.], [1, 2.], [1, 2.], [1., 2.]], dtype=mstype.float32)
            >>> print(csr_tensor.mm(dense_matrix))
            [[2. 4.]
            [1. 2.]]
        """
        if isinstance(matrix, CSRTensor):
            return tensor_operator_registry.get("csr_mm")(self, matrix)
        validator.check_value_type('matrix', matrix, (Tensor, Tensor_,), 'CSRTensor.mm')
        return tensor_operator_registry.get("csr_mm_akg")()(self.indptr, self.indices, self.values,
                                                            self.shape, matrix)

    def sum(self, axis: int) -> Tensor:
        """
        Reduces a dimension of a CSRTensor by summing all elements in the dimension.

        Note:
            Currently only supports CPU backend with LLVM 12.0.1 installed.

        Args:
            axis (int): The dimensions to reduce.

        Returns:
            Tensor, the dtype is the same as `CSRTensor.values`.

        Supported Platforms:
            ``GPU`` ``CPU``

        Examples:
            >>> from mindspore import Tensor, CSRTensor
            >>> from mindspore import dtype as mstype
            >>> indptr = Tensor([0, 1, 2], dtype=mstype.int32)
            >>> indices = Tensor([0, 1], dtype=mstype.int32)
            >>> values = Tensor([2, 1], dtype=mstype.float32)
            >>> dense_shape = (2, 4)
            >>> csr_tensor = CSRTensor(indptr, indices, values, dense_shape)
            >>> print(csr_tensor.sum(1))
            [[2.]
            [1.]]
        """
        return tensor_operator_registry.get("csr_reduce_sum")(self, axis)

    def abs(self) -> CSRTensor:
        """
        Return absolute value element-wisely.

        Returns:
            CSRTensor, with all values being non-negative.

        Supported Platforms:
            ``Ascend`` ``GPU`` ``CPU``
        """
        data = self.values.abs()
        return CSRTensor(self.indptr, self.indices, data, self.shape)

    def add(self, b: CSRTensor, alpha: Tensor, beta: Tensor) -> CSRTensor:
        """
        Addition of two CSR Tensors : C = alpha * A + beta * B

        Args:
            b (CSRTensor): Sparse CSR Tensor.
            alpha(Tensor): Dense Tensor, its shape must be able to broadcast to self.
            beta(Tensor): Dense Tensor, its shape must be able to broadcast to b.

        Returns:
            CSRTensor.

        Supported Platforms:
            ``GPU`` ``CPU``

        Examples:
            >>> from mindspore import Tensor, CSRTensor
            >>> import mindspore.common.dtype as mstype
            >>> indptr = Tensor([0, 1, 2], dtype=mstype.int32)
            >>> indices = Tensor([0, 1], dtype=mstype.int32)
            >>> values_a = Tensor([2, 1], dtype=mstype.float32)
            >>> values_b = Tensor([1, 2], dtype=mstype.float32)
            >>> dense_shape = (2, 4)
            >>> alpha = Tensor(1, mstype.float32)
            >>> beta = Tensor(1, mstype.float32)
            >>> a = CSRTensor(indptr, indices, values_a, dense_shape)
            >>> b = CSRTensor(indptr, indices, values_b, dense_shape)
            >>> print(a.add(b, alpha, beta))
                CSRTensor(shape=[2,4], dtype=Float32,
                          indptr=Tensor(shape=[3], dtype=Int32, value = [0, 1, 2]),
                          indices=Tensor(shape=[2], dtype=Int32, value = [0, 1]),
                          values=Tensor(shape=[2], dtype=Float32, value = [3.0, 3.0]))
        """
        return tensor_operator_registry.get('csr_add')(self, b, alpha, beta)
