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
"""Private sparse representations for retained framework execution paths."""
from __future__ import absolute_import, annotations

__all__ = []

from typing import Tuple, Union

from mindspore import log as logger
from mindspore.common import dtype as mstype
from mindspore.common._register_for_tensor import tensor_operator_registry
from mindspore.common.tensor import Tensor
from mindspore._c_expression import COOTensor as COOTensor_
from mindspore._c_expression import CSRTensor as CSRTensor_
from mindspore._c_expression import RowTensor as RowTensor_
from mindspore._c_expression import TensorPy as Tensor_
from mindspore import _checkparam as validator
from mindspore._checkparam import is_stub_tensor


class RowTensorInner(RowTensor_):
    """Private RowTensorInner representation for retained framework paths."""

    def __init__(self, indices=None, values=None, shape=None, row_tensor=None):
        """Init RowTensor"""
        self.init_finished = False
        # Directly init a RowTensor from another RowTensor
        if row_tensor is not None:
            if not isinstance(row_tensor, (RowTensorInner, RowTensor_)):
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


class COOTensor(COOTensor_):
    """Private COOTensor representation for retained framework paths."""

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
        """Returns a coalesced copy of an uncoalesced sparse tensor."""
        shape = Tensor(self.shape)
        res_indices, res_values, _ = tensor_operator_registry.get("coalesce")(self.indices.transpose(),
                                                                              self.values, shape)
        return COOTensor(res_indices.transpose(), res_values, self.shape)

    def to_csr(self) -> CSRTensor:
        """Converts COOTensor to CSRTensor."""
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
        """Converts COOTensor to Dense Tensor."""
        zeros_tensor = tensor_operator_registry.get("zeros")(self.shape, self.values.dtype)
        return tensor_operator_registry.get("tensor_scatter_add")(
            zeros_tensor, self.indices, self.values)

    def astype(self, dtype: mstype) -> COOTensor:
        """Return a copy of the COOTensor, cast its values to a specified type."""
        data = self.values.astype(dtype)
        return COOTensor(self.indices, data, self.shape)

    def to_tuple(self) -> Tuple[Tensor, Tensor, Tuple[int, ...]]:
        """Return indices, values and shape as a tuple."""
        return self.indices, self.values, self.shape

    def abs(self) -> COOTensor:
        """Return absolute value element-wisely."""
        data = self.values.abs()
        return COOTensor(self.indices, data, self.shape)

    def add(self, other: COOTensor, thresh: Tensor) -> COOTensor:
        """Return the sum with another COOTensor."""
        return tensor_operator_registry.get('coo_add')(self, other, thresh)


class CSRTensor(CSRTensor_):
    """Private CSRTensor representation for retained framework paths."""

    def __init__(self, indptr=None, indices=None, values=None, shape=None, csr_tensor=None):
        """Init CSRTensor"""
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
        """Return indptr, indices, values and shape as a tuple."""
        return self.indptr, self.indices, self.values, self.shape

    def to_coo(self) -> COOTensor:
        """Converts CSRTensor to COOTensor."""
        if self.ndim != 2:
            raise ValueError("Currently only support 2-D CSRTensor when converting to COOTensor.")
        row_indices = tensor_operator_registry.get("csr2coo")(self.indptr, self.values.shape[0])
        coo_indices = tensor_operator_registry.get("stack")((row_indices, self.indices), 1)
        return COOTensor(coo_indices, self.values, self.shape)

    def to_dense(self) -> Tensor:
        """Converts CSRTensor to Dense Tensor."""
        return tensor_operator_registry.get("csr_to_dense")(self)

    def astype(self, dtype: mstype) -> CSRTensor:
        """Return a copy of the CSRTensor, cast its values to a specified type."""
        data = self.values.astype(dtype)
        return CSRTensor(self.indptr, self.indices, data, self.shape)

    def mv(self, dense_vector: Tensor) -> Tensor:
        """Return the matrix multiplication result of the right-multiply dense matrix of the CSRTensor.
        The CSRTensor with shape `[M, N]` needs to adapt the dense vector with shape `[N, 1]`
        to get the dense vector with result `[M, 1]`."""
        validator.check_value_type('dense_vector', dense_vector, (Tensor, Tensor_,), 'CSRTensor.mv')
        return tensor_operator_registry.get("csr_mv")(self, dense_vector)

    def mm(self, matrix: Union[Tensor, CSRTensor]) -> Union[Tensor, CSRTensor]:
        """Return the matrix multiplication result of the right-multiply matrix(dense or CSRTensor) of the CSRTensor.
        The CSRTensor with shape `[M, N]` needs to adapt the right matrix with shape `[N, K]`
        to get the dense matrix or CSRTensor with result `[M, K]`."""
        if isinstance(matrix, CSRTensor):
            return tensor_operator_registry.get("csr_mm")(self, matrix)
        validator.check_value_type('matrix', matrix, (Tensor, Tensor_,), 'CSRTensor.mm')
        return tensor_operator_registry.get("csr_mm_akg")()(self.indptr, self.indices, self.values,
                                                            self.shape, matrix)

    def sum(self, axis: int) -> Tensor:
        """Reduces a dimension of a CSRTensor by summing all elements in the dimension."""
        return tensor_operator_registry.get("csr_reduce_sum")(self, axis)

    def abs(self) -> CSRTensor:
        """Return absolute value element-wisely."""
        data = self.values.abs()
        return CSRTensor(self.indptr, self.indices, data, self.shape)

    def add(self, b: CSRTensor, alpha: Tensor, beta: Tensor) -> CSRTensor:
        """Addition of two CSR Tensors : C = alpha * A + beta * B"""
        return tensor_operator_registry.get('csr_add')(self, b, alpha, beta)
