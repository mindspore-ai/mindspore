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

"""Defines sparse operators with functional form."""

from __future__ import absolute_import
from typing import Tuple
from mindspore.ops.operations.sparse_ops import (
    DenseToCSRSparseMatrix,
    CSRSparseMatrixToSparseTensor,
    SparseConcat,
    SparseAdd,
    SparseMatrixAdd,
    SparseMatrixSoftmax,
    SparseMatrixSparseMatMul,
    CSRSparseMatrixToDense
)
from mindspore import ops
from mindspore.common import dtype as mstype
from mindspore.ops.primitive import constexpr, Primitive
from mindspore.ops.operations.array_ops import GatherNd, Coalesce
from mindspore.ops.operations import _csr_ops
from mindspore.ops import functional as F
from mindspore.common import CSRTensor, COOTensor, Tensor
from mindspore.ops.composite.multitype_ops._constexpr_utils import raise_value_error, raise_type_error, make_tensor,\
    promote_binary_dtype

# utility functions and values
gather_nd = GatherNd()
dense_to_csr = DenseToCSRSparseMatrix()
csr_sparse_matrix_to_sparse_tensor = CSRSparseMatrixToSparseTensor()
batch_csr_pointers_empty = Tensor([0, -1], dtype=mstype.int32)
coalesce_op = Coalesce()
csr_sparse_matrix_to_dense = CSRSparseMatrixToDense()


@constexpr
def print_info(info):
    """Print given error info"""
    print(info)


@constexpr
def _make_tensor(data):
    """Make Tensor"""
    return Tensor(data)


@constexpr
def _make_tensor_with_dtype(data, dtype):
    """Make Tensor with specific datatype"""
    return Tensor(data, dtype=dtype)


def _convert_shape(shape):
    """Temporary solution to get shape value, will be removed when shape op is supported."""
    if F.is_sequence_shape_unknown(shape):
        return (-2,)
    shape = [-1 if not F.isconstant(i) else i for i in shape]
    return tuple(shape)


def is_scalar(tensor):
    """Determine whether tensor input is a scalar tensor."""
    if tensor.size != 1:
        return False
    return len(tensor.shape) <= 2


def promote_tensor(tensor_1, tensor_2):
    """promote Tensor"""
    dtype = promote_binary_dtype(tensor_1.dtype, tensor_2.dtype)
    return tensor_1.astype(dtype), tensor_2.astype(dtype)


def promote_csr(csr_tensor_1, csr_tensor_2):
    """Type promotion for CSR tensor."""
    indptr_1, indptr_2 = promote_tensor(csr_tensor_1.indptr, csr_tensor_2.indptr)
    indices_1, indices_2 = promote_tensor(csr_tensor_1.indices, csr_tensor_2.indices)
    values_1, values_2 = promote_tensor(csr_tensor_1.values, csr_tensor_2.values)
    csr_tensor_1 = CSRTensor(indptr_1, indices_1, values_1, csr_tensor_1.shape)
    csr_tensor_2 = CSRTensor(indptr_2, indices_2, values_2, csr_tensor_2.shape)
    return csr_tensor_1, csr_tensor_2


def promote_coo(coo_tensor_1, coo_tensor_2):
    """promote COO-Tensor"""
    indices_1, indices_2 = promote_tensor(coo_tensor_1.indices, coo_tensor_2.indices)
    values_1, values_2 = promote_tensor(coo_tensor_1.values, coo_tensor_2.values)
    coo_tensor_1 = COOTensor(indices_1, values_1, coo_tensor_1.shape)
    coo_tensor_2 = COOTensor(indices_2, values_2, coo_tensor_2.shape)
    return coo_tensor_1, coo_tensor_2


def coalesce(x_indices: Tensor, x_values: Tensor, x_shape: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Returns the coalesced sparse tensor of the input.

    Args:
        - **x_indices** (Tensor) - A 2-D Tensor, represents the indices of the nonzero elements of the sparse tensor.
          Supported data type is int64. It's elements should be non-negative. The shape is :math:`(y, x)`.
        - **x_values** (Tensor) - A 1-D Tensor, represents the values corresponding to the indices in `x_indices`.
          Supported data types are float16 and float32. The shape is :math:`(x,)`.
        - **x_shape** (Tensor) - A 1-D Tensor, specifies the shape of the sparse tensor.
          Supported data type is int64. The shape is :math:`(y,)`.

    Returns:
        - **y_indices** (Tensor) - A 2-D Tensor, represents the indices of the nonzero elements of the sparse tensor.
          Data type is int64. It's elements are non-negative. The shape is :math:`(y, z)`.
          `z` represents the number of different indices in `x_indices`.
        - **y_values** (Tensor) - A 1-D Tensor, represents the values corresponding to the indices in `y_indices`.
          Data type is the same as `x_values`'s. The shape is :math:`(z,)`.
        - **y_shape** (Tensor) - A 1-D Tensor, specifies the shape of the sparse tensor.
          Data type is int64. The shape is :math:`(y,)`.

    Raises:
        TypeError: If the data type of `x_values` is neither float32 nor float16.
        TypeError: If any of the data types of `x_indices` and `x_shape` is not int64.
        ValueError: If any of `x_values` and `x_shape` is not a 1-D tensor.
        ValueError: If `x_indices` is not a 2-D tensor.
        ValueError: If sizes of second dimension of `x_indices` and first dimension of `x_values` are not the same.
        ValueError: If sizes of first dimension of `x_indices` and first dimension of `x_shape` are not the same.
        ValueError: If any of the values of elements of `x_indices` is negative.
        ValueError: If any of the values of elements of `x_indices` exceed the limit set by `x_shape`.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import mindspore.ops as ops
        >>> from mindspore import Tensor
        >>> x_indices = Tensor([[0, 0, 1], [1, 1, 2]], dtype=ms.int64)
        >>> x_values = Tensor([1, 5, 4], dtype=ms.float32)
        >>> x_shape = Tensor([3, 3], dtype=ms.int64)
        >>> y_indices, y_values, y_shape = ops.Coalesce()(x_indices, x_values, x_shape)
        >>> print(y_indices)
        [[0 1]
         [1 2]]
        >>> print(y_values)
        [6. 4.]
        >>> print(y_shape)
        [3 3]
    """
    return coalesce_op(x_indices, x_values, x_shape)


coo2csr = _csr_ops.COO2CSR()

coo_tensor_get_dense_shape = Primitive('COOTensorGetDenseShape')

coo_tensor_get_indices = Primitive('COOTensorGetIndices')

coo_tensor_get_values = Primitive('COOTensorGetValues')


def csr_div(x: CSRTensor, y: Tensor) -> Tensor:
    """
    Returns x / y where x is CSRTensor and y is Tensor.

    Note:
        This function returns the results of dense Tensor, represents the non-zero
        values of the CSRTensor. If user expects a CSRTensor as output, please directly
        use `/` operator instead. Only support dense tensor broadcast to sparse tensor
        at the moment.

    Args:
        x (CSRTensor): Sparse CSR Tensor.
        y (Tensor): Dense Tensor, its shape must be able to broadcast to x.

    Returns:
        Dense Tensor, represents the non-zero values of the result.

    Supported Platforms:
        ``GPU`` ``CPU``
    """
    if isinstance(y, (int, float, bool)):
        y = _make_tensor(y)
    if is_scalar(y):
        if y.ndim > x.ndim:
            raise_value_error("dense tensor cannot broadcast to the sparse tensor.")
        return (x.values / y).reshape(x.values.shape)
    x_values, y = promote_tensor(x.values, y)
    res_values = _csr_ops.CSRDiv()(x.indptr, x.indices, x_values, x.shape, y)
    return CSRTensor(x.indptr, x.indices, res_values, x.shape)


csr_gather = _csr_ops.CSRGather()


def csr_mul(x: CSRTensor, y: Tensor) -> CSRTensor:
    """
    Returns x * y where x is CSRTensor and y is Tensor.

    Args:
        x (CSRTensor): Sparse CSR Tensor.
        y (Tensor): Dense Tensor, its shape must be able to broadcast to x.

    Returns:
        CSRTensor.

    Supported Platforms:
        ``GPU`` ``CPU``
    """
    if isinstance(y, (int, float, bool)):
        y = _make_tensor(y)
    if is_scalar(y):
        if y.ndim > x.ndim:
            raise_value_error("dense tensor cannot broadcast to the sparse tensor.")
        return (x.values * y).reshape(x.values.shape)
    x_values, y = promote_tensor(x.values, y)
    res_values = _csr_ops.CSRMul()(x.indptr, x.indices, x_values, x.shape, y)
    return CSRTensor(x.indptr, x.indices, res_values, x.shape)


def csr_mv(csr_tensor: CSRTensor, dense: Tensor) -> Tensor:
    """
    Sparse matrix-vector multiplication.

    Args:
        csr_tensor (CSRTensor): Sparse CSR Tensor.
        dense (Tensor): Dense Tensor.

    Returns:
        Dense Tensor.

    Supported Platforms:
        ``GPU`` ``CPU``
    """
    csr_tensor_values, dense = promote_tensor(csr_tensor.values, dense)
    return _csr_ops.CSRMV()(csr_tensor.indptr, csr_tensor.indices, csr_tensor_values, csr_tensor.shape, dense)


def csr_mm(a: CSRTensor, b: CSRTensor, trans_a: bool = False, trans_b: bool = False,
           adjoint_a: bool = False, adjoint_b: bool = False):
    """
    Return the matrix multiplication result of the right-multiply matrix (dense or CSRTensor) of the CSRTensor.
    The CSRTensor with shape `[M, N]` needs to adapt the right matrix with shape `[N, K]`
    to get the dense matrix or CSRTensor with result `[M, K]`.

    Note:
        If right matrix is CSRTensor, currently only supports GPU backend.
        If right matrix is Tensor, currently supports CPU backend with LLVM no lower than 12.0.1, and GPU backend.

    Args:
        a (CSRTensor): Sparse CSR Tensor, rank should be 2.
        b (CSRTensor): Sparse CSR Tensor, rank should be 2.
        trans_a (bool, optional): whether to transpose CSRTensor a. Default: ``False`` .
        trans_b (bool, optional): whether to transpose CSRTensor b. Default: ``False`` .
        adjoint_a (bool, optional): whether to adjoint CSRTensor a. Default: ``False`` .
        adjoint_b (bool, optional): whether to adjoint CSRTensor b. Default: ``False`` .

    Returns:
        CSRTensor.

    Supported Platforms:
        ``GPU``

    Examples:
        >>> from mindspore import Tensor, CSRTensor
        >>> from mindspore import dtype as mstype
        >>> import mindspore.ops as ops
        >>> a_shape = (4, 5)
        >>> a_indptr = Tensor([0, 1, 1, 3, 4], dtype=mstype.int32)
        >>> a_indices = Tensor([0, 3, 4, 0],dtype=mstype.int32)
        >>> a_values = Tensor([1.0, 5.0, -1.0, -2.0], dtype=mstype.float32)
        >>> b_shape = (5, 3)
        >>> b_indptr = Tensor([0, 1, 1, 3, 3, 3], dtype=mstype.int32)
        >>> b_indices = Tensor([0, 0, 1],dtype=mstype.int32)
        >>> b_values = Tensor([2.0, 7.0, 8.0], dtype=mstype.float32)
        >>> a = CSRTensor(a_indptr, a_indices, a_values, a_shape)
        >>> b = CSRTensor(b_indptr, b_indices, b_values, b_shape)
        >>> c = ops.csr_mm(a, b)
        >>> print(c.shape)
        (4, 3)
        >>> print(c.values)
        [2. -4.]
        >>> print(c.indptr)
        [0 1 1 1 2]
        >>> print(c.indices)
        [0 0]
    """
    if not isinstance(a, CSRTensor) or not isinstance(b, CSRTensor):
        raise_type_error("For functional operator csr_mm, inputs a and b must be type of CSRTensor currently.")
    a_batch_pointers = make_tensor([0, a.values.shape[0]], a.indices.dtype)
    b_batch_pointers = make_tensor([0, b.values.shape[0]], b.indices.dtype)
    a_shape = make_tensor(a.shape, a.indices.dtype)
    b_shape = make_tensor(b.shape, b.indices.dtype)
    sparse_matrix_sparse_matmul = SparseMatrixSparseMatMul(transpose_a=trans_a,
                                                           transpose_b=trans_b,
                                                           adjoint_a=adjoint_a,
                                                           adjoint_b=adjoint_b)

    _, _, c_indptr, c_indices, c_values = sparse_matrix_sparse_matmul(a_shape,
                                                                      a_batch_pointers,
                                                                      a.indptr,
                                                                      a.indices,
                                                                      a.values,
                                                                      b_shape,
                                                                      b_batch_pointers,
                                                                      b.indptr,
                                                                      b.indices,
                                                                      b.values)
    if trans_a or adjoint_a:
        return CSRTensor(c_indptr, c_indices, c_values, (a.shape[1], b.shape[1]))
    if trans_b or adjoint_b:
        return CSRTensor(c_indptr, c_indices, c_values, (a.shape[0], b.shape[0]))
    return CSRTensor(c_indptr, c_indices, c_values, (a.shape[0], b.shape[1]))


def csr_reduce_sum(csr_tensor: CSRTensor, axis: int) -> Tensor:
    """
    Reduces a dimension of a CSRTensor by summing all elements in the dimension.

    Args:
        csr_tensor (CSRTensor): Sparse CSR Tensor.
        axis (int): Axis to be reduced.

    Returns:
        Dense Tensor, represents the non-zero values of the result.

    Supported Platforms:
        ``GPU`` ``CPU``
    """
    return _csr_ops.CSRReduceSum()(csr_tensor.indptr, csr_tensor.indices, csr_tensor.values, csr_tensor.shape, axis)


def csr_to_coo(tensor: CSRTensor) -> COOTensor:
    """
    Converts a CSRTensor to COOTensor.

    Note:
        Only 2-D CSRTensor is supported for now.

    Args:
        tensor (CSRTensor): A CSRTensor, must be 2-D.

    Returns:
        2D COOTensor, the input tensor stored in COO format.

    Raises:
        TypeError: If input is not a COOTensor.
        ValueError: If input tensor is not 2-D.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> from mindspore import Tensor, CSRTensor
        >>> indptr = Tensor([0, 1, 2]).astype("int32")
        >>> indices = Tensor([0, 1]).astype("int32")
        >>> values = Tensor([2, 1]).astype("float32")
        >>> shape = (2, 4)
        >>> x = CSRTensor(indptr, indices, values, shape)
        >>> output = ops.csr_to_coo(x)
        >>> print(output.indices)
        [[0 0]
        [1 1]]
    """
    if not isinstance(tensor, CSRTensor):
        raise_type_error("For functional operator csr_to_coo, input argument must be a CSRTensor.")
    if len(_convert_shape(tensor.shape)) > 2:
        raise_value_error("Currently only support 2-D CSRTensor when converting to COOTensor.")
    shape = tensor.shape
    indices, values, _ = csr_sparse_matrix_to_sparse_tensor(Tensor(shape, mstype.int32), batch_csr_pointers_empty,
                                                            tensor.indptr, tensor.indices, tensor.values)
    return COOTensor(indices, values, _convert_shape(shape))


def csr_to_dense(csr_tensor: CSRTensor) -> Tensor:
    """
    Converts a CSRTensor to its dense form.

    Note:
        Only 2-D CSRTensor is supported for now.

    Args:
        csr_tensor (CSRTensor): A CSRTensor, must be 2-D.

    Returns:
        Tensor.

    Raises:
        TypeError: If input is not a CSRTensor.
        ValueError: If input CSRTensor is not 2-D.

    Supported Platforms:
        ``GPU``

    Examples:
        >>> from mindspore import Tensor, CSRTensor, ops
        >>> indptr = Tensor([0, 1, 2]).astype("int32")
        >>> indices = Tensor([0, 1]).astype("int32")
        >>> values = Tensor([2, 1]).astype("float32")
        >>> shape = (2, 4)
        >>> x = CSRTensor(indptr, indices, values, shape)
        >>> output = ops.csr_to_dense(x)
        >>> print(output)
    """
    if not isinstance(csr_tensor, CSRTensor):
        raise_type_error("For functional operator csr_to_dense, input argument must be a CSRTensor.")
    if len(csr_tensor.shape) > 2:
        raise_value_error("Currently only support 2-D CSRTensor when converting to COOTensor.")

    shape = _convert_shape(csr_tensor.shape)
    dense_shape = Tensor(shape, dtype=mstype.int32)
    batch_pointers = ops.concat((make_tensor([0]), ops.TensorShape()(csr_tensor.values))).astype("int32")
    row_pointers = csr_tensor.indptr
    col_indices = csr_tensor.indices
    values = csr_tensor.values

    valid_indices_dtype = [mstype.int32, mstype.int64]
    if row_pointers.dtype in valid_indices_dtype and col_indices.dtype in valid_indices_dtype:
        if mstype.int64 in (row_pointers.dtype, col_indices.dtype):
            return csr_sparse_matrix_to_dense(dense_shape.astype(mstype.int64), batch_pointers.astype(mstype.int64),
                                              row_pointers.astype(mstype.int64), col_indices.astype(mstype.int64),
                                              values)

    return csr_sparse_matrix_to_dense(dense_shape, batch_pointers, row_pointers, col_indices, values)


# deprecated, will be removed once `csr_to_coo` supports all backends.
csr2coo = _csr_ops.CSR2COO()

csr_tensor_get_dense_shape = Primitive('CSRTensorGetDenseShape')

csr_tensor_get_indices = Primitive('CSRTensorGetIndices')

csr_tensor_get_indptr = Primitive('CSRTensorGetIndptr')

csr_tensor_get_values = Primitive('CSRTensorGetValues')


def dense_to_sparse_coo(tensor: Tensor) -> COOTensor:
    """
    Convert a Tensor to COOTensor.

    Note:
        Only 2-D tensor is supported for now.

    Args:
        tensor (Tensor): A dense tensor, must be 2-D.

    Returns:
        COOTensor, a sparse representation of the original dense tensor, containing the following parts.

        - indices (Tensor): 2-D integer tensor, indicates the positions of `values` of the dense tensor.
        - values (Tensor): 1-D tensor, indicates the non-zero values of the dense tensor.
        - shape (tuple(int)): the shape of the COOTensor, is the same as the original dense tensor.

    Raises:
        TypeError: If input is not a tensor.
        ValueError: If input tensor is not 2-D.

    Supported Platforms:
        ``GPU``

    Examples:
        >>> from mindspore import Tensor, ops
        >>> import mindspore as ms
        >>> x = Tensor([[1, 0], [-5, 0]], ms.float32)
        >>> output = ops.dense_to_sparse_coo(x)
        >>> print(output.indices)
        [[0 0]
        [1 0]]
        >>> print(output.values)
        [ 1. -5.]
        >>> print(output.shape)
        (2, 2)
    """
    if not isinstance(tensor, Tensor):
        raise_type_error("For functional operator dense_to_sparse_coo, input argument must be a Tensor.")
    if len(_convert_shape(tensor.shape)) > 2:
        raise_value_error("Currently only support 2-D Tensor when converting to COOTensor.")
    indices = tensor.nonzero().astype("int32")
    values = gather_nd(tensor, indices)
    return COOTensor(indices, values, _convert_shape(tensor.shape))


def dense_to_sparse_csr(tensor: Tensor) -> CSRTensor:
    """
    Convert a Tensor to CSRTensor.

    Note:
        Only 2-D tensor is supported for now.

    Args:
        tensor (Tensor): A dense tensor, must be 2-D.

    Returns:
        CSRTensor, a sparse representation of the original dense tensor, containing the following parts.

        - indptr (Tensor): 1-D integer tensor, indicates the start and end point for `values` in each row.
        - indices (Tensor): 1-D integer tensor, indicates the column positions of all non-zero values of the input.
        - values (Tensor): 1-D tensor, indicates the non-zero values of the dense tensor.
        - shape (tuple(int)): the shape of the CSRTensor, is the same as the original dense tensor.

    Raises:
        TypeError: If input is not a tensor.
        ValueError: If input tensor is not 2-D.

    Supported Platforms:
        ``GPU``

    Examples:
        >>> from mindspore import Tensor, ops
        >>> import mindspore as ms
        >>> x = Tensor([[1, 0], [-5, 0]], ms.float32)
        >>> output = ops.dense_to_sparse_csr(x)
        >>> print(output.indptr)
        [0 1 2]
        >>> print(output.indices)
        [0 0]
        >>> print(output.shape)
        (2, 2)
    """
    if not isinstance(tensor, Tensor):
        raise_type_error("For functional operator dense_to_sparse_csr, input argument must be a Tensor.")
    if len(_convert_shape(tensor.shape)) > 2:
        raise_value_error("Currently only support 2-D Tensor when converting to CSRTensor.")
    indices = tensor.nonzero().astype("int32")
    _, _, indptr, indices, values = dense_to_csr(tensor, indices)
    return CSRTensor(indptr, indices, values, _convert_shape(tensor.shape))


def make_sparse_tensor(indices, values, dense_shape):
    """Call make_coo_tensor in this function."""
    print_info("WARNING: 'SparseTensor' is deprecated from version 1.7 and will be removed in a future version. " +
               "Please use 'COOTensor' instead.")
    return make_coo_tensor(indices, values, dense_shape)


def make_row_tensor(indices, values, dense_shape):
    """Call make_row_tensor_inner in this function."""
    print_info("WARNING: 'RowTensor' is deprecated from version 2.0 and will be removed in a future version.")
    return make_row_tensor_inner(indices, values, dense_shape)


make_coo_tensor = Primitive('MakeCOOTensor')

make_csr_tensor = Primitive('MakeCSRTensor')

make_row_tensor_inner = Primitive('MakeRowTensor')

make_map_parameter = Primitive('MakeMapParameter')

row_tensor_get_values = Primitive('RowTensorGetValues')

row_tensor_get_indices = Primitive('RowTensorGetIndices')

row_tensor_get_dense_shape = Primitive('RowTensorGetDenseShape')

row_tensor_add = Primitive('RowTensorAdd')


@constexpr
def _calc_out_shape(sp_input, concat_dim):
    "calculating the COOTensor output shape in coo_concat"
    if isinstance(sp_input[0], tuple):
        out_shape_list = list(sp_input[0][2])
    else:
        out_shape_list = list(sp_input[0].shape)
    for i in range(1, len(sp_input)):
        if isinstance(sp_input[i], tuple):
            out_shape_list[concat_dim] += sp_input[i][2][concat_dim]
        else:
            out_shape_list[concat_dim] += sp_input[i].shape[concat_dim]
    return tuple(out_shape_list)


@constexpr
def _set_coo_concat_input(sp_input):
    "split COOTensor to normal tensor"
    if len(sp_input) < 2:
        raise_value_error("For coo_concat, not support COOTensor input number < 2.")
    in_indices = []
    in_values = []
    in_shapes = []
    for element in sp_input:
        if isinstance(element, tuple):
            in_indices.append(element[0])
            in_values.append(element[1])
            in_shapes.append(Tensor(element[2], dtype=mstype.int64))
        else:
            in_indices.append(element.indices)
            in_values.append(element.values)
            in_shapes.append(Tensor(element.shape, dtype=mstype.int64))
    return in_indices, in_values, in_shapes


def coo_concat(sp_input, concat_dim=0):
    """
    concatenates the input SparseTensor(COO format) along the specified dimension.

    .. warning::
        This is an experimental API that is subjected to change or deletion. Only supported on CPU now.

    Args:
        sp_input (Union[list(COOTensor), tuple(COOTensor)]) - the list of SparseTensor which need to concatenates.
            for COOTensor input.
        concat_dim (scalar): decide the dimension to concatenation along.
            The value must be in range [-rank, rank), where rank is the number of dimensions in each input
            SparseTensor. Default is 0.

    Returns:
        - **output** (COOtensor) - the result of concatenates the input SparseTensor along the
          specified dimension. OutShape: OutShape[non concat_dim] is equal to InShape[non concat_dim] and
          OutShape[concat_dim] is all input concat_dim axis shape accumulate.

    Raises:
        ValueError: If only one sparse tensor input.
        ValueError: If Input COOTensor shape dim > 3. COOtensor shape dim size must be 2 now.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> from mindspore import Tensor, ops, COOTensor
        >>> from mindspore import dtype as mstype
        >>> indices0 = Tensor([[0, 1], [1, 2]], dtype=mstype.int64)
        >>> values0 = Tensor([1, 2], dtype=mstype.int32)
        >>> shape0 = (3, 4)
        >>> input0 = COOTensor(indices0, values0, shape0)
        >>> indices1 = Tensor([[0, 0], [1, 1]], dtype=mstype.int64)
        >>> values1 = Tensor([3, 4], dtype=mstype.int32)
        >>> shape1 = (3, 4)
        >>> input1 = COOTensor(indices1, values1, shape1)
        >>> concat_dim = 1
        >>> out = ops.coo_concat((input0, input1), concat_dim)
        >>> print(out)
        COOTensor(shape=[3, 8], dtype=Int32, indices=Tensor(shape=[4, 2], dtype=Int64, value=
        [[0 1]
         [0 4]
         [1 2]
         [1 5]]), values=Tensor(shape=[4], dtype=Int32, value=[1 3 2 4]))
    """
    coo_concat_op = SparseConcat(concat_dim)
    in_indices, in_values, in_shapes = _set_coo_concat_input(sp_input)
    indices, values, _ = coo_concat_op(in_indices, in_values, in_shapes)
    out_shape = _calc_out_shape(sp_input, concat_dim)
    return COOTensor(indices, values, out_shape)


def coo_add(x1: COOTensor, x2: COOTensor, thresh: Tensor) -> COOTensor:
    """
    Computes the sum of x1(COOTensor) and x2(COOTensor), and return a new COOTensor
    based on the computed result and `thresh`.

    Args:
        x1 (COOTensor): the first COOTensor to sum.
        x2 (COOTensor): the second COOTensor to sum.
        thresh (Tensor): A 0-D Tensor, represents the magnitude threshold that determines
            if an output value/index pair take place. Its dtype
            should match that of the values if they are real. If output's
            value is less than the `thresh`, it will vanish.

    Returns:
        A COOTensor, the result of sum.

    Raises:
        ValueError: If any input(x1/x2)'s indices's dim is not equal to 2.
        ValueError: If any input(x1/x2)'s values's dim is not equal to 1.
        ValueError: If any input(x1/x2)'s shape's dim is not equal to 1.
        ValueError: If thresh's dim is not equal to 0.
        TypeError: If any input(x1/x2)'s indices's type is not equal to int64.
        TypeError: If any input(x1/x2)'s shape's type is not equal to int64.
        ValueError: If any input(x1/x2)'s indices's length is not equal to
            its values's length.
        TypeError: If any input(x1/x2)'s values's type is not equal to anf of
            (int8/int16/int32/int64/float32/float64/complex64/complex128).
        TypeError: If thresh's type is not equal to anf of
            (int8/int16/int32/int64/float32/float64).
        TypeError: If x1's indices's type is not equal to x2's indices's type.
        TypeError: If x1's values's type is not equal to x2's values's type.
        TypeError: If x1's shape's type is not equal to x2's shape's type.
        TypeError: If (x1/x2)'s value's type is not matched with thresh's type.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> from mindspore import Tensor, COOTensor
        >>> from mindspore import dtype as mstype
        >>> from mindspore import context
        >>> from mindspore import ops
        >>> indics0 = Tensor([[0, 1], [1, 2]], dtype=mstype.int64)
        >>> values0 = Tensor([1, 2], dtype=mstype.int32)
        >>> shape0 = (3, 4)
        >>> input0 = COOTensor(indics0, values0, shape0)
        >>> indics1 = Tensor([[0, 0], [1, 1]], dtype=mstype.int64)
        >>> values1 = Tensor([3, 4], dtype=mstype.int32)
        >>> shape1 = (3, 4)
        >>> input1 = COOTensor(indics1, values1, shape1)
        >>> thres = Tensor(0, dtype=mstype.int32)
        >>> out = ops.coo_add(input0, input1, thres)
        >>> print(out)
        COOTensor(shape=[3, 4], dtype=Int32, indices=Tensor(shape=[4, 2], dtype=Int64, value=
        [[0 0]
         [0 1]
         [1 1]
         [1 2]]), values=Tensor(shape=[4], dtype=Int32, value=[3 1 4 2]))
    """
    x1, x2 = promote_coo(x1, x2)
    thresh = thresh.astype(x1.dtype)
    x1_indices = x1.indices
    x1_values = x1.values
    x2_indices = x2.indices
    x2_values = x2.values
    den_shp = make_tensor(x1.shape, x1_indices.dtype)
    add_op = SparseAdd()
    indices, values, _ = add_op(x1_indices, x1_values, den_shp, x2_indices, x2_values, den_shp, thresh)
    return COOTensor(indices, values, x1.shape)


def csr_softmax(logits: CSRTensor, dtype: mstype):
    """
    Calculates the softmax of a CSRTensorMatrix.

    Args:
        logits (CSRTensor): Input sparse CSRTensor.
        dtype (dtype): Input data type.

    Returns:
        CSRTensor, a CSRTensor containing

        - **indptr** - Indicates the start and end point for non-zero values in each row.
        - **indices** - The column positions of all non-zero values of the input.
        - **values** - The non-zero values of the dense tensor.
        - **shape** - The shape of the CSRTensor.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> import mindspore as ms
        >>> import mindspore.ops as ops
        >>> import mindspore.common.dtype as mstype
        >>> from mindspore import Tensor, CSRTensor
        >>> logits_indptr = Tensor([0, 4, 6], dtype=mstype.int32)
        >>> logits_indices = Tensor([0, 2, 3, 4, 3, 4], dtype=mstype.int32)
        >>> logits_values = Tensor([1, 2, 3, 4, 1, 2], dtype=mstype.float32)
        >>> shape = (2, 6)
        >>> logits = CSRTensor(logits_indptr, logits_indices, logits_values, shape)
        >>> out = ops.csr_softmax(logits, dtype=mstype.float32)
        >>> print(out)
        CSRTensor(shape=[2, 6], dtype=Float32, indptr=Tensor(shape=[3], dtype=Int32, value=[0 4 6]),
                       indices=Tensor(shape=[6], dtype=Int32, value=[0 2 3 4 3 4]),
                       values=Tensor(shape=[6], dtype=Float32, value=[ 3.20586003e-02  8.71443152e-02  2.36882806e-01
                       6.43914223e-01  2.68941432e-01  7.31058598e-01]))
    """
    if not isinstance(logits, CSRTensor):
        raise_type_error("For functional operator sparse_matrix_softmax, logits must be type of CSRTensor.")
    sparse_matrix_softmax_op = SparseMatrixSoftmax(dtype)
    logits_batch_pointers = make_tensor([0, logits.values.shape[0]], mstype.int32)
    logits_shape = make_tensor(logits.shape, mstype.int32)
    shape, _, indptr, indices, values = sparse_matrix_softmax_op(logits_shape, logits_batch_pointers, logits.indptr,
                                                                 logits.indices, logits.values)
    output_shape = tuple(shape.asnumpy().tolist())
    return CSRTensor(indptr=indptr, indices=indices, values=values, shape=output_shape)


def csr_add(a: CSRTensor, b: CSRTensor, alpha: Tensor, beta: Tensor) -> CSRTensor:
    """
    Computes the linear combination of two input CSRTensors a and b.

    .. math::

        out = alpha * a + beta * b

    where both :math:`a` and :math:`b` are CSRTensor, :math:`alpha` and :math:`beta` are both Tensor

    Note:
        The user need to ensure that the input sparse matrix is legal.
        Otherwise, the behavior of the operator is undefined.
        For example, when there are multiple elements in the same position, the
        operator may return an error of fail execute.

    Args:
        a (CSRTensor): Input sparse CSRTensor.
        b (CSRTensor): Input sparse CSRTensor.
        alpha(Tensor): Dense Tensor, its shape must be able to broadcast to a.
        beta(Tensor): Dense Tensor, its shape must be able to broadcast to b.

    Returns:
        CSRTensor, a CSRTensor containing the following parts.

        - **indptr** -  Indicates the start and end point for non-zero values in each row.
        - **indices** - The column positions of all non-zero values of the input.
        - **values** - The non-zero values of the dense tensor.
        - **shape** - The shape of the CSRTensor.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.common.dtype as mstype
        >>> from mindspore import Tensor, CSRTensor
        >>> import mindspore.ops as ops
        >>> a_indptr = Tensor([0, 1, 2], dtype=mstype.int32)
        >>> a_indices = Tensor([0, 1], dtype=mstype.int32)
        >>> a_values = Tensor([1, 2], dtype=mstype.float32)
        >>> shape = (2, 6)
        >>> b_indptr = Tensor([0, 1, 2], dtype=mstype.int32)
        >>> b_indices = Tensor([0, 1], dtype=mstype.int32)
        >>> b_values = Tensor([1, 2], dtype=mstype.float32)
        >>> alpha = Tensor(1, mstype.float32)
        >>> beta = Tensor(1, mstype.float32)
        >>> csra = CSRTensor(a_indptr, a_indices, a_values, shape)
        >>> csrb = CSRTensor(b_indptr, b_indices, b_values, shape)
        >>> out = ops.csr_add(csra, csrb, alpha, beta)
        >>> print(out)
        CSRTensor(shape=[2, 6], dtype=Float32, \
                  indptr=Tensor(shape=[3], dtype=Int32, value=[0 1 2]), \
                  indices=Tensor(shape=[2], dtype=Int32, value=[0 1]), \
                  values=Tensor(shape=[2], dtype=Float32, value=[ 2.00000000e+00  4.00000000e+00]))
    """
    if not isinstance(a, CSRTensor) or not isinstance(b, CSRTensor):
        raise_type_error("For functional operator csr_add, both inputs a and b must be type of CSRTensor.")
    if not isinstance(alpha, Tensor) or not isinstance(beta, Tensor):
        raise_type_error("For functional operator csr_add, both inputs alpha and beta must be Tensor.")
    csr_add_op = SparseMatrixAdd()
    a, b = promote_csr(a, b)
    alpha = alpha.astype(a.dtype)
    beta = beta.astype(a.dtype)
    a_batch_pointers = ops.concat((make_tensor([0]), ops.TensorShape()(a.values))).astype(a.indptr.dtype)
    b_batch_pointers = ops.concat((make_tensor([0]), ops.TensorShape()(b.values))).astype(b.indptr.dtype)
    a_shape = make_tensor(a.shape, a.indptr.dtype)
    b_shape = make_tensor(b.shape, b.indptr.dtype)
    _, _, indptr, indices, values = csr_add_op(a_shape, a_batch_pointers, a.indptr, a.indices, a.values,
                                               b_shape, b_batch_pointers, b.indptr, b.indices, b.values,
                                               alpha, beta)
    return CSRTensor(indptr, indices, values, a.shape)


__all__ = [
    'coalesce',
    'coo2csr',
    'coo_tensor_get_dense_shape',
    'coo_tensor_get_indices',
    'coo_tensor_get_values',
    'csr_div',
    'csr_gather',
    'csr_mm',
    'csr_mul',
    'csr_mv',
    'csr_reduce_sum',
    'csr_to_coo',
    'csr2coo',
    'csr_tensor_get_dense_shape',
    'csr_tensor_get_indices',
    'csr_tensor_get_indptr',
    'csr_tensor_get_values',
    'dense_to_sparse_coo',
    'dense_to_sparse_csr',
    'make_sparse_tensor',
    'make_coo_tensor',
    'make_csr_tensor',
    'make_row_tensor',
    'make_row_tensor_inner',
    'make_map_parameter',
    'row_tensor_get_values',
    'row_tensor_get_indices',
    'row_tensor_get_dense_shape',
    'row_tensor_add',
    'coo_add',
    'coo_concat',
    'csr_add',
    'csr_softmax',
    'csr_to_dense'
]

__all__.sort()
