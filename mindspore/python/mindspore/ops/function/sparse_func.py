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

from ..primitive import constexpr, Primitive
from ..operations.sparse_ops import (
    DenseToCSRSparseMatrix,
    CSRSparseMatrixToSparseTensor,
    SparseConcat,
    SparseAdd
)
from ..operations.array_ops import GatherNd, Coalesce
from ..operations import _csr_ops
from ...common import CSRTensor, COOTensor, Tensor
from ...common import dtype as mstype
from ..composite.multitype_ops._constexpr_utils import raise_value_error, raise_type_error

# utility functions and values
gather_nd = GatherNd()
dense_to_csr = DenseToCSRSparseMatrix()
csr_sparse_matrix_to_sparse_tensor = CSRSparseMatrixToSparseTensor()
batch_csr_pointers_empty = Tensor([0, -1], dtype=mstype.int32)
coalesce_op = Coalesce()


@constexpr
def print_info(info):
    """Print given error info"""
    print(info)


@constexpr
def _make_tensor(data):
    """Make Tensor"""
    return Tensor(data)


def is_scalar(tensor):
    """Determine whether tensor input is a scalar tensor."""
    if tensor.size != 1:
        return False
    return len(tensor.shape) <= 2


def coalesce(x_indices, x_values, x_shape):
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
        ``GPU``

    Examples:
        >>> import mindspore
        >>> import mindspore.ops as ops
        >>> from mindspore import Tensor
        >>> x_indices = Tensor([[0, 0, 1], [1, 1, 2]], dtype=ms.int64)
        >>> x_values = Tensor([1, 5, 4], dtype=ms.float32)
        >>> x_shape = Tensor([3, 3], dtype=ms.int64)
        >>> y_indices, y_values, y_shape = ops.coalesce(x_indices, x_values, x_shape)
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


def csr_div(x, y):
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
    return _csr_ops.CSRDiv()(x.indptr, x.indices, x.values, x.shape, y)


csr_gather = _csr_ops.CSRGather()


def csr_mul(x, y):
    """
    Returns x * y where x is CSRTensor and y is Tensor.

    Note:
        This function returns the results of dense Tensor, represents the non-zero
        values of the CSRTensor. If user expects a CSRTensor as output, please directly
        use `*` operator instead. Only support dense tensor broadcast to sparse tensor
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
        return (x.values * y).reshape(x.values.shape)
    return _csr_ops.CSRMul()(x.indptr, x.indices, x.values, x.shape, y)


def csr_mv(csr_tensor, dense):
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
    return _csr_ops.CSRMV()(csr_tensor.indptr, csr_tensor.indices, csr_tensor.values, csr_tensor.shape, dense)


def csr_reduce_sum(csr_tensor, axis):
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


def csr_to_coo(tensor):
    """
    Converts a CSRTensor to COOTensor.

    Note:
        Only 2-D CSRTensor is supported for now.

    Args:
        tensor: A CSRTensor, must be 2-D.

    Returns:
        2D COOTensor, the input tensor stored in COO format.

    Raises:
        TypeError: If input is not a COOTensor.
        ValueError: If input tensor is not 2-D.

    Supported Platforms:
        ``GPU``

    Examples:
        >>> from mindspore import Tensor, CSRTensor
        >>> indptr = Tensor([0, 1, 2]).astype("int32")
        >>> indices = Tensor([0, 1]).astype("int32")
        >>> values = Tensor([2, 1]).astype("float32")
        >>> shape = (2, 4)
        >>> x = CSRTensor(indptr, indices, values, shape)
        >>> output = ops.csr_to_coo(x)
        >>> print(output)
    """
    if not isinstance(tensor, CSRTensor):
        raise_type_error("For functional operator csr_to_coo, input argument must be a CSRTensor.")
    if len(tensor.shape) != 2:
        raise_value_error("Currently only support 2-D CSRTensor when converting to COOTensor.")
    shape = tensor.shape
    indices, values, _ = csr_sparse_matrix_to_sparse_tensor(Tensor(shape, dtype=mstype.int32), batch_csr_pointers_empty,
                                                            tensor.indptr, tensor.indices, tensor.values)
    return COOTensor(indices, values, shape)


# deprecated, will be removed once `csr_to_coo` supports all backends.
csr2coo = _csr_ops.CSR2COO()


csr_tensor_get_dense_shape = Primitive('CSRTensorGetDenseShape')


csr_tensor_get_indices = Primitive('CSRTensorGetIndices')


csr_tensor_get_indptr = Primitive('CSRTensorGetIndptr')


csr_tensor_get_values = Primitive('CSRTensorGetValues')


def dense_to_sparse_coo(tensor):
    """
    Convert a Tensor to COOTensor.

    Note:
        Only 2-D tensor is supported for now.

    Args:
        tensor: A dense tensor, must be 2-D.

    Returns:
        COOTensor, a sparse representation of the original dense tensor, containing:

        - indices (Tensor): 2-D integer tensor, indicates the positions of `values` of the dense tensor.
        - values (Tensor): 1-D tensor, indicates the non-zero values of the dense tensor.
        - shape (tuple(int)): the shape of the COOTensor, is the same as the original dense tensor.

    Raises:
        TypeError: If input is not a tensor.
        ValueError: If input tensor is not 2-D.

    Supported Platforms:
        ``GPU``

    Examples:
        >>> from mindspore import Tensor
        >>> import mindspore as ms
        >>> x = Tensor([[1, 0], [-5, 0]], ms.float32)
        >>> output = ops.dense_to_sparse_coo(x)
        >>> print(output)
    """
    if not isinstance(tensor, Tensor):
        raise_type_error("For functional operator dense_to_sparse_coo, input argument must be a Tensor.")
    if len(tensor.shape) != 2:
        raise_value_error("Currently only support 2-D Tensor when converting to COOTensor.")
    indices = tensor.nonzero().astype("int32")
    values = gather_nd(tensor, indices)
    return COOTensor(indices, values, tensor.shape)


def dense_to_sparse_csr(tensor):
    """
    Convert a Tensor to CSRTensor.

    Note:
        Only 2-D tensor is supported for now.

    Args:
        tensor: A dense tensor, must be 2-D.

    Returns:
        CSRTensor, a sparse representation of the original dense tensor, containing:

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
        >>> from mindspore import Tensor
        >>> import mindspore as ms
        >>> x = Tensor([[1, 0], [-5, 0]], ms.float32)
        >>> output = ops.dense_to_sparse_csr(x)
        >>> print(output)
    """
    if not isinstance(tensor, Tensor):
        raise_type_error("For functional operator dense_to_sparse_csr, input argument must be a Tensor.")
    if len(tensor.shape) != 2:
        raise_value_error("Currently only support 2-D Tensor when converting to CSRTensor.")
    indices = tensor.nonzero().astype("int32")
    _, _, indptr, indices, values = dense_to_csr(tensor, indices)
    return CSRTensor(indptr, indices, values, tensor.shape)


def make_sparse_tensor(indices, values, dense_shape):
    """Call make_coo_tensor in this function."""
    print_info("WARNING: 'SparseTensor' is deprecated from version 1.7 and will be removed in a future version. " +
               "Please use 'COOTensor' instead.")
    return make_coo_tensor(indices, values, dense_shape)


make_coo_tensor = Primitive('MakeCOOTensor')


make_csr_tensor = Primitive('MakeCSRTensor')


make_row_tensor = Primitive('MakeRowTensor')


row_tensor_get_values = Primitive('RowTensorGetValues')


row_tensor_get_indices = Primitive('RowTensorGetIndices')


row_tensor_get_dense_shape = Primitive('RowTensorGetDenseShape')


row_tensor_add = Primitive('RowTensorAdd')


@constexpr
def _calc_out_shape(sp_input, concat_dim):
    "calculating the COOTensor output shape in sparse_concat"
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


def sparse_concat(sp_input, concat_dim):
    """
    concatenates the input SparseTensor(COO format) along the specified dimension.

    .. note:
        demo API now, and only supported CPU

    Args:
        sp_input (Union[list(COOTensor), tuple(COOTensor)) - the list of SparseTensor which need to concatenates.
            for COOTensor input.
        concat_dim (scalar): decide the dimension to concatenation along.
            The value must be in range [-rank, rank), where rank is the number of dimensions in each input
            SparseTensor.

    Outputs:
        - **output** (COOtensor) - the result of concatenates the input SparseTensor along the
            specified dimension.

    Raises:
        ValueError: If only one sparse tensor input.
        ValueError: If Input COOTensor shape dim > 3. COOtensor shape dim size must be 2 now

    Supported Platforms:
        ``CPU``

    Examples:
        >>> indics0 = Tensor([[0, 1], [1, 2]], dtype=mstype.int32)
        >>> values0 = Tensor([1, 2], dtype=mstype.int32)
        >>> shape0 = (3, 4)
        >>> input0 = COOTensor(indics0, values0, shape0)
        >>> indics1 = Tensor([[0, 0], [1, 1]], dtype=mstype.int32)
        >>> values1 = Tensor([3, 4], dtype=mstype.int32)
        >>> shape1 = (3, 4)
        >>> input1 = COOTensor(indics1, values1, shape1)
        >>> sparse_concat = ops.SparseConcat()
        >>> concat_dim = 1
        >>> out = sparse_concat((input0, input1), concat_dim)
        >>> print(out)
        shape = [3 4]
        [0 1]: "1"
        [0 4]: "3"
        [1 2]: "4"
        [1 5]: "2"
    """
    if len(sp_input) < 2:
        raise_value_error("For sparse_concat, not support COOTensor input number < 2.")
    sparse_concat_op = SparseConcat(concat_dim)
    indices = sp_input[0].indices
    values = sp_input[0].values
    sp_input_0_shape = sp_input[0].shape
    shape = Tensor(sp_input_0_shape)
    for i in range(1, len(sp_input)):
        in_indices = (indices, sp_input[i].indices)
        in_values = (values, sp_input[i].values)
        sp_input_i_shape = sp_input[i].shape
        in_shapes = (shape, Tensor(sp_input_i_shape))
        indices, values, shape = sparse_concat_op(in_indices, in_values, in_shapes)
    out_shape = _calc_out_shape(sp_input, concat_dim)
    return COOTensor(indices, values, out_shape)


def sparse_add(x1, x2, thresh):
    """
    sum the input SparseTensor(COO format).

    Args:
        x1 (COOTensor): the first SparseTensor to sum.
        x2 (COOTensor): the second SparseTensor to sum.
        thresh (Tensor): A 0-D Tensor, represents the magnitude threshold that determines
            if an output value/index pair pair take space.

    Returns:
        A COOTensor, the result of sum the input SparseTensor.

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
        TypeError: If (x1/x2)'s value's type is not match to thresh's type.

    Supported Platforms:
        ``CPU`` ``GPU``

    Examples:
        >>> indics0 = Tensor([[0, 1], [1, 2]], dtype=mstype.int64)
        >>> values0 = Tensor([1, 2], dtype=mstype.int32)
        >>> shape0 = (3, 4)
        >>> input0 = COOTensor(indics0, values0, shape0)
        >>> indics1 = Tensor([[0, 0], [1, 1]], dtype=mstype.int64)
        >>> values1 = Tensor([3, 4], dtype=mstype.int32)
        >>> shape1 = (3, 4)
        >>> input1 = COOTensor(indics1, values1, shape1)
        >>> thres = Tensor(0, dtype=mstype.int32)
        >>> out = F.sparse_add(input0, input1, thres)
        >>> print(out)
        COOTensor(shape = [3, 4], dtype = Int32, indices=Tensor(shape=[2,2],
        dtype = Int64, value=[[0 1], [1 2]]),  values=Tensor(shape[2],
        dtype=Int32, value=[4 6]))
    """
    x1_indices = x1.indices
    x1_values = x1.values
    x2_indices = x2.indices
    x2_values = x2.values
    den_shp = Tensor(x1.shape)
    add_op = SparseAdd()
    indices, values, _ = add_op(x1_indices, x1_values, den_shp, x2_indices, x2_values, den_shp, thresh)
    return COOTensor(indices, values, x1.shape)


__all__ = [
    'coalesce',
    'coo2csr',
    'coo_tensor_get_dense_shape',
    'coo_tensor_get_indices',
    'coo_tensor_get_values',
    'csr_div',
    'csr_gather',
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
    'row_tensor_get_values',
    'row_tensor_get_indices',
    'row_tensor_get_dense_shape',
    'row_tensor_add',
    'sparse_add',
    'sparse_concat',
]

__all__.sort()
