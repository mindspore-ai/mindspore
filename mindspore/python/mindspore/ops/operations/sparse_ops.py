# coding: utf-8

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

"""Operators for sparse operators."""

from ..._checkparam import Validator as validator
from ...common import dtype as mstype
from ..primitive import PrimitiveWithInfer, Primitive, prim_attr_register


class SparseToDense(PrimitiveWithInfer):
    """
    Converts a sparse representation into a dense tensor.

    Inputs:
        - **indices** (Tensor) - A 2-D Tensor, represents the position of the element in the sparse tensor.
          Support int32, int64, each element value should be a non-negative int number. The shape is :math:`(n, 2)`.
        - **values** (Tensor) - A 1-D Tensor, represents the value corresponding to the position in the `indices`.
          The shape should be :math:`(n,)`.
        - **sparse_shape** (tuple(int)) - A positive int tuple which specifies the shape of sparse tensor,
          should have 2 elements, represent sparse tensor shape is :math:`(N, C)`.

    Returns:
        Tensor, converted from sparse tensor. The dtype is same as `values`, and the shape is `sparse_shape`.

    Raises:
        TypeError: If the dtype of `indices` is neither int32 nor int64.
        ValueError: If `sparse_shape`, shape of `indices` and shape of `values` don't meet the parameter description.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> indices = Tensor([[0, 1], [1, 2]])
        >>> values = Tensor([1, 2], dtype=ms.float32)
        >>> sparse_shape = (3, 4)
        >>> sparse_to_dense = ops.SparseToDense()
        >>> out = sparse_to_dense(indices, values, sparse_shape)
        >>> print(out)
        [[0. 1. 0. 0.]
         [0. 0. 2. 0.]
         [0. 0. 0. 0.]]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize SparseToDense."""
        self.init_prim_io_names(inputs=['indices', 'values', 'dense_shape'], outputs=['output'])

    def __infer__(self, indices, values, sparse_shape):
        validator.check_tensor_dtype_valid('indices', indices['dtype'], [mstype.int32, mstype.int64], self.name)
        validator.check_tensor_dtype_valid('values', values['dtype'], mstype.number_type + (mstype.bool_,), self.name)
        indices_shape = indices['shape']
        if len(indices_shape) != 2:
            raise ValueError(f"For '{self.name}', the 'indices' must be a 2-D tensor, "
                             f"but got 'indices' shape: {indices_shape}.")
        values_shape = values['shape']
        if len(values_shape) != 1 or values_shape[0] != indices_shape[0]:
            raise ValueError(f"For '{self.name}', the 'values' must be a 1-D tensor and the first dimension length "
                             f"must be equal to the first dimension length of 'indices', "
                             f"but got 'indices' shape: {indices_shape}, 'values' shape: {values_shape}.")
        sparse_shape_v = sparse_shape['value']
        for i in sparse_shape_v:
            if isinstance(i, bool) or not isinstance(i, int) or i <= 0:
                raise ValueError(f"For '{self.name}', all elements in 'sparse_shape' must be "
                                 f"positive int number, but got 'sparse_shape': {sparse_shape_v}.")
        if len(sparse_shape_v) != indices_shape[1]:
            raise ValueError(f"For '{self.name}', the length of 'sparse_shape' must be equal to the second dimension "
                             f"length of 'indices', but got the second dimension length of 'indices': "
                             f"{indices_shape[1]}, length of 'sparse_shape': {len(sparse_shape_v)}.")
        out = {'shape': sparse_shape['value'],
               'dtype': values['dtype'],
               'value': None}
        return out


class SparseTensorDenseMatmul(PrimitiveWithInfer):
    """
    Multiplies sparse matrix `A` by dense matrix `B`.
    The rank of sparse matrix and dense matrix must be equal to `2`.

    Args:
        adjoint_st (bool): If true, sparse tensor is transposed before multiplication. Default: False.
        adjoint_dt (bool): If true, dense tensor is transposed before multiplication. Default: False.

    Inputs:
        - **indices** (Tensor) - A 2-D Tensor, represents the position of the element in the sparse tensor.
          Support int32, int64, each element value should be a non-negative int number. The shape is :math:`(n, 2)`.
        - **values** (Tensor) - A 1-D Tensor, represents the value corresponding to the position in the `indices`.
          Support float16, float32, float64, int32, int64. The shape should be :math:`(n,)`.
        - **sparse_shape** (tuple(int)) - A positive int tuple which specifies the shape of sparse tensor,
          should have 2 elements, represent sparse tensor shape is :math:`(N, C)`.
        - **dense** (Tensor) - A 2-D Tensor, the dtype is same as `values`.
          If `adjoint_st` is False and `adjoint_dt` is False, the shape must be :math:`(C, M)`.
          If `adjoint_st` is False and `adjoint_dt` is True, the shape must be :math:`(M, C)`.
          If `adjoint_st` is True and `adjoint_dt` is False, the shape must be :math:`(N, M)`.
          If `adjoint_st` is True and `adjoint_dt` is True, the shape must be :math:`(M, N)`.

    Outputs:
        Tensor, the dtype is the same as `values`.
        If `adjoint_st` is False, the shape is :math:`(N, M)`.
        If `adjoint_st` is True, the shape is :math:`(C, M)`.

    Raises:
        TypeError: If the type of `adjoint_st` or `adjoint_dt` is not bool, or the dtype of `indices`,
            dtype of `values` and dtype of `dense` don't meet the parameter description.
        ValueError: If `sparse_shape`, shape of `indices`, shape of `values`,
            and shape of `dense` don't meet the parameter description.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> indices = Tensor([[0, 1], [1, 2]], dtype=ms.int32)
        >>> values = Tensor([1, 2], dtype=ms.float32)
        >>> sparse_shape = (3, 4)
        >>> dense = Tensor([[1,1], [2,2], [3,3 ], [4, 4]], dtype=ms.float32)
        >>> sparse_dense_matmul = ops.SparseTensorDenseMatmul()
        >>> out = sparse_dense_matmul(indices, values, sparse_shape, dense)
        >>> print(out)
        [[2. 2.]
         [6. 6.]
         [0. 0.]]
    """

    @prim_attr_register
    def __init__(self, adjoint_st=False, adjoint_dt=False):
        """Initialize SparseTensorDenseMatmul"""
        self.adjoint_st = adjoint_st
        self.adjoint_dt = adjoint_dt
        self.init_prim_io_names(inputs=['indices', 'values', 'sparse_shape', 'dense'],
                                outputs=['output'])
        self.add_prim_attr('adjoint_st', self.adjoint_st)
        self.add_prim_attr('adjoint_dt', self.adjoint_dt)
        validator.check_value_type("adjoint_st", adjoint_st, [bool], self.name)
        validator.check_value_type("adjoint_dt", adjoint_dt, [bool], self.name)

    def __infer__(self, indices, values, sparse_shape, dense):
        validator.check_tensor_dtype_valid('indices', indices['dtype'], [mstype.int32, mstype.int64], self.name)
        valid_types = (mstype.float16, mstype.float32, mstype.float64, mstype.int32, mstype.int64)
        args = {'values': values['dtype'], 'dense': dense['dtype']}
        validator.check_tensors_dtypes_same_and_valid(args, valid_types, self.name)
        indices_shape = indices['shape']
        if len(indices_shape) != 2 or indices_shape[1] != 2:
            raise ValueError(f"For '{self.name}', the 'indices' must be a 2-D tensor and "
                             f"the second dimension length must be 2, but got 'indices' shape: {indices_shape}.")
        values_shape = values['shape']
        if len(values_shape) != 1 or values_shape[0] != indices_shape[0]:
            raise ValueError(f"For '{self.name}', the 'values' must be a 1-D tensor and "
                             f"the first dimension length must be equal to the first dimension length of 'indices', "
                             f"but got 'indices' shape: {indices_shape}, 'values' shape: {values_shape}.")
        a_shape = sparse_shape['value'][::-1] if self.adjoint_st else sparse_shape['value']
        b_shape = dense['shape'][::-1] if self.adjoint_dt else dense['shape']
        for i in a_shape:
            if isinstance(i, bool) or not isinstance(i, int) or i <= 0:
                raise ValueError(f"For '{self.name}', all elements in 'sparse_shape' must be "
                                 f"positive int number, but got 'sparse_shape': {a_shape}.")
        if len(a_shape) != 2 or len(b_shape) != 2:
            raise ValueError(f"For '{self.name}', both the length of 'sparse_shape' and the tensor "
                             f"rank of 'dense' must be equal to 2, but got the length of "
                             f"'sparse_shape': {len(a_shape)}, "
                             f"the tensor rank of 'dense': {len(b_shape)}.")
        if a_shape[1] != b_shape[0]:
            raise ValueError(f"For '{self.name}', the second dimension length of 'sparse_shape' must be equal to the "
                             f"first dimension length of 'dense', but got "
                             f"the tensor shape of 'sparse': {a_shape} and the tensor shape of 'dense': {b_shape}. "
                             f"Don't meet the condition for matmul")
        out_shape = [a_shape[0], b_shape[1]]
        out = {'shape': tuple(out_shape),
               'dtype': values['dtype'],
               'value': None}
        return out


class DenseToCSRSparseMatrix(Primitive):
    """
    Converts a dense matrix(maybe batched) to its CSR sparse form.

    .. warning::
        This is an experimental prototype that is subject to change and/or deletion.

    Inputs:
        - **dense_input** (Tensor) - A 2-D or 3-D Tensor. It represents the input dense matrix.
        - **indices** (Tensor) - A 2-D Tensor. It represents indices of all the nonzero elements.

    Outputs:
        - **y_dense_shape** (Tensor) - A 1-D Tensor. It represents the dense form shape of
          the output CSR sparse matrix, the shape of which should be :math:`(2,)` or :math:`(3,)`.
        - **y_batch_pointers** (Tensor) - A 1-D Tensor. Supposing the output CSR sparse matrix is of
          batch size `n`, it should have shape :math:`(n+1,)`, while the `i`-th element of which stores
          acummulated counts of nonzero values of the first `i - 1` batches.
        - **y_row_pointers** (Tensor) - A 1-D Tensor. Supposing the output CSR sparse matrix is of
          batch size `n` and row number `m`, it can be divided into `n` parts, each part of length
          `m + 1`. The `i`-th element of each :math:`(m+1,)` vector stores acummulated counts of
          nonzero values of the first `i - 1` rows in the corresponding batch.
        - **y_col_indices** (Tensor) - A 1-D Tensor. It represents column indices of the nonzero values
          in the output CSR sparse matrix.
        - **y_values** (Tensor) - A 1-D Tensor. It represents all the nonzero values in the
          output CSR sparse matrix.

    Raises:
        TypeError: If the dtype of `indices` is not int32 or int64.
        TypeError: If the dtype of `dense_input` is not float32, float64, complex64 or complex128.
        ValueError: If either of the inputs is not a tensor.
        ValueError: If rank of `dense_input` is not 2 or 3.
        ValueError: If rank of `indices` is not 2.
        ValueError: If shape[1] of `indices` and rank of `dense_input` is not the same.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> x = Tensor([[[1., 0.], [0., 2.]]], dtype=ms.float32)
        >>> indices = Tensor([[0, 0, 0], [0, 1, 1]], dtype=ms.int32)
        >>> dense_to_csr = ops.DenseToCSRSparseMatrix()
        >>> out = dense_to_csr(x, indices)
        >>> print(out[0])
        [1 2 2]
        >>> print(out[1])
        [0 2]
        >>> print(out[2])
        [0 1 2]
        >>> print(out[3])
        [0 1]
        >>> print(out[4])
        [1. 2.]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize DenseToCSRSparseMatrix"""
        self.init_prim_io_names(
            inputs=['dense_input', 'indices'],
            outputs=['y_dense_shape', 'y_batch_pointers', 'y_row_pointers', 'y_col_indices', 'y_values'])
