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
from ..primitive import PrimitiveWithInfer, prim_attr_register


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
        ValueError: If `sparse_shape`, shape of `indices and shape of `values` don't meet the parameter description.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> indices = Tensor([[0, 1], [1, 2]])
        >>> values = Tensor([1, 2], dtype=ms.float32)
        >>> sparse_shape = (3, 4)
        >>> sparse_to_dense = ops.SparseToDense()
        >>> out = sparse_to_dense(indices, values, sparse_shape)
        >>> print(out)
        [[0 1 0 0]
         [0 0 2 0]
         [0 0 0 0]]
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
            raise ValueError("SparseToDense requires 'indices' must be a 2-D Tensor, "
                             f"but got 'indices' shape: {indices_shape}")
        values_shape = values['shape']
        if len(values_shape) != 1 or values_shape[0] != indices_shape[0]:
            raise ValueError("SparseToDense requires 'values' must be a 1-D Tensor and "
                             "the first dimension length must be equal to the first dimension length of 'indices', "
                             f"but got 'indices' shape: {indices_shape}, 'values' shape: {values_shape}")
        sparse_shape_v = sparse_shape['value']
        for i in sparse_shape_v:
            if isinstance(i, bool) or not isinstance(i, int) or i <= 0:
                raise ValueError("SparseToDense requires all elements in 'sparse_shape' must be "
                                 f"positive int number, but got 'sparse_shape': {sparse_shape_v}")
        if len(sparse_shape_v) != indices_shape[1]:
            raise ValueError("SparseToDense requires the 'sparse_shape' length should be equal to the 'indices' "
                             "second dimension length, but got the 'indices' second dimension length: "
                             f"{indices_shape[1]}, 'sparse_shape' length: {len(sparse_shape_v)}")
        out = {'shape': sparse_shape['value'],
               'dtype': values['dtype'],
               'value': None}
        return out


class SparseTensorDenseMatmul(PrimitiveWithInfer):
    """
    Multiplies sparse matrix `A` by dense matrix `B`.
    The rank of sparse matrix and dense matrix must equal to `2`.

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
        ValueError: If `sparse_shape`, shape of `indices, shape of `values`,
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
        [[2 2]
         [6 6]
         [0 0]]
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
            raise ValueError("SparseTensorDenseMatmul requires 'indices' must be a 2-D Tensor and "
                             f"the second dimension length must be 2, but got 'indices' shape: {indices_shape}")
        values_shape = values['shape']
        if len(values_shape) != 1 or values_shape[0] != indices_shape[0]:
            raise ValueError("SparseTensorDenseMatmul requires 'value's must be a 1-D Tensor and "
                             f"the first dimension length must be equal to the first dimension length of 'indices', "
                             f"but got 'indices' shape: {indices_shape}, 'values' shape: {values_shape}")
        a_shape = sparse_shape['value'][::-1] if self.adjoint_st else sparse_shape['value']
        b_shape = dense['shape'][::-1] if self.adjoint_dt else dense['shape']
        for i in a_shape:
            if isinstance(i, bool) or not isinstance(i, int) or i <= 0:
                raise ValueError("SparseTensorDenseMatmul requires all elements in 'sparse_shape' must be "
                                 f"positive int number, but got sparse shape: {a_shape}")
        if len(a_shape) != 2 or len(b_shape) != 2:
            raise ValueError("SparseTensorDenseMatmul requires both the 'sparse_shape' length and the dense tensor "
                             f"rank should be equal to 2, but got 'sparse_shape' length: {len(a_shape)}, "
                             f"dense tensor rank: {len(b_shape)}")
        if a_shape[1] != b_shape[0]:
            raise ValueError(f"The sparse tensor shape: {a_shape} and the dense tensor shape: {b_shape} "
                             f"don't meet the condition for matmul")
        out_shape = [a_shape[0], b_shape[1]]
        out = {'shape': tuple(out_shape),
               'dtype': values['dtype'],
               'value': None}
        return out
