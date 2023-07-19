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
"""Sparse related tools."""
from __future__ import absolute_import

from mindspore import log as logger
from mindspore.ops import operations as P
from mindspore.nn.cell import Cell


class SparseToDense(Cell):
    """
    Converts a sparse tensor(COOTensor) into dense.

    In Python, for the ease of use, three tensors are collected into a SparseTensor class.
    MindSpore uses three independent dense tensors: indices, value and dense shape to represent the sparse tensor.
    Separate indexes, values and dense shape tensors can be wrapped in a Sparse Tensor object
    before :class:`mindspore.ops.SparseToDense` is called.

    Note:
        'nn.SparseToDense' is deprecated from version 2.0, and will be removed in a future version, please use
        COOTensor.to_dense() instead.

    Inputs:
        - **coo_tensor** (:class:`mindspore.COOTensor`) - the sparse COOTensor to convert.

    Outputs:
        Tensor, converted from sparse tensor.

    Raises:
        TypeError: If `sparse_tensor.indices` is not a Tensor.
        TypeError: If `sparse_tensor.values` is not a Tensor.
        TypeError: If `sparse_tensor.shape` is not a tuple.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> import mindspore as ms
        >>> from mindspore import Tensor, COOTensor
        >>> import mindspore.nn as nn
        >>> ms.set_context(mode=ms.PYNATIVE_MODE)
        >>> indices = Tensor([[0, 1], [1, 2]])
        >>> values = Tensor([1, 2], dtype=ms.int32)
        >>> dense_shape = (3, 4)
        >>> class Net(nn.Cell):
        ...     def __init__(self, dense_shape):
        ...         super(Net, self).__init__()
        ...         self.dense_shape = dense_shape
        ...         self.op = nn.SparseToDense()
        ...
        ...     def construct(self, indices, values):
        ...         x = COOTensor(indices, values, self.dense_shape)
        ...         return self.op(x)
        ...
        >>> print(Net(dense_shape)(indices, values))
        [[0 1 0 0]
         [0 0 2 0]
         [0 0 0 0]]
    """

    def __init__(self):
        """Initialize SparseToDense."""
        logger.warning("'nn.SparseToDense' is deprecated from version 2.0 and will be removed in a future version. " +
                       "Please use 'COOTensor.to_dense()' instead.")
        super(SparseToDense, self).__init__()
        self.sparse_to_dense = P.SparseToDense()

    def construct(self, sparse_tensor):
        return self.sparse_to_dense(sparse_tensor.indices,
                                    sparse_tensor.values,
                                    sparse_tensor.shape)


class SparseTensorDenseMatmul(Cell):
    """
    Multiplies sparse matrix `a` and dense matrix `b`.
    The rank of sparse matrix and dense matrix must be equal to `2`.

    Args:
        adjoint_st (bool): If ``true`` , sparse tensor is transposed before multiplication. Default: ``False`` .
        adjoint_dt (bool): If ``true`` , dense tensor is transposed before multiplication. Default: ``False`` .

    Inputs:
        - **indices** (Tensor) - A 2-D Tensor, represents the position of the element in the sparse tensor.
          Support int32, int64, each element value should be non-negative. The shape is :math:`(n, 2)`.
        - **values** (Tensor) - A 1-D Tensor, represents the value corresponding to the position in the `indices`.
          Support float16, float32, float64, int32, int64. The shape should be :math:`(n,)`.
        - **sparse_shape** (tuple) - A positive int tuple which specifies the shape of sparse tensor,
          should have 2 elements, represent sparse tensor shape is :math:`(N, C)`.
        - **dense** (Tensor) - A 2-D Tensor, the dtype is same as `values`.
          If `adjoint_st` is ``False`` and `adjoint_dt` is ``False`` , the shape must be :math:`(C, M)`.
          If `adjoint_st` is ``False`` and `adjoint_dt` is ``True`` , the shape must be :math:`(M, C)`.
          If `adjoint_st` is ``True`` and `adjoint_dt` is ``False`` , the shape must be :math:`(N, M)`.
          If `adjoint_st` is ``True`` and `adjoint_dt` is ``True`` , the shape must be :math:`(M, N)`.

    Outputs:
        Tensor, the dtype is the same as `values`.
        If `adjoint_st` is ``False`` , the shape is :math:`(N, M)`.
        If `adjoint_st` is ``True`` , the shape is :math:`(C, M)`.

    Raises:
        TypeError: If the type of `adjoint_st` or `adjoint_dt` is not bool, or the dtype of `indices`,
            dtype of `values` and dtype of `dense` don't meet the parameter description.
        ValueError: If `sparse_shape`, shape of `indices`, shape of `values`,
            and shape of `dense` don't meet the parameter description.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> import mindspore as ms
        >>> from mindspore import Tensor
        >>> from mindspore import nn
        >>> indices = Tensor([[0, 1], [1, 2]], dtype=ms.int32)
        >>> values = Tensor([1, 2], dtype=ms.float32)
        >>> sparse_shape = (3, 4)
        >>> dense = Tensor([[1, 1], [2, 2], [3, 3], [4, 4]], dtype=ms.float32)
        >>> sparse_dense_matmul = nn.SparseTensorDenseMatmul()
        >>> out = sparse_dense_matmul(indices, values, sparse_shape, dense)
        >>> print(out)
        [[2. 2.]
         [6. 6.]
         [0. 0.]]
    """

    def __init__(self, adjoint_st=False, adjoint_dt=False):
        """Initialize SparseTensorDenseMatmul"""
        super(SparseTensorDenseMatmul, self).__init__()
        self.adj_st = adjoint_st
        self.adj_dt = adjoint_dt
        self.sparse_dense_matmul = P.SparseTensorDenseMatmul(adjoint_st=self.adj_st, adjoint_dt=self.adj_dt)

    def construct(self, indices, values, sparse_shape, dense):
        return self.sparse_dense_matmul(indices, values, sparse_shape, dense)
