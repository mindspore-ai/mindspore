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

from ..operations.sparse_ops import DenseToCSRSparseMatrix, CSRSparseMatrixToSparseTensor
from ..operations.array_ops import GatherNd
from ...common import CSRTensor, COOTensor, Tensor
from ...common import dtype as mstype
from ..composite.multitype_ops._constexpr_utils import raise_value_error, raise_type_error


gather_nd = GatherNd()
dense_to_csr = DenseToCSRSparseMatrix()
csr_sparse_matrix_to_sparse_tensor = CSRSparseMatrixToSparseTensor()
batch_csr_pointers_empty = Tensor([0, -1], dtype=mstype.int32)


def dense_to_sparse_coo(tensor):
    """
    Convert a Tensor to COOTensor.

    Note:
        Only 2-D tensor is supported for now.

    Args:
        tensor: A dense tensor, must be 2-D.

    Returns:
        COOTensor, a 2-D coo_tensor, containing:
        indices: the positions of all non-zero values of the input.
        values: the non-zero values of the dense tensor.
        shape: the shape of the coo_tensor, length is 2.

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
        CSRTensor, a 2-D csr_tensor, containing:
        indptr: indicates the start and end point for `values` in each row.
        indices: the column positions of all non-zero values of the input.
        values: the non-zero values of the dense tensor.
        shape: the shape of the csr_tensor, length is 2.

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

__all__ = [
    'dense_to_sparse_coo',
    'dense_to_sparse_csr',
    'csr_to_coo'
]

__all__.sort()
