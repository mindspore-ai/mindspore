"""Define the grad rules of math related operations."""

from mindspore.ops import functional as F
from mindspore.ops._grad.grad_base import bprop_getters
from mindspore.ops.composite.multitype_ops.zeros_like_impl import zeros_like
from mindspore.ops.operations.sparse_ops import SparseTensorDenseAdd
from mindspore.ops.operations.sparse_ops import SparseMatrixTranspose


@bprop_getters.register(SparseTensorDenseAdd)
def get_bprop_sparse_tensor_dense_add(self):
    """Grad definition for `SparseTensorDenseAdd` operation."""
    def bprop(x1_indices, x1_values, x1_shape, x2, out, dout):
        return (zeros_like(x1_indices), F.gather_nd(dout, x1_indices), zeros_like(x1_shape), dout,)
    return bprop


@bprop_getters.register(SparseMatrixTranspose)
def get_bprop_sparse_matrix_transpose(self):
    """Grad definition for 'SparseMatrixTranspose' operation"""
    sparse_transpose = SparseMatrixTranspose(conjugate=self.conjugate)

    def bprop(x_dense_shape, x_batch_pointers, x_row_pointers, x_col_indices, x_values, out, dout):
        dx = sparse_transpose(dout[0], dout[1], dout[2], dout[3], dout[4])
        dx_all = (dx[0], dx[1], dx[2], dx[3], dx[4])
        return dx_all
    return bprop
