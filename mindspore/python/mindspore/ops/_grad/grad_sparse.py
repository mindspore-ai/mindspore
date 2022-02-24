# Copyright 2020 Huawei Technologies Co., Ltd
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

"""bprop primitives"""
from ...common import dtype as mstype
from .. import functional as F
from .. import operations as P
from ..operations import _csr_ops
from ..composite.multitype_ops.zeros_like_impl import zeros_like
from .grad_base import bprops, bprop_getters


# Unused parameters are placeholders.


@bprops.register("MakeCOOTensor")
def bprop_make_coo_tensor(indices, values, dense_shape, out, dout):
    """Backpropagator for primitive `MakeCOOTensor`."""
    return zeros_like(indices), F.coo_tensor_get_values(dout), ()


@bprops.register("COOTensorGetIndices")
def bprop_sparse_tensor_get_indices(sparse_tensor, out, dout):
    """Backpropagator for primitive `COOTensorGetIndices`."""
    return (zeros_like(sparse_tensor),)


@bprops.register("COOTensorGetValues")
def bprop_sparse_tensor_get_values(sparse_tensor, out, dout):
    """Backpropagator for primitive `COOTensorGetValues`."""
    return F.make_coo_tensor(F.coo_tensor_get_indices(sparse_tensor),
                             dout,
                             F.coo_tensor_get_dense_shape(sparse_tensor))


@bprops.register("COOTensorrGetDenseShape")
def bprop_sparse_tensor_get_dense_shape(sparse_tensor, out, dout):
    """Backpropagator for primitive `COOTensorGetDenseShape`."""
    return (zeros_like(sparse_tensor),)


@bprop_getters.register(P.SparseToDense)
def get_bprop_sparse_to_dense(self):
    """Generate bprop for SparseToDense"""

    def bprop(indices, values, dense_shape, out, dout):
        return zeros_like(indices), F.gather_nd(dout, indices), zeros_like(dense_shape)

    return bprop


@bprop_getters.register(P.SparseTensorDenseMatmul)
def get_bprop_sparse_tensor_dense_matmul(self):
    """Generate bprop for SparseTensorDenseMatmul"""
    adj_s = self.adjoint_st
    adj_d = self.adjoint_dt
    sparse_tensor_dense_mat_mul = P.SparseTensorDenseMatmul(not adj_s)

    def bprop(indices, values, dense_shape, dense, out, dout):
        dense_grad = sparse_tensor_dense_mat_mul(indices, values, dense_shape, dout)
        perm = (1, 0)
        if adj_d:
            dense_grad = F.transpose(dense_grad, perm)
        rows = indices[:, 0]
        cols = indices[:, 1]
        parts_a = F.gather(dout, cols if adj_s else rows, 0)
        parts_b = F.gather(F.transpose(dense, perm) if adj_d else dense, rows if adj_s else cols, 0)
        values_grad = F.reduce_sum(parts_a * parts_b, 1)
        return zeros_like(indices), values_grad, zeros_like(dense_shape), dense_grad
    return bprop

@bprop_getters.register(_csr_ops.CSRReduceSum)
def get_bprop_csr_reduce_sum(self):
    "Back-propagation for CSRReduceSum."
    def bprop(csr_tensor, axis, out, dout):
        indptr = csr_tensor.indptr
        indices = csr_tensor.indices
        shape = csr_tensor.shape

        output_shape_kept_dims = F.reduced_shape(shape, axis)
        tile_scaling = F.tuple_div(shape, output_shape_kept_dims)
        values_grad_dense = F.tile(F.reshape(dout, output_shape_kept_dims), tile_scaling)
        values_grad = F.csr_gather(indptr, indices, values_grad_dense, shape)
        return F.make_csr_tensor(indptr, indices, values_grad, shape), zeros_like(axis)
    return bprop

@bprop_getters.register(_csr_ops.CSRMV)
def get_bprop_csr_mv(self):
    "Back-propagation for CSRMV."
    def bprop(csr_tensor, dense, out, dout):
        indptr = F.csr_tensor_get_indptr(csr_tensor)
        indices = F.csr_tensor_get_indices(csr_tensor)
        values = F.csr_tensor_get_values(csr_tensor)
        dense_shape = csr_tensor.shape

        rows = F.csr2coo(indptr, indices.shape[0])
        idx_dtype = rows.dtype
        rows_transposed, cols_indexing = F.sort(indices.astype(mstype.float32))
        rows_transposed = rows_transposed.astype(idx_dtype)
        cols_transposed = rows[cols_indexing]
        values_transposed = values[cols_indexing]
        indptr_transposed = F.coo2csr(rows_transposed, dense_shape[1])
        csr_tensor_transposed = F.make_csr_tensor(
            indptr_transposed, cols_transposed, values_transposed, (dense_shape[1], dense_shape[0]))

        dense_grad = F.csr_mv(csr_tensor_transposed, dout)
        parts_a = F.gather(dout, rows, 0)
        parts_b = F.gather(dense, indices, 0)
        values_grad = F.reduce_sum(parts_a * parts_b, 1)
        return F.make_csr_tensor(indptr, indices, values_grad, csr_tensor.shape), dense_grad
    return bprop

@bprop_getters.register(_csr_ops.CSRMul)
def get_bprop_csr_mul(self):
    "Back-propagation for CSRMul."
    def bprop(csr_tensor, dense, out, dout):
        indptr = csr_tensor.indptr
        indices = csr_tensor.indices
        values = csr_tensor.values
        shape = csr_tensor.shape

        csr_tensor_grad_value = F.csr_mul(F.make_csr_tensor(indptr, indices, dout, shape), dense)
        csr_tensor_grad = F.make_csr_tensor(indptr, indices, csr_tensor_grad_value, shape)
        dense_grad_value = F.mul(dout, values)
        dense_grad = F.make_csr_tensor(indptr, indices, dense_grad_value, shape)
        if len(dense.shape) == 1 or dense.shape[0] == 1:
            dense_grad = F.csr_reduce_sum(dense_grad, 0)
        elif dense.shape[1] == 1:
            dense_grad = F.csr_reduce_sum(dense_grad, 1)
        else:
            row = F.csr2coo(indptr, indices.shape[0])
            coo_idx = P.Stack(-1)((row, indices))
            dense_grad = F.tensor_scatter_update(zeros_like(dense), coo_idx, dense_grad_value)
        return csr_tensor_grad, dense_grad
    return bprop

@bprop_getters.register(_csr_ops.CSR2COO)
def get_bprop_csr2coo(self):
    def bprop(indptr, nnz, out, dout):
        return zeros_like(dout)
    return bprop

@bprop_getters.register(_csr_ops.COO2CSR)
def get_bprop_coo2csr(self):
    def bprop(row_indices, height, out, dout):
        return zeros_like(dout)
    return bprop
