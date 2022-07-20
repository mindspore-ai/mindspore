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

"""bprop primitives"""
from mindspore.ops.operations.sparse_ops import CSRSparseMatrixToSparseTensor
from mindspore.ops.operations.sparse_ops import SparseTensorToCSRSparseMatrix
from .._grad.grad_base import bprop_getters


# Unused parameters are placeholders.

@bprop_getters.register(SparseTensorToCSRSparseMatrix)
def get_bprop_sparse_tensor_to_csr_sparse_matrix(self):
    """Grad definition for 'SparseTensorToCSRSparseMatrix' operation"""
    op = CSRSparseMatrixToSparseTensor()

    def bprop(x_indices, x_values, x_dense_shape, out, dout):
        dx = op(dout[0], dout[1], dout[2], dout[3], dout[4])
        dx_all = (dx[0], dx[1], dx[2])
        return dx_all

    return bprop


@bprop_getters.register(CSRSparseMatrixToSparseTensor)
def get_bprop_csr_sparse_matrix_to_sparse_tensor(self):
    """Grad definition for 'CSRSparseMatrixToSparseTensor' operation"""
    op = SparseTensorToCSRSparseMatrix()

    def bprop(x_dense_shape, x_batch_pointers, x_row_pointers, x_col_indices, x_values, out, dout):
        dx = op(dout[0], dout[1], dout[2])
        dx_all = (dx[0], dx[1], dx[2], dx[3], dx[4])
        return dx_all

    return bprop
