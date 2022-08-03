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
from mindspore.ops.operations.sparse_ops import SparseSegmentSqrtN
from mindspore.ops.operations.sparse_ops import SparseSegmentSqrtNWithNumSegments
from mindspore.common import dtype as mstype
from .. import functional as F
from .. import operations as P
from ..composite.multitype_ops.zeros_like_impl import zeros_like
from ..operations import _grad_ops as G
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


@bprop_getters.register(SparseSegmentSqrtN)
def get_bprop_sparse_segment_sqrt_n(self):
    """Grad definition for `SparseSegmentSqrtN` operation."""
    input_grad = G.SparseSegmentSqrtNGrad()
    shape = P.Shape()

    def bprop(x, indices, segment_ids, out, dout):
        output_dim0 = F.scalar_to_tensor(shape(x)[0], mstype.int32)
        indices = F.cast(indices, mstype.int32)
        segment_ids = F.cast(segment_ids, mstype.int32)
        dx = input_grad(dout, indices, segment_ids, output_dim0)
        return dx, zeros_like(indices), zeros_like(segment_ids)

    return bprop


@bprop_getters.register(SparseSegmentSqrtNWithNumSegments)
def get_bprop_sparse_segment_sqrt_n_with_num_segments(self):
    """Grad definition for `SparseSegmentSqrtNWithNumSegments` operation."""
    input_grad = G.SparseSegmentSqrtNGrad()
    shape = P.Shape()

    def bprop(x, indices, segment_ids, num_segments, out, dout):
        output_dim0 = F.scalar_to_tensor(shape(x)[0], mstype.int32)
        indices = F.cast(indices, mstype.int32)
        segment_ids = F.cast(segment_ids, mstype.int32)
        dx = input_grad(dout, indices, segment_ids, output_dim0)
        all_d = (dx, zeros_like(indices), zeros_like(segment_ids), zeros_like(num_segments))
        return all_d

    return bprop
