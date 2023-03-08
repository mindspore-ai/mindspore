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
from mindspore.ops.operations.sparse_ops import RaggedTensorToSparse
from mindspore.ops.operations.sparse_ops import CSRSparseMatrixToSparseTensor
from mindspore.ops.operations.sparse_ops import SparseReorder
from mindspore.ops.operations.sparse_ops import SparseTensorToCSRSparseMatrix
from mindspore.ops.operations.sparse_ops import SparseToDenseV2
from mindspore.ops.operations.sparse_ops import SparseSoftmax
from mindspore.ops.operations.sparse_ops import SparseDenseCwiseAdd
from mindspore.ops.operations.sparse_ops import SparseSegmentSum
from mindspore.ops.operations.sparse_ops import SparseSegmentSumWithNumSegments
from mindspore.ops.operations.sparse_ops import SparseSegmentSqrtN
from mindspore.ops.operations.sparse_ops import SparseSegmentSqrtNWithNumSegments
from mindspore.ops.operations.sparse_ops import SparseFillEmptyRows
from mindspore.ops.operations.sparse_ops import SparseSegmentMeanWithNumSegments
from mindspore.ops.operations.sparse_ops import SparseSlice
from mindspore.ops.operations.sparse_ops import SparseDenseCwiseMul
from mindspore.ops.operations.sparse_ops import SparseDenseCwiseDiv
from mindspore.ops.operations.sparse_ops import SparseTensorDenseAdd
from mindspore.ops.operations.sparse_ops import Sspaddmm
from mindspore.ops.operations._inner_ops import IsSubClass
import mindspore as ms
from mindspore.ops.operations import _map_tensor_ops
from mindspore.common import dtype as mstype
from mindspore import Tensor
from mindspore.ops.primitive import constexpr
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore.ops.composite.multitype_ops.zeros_like_impl import zeros_like
from mindspore.ops.operations import _grad_ops as G
from mindspore.ops._grad.grad_base import bprop_getters
from mindspore import context

# Unused parameters are placeholders.
dyn_shape_op = P.TensorShape()
is_sub_class = IsSubClass()


@bprop_getters.register(SparseDenseCwiseMul)
def get_bprop_sparse_dense_cwise_mul(self):
    """Grad definition for 'SparseDenseCwiseMul' operation"""

    slice_op = P.Slice()
    ones = P.Ones()
    shape = P.Shape()
    cast = P.Cast()
    gather_nd = P.GatherNd()
    size_op = P.Size()
    to_array = P.TupleToArray()
    sparse_tensor_dense_add = SparseTensorDenseAdd()

    def bprop(x1_indices, x1_values, x1_shape, x2, out, dout):
        x2_shape = cast(to_array(shape(x2)), mstype.int64)
        augmented_x2_shape = P.Concat(0)((ones(size_op(x1_shape) - size_op(x2_shape), mstype.int64), x2_shape))
        scaling = x1_shape // augmented_x2_shape
        scaled_indices = x1_indices // scaling
        scaled_indices = cast(slice_op(scaled_indices, [0, size_op(x1_shape) - size_op(x2_shape)], [-1, -1]),
                              mstype.int64)
        dense_vals = gather_nd(x2, scaled_indices)
        dx1 = dout * dense_vals
        dx2_val = dout * x1_values
        dx2 = sparse_tensor_dense_add(scaled_indices, dx2_val, x2_shape, zeros_like(x2))
        d_all = (zeros_like(x1_indices), dx1, zeros_like(x1_shape), dx2)
        return d_all

    return bprop


@bprop_getters.register(SparseDenseCwiseDiv)
def get_bprop_sparse_dense_cwise_div(self):
    """Grad definition for 'SparseDenseCwiseDiv' operation"""

    slice_op = P.Slice()
    ones = P.Ones()
    shape = P.Shape()
    cast = P.Cast()
    gather_nd = P.GatherNd()
    size_op = P.Size()
    to_array = P.TupleToArray()
    sparse_tensor_dense_add = SparseTensorDenseAdd()

    def bprop(x1_indices, x1_values, x1_shape, x2, out, dout):

        x2_shape = cast(to_array(shape(x2)), mstype.int64)
        augmented_x2_shape = P.Concat(0)((ones(size_op(x1_shape) - size_op(x2_shape), mstype.int64), x2_shape))
        scaling = x1_shape // augmented_x2_shape
        scaled_indices = x1_indices // scaling
        scaled_indices = cast(slice_op(scaled_indices, [0, size_op(x1_shape) - size_op(x2_shape)],
                                       [-1, -1]), mstype.int64)
        dense_vals = gather_nd(x2, scaled_indices)
        dx1 = dout / dense_vals
        dense_vals_2 = dense_vals * dense_vals
        w = x1_values / dense_vals_2
        w = (-1) * w
        dx2_val = dout * w
        dx2 = sparse_tensor_dense_add(scaled_indices, dx2_val, x2_shape,
                                      zeros_like(x2))
        d_all = (zeros_like(x1_indices), dx1, zeros_like(x1_shape), dx2)
        return d_all

    return bprop



@constexpr
def _create_tensor(data, dtype):
    return Tensor(data, dtype=dtype)


@bprop_getters.register(SparseSoftmax)
def get_bprop_sparse_softmax(self):
    """Generate bprop for SparseSoftmax"""
    sparse_to_dense = SparseToDenseV2()
    sparse_dense_cwise_add = SparseDenseCwiseAdd()
    reduce_sum = P.ReduceSum(keep_dims=True)
    mul = P.Mul()

    def bprop(indices, values, shape, out, dout):
        default_values = _create_tensor(0, values.dtype)
        out_dout = mul(out, dout)
        sp_product = sparse_to_dense(indices, shape, out_dout, default_values)
        sum_reduced = -1 * reduce_sum(sp_product, -1)
        sp_sum = sparse_dense_cwise_add(indices, dout, shape, sum_reduced)
        grad_x = mul(sp_sum, out)
        return zeros_like(indices), grad_x, zeros_like(shape)

    return bprop


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


@bprop_getters.register(SparseToDenseV2)
def get_bprop_sparse_to_dense_v2(self):
    """Generate bprop for SparseToDenseV2"""

    def bprop(indices, output_shape, values, default_value, out, dout):
        sparse_values_grad = F.gather_nd(dout, indices)
        default_value_grad = F.reduce_sum(dout) - F.reduce_sum(sparse_values_grad)
        result_all = (zeros_like(indices), zeros_like(output_shape), sparse_values_grad, default_value_grad)
        return result_all

    return bprop


@bprop_getters.register(SparseSegmentSqrtN)
def get_bprop_sparse_segment_sqrt_n(self):
    """Grad definition for `SparseSegmentSqrtN` operation."""
    input_grad = G.SparseSegmentSqrtNGrad()
    shape = P.Shape()

    def bprop(x, indices, segment_ids, out, dout):
        shape_x = shape(x)
        if F.is_sequence_value_unknown(shape_x):
            shape_x = dyn_shape_op(x)
        output_dim0 = P.Cast()(shape_x[0], mstype.int32)
        indices = F.cast(indices, mstype.int32)
        segment_ids = F.cast(segment_ids, mstype.int32)
        dx = input_grad(dout, indices, segment_ids, output_dim0)
        all_d = (dx, zeros_like(indices), zeros_like(segment_ids))
        return all_d

    return bprop


@bprop_getters.register(SparseSegmentSqrtNWithNumSegments)
def get_bprop_sparse_segment_sqrt_n_with_num_segments(self):
    """Grad definition for `SparseSegmentSqrtNWithNumSegments` operation."""
    input_grad = G.SparseSegmentSqrtNGrad()
    shape = P.Shape()

    def bprop(x, indices, segment_ids, num_segments, out, dout):
        shape_x = shape(x)
        if F.is_sequence_value_unknown(shape_x):
            shape_x = dyn_shape_op(x)
        output_dim0 = P.Cast()(shape_x[0], mstype.int32)
        indices = F.cast(indices, mstype.int32)
        segment_ids = F.cast(segment_ids, mstype.int32)
        dx = input_grad(dout, indices, segment_ids, output_dim0)
        all_d = (dx, zeros_like(indices), zeros_like(segment_ids), zeros_like(num_segments))
        return all_d

    return bprop


@bprop_getters.register(SparseSegmentSum)
def get_bprop_sparse_segment_sum(self):
    """Grad definition for `SparseSegmentSum` operation."""
    gather = P.Gather()
    unsorted_segment_sum = P.UnsortedSegmentSum()
    shape = P.Shape()

    def bprop_gpu(x, indices, segment_ids, out, dout):
        input_grad = G.SparseSegmentSumGrad()
        shape_x = shape(x)
        if F.is_sequence_value_unknown(shape_x):
            shape_x = dyn_shape_op(x)
        output_dim0 = P.Cast()(shape_x[0], mstype.int32)
        indices = F.cast(indices, mstype.int32)
        segment_ids = F.cast(segment_ids, mstype.int32)
        dx = input_grad(dout, indices, segment_ids, output_dim0)
        all_d = (dx, zeros_like(indices), zeros_like(segment_ids))
        return all_d

    def bprop(x, indices, segment_ids, out, dout):
        shape_x = dyn_shape_op(x)
        output_dim0 = P.Cast()(shape_x[0], mstype.int32)
        segment_ids = F.cast(segment_ids, mstype.int32)
        input0 = gather(dout, segment_ids, 0)
        input0 = F.cast(input0, mstype.float32)
        indices = F.cast(indices, mstype.int32)
        dx = unsorted_segment_sum(input0, indices, output_dim0)
        dx = F.cast(dx, F.dtype(dout))
        return dx, zeros_like(indices), zeros_like(segment_ids)

    if context.get_context('device_target') == "GPU":
        return bprop_gpu

    return bprop


@bprop_getters.register(SparseFillEmptyRows)
def get_bprop_sparsefillemptyrows(self):
    """Grad definition for `SparseFillEmptyRows` operation."""
    op = G.SparseFillEmptyRowsGrad()

    def bprop(indices, values, dense_shape, default_value, out, dout):
        dx = op(out[3], dout[1])
        dx_all = (zeros_like(indices), dx[0], zeros_like(dense_shape), dx[1])
        return dx_all
    return bprop


@bprop_getters.register(SparseSegmentSumWithNumSegments)
def get_bprop_sparse_segment_sum_with_num_segments(self):
    """Grad definition for `SparseSegmentSumWithNumSegments` operation."""
    gather = P.Gather()
    unsorted_segment_sum = P.UnsortedSegmentSum()
    shape = P.Shape()

    def bprop_gpu(x, indices, segment_ids, num_segments, out, dout):
        input_grad = G.SparseSegmentSumGrad()
        shape_x = shape(x)
        if F.is_sequence_value_unknown(shape_x):
            shape_x = dyn_shape_op(x)
        output_dim0 = P.Cast()(shape_x[0], mstype.int32)
        indices = F.cast(indices, mstype.int32)
        segment_ids = F.cast(segment_ids, mstype.int32)
        dx = input_grad(dout, indices, segment_ids, output_dim0)
        all_d = (dx, zeros_like(indices), zeros_like(segment_ids), zeros_like(num_segments))
        return all_d

    def bprop(x, indices, segment_ids, num_segments, out, dout):
        shape_x = dyn_shape_op(x)
        output_dim0 = P.Cast()(shape_x[0], mstype.int32)
        segment_ids = F.cast(segment_ids, mstype.int32)
        input0 = gather(dout, segment_ids, 0)
        input0 = F.cast(input0, mstype.float32)
        indices = F.cast(indices, mstype.int32)
        dx = unsorted_segment_sum(input0, indices, output_dim0)
        dx = F.cast(dx, F.dtype(dout))
        all_d = (dx, zeros_like(indices), zeros_like(segment_ids), zeros_like(num_segments))
        return all_d

    if context.get_context('device_target') == "GPU":
        return bprop_gpu

    return bprop


@bprop_getters.register(SparseSegmentMeanWithNumSegments)
def get_bprop_sparse_segment_mean_with_num_segments(self):
    """Grad definition for `SparseSegmentMeanWithNumSegments` operation."""
    input_grad = G.SparseSegmentMeanGrad()
    shape = P.Shape()

    def bprop(x, indices, segment_ids, num_segments, out, dout):
        x_shp = shape(x)
        if F.is_sequence_value_unknown(x_shp):
            x_shp = dyn_shape_op(x)
            output_dim0 = F.cast(x_shp[0], mstype.int32)
        else:
            output_dim0 = F.scalar_to_tensor(x_shp[0], mstype.int32)
        indices = F.cast(indices, mstype.int32)
        segment_ids = F.cast(segment_ids, mstype.int32)
        dx = input_grad(dout, indices, segment_ids, output_dim0)
        all_d = (dx, zeros_like(indices), zeros_like(segment_ids), zeros_like(num_segments))
        return all_d

    return bprop


@bprop_getters.register(SparseSlice)
def get_bprop_sparse_slice(self):
    """Grad definition for `SparseSlice` operation."""
    sparse_slice_grad = G.SparseSliceGrad()

    def bprop(indices, values, shape, start, size, out, dout):
        grad_op = sparse_slice_grad(dout[1], indices, start, out[0])
        result_all = (zeros_like(indices), grad_op, zeros_like(shape), zeros_like(start), zeros_like(size))
        return result_all

    return bprop


@bprop_getters.register(Sspaddmm)
def get_bprop_sspaddmm(self):
    """Grad definition for `Sspaddmm` operation."""
    def bprop(x1_indices, x1_values, x1_shape, x2_indices, x2_values, x2_shape, x3_dense, alpha, beta, out, dout):
        dx_all = (zeros_like(x1_indices), zeros_like(x1_values), zeros_like(x1_shape),
                  zeros_like(x2_indices), zeros_like(x2_values), zeros_like(x2_shape),
                  zeros_like(x3_dense), zeros_like(alpha), zeros_like(beta))
        return dx_all

    return bprop


@bprop_getters.register(RaggedTensorToSparse)
def get_bprop_ragged_tensor_to_sparse(self):
    """Grad definition for `RaggedTensorToSparse` operation."""
    shape = P.Shape()
    reshape = P.Reshape()

    def bprop(rt_nested_splits, rt_dense_values, out, dout):
        ragged_values_shape = shape(rt_dense_values)
        ragged_values_grad = reshape(dout[1], ragged_values_shape)

        if is_sub_class(F.typeof(rt_nested_splits), ms.list_):
            split = []
            for i in enumerate(rt_nested_splits):
                split.append(zeros_like(i))
            all_d = (split, ragged_values_grad)
            return all_d
        split = ()
        for i in enumerate(rt_nested_splits):
            split = split + (zeros_like(i),)
        all_d = (split, ragged_values_grad)
        return all_d

    return bprop


@bprop_getters.register(SparseReorder)
def get_bprop_sparse_reorder(self):
    """Grad definition for `SparseReorder` operation."""
    sparse_reorder_op = SparseReorder()
    range_op = P.Range()
    gather_op = P.Gather()

    def bprop(indices, values, shape, out, dout):
        num_entries = F.shape(indices)[0]
        start = Tensor(0, dtype=mstype.int32)
        limit = P.Cast()(num_entries, mstype.int32)
        delta = Tensor(1, dtype=mstype.int32)
        entry_indices = range_op(start, limit, delta)
        output = sparse_reorder_op(indices, entry_indices, shape)
        inverted_permutation = F.sort(output[1].astype(mstype.float32))[1]
        axis = 0
        return zeros_like(indices), gather_op(dout[1], inverted_permutation, axis), zeros_like(shape)

    return bprop


@bprop_getters.register(_map_tensor_ops.MapTensorGet)
def get_bprop_map_tensor_get(self):
    """Grad definition for `MapTensorGet` operation."""
    grad_op = G.MapTensorGetGrad()

    def bprop(map_tensor, key_tensor, out, dout):
        grad_map_tensor = grad_op(map_tensor, key_tensor, dout)
        return grad_map_tensor, zeros_like(key_tensor)
    return bprop
