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

"""array_ops"""

from .. import operations as P
from ..operations import _grad_ops as G
from ..operations import _inner_ops as inner
from ..composite.multitype_ops.zeros_like_impl import zeros_like
from ..functional import broadcast_gradient_args
from .. import functional as F
from .grad_base import bprop_getters
from ..primitive import constexpr
from ... import context
from ...common import dtype as mstype

reduce_sum = P.ReduceSum()
unsorted_segment_sum = P.UnsortedSegmentSum()
transpose = P.Transpose()
shape_op = P.Shape()
reshape = P.Reshape()
size_op = P.Size()
invert_permutation = P.InvertPermutation()
logical_and = P.LogicalAnd()


@bprop_getters.register(P.Fill)
def get_bprop_fill(self):
    """Generate bprop for Fill"""

    def bprop(dtype, dims, x, out, dout):
        return zeros_like(dims), zeros_like(x)

    return bprop


@bprop_getters.register(P.DType)
def get_bprop_dtype(self):
    """Generate bprop for DType"""

    def bprop(x, out, dout):
        return (zeros_like(x),)

    return bprop


@bprop_getters.register(P.Cast)
def get_bprop_cast(self):
    """Generate bprop for Cast"""
    cast = P.Cast()
    get_dtype = P.DType()

    def bprop(x, t, out, dout):
        dx = cast(dout, get_dtype(x))
        return dx, zeros_like(t)

    return bprop


@bprop_getters.register(P.Shape)
def get_bprop_shape(self):
    """Generate bprop for Shape"""

    def bprop(x, out, dout):
        return (zeros_like(x),)

    return bprop


@bprop_getters.register(P.Split)
def get_bprop_split(self):
    """Generate bprop for Split"""
    axis = self.axis

    def bprop(x, out, dout):
        concat_op = P.Concat(axis)
        dx = concat_op(dout)
        return (dx,)

    return bprop


@bprop_getters.register(P.Rank)
def get_bprop_rank(self):
    """Generate bprop for Rank"""

    def bprop(x, out, dout):
        return (zeros_like(x),)

    return bprop


@bprop_getters.register(P.Reshape)
def get_bprop_reshape(self):
    """Generate bprop for Reshape"""

    def bprop(x, shp, out, dout):
        shapex = shape_op(x)
        return reshape(dout, shapex), zeros_like(shp)

    return bprop


@bprop_getters.register(P.ExpandDims)
def get_bprop_expand_dims(self):
    """Generate bprop for ExpandDims"""

    def bprop(x, axis, out, dout):
        shapex = shape_op(x)
        return reshape(dout, shapex), zeros_like(axis)

    return bprop


@bprop_getters.register(P.Squeeze)
def get_bprop_squeeze(self):
    """Generate bprop for Squeeze"""

    def bprop(x, out, dout):
        shapex = shape_op(x)
        return (reshape(dout, shapex),)

    return bprop


@bprop_getters.register(P.Flatten)
def get_bprop_flatten(self):
    """Generate bprop for Flatten"""
    flatten_grad = G.FlattenGrad()

    def bprop(x, out, dout):
        dx = flatten_grad(dout, shape_op(x))
        return (dx,)

    return bprop


@constexpr
def _tile_shape(multiples, shapex):
    """Calculate [1,2], [3, 4] -> [1,3,2,4]."""
    len_muli = len(multiples)
    rank = len(shapex)
    len_cmp = len_muli - rank
    max_len = max(len_muli, rank)
    i = 0
    j = 0
    ret = []
    while (i < max_len) and (j < max_len):
        if len_cmp == 0:
            ret.append(multiples[i])
            ret.append(shapex[j])
            i += 1
            j += 1
        elif len_cmp > 0:
            ret.append(multiples[i])
            ret.append(1)
            i += 1
            len_cmp -= 1
        else:
            ret.append(1)
            ret.append(shapex[j])
            len_cmp += 1
    return tuple(ret)


@bprop_getters.register(P.Tile)
def get_bprop_tile(self):
    """Generate bprop for Tile"""

    def bprop(x, multiples, out, dout):
        shapex = shape_op(x)
        r_shape = _tile_shape(multiples, shapex)
        # 0 represents the start index, and 2 represents the step
        axis = F.make_range(0, len(r_shape), 2)
        dx = reduce_sum(reshape(dout, r_shape), axis)
        dx = reshape(dx, shapex)
        return dx, zeros_like(multiples)

    return bprop


@bprop_getters.register(inner.EmbeddingLookup)
def get_bprop_embedding_lookup(self):
    """Generate bprop for EmbeddingLookup"""
    host_sub = P.Sub().add_prim_attr('primitive_target', 'CPU')
    host_reshape = P.Reshape().add_prim_attr('primitive_target', 'CPU')
    def bprop_sparse(x, indices, offset, reduce_scatter_flag, split_num, out, dout):
        x_shp = shape_op(x)
        if reduce_scatter_flag is True:
            elu_grad = G.EmbeddingLookupCommGrad()
            actual_dout = elu_grad(dout, split_num)
        else:
            actual_dout = dout
        new_indices = host_sub(indices, offset)
        # Reshape the 'new_indices'
        new_indices_shape_changed = (size_op(new_indices),)
        new_indices = host_reshape(new_indices, new_indices_shape_changed)
        # Reshape the 'actual_dout'
        x_shp_tail = x_shp[1:]
        actual_dout_shape_changed = new_indices_shape_changed + x_shp_tail
        actual_dout = host_reshape(actual_dout, actual_dout_shape_changed)
        return (new_indices, actual_dout, x_shp), zeros_like(indices), zeros_like(offset), \
               zeros_like(reduce_scatter_flag), zeros_like(split_num)
    return bprop_sparse


@bprop_getters.register(P.Transpose)
def get_bprop_transpose(self):
    """Generate bprop for Transpose"""

    def bprop(x, perm, out, dout):
        return transpose(dout, invert_permutation(perm)), zeros_like(perm)

    return bprop


@bprop_getters.register(P.Concat)
def get_bprop_concat(self):
    """Generate bprop for Concat"""
    axis = self.axis

    def bprop(x, out, dout):
        dx = ()
        out_offset = G.ConcatOffset(F.tuple_len(x), axis)(x)
        for i in range(F.tuple_len(x)):
            slice_out = P.Slice()(dout, out_offset[i], shape_op(x[i]))
            dx = dx + (slice_out,)
        return (dx,)

    return bprop


@constexpr
def _slice_grad_pad(begins, sizes, shapes):
    pads = tuple((begin, shape - begin - size) for begin, size, shape in zip(begins, sizes, shapes))
    return pads


@bprop_getters.register(P.Slice)
def get_bprop_slice(self):
    """Generate bprop for Slice"""

    def bprop(x, begin, size, out, dout):
        dx = P.Pad(_slice_grad_pad(begin, size, shape_op(x)))(dout)
        return (dx, zeros_like(begin), zeros_like(size))

    def bprop_grad(x, begin, size, out, dout):
        dx = dx = G.SliceGrad()(dout, x, begin, size)
        return (dx, zeros_like(begin), zeros_like(size))

    if context.get_context('device_target') == "GPU" or context.get_context('device_target') == "CPU":
        return bprop_grad
    return bprop


@constexpr
def _generate_shape_index(out_shape, indices_shape, axis):
    out_rank = len(out_shape)
    ind_rank = len(indices_shape)
    if axis < 0:
        axis += out_rank - ind_rank + 1
    perm_part1 = tuple(range(axis, axis + ind_rank))
    index = tuple(range(out_rank))
    perm = perm_part1 + index[:axis] + index[axis + ind_rank:]
    return perm


@constexpr
def _generate_inverse_index(x_shape, axis):
    x_rank = len(x_shape)
    index = tuple(range(x_rank))
    if axis < 0:
        axis += x_rank
    perm = index[1:1 + axis] + (0,) + index[1 + axis:]
    return perm


@bprop_getters.register(P.GatherV2)
def get_bprop_gather_v2(self):
    """Generate bprop for GatherV2"""

    def bprop(x, indices, axis, out, dout):
        if F.rank(dout) == 0:
            dout = P.ExpandDims()(dout, -1)
        if F.rank(indices) == 0:
            indices = P.ExpandDims()(indices, -1)
        x_shp = shape_op(x)
        out_shp = shape_op(dout)
        ind_shp = shape_op(indices)
        # Example: out_shape:(3,2,3) axis 1 -> (1,0,2)
        perm_1 = _generate_shape_index(out_shp, ind_shp, axis)
        values_transpose = transpose(dout, perm_1)
        params_grad = unsorted_segment_sum(values_transpose, indices, shape_op(x)[axis])
        # Example: out_shape:(3,2,3) axis 2 -> (1,2,0)
        perm_2 = _generate_inverse_index(x_shp, axis)
        params_grad = transpose(params_grad, perm_2)
        return params_grad, zeros_like(indices), zeros_like(axis)

    return bprop


@bprop_getters.register(P.SparseGatherV2)
def get_bprop_sparse_gather_v2(self):
    """Generate bprop for SparseGatherV2"""

    def bprop(x, indices, axis, out, dout):
        x_shp = shape_op(x)
        if axis == 0:
            indices_size = (size_op(indices),)
            x_tail_shp = x_shp[1:]
            values_shape = indices_size + x_tail_shp
            values = reshape(dout, values_shape)
            indices = reshape(indices, indices_size)
            return (indices, values, x_shp), zeros_like(indices), zeros_like(axis)
        if F.rank(dout) == 0:
            dout = P.ExpandDims()(dout, -1)
        if F.rank(indices) == 0:
            indices = P.ExpandDims()(indices, -1)
        out_shp = shape_op(dout)
        ind_shp = shape_op(indices)
        # Example: out_shape:(3,2,3) axis 1 -> (1,0,2)
        perm_1 = _generate_shape_index(out_shp, ind_shp, axis)
        values_transpose = transpose(dout, perm_1)
        params_grad = unsorted_segment_sum(values_transpose, indices, shape_op(x)[axis])
        # Example: out_shape:(3,2,3) axis 2 -> (1,2,0)
        perm_2 = _generate_inverse_index(x_shp, axis)
        params_grad = transpose(params_grad, perm_2)
        return params_grad, zeros_like(indices), zeros_like(axis)

    return bprop


@bprop_getters.register(inner.Range)
def get_bprop_range(self):
    """Generate bprop for Range"""

    def bprop(x, out, dout):
        return (zeros_like(x),)
    return bprop


@bprop_getters.register(P.Pack)
def get_bprop_pack(self):
    """Generate bprop for Pack"""
    axis = self.axis

    def bprop(x, out, dout):
        pack_grad = P.Unpack(axis)
        out = pack_grad(dout)
        return (out,)

    return bprop


@bprop_getters.register(P.Unpack)
def get_bprop_unpack(self):
    """Generate bprop for Unpack"""
    axis = self.axis

    def bprop(x, out, dout):
        unpack_grad = P.Pack(axis)
        out = unpack_grad(dout)
        return (out,)

    return bprop


@bprop_getters.register(P.StridedSlice)
def get_bprop_strided_slice(self):
    """Generate bprop for StridedSlice"""
    input_grad = G.StridedSliceGrad(self.begin_mask,
                                    self.end_mask,
                                    self.ellipsis_mask,
                                    self.new_axis_mask,
                                    self.shrink_axis_mask)

    def bprop(x, begin, end, strides, out, dout):
        dx = input_grad(dout, shape_op(x), begin, end, strides)
        return dx, zeros_like(begin), zeros_like(end), zeros_like(strides)

    return bprop


@bprop_getters.register(P.Eye)
def get_bprop_eye(self):
    """Generate bprop for Eye"""

    def bprop(n, m, t, out, dout):
        return zeros_like(n), zeros_like(m), zeros_like(t)

    return bprop


@bprop_getters.register(P.Select)
def get_bprop_select(self):
    """Generate bprop for Select"""
    select = P.Select()

    def bprop(cond, x, y, out, dout):
        return zeros_like(cond), select(cond, dout, zeros_like(x)), select(cond, zeros_like(y), dout)

    return bprop


@bprop_getters.register(P.OnesLike)
def get_bprop_oneslike(self):
    """Generate bprop for OnesLike"""

    def bprop(x, out, dout):
        return (zeros_like(x),)

    return bprop


@bprop_getters.register(P.ZerosLike)
def get_bprop_zeroslike(self):
    """Generate bprop for OnesLike"""

    def bprop(x, out, dout):
        return (zeros_like(x),)

    return bprop


@bprop_getters.register(P.ResizeNearestNeighbor)
def get_bprop_resize_nearest_neighbor(self):
    """Generate bprop for ResizeNearestNeighbor"""
    op = G.ResizeNearestNeighborGrad(self.align_corners)

    def bprop(inputs, out, dout):
        shp = shape_op(inputs)
        # 2 and 3 represent the height and width
        shp = (shp[2], shp[3])
        return (op(dout, shp),)

    return bprop


@bprop_getters.register(P.GatherNd)
def get_bprop_gather_nd(self):
    """Generate bprop for GatherNd"""
    op = P.ScatterNd()

    def bprop(x, indices, out, dout):
        shp = shape_op(x)
        return op(indices, dout, shp), zeros_like(indices)

    return bprop


@bprop_getters.register(P.ScatterNd)
def get_bprop_scatter_nd(self):
    """Generate bprop for ScatterNd"""
    op = P.GatherNd()

    def bprop(indices, x, shape, out, dout):
        return zeros_like(indices), op(dout, indices), zeros_like(shape)

    return bprop


@bprop_getters.register(P.ScatterNdUpdate)
def get_bprop_scatter_nd_update(self):
    """Generate bprop for ScatterNdUpdate"""
    op = P.GatherNd()

    def bprop(x, indices, update, out, dout):
        return dout, zeros_like(indices), op(dout, indices)

    return bprop


@bprop_getters.register(P.TensorScatterUpdate)
def get_bprop_tensor_scatter_update(self):
    """Generate bprop for TensorScatterUpdate"""
    gather_nd = P.GatherNd()
    tensor_scatter_update = P.TensorScatterUpdate()

    def bprop(x, indices, update, out, dout):
        x_grad = tensor_scatter_update(dout, indices, zeros_like(update))
        update_grad = gather_nd(dout, indices)
        return x_grad, zeros_like(indices), update_grad

    return bprop


@bprop_getters.register(P.ScatterMax)
def get_bprop_scatter_max(self):
    """Generate bprop for ScatterMax"""
    gather = P.GatherV2()

    def bprop(x, indices, update, out, dout):
        return dout, zeros_like(indices), gather(dout, indices, 0)

    return bprop


@bprop_getters.register(P.Argmax)
def get_bprop_argmax(self):
    """Generate bprop for Argmax"""

    def bprop(x, out, dout):
        return (zeros_like(x),)

    return bprop


@bprop_getters.register(P.Argmin)
def get_bprop_argmin(self):
    """Generate bprop for Argmin"""

    def bprop(x, out, dout):
        return (zeros_like(x),)

    return bprop


@bprop_getters.register(P.SpaceToDepth)
def get_bprop_space_to_depth(self):
    """Generate bprop for SpaceToDepth"""
    op = P.DepthToSpace(self.block_size)

    def bprop(x, out, dout):
        return (op(dout),)

    return bprop


@bprop_getters.register(P.DepthToSpace)
def get_bprop_depth_to_space(self):
    """Generate bprop for DepthToSpace"""
    op = P.SpaceToDepth(self.block_size)

    def bprop(x, out, dout):
        return (op(dout),)

    return bprop


@bprop_getters.register(P.Diag)
def get_bprop_diag(self):
    """Generate bprop for Diag"""
    op = P.DiagPart()

    def bprop(x, out, dout):
        return (op(dout),)

    return bprop


@bprop_getters.register(P.DiagPart)
def get_bprop_diag_part(self):
    """Generate bprop for DiagPart"""
    op = P.Diag()

    def bprop(x, out, dout):
        return (op(dout),)

    return bprop


def _GatherDropNegatives(params,
                         ids,
                         zero_clipped_indices=None,
                         is_positive=None):
    """Helper function for unsorted segment ops."""
    maximum = P.Maximum()
    gather = P.GatherV2()
    greater_equal = P.GreaterEqual()
    rank = P.Rank()
    fill = P.Fill()
    select = P.Select()

    if zero_clipped_indices is None:
        zero_clipped_indices = maximum(ids, zeros_like(ids))
    gathered = gather(params, zero_clipped_indices, 0)
    if is_positive is None:
        is_positive = greater_equal(ids, 0)
        is_positive_shape = shape_op(is_positive)
        broadcastable_shape = is_positive_shape
        for _ in range(rank(gathered) - rank(is_positive)):
            broadcastable_shape += (1,)
        is_positive = reshape(is_positive, broadcastable_shape)
        gathered_shape = shape_op(gathered)
        is_positive = logical_and(is_positive, fill(mstype.bool_, gathered_shape, 1))
    zero_slice = zeros_like(gathered)
    return (select(is_positive, gathered, zero_slice), zero_clipped_indices, is_positive)


@bprop_getters.register(P.UnsortedSegmentMin)
def get_bprop_unsorted_segment_min(self):
    """Generate bprop for UnsortedSegmentMin"""
    equal = P.Equal()
    cast = P.Cast()
    divide = P.RealDiv()
    get_dtype = P.DType()
    select = P.Select()

    def bprop(x, segment_ids, num_segments, out, dout):
        gathered_outputs, zero_clipped_indices, is_positive = _GatherDropNegatives(out, segment_ids)
        is_selected = equal(x, gathered_outputs)
        is_selected = logical_and(is_selected, is_positive)
        num_selected = unsorted_segment_sum(cast(is_selected, get_dtype(dout)),
                                            segment_ids, num_segments)
        weighted_grads = divide(dout, num_selected)
        gathered_grads, _, _ = _GatherDropNegatives(weighted_grads, None,
                                                    zero_clipped_indices, is_positive)
        zeros = zeros_like(gathered_grads)
        return select(is_selected, gathered_grads, zeros), zeros_like(segment_ids), zeros_like(num_segments)
    return bprop


@bprop_getters.register(P.SpaceToBatch)
def get_bprop_space_to_batch(self):
    """Generate bprop for SpaceToBatch"""
    space_to_batch_grad = P.BatchToSpace(self.block_size, self.paddings)

    def bprop(x, out, dout):
        dx = space_to_batch_grad(dout)
        return (dx,)

    return bprop


@bprop_getters.register(P.BatchToSpace)
def get_bprop_batch_to_space(self):
    """Generate bprop for BatchToSpace"""
    batch_to_space_grad = P.SpaceToBatch(self.block_size, self.crops)

    def bprop(x, out, dout):
        dx = batch_to_space_grad(dout)
        return (dx,)

    return bprop


@bprop_getters.register(P.SpaceToBatchND)
def get_bprop_space_to_batch_nd(self):
    """Generate bprop for SpaceToBatchND"""
    space_to_batch_nd_grad = P.BatchToSpaceND(self.block_shape, self.paddings)
    def bprop(x, out, dout):
        dx = space_to_batch_nd_grad(dout)
        return (dx,)
    return bprop


@bprop_getters.register(P.BatchToSpaceND)
def get_bprop_batch_to_space_nd(self):
    """Generate bprop for BatchToSpaceND"""
    batch_to_space_nd_grad = P.SpaceToBatchND(self.block_shape, self.crops)
    def bprop(x, out, dout):
        dx = batch_to_space_nd_grad(dout)
        return (dx,)
    return bprop

@bprop_getters.register(P.BroadcastTo)
def get_bprop_broadcast_to(self):
    """Generate bprop for BroadcastTo"""
    reduce_keep_dim = P.ReduceSum(keep_dims=True)
    broadcast_shape = self.shape

    def bprop(x, out, dout):
        x_shape = shape_op(x)
        _, reduction_axes = broadcast_gradient_args(broadcast_shape, x_shape)
        reduced_grad = reduce_keep_dim(dout, reduction_axes)
        dx = reshape(reduced_grad, x_shape)
        return (dx,)
    return bprop
