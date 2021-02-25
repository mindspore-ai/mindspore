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

"""array_ops"""

import numpy as np
import mindspore as ms
from mindspore.ops import composite as C
from mindspore.common.tensor import Tensor
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
from ...common.tensor import RowTensor

reduce_sum = P.ReduceSum()
unsorted_segment_sum = P.UnsortedSegmentSum()
transpose = P.Transpose()
shape_op = P.Shape()
dyn_shape_op = P.DynamicShape()
reshape = P.Reshape()
size_op = P.Size()
invert_permutation = P.InvertPermutation()
logical_and = P.LogicalAnd()
is_sub_class = P.IsSubClass()


@bprop_getters.register(P.Fill)
def get_bprop_fill(self):
    """Generate bprop for Fill"""

    def bprop(dtype, dims, x, out, dout):
        return zeros_like(dims), zeros_like(x)

    return bprop


@bprop_getters.register(P.Ones)
def get_bprop_ones(self):
    """Generate bprop for Ones"""

    def bprop(dims, dtype, out, dout):
        return zeros_like(dims)

    return bprop


@bprop_getters.register(P.Zeros)
def get_bprop_zeros(self):
    """Generate bprop for Zeros"""

    def bprop(dims, dtype, out, dout):
        return zeros_like(dims)

    return bprop


@bprop_getters.register(P.DType)
def get_bprop_dtype(self):
    """Generate bprop for DType"""

    def bprop(x, out, dout):
        return (zeros_like(x),)

    return bprop


dout_cast = C.MultitypeFuncGraph("dout_cast")


@dout_cast.register("Tensor", "Tensor")
def dout_cast_tensor(dout, x):
    cast = P.Cast()
    get_dtype = P.DType()
    dx = cast(dout, get_dtype(x))
    return dx


@dout_cast.register("Number", "Number")
def dout_cast_number(dout, x):
    cast = P.Cast()
    get_dtype = P.DType()
    dx = cast(dout, get_dtype(x))
    return dx


@dout_cast.register("RowTensor", "Tensor")
def dout_cast_row_tensor(dout, x):
    cast = P.Cast()
    get_dtype = P.DType()
    values = cast(dout.values, get_dtype(x))
    return RowTensor(dout.indices, values, dout.dense_shape)


@bprop_getters.register(P.Cast)
def get_bprop_cast(self):
    """Generate bprop for Cast"""
    cast = P.Cast()
    get_dtype = P.DType()

    def bprop(x, t, out, dout):
        dx = cast(dout, get_dtype(x))
        return dx, zeros_like(t)

    def bprop_sparse(x, t, out, dout):
        dx = dout_cast(dout, x)
        return dx, zeros_like(t)

    if context.get_context('enable_sparse'):
        return bprop_sparse

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


@bprop_getters.register(P.EmbeddingLookup)
def get_bprop_embedding_lookup(self):
    """Generate bprop for EmbeddingLookup"""
    sub_op = P.Sub()
    reshape_op = P.Reshape()

    def bprop_sparse(x, indices, offset, out, dout):
        x_shp = shape_op(x)
        new_indices = sub_op(indices, offset)
        # Reshape the 'new_indices'
        new_indices_shape_changed = (size_op(new_indices),)
        new_indices = reshape_op(new_indices, new_indices_shape_changed)
        x_shp_tail = x_shp[1:]
        actual_dout_shape_changed = new_indices_shape_changed + x_shp_tail
        # Reshape the 'actual_dout' on device
        actual_dout = reshape_op(dout, actual_dout_shape_changed)
        return RowTensor(new_indices, actual_dout, x_shp), zeros_like(indices), zeros_like(offset)

    return bprop_sparse


@constexpr
def make_begin(shp):
    begin = tuple([0 for _ in shp])
    return begin


@bprop_getters.register(P.Padding)
def get_bprop_padding(self):
    """Grad definition for `Padding` operation."""

    def bprop(x, out, dout):
        shp = shape_op(x)
        begin = make_begin(shp)
        dx = P.Slice()(dout, begin, shp)
        return (dx,)

    return bprop


@bprop_getters.register(P.Transpose)
def get_bprop_transpose(self):
    """Generate bprop for Transpose"""

    def bprop(x, perm, out, dout):
        return transpose(dout, invert_permutation(perm)), zeros_like(perm)

    return bprop


@constexpr
def _concat_grad_uniform(input_shapes, input_nums):
    """Helper function for bprop of Concat"""
    is_uniform = True
    for i in range(1, input_nums):
        if input_shapes[i - 1] != input_shapes[i]:
            is_uniform = False
            break
    return is_uniform


@bprop_getters.register(P.Concat)
def get_bprop_concat(self):
    """Generate bprop for Concat"""
    axis = self.axis

    def bprop(x, out, dout):
        dx = ()
        out_offset = G.ConcatOffset(F.tuple_len(x), axis)(x)
        input_nums = F.tuple_len(x)
        input_shapes = ()
        for i in range(input_nums):
            input_shapes = input_shapes + (shape_op(x[i]),)
        is_uniform = _concat_grad_uniform(input_shapes, input_nums)
        if is_uniform:
            dx = P.Split(axis, input_nums)(dout)
        else:
            for i in range(input_nums):
                slice_out = P.Slice()(dout, out_offset[i], input_shapes[i])
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
        dx = G.SliceGrad()(dout, x, begin, size)
        return (dx, zeros_like(begin), zeros_like(size))

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


@constexpr
def _regenerate_output_shape(x_shp, ind_shp, axis):
    rank = len(x_shp)
    if axis < 0:
        axis += rank
    out_shape = x_shp[:axis] + ind_shp + x_shp[axis + 1:]
    return out_shape


@bprop_getters.register(P.Gather)
@bprop_getters.register(P.GatherV2)
def get_bprop_gather_v2(self):
    """Generate bprop for GatherV2"""

    def bprop(x, indices, axis, out, dout):
        orig_indices = indices
        if F.rank(dout) == 0:
            dout = P.ExpandDims()(dout, -1)
        if F.rank(indices) == 0:
            indices = P.ExpandDims()(indices, -1)
            x_shp = shape_op(x)
            ind_shp = shape_op(indices)
            out_shp = _regenerate_output_shape(x_shp, ind_shp, axis)
            dout = reshape(dout, out_shp)

        x_shp = shape_op(x)
        out_shp = shape_op(dout)
        ind_shp = shape_op(indices)
        # Example: out_shape:(3,2,3) axis 1 -> (1,0,2)
        perm_1 = _generate_shape_index(out_shp, ind_shp, axis)
        values_transpose = transpose(dout, perm_1)
        if -1 in shape_op(x):
            params_grad = unsorted_segment_sum(values_transpose, indices, dyn_shape_op(x)[axis])
        else:
            params_grad = unsorted_segment_sum(values_transpose, indices, shape_op(x)[axis])
        # Example: out_shape:(3,2,3) axis 2 -> (1,2,0)
        perm_2 = _generate_inverse_index(x_shp, axis)
        params_grad = transpose(params_grad, perm_2)
        return params_grad, zeros_like(orig_indices), zeros_like(axis)

    return bprop


@bprop_getters.register(P.GatherD)
def get_bprop_gather_d(self):
    """Generate bprop for GatherD"""

    def bprop(x, dim, index, out, dout):
        x_shp = shape_op(x)
        dx = G.GatherDGrad(dim, x_shp)(index, dout)
        return dx, zeros_like(dim), zeros_like(index)

    return bprop


@bprop_getters.register(P.SparseGatherV2)
def get_bprop_sparse_gather_v2(self):
    """Generate bprop for SparseGatherV2"""

    def bprop(x, indices, axis, out, dout):
        x_shp = shape_op(x)
        if axis == 0:
            indices_size = (size_op(indices),)
            if len(x_shp) <= 1:
                x_tail_shp = ()
            else:
                x_tail_shp = x_shp[1:]
            values_shape = indices_size + x_tail_shp
            values = reshape(dout, values_shape)
            indices_new = reshape(indices, indices_size)
            return RowTensor(indices_new, values, x_shp), zeros_like(indices), zeros_like(axis)
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


@constexpr
def _range_op(start, limit, delta, dtype):
    """helper function for grad of Sort"""
    output_tensor = Tensor(list(range(start, limit, delta)), dtype)
    return output_tensor


@constexpr
def _get_1d_shape(in_shape):
    """helper function for grad of Sort"""
    out_shape = 1
    for i in in_shape:
        out_shape *= i
    return (out_shape,)


@constexpr
def _get_transposition(axis, rank):
    """helper function for grad of Sort"""
    if axis < 0:
        axis += rank
    transposition = np.r_[np.arange(axis), [rank - 1], np.arange(axis + 1, rank - 1), [axis]]
    trans = tuple(transposition.tolist())
    return trans


@bprop_getters.register(P.Sort)
def get_bprop_sort(self):
    """Grad definition for `Sort` operation."""
    axis = self.axis
    descending = self.descending
    scatter = P.ScatterNd()
    expand_dims = P.ExpandDims()
    reshape_op = P.Reshape()
    dtype = P.DType()
    topk = P.TopK()
    neg = P.Neg()
    tranpose = P.Transpose()

    def bprop(input_x, out, dout):
        x_shape = input_x.shape
        k = x_shape[axis]
        rank = F.rank(input_x)
        dvalue = dout[0]
        if not descending:
            input_x = neg(input_x)
            dvalue = neg(dvalue)
        if axis == -1 or (axis + 1) == rank:
            transposition = None
            top_k_input = input_x
        else:
            transposition = _get_transposition(axis, rank)
            top_k_input = tranpose(input_x, transposition)

        _, indices = topk(top_k_input, k)
        ind_shape = indices.shape
        top_k_input_shape = top_k_input.shape
        in_lastdim = top_k_input_shape[-1]
        ind_lastdim = ind_shape[-1]
        ind_2d = reshape_op(indices, (-1, ind_lastdim))
        outer_dim = ind_2d.shape[0]

        # [0, outterdim, 2*outerdim, ..., (k-1)*outerdim]
        indices_dtype = dtype(indices)
        range_flatten_index = _range_op(0, outer_dim * in_lastdim, in_lastdim, indices_dtype)

        # expand_dims to (k, 1), then broadcast
        ind = reshape_op(ind_2d + expand_dims(range_flatten_index, -1), (-1,))
        x_shape_1d = _get_1d_shape(top_k_input_shape)

        if transposition is not None:
            dvalue = tranpose(dvalue, invert_permutation(transposition))
            out_grad = reshape_op(
                scatter(expand_dims(ind, -1), reshape_op(dvalue, (-1,)), x_shape_1d), top_k_input_shape)
            dx = tranpose(out_grad, invert_permutation(transposition))
        else:
            dx = reshape_op(scatter(expand_dims(ind, -1), reshape_op(dvalue, (-1,)), x_shape_1d), top_k_input_shape)
        if not descending:
            dx = neg(dx)
        return (dx,)

    return bprop


@bprop_getters.register(P.Identity)
def get_bprop_identity(self):
    """Generate bprop for Identity"""

    def bprop(x, out, dout):
        return (dout,)

    return bprop


@bprop_getters.register(inner.Range)
def get_bprop_range(self):
    """Generate bprop for Range"""

    def bprop(x, out, dout):
        return (zeros_like(x),)

    return bprop


@bprop_getters.register(P.Pack)
@bprop_getters.register(P.Stack)
def get_bprop_stack(self):
    """Generate bprop for Stack"""
    axis = self.axis

    def bprop(x, out, dout):
        stack_grad = P.Unstack(axis)
        out = stack_grad(dout)
        if is_sub_class(F.typeof(x), ms.list_):
            ret = []
            for item in out:
                ret.append(item)
            return (ret,)
        return (out,)

    return bprop


@bprop_getters.register(P.ReverseV2)
def get_bprop_reverse_v2(self):
    """Generate bprop for ReverseV2"""
    axis = self.axis

    def bprop(x, out, dout):
        reverse_grad = P.ReverseV2(axis)
        dx = reverse_grad(dout)
        return (dx,)

    return bprop


@bprop_getters.register(P.Unstack)
def get_bprop_unstack(self):
    """Generate bprop for Unstack"""
    axis = self.axis

    def bprop(x, out, dout):
        unstack_grad = P.Stack(axis)
        out = unstack_grad(dout)
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


@bprop_getters.register(P.ScatterNonAliasingAdd)
def get_bprop_scatter_non_aliasing_add_update(self):
    """Generate bprop for ScatterNonAliasingAdd"""
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
    gather = P.Gather()

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


def _gather_drop_negatives(params,
                           ids,
                           zero_clipped_indices=None,
                           is_positive=None):
    """Helper function for unsorted segment ops."""
    maximum = P.Maximum()
    gather = P.Gather()
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


def _unsorted_segment_min_or_max_grad(x, segment_ids, num_segments, out, dout):
    """Gradient for UnsortedSegmentMin or UnsortedSegmentMax"""
    equal = P.Equal()
    cast = P.Cast()
    divide = P.RealDiv()
    get_dtype = P.DType()
    select = P.Select()

    gathered_outputs, zero_clipped_indices, is_positive = _gather_drop_negatives(out, segment_ids, None, None)
    is_selected = equal(x, gathered_outputs)
    is_selected = logical_and(is_selected, is_positive)
    num_selected = unsorted_segment_sum(cast(is_selected, get_dtype(dout)),
                                        segment_ids, num_segments)
    weighted_grads = divide(dout, num_selected)
    gathered_grads, _, _ = _gather_drop_negatives(weighted_grads, None,
                                                  zero_clipped_indices, is_positive)
    zeros = zeros_like(gathered_grads)
    return select(is_selected, gathered_grads, zeros), zeros_like(segment_ids), zeros_like(num_segments)


@bprop_getters.register(P.UnsortedSegmentSum)
def get_bprop_unsorted_segment_sum(self):
    """Generate bprop for UnsortedSegmentSum"""

    def bprop(x, segment_ids, num_segments, out, dout):
        return _gather_drop_negatives(dout, segment_ids, None, None)[0], zeros_like(segment_ids), \
               zeros_like(num_segments)

    return bprop


@bprop_getters.register(P.UnsortedSegmentMin)
def get_bprop_unsorted_segment_min(self):
    """Generate bprop for UnsortedSegmentMin"""

    def bprop(x, segment_ids, num_segments, out, dout):
        return _unsorted_segment_min_or_max_grad(x, segment_ids, num_segments, out, dout)

    return bprop


@bprop_getters.register(P.UnsortedSegmentMax)
def get_bprop_unsorted_segment_max(self):
    """Generate bprop for UnsortedSegmentMax"""

    def bprop(x, segment_ids, num_segments, out, dout):
        return _unsorted_segment_min_or_max_grad(x, segment_ids, num_segments, out, dout)

    return bprop


@bprop_getters.register(P.UnsortedSegmentProd)
def get_bprop_unsorted_segment_prod(self):
    """Generate bprop for UnsortedSegmentProd"""
    equal = P.Equal()
    cast = P.Cast()
    select = P.Select()
    gather = P.Gather()
    greater = P.Greater()
    ones_like = P.OnesLike()
    maximum = P.Maximum()
    unsorted_segment_prod = P.UnsortedSegmentProd()

    def bprop(x, segment_ids, num_segments, out, dout):
        is_zero = equal(x, 0)
        num_zero = unsorted_segment_sum(cast(is_zero, mstype.int32), segment_ids, num_segments)
        grad = select(greater(num_zero, 1), zeros_like(dout), dout)
        non_zero_data = select(is_zero, ones_like(x), x)
        non_zero_prod = unsorted_segment_prod(non_zero_data, segment_ids, num_segments)
        zero_clipped_indices = maximum(segment_ids, zeros_like(segment_ids))
        gathered_prod = gather(out, zero_clipped_indices, 0)
        gathered_non_zero_prod = gather(non_zero_prod, zero_clipped_indices, 0)
        prod_divided_by_x = gathered_prod / x
        partial_derivative = select(is_zero, gathered_non_zero_prod, prod_divided_by_x)
        gathered_grad, _, _ = _gather_drop_negatives(grad, segment_ids, zero_clipped_indices, None)
        dx = gathered_grad * partial_derivative
        return dx, zeros_like(segment_ids), zeros_like(num_segments)

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
        dout_shape = shape_op(dout)

        if x_shape == dout_shape:
            return (dout,)
        _, reduction_axes = broadcast_gradient_args(broadcast_shape, x_shape)
        reduced_grad = reduce_keep_dim(dout, reduction_axes)
        dx = reshape(reduced_grad, x_shape)
        return (dx,)

    return bprop


@bprop_getters.register(P.ReverseSequence)
def get_bprop_reverse_sequence(self):
    """Generate bprop for ReverseSequence"""
    reverse_sequence_grad = P.ReverseSequence(batch_dim=self.batch_dim_, seq_dim=self.seq_dim_)

    def bprop(x, seq_lengths, out, dout):
        dx = reverse_sequence_grad(dout, seq_lengths)
        return dx, zeros_like(seq_lengths)

    return bprop


@bprop_getters.register(P.TransShape)
def get_bprop_trans_shape(self):
    """Generate bprop for TransShape"""
    op = P.TransShape()

    def bprop(x, shape, out, dout):
        dx = op(dout, shape_op(x))
        return (dx, zeros_like(shape))

    return bprop


@bprop_getters.register(P.Unique)
def get_bprop_unique(self):
    """Generate bprop for Unique"""
    op = G.UniqueGrad()

    def bprop(x, out, dout):
        dx = op(dout, out)
        return (dx,)

    return bprop
