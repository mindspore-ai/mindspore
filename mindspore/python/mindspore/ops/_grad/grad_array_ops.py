# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.ops.operations import _grad_ops as G
from mindspore.ops.operations.array_ops import Fills, NonZero
from mindspore.ops.composite.multitype_ops.zeros_like_impl import zeros_like
from mindspore.ops.functional import broadcast_gradient_args
from mindspore.ops import functional as F
from mindspore.ops._grad.grad_base import bprop_getters, create_tensor_by_element
from mindspore.ops.primitive import constexpr
from mindspore.ops.primitive import _primexpr
from mindspore.common import dtype as mstype
from mindspore.common.sparse_tensor import RowTensorInner
from mindspore.ops._utils.utils import range_op, get_1d_shape, generate_shape_index
from mindspore.ops._grad.grad_base import dyn_rank, convert_to_tensor, dyn_ones, dyn_fill
from mindspore.ops._grad.grad_base import sum_grad_reduce_axis
from mindspore.ops.operations._inner_ops import DynamicBroadcastGradientArgs
from ..operations._inner_ops import DynamicBroadcastGradientArgs, IsSubClass

reduce_sum = P.ReduceSum()
unsorted_segment_sum = P.UnsortedSegmentSum()
transpose = P.Transpose()
shape_op = P.Shape()
dyn_shape_op = P.TensorShape()
reshape = P.Reshape()
size_op = P.Size()
invert_permutation = P.InvertPermutation()
logical_and = P.LogicalAnd()
is_sub_class = IsSubClass()


@bprop_getters.register(P.Fill)
def get_bprop_fill(self):
    """Generate bprop for Fill"""

    def bprop(dtype, dims, x, out, dout):
        return zeros_like(dims), zeros_like(x)

    return bprop


@bprop_getters.register(Fills)
def get_bprop_fills(self):
    """Generate bprop for Fills."""

    def bprop(x, value, out, dout):
        return zeros_like(x), zeros_like(value)

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


@bprop_getters.register(P.Shape)
def get_bprop_shape(self):
    """Generate bprop for Shape"""

    def bprop(x, out, dout):
        return (zeros_like(x),)

    return bprop


@bprop_getters.register(P.DynamicShape)
def get_bprop_dynamicshape(self):
    """Generate bprop for DynamicShape"""

    def bprop(x, out, dout):
        return (zeros_like(x),)

    return bprop


@bprop_getters.register(P.TensorShape)
def get_bprop_tensorshape(self):
    """Generate bprop for TensorShape"""

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
        if F.is_sequence_value_unknown(shapex):
            shapex = dyn_shape_op(x)
        return reshape(dout, shapex), zeros_like(shp)

    return bprop


@bprop_getters.register(P.ExpandDims)
def get_bprop_expand_dims(self):
    """Generate bprop for ExpandDims"""

    def bprop(x, axis, out, dout):
        shapex = shape_op(x)
        if F.is_sequence_value_unknown(shapex):
            shapex = dyn_shape_op(x)
        return reshape(dout, shapex), zeros_like(axis)

    return bprop


@bprop_getters.register(P.Squeeze)
def get_bprop_squeeze(self):
    """Generate bprop for Squeeze"""

    def bprop(x, out, dout):
        shapex = shape_op(x)
        if F.is_sequence_value_unknown(shapex):
            shapex = dyn_shape_op(x)
        return (reshape(dout, shapex),)

    return bprop


@bprop_getters.register(P.Flatten)
def get_bprop_flatten(self):
    """Generate bprop for Flatten"""
    flatten_grad = P.Reshape()

    def bprop(x, out, dout):
        shape_x = shape_op(x)
        if F.is_sequence_value_unknown(shape_x):
            shape_x = dyn_shape_op(x)
        dx = flatten_grad(dout, shape_x)
        return (dx,)

    return bprop


@_primexpr
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
            j += 1
            len_cmp += 1
    return tuple(ret)


@bprop_getters.register(P.Tile)
def get_bprop_tile(self):
    """Generate bprop for Tile"""
    cast = P.Cast()
    concat = P.Concat()
    stridedslice = P.StridedSlice()

    def get_reduce_axis(r_shape):
        """
        reshape grad to r_shape, and reduce along all even dimensions to get the result with input_shape
        For example:
        input_shape = [20, 30, 40]
        multiples = [2, 3, 4]
        r_shape = [2, 20, 3, 30, 4, 40]
        axis = [0, 2, 4]
        """
        rankr = dyn_shape_op(r_shape)[0]
        tmp = range_op(0, 20, 2, mstype.int64)
        return stridedslice(tmp, (0,), F.expand_dims(rankr // 2, 0), (1,))

    def bprop(x, multiples, out, dout):
        shapex = shape_op(x)
        if F.is_sequence_value_unknown(shapex):
            shapex = dyn_shape_op(x)
        if isinstance(multiples, tuple) and isinstance(shapex, tuple):
            r_shape = _tile_shape(multiples, shapex)
            # 0 represents the start index, and 2 represents the step
            axis = F.make_range(0, len(r_shape), 2)
        else:
            shapex = dyn_shape_op(x)
            shapey = create_tensor_by_element(multiples)
            rankx = dyn_rank(x)
            ranky = dyn_shape_op(shapey)[0]
            offset = F.expand_dims(ranky - rankx + 1, 0)
            shape_x = concat((dyn_ones(offset, mstype.int64), shapex))
            shape_x = shape_x[1:]
            shapey = concat((P.Ones()((1,), mstype.int64), shapey))
            shapey = shapey[1:]
            tile_shape = P.Stack(1)((shapey, shape_x))
            r_shape = P.Reshape()(tile_shape, (-1,))
            axis = get_reduce_axis(r_shape)

        dout_reshaped = P.Reshape()(dout, r_shape)
        dout_origin_dtype = dout_reshaped.dtype
        # Currently, for Ascend and GPU, the reduce_sum's input does not support int16, int32 and int64.
        if dout_origin_dtype in (mstype.int16, mstype.int32, mstype.int64):
            dout_reshaped = cast(dout_reshaped, mstype.float32)
            dx = reduce_sum(dout_reshaped, axis)
            dx = cast(dx, dout_origin_dtype)
        else:
            dx = reduce_sum(dout_reshaped, axis)
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
        if F.is_sequence_value_unknown(x_shp):
            raise RuntimeError("Now, EmbeddingLookup op's grad don't support Dynamic Sense!")
        new_indices = sub_op(indices, offset)
        indices_size = size_op(new_indices)
        if indices_size > 0:
            # Reshape the 'new_indices'
            new_indices_shape_changed = (indices_size,)
            new_indices = reshape_op(new_indices, new_indices_shape_changed)
        else:
            new_indices_shape_changed = ()
        x_shp_tail = x_shp[1:]
        actual_dout_shape_changed = new_indices_shape_changed + x_shp_tail
        # Reshape the 'actual_dout' on device
        actual_dout = reshape_op(dout, actual_dout_shape_changed)
        return RowTensorInner(new_indices, actual_dout, x_shp), zeros_like(indices), zeros_like(offset)

    return bprop_sparse


@_primexpr
def make_begin(shp):
    """Creates a tuple with zero according to the shape."""
    begin = tuple([0 for _ in shp])
    return begin


def make_dynamic_begin(shp):
    """Creates a tuple with zero according to the shape."""
    begin = zeros_like(shp)
    return begin


@bprop_getters.register(P.Padding)
def get_bprop_padding(self):
    """Grad definition for `Padding` operation."""

    def bprop(x, out, dout):
        shp = shape_op(x)
        begin = ()
        if F.is_sequence_value_unknown(shp):
            shp = dyn_shape_op(x)
            begin = make_dynamic_begin(shp)
        else:
            begin = make_begin(shp)
        dx = P.Slice()(dout, begin, shp)
        return (dx,)

    return bprop


@_primexpr
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
        out_offset = G.ConcatOffset(len(x), axis)(x)
        input_nums = len(x)
        input_shapes = ()
        if isinstance(out_offset, tuple):
            for i in range(input_nums):
                input_shapes = input_shapes + (shape_op(x[i]),)
            is_uniform = _concat_grad_uniform(input_shapes, input_nums)
        else:
            # for dynamic shape
            for i in range(input_nums):
                input_shapes = input_shapes + (dyn_shape_op(x[i]),)
            is_uniform = False

        if isinstance(x, list):
            dx = []
            if is_uniform:
                dx_tuple = P.Split(axis, input_nums)(dout)
                for _, i in enumerate(dx_tuple):
                    dx = dx + [i,]
            else:
                for i in range(input_nums):
                    slice_out = P.Slice()(dout, out_offset[i], input_shapes[i])
                    dx = dx + [slice_out,]
        else:
            dx = ()
            if is_uniform:
                dx = P.Split(axis, input_nums)(dout)
            else:
                for i in range(input_nums):
                    slice_out = P.Slice()(dout, out_offset[i], input_shapes[i])
                    dx = dx + (slice_out,)
        return (dx,)

    return bprop


@bprop_getters.register(P.Slice)
def get_bprop_slice(self):
    """Generate bprop for Slice"""

    def bprop(x, begin, size, out, dout):
        dx = G.SliceGrad()(dout, x, begin, size)
        return (dx, zeros_like(begin), zeros_like(size))

    return bprop


@_primexpr
def _generate_inverse_index(x_shape, axis, batch_dims=0):
    x_rank = len(x_shape)
    index = tuple(range(x_rank))
    if axis < 0:
        axis += x_rank
    perm = index[:batch_dims] + index[batch_dims + 1:1 + axis] + (index[batch_dims],) + index[1 + axis:]
    return perm


@_primexpr
def _regenerate_output_shape(x_shp, ind_shp, axis):
    rank = len(x_shp)
    if axis < 0:
        axis += rank
    out_shape = x_shp[:axis] + ind_shp + x_shp[axis + 1:]
    return out_shape


def _dyn_regenerate_output_shape(x_shp, ind_shp, axis):
    """Get reshape new_shape"""
    rank = dyn_shape_op(x_shp)[0]
    if axis < 0:
        axis += rank
    out_shape = P.Concat(0)((x_shp[:axis], ind_shp, x_shp[axis + 1:]))
    return out_shape


def _dyn_generate_shape_index(out_shape, indices_shape, axis, batch_dims=0):
    """Get tranpose order"""
    out_rank = F.reshape(dyn_shape_op(out_shape), ())
    ind_rank = F.reshape(dyn_shape_op(indices_shape), ())
    if axis < 0:
        axis += out_rank - ind_rank + 1
    perm_part1 = P.Range()(F.cast(0, mstype.int32), F.cast(20, mstype.int32), F.cast(1, mstype.int32))
    ind_end = axis + ind_rank - batch_dims
    perm_part1 = perm_part1[axis: ind_end]
    index = P.Range()(F.cast(0, mstype.int32), F.cast(out_rank, mstype.int32), F.cast(1, mstype.int32))
    perm = F.hstack((index[:batch_dims], perm_part1, index[batch_dims:axis], index[ind_end:]))
    return perm


def _dyn_generate_inverse_index(x_shp, axis, batch_dims=0):
    """Get tranpose order"""
    x_rank = F.reshape(dyn_shape_op(x_shp), ())
    index = P.Range()(F.cast(0, mstype.int32), F.cast(x_rank, mstype.int32), F.cast(1, mstype.int32))
    if axis < 0:
        axis += x_rank
    perm = F.hstack((index[:batch_dims], index[batch_dims + 1:1 + axis], index[batch_dims], index[1 + axis:]))
    return perm


def calculate_batch_gather(values, indices, x_shape, axis, batch_dims):
    """Calculate gather grad with batch_dims"""
    values_shape = dyn_shape_op(values)
    batch_size = F.prod(x_shape[:batch_dims])
    batch_size = F.cast(batch_size, mstype.int32)
    axis_dim = F.cast(x_shape[axis], mstype.int32)

    # Move batch dimension to first non-batch dimension
    values = values.reshape((-1,) + values.shape[batch_dims:])
    indices = indices.reshape((-1,) + indices.shape[batch_dims:])
    offset = P.Range()(F.cast(0, mstype.int32), batch_size * axis_dim, axis_dim)
    offset_shape = F.hstack([batch_size] + [Tensor(1, dtype=mstype.int32) for _ in range(len(indices.shape) - 1)])
    offset = reshape(offset, offset_shape)
    indices = indices + offset
    num_segments = batch_size * axis_dim
    params_grad = unsorted_segment_sum(values, indices, num_segments)
    grad_shape = dyn_shape_op(params_grad)
    ret_shape = F.hstack([values_shape[:batch_dims], F.cast(axis_dim, mstype.int64), grad_shape[1:]])
    params_grad = reshape(params_grad, ret_shape)
    return params_grad


@bprop_getters.register(P.Gather)
@bprop_getters.register(P.GatherV2)
def get_bprop_gather_v2(self):
    """Generate bprop for GatherV2"""
    batch_dims = self.batch_dims

    def _dyn_bprop_gather_v2(x, indices, axis, dout):
        """dyn shape bprop for GatherV2"""
        orig_indices = indices
        x_shp = dyn_shape_op(x)
        ind_shp = dyn_shape_op(indices)
        out_shp = dyn_shape_op(dout)

        if F.rank(dout) == 0:
            dout = P.ExpandDims()(dout, -1)
        if F.rank(indices) == 0:
            indices = P.ExpandDims()(indices, -1)
            out_shp = _dyn_regenerate_output_shape(x_shp, ind_shp, axis)
            dout = reshape(dout, out_shp)

        # Example: out_shape:(3,2,3) axis 1 -> (1,0,2)
        perm_1 = _dyn_generate_shape_index(out_shp, ind_shp, axis, batch_dims)
        values_transpose = transpose(dout, perm_1)
        if batch_dims > 0:
            params_grad = calculate_batch_gather(values_transpose, indices, x_shp, axis, batch_dims)
        else:
            params_grad = unsorted_segment_sum(values_transpose, indices, x_shp[axis])
        perm_2 = _dyn_generate_inverse_index(x_shp, axis, batch_dims)
        params_grad = transpose(params_grad, perm_2)
        return params_grad, zeros_like(orig_indices), zeros_like(axis)

    def bprop(x, indices, axis, out, dout):
        is_mutable, axis = convert_to_tensor(axis)
        if (F.is_sequence_value_unknown(shape_op(x)) or F.is_sequence_value_unknown(shape_op(indices)) or \
                F.is_sequence_value_unknown(shape_op(dout))) and is_mutable:
            return _dyn_bprop_gather_v2(x, indices, axis, dout)
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
        perm_1 = generate_shape_index(out_shp, ind_shp, axis, batch_dims)
        values_transpose = transpose(dout, perm_1)
        dyn_x_sape = dyn_shape_op(x)
        if batch_dims > 0:
            params_grad = calculate_batch_gather(values_transpose, indices, dyn_x_sape, axis, batch_dims)
        else:
            params_grad = unsorted_segment_sum(values_transpose, indices, dyn_x_sape[axis])
        # Example: out_shape:(3,2,3) axis 2 -> (1,2,0)
        perm_2 = _generate_inverse_index(x_shp, axis, batch_dims)
        params_grad = transpose(params_grad, perm_2)
        return params_grad, zeros_like(orig_indices), zeros_like(axis)

    return bprop


@bprop_getters.register(P.GatherD)
def get_bprop_gather_d(self):
    """Generate bprop for GatherD"""

    def bprop(x, dim, index, out, dout):
        dx = G.GatherDGradV2()(x, dim, index, dout)
        return dx, zeros_like(dim), zeros_like(index)

    return bprop


@bprop_getters.register(G.GatherDGrad)
def get_bprop_gather_d_grad(self):
    """Generate bprop for GatherDGrad"""
    op = P.Gather()
    dim = self.dim
    x_shp = self.out_shape

    def bprop(index, x, out, dout):
        index_shp = shape_op(index)
        dim_before_axis = 1
        for i in range(dim):
            dim_before_axis *= x_shp[i]
        dim_at_axis_index = index_shp[dim]
        dim_at_axis_output = x_shp[dim]
        dim_after_axis = 1
        for i in range(dim + 1, len(x_shp)):
            dim_after_axis *= x_shp[i]
        element = dim_before_axis * dim_at_axis_index * dim_after_axis
        id_ = range_op(0, element, 1, index.dtype)
        i = id_ // (dim_at_axis_index * dim_after_axis)
        k = id_ % dim_after_axis
        j = P.Cast()(index < 0, index.dtype)
        j_read = dim_at_axis_index * j + index
        j_read = P.Reshape()(j_read, (-1,))
        read_id = i * dim_at_axis_output * dim_after_axis + j_read * dim_after_axis + k
        dout = P.Reshape()(dout, (-1,))
        dx = op(dout, read_id, 0)
        dx = P.Reshape()(dx, shape_op(x))
        return zeros_like(index), dx

    return bprop


@bprop_getters.register(G.GatherDGradV2)
def get_bprop_gather_d_grad_v2(self):
    """Generate bprop for GatherDGradV2"""
    op = P.Gather()
    dim = self.dim

    def bprop(index, x, out, dout):
        index_shp = shape_op(index)
        dim_before_axis = 1
        x_shp = shape_op(x)
        for i in range(dim):
            dim_before_axis *= x_shp[i]
        dim_at_axis_index = index_shp[dim]
        dim_at_axis_output = x_shp[dim]
        dim_after_axis = 1
        for i in range(dim + 1, len(x_shp)):
            dim_after_axis *= x_shp[i]
        element = dim_before_axis * dim_at_axis_index * dim_after_axis
        id_ = range_op(0, element, 1, index.dtype)
        i = id_ // (dim_at_axis_index * dim_after_axis)
        k = id_ % dim_after_axis
        j = P.Cast()(index < 0, index.dtype)
        j_read = dim_at_axis_index * j + index
        j_read = P.Reshape()(j_read, (-1,))
        read_id = i * dim_at_axis_output * dim_after_axis + j_read * dim_after_axis + k
        dout = P.Reshape()(dout, (-1,))
        dx = op(dout, read_id, 0)
        dx = P.Reshape()(dx, shape_op(x))
        return zeros_like(index), dx

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
            return RowTensorInner(indices_new, values, x_shp), zeros_like(indices), zeros_like(axis)
        if F.rank(dout) == 0:
            dout = P.ExpandDims()(dout, -1)
        if F.rank(indices) == 0:
            indices = P.ExpandDims()(indices, -1)
        out_shp = shape_op(dout)
        ind_shp = shape_op(indices)
        # Example: out_shape:(3,2,3) axis 1 -> (1,0,2)
        perm_1 = generate_shape_index(out_shp, ind_shp, axis)
        values_transpose = transpose(dout, perm_1)
        params_grad = unsorted_segment_sum(values_transpose, indices, shape_op(x)[axis])
        # Example: out_shape:(3,2,3) axis 2 -> (1,2,0)
        perm_2 = _generate_inverse_index(x_shp, axis)
        params_grad = transpose(params_grad, perm_2)
        return params_grad, zeros_like(indices), zeros_like(axis)

    return bprop


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

        indices_dtype = dtype(indices)
        range_flatten_index = range_op(0, outer_dim * in_lastdim, in_lastdim, indices_dtype)

        # expand_dims to (k, 1), then broadcast
        ind = reshape_op(ind_2d + expand_dims(range_flatten_index, -1), (-1,))
        x_shape_1d = get_1d_shape(top_k_input_shape)

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


@bprop_getters.register(P.Range)
def get_bprop_range(self):
    """Generate bprop for Range"""

    def bprop(start, limit, delta, out, dout):
        return (zeros_like(start), zeros_like(limit), zeros_like(delta))

    return bprop


@bprop_getters.register(P.Pack)
@bprop_getters.register(P.Stack)
def get_bprop_stack(self):
    """Generate bprop for Stack"""
    axis = self.axis

    def bprop(x, out, dout):
        stack_grad = P.Unstack(num=len(x), axis=axis)
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
        x_shape = shape_op(x)
        if F.is_sequence_value_unknown(x_shape):
            x_shape = dyn_shape_op(x)
        dx = input_grad(dout, x_shape, begin, end, strides)
        return dx, zeros_like(begin), zeros_like(end), zeros_like(strides)

    return bprop


@bprop_getters.register(G.StridedSliceGrad)
def get_bprop_strided_slice_grad(self):
    """Generate bprop for StridedSliceGrad"""
    strided_slice = P.StridedSlice(begin_mask=self.begin_mask,
                                   end_mask=self.end_mask,
                                   ellipsis_mask=self.ellipsis_mask,
                                   new_axis_mask=self.new_axis_mask,
                                   shrink_axis_mask=self.shrink_axis_mask)

    def bprop(dy, shapex, begin, end, strides, out, dout):
        return strided_slice(dout, begin, end, strides), zeros_like(shapex), zeros_like(begin), zeros_like(end), \
               zeros_like(strides)

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
    """Generate bprop for ZerosLike"""

    def bprop(x, out, dout):
        return (zeros_like(x),)

    return bprop


@bprop_getters.register(P.ResizeNearestNeighbor)
def get_bprop_resize_nearest_neighbor(self):
    """Generate bprop for ResizeNearestNeighbor"""
    op = G.ResizeNearestNeighborGrad(self.align_corners)
    tensor_shape = P.TensorShape()

    def bprop(inputs, out, dout):
        if F.is_sequence_value_unknown(shape_op(inputs)) or F.is_sequence_shape_unknown(shape_op(inputs)):
            shp = tensor_shape(inputs)
        else:
            shp = shape_op(inputs)
        # 2 and 3 represent the height and width
        shp = shp[2:]
        return (op(dout, shp),)

    return bprop


@bprop_getters.register(P.GatherNd)
def get_bprop_gather_nd(self):
    """Generate bprop for GatherNd"""
    op = P.ScatterNd()

    def bprop(x, indices, out, dout):
        shp = shape_op(x)
        if F.is_sequence_value_unknown(shp):
            shp = dyn_shape_op(x)
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


@bprop_getters.register(P.TensorScatterAdd)
def get_bprop_tensor_scatter_add(self):
    """Generate bprop for TensorScatterAdd"""
    gather_nd = P.GatherNd()

    def bprop(x, indices, update, out, dout):
        update_grad = gather_nd(dout, indices)
        return dout, zeros_like(indices), update_grad

    return bprop


@bprop_getters.register(P.ScatterMax)
def get_bprop_scatter_max(self):
    """Generate bprop for ScatterMax"""
    gather = P.Gather()

    def bprop(x, indices, update, out, dout):
        return dout, zeros_like(indices), gather(dout, indices, 0)

    return bprop


@bprop_getters.register(P.ScatterMin)
def get_bprop_scatter_min(self):
    """Generate bprop for ScatterMin"""
    gather = P.Gather()

    def bprop(x, indices, update, out, dout):
        return dout, zeros_like(indices), gather(dout, indices, 0)

    return bprop


@bprop_getters.register(P.ScatterUpdate)
def get_bprop_scatter_update(self):
    """Generate bprop for ScatterUpdate"""
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
    zero_slice = zeros_like(gathered)
    if is_positive is None:
        is_positive = greater_equal(ids, 0)
        is_positive_shape = shape_op(is_positive)
        gathered_shape = shape_op(gathered)
        if F.is_sequence_value_unknown(gathered_shape) or F.is_sequence_value_unknown(is_positive_shape):
            gathered_shape = dyn_shape_op(gathered)
            rank_gathered = dyn_rank(gathered)
            fill_gathered = dyn_fill(mstype.int64, gathered_shape, 1)
            is_positive_shape = dyn_shape_op(is_positive)
            rank_positive = dyn_rank(is_positive)
            if rank_gathered - rank_positive > 0:
                padded_size = F.expand_dims(rank_gathered - rank_positive, 0)
                padded_shape = dyn_ones(padded_size, is_positive_shape.dtype)
                is_positive_shape = P.Concat(-1)((is_positive_shape, padded_shape))
            is_positive = reshape(is_positive, is_positive_shape)
            is_positive = logical_and(is_positive, F.cast(fill_gathered, mstype.bool_))
        else:
            broadcastable_shape = is_positive_shape
            for _ in range(rank(gathered) - rank(is_positive)):
                broadcastable_shape += (1,)
            is_positive = reshape(is_positive, broadcastable_shape)
            is_positive = logical_and(is_positive, fill(mstype.bool_, gathered_shape, 1))
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
        if x.dtype == mstype.complex64 or x.dtype == mstype.complex128:
            is_zero = equal(x, F.scalar_to_tensor(0).astype(x.dtype))
        else:
            is_zero = equal(cast(x, mstype.float32), F.scalar_to_tensor(0).astype(np.float32))
        num_zero = unsorted_segment_sum(cast(is_zero, mstype.int32), segment_ids, num_segments)
        grad = select(greater(num_zero, 1), zeros_like(dout), dout)
        if x.dtype == mstype.complex64 or x.dtype == mstype.complex128:
            non_zero_data = select(is_zero, ones_like(x), x)
        else:
            temp_var = ones_like(cast(x, mstype.float32))
            non_zero_data = select(is_zero, cast(temp_var, x.dtype), x)
        non_zero_prod = unsorted_segment_prod(non_zero_data, segment_ids, num_segments)
        zero_clipped_indices = maximum(segment_ids, zeros_like(segment_ids))
        gathered_prod = gather(out, zero_clipped_indices, 0)
        gathered_non_zero_prod = gather(non_zero_prod, zero_clipped_indices, 0)
        if x.dtype == mstype.uint32 or x.dtype == mstype.uint64:
            prod_divided_by_x = cast(gathered_prod, mstype.float32) / cast(x, mstype.float32)
        else:
            prod_divided_by_x = gathered_prod / x
        partial_derivative = select(is_zero, gathered_non_zero_prod,
                                    cast(prod_divided_by_x, gathered_non_zero_prod.dtype))
        gathered_grad, _, _ = _gather_drop_negatives(grad, segment_ids, zero_clipped_indices, None)
        if x.dtype == mstype.uint32 or x.dtype == mstype.uint64:
            temp_dx = cast(gathered_grad, mstype.float32) * cast(partial_derivative, mstype.float32)
            dx = cast(temp_dx, x.dtype)
        else:
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

    def bprop(x, out, dout):
        x_shape = shape_op(x)
        dout_shape = shape_op(dout)
        broadcast_shape = shape_op(out)
        dynamic = F.is_sequence_value_unknown(x_shape) or F.is_sequence_value_unknown(dout_shape)
        if not dynamic and x_shape == dout_shape:
            return (dout,)
        dynamic = dynamic or F.is_sequence_value_unknown(broadcast_shape)
        out_type = dout.dtype
        if not dynamic:
            _, reduction_axes = broadcast_gradient_args(broadcast_shape, x_shape)
            if out_type in (ms.int16, ms.int32, ms.int64):
                dout = P.Cast()(dout, ms.float32)
                reduced_grad = reduce_keep_dim(dout, reduction_axes)
                reduced_grad = P.Cast()(reduced_grad, out_type)
            else:
                reduced_grad = reduce_keep_dim(dout, reduction_axes)
            dx = reshape(reduced_grad, x_shape)
        else:
            x_shape = dyn_shape_op(x)
            broadcast_shape = dyn_shape_op(out)
            _, reduction_axes = DynamicBroadcastGradientArgs()(broadcast_shape, x_shape)
            if out_type in (ms.int16, ms.int32, ms.int64):
                dout = P.Cast()(dout, ms.float32)
                reduced_grad = sum_grad_reduce_axis(dout, reduction_axes, keep_dims=True)
                reduced_grad = P.Cast()(reduced_grad, out_type)
            else:
                reduced_grad = sum_grad_reduce_axis(dout, reduction_axes, keep_dims=True)
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


@bprop_getters.register(P.MaskedSelect)
def get_bprop_masked_select(self):
    """Generate bprop for MaskedSelect"""
    op = G.MaskedSelectGrad()

    def bprop(x, mask, out, dout):
        dx = op(x, mask, dout)
        return (dx, zeros_like(mask))

    return bprop


@bprop_getters.register(NonZero)
def get_bprop_non_zero(self):
    """Generate bprop for NonZero"""

    def bprop(x, out, dout):
        return (zeros_like(x),)

    return bprop
