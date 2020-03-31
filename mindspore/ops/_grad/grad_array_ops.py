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
from ..composite.multitype_ops.zeros_like_impl import zeros_like
from .. import functional as F
from .grad_base import bprop_getters
from ..primitive import constexpr
from ... import context

reduce_sum = P.ReduceSum()
unsorted_segment_sum = P.UnsortedSegmentSum()
transpose = P.Transpose()
shape_op = P.Shape()
reshape = P.Reshape()
invert_permutation = P.InvertPermutation()


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
        out_offset = P.ConcatOffset(F.tuple_len(x), axis)(x)
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
        return (dx,)

    def bprop_gpu(x, begin, size, out, dout):
        dx = dx = G.SliceGrad()(dout, x, begin, size)
        return (dx,)

    if context.get_context('device_target') == "GPU":
        return bprop_gpu
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
        return params_grad, zeros_like(indices)
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
