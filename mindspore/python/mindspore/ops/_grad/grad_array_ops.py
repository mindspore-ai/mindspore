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

from mindspore.ops import operations as P
from mindspore.ops.operations import _grad_ops as G
from mindspore.ops.composite.multitype_ops.zeros_like_impl import zeros_like
from mindspore.ops import functional as F
from mindspore.ops._grad.grad_base import bprop_getters
from mindspore.ops.primitive import _primexpr
from mindspore.common.sparse_tensor import RowTensorInner
from mindspore.ops._utils.utils import generate_shape_index

reduce_sum = P.ReduceSum()
unsorted_segment_sum = P.UnsortedSegmentSum()
transpose = P.Transpose()
shape_op = P.Shape()
dyn_shape_op = P.TensorShape()
reshape = P.Reshape()
size_op = P.Size()
invert_permutation = P.InvertPermutation()
logical_and = P.LogicalAnd()


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
def _generate_inverse_index(x_shape, axis, batch_dims=0):
    x_rank = len(x_shape)
    index = tuple(range(x_rank))
    if axis < 0:
        axis += x_rank
    perm = index[:batch_dims] + index[batch_dims + 1:1 + axis] + (index[batch_dims],) + index[1 + axis:]
    return perm


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


@bprop_getters.register(P.Unstack)
def get_bprop_unstack(self):
    """Generate bprop for Unstack"""
    axis = self.axis

    def bprop(x, out, dout):
        unstack_grad = P.Stack(axis)
        out = unstack_grad(dout)
        return (out,)

    return bprop


@bprop_getters.register(P.Eye)
def get_bprop_eye(self):
    """Generate bprop for Eye"""

    def bprop(n, m, t, out, dout):
        return zeros_like(n), zeros_like(m), zeros_like(t)

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


@bprop_getters.register(P.ScatterUpdate)
def get_bprop_scatter_update(self):
    """Generate bprop for ScatterUpdate"""
    gather = P.Gather()

    def bprop(x, indices, update, out, dout):
        return dout, zeros_like(indices), gather(dout, indices, 0)

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
