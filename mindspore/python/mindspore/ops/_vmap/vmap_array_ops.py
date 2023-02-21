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

"""array_ops vmap impl."""
from __future__ import absolute_import

import mindspore
import mindspore.numpy as mnp
from mindspore import ops
from mindspore.common import Tensor
from mindspore._c_expression import Tensor as Tensor_
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops import constexpr
from mindspore.ops.primitive import _primexpr
from mindspore.ops.operations._grad_ops import MaskedSelectGrad
from mindspore.ops.operations import _grad_ops as G
from mindspore.ops.operations.array_ops import Fills, UniqueConsecutive, Col2Im, NonZero, IndexFill, \
    TensorScatterElements
from mindspore.ops.operations.random_ops import RandomPoisson
from mindspore.ops.primitive import Primitive
from mindspore.ops._vmap.vmap_base import vmap_rules_getters, vmap_general_preprocess, _bdim_at_front, \
    _raise_value_error, _vmap_clone_prim, _handle_broadcasting, get_unsupported_dynamic_vmap_rule, _broadcast_by_axis, \
    get_unop_vmap_rule, _get_reduce_out_dim, _get_reduce_batch_axis, \
    _bdim_at_any
from mindspore.ops.function import _VmapGeneralRule


@vmap_rules_getters.register(P.NoRepeatNGram)
def get_no_repeat_ngram_vmap_rule(prim, axis_size):
    """VmapRule for `NoRepeatNGram` operation."""

    def vmap_rule(state_seq_bdim, log_probs_bdim):
        is_all_none, result = vmap_general_preprocess(prim, state_seq_bdim, log_probs_bdim)
        if is_all_none:
            return result

        state_seq, state_seq_dim = state_seq_bdim
        log_probs, log_probs_dim = log_probs_bdim
        state_seq = _bdim_at_front(state_seq, state_seq_dim, axis_size)
        log_probs = _bdim_at_front(log_probs, log_probs_dim, axis_size)
        s_ori_shape = F.shape(state_seq)
        l_ori_shape = F.shape(log_probs)
        state_seq = F.reshape(state_seq, (-1,) + s_ori_shape[-2:])
        log_probs = F.reshape(log_probs, (-1,) + l_ori_shape[-2:])
        out = prim(state_seq, log_probs)
        out = F.reshape(out, l_ori_shape)
        return out, 0

    return vmap_rule


@vmap_rules_getters.register("Cast")
def get_cast_vmap_rule(prim, axis_size):
    """VmapRule for `Cast` operation."""
    if isinstance(prim, str):
        prim_name = prim
        prim = Primitive(prim)
    else:
        prim_name = prim.name

    def vmap_rule(input_bdim, type_bdim):
        input_x, x_dim = input_bdim
        dtype, type_dim = type_bdim
        if type_dim is not None:
            _raise_value_error("The source axis of 'type' in `{}` must be None, "
                               "but got {}.".format(prim_name, type_dim))
        out = prim(input_x, dtype)
        return out, x_dim

    return vmap_rule


@vmap_rules_getters.register(P.Argmin)
def get_argmin_vmap_rule(prim, axis_size):
    """VmapRule for `Argmin` operations."""
    if isinstance(prim, str):
        axis = -1
        output_type = mindspore.int32
    else:
        axis = prim.axis
        output_type = prim.output_type

    def vmap_rule(x_bdim):
        is_all_none, result = vmap_general_preprocess(prim, x_bdim)
        if is_all_none:
            return result
        var, x_dim = x_bdim
        x_ndim = ops.rank(var)
        batch_axis = _get_reduce_batch_axis(axis, x_dim, x_ndim)
        out = P.Argmin(batch_axis, output_type)(var)
        out_dim = _get_reduce_out_dim(x_dim, batch_axis)
        return out, out_dim

    return vmap_rule


@vmap_rules_getters.register(P.ArgMaxWithValue)
@vmap_rules_getters.register(P.ArgMinWithValue)
def get_arg_min_max_with_value_vmap_rule(prim, axis_size):
    """VmapRule for `ArgMaxWithValue` and `ArgMinWithValue` operations."""
    if isinstance(prim, str):
        axis = -1
        keep_dims = False
    else:
        axis = prim.axis
        keep_dims = prim.keep_dims

    cum_fun_map = {
        "ArgMaxWithValue": P.ArgMaxWithValue,
        "ArgMinWithValue": P.ArgMinWithValue,
    }
    prim_class = cum_fun_map.get(prim.name)

    def vmap_rule(x_bdim):
        is_all_none, result = vmap_general_preprocess(prim, x_bdim)
        if is_all_none:
            return result
        var, x_dim = x_bdim
        x_ndim = ops.rank(var)
        batch_axis = _get_reduce_batch_axis(axis, x_dim, x_ndim)
        index, out = prim_class(batch_axis, keep_dims)(var)
        out_dim = _get_reduce_out_dim(x_dim, batch_axis, keep_dims)
        return (index, out_dim), (out, out_dim)

    return vmap_rule


@_primexpr
def _get_prefix(indices_shape, axis_size, indices_dtype):
    """
    Generate prefix by indices shape, whose -1 axis value is the index value of axis 0.
    eg. if the indices is Tensor([[[1, 2], [2, 3]],
                                  [[2, 3], [3, 4]]])
    we got the indices_shape (2, 2, 2),
    the generated prefix is a Tensor([[[0], [0]],
                                      [[1], [1]]])
    """

    indices_len = len(indices_shape)

    if indices_len == 1:
        prefix = P.Range()(Tensor(0, indices_dtype), P.Fill()(
            indices_dtype, (), axis_size), Tensor(1, indices_dtype))
        return prefix

    indices_end = indices_len - 1
    prefix_shape = ()
    expand_shape = ()
    for i, element in enumerate(indices_shape):
        if i == indices_end:
            prefix_shape = prefix_shape + (1,)
        else:
            prefix_shape = prefix_shape + (element,)
        if i == 0:
            expand_shape = expand_shape + (element,)
        else:
            expand_shape = expand_shape + (1,)

    prefix = P.BroadcastTo(prefix_shape)(P.Reshape()(P.Range()(Tensor(
        0, indices_dtype), Tensor(axis_size, indices_dtype), Tensor(1, indices_dtype)), expand_shape))
    return prefix


@vmap_rules_getters.register(P.Transpose)
def get_transpose_vmap_rule(prim, axis_size):
    """VmapRule for `Transpose` operation."""
    if isinstance(prim, str):
        prim = Primitive(prim)

    @_primexpr
    def _get_transpose_batch_perm(dim, perm, x_rank):
        """Generate batch_perm based on the original perm of transpose operation and dim of the input."""
        if dim < 0:
            dim = dim + x_rank
        batch_perm = (dim,)

        perm_len = len(perm)

        for i in perm:
            if i < 0:
                i += perm_len

            if i < dim:
                batch_perm = batch_perm + (i,)
            else:
                index = i + 1
                batch_perm = batch_perm + (index,)
        return batch_perm

    def vmap_rule(x_bdim, perm_bdim):
        is_all_none, result = vmap_general_preprocess(prim, x_bdim, perm_bdim)
        if is_all_none:
            return result

        x, dim = x_bdim
        perm, perm_dim = perm_bdim
        if perm_dim is not None:
            _raise_value_error("The source axis of perm in `Transpose` must be None, "
                               "but got {}.".format(perm_dim))
        x_rank = F.rank(x)
        batch_perm = _get_transpose_batch_perm(dim, perm, x_rank)
        out = prim(x, batch_perm)
        return out, 0

    return vmap_rule


@vmap_rules_getters.register(P.Tile)
def get_tile_vmap_rule(prim, axis_size):
    """VmapRule for `P.Tile` operation."""
    if isinstance(prim, str):
        prim = Primitive(prim)

    @_primexpr
    def _get_batch_multiples(input_shape, dim, multiples):
        input_ndim = len(input_shape)
        multiples_ndim = len(multiples)
        if multiples_ndim < input_ndim - 1:
            multiples = (1,) * (input_ndim - 1 - multiples_ndim) + multiples

        rev_dim = input_ndim - 1 - dim
        if rev_dim == 0:
            return multiples + (1,), multiples_ndim

        batch_multiples = list(multiples)
        batch_multiples.insert(-rev_dim, 1)
        return tuple(batch_multiples), multiples_ndim - rev_dim

    def vmap_rule(input_bdim, multiples_bdim):
        is_all_none, result = vmap_general_preprocess(prim, input_bdim, multiples_bdim)
        if is_all_none:
            return result

        input_x, dim = input_bdim
        multiples, multiples_dim = multiples_bdim
        if multiples_dim is not None:
            _raise_value_error("The source axis of shape in `Tile` must be None, but got {}.".format(multiples_dim))

        input_shape = F.shape(input_x)
        batch_multiples, out_dim = _get_batch_multiples(input_shape, dim, multiples)
        repeat_tensor = P.Tile()(input_x, batch_multiples)
        return repeat_tensor, out_dim

    return vmap_rule


@vmap_rules_getters.register(P.Concat)
def get_concat_vmap_rule(prim, axis_size):
    """VmapRule for `Concat` operation."""
    if isinstance(prim, str):
        prim = P.Concat(0)
        new_axis = 0
    else:
        new_axis = prim.axis
    if new_axis >= 0:
        new_axis += 1

    def vmap_rule(*inputs_bdim):
        is_all_none, result = vmap_general_preprocess(prim, *inputs_bdim)
        if is_all_none:
            return result

        if not isinstance(inputs_bdim, (tuple, list)):
            _raise_value_error("The 'x' of P.Concat is neither tuple nor list.")

        args = inputs_bdim[0]
        vals = ()
        for each_arg in args:
            x, bdim = each_arg
            x = _bdim_at_front(x, bdim, axis_size)
            vals = vals + (x,)

        out = P.Concat(new_axis)(vals)
        return out, 0

    return vmap_rule


@vmap_rules_getters.register(P.Stack)
def get_stack_vmap_rule(prim, axis_size):
    """VmapRule for `Stack` operation."""
    if isinstance(prim, str):
        prim = P.Stack(0)
        new_axis = 0
    else:
        new_axis = prim.axis
    if new_axis >= 0:
        new_axis += 1

    def vmap_rule(*inputs_bdim):
        is_all_none, result = vmap_general_preprocess(prim, *inputs_bdim)
        if is_all_none:
            return result

        if not isinstance(inputs_bdim, (tuple, list)):
            _raise_value_error("The 'x' of P.Stack is neither tuple nor list.")

        args = inputs_bdim[0]
        vals = ()
        for each_arg in args:
            x, bdim = each_arg
            x = _bdim_at_front(x, bdim, axis_size)
            vals = vals + (x,)

        out = P.Stack(new_axis)(vals)
        return out, 0

    return vmap_rule


@vmap_rules_getters.register(P.Unstack)
def get_unstack_vmap_rule(prim, axis_size):
    """VmapRule for `Unstack` operation."""
    if isinstance(prim, str):
        prim = P.Unstack(0)
        new_axis = 0
    else:
        new_axis = prim.axis
    if new_axis >= 0:
        new_axis += 1

    def vmap_rule(inputs_bdim):
        is_all_none, result = vmap_general_preprocess(prim, inputs_bdim)
        if is_all_none:
            return result

        x, bdim = inputs_bdim
        x = _bdim_at_front(x, bdim, axis_size)

        outputs = P.Unstack(new_axis)(x)
        outputs_tuple = ()
        for output in outputs:
            outputs_tuple = outputs_tuple + ((output, 0),)
        return outputs_tuple

    return vmap_rule


@vmap_rules_getters.register(P.Reshape)
def get_reshape_vmap_rule(prim, axis_size):
    """VmapRule for `Reshape` operation."""


    @_primexpr
    def get_batch_shape(x_shape, x_dim, target_shape, axis_size):
        if x_dim == 0:
            return (axis_size,) + target_shape, 0, False

        if x_dim in (len(x_shape) - 1, -1):
            return target_shape + (axis_size,), len(target_shape), False

        neg_index = -1
        dim_prod = 1
        for i, shp_i in enumerate(target_shape):
            if shp_i == -1:
                neg_index = i
            else:
                dim_prod *= shp_i
        arr_prod = 1
        for i in x_shape:
            arr_prod *= i
        target_shape_list = list(target_shape)
        if neg_index != -1:
            neg_index_size = int(arr_prod // (dim_prod * axis_size))
            target_shape_list[neg_index] = neg_index_size

        arr_prod_before_dim = 1
        for i in x_shape[:x_dim]:
            arr_prod_before_dim *= i
        dim_prod = 1
        for i, shp_i in enumerate(target_shape_list, start=1):
            dim_prod *= shp_i
            if dim_prod == arr_prod_before_dim:
                return tuple(target_shape_list[:i]) + (axis_size,) + tuple(target_shape_list[i:]), i, False
            if dim_prod > arr_prod_before_dim:
                return 0, 0, True

        return 0, 0, True

    def vmap_rule(operand_bdim, shape_bdim):
        is_all_none, result = vmap_general_preprocess(prim, operand_bdim, shape_bdim)
        if is_all_none:
            return result

        x, dim = operand_bdim
        shape, shape_dim = shape_bdim
        if shape_dim is not None:
            _raise_value_error("The source axis of shape in `Reshape` must be None, but got {}.".format(shape_dim))

        x_shape = F.shape(x)
        batch_shape, out_axis, need_moveaxis = get_batch_shape(x_shape, dim, shape, axis_size)
        if need_moveaxis:
            # for such case: `x_shape` is (2, 3, 4, 5, 6), `x_dim` is 3, and `shape` is (-1,)
            x = mnp.moveaxis(x, dim, 0)
            batch_shape = (axis_size,) + shape
            out = prim(x, batch_shape)
            return out, 0

        out = prim(x, batch_shape)
        return out, out_axis

    return vmap_rule


@vmap_rules_getters.register(P.ReverseSequence)
def get_reverse_sequence_vmap_rule(prim, axis_size):
    """VmapRule for `ReverseSequence` operation."""
    if isinstance(prim, str):
        prim = Primitive(prim)
    reshape = P.Reshape()
    batch_dim = prim.batch_dim_
    seq_dim = prim.seq_dim_

    @_primexpr
    def get_batch_seq_dim(dim, batch_dim_, seq_dim_):
        if dim is None:
            batch_dim_ += 1
            seq_dim_ += 1
        else:
            if seq_dim_ == dim:
                seq_dim_ += 1
                if seq_dim_ == batch_dim_:
                    batch_dim_ += 1
            elif batch_dim_ == dim:
                batch_dim_ += 1
                if seq_dim_ == batch_dim_:
                    seq_dim_ += 1
        return batch_dim_, seq_dim_

    @_primexpr
    def get_seq_dim(dim, batch_dim_, seq_dim_):
        if dim is None:
            return seq_dim_
        if seq_dim_ < dim and seq_dim_ < batch_dim_:
            seq_dim_ = seq_dim_ + 1
        elif seq_dim_ > dim and seq_dim_ > batch_dim_:
            seq_dim_ = seq_dim_ - 1
        return seq_dim_

    def vmap_rule(x_bdim, seq_lengths_bdim):
        is_all_none, result = vmap_general_preprocess(prim, x_bdim, seq_lengths_bdim)
        if is_all_none:
            return result
        x, dim = x_bdim
        seq_lengths, seq_lengths_dim = seq_lengths_bdim
        seq_lengths = _bdim_at_front(seq_lengths, seq_lengths_dim, axis_size)
        origin_shape = x.shape
        batch_dim_, seq_dim_ = get_batch_seq_dim(dim, batch_dim, seq_dim)
        if dim is None:
            x = _bdim_at_front(x, dim, axis_size)
            origin_shape = x.shape
            x = mnp.moveaxis(x, batch_dim_, 1)
            real_dim = 0
        else:
            x = mnp.moveaxis(x, [dim, batch_dim_], [0, 1])
            real_dim = dim
        shape = x.shape
        shape = (shape[0] * shape[1],) + tuple(shape[2:])
        x = reshape(x, shape)
        seq_dim_ = get_seq_dim(dim, batch_dim_, seq_dim_)
        seq_lengths = reshape(seq_lengths, (-1,))
        x = P.ReverseSequence(seq_dim=seq_dim_)(x, seq_lengths)
        shape = x.shape
        shape = (origin_shape[real_dim], origin_shape[batch_dim_],) + tuple(shape[1:])
        out = reshape(x, shape)
        if batch_dim_ not in (0, 1):
            out = mnp.moveaxis(out, 1, batch_dim_)
        return out, 0

    return vmap_rule


@vmap_rules_getters.register(P.Flatten)
def get_flatten_vmap_rule(prim, axis_size):
    """VmapRule for `Flatten` operation."""

    def vmap_rule(x_bdim):
        is_all_none, result = vmap_general_preprocess(prim, x_bdim)
        if is_all_none:
            return result

        x, x_dim = x_bdim
        x = _bdim_at_front(x, x_dim, axis_size)
        x_shape = F.shape(x)
        output = F.reshape(x, x_shape[0:2] + (-1,))
        return output, 0

    return vmap_rule


@vmap_rules_getters.register(P.Select)
def get_select_vmap_rule(prim, axis_size):
    """VmapRule for 'Select' operation."""
    if isinstance(prim, str):
        prim = P.Select()

    def vmap_rule(condition_bdim, x_bdim, y_bdim):
        is_all_none, result = vmap_general_preprocess(prim, condition_bdim, x_bdim, y_bdim)
        if is_all_none:
            return result

        condition, condition_dim = condition_bdim
        x, x_dim = x_bdim
        y, y_dim = y_bdim

        condition = _bdim_at_front(condition, condition_dim, axis_size)
        x = _bdim_at_front(x, x_dim, axis_size)
        y = _bdim_at_front(y, y_dim, axis_size)

        out = prim(condition, x, y)

        return out, 0

    return vmap_rule


@vmap_rules_getters.register(P.ScatterNd)
def get_scatter_nd_vmap_rule(prim, axis_size):
    """
    VmapRule for `ScatterNd` operation.

    An example for the rule:
    --- inputs info
        indices.shape = [10, 3, 2, 2]
        updates.shape = [10, 3, 2, 5]
        shape         =     [6, 4, 5]
    the first dim (10) is batch.
    the shape without batch dim are:
        indices.shape = [3, 2, 2]
        updates.shape = [3, 2, 5]
        shape         = [6, 4, 5]
    --- step 1
    Change the `shape` to `[60, 4, 5]`, set the indices `offset` to 6 (original first dim).
    Since there's a constraint `updates.shape = indices.shape[:-1] + shape[indices.shape[-1]:]` in the `ScatterNd` op,
    so the `shape` with a batch dim is invalid, but its first dim can be changed.
    --- step 2
    Generate an constant offset tensor for the indices, which `indices_offset.shape = [10, 1, 1, 2]`,
    for i in [0, 10), set `indices_offset[i, :, :, 0] = i * offset`.
    The output batch dim was concat by original 0-axis, so the indices should be offset.
    Only the 0-dim of output is changed, so only the `indices_offset[i,:,:,0]` is set, and the `indices_offset[i,:,:,1]`
    is leave as zero.
    --- step 3
    Add the `indices_offset` with `indices`.
    --- step 4
    Call `ScatterNd` with new `indices`, old `updates`, and new `shape (60, 4, 5)`.
    --- step 5
    Reshape the output tensor to `[10, 6, 4, 5]`
    """

    @_primexpr
    def _refine_shape(shape, bdim_size):
        offset = shape[0]
        return (bdim_size * shape[0],) + tuple(shape[1:]), offset, (bdim_size,) + tuple(shape)

    @_primexpr
    def _gen_indices_offset(shape, offset):
        # original rank(indices.shape) is required >= 2, so indices with batch dim's rank >= 3.
        shape = (shape[0],) + (1,) * (len(shape) - 2) + (shape[-1],)
        val = P.Zeros()((shape[0], shape[-1]), mindspore.int32)
        for i in range(shape[0]):
            val[i, 0] = i * offset
        return P.Reshape()(val, shape)

    if isinstance(prim, str):
        prim = Primitive(prim)

    def vmap_rule(indices_bdim, updates_bdim, shape_bdim):
        is_all_none, result = vmap_general_preprocess(prim, indices_bdim, updates_bdim, shape_bdim)
        if is_all_none:
            return result
        indices, indices_dim = indices_bdim
        updates, updates_dim = updates_bdim
        shape, shape_dim = shape_bdim
        if shape_dim is not None:
            _raise_value_error("The source axis of `shape` in `{}` must be None, "
                               "but got {}.".format(prim.name, shape_dim))
        indices = _bdim_at_front(indices, indices_dim, axis_size)
        updates = _bdim_at_front(updates, updates_dim, axis_size)
        new_shape, offset, out_shape = _refine_shape(shape, axis_size)
        indices_shape = F.shape(indices)
        indices_dtype = F.dtype(indices)
        offset_val = _gen_indices_offset(indices_shape, offset)
        indices_offset = P.Cast()(offset_val, indices_dtype)
        new_indices = P.Add()(indices, indices_offset)
        out = prim(new_indices, updates, new_shape)
        real_out = P.Reshape()(out, out_shape)
        return real_out, 0

    return vmap_rule


@vmap_rules_getters.register(P.ScatterAdd)
@vmap_rules_getters.register(P.ScatterMul)
@vmap_rules_getters.register(P.ScatterMin)
@vmap_rules_getters.register(P.ScatterMax)
@vmap_rules_getters.register(P.ScatterDiv)
@vmap_rules_getters.register(P.ScatterNdAdd)
@vmap_rules_getters.register(P.ScatterNdSub)
@vmap_rules_getters.register(P.ScatterNdMin)
@vmap_rules_getters.register(P.ScatterNdMax)
@vmap_rules_getters.register(P.array_ops.ScatterNdMul)
@vmap_rules_getters.register(P.ScatterNdDiv)
@vmap_rules_getters.register(P.ScatterNdUpdate)
@vmap_rules_getters.register(P.ScatterUpdate)
def get_scatter_op_vmap_rule(prim, axis_size):
    """
    VmapRule for `Scatter*` operations, such as `ScatterAdd`, `ScatterNdAdd`, `ScatterMin` and `ScatterMax`.
    scatter_func_map: high-dimensional implementation for recording Scatter class operators
    and ScatterNd class operators.
    scatter_func_list: used to record all Scatter class operators.
    """
    scatter_func_map = {
        "ScatterAdd": P.ScatterNdAdd,
        "ScatterMul": P.array_ops.ScatterNdMul,
        "ScatterMin": P.ScatterNdMin,
        "ScatterMax": P.ScatterNdMax,
        "ScatterDiv": P.ScatterNdDiv,
        "ScatterNdAdd": P.ScatterNdAdd,
        "ScatterNdSub": P.ScatterNdSub,
        "ScatterNdMin": P.ScatterNdMin,
        "ScatterNdMax": P.ScatterNdMax,
        "ScatterNdMul": P.array_ops.ScatterNdMul,
        "ScatterNdDiv": P.ScatterNdDiv,
        "ScatterNdUpdate": P.ScatterNdUpdate,
        "ScatterUpdate": P.ScatterNdUpdate
    }
    scatter_func_list = ["ScatterAdd", "ScatterMul", "ScatterMin", "ScatterMax", "ScatterDiv", "ScatterUpdate"]
    if isinstance(prim, str):
        prim_name = prim
        prim = Primitive(prim)
        use_locking = False
    else:
        prim_name = prim.name
        use_locking = prim.use_locking

    scatter_func = scatter_func_map.get(prim_name)(use_locking)
    concat = P.Concat(-1)

    def vmap_rule(ref_bdim, indices_bdim, updates_bdim, u_monad):
        ref, ref_dim = ref_bdim
        indices, indices_dim = indices_bdim
        updates, updates_dim = updates_bdim

        if ref_dim is None:
            if indices_dim is not None or updates_dim is not None:
                _raise_value_error("The source axis of `ref` is None, but the source axis of "
                                   "`indices` or `updates` is not None. The execution order of "
                                   "operator `{}` cannot be guaranteed.".format(prim_name))
            out = prim(ref, indices, updates, u_monad)
        elif ref_dim == 0:
            indices = _bdim_at_front(indices, indices_dim, axis_size)
            updates = _bdim_at_front(updates, updates_dim, axis_size)
            if prim_name in scatter_func_list:
                indices = F.expand_dims(indices, -1)

            indices_shape = F.shape(indices)
            prefix = _get_prefix(indices_shape, axis_size, F.dtype(indices))
            indices = concat((prefix, indices))
            out = scatter_func(ref, indices, updates, u_monad)
        else:
            _raise_value_error("The source axis of `ref` in `{}` must be 0 or None, "
                               "but got {}.".format(prim_name, ref_dim))
            out = None
        return out, ref_dim

    return vmap_rule


@vmap_rules_getters.register(G.SliceGrad)
def get_slice_grad_vmap_rule(prim, axis_size):
    """VmapRule for `SliceGrad` operation."""
    if isinstance(prim, str):
        prim_name = prim
        prim = Primitive(prim)
    else:
        prim_name = prim.name

    def vmap_rule(dy_bdim, x_bdim, begin_bdim, size_bdim):
        is_all_none, result = vmap_general_preprocess(prim, dy_bdim, x_bdim, begin_bdim, size_bdim)
        if is_all_none:
            return result

        dy, dy_dim = dy_bdim
        x, x_dim = x_bdim
        begin, begin_dim = begin_bdim
        size, size_dim = size_bdim

        if begin_dim is not None:
            _raise_value_error("The source axis of `begin` in {} only supports None currently, "
                               "but got {}.".format(prim_name, begin_dim))
        if size_dim is not None:
            _raise_value_error("The source axis of `size` in {} must be None, but got {}.".format(prim_name, size_dim))

        dy = _bdim_at_front(dy, dy_dim, axis_size)
        x = _bdim_at_front(x, x_dim, axis_size)

        batch_begin = (0,) + begin
        batch_size = (axis_size,) + size

        out = prim(dy, x, batch_begin, batch_size)

        return out, 0

    return vmap_rule


@vmap_rules_getters.register(P.TensorScatterAdd)
@vmap_rules_getters.register(P.TensorScatterSub)
@vmap_rules_getters.register(P.TensorScatterMul)
@vmap_rules_getters.register(P.TensorScatterDiv)
@vmap_rules_getters.register(P.TensorScatterMax)
def get_tensor_scatter_op_vmap_rule(prim, axis_size):
    """
    VmapRule for `TensorScatter*` operations, such as `TensorScatterMul`.
    tensor_scatter_func_map: TensorScatter implementation for recording TensorScatter class operators.
    """
    tensor_scatter_func_map = {
        "TensorScatterAdd": P.TensorScatterAdd,
        "TensorScatterSub": P.TensorScatterSub,
        "TensorScatterMul": P.TensorScatterMul,
        "TensorScatterDiv": P.TensorScatterDiv,
        "TensorScatterMax": P.TensorScatterMax,
    }
    if isinstance(prim, str):
        prim_name = prim
        prim = Primitive(prim)
    else:
        prim_name = prim.name

    tensor_scatter_func = tensor_scatter_func_map.get(prim_name)()
    concat = P.Concat(-1)

    def vmap_rule(input_x_bdim, indices_bdim, updates_bdim):
        is_all_none, result = vmap_general_preprocess(prim, input_x_bdim, indices_bdim, updates_bdim)
        if is_all_none:
            return result
        input_x, input_x_dim = input_x_bdim
        indices, indices_dim = indices_bdim
        updates, updates_dim = updates_bdim

        input_x = _bdim_at_front(input_x, input_x_dim, axis_size)
        indices = _bdim_at_front(indices, indices_dim, axis_size)
        updates = _bdim_at_front(updates, updates_dim, axis_size)

        indices_shape = F.shape(indices)
        prefix = _get_prefix(indices_shape, axis_size, F.dtype(indices))
        indices = concat((prefix, indices))
        out = tensor_scatter_func(input_x, indices, updates)
        return out, input_x_dim

    return vmap_rule


@vmap_rules_getters.register(P.UnsortedSegmentMin)
@vmap_rules_getters.register(P.UnsortedSegmentMax)
@vmap_rules_getters.register(P.UnsortedSegmentProd)
@vmap_rules_getters.register(P.UnsortedSegmentSum)
def get_unsorted_segment_arithmetic_vmap_rule(prim, axis_size):
    """VmapRule for `UnsortedSegment*` operation."""

    unsorted_segment_func_map = {
        "UnsortedSegmentMin": P.UnsortedSegmentMin,
        "UnsortedSegmentMax": P.UnsortedSegmentMax,
        "UnsortedSegmentProd": P.UnsortedSegmentProd,
        "UnsortedSegmentSum": P.UnsortedSegmentSum,
    }
    prim_name = prim.name
    unsorted_segment_func = unsorted_segment_func_map.get(prim_name)()

    if hasattr(prim, 'batch_rank'):
        batch_rank = prim.batch_rank + 1
    else:
        batch_rank = 1

    unsorted_segment_func.add_prim_attr('batch_rank', batch_rank)

    def vmap_rule(input_bdim, segment_ids_bdim, num_segment_bdim):
        is_all_none, result = vmap_general_preprocess(prim, input_bdim, segment_ids_bdim, num_segment_bdim)
        if is_all_none:
            return result

        # num_segment affect output shape, must be none
        num_segment, num_segment_dim = num_segment_bdim
        if num_segment_dim is not None:
            _raise_value_error("The source axis of `num_segment` in `{}` must be None, "
                               "but got {}.".format(prim_name, num_segment_dim))

        input_value, input_dim = input_bdim
        segment_ids, segment_ids_dim = segment_ids_bdim

        input_value = _bdim_at_front(input_value, input_dim, axis_size)
        segment_ids = _bdim_at_front(segment_ids, segment_ids_dim, axis_size)

        out = unsorted_segment_func(input_value, segment_ids, num_segment)
        return out, 0

    return vmap_rule


@vmap_rules_getters.register(P.Fill)
def get_fill_vmap_rule(prim, axis_size):
    """VmapRule for `Fill` operation."""
    if isinstance(prim, str):
        prim = Primitive(prim)
    cast_op = P.Cast()

    def vmap_rule(dtype_bdim, shape_bdim, value_bdim):
        is_all_none, result = vmap_general_preprocess(prim, dtype_bdim, shape_bdim, value_bdim)
        if is_all_none:
            return result
        dtype, type_dim = dtype_bdim
        if type_dim is not None:
            _raise_value_error("The source axis of `type` in `P.Fill` must be None, but got {}.".format(type_dim))
        value_shape, shape_dim = shape_bdim
        if shape_dim is not None:
            _raise_value_error("The source axis of `shape` in `P.Fill` must be None, but got {}.".format(shape_dim))
        value, vdim = value_bdim
        value_rank = F.rank(value)
        if value_rank != 1 or vdim != 0:
            _raise_value_error("The `value` in `P.Fill` must be constant value, thus the value only "
                               "can be rank: 1 with source axis: 0 in vmap scope, but got value rank: "
                               "{} with source axis: {}.".format(value_rank, vdim))
        value = cast_op(value, dtype)
        value = F.reshape(value, (axis_size,) + (1,) * len(value_shape))
        out = P.BroadcastTo((axis_size,) + value_shape)(value)
        return out, 0

    return vmap_rule


@vmap_rules_getters.register(P.FillV2)
def get_fill_v2_vmap_rule(prim, axis_size):
    """VmapRule for `FillV2` operation."""
    if isinstance(prim, str):
        prim = Primitive(prim)

    def vmap_rule(shape_bdim, value_bdim):
        is_all_none, result = vmap_general_preprocess(prim, shape_bdim,
                                                      value_bdim)
        if is_all_none:
            return result
        value_shape, shape_dim = shape_bdim
        if isinstance(value_shape, (Tensor_, Tensor)):
            _raise_value_error(
                "For `P.FillV2`, when the `shape` is a tensor, VMAP is not supported!"
            )
        if shape_dim is not None:
            _raise_value_error(
                "The source axis of `shape` in `P.FillV2` must be None, but got {}."
                .format(shape_dim))
        value, vdim = value_bdim
        value_rank = F.rank(value)
        if value_rank != 1 or vdim != 0:
            _raise_value_error(
                "The `value` in `P.FillV2` must be constant value, thus the value only "
                "can be rank: 1 with source axis: 0 in vmap scope, but got value rank: "
                "{} with source axis: {}.".format(value_rank, vdim))
        value = F.reshape(value, (axis_size,) + (1,) * len(value_shape))
        out = P.BroadcastTo((axis_size,) + value_shape)(value)
        return out, 0

    return vmap_rule


@vmap_rules_getters.register(Fills)
def get_fills_vmap_rule(prim, axis_size):
    """VmapRule for `Fills` operation."""
    if isinstance(prim, str):
        prim = Primitive(prim)
    cast_op = P.Cast()

    def vmap_rule(x_bdim, value_bdim):
        is_all_none, result = vmap_general_preprocess(prim, x_bdim, value_bdim)
        if is_all_none:
            return result
        x, x_batch_dim = x_bdim
        value, value_batch_dim = value_bdim
        out_type = x.dtype
        out_shape = x.shape
        value = cast_op(value, out_type)
        if value_batch_dim is None:
            out = P.BroadcastTo(out_shape)(value)
            return out, x_batch_dim
        value_rank = F.rank(value)
        if value_rank != 1 or value_batch_dim != 0:
            _raise_value_error("The `value` in `F.fills` only accept scalar or 0-dims tensor, thus the value only "
                               "can be rank: 1 with source axis: 0 in vmap scope, but got value rank: "
                               "{} with source axis: {}.".format(value_rank, value_batch_dim))
        if x_batch_dim is None:
            value = F.reshape(value, (axis_size,) + (1,) * len(out_shape))
            out = P.BroadcastTo((axis_size,) + out_shape)(value)
        else:
            x = _bdim_at_front(x, x_batch_dim, axis_size)
            out_shape = x.shape
            value = F.reshape(value, (axis_size,) + (1,) * (len(out_shape) - 1))
            out = P.BroadcastTo(out_shape)(value)
        return out, 0

    return vmap_rule


@vmap_rules_getters.register(P.Range)
def get_range_vmap_rule(prim, axis_size):
    """VmapRule for `Range` operation."""
    if isinstance(prim, str):
        prim = Primitive(prim)

    def vmap_rule(start_bdim, limit_bdim, delta_bdim):
        is_all_none, result = vmap_general_preprocess(prim, start_bdim, limit_bdim, delta_bdim)
        if not is_all_none:
            _, start_dim = start_bdim
            _, limit_dim = limit_bdim
            _, delta_dim = delta_bdim
            _raise_value_error("For operator Range, all axis for inputs should be None, but got start_dim: {},"
                               " limit_dim: {} and delta_dim: {}.".format(start_dim, limit_dim, delta_dim))
        return result

    return vmap_rule


@vmap_rules_getters.register(P.UniqueWithPad)
def get_unique_with_pad_vmap_rule(prim, axis_size):
    """VmapRule for `UniqueWithPad` operations.
    if isinstance(prim, str):
        prim = P.UniqueWithPad()

    prim_vmap = _VmapGeneralRule(prim, axis_size)

    def vmap_rule(x_bdim, pad_num_bdim):
        return prim_vmap(x_bdim, pad_num_bdim)

    return vmap_rule
    """
    if hasattr(prim, 'batch_rank'):
        batch_rank = prim.batch_rank + 1
    else:
        batch_rank = 1

    batch_prim = _vmap_clone_prim(prim)
    batch_prim.add_prim_attr("batch_rank", batch_rank)

    def vmap_rule(x_bdim, pad_num_bdim):
        x, x_dim = x_bdim
        pad_num, pad_num_dim = pad_num_bdim

        x = _bdim_at_front(x, x_dim, axis_size)
        pad_num = _bdim_at_front(pad_num, pad_num_dim, axis_size)
        y, idx = batch_prim(x, pad_num)
        return (y, 0), (idx, 0)

    return vmap_rule


@vmap_rules_getters.register(P.array_ops.MatrixDiagV3)
def get_matrix_diag_v3_vmap_rule(prim, axis_size):
    """VmapRule for `MatrixDiagV3` operation."""
    if isinstance(prim, str):
        prim_name = prim
        prim = P.array_ops.MatrixDiagV3()
    else:
        prim_name = prim.name

    def vmap_rule(x_bdim, k_bdim, num_rows_bdim, num_cols_bdim, padding_value_bdim):
        is_all_none, result = vmap_general_preprocess(prim, x_bdim, k_bdim, num_rows_bdim, num_cols_bdim,
                                                      padding_value_bdim)
        if is_all_none:
            return result

        x, x_dim = x_bdim
        k, k_dim = k_bdim
        num_rows, num_rows_dim = num_rows_bdim
        num_cols, num_cols_dim = num_cols_bdim
        padding_value, padding_value_dim = padding_value_bdim
        if k_dim is not None:
            _raise_value_error("The source axis of `k` in {} must be None, but got {}.".format(prim_name, k_dim))
        if num_rows_dim is not None:
            _raise_value_error(
                "The source axis of `num_rows` in {} must be None, but got {}.".format(prim_name, num_rows_dim))
        if num_cols_dim is not None:
            _raise_value_error(
                "The source axis of `num_cols` in {} must be None, but got {}.".format(prim_name, num_cols_dim))
        if padding_value_dim is not None:
            _raise_value_error("The source axis of `padding_value` in {} must be None, "
                               "but got {}.".format(prim_name, padding_value_dim))

        x = _bdim_at_front(x, x_dim, axis_size)
        out = prim(x, k, num_rows, num_cols, padding_value)
        return out, 0

    return vmap_rule


@vmap_rules_getters.register(P.TensorShape)
def get_tensor_shape_vmap_rule(prim, axis_size):
    """VmapRule for `TensorShape` operation."""
    if isinstance(prim, str):
        prim = Primitive(prim)

    def vmap_rule(input_bdim):
        is_all_none, result = vmap_general_preprocess(prim, input_bdim)
        if is_all_none:
            return result

        input_x, x_dim = input_bdim
        sub_x = P.Unstack(x_dim)(input_x)[0]
        out = prim(sub_x)

        return out, None

    return vmap_rule


@constexpr
def _get_one_hot_vmap_axis(orig_axis, ndim, indices_dim):
    """Find vmap axis for OneHot."""
    if orig_axis >= 0 and indices_dim <= orig_axis:
        return orig_axis + 1, indices_dim
    if orig_axis == -1:
        if indices_dim == (ndim - 1):
            return ndim - 1, indices_dim + 1
        return orig_axis, indices_dim
    return orig_axis, indices_dim + 1


@vmap_rules_getters.register(P.OneHot)
def get_one_hot_vmap_rule(prim, axis_size):
    """VmapRule for `OneHot` operation."""
    if isinstance(prim, str):
        prim_name = prim
        prim = Primitive(prim)
    else:
        prim_name = prim.name

    axis = prim.axis

    def vmap_rule(indices_bdim, depth_bdim, on_value_bdim, off_value_bdim):
        is_all_none, result = vmap_general_preprocess(prim, indices_bdim, depth_bdim, on_value_bdim, off_value_bdim)
        if is_all_none:
            return result

        indices, indices_dim = indices_bdim
        depth, depth_dim = depth_bdim
        on_value, on_value_dim = on_value_bdim
        off_value, off_value_dim = off_value_bdim

        if depth_dim is not None:
            _raise_value_error(
                "The source axis of `depth` in {} must be None, but got {}.".format(prim_name, depth_dim))

        if on_value_dim is not None:
            _raise_value_error(
                "The source axis of `on_value` in {} must be None, but got {}.".format(prim_name, on_value_dim))

        if off_value_dim is not None:
            _raise_value_error(
                "The source axis of `off_value` in {} must be None, but got {}.".format(prim_name, off_value_dim))
        ndim = F.rank(indices)
        new_axis, new_bd = _get_one_hot_vmap_axis(axis, ndim, indices_dim)
        out = P.OneHot(new_axis)(indices, depth, on_value, off_value)

        return out, new_bd

    return vmap_rule


@vmap_rules_getters.register(P.MaskedSelect)
def get_masked_select_vmap_rule(prim, axis_size):
    """VmapRule for `MaskedSelect`."""

    def vmap_rule(x_bdim, mask_bdim):
        is_all_none, result = vmap_general_preprocess(prim, x_bdim, mask_bdim)
        if is_all_none:
            return result

        x, x_dim = x_bdim
        mask, mask_dim = mask_bdim
        if mask_dim is not None:
            _raise_value_error("The source axis of `mask` in `P.MaskedSelect` must be None, "
                               "but got {}.".format(mask_dim))

        x = _bdim_at_front(x, x_dim, axis_size)
        mask = _bdim_at_front(mask, mask_dim, axis_size)
        x_shape = F.shape(x)
        mask_shape = F.shape(mask)
        x = _handle_broadcasting(x, x_shape, mask_shape)
        out = prim(x, mask)
        x_rank = F.rank(x)
        if x_rank > 1:
            out = F.reshape(out, (x_shape[0], -1))
        return out, 0

    return vmap_rule


@vmap_rules_getters.register(MaskedSelectGrad)
def get_masked_select_grad_vmap_rule(prim, axis_size):
    """VmapRule for `MaskedSelect`."""

    def vmap_rule(x_bdim, mask_bdim, outgrad_bdim):
        is_all_none, result = vmap_general_preprocess(prim, x_bdim, mask_bdim, outgrad_bdim)
        if is_all_none:
            return result

        x, x_dim = x_bdim
        mask, mask_dim = mask_bdim
        outgrad, outgrad_dim = outgrad_bdim
        if mask_dim is not None:
            _raise_value_error("The source axis of `mask` in `P.MaskedSelect` must be None, "
                               "but got {}.".format(mask_dim))

        x = _bdim_at_front(x, x_dim, axis_size)
        mask = _bdim_at_front(mask, mask_dim, axis_size)
        outgrad = _bdim_at_front(outgrad, outgrad_dim, axis_size)
        outgrad_shape = F.shape(outgrad)
        outgrad = F.reshape(outgrad, (outgrad_shape[0] * outgrad_shape[1],))
        x_grad = prim(x, mask, outgrad)
        return x_grad, 0

    return vmap_rule


@vmap_rules_getters.register(P.array_ops.MatrixBandPart)
def get_matrix_band_part_vmap_rule(prim, axis_size):
    """VmapRule for `MatrixBandPart` operation."""
    if isinstance(prim, str):
        prim = Primitive(prim)

    if hasattr(prim, 'batch_rank'):
        batch_rank = prim.batch_rank + 1
    else:
        batch_rank = 1

    batch_prim = P.array_ops.MatrixBandPart()
    batch_prim.add_prim_attr('batch_rank', batch_rank)

    def vmap_rule(x_bdim, lower_bdim, upper_bdim):
        is_all_none, result = vmap_general_preprocess(prim, x_bdim, lower_bdim, upper_bdim)
        if is_all_none:
            return result

        x, x_dim = x_bdim
        lower, lower_dim = lower_bdim
        upper, upper_dim = upper_bdim

        x = _bdim_at_front(x, x_dim, axis_size)
        lower = _bdim_at_front(lower, lower_dim, axis_size)
        upper = _bdim_at_front(upper, upper_dim, axis_size)

        out = batch_prim(x, lower, upper)
        return out, 0

    return vmap_rule


@vmap_rules_getters.register(P.array_ops.MatrixDiagPartV3)
def get_matrix_diag_part_v3_vmap_rule(prim, axis_size):
    """VmapRule for `MatrixBandPart` operation."""
    if isinstance(prim, str):
        prim_name = prim
        align = "RIGHT_LEFT"
    else:
        prim_name = prim.name
        align = prim.align

    matrix_diag_part = P.array_ops.MatrixDiagPartV3(align=align)

    def vmap_rule(x_bdim, k_bdim, padding_value_bdim):
        is_all_none, result = vmap_general_preprocess(prim, x_bdim, k_bdim, padding_value_bdim)
        if is_all_none:
            return result

        x, x_dim = x_bdim
        k, k_dim = k_bdim
        padding_value, padding_value_dim = padding_value_bdim
        if k_dim is not None:
            _raise_value_error("The source axis of `k` in {} must be None, but got {}.".format(prim_name, k_dim))
        if padding_value_dim is not None:
            _raise_value_error("The source axis of `padding_value` in {} must be None, "
                               "but got {}.".format(prim_name, padding_value_dim))

        x = _bdim_at_front(x, x_dim, axis_size)
        out = matrix_diag_part(x, k, padding_value)

        return out, 0

    return vmap_rule


@vmap_rules_getters.register(P.array_ops.MatrixSetDiagV3)
def get_matrix_set_diag_v3_vmap_rule(prim, axis_size):
    """VmapRule for `MatrixSetDiagV3` operation."""
    if isinstance(prim, str):
        prim_name = prim
        align = "RIGHT_LEFT"
    else:
        prim_name = prim.name
        align = prim.align

    matrix_set_diag_op = P.array_ops.MatrixSetDiagV3(align=align)

    def vmap_rule(x_bdim, diagonal_bdim, k_bdim):
        is_all_none, result = vmap_general_preprocess(prim, x_bdim, diagonal_bdim, k_bdim)
        if is_all_none:
            return result

        x, x_dim = x_bdim
        k, k_dim = k_bdim
        diagonal, diagonal_dim = diagonal_bdim
        if k_dim is not None:
            _raise_value_error("The source axis of `k` in {} must be None, but got {}.".format(prim_name, k_dim))

        x = _bdim_at_front(x, x_dim, axis_size)
        diagonal = _bdim_at_front(diagonal, diagonal_dim, axis_size)
        out = matrix_set_diag_op(x, diagonal, k)

        return out, 0

    return vmap_rule


@vmap_rules_getters.register(P.Padding)
def get_padding_vmap_rule(prim, axis_size):
    """VmapRule for `Padding` operation."""
    if isinstance(prim, str):
        prim = Primitive(prim)

    def vmap_rule(x_bdim):
        is_all_none, result = vmap_general_preprocess(prim, x_bdim)
        if is_all_none:
            return result

        x, x_dim = x_bdim
        if F.rank(x) and x_dim in (-1, F.rank(x) - 1):
            x = _bdim_at_front(x, x_dim, axis_size)
            output = prim(x)
            return output, 0
        output = prim(x)
        return output, x_dim

    return vmap_rule


@vmap_rules_getters.register(P.Ger)
def get_ger_vmap_rule(prim, axis_size):
    """VmapRule for `Ger`."""
    if hasattr(prim, 'batch_rank'):
        batch_rank = prim.batch_rank + 1
    else:
        batch_rank = 1

    batch_prim = P.Ger()
    batch_prim.add_prim_attr('batch_rank', batch_rank)

    def vmap_rule(x1_bdim, x2_bdim):
        is_all_none, result = vmap_general_preprocess(prim, x1_bdim, x2_bdim)
        if is_all_none:
            return result

        x1, x1_dim = x1_bdim
        x2, x2_dim = x2_bdim
        x1 = _bdim_at_front(x1, x1_dim, axis_size)
        x2 = _bdim_at_front(x2, x2_dim, axis_size)
        out = batch_prim(x1, x2)
        return out, 0

    return vmap_rule


@vmap_rules_getters.register(P.GatherD)
def get_gatherd_vmap_rule(prim, axis_size):
    """VmapRule for GatherD operations."""
    if isinstance(prim, str):
        prim = Primitive(prim)

    def vmap_rule(x_bdim, dim_bdim, index_bdim):
        is_all_none, result = vmap_general_preprocess(prim, x_bdim, dim_bdim, index_bdim)
        if is_all_none:
            return result

        x, x_dim = x_bdim
        dim_value, axis_dim = dim_bdim
        index, index_dim = index_bdim

        # `dim` will be a Tensor in dynamic shape case, do not support its vamp.
        if axis_dim is not None:
            _raise_value_error("The source axis of `dim` in `GatherD` must be None, "
                               "but got {}.".format(axis_dim))
        if not isinstance(dim_value, int):
            _raise_value_error("The `dim` in `GatherD` must be a int, but got {}.".format(dim_value))

        out_dim = index_dim

        # Broadcast if needed.
        if x_dim is None:
            x = _broadcast_by_axis(x, index_dim, axis_size)
        elif index_dim is None:
            index = _broadcast_by_axis(index, x_dim, axis_size)
            out_dim = x_dim
        elif x_dim != index_dim:
            mnp.moveaxis(x, x_dim, index_dim)

        # Adapt `dim` to vmap case.
        x_ndim = ops.rank(x)
        dim_value = _get_reduce_batch_axis(dim_value, x_dim, x_ndim)

        out = prim(x, dim_value, index)
        return out, out_dim

    return vmap_rule


@vmap_rules_getters.register(G.GatherDGradV2)
def get_gatherd_grad_v2_vmap_rule(prim, axis_size):
    """VmapRule for GatherDGradV2 operations."""
    if isinstance(prim, str):
        prim = Primitive(prim)

    def _update_dim(dim, x_rank, batch_dim):
        pdim = dim
        if pdim < 0:
            pdim += x_rank
        if pdim < 0 or pdim >= x_rank:
            _raise_value_error(
                "The `dim` in `GatherDGradV2` must be in range [{}, {}], but got {}.".format(-x_rank, x_rank - 1, dim))
        if pdim >= batch_dim:
            return pdim + 1
        if dim < 0:
            return pdim
        return dim

    def vmap_rule(x_bdim, dim_bdim, index_bdim, grad_bdim):
        is_all_none, result = vmap_general_preprocess(prim, x_bdim, dim_bdim, index_bdim, grad_bdim)
        if is_all_none:
            return result

        x, x_dim = x_bdim
        dim, dim_dim = dim_bdim
        if dim_dim is not None:
            _raise_value_error("The dim of 'dim' in `GatherDGradV2` must be None, but got {}.".format(dim_dim))
        index, index_dim = index_bdim
        grad, grad_dim = grad_bdim
        batch_dim = 0
        if x_dim is not None:
            batch_dim = x_dim
        elif index_dim is not None:
            batch_dim = index_dim
        elif grad_dim is not None:
            batch_dim = grad_dim

        x = _bdim_at_any(x, x_dim, batch_dim, axis_size)
        index = _bdim_at_any(index, index_dim, batch_dim, axis_size)
        grad = _bdim_at_any(grad, grad_dim, batch_dim, axis_size)
        x_rank = F.rank(x) - 1
        # Adjust dim if needed
        dim = _update_dim(dim, x_rank, batch_dim)
        out = prim(x, dim, index, grad)
        return (out, batch_dim)

    return vmap_rule


@vmap_rules_getters.register(P.SpaceToBatchND)
def get_space_to_batch_nd_vmap_rule(prim, axis_size):
    """VmapRule for `SpaceToBatchND`."""

    def vmap_rule(input_xdim):
        is_all_none, result = vmap_general_preprocess(prim, input_xdim)
        if is_all_none:
            return result

        x, x_dim = input_xdim
        x_trans = mnp.moveaxis(x, x_dim, 1)
        out = prim(x_trans)
        return out, 1

    return vmap_rule


@vmap_rules_getters.register(P.BatchToSpaceND)
def get_batch_to_space_nd_vmap_rule(prim, axis_size):
    """VmapRule for `BatchToSpaceND`."""

    def vmap_rule(input_xdim):
        is_all_none, result = vmap_general_preprocess(prim, input_xdim)
        if is_all_none:
            return result

        x, x_dim = input_xdim
        x_trans = mnp.moveaxis(x, x_dim, 1)
        out = prim(x_trans)
        return out, 1

    return vmap_rule


@vmap_rules_getters.register(P.GatherNd)
def get_gather_nd_vmap_rule(prim, axis_size):
    """VmapRule for GatherND operations."""
    if isinstance(prim, str):
        prim = P.GatherNd()
    concat = P.Concat(-1)

    def vmap_rule(x_bdim, indices_bdim):
        is_all_none, result = vmap_general_preprocess(prim, x_bdim, indices_bdim)
        if is_all_none:
            return result
        x, x_dim = x_bdim
        indices, indices_dim = indices_bdim
        x = _bdim_at_front(x, x_dim, axis_size)
        indices = _bdim_at_front(indices, indices_dim, axis_size)
        indices_shape = F.shape(indices)
        prefix = _get_prefix(indices_shape, axis_size, F.dtype(indices))
        indices = concat((prefix, indices))
        out = prim(x, indices)
        return out, 0

    return vmap_rule


@vmap_rules_getters.register(P.Meshgrid)
def get_meshgrid_vmap_rule(prim, axis_size):
    """VmapRule for `P.Meshgrid` operation."""
    if isinstance(prim, str):
        prim = Primitive(prim)
    indexing = prim.indexing

    def vmap_rule(*inputs_bdim):
        is_all_none, result = vmap_general_preprocess(prim, *inputs_bdim)
        if is_all_none:
            return result

        if not isinstance(inputs_bdim, (tuple)):
            _raise_value_error("The inputs of P.Meshgrid is not tuple.")
        args = inputs_bdim[0]
        if len(args) <= 1:
            _raise_value_error(
                "The input number of P.Meshgrid must be greater than 1.")

        output_shape = []
        ones_shape = []
        for each_arg in args:
            x, bdim = each_arg
            if bdim is None:
                _raise_value_error(
                    "For Meshgrid vmap, the axis of each input must be same.")
            x = _bdim_at_front(x, bdim, axis_size)
            if F.rank(x) != 2:
                _raise_value_error(
                    "Each input of Meshgrid must be 1D, but got {}.".format(F.rank(x) - 1))
            output_shape.append(F.shape(x)[-1])
            ones_shape.append(1)
        output_shape.insert(0, axis_size)
        ones_shape.insert(0, axis_size)

        if indexing == "xy":
            output_shape[1], output_shape[2] = output_shape[2], output_shape[1]
        shape = tuple(output_shape)

        input_0, _ = args[0]
        dtype = F.dtype(input_0)
        ones_tensor = F.fill(dtype, shape, 1)

        index = 0
        vals_out_tuple = ()
        for each_arg in args:
            x, bdim = each_arg
            x = _bdim_at_front(x, bdim, axis_size)
            shape_index = (1 - index) if (index <= 1 and indexing == "xy") else index
            ones_shape[shape_index + 1] = output_shape[shape_index + 1]
            x = P.Reshape()(x, tuple(ones_shape))
            output = P.Mul()(x, ones_tensor)
            vals_out_tuple = vals_out_tuple + ((output, 0),)
            ones_shape[shape_index + 1] = 1
            index = index + 1

        return vals_out_tuple

    return vmap_rule


@vmap_rules_getters.register(P.MaskedFill)
def get_masked_fill_vmap_rule(prim, axis_size):
    """VmapRule for `MaskedFill` operation."""
    if hasattr(prim, 'batch_rank'):
        batch_rank = prim.batch_rank + 1
    else:
        batch_rank = 1

    batch_prim = P.MaskedFill()
    batch_prim.add_prim_attr('batch_rank', batch_rank)

    def vmap_rule(input_bdim, mask_bdim, value_bdim):
        is_all_none, result = vmap_general_preprocess(prim, input_bdim, mask_bdim, value_bdim)
        if is_all_none:
            return result

        input_x, x_dim = input_bdim
        mask, mask_dim = mask_bdim
        value, value_dim = value_bdim
        input_x = _bdim_at_front(input_x, x_dim, axis_size)
        mask = _bdim_at_front(mask, mask_dim, axis_size)
        value = _bdim_at_front(value, value_dim, axis_size)
        out = batch_prim(input_x, mask, value)
        return out, 0

    return vmap_rule


@vmap_rules_getters.register(P.Gather)
def get_gather_vmap_rule(prim, axis_size):
    """VmapRule for `Gather` operation. """
    if isinstance(prim, str):
        prim_name = prim
        prim = P.Gather()
    else:
        prim_name = prim.name

    @_primexpr
    def process_axis(axis, x_shape_size, has_xdim: bool, has_idim: bool):
        if has_xdim and has_idim:
            if axis < 0:
                axis = x_shape_size - 1 + axis
        elif has_xdim:
            if axis >= 0:
                axis = axis + 1
        else:
            if axis < 0:
                axis = x_shape_size + axis

        return axis

    @_primexpr
    def get_x_dst_shape(x_shape, axis):
        target_axis_size = x_shape[axis + 1]
        x_dst_shape = x_shape[0:axis] + (axis_size * target_axis_size,) + x_shape[axis + 2:]
        max_axis_size = axis_size * target_axis_size

        return target_axis_size, x_dst_shape, max_axis_size

    def vmap_rule(x_bdim, indices_bdim, axis_bdim):
        is_all_none, result = vmap_general_preprocess(prim, x_bdim, indices_bdim)
        if is_all_none:
            return result

        x, x_dim = x_bdim
        indices, indices_dim = indices_bdim
        axis, axis_dim = axis_bdim

        if axis_dim is not None:
            _raise_value_error("The source axis of `axis` in {} must be None, but got {}.".format(prim_name, axis_dim))

        x_shape_len = len(x.shape)

        if x_dim is not None and indices_dim is None:
            x = _bdim_at_front(x, x_dim, axis_size)
            axis = process_axis(axis, x_shape_len, True, False)
            output = prim(x, indices, axis)
            return output, 0

        if x_dim is None and indices_dim is not None:
            indices = _bdim_at_front(indices, indices_dim, axis_size)
            axis = process_axis(axis, x_shape_len, False, True)
            output = prim(x, indices, axis)
            return output, axis

        x = _bdim_at_front(x, x_dim, axis_size)
        indices = _bdim_at_front(indices, indices_dim, axis_size)

        axis = process_axis(axis, x_shape_len, True, True)

        x = mnp.moveaxis(x, 0, axis)

        x_shape = x.shape
        target_axis_size, x_dst_shape, max_axis_size = get_x_dst_shape(x_shape, axis)

        x = x.reshape(x_dst_shape)

        counts_shape = indices.shape
        counts = mnp.arange(0, axis_size, 1)
        counts = F.mul(counts, target_axis_size)
        counts = P.BroadcastTo(counts_shape[1:] + (axis_size,))(counts)
        counts = mnp.moveaxis(counts, -1, 0)

        indices_out_of_bound = mnp.where(indices > target_axis_size - 1, x=max_axis_size, y=0)

        indices = F.add(indices, counts)
        indices = F.add(indices, indices_out_of_bound)

        output = prim(x, indices, axis)

        return output, axis

    return vmap_rule


@vmap_rules_getters.register(TensorScatterElements)
def get_tensor_scatter_elements_vmap_rule(prim, axis_size):
    """VmapRule for TensorScatterElements operations."""
    if isinstance(prim, str):
        axis = 0
        reduction = 'none'
    else:
        axis = prim.axis
        reduction = prim.reduction

    def two_dims_are_none(i_bdim, j_no_dim, k_no_dim, axis_size):
        i, i_dim = i_bdim
        j = _broadcast_by_axis(j_no_dim, i_dim, axis_size)
        k = _broadcast_by_axis(k_no_dim, i_dim, axis_size)
        new_inputs = (i, j, k)
        return (new_inputs, i_dim)

    def one_dim_is_none(i_bdim, j_bdim, k_no_dim, axis_size):
        i, i_dim = i_bdim
        j, j_dim = j_bdim
        mnp.moveaxis(j, j_dim, i_dim)
        k = _broadcast_by_axis(k_no_dim, i_dim, axis_size)
        new_inputs = (i, j, k)
        return (new_inputs, i_dim)

    def no_dim_is_none(i_bdim, j_bdim, k_bdim):
        i, i_dim = i_bdim
        j, j_dim = j_bdim
        k, k_dim = k_bdim
        mnp.moveaxis(j, j_dim, i_dim)
        mnp.moveaxis(k, k_dim, i_dim)
        new_inputs = (i, j, k)
        return new_inputs, i_dim

    def vmap_rule(x_bdim, index_bdim, update_bdim):
        is_all_none, result = vmap_general_preprocess(
            prim, x_bdim, index_bdim, update_bdim)
        if is_all_none:
            return result

        x, x_dim = x_bdim
        index, index_dim = index_bdim
        update, update_dim = update_bdim

        numbers = [x_dim, index_dim, update_dim].count(None)
        if numbers == 2:
            if x_dim is not None:
                inputs, out_dim = two_dims_are_none(
                    x_bdim, index, update, axis_size)
                x, index, update = inputs
            elif index_dim is not None:
                inputs, out_dim = two_dims_are_none(
                    index_bdim, x, update, axis_size)
                index, x, update = inputs
            else:
                inputs, out_dim = two_dims_are_none(
                    update_bdim, x, index, axis_size)
                update, x, index = inputs
        elif numbers == 1:
            if x_dim is None:
                inputs, out_dim = one_dim_is_none(
                    index_bdim, update_bdim, x, axis_size)
                index, update, x = inputs
            elif index_dim is None:
                inputs, out_dim = one_dim_is_none(
                    x_bdim, update_bdim, index, axis_size)
                x, update, index = inputs
            else:
                inputs, out_dim = one_dim_is_none(
                    x_bdim, index_bdim, update, axis_size)
                x, index, update = inputs
        else:
            inputs, out_dim = no_dim_is_none(x_bdim, index_bdim, update_bdim)
            x, index, update = inputs

        # Adapt `axis` to vmap case.
        new_axis = axis + 1 if axis >= out_dim else axis

        out = TensorScatterElements(new_axis, reduction)(x, index, update)
        return out, out_dim

    return vmap_rule


@vmap_rules_getters.register(IndexFill)
def get_index_fill_rule(prim, axis_size):
    """VmapRule for `IndexFill` operation."""
    prim_vmap = _VmapGeneralRule(prim, axis_size)

    def vmap_rule(x_bdim, dim_bdim, index_bdim, value_bdim):
        is_all_none, result = vmap_general_preprocess(prim, x_bdim, dim_bdim, index_bdim, value_bdim)
        if is_all_none:
            return result

        x, x_dim = x_bdim
        dim, dim_dim = dim_bdim
        index, index_dim = index_bdim
        value, value_dim = value_bdim
        if dim_dim is not None or index_dim is not None or value_dim is not None:
            return prim_vmap(x_bdim, dim_bdim, index_bdim, value_bdim)

        x = _bdim_at_front(x, x_dim, axis_size)
        new_dim = F.select(dim < 0, dim, dim + 1)
        out = prim(x, new_dim, index, value)
        return out, 0

    return vmap_rule


@vmap_rules_getters.register(P.DataFormatDimMap)
def get_data_format_dim_map_vmap_rule(prim, axis_size):
    """VmapRule for `DataFormatDimMap`"""
    if hasattr(prim, 'batch_rank'):
        batch_rank = prim.batch_rank + 1
    else:
        batch_rank = 1

    batch_prim = P.DataFormatDimMap()
    batch_prim.add_prim_attr('batch_rank', batch_rank)

    def vmap_rule(x_bdim):
        is_all_none, result = vmap_general_preprocess(prim, x_bdim)
        if is_all_none:
            return result
        x, x_dim = x_bdim
        x = _bdim_at_front(x, x_dim, axis_size)
        out = batch_prim(x)
        return out, 0

    return vmap_rule


@vmap_rules_getters.register(P.ExpandDims)
def get_expand_dims_vmap_rule(prim, axis_size):
    """VmapRule for `ExpandDims`."""

    @_primexpr
    def process_axis(axis, rank, x_dim):
        if axis < 0:
            axis += rank
        axis_processed = axis + 1 if x_dim <= axis else axis
        x_dim = x_dim if x_dim < axis_processed else x_dim + 1
        return axis_processed, x_dim

    def vmap_rule(x_bdim, axis_bdim):
        is_all_none, result = vmap_general_preprocess(prim, x_bdim, axis_bdim)
        if is_all_none:
            return result

        x, x_dim = x_bdim
        axis, axis_dim = axis_bdim
        rank = ops.rank(x)

        if axis_dim is not None:
            _raise_value_error("The source axis of shape in `ExpandDims` must be None, but got {}.".format(axis_dim))

        axis, x_dim = process_axis(axis, rank, x_dim)
        output = prim(x, axis)
        return output, x_dim

    return vmap_rule


@vmap_rules_getters.register(P.Diag)
def get_diag_vmap_rule(prim, axis_size):
    """VmapRule for `Diag` operations."""
    if hasattr(prim, 'batch_rank'):
        batch_rank = prim.batch_rank + 1
    else:
        batch_rank = 1

    batch_prim = P.Diag()
    batch_prim.add_prim_attr('batch_rank', batch_rank)

    def vmap_rule(x_bdim):
        is_all_none, result = vmap_general_preprocess(prim, x_bdim)
        if is_all_none:
            return result
        x, x_dim = x_bdim
        x = _bdim_at_front(x, x_dim, axis_size)
        out = batch_prim(x)
        return out, 0

    return vmap_rule


@vmap_rules_getters.register(P.Slice)
def get_slice_vmap_rule(prim, axis_size):
    """VmapRule for `Slice` operation."""
    if isinstance(prim, str):
        prim_name = prim
        prim = Primitive(prim)
    else:
        prim_name = prim.name

    def vmap_rule(x_bdim, begin_bdim, size_bdim):
        is_all_none, result = vmap_general_preprocess(prim, x_bdim, begin_bdim, size_bdim)
        if is_all_none:
            return result

        x, x_dim = x_bdim
        begin, begin_dim = begin_bdim
        size, size_dim = size_bdim

        if begin_dim is not None:
            _raise_value_error("The source axis of `begin` in {} only supports None currently, "
                               "but got {}.".format(prim_name, begin_dim))
        if size_dim is not None:
            _raise_value_error("The source axis of `size` in {} must be None, but got {}.".format(prim_name, size_dim))

        x = _bdim_at_front(x, x_dim, axis_size)

        batch_begin = (0,) + begin
        batch_size = (axis_size,) + size

        out = prim(x, batch_begin, batch_size)

        return out, 0

    return vmap_rule


@vmap_rules_getters.register(P.Squeeze)
def get_squeeze_vmap_rule(prim, axis_size):
    """VmapRule for `Squeeze`."""
    if hasattr(prim, 'axis'):
        prim_axis = prim.axis
    else:
        prim_axis = None

    @_primexpr
    def move_axis(axes):
        new_axis = ()
        for axis in axes:
            if axis < 0:
                new_axis = new_axis + (axis,)
            else:
                new_axis = new_axis + (axis + 1,)
        return new_axis

    @_primexpr
    def generate_all_axis_except_first(x_rank):
        new_axis = ()
        for i in range(1, x_rank, 1):
            new_axis += (i,)
        return new_axis

    def vmap_rule(x_bdim):
        is_all_none, result = vmap_general_preprocess(prim, x_bdim)
        if is_all_none:
            return result

        x, x_dim = x_bdim
        x = _bdim_at_front(x, x_dim, axis_size)

        if prim_axis is None:
            if axis_size == 1:
                new_axis = generate_all_axis_except_first(F.rank(x))
                batch_squeeze = P.Squeeze(axis=new_axis)
                out = batch_squeeze(x)
                return out, 0

            out = prim(x)
            return out, 0

        new_axis = move_axis(prim_axis)
        batch_squeeze = P.Squeeze(axis=new_axis)
        out = batch_squeeze(x)
        return out, 0

    return vmap_rule


@vmap_rules_getters.register(P.StridedSlice)
def get_stridedslice_vmap_rule(prim, axis_size):
    """VmapRule for `StridedSlice`."""
    new_begin_mask = prim.begin_mask * 2 + 1
    new_end_mask = prim.end_mask * 2 + 1
    new_ellipsis_mask = prim.ellipsis_mask
    new_new_axis_mask = prim.new_axis_mask * 2
    new_shrink_axis_mask = prim.shrink_axis_mask * 2
    batch_stridedslice = P.StridedSlice(new_begin_mask, new_end_mask, new_ellipsis_mask, new_new_axis_mask, \
                                        new_shrink_axis_mask)

    @_primexpr
    def get_new_begin_end_strided(begin, end, strided):
        new_begin = (0,) + begin
        new_end = (0,) + end
        new_strided = (1,) + strided
        return new_begin, new_end, new_strided

    def vmap_rule(x_bdim, begin_bdim, end_bdim, strided_bdim):
        is_all_none, result = vmap_general_preprocess(prim, x_bdim, begin_bdim, end_bdim, strided_bdim)
        if is_all_none:
            return result

        x, x_dim = x_bdim
        begin, begin_dim = begin_bdim
        end, end_dim = end_bdim
        strided, strided_dim = strided_bdim

        if any(dim is not None for dim in [begin_dim, end_dim, strided_dim]):
            _raise_value_error("vmap of `StridedSlice` not support `begin`, `end` or `strided` has batch dimension, "
                               "but got {}, {}, {}".format(begin_dim, end_dim, strided_dim))

        # x_dim is not None, and the others are None
        x = _bdim_at_front(x, x_dim, axis_size)
        new_begin, new_end, new_strided = get_new_begin_end_strided(begin, end, strided)
        result = batch_stridedslice(x, new_begin, new_end, new_strided)
        return result, 0

    return vmap_rule


@vmap_rules_getters.register(G.StridedSliceGrad)
def get_stridedslice_grad_vmap_rule(prim, axis_size):
    """VmapRule for `StridedSliceGrad`."""
    new_begin_mask = prim.begin_mask * 2 + 1
    new_end_mask = prim.end_mask * 2 + 1
    new_ellipsis_mask = prim.ellipsis_mask
    new_new_axis_mask = prim.new_axis_mask * 2
    new_shrink_axis_mask = prim.shrink_axis_mask * 2
    batch_stridedslice_grad = G.StridedSliceGrad(new_begin_mask, new_end_mask, new_ellipsis_mask, new_new_axis_mask, \
                                                 new_shrink_axis_mask)

    @_primexpr
    def get_new_xshape_begin_end_strided(xshape, begin, end, strided):
        new_xshape = (axis_size,) + xshape
        new_begin = (0,) + begin
        new_end = (axis_size,) + end
        new_strided = (1,) + strided
        return new_xshape, new_begin, new_end, new_strided

    def vmap_rule(dy_bdim, xshape_bdim, begin_bdim, end_bdim, strided_bdim):
        is_all_none, result = vmap_general_preprocess(prim, dy_bdim, xshape_bdim, begin_bdim, end_bdim, strided_bdim)
        if is_all_none:
            return result

        dy, dy_dim = dy_bdim
        xshape, xshape_dim = xshape_bdim
        begin, begin_dim = begin_bdim
        end, end_dim = end_bdim
        strided, strided_dim = strided_bdim

        if any(dim is not None for dim in [xshape_dim, begin_dim, end_dim, strided_dim]):
            _raise_value_error("vmap of `StridedSliceGrad` not support `xshape`, `begin`, "
                               "`end` or `strided` has batch dimension, "
                               "but got {}, {}, {}, {}".format(xshape_dim, begin_dim, end_dim, strided_dim))

        # dy_dim and x_dim are not None, and others are None
        dy = _bdim_at_front(dy, dy_dim, axis_size)

        new_xshape, new_begin, new_end, new_strided = get_new_xshape_begin_end_strided(xshape, begin, end, strided)

        result = batch_stridedslice_grad(dy, new_xshape, new_begin, new_end, new_strided)
        return result, 0

    return vmap_rule


@vmap_rules_getters.register(P.TopK)
def get_topk_vmap_rule(prim, axis_size):
    """VmapRule for `TopK` operation."""
    if isinstance(prim, str):
        prim_name = prim
        prim = Primitive(prim)
    else:
        prim_name = prim.name

    def vmap_rule(x_bdim, k_bdim):
        is_all_none, result = vmap_general_preprocess(prim, x_bdim, k_bdim)
        if is_all_none:
            return result

        x, x_dim = x_bdim
        k, k_dim = k_bdim

        if k_dim is not None:
            _raise_value_error("The source axis of `k` in {} must be None, but got {}.".format(prim_name, k_dim))

        if F.rank(x) and x_dim in (-1, F.rank(x) - 1):
            x = _bdim_at_front(x, x_dim, axis_size)
            values, indices = prim(x, k)
            return (values, 0), (indices, 0)

        values, indices = prim(x, k)
        return (values, x_dim), (indices, x_dim)

    return vmap_rule


@vmap_rules_getters.register(P.Im2Col)
def get_im2col_vmap_rule(prim, axis_size):
    """VmapRule for `Im2Col` operations."""
    if isinstance(prim, str):
        prim = Primitive(prim)

    def vmap_rule(x_bdim):
        is_all_none, result = vmap_general_preprocess(prim, x_bdim)
        if is_all_none:
            return result
        x, x_dim = x_bdim
        x = _bdim_at_front(x, x_dim, axis_size)
        x_shape = x.shape
        x_new_shape = (-1,) + x_shape[2:]
        x = x.reshape(x_new_shape)

        out = prim(x)
        out_shape = out.shape
        original_shape = x_shape[:2] + out_shape[1:]
        out = out.reshape(original_shape)
        return out, 0

    return vmap_rule


@vmap_rules_getters.register(P.Split)
def get_split_vmap_rule(prim, axis_size):
    """VmapRule for `Split`."""

    axis = prim.axis
    if axis >= 0:
        axis += 1
    batch_prim = P.Split(axis, prim.output_num)

    def vmap_rule(x_bdim):
        is_all_none, result = vmap_general_preprocess(prim, x_bdim)
        if is_all_none:
            return result
        x, x_dim = x_bdim
        x = _bdim_at_front(x, x_dim, axis_size)
        outputs = batch_prim(x)
        output = ()
        for out in outputs:
            output = output + ((out, 0),)
        return output

    return vmap_rule


get_unsupported_dynamic_vmap_rule = vmap_rules_getters.register(NonZero)(get_unsupported_dynamic_vmap_rule)
get_unsupported_dynamic_vmap_rule = vmap_rules_getters.register(P.Unique)(get_unsupported_dynamic_vmap_rule)
get_unsupported_dynamic_vmap_rule = \
    vmap_rules_getters.register(UniqueConsecutive)(get_unsupported_dynamic_vmap_rule)
get_unsupported_dynamic_vmap_rule = vmap_rules_getters.register(Col2Im)(get_unsupported_dynamic_vmap_rule)
get_unsupported_dynamic_vmap_rule = vmap_rules_getters.register(RandomPoisson)(get_unsupported_dynamic_vmap_rule)
get_unop_vmap_rule = vmap_rules_getters.register(P.ZerosLike)(get_unop_vmap_rule)
get_unop_vmap_rule = vmap_rules_getters.register(P.OnesLike)(get_unop_vmap_rule)
