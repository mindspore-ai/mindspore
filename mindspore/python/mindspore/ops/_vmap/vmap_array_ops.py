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

import mindspore.numpy as mnp
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops import constexpr
from ..primitive import Primitive
from .._vmap.vmap_base import vmap_rules_getters, vmap_general_preprocess, _bdim_at_front, _raise_value_error, \
    _handle_broadcasting
from ..operations.array_ops import Fills


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
        return (out, x_dim)

    return vmap_rule


@constexpr
def _get_transpose_batch_perm(dim, perm, x_rank):
    """Generate batch_perm based on the original perm of transpose operation and dim of the input."""
    if dim < 0:
        dim = dim + x_rank
    batch_perm = (dim,)
    for i in perm:
        if i < dim:
            batch_perm = batch_perm + (i,)
        else:
            index = i + 1
            batch_perm = batch_perm + (index,)
    return batch_perm


@constexpr
def _get_prefix_expand_shape(indices_shape):
    """Generate prefix and expand shape by indices shape."""
    indices_end = len(indices_shape) - 1
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
    return prefix_shape, expand_shape


@vmap_rules_getters.register(P.Transpose)
def get_transpose_vmap_rule(prim, axis_size):
    """VmapRule for `Transpose` operation."""
    if isinstance(prim, str):
        prim = Primitive(prim)

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
        return (out, 0)

    return vmap_rule


@constexpr
def _get_tile_shape(input_shape, multiples):
    input_ndim = len(input_shape)
    multiples_ndim = len(multiples)
    input_shape = (input_shape[0],) + (1,) * (multiples_ndim - input_ndim + 1) + input_shape[1:]
    multiples = (1,) * (input_ndim - multiples_ndim) + multiples
    input_expand_shape = (input_shape[0],) + tuple([j for i in input_shape[1:] for j in [1, i]])
    repeat_shape = (input_shape[0],) + tuple([k for pair in zip(multiples[1:], input_shape[1:]) for k in pair])
    output_shape = tuple([a * b for a, b in zip(input_shape, multiples)])
    return input_expand_shape, repeat_shape, output_shape


@vmap_rules_getters.register(P.Tile)
def get_tile_vmap_rule(prim, axis_size):
    """VmapRule for `P.Tile` operation."""
    if isinstance(prim, str):
        prim = Primitive(prim)

    def vmap_rule(input_bdim, multiples_bdim):
        is_all_none, result = vmap_general_preprocess(prim, input_bdim, multiples_bdim)
        if is_all_none:
            return result

        input_x, dim = input_bdim
        multiples, multiples_dim = multiples_bdim
        if multiples_dim is not None:
            _raise_value_error("The source axis of shape in `Tile` must be None, but got {}.".format(multiples_dim))

        input_x = _bdim_at_front(input_x, dim, axis_size)
        input_shape = F.shape(input_x)
        input_expand_shape, repeat_shape, output_shape = _get_tile_shape(input_shape, multiples)
        expand_input = F.reshape(input_x, input_expand_shape)
        repeat_tensor = P.BroadcastTo(repeat_shape)(expand_input)
        output = F.reshape(repeat_tensor, output_shape)
        return (output, 0)

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
        return (out, 0)

    return vmap_rule


@vmap_rules_getters.register(P.Reshape)
def get_reshape_vmap_rule(prim, axis_size):
    """VmapRule for `Reshape` operation."""

    def vmap_rule(operand_bdim, shape_bdim):
        is_all_none, result = vmap_general_preprocess(prim, operand_bdim, shape_bdim)
        if is_all_none:
            return result

        x, dim = operand_bdim
        shape, shape_dim = shape_bdim
        if shape_dim is not None:
            _raise_value_error("The source axis of shape in `Reshape` must be None, but got {}.".format(shape_dim))

        x = mnp.moveaxis(x, dim, 0)
        axis_size = F.shape(x)[0]
        batch_shape = (axis_size,) + shape

        out = prim(x, batch_shape)
        return (out, 0)

    return vmap_rule


@vmap_rules_getters.register(P.Select)
def get_select_vmap_rule(prim, axis_size):
    """VmapRule for 'Select' operation."""

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

        return (out, 0)

    return vmap_rule


@vmap_rules_getters.register(P.ScatterAdd)
@vmap_rules_getters.register(P.ScatterMin)
@vmap_rules_getters.register(P.ScatterMax)
@vmap_rules_getters.register(P.ScatterNdAdd)
@vmap_rules_getters.register(P.ScatterNdSub)
def get_scatter_add_vmap_rule(prim, axis_size):
    """
    VmapRule for `Scatter*` operations, such as `ScatterAdd`, `ScatterNdAdd`, `ScatterMin` and `ScatterMax`.
    sactter_func_map: high-dimensional implementation for recording Scatter class operators
    and ScatterNd class operators.
    sactter_func_list: used to record all Scatter class operators.
    """
    sactter_func_map = {
        "ScatterAdd": P.ScatterNdAdd,
        "ScatterMin": P.ScatterNdMin,
        "ScatterMax": P.ScatterNdMax,
        "ScatterNdAdd": P.ScatterNdAdd,
        "ScatterNdSub": P.ScatterNdSub,
    }
    sactter_func_list = ["ScatterAdd", "ScatterMin", "ScatterMax"]
    if isinstance(prim, str):
        prim_name = prim
        prim = Primitive(prim)
        use_locking = False
    else:
        prim_name = prim.name
        use_locking = prim.use_locking

    scatter_func = sactter_func_map.get(prim_name)(use_locking)
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
            if prim_name in sactter_func_list:
                indices = F.expand_dims(indices, -1)

            indices_shape = F.shape(indices)
            prefix_shape, expand_shape = _get_prefix_expand_shape(indices_shape)
            prefix = P.BroadcastTo(prefix_shape)(F.reshape(mnp.arange(axis_size), expand_shape))
            indices = concat((prefix, indices))
            out = scatter_func(ref, indices, updates, u_monad)
        else:
            _raise_value_error("The source axis of `ref` in `{}` must be 0 or None, "
                               "but got {}.".format(prim_name, ref_dim))
            out = None
        return (out, ref_dim)

    return vmap_rule


@vmap_rules_getters.register(P.TensorScatterAdd)
@vmap_rules_getters.register(P.TensorScatterSub)
@vmap_rules_getters.register(P.TensorScatterMul)
def get_tensor_scatter_op_vmap_rule(prim, axis_size):
    """
    VmapRule for `TensorScatter*` operations, such as `TensorScatterMul`.
    tensor_scatter_func_map: TensorScatter implementation for recording TensorScatter class operators.
    """
    tensor_scatter_func_map = {
        "TensorScatterAdd": P.TensorScatterAdd,
        "TensorScatterSub": P.TensorScatterSub,
        "TensorScatterMul": P.TensorScatterMul,
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
        prefix_shape, expand_shape = _get_prefix_expand_shape(indices_shape)
        prefix = P.BroadcastTo(prefix_shape)(F.reshape(mnp.arange(axis_size), expand_shape))
        indices = concat((prefix, indices))
        out = tensor_scatter_func(input_x, indices, updates)
        return out, input_x_dim

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
        return (out, 0)

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
        x, x_dim = x_bdim
        value, value_dim = value_bdim
        out_type = x.dtype
        out_shape = x.shape
        value = cast_op(value, out_type)
        if value_dim is None:
            out = P.BroadcastTo(out_shape)(value)
            return out, x_dim
        value = _bdim_at_front(value, value_dim, axis_size)
        if x_dim is None:
            value = F.reshape(value, (axis_size,) + (1,) * len(out_shape))
            out = P.BroadcastTo((axis_size,) + out_shape)(value)
        else:
            x = _bdim_at_front(x, x_dim, axis_size)
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

        return (out, None)

    return vmap_rule


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
        new_axis = axis
        if axis >= 0 and indices_dim <= axis:
            new_axis = axis + 1
        elif indices_dim == (ndim - 1):
            if axis in (-1, (ndim - 1)):
                new_axis = ndim - 1
        out = P.OneHot(new_axis)(indices, depth, on_value, off_value)

        return (out, indices_dim)

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
        return (out, 0)

    return vmap_rule


@vmap_rules_getters.register(P.Ger)
def get_ger_vmap_rule(prim, axis_size):
    """VmapRule for `Ger`."""

    def vmap_rule(x1_bdim, x2_bdim):
        is_all_none, result = vmap_general_preprocess(prim, x1_bdim, x2_bdim)
        if is_all_none:
            return result

        x1, x1_dim = x1_bdim
        x2, x2_dim = x2_bdim
        x1 = _bdim_at_front(x1, x1_dim, axis_size)
        x2 = _bdim_at_front(x2, x2_dim, axis_size)
        out = prim(x1, x2)
        return (out, 0)

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
        return (out, 1)

    return vmap_rule


@vmap_rules_getters.register(P.GatherNd)
def get_gather_nd_vmap_rule(prim, axis_size):
    """VmapRule for GatherND operations."""
    if isinstance(prim, str):
        prim = Primitive(prim)
    concat = P.Concat(-1)
    gather_nd = P.GatherNd()

    def vmap_rule(x_bdim, indices_bdim):
        is_all_none, result = vmap_general_preprocess(prim, x_bdim, indices_bdim)
        if is_all_none:
            return result
        x, x_dim = x_bdim
        indices, indices_dim = indices_bdim
        x = _bdim_at_front(x, x_dim, axis_size)
        indices = _bdim_at_front(indices, indices_dim, axis_size)
        indices_shape = F.shape(indices)
        prefix_shape, expand_shape = _get_prefix_expand_shape(indices_shape)
        prefix = P.BroadcastTo(prefix_shape)(F.reshape(mnp.arange(axis_size), expand_shape))
        indices = concat((prefix, indices))
        out = gather_nd(x, indices)
        out = prim(x, indices)
        return (out, 0)

    return vmap_rule
