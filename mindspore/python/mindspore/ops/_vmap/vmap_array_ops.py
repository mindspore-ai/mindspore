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
from .._vmap.vmap_base import vmap_rules_getters, vmap_general_preprocess, _bdim_at_front, _raise_value_error


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
        batch_perm = (dim,)
        for i in perm:
            if i < dim:
                batch_perm = batch_perm + (i,)
            else:
                index = i + 1
                batch_perm = batch_perm + (index,)

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


@vmap_rules_getters.register(P.ScatterAdd)
@vmap_rules_getters.register(P.ScatterNdAdd)
def get_scatter_add_vmap_rule(prim, axis_size):
    """VmapRule for `Scatter*` operations, such as `ScatterAdd` and `ScatterNdAdd`."""
    if isinstance(prim, str):
        prim_name = prim
        prim = Primitive(prim)
        use_locking = False
    else:
        prim_name = prim.name
        use_locking = prim.use_locking

    scatter_nd_add = P.ScatterNdAdd(use_locking)
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
            if prim_name == "ScatterAdd":
                indices = F.expand_dims(indices, -1)

            indices_shape = F.shape(indices)
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
            prefix = P.BroadcastTo(prefix_shape)(F.reshape(mnp.arange(axis_size), expand_shape))
            indices = concat((prefix, indices))
            out = scatter_nd_add(ref, indices, updates, u_monad)
        else:
            _raise_value_error("The source axis of `ref` in `{}` must be 0 or None, "
                               "but got {}.".format(prim_name, ref_dim))
            out = None
        return (out, ref_dim)

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


@vmap_rules_getters.register(P.Ger)
def get_fast_gelu_grad_vmap_rule(prim, axis_size):
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
