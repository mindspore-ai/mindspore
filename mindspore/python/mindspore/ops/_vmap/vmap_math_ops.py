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

"""math_ops vmap impl."""

import mindspore.numpy as mnp
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops import constexpr
from ..primitive import Primitive
from .._vmap.vmap_base import vmap_rules_getters, vmap_general_preprocess, get_assign_vmap_rule, \
    get_unop_vmap_rule, _raise_value_error, _bdim_at_front, _broadcast_by_axis


@constexpr
def _broadcast_shape(nd, x_ndim, x_shape):
    return x_shape + (1,) * (nd - x_ndim)


@constexpr
def _get_broadcast_shape(x_shape, y_shape):
    """Get the broadcast shape for _handle_broadcasting."""
    x_len = len(x_shape)
    y_len = len(y_shape)

    if x_len >= y_len:
        return x_shape

    # scalar case
    if not x_len:
        return (1,) * y_len

    broadcast_shape = ()
    x_index = x_len - 1

    for i in range(y_len - 1, 0, -1):
        if x_index == 0:
            broadcast_shape = (1,) + broadcast_shape
        elif x_shape[x_index] == y_shape[i] or y_shape[i] == 1 or x_shape[x_index] == 1:
            broadcast_shape = (x_shape[x_index],) + broadcast_shape
            x_index -= 1
        else:
            broadcast_shape = (1,) + broadcast_shape

    finnal_shape = (x_shape[0],) + tuple(broadcast_shape)
    return finnal_shape


def _handle_broadcasting(x, x_shape, y_shape):
    """Handle the broadcasting for the broadcast_binary_op."""
    broadcast_shape = _get_broadcast_shape(x_shape, y_shape)
    return F.reshape(x, broadcast_shape)


@vmap_rules_getters.register(P.Add)
@vmap_rules_getters.register(P.Sub)
@vmap_rules_getters.register(P.Mul)
@vmap_rules_getters.register(P.Div)
@vmap_rules_getters.register(P.RealDiv)
@vmap_rules_getters.register(P.FloorDiv)
@vmap_rules_getters.register(P.Maximum)
@vmap_rules_getters.register(P.Minimum)
@vmap_rules_getters.register(P.Atan2)
@vmap_rules_getters.register(P.Pow)
@vmap_rules_getters.register(P.Mod)
@vmap_rules_getters.register(P.Equal)
@vmap_rules_getters.register(P.GreaterEqual)
@vmap_rules_getters.register(P.Greater)
@vmap_rules_getters.register(P.LessEqual)
@vmap_rules_getters.register(P.Less)
@vmap_rules_getters.register(P.NotEqual)
@vmap_rules_getters.register(P.LogicalOr)
@vmap_rules_getters.register(P.LogicalAnd)
@vmap_rules_getters.register(P.BitwiseAnd)
@vmap_rules_getters.register(P.BitwiseOr)
@vmap_rules_getters.register(P.BitwiseXor)
def get_broadcast_binary_op_vmap_rule(prim, axis_size):
    """VmapRule for binary operations with broadcasting, such as `Add` and `Sub`."""

    def vmap_rule(x_bdim, y_bdim):
        is_all_none, result = vmap_general_preprocess(prim, x_bdim, y_bdim)
        if is_all_none:
            return result

        x, x_dim = x_bdim
        y, y_dim = y_bdim
        x_shape = F.shape(x)
        y_shape = F.shape(y)
        if x_dim == y_dim and x_shape == y_shape:
            out = prim(x, y)
            return (out, x_dim)

        if F.rank(x):
            x = _bdim_at_front(x, x_dim, 1)
        if F.rank(y):
            y = _bdim_at_front(y, y_dim, 1)
        x_shape = F.shape(x)
        y_shape = F.shape(y)
        x = _handle_broadcasting(x, x_shape, y_shape)
        y = _handle_broadcasting(y, y_shape, x_shape)

        out = prim(x, y)
        return (out, 0)

    return vmap_rule


@vmap_rules_getters.register(P.AddN)
def get_add_n_vmap_rule(prim, axis_size):
    """VmapRule for AddN operation."""
    if isinstance(prim, str):
        prim = Primitive(prim)

    def vmap_rule(*inputs_bdim):
        is_all_none, result = vmap_general_preprocess(prim, *inputs_bdim)
        if is_all_none:
            return result

        if not isinstance(inputs_bdim, (tuple, list)):
            _raise_value_error("The 'x' of P.AddN is neither tuple nor list.")

        args = inputs_bdim[0]
        vals = ()
        for each_arg in args:
            x, bdim = each_arg
            x = _bdim_at_front(x, bdim, axis_size)
            vals = vals + (x,)

        out = prim(vals)
        return (out, 0)

    return vmap_rule


@vmap_rules_getters.register(P.MatMul)
@vmap_rules_getters.register(P.BatchMatMul)
def get_matmul_vmap_rule(prim, axis_size):
    """VmapRule for `*MatMul` operation."""
    if isinstance(prim, str):
        prim = Primitive(prim)
        transpose_a = False
        transpose_b = False
    else:
        transpose_a = prim.transpose_a
        transpose_b = prim.transpose_b
    batch_matmul = P.BatchMatMul(transpose_a, transpose_b)

    def vmap_rule(a_bdim, b_bdim):
        is_all_none, result = vmap_general_preprocess(prim, a_bdim, b_bdim)
        if is_all_none:
            return result

        a, a_dim = a_bdim
        b, b_dim = b_bdim
        a = _bdim_at_front(a, a_dim, axis_size)
        b = _bdim_at_front(b, b_dim, axis_size)

        out = batch_matmul(a, b)
        return (out, 0)

    return vmap_rule


@vmap_rules_getters.register(P.BroadcastTo)
def get_broadcast_to_vmap_rule(prim, axis_size):
    """VmapRule for `BroadcastTo` operation."""
    shape = prim.shape

    def vmap_rule(operand_bdim):
        is_all_none, result = vmap_general_preprocess(prim, operand_bdim)
        if is_all_none:
            return result

        x, dim = operand_bdim
        x = mnp.moveaxis(x, dim, 0)
        axis_size = F.shape(x)[0]
        batch_shape = (axis_size,) + shape

        out = P.BroadcastTo(batch_shape)(x)
        return (out, 0)

    return vmap_rule


@constexpr
def _get_reduce_batch_axis(axis, x_dim, x_ndim):
    """get batch_axis for reduce* operation."""
    if not isinstance(axis, tuple):
        axis = (axis,)

    batch_axis = ()
    if axis:
        for index in axis:
            if index < x_dim:
                batch_axis = batch_axis + (index,)
            else:
                batch_axis = batch_axis + (index + 1,)
    else:
        batch_axis_list = [index for index in range(x_ndim)]
        del batch_axis_list[x_dim]
        batch_axis = tuple(batch_axis_list)
    return batch_axis


@constexpr
def _get_reduce_out_dim(keep_dims, x_dim, x_ndim, batch_axis):
    """get out_dim for reduce* operation."""
    if keep_dims:
        out_dim = x_dim
    else:
        out_dim = 0
        for i in range(x_ndim):
            if i == x_dim:
                break
            if i in batch_axis:
                continue
            else:
                out_dim += 1
    return out_dim


@vmap_rules_getters.register(P.ReduceSum)
@vmap_rules_getters.register(P.ReduceMax)
@vmap_rules_getters.register(P.ReduceMin)
def get_reducer_vmap_rule(prim, axis_size):
    """VmapRule for reduce operations, such as `ReduceSum`."""
    keep_dims = prim.keep_dims
    prim_name = prim.name

    def vmap_rule(operand_bdim, axis_bdim):
        is_all_none, result = vmap_general_preprocess(prim, operand_bdim, axis_bdim)
        if is_all_none:
            return result

        x, x_dim = operand_bdim
        axis, axis_dim = axis_bdim
        if axis_dim is not None:
            _raise_value_error("The source axis of `axis` in `{}` must be None, "
                               "but got {}.".format(prim_name, axis_dim))

        x_ndim = F.rank(x)
        batch_axis = _get_reduce_batch_axis(axis, x_dim, x_ndim)

        out = prim(x, batch_axis)
        out_dim = _get_reduce_out_dim(keep_dims, x_dim, x_ndim, batch_axis)
        return (out, out_dim)

    return vmap_rule


@constexpr
def _get_index_add_batch_axis(axis, x_dim, x_ndim):
    """get batch_axis for IndexAdd."""
    # case1: batch not exists
    if x_dim is None:
        return axis
    # case2: batch exists
    if axis < -x_ndim or x_dim >= x_ndim:
        raise ValueError("'axis' of 'IndexAdd' should be in range [{}, {}), but got {}".format(-x_ndim, x_ndim, axis))
    if axis < 0:
        axis = axis + x_ndim
    if x_dim > axis:
        return axis
    return axis + 1


@vmap_rules_getters.register(P.IndexAdd)
def get_index_add_vmap_rule(prim, axis_size):
    """VmapRule for IndexAdd."""
    axis = prim.axis

    def vmap_rule(x_bdim, indices_bdim, y_bdim, u_monad):
        x, x_dim = x_bdim
        indices, indices_dim = indices_bdim
        y, y_dim = y_bdim

        if indices_dim is not None:
            _raise_value_error("The batch dimension of 'indices' in 'IndexAdd' must be None, but got {}."
                               .format(indices_dim))
        if x_dim is None and y_dim is not None:
            _raise_value_error("The batch dimension of 'x' in 'IndexAdd' is None, so the batch dimension of 'y' "
                               "must be None, but got {}. ".format(y_dim))

        # update axis
        new_axis = _get_index_add_batch_axis(axis, x_dim, F.rank(x))
        op = P.IndexAdd(new_axis)

        if x_dim == y_dim:
            out = op(x, indices, y, u_monad)
            return (out, x_dim)
        if y_dim is None:
            y = _broadcast_by_axis(y, x_dim, axis_size)
        else:
            y = mnp.moveaxis(y, y_dim, x_dim)
        out = op(x, indices, y, u_monad)
        return (out, x_dim)

    return vmap_rule


get_assign_vmap_rule = vmap_rules_getters.register(P.AssignAdd)(get_assign_vmap_rule)
get_assign_vmap_rule = vmap_rules_getters.register(P.AssignSub)(get_assign_vmap_rule)
get_unop_vmap_rule = vmap_rules_getters.register(P.Abs)(get_unop_vmap_rule)
get_unop_vmap_rule = vmap_rules_getters.register(P.ACos)(get_unop_vmap_rule)
get_unop_vmap_rule = vmap_rules_getters.register(P.Acosh)(get_unop_vmap_rule)
get_unop_vmap_rule = vmap_rules_getters.register(P.Asin)(get_unop_vmap_rule)
get_unop_vmap_rule = vmap_rules_getters.register(P.Asinh)(get_unop_vmap_rule)
get_unop_vmap_rule = vmap_rules_getters.register(P.Atan)(get_unop_vmap_rule)
get_unop_vmap_rule = vmap_rules_getters.register(P.Atanh)(get_unop_vmap_rule)
get_unop_vmap_rule = vmap_rules_getters.register(P.Cos)(get_unop_vmap_rule)
get_unop_vmap_rule = vmap_rules_getters.register(P.Cosh)(get_unop_vmap_rule)
get_unop_vmap_rule = vmap_rules_getters.register(P.Sin)(get_unop_vmap_rule)
get_unop_vmap_rule = vmap_rules_getters.register(P.Sinh)(get_unop_vmap_rule)
get_unop_vmap_rule = vmap_rules_getters.register(P.Tan)(get_unop_vmap_rule)
get_unop_vmap_rule = vmap_rules_getters.register(P.Tanh)(get_unop_vmap_rule)
get_unop_vmap_rule = vmap_rules_getters.register(P.Ceil)(get_unop_vmap_rule)
get_unop_vmap_rule = vmap_rules_getters.register(P.Erf)(get_unop_vmap_rule)
get_unop_vmap_rule = vmap_rules_getters.register(P.Erfc)(get_unop_vmap_rule)
get_unop_vmap_rule = vmap_rules_getters.register(P.Erfinv)(get_unop_vmap_rule)
get_unop_vmap_rule = vmap_rules_getters.register(P.Exp)(get_unop_vmap_rule)
get_unop_vmap_rule = vmap_rules_getters.register(P.Expm1)(get_unop_vmap_rule)
get_unop_vmap_rule = vmap_rules_getters.register(P.Floor)(get_unop_vmap_rule)
get_unop_vmap_rule = vmap_rules_getters.register(P.Log)(get_unop_vmap_rule)
get_unop_vmap_rule = vmap_rules_getters.register(P.Log1p)(get_unop_vmap_rule)
get_unop_vmap_rule = vmap_rules_getters.register(P.LogicalNot)(get_unop_vmap_rule)
get_unop_vmap_rule = vmap_rules_getters.register(P.Neg)(get_unop_vmap_rule)
get_unop_vmap_rule = vmap_rules_getters.register(P.Reciprocal)(get_unop_vmap_rule)
get_unop_vmap_rule = vmap_rules_getters.register(P.Rint)(get_unop_vmap_rule)
get_unop_vmap_rule = vmap_rules_getters.register(P.Round)(get_unop_vmap_rule)
get_unop_vmap_rule = vmap_rules_getters.register(P.Rsqrt)(get_unop_vmap_rule)
get_unop_vmap_rule = vmap_rules_getters.register(P.Sigmoid)(get_unop_vmap_rule)
get_unop_vmap_rule = vmap_rules_getters.register(P.Sqrt)(get_unop_vmap_rule)
get_unop_vmap_rule = vmap_rules_getters.register(P.Sign)(get_unop_vmap_rule)
get_unop_vmap_rule = vmap_rules_getters.register(P.Real)(get_unop_vmap_rule)
get_unop_vmap_rule = vmap_rules_getters.register(P.Imag)(get_unop_vmap_rule)
get_unop_vmap_rule = vmap_rules_getters.register(P.IsNan)(get_unop_vmap_rule)
get_unop_vmap_rule = vmap_rules_getters.register(P.IsInf)(get_unop_vmap_rule)
get_unop_vmap_rule = vmap_rules_getters.register(P.IsFinite)(get_unop_vmap_rule)
get_unop_vmap_rule = vmap_rules_getters.register(P.BesselJ0)(get_unop_vmap_rule)
get_unop_vmap_rule = vmap_rules_getters.register(P.BesselJ1)(get_unop_vmap_rule)
get_unop_vmap_rule = vmap_rules_getters.register(P.BesselI0)(get_unop_vmap_rule)
get_unop_vmap_rule = vmap_rules_getters.register(P.BesselI0e)(get_unop_vmap_rule)
get_unop_vmap_rule = vmap_rules_getters.register(P.BesselK0)(get_unop_vmap_rule)
get_unop_vmap_rule = vmap_rules_getters.register(P.BesselK0e)(get_unop_vmap_rule)
get_unop_vmap_rule = vmap_rules_getters.register(P.BesselY0)(get_unop_vmap_rule)
get_unop_vmap_rule = vmap_rules_getters.register(P.BesselY1)(get_unop_vmap_rule)
