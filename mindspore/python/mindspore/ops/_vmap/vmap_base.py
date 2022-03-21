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

"""vmap base functions"""

import mindspore.numpy as mnp
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops import constexpr
from .._register_for_op import Registry
from ..composite import HyperMap, _VmapGeneralPreprocess
from ..primitive import Primitive
from ...common import Tensor


vmap_rules_getters = Registry()
vmap_rules = Registry()


def get_vmap_rule(prim, axis_size):
    """get vmap rule function by primitive obj or prim name for c++"""
    out = vmap_rules_getters.get(prim, None)
    if out:
        return out(prim, axis_size)
    return vmap_rules.get(prim, None)


@constexpr
def _raise_value_error(info, param=None):
    """Constexpr for raise_value_error."""
    if param is None:
        raise ValueError(info)
    raise ValueError(info + f"{param}")


@constexpr
def _get_broadcast_shape(x_shape, dst, axis_size):
    """Get the target shape for broadcast array."""
    x_ndim = len(x_shape)
    broadcast_ndim = x_ndim + 1

    if dst < -broadcast_ndim or dst >= broadcast_ndim:
        _raise_value_error("ValueError: destination axis is out of bounds for array of dimension")
    if dst < 0:
        dst = broadcast_ndim + dst

    target_shape = list(x_shape)
    target_shape.insert(dst, axis_size)
    return tuple(target_shape)


def _broadcast_by_axis(x, dst: int, axis_size: int):
    """Broadcasts an array to a new shape alone the destination axis."""
    if not isinstance(x, Tensor):
        if dst == 0:
            x = [x] * axis_size
            return Tensor(x)
        _raise_value_error("ValueError: destination axis is out of bounds for array of dimension")

    x_shape = F.shape(x)
    target_shape = _get_broadcast_shape(x_shape, dst, axis_size)
    x = F.expand_dims(x, dst)
    return P.BroadcastTo(target_shape)(x)


def vmap_bind_all_none(inputs):
    results = ()
    if isinstance(inputs, tuple):
        for res in inputs:
            results = results + ((res, None),)
        return results
    return (inputs, None)


vmap_general_preprocess = _VmapGeneralPreprocess()


def vmap_general_rule(prim, axis_size):
    """
    When the primitive does not registered the relevant specific VmapRule, it attempts to get
    this the general rule. The general rule is combining loop and stack operators to simulate
    the behavior of Vmap. Noted that, general rules does not guarantee the correctness of
    execution results.
    Currently, only the following types of primitives are supported:
    1、 Most calculation operations, whose inputs are tensors, scalars or both of them.
       (If all elements in a tuple are scalars, it is also considered scalar.)
    2、 Operators with indefinite inputs length, such as `AddN`, whose inputs is wrapped into a tuple.
    In other words, we do not support any tuple wrapped variables except for the special cases
    listed above.
    """

    if isinstance(prim, str):
        prim_name = prim
        prim = Primitive(prim)
    else:
        prim_name = prim.name
    common_map = HyperMap()

    def loop_stack(*args):
        is_all_none, result = vmap_general_preprocess(prim, *args)
        if is_all_none:
            return result

        wrapped_tuple = False
        # Handle case such as args:(((A, 0), (B, 1)),)
        if len(args) == 1 and isinstance(args[0][-1], tuple):
            wrapped_tuple = True
            args = args[0]

        vals_in_tuple = ()
        for val_in in args:
            val, dim = val_in
            out = ()
            if dim is None:
                # Handle case such as args:(..., (A, None), (1, None), ...)
                for _ in range(axis_size):
                    out = out + (val,)
            else:
                if isinstance(val, Tensor):
                    # Handle case such as args:(..., (A, 0), (B, 1), ...)
                    out = P.Unstack(dim)(val)
                else:
                    _raise_value_error("A variable of type other than `Tensor` is accepted, "
                                       "but the source axis is not `None`")

            vals_in_tuple = vals_in_tuple + (out,)

        if wrapped_tuple:
            output = ()
            for sub_tuple in zip(*vals_in_tuple):
                out = prim(sub_tuple)
                output = output + (out,)
        else:
            output = common_map(prim, *vals_in_tuple)

        vals_out_tuple = ()
        if isinstance(output[0], tuple):
            for res in zip(**output):
                if not isinstance(res[0], Tensor):
                    _raise_value_error("The output of the operator is not of the Tensor type, "
                                       "a specific vmap rule is required for op: ", prim_name)
                out = F.stack(res)
                vals_out_tuple = vals_out_tuple + ((out, 0),)
        else:
            out = F.stack(output)
            vals_out_tuple = vals_out_tuple + (out, 0)
        return vals_out_tuple
    return loop_stack


def vmap_monad_rule(prim, axis_size):
    """
    When the monad primitive does not registered the relevant specific VmapRule, it attempts to get
    this the general monad rule. Currently, only all inputs with the source axis of `None` can be
    supported.
    """
    if isinstance(prim, str):
        prim_name = prim
        prim = Primitive(prim)
    else:
        prim_name = prim.name

    def vmap_rule(*args):
        vals = ()
        args_len = len(args)
        for index, val_in in enumerate(args, 1):
            # Only the monad tag can not be tuple
            if index == args_len:
                vals = vals + (val_in,)
            if not isinstance(val_in, tuple):
                _raise_value_error("vmap currently not support the side effect op: ", prim_name)
            else:
                val, dim = val_in
                if dim is not None:
                    _raise_value_error("vmap currently not support the side effect op: ", prim_name)
                vals = vals + (val,)
        out = prim(*vals)
        return (out, None)
    return vmap_rule


@vmap_rules_getters.register("list_getitem")
@vmap_rules_getters.register("TupleGetItem")
def get_seq_get_item_vmap_rule(prim, axis_size):
    """VmapRule for `list_getitem` or `TupleGetItem` operation."""
    if isinstance(prim, str):
        prim = Primitive(prim)
    def vmap_rule(inputs_seq, index_in):
        index, _ = index_in
        out = prim(inputs_seq, index)
        return out

    return vmap_rule


@vmap_rules_getters.register(P.Sin)
@vmap_rules_getters.register(P.Cos)
def get_unop_vmap_rule(prim, axis_size):
    """VmapRule for unary operations, such as `Sin` and `Cos`."""
    if isinstance(prim, str):
        prim = Primitive(prim)
    def vmap_rule(x_in):
        var, dim = x_in
        out = prim(var)
        return (out, dim)

    return vmap_rule


def _bdim_at_front(x, src, axis_size):
    if src is None:
        return _broadcast_by_axis(x, 0, axis_size)
    return mnp.moveaxis(x, src, 0)


def _handle_scalar_broadcasting(nd, x):
    x_shape = F.shape(x)
    x_ndim = len(x_shape)
    if nd == x_ndim:
        return x
    return F.reshape(x, x_shape + (1,) * (nd - x_ndim))


@vmap_rules_getters.register(P.Add)
@vmap_rules_getters.register(P.Sub)
@vmap_rules_getters.register(P.Mul)
@vmap_rules_getters.register(P.Div)
def get_math_binary_op_vmap_rule(prim, axis_size):
    """VmapRule for binary operations, such as `Add` and `Sub`."""
    def vmap_rule(x_in, y_in):
        is_all_none, result = vmap_general_preprocess(prim, x_in, y_in)
        if is_all_none:
            return result

        x, x_bdim = x_in
        y, y_bdim = y_in
        x = _bdim_at_front(x, x_bdim, axis_size)
        y = _bdim_at_front(y, y_bdim, axis_size)
        x_nd = F.rank(x)
        y_nd = F.rank(y)
        ndim = x_nd
        if y_nd > ndim:
            ndim = y_nd

        x = _handle_scalar_broadcasting(ndim, x)
        y = _handle_scalar_broadcasting(ndim, y)

        out = prim(x, y)
        return (out, 0)

    return vmap_rule


@vmap_rules_getters.register(P.Transpose)
def get_transpose_vmap_rule(prim, axis_size):
    """VmapRule for `Transpose` operation."""
    def vmap_rule(x_in, perm_in):
        is_all_none, result = vmap_general_preprocess(prim, x_in, perm_in)
        if is_all_none:
            return result

        x, dim = x_in
        perm, perm_dim = perm_in
        if perm_dim is not None:
            _raise_value_error("The source axis of perm in `Transpose` must be None, but got ", perm_dim)
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


@vmap_rules_getters.register(P.Reshape)
def get_reshape_vmap_rule(prim, axis_size):
    """VmapRule for `Reshape` operation."""
    def vmap_rule(operand_in, shape_in):
        is_all_none, result = vmap_general_preprocess(prim, operand_in, shape_in)
        if is_all_none:
            return result

        x, dim = operand_in
        shape, shape_dim = shape_in
        if shape_dim is not None:
            _raise_value_error("The source axis of shape in `Reshape` must be None, but got ", shape_dim)

        x = mnp.moveaxis(x, dim, 0)
        axis_size = F.shape(x)[0]
        batch_shape = (axis_size,) + shape

        out = prim(x, batch_shape)
        return (out, 0)

    return vmap_rule


@vmap_rules_getters.register(P.BroadcastTo)
def get_broadcast_to_vmap_rule(prim, axis_size):
    """VmapRule for `BroadcastTo` operation."""
    shape = prim.shape
    def vmap_rule(operand_in):
        is_all_none, result = vmap_general_preprocess(prim, operand_in)
        if is_all_none:
            return result

        x, dim = operand_in
        x = mnp.moveaxis(x, dim, 0)
        axis_size = F.shape(x)[0]
        batch_shape = (axis_size,) + shape

        out = P.BroadcastTo(batch_shape)(x)
        return (out, 0)

    return vmap_rule


@vmap_rules_getters.register(P.ReduceSum)
@vmap_rules_getters.register(P.ReduceMax)
@vmap_rules_getters.register(P.ReduceMin)
def get_reducer_vmap_rule(prim, axis_size):
    """VmapRule for reduce operations, such as `ReduceSum`."""
    keep_dims = prim.keep_dims
    prim_name = prim.name
    def vmap_rule(operand_in, axis_in):
        is_all_none, result = vmap_general_preprocess(prim, operand_in, axis_in)
        if is_all_none:
            return result

        x, dim = operand_in
        axis, axis_dim = axis_in
        if axis_dim is not None:
            _raise_value_error("The source axis of `axis` in `" + prim_name + "` must be None, but got ", axis_dim)

        if not isinstance(axis, tuple):
            axis = (axis,)

        x_ndim = F.rank(x)
        batch_axis = ()
        if axis:
            for index in axis:
                if index < dim:
                    batch_axis = batch_axis + (index,)
                else:
                    batch_axis = batch_axis + (index + 1,)
        else:
            for index in range(x_ndim):
                if index != dim:
                    batch_axis = batch_axis + (index,)

        out = prim(x, batch_axis)
        if keep_dims:
            out_dim = dim
        else:
            out_dim = 0
            for i in range(x_ndim):
                if i == dim:
                    break
                if i in batch_axis:
                    continue
                else:
                    out_dim += 1
        return (out, out_dim)

    return vmap_rule

@vmap_rules_getters.register("Switch")
@vmap_rules_getters.register("Partial")
def get_partical_vmap_rule(prim, axis_size):
    """VmapRule for `Partial` and `Switch` operation."""
    if isinstance(prim, str):
        prim_name = prim
        prim = Primitive(prim)
    else:
        prim_name = prim.name
    def vmap_rule(*args):
        vals = ()
        for val_in in args:
            if not isinstance(val_in, tuple):
                vals = vals + (val_in,)
            else:
                val, dim = val_in
                if dim is not None:
                    _raise_value_error("The source axis of args in " + prim_name + " must be None, but got ", dim)
                vals = vals + (val,)

        out = prim(*vals)
        return out

    return vmap_rule


@vmap_rules_getters.register("Load")
def get_load_vmap_rule(prim, axis_size):
    """VmapRule for `Load` operation."""
    if isinstance(prim, str):
        prim = Primitive(prim)
    def vmap_rule(ref_in, u_monad):
        var, dim = ref_in
        out = prim(var, u_monad)
        return (out, dim)

    return vmap_rule


@vmap_rules_getters.register(P.Assign)
@vmap_rules_getters.register(P.AssignAdd)
def get_assign_vmap_rule(prim, axis_size):
    """VmapRule for `Assign*` operations, such as `Assign` and `AssignAdd`."""
    if isinstance(prim, str):
        prim_name = prim
        prim = Primitive(prim)
    else:
        prim_name = prim.name
    def vmap_rule(variable, value, u_monad):
        var, var_dim = variable
        val, val_dim = value

        if var_dim is None:
            if val_dim is not None:
                _raise_value_error("The source axis of `variable` is None, but the source "
                                   "axis of `value` is not None. The execution order of "
                                   "operator " + prim_name + " cannot be guaranteed")
        else:
            if val_dim is None:
                val = _broadcast_by_axis(val, var_dim, axis_size)
            else:
                val = mnp.moveaxis(val, val_dim, var_dim)
        out = prim(var, val, u_monad)
        return (out, var_dim)

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

    def vmap_rule(ref_batch, indices_batch, updates_batch, u_monad):
        ref, ref_dim = ref_batch
        indices, indices_dim = indices_batch
        updates, updates_dim = updates_batch

        if ref_dim is None:
            if indices_dim is not None or updates_dim is not None:
                _raise_value_error("The source axis of `ref` is None, but the source "
                                   "axis of `indices` or `updates` is not None. The execution order of "
                                   "operator " + prim_name + " cannot be guaranteed")
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
            _raise_value_error("The source axis of `ref` in " + prim_name +
                               " must be 0 or None, but got ", ref_dim)
            out = None
        return (out, ref_dim)

    return vmap_rule


@vmap_rules_getters.register(P.Print)
def get_print_vmap_rule(prim, axis_size):
    """VmapRule for `Print` operation."""
    if isinstance(prim, str):
        prim = Primitive(prim)
    def vmap_rule(*args):
        vals = ()
        args_len = len(args)
        for index, val_in in enumerate(args, 1):
            # Only the monad tag can not be tuple
            if index == args_len:
                vals = vals + (val_in,)
                break
            if not isinstance(val_in, tuple):
                _raise_value_error("The received args does not contain axis information in P.Print")
            else:
                val, dim = val_in
                if dim is None:
                    vals = vals + (val,)
                else:
                    vals = vals + ("(", val, ", dim: ", dim, ")")

        out = prim(*vals)
        return out

    return vmap_rule
