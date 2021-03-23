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

"""bprop primitives"""
from mindspore.ops import _constants
from ..operations import _grad_ops as G
from .. import functional as F
from .. import operations as P
from ..composite import multitype_ops as C
from .grad_base import bprops

get_dtype = P.DType()
# Unused parameters are placeholders.


@bprops.register("MaximumGrad")
def bprop_maximum_grad_grad(x, y, z, out, dout):
    """Backpropagator for primitive `MaximumGrad`."""
    out0 = F.cast(out[0] != 0, get_dtype(dout[0]))
    out1 = F.cast(out[1] != 0, get_dtype(dout[1]))
    dz = out0 * dout[0] + out1 * dout[1]
    return F.zeros_like(x), F.zeros_like(y), dz


@bprops.register("MinimumGrad")
def bprop_minimum_grad_grad(x, y, z, out, dout):
    """Backpropagator for primitive `MinimumGrad`."""
    out0 = F.cast(out[0] != 0, get_dtype(dout[0]))
    out1 = F.cast(out[1] != 0, get_dtype(dout[1]))
    dz = out0 * dout[0] + out1 * dout[1]
    return F.zeros_like(x), F.zeros_like(y), dz


@bprops.register("ReluGrad")
def bprop_relu_grad_grad(x, y, out, dout):
    """Backpropagator for primitive `ReluGrad`."""
    input_grad = G.ReluGrad()
    dy = input_grad(dout, y)
    return dy, F.zeros_like(y)


@bprops.register(_constants.kScalarAdd)
def bprop_scalar_add(x, y, out, dout):
    """Backpropagator for primitive `scalar_add`."""
    return dout, dout


@bprops.register(_constants.kScalarMul)
def bprop_scalar_mul(x, y, out, dout):
    """Backpropagator for primitive `scalar_mul`."""
    return dout*y, dout*x


@bprops.register(_constants.kScalarSub)
def bprop_scalar_sub(x, y, out, dout):
    """Backpropagator for primitive `scalar_sub`."""
    return dout, -dout


@bprops.register(_constants.kScalarDiv)
def bprop_scalar_div(x, y, out, dout):
    """Backpropagator for primitive `scalar_div`."""
    return dout/y, (-dout) * (out/y)


@bprops.register(_constants.kScalarPow)
def bprop_scalar_pow(x, y, out, dout):
    """Backpropagator for primitive `scalar_pow`."""
    return dout * (y * (x ** (y-1))), dout * (F.scalar_log(x) * out)


@bprops.register("scalar_exp")
def bprop_scalar_exp(x, out, dout):
    """Backpropagator for primitive `scalar_exp`."""
    return (dout * out,)


@bprops.register(_constants.kScalarUadd)
def bprop_scalar_uadd(x, out, dout):
    """Backpropagator for primitive `scalar_uadd`."""
    return (dout,)


@bprops.register(_constants.kScalarUsub)
def bprop_scalar_usub(x, out, dout):
    """Backpropagator for primitive `scalar_usub`."""
    return (-dout,)


@bprops.register("scalar_gt")
def bprop_scalar_gt(x, y, out, dout):
    """Backpropagator for primitive `scalar_gt`."""
    return C.zeros_like(x), C.zeros_like(y)


@bprops.register("scalar_lt")
def bprop_scalar_lt(x, y, out, dout):
    """Backpropagator for primitive `scalar_lt`."""
    return C.zeros_like(x), C.zeros_like(y)


@bprops.register("scalar_ge")
def bprop_scalar_ge(x, y, out, dout):
    """Backpropagator for primitive `scalar_ge`."""
    return C.zeros_like(x), C.zeros_like(y)


@bprops.register("scalar_le")
def bprop_scalar_le(x, y, out, dout):
    """Backpropagator for primitive `scalar_le`."""
    return C.zeros_like(x), C.zeros_like(y)


@bprops.register("scalar_eq")
def bprop_scalar_eq(x, y, out, dout):
    """Backpropagator for primitive `scalar_eq`."""
    return C.zeros_like(x), C.zeros_like(y)


@bprops.register("scalar_ne")
def bprop_scalar_ne(x, y, out, dout):
    """Backpropagator for primitive `scalar_eq`."""
    return C.zeros_like(x), C.zeros_like(y)


@bprops.register("scalar_cast")
def bprop_scalar_cast(x, t, out, dout):
    """Backpropagator for primitive `scalar_cast`."""
    return F.scalar_cast(dout, F.typeof(x)), t


@bprops.register(_constants.kTupleGetItem)
def bprop_tuple_getitem(data, idx, out, dout):
    """Backpropagator for primitive `tuple_getitem`."""
    return F.tuple_setitem(C.zeros_like(data), idx, dout), C.zeros_like(idx)


@bprops.register("list_getitem")
def bprop_list_getitem(data, idx, out, dout):
    """Backpropagator for primitive `list_getitem`."""
    return F.list_setitem(C.zeros_like(data), idx, dout), C.zeros_like(idx)


@bprops.register("identity")
def bprop_identity(x, out, dout):
    """Backpropagator for primitive `identity`."""
    return (dout,)


@bprops.register("make_ref")
def bprop_make_ref(key, x, y, out, dout):
    """Backpropagator for primitive `make_ref`."""
    return (C.zeros_like(key), dout, C.zeros_like(y))


@bprops.register("get_ref_value")
def bprop_get_ref_value(x, out, dout):
    """Backpropagator for primitive `get_ref_value`."""
    return (dout,)


@bprops.register("get_ref_key")
def bprop_get_ref_key(x, out, dout):
    """Backpropagator for primitive `get_ref_key`."""
    return (C.zeros_like(x),)


@bprops.register("scalar_to_array")
def bprop_scalar_to_array(x, out, dout):
    """Backpropagator for primitive `scalar_to_array`."""
    return (F.array_to_scalar(dout),)


@bprops.register("array_to_scalar")
def bprop_array_to_scalar(x, out, dout):
    """Backpropagator for primitive `array_to_scalar`."""
    return (F.scalar_to_array(dout),)


@bprops.register("reshape")
def bprop_reshape(xs, shp, out, dout):
    """Backpropagator for primitive `reshape`."""
    return F.reshape(dout, F.shape(xs)), C.zeros_like(shp)


@bprops.register("distribute")
def bprop_distribute(arr, shp, out, dout):
    """Backpropagator for primitive `distribute`."""
    return F.array_reduce(F.scalar_add, dout, F.shape(arr)), C.zeros_like(shp)


@bprops.register("shape")
def bprop_shape(arr, out, dout):
    """Backpropagator for primitive `shape`."""
    return (C.zeros_like(arr),)


@bprops.register("broadcast_shape")
def bprop_broadcast_shape(shp1, shp2, out, dout):
    """Backpropagator for primitive `broadcast_shape`."""
    return C.zeros_like(shp1), C.zeros_like(shp2)


@bprops.register("J")
def bprop_j(x, out, dout):
    """Backpropagator for primitive `J`."""
    return (F.jinv(dout),)


@bprops.register("array_reduce")
def bprop_array_reduce(fn, x, shp, out, dout):
    """Backpropagator for primitive `array_reduce`."""
    return F.distribute(dout, F.shape(x)), C.zeros_like(shp)


@bprops.register("Depend")
def bprop_depend(x, y, out, dout):
    """Backpropagator for primitive `depend`."""
    return dout, C.zeros_like(y)


@bprops.register("embed")
def bprop_embed(x, out, dout):
    """Backpropagator for primitive `embed`."""
    return (C.zeros_like(x),)


@bprops.register("bool_not")
def bprop_bool_not(x, out, dout):
    """Backpropagator for primitive `bool_not`."""
    return (C.zeros_like(x),)


@bprops.register("bool_or")
def bprop_bool_or(x, y, out, dout):
    """Backpropagator for primitive `bool_or`."""
    return C.zeros_like(x), C.zeros_like(y)


@bprops.register("stop_gradient")
def bprop_stop_gradient(x, out, dout):
    """Backpropagator for primitive `stop_gradient`."""
    return (C.zeros_like(x),)


@bprops.register("bool_and")
def bprop_bool_and(x, y, out, dout):
    """Backpropagator for primitive `bool_and`."""
    return C.zeros_like(x), C.zeros_like(y)


@bprops.register("Switch")
def bprop_switch(cond, tb, fb, out, dout):
    """Backpropagator for primitive `switch`."""
    return C.zeros_like(cond), F.switch(cond, dout, C.zeros_like(tb)), \
        F.switch(cond, C.zeros_like(fb), dout)


def _fprop_switch_layer(index, layers):
    """Backpropagator for primitive `switch_layer`."""
    def _bprop_switch_layer(dout):
        return dout, C.zeros_like(index), ()
    return F.switch_layer(index, layers), _bprop_switch_layer


@bprops.register("UpdateState")
def bprop_update_state(u_monad, x, out, dout):
    """Backpropagator for primitive `UpdateState`."""
    return C.zeros_like(u_monad), C.zeros_like(x)


@bprops.register("Load")
def bprop_load(param, u_monad, out, dout):
    """Backpropagator for primitive `load`."""
    return dout, C.zeros_like(u_monad)
