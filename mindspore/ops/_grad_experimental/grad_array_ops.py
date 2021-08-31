# Copyright 2021 Huawei Technologies Co., Ltd
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

from ...common import dtype as mstype
from .._grad.grad_math_ops import binop_grad_common
from .._grad.grad_base import bprop_getters
from ..composite.multitype_ops.zeros_like_impl import zeros_like
from .. import functional as F
from .. import operations as P

@bprop_getters.register(P.MaskedFill)
def get_bprop_masked_select(self):
    """Generate bprop for MaskedFill"""
    mul_op = P.Mul()
    sum_op = P.ReduceSum()
    is_instance_op = P.IsInstance()

    def bprop(input_data, mask, value, out, dout):
        mask = F.cast(mask, mstype.float32)
        dinput = mul_op(dout, (1 - mask))
        dvalue = mul_op(dout, mask)
        dinput, dvalue = binop_grad_common(input_data, mask, dinput, dvalue)
        dvalue = sum_op(dvalue)
        dinput = F.cast(dinput, F.dtype(input_data))
        if is_instance_op(value, mstype.number) is True:
            dvalue = 0
        else:
            dvalue = F.cast(dvalue, F.dtype(value))
        return dinput, zeros_like(mask), dvalue

    return bprop


@bprop_getters.register(P.TensorScatterSub)
def get_bprop_tensor_scatter_sub(self):
    """Generate bprop for TensorScatterSub"""
    gather_nd = P.GatherNd()
    neg = P.Neg()

    def bprop(x, indices, update, out, dout):
        update_grad = neg(gather_nd(dout, indices))
        return dout, zeros_like(indices), update_grad

    return bprop


def tensor_scatter_possible_replacement(x, indices, updates, out, dout):
    """bpropr for any TensorScatter* op that possibly replaces values in the input tensor"""
    gather_nd = P.GatherNd()
    scatter_nd = P.ScatterNd()
    equal = P.Equal()
    shape = P.Shape()

    x_indicators = F.cast(equal(x, out), mstype.int32)
    possibly_updated = gather_nd(out, indices)
    out_indicators = F.cast(equal(updates, possibly_updated), mstype.int32)
    scattered_out_indicators = scatter_nd(indices, out_indicators, shape(x))
    indicators = x_indicators + scattered_out_indicators
    dx = dout * F.cast(x_indicators, F.dtype(dout)) / F.cast(indicators, F.dtype(dout))
    dupdates = gather_nd(dout / F.cast(indicators, F.dtype(dout)), indices) * F.cast(out_indicators, F.dtype(dout))

    return F.cast(dx, F.dtype(x)), zeros_like(indices), F.cast(dupdates, F.dtype(updates))


@bprop_getters.register(P.TensorScatterMax)
def get_bprop_tensor_scatter_max(self):
    """Generate bprop for TensorScatterMax"""
    def bprop(x, indices, updates, out, dout):
        return tensor_scatter_possible_replacement(x, indices, updates, out, dout)

    return bprop


@bprop_getters.register(P.TensorScatterMin)
def get_bprop_tensor_scatter_min(self):
    """Generate bprop for TensorScatterMin"""
    def bprop(x, indices, updates, out, dout):
        return tensor_scatter_possible_replacement(x, indices, updates, out, dout)

    return bprop


@bprop_getters.register(P.SplitV)
def get_bprop_split_v(self):
    """Generate bprop for SplitV"""
    split_dim = self.split_dim
    concat_op = P.Concat(split_dim)

    def bprop(x_input, output, dout):
        dx = concat_op(dout)
        return (dx,)

    return bprop
