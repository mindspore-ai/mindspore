# Copyright 2023 Huawei Technologies Co., Ltd
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

"""grad_sequence_ops"""

from mindspore.ops.operations import _sequence_ops as seq
from mindspore.ops.composite.multitype_ops.zeros_like_impl import zeros_like
from mindspore.ops._grad.grad_base import bprop_getters
from mindspore.ops.primitive import Primitive


tuple_setitem = Primitive("tuple_setitem")


@bprop_getters.register(seq.SequenceCount)
def get_bprop_count(self):
    """Generate bprop for SequenceCount"""

    def bprop(x, y, out, dout):
        return (zeros_like(x), zeros_like(y))

    return bprop


@bprop_getters.register(seq.sequence_len)
def get_bprop_sequence_len(self):
    """Generate bprop for sequence_len"""
    def bprop(x, out, dout):
        return (zeros_like(x),)

    return bprop


@bprop_getters.register(seq.make_range)
def get_bprop_range(self):
    """Generate bprop for make_range"""
    def bprop(start, limit, delta, out, dout):
        return (zeros_like(start), zeros_like(limit), zeros_like(delta))

    return bprop


@bprop_getters.register(seq.SequenceAdd)
def get_bprop_sequence_add(self):
    """Generate bprop for SequenceAdd"""
    def bprop(x, y, out, dout):
        out_offset = seq.SequenceAddOffset()(x, y)
        dx = seq.SequenceSlice()(dout, out_offset[0], len(x), 1)
        dy = seq.SequenceSlice()(dout, out_offset[1], len(x) + len(y), 1)

        return (dx, dy)

    return bprop


@bprop_getters.register(seq.SequenceSlice)
def get_bprop_slice(self):
    """Generate bprop for SequenceSlice"""

    def bprop(x, start, stop, step, out, dout):
        dx = seq.SequenceSliceGrad()(dout, x, start, stop, step)
        return (dx, zeros_like(start), zeros_like(stop), zeros_like(step))

    return bprop


@bprop_getters.register(seq.SequenceIndex)
def get_bprop_index(self):
    """Generate bprop for SequenceIndex"""

    def bprop(x, y, out, dout):
        return (zeros_like(x), zeros_like(y))

    return bprop


@bprop_getters.register("tuple_setitem")
@bprop_getters.register("list_setitem")
def get_bprop_setitem(self):
    """Generate bprop for TupleSetItem and ListSetItem"""

    def bprop(x, idx, value, out, dout):
        d_x = tuple_setitem(dout, idx, 0)
        d_value = dout[idx]
        d_idx = 0
        return d_x, zeros_like(d_idx), d_value

    return bprop


@bprop_getters.register(seq.SequenceMul)
def get_bprop_mul(self):
    """Generate bprop for SequenceMul"""

    def bprop(x, y, out, dout):
        dx = x
        for i in range(len(x)):
            dx = tuple_setitem(dx, i, dout[i])
        return (dx, zeros_like(y))

    return bprop


@bprop_getters.register(seq.SequenceMin)
@bprop_getters.register(seq.SequenceMax)
def get_bprop_max_min(self):
    """Generate bprop for SequenceMax and SequenceMax"""

    def bprop(x, out, dout):
        index = seq.SequenceIndex()(x, out)
        dx = tuple_setitem(zeros_like(x), index, dout)
        return (dx,)

    return bprop


@bprop_getters.register("tuple_greater_than")
@bprop_getters.register("list_greater_than")
@bprop_getters.register("tuple_greater_equal")
@bprop_getters.register("list_greater_equal")
def get_bprop_greater(self):
    """Generate bprop for tuple_greater_than, list_greater_than,
    tuple_greater_equal, list_greater_equal.
    """

    def bprop(x, y, out, dout):
        return (zeros_like(x), zeros_like(y))

    return bprop
