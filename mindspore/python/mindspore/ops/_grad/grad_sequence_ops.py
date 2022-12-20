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

"""sequence_ops"""

from mindspore.ops.operations import _sequence_ops as seq
from mindspore.ops.composite.multitype_ops.zeros_like_impl import zeros_like
from mindspore.ops._grad.grad_base import bprop_getters
from mindspore.ops.primitive import Primitive


@bprop_getters.register(seq.SequenceCount)
def get_bprop_count(self):
    """Generate bprop for SequenceCount"""

    def bprop(x, y, out, dout):
        return (zeros_like(x), zeros_like(y))

    return bprop


@bprop_getters.register(seq.SequenceIndex)
def get_bprop_index(self):
    """Generate bprop for SequenceIndex"""

    def bprop(x, y, out, dout):
        return (zeros_like(x), zeros_like(y))

    return bprop


@bprop_getters.register(seq.SequenceMul)
def get_bprop_mul(self):
    """Generate bprop for SequenceMul"""
    tuple_set_item = Primitive("TupleSetItem")

    def bprop(x, y, out, dout):
        dx = x
        for i in range(len(x)):
            dx = tuple_set_item(dx, i, dout[i])
        return (dx, zeros_like(y))

    return bprop
