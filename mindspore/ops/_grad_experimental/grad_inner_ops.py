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

"""inner_ops"""

from .._grad.grad_base import bprop_getters
from ..operations import _inner_ops as inner
from .. import functional as F
from ..composite.multitype_ops.zeros_like_impl import zeros_like


@bprop_getters.register(inner.TensorCopySlices)
def get_bprop_tensor_copy_slices(self):
    """Generate bprop for TensorCopySlices"""
    tensor_copy_slices = inner.TensorCopySlices()

    def bprop(x, update, begin, end, stride, out, dout):
        x_grad = tensor_copy_slices(dout, zeros_like(update), begin, end, stride)
        update_grad = F.strided_slice(dout, begin, end, stride)
        return x_grad, update_grad, zeros_like(begin), zeros_like(end), zeros_like(stride)

    return bprop


@bprop_getters.register(inner.Roll)
def get_bprop_roll(self):
    """Generate bprop for Roll"""
    shift = self.shift
    axis = self.axis
    roll_grad = inner.Roll(-shift, axis)

    def bprop(x_input, out, dout):
        dx = roll_grad(dout)
        return (dx,)

    return bprop
