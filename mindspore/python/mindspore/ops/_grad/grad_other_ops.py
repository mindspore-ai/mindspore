# Copyright 2020-2022 Huawei Technologies Co., Ltd
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

"""Generate bprop for other ops"""

from mindspore.ops import operations as P
from mindspore.ops.operations import _grad_ops as G
from mindspore.ops.operations import _inner_ops as inner
from mindspore.ops.composite.multitype_ops.zeros_like_impl import zeros_like
from mindspore.ops._grad.grad_base import bprop_getters

# Unused parameters are placeholders.


@bprop_getters.register(P.Assign)
def get_bprop_assign(self):
    """Generate bprop for Assign"""

    def bprop(x, y, out, dout):
        return (dout, zeros_like(y))
    return bprop


@bprop_getters.register(P.InvertPermutation)
def get_bprop_invert_permutation(self):
    """Generate bprop for InvertPermutation"""

    def bprop(x, out, dout):
        return (zeros_like(x),)
    return bprop


@bprop_getters.register(P.IOU)
def get_bprop_iou(self):
    """Generate bprop for IOU"""

    def bprop(x, y, out, dout):
        return zeros_like(x), zeros_like(y)
    return bprop


@bprop_getters.register(inner.SyncBatchNorm)
def get_bprop_sync_batch_norm(self):
    """Grad definition for `SyncBatchNorm` operation."""
    input_grad = G.SyncBatchNormGrad(self.epsilon, self.group, self.device_num)

    def bprop(x, scale, b, mean, variance, out, dout):
        saved_mean = out[3]
        saved_variance = out[4]
        out = input_grad(dout[0], x, scale, saved_mean, saved_variance)
        dx = out[0]
        dscale = out[1]
        dbias = out[2]
        return dx, dscale, dbias, zeros_like(mean), zeros_like(variance)
    return bprop


@bprop_getters.register(inner.GpuConvertToDynamicShape)
def get_bprop_gpu_convert_to_dynamic_shape(self):
    """Get backprop for GpuConvertToDynamicShape."""

    def bprop(x, out, dout):
        return (dout,)
    return bprop


@bprop_getters.register(P._DynamicLossScale) # pylint: disable=W0212
def get_bprop_dynamic_loss_scale(self):
    """Get backprop for dynamic_loss_scale."""
    mul = P.Mul()
    mul.add_prim_attr('split_overflow', True)
    mul.add_prim_attr('layer_overflow', self.layer)

    def bprop(x, loss_scale, out, dout):
        res = mul(dout, loss_scale)
        return res, zeros_like(loss_scale)
    return bprop
