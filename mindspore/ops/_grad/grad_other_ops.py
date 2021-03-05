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

"""Generate bprop for other ops"""

from .. import operations as P
from .. import composite as C
from ..operations import _grad_ops as G
from ..operations import _inner_ops as inner
from ..composite.multitype_ops.zeros_like_impl import zeros_like
from .grad_base import bprop_getters

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


@bprop_getters.register(P.PQC)
def bprop_pqc(self):
    """Generate bprop for PQC"""

    t = P.Transpose()
    mul = P.Mul()
    sum_ = P.ReduceSum()

    def bprop(encoder_data, ansatz_data, out, dout):
        dx = t(out[1], (2, 0, 1))
        dx = mul(dout[0], dx)
        dx = sum_(dx, 2)
        dx = t(dx, (1, 0))
        dy = C.tensor_dot(dout[0], out[2], ((0, 1), (0, 1)))
        return dx, dy
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
    def bprop(x, out, dout):
        return (dout,)
    return bprop
