# Copyright 2021-2023 Huawei Technologies Co., Ltd
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
from __future__ import absolute_import

from mindspore.ops.operations import _inner_ops as inner
from mindspore.ops.operations import _grad_ops as G
from mindspore.ops import operations as P
from mindspore.ops.composite.multitype_ops.zeros_like_impl import zeros_like
from mindspore.ops._grad_experimental.grad_base import bprop_getters


@bprop_getters.register(inner.ParallelResizeBilinear)
def get_bprop_parallel_resize_bilinear(self):
    """Grad definition for `ParallelResizeBilinear` operation."""
    grad = G.ParallelResizeBilinearGrad(self.ori_image_size, self.src_start_w, self.dst_start_w,
                                        self.align_corners)

    def bprop(x, size, out, dout):
        dx = grad(dout, x, size)
        return dx, zeros_like(size)

    return bprop


@bprop_getters.register(inner.PsROIPooling)
def get_bprop_ps_roi_pooling(self):
    """Grad definition for `PsROIPooling` operation."""
    shape_op = P.Shape()
    pooled_height = self.pooled_height
    pooled_width = self.pooled_width
    spatial_scale = self.spatial_scale
    out_dim = self.out_dim
    num_rois = self.num_rois

    def bprop(inputs, rois, out, dout):
        mapping_channel = out[1]
        inputs_shape = shape_op(inputs)
        batch_size = inputs_shape[0]
        channels = inputs_shape[1]
        height = inputs_shape[2]
        width = inputs_shape[3]

        dx = G.PsROIPoolingGrad(
            batch_size,
            channels,
            height,
            width,
            num_rois,
            pooled_height,
            pooled_width,
            spatial_scale,
            out_dim
        )(dout[0], rois, mapping_channel)

        return dx, zeros_like(rois)

    return bprop
