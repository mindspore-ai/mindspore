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

"""image_ops"""

from .._grad.grad_base import bprop_getters
from .. import operations as P
from .. import functional as F
from ...common import dtype as mstype
from ..operations.image_ops import ResizeBicubic
from ..operations._grad_ops import ResizeBicubicGrad


@bprop_getters.register(ResizeBicubic)
def get_bprop_resize_bicubic(self):
    """Grad definition for `ResizeBicubic` operation."""
    resize_bicubic_grad = ResizeBicubicGrad(align_corners=self.align_corners,
                                            half_pixel_centers=self.half_pixel_centers)
    def bprop(images, size, out, dout):
        images_type = F.dtype(images)
        type_list = [mstype.int8, mstype.uint8, mstype.int16, mstype.uint16, mstype.int32,
                     mstype.int64, mstype.float16]
        if images_type in type_list:
            images = F.cast(images, mstype.float64)
        dx = resize_bicubic_grad(dout, images)
        return (dx, P.ZerosLike()(size))
    return bprop
