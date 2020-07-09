# Copyright 2020 Huawei Technologies Co., Ltd
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

from .. import operations as P
from ..operations import _grad_ops as G
from ..operations import _inner_ops as inner
from ..composite.multitype_ops.zeros_like_impl import zeros_like
from .grad_base import bprop_getters


@bprop_getters.register(inner.StridedSliceAICPU)
def get_bprop_strided_slice_aicpu(self):
    """Generate bprop for StridedSlice"""
    shape_op = P.Shape()
    input_grad = G.StridedSliceGradAICPU(self.begin_mask,
                                         self.end_mask,
                                         self.ellipsis_mask,
                                         self.new_axis_mask,
                                         self.shrink_axis_mask)

    def bprop(x, begin, end, strides, out, dout):
        dx = input_grad(dout, shape_op(x), begin, end, strides)
        return dx, zeros_like(begin), zeros_like(end), zeros_like(strides)

    return bprop
