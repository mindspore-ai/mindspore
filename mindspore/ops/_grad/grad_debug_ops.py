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

"""Generate bprop for debug ops"""

from .. import operations as P
from ..composite.multitype_ops.zeros_like_impl import zeros_like
from .grad_base import bprop_getters

# Unused parameters are placeholders.


@bprop_getters.register(P.ScalarSummary)
def get_bprop_scalar_summary(self):
    """Generate bprop for ScalarSummary"""

    def bprop(tag, x, out, dout):
        return tag, zeros_like(x)
    return bprop


@bprop_getters.register(P.TensorSummary)
def get_bprop_tensor_summary(self):
    """Generate bprop for TensorSummary"""

    def bprop(tag, x, out, dout):
        return tag, zeros_like(x)
    return bprop


@bprop_getters.register(P.ImageSummary)
def get_bprop_image_summary(self):
    """Generate bprop for ImageSummary"""

    def bprop(tag, x, out, dout):
        return tag, zeros_like(x)
    return bprop


@bprop_getters.register(P.HistogramSummary)
def get_bprop_histogram_summary(self):
    """Generate bprop for HistogramSummary"""

    def bprop(tag, x, out, dout):
        return tag, zeros_like(x)
    return bprop


@bprop_getters.register(P.InsertGradientOf)
def get_bprop_insert_gradient_of(self):
    """Generate bprop for InsertGradientOf"""
    f = self.f

    def bprop(x, out, dout):
        return (f(dout),)
    return bprop
