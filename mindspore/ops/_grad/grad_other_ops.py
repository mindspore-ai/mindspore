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

"""Generate bprop for other ops"""

from .. import operations as P
from ..composite.multitype_ops.zeros_like_impl import zeros_like
from .grad_base import bprop_getters

# Unused parameters are placeholders.


@bprop_getters.register(P.Assign)
def get_bprop_assign(self):
    """Generate bprop for Assign"""
    def bprop(x, y, out, dout):
        return (x, zeros_like(y))
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
