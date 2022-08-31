# Copyright 2022 Huawei Technologies Co., Ltd
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

"""Vmap operations."""
from __future__ import absolute_import

from mindspore._c_expression import VmapGeneralPreprocess_, VmapGeneralRulePyAdapter_


class _VmapGeneralPreprocess(VmapGeneralPreprocess_):
    """
    General preprocessing of VmapRules. If the source axes of all inputs are `None`,
    means that vectorization is not performed, taking out the original input and call
    the primitive directly.
    """
    def __init__(self):
        VmapGeneralPreprocess_.__init__(self, "VmapGeneralPreprocess")


class _VmapGeneralRule(VmapGeneralRulePyAdapter_):
    """
    General rule python adapter is a adapter for general rule in c++. Some operators can
    implement loop-stack method in their vmaprule by calling this adapter.
    """
    def __init__(self, prim, axis_size):
        VmapGeneralRulePyAdapter_.__init__(self, 'vmapgeneralrule', prim, axis_size)
