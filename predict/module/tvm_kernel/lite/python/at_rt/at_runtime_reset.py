# Copyright 2019 Huawei Technologies Co., Ltd
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
"""
This module is Using to make Lite Runtime funcitons instead TVM Runtime funcitons while codegen.
"""

import os
from tvm import codegen

class AtRuntimeReset():
    """Using this class to make Lite Runtime funcitons instead TVM Runtime funcitons while codegen
       Usage like:
           with at_runtime_reset.AtRuntimeReset():
           fadd = tvm.build(s, [A, B], tgt, target_host = tgt_host, name = "myadd")
       then the module fadd will using Lite runtime functions.
    """

    def __enter__(self):
        if os.getenv("TVM_RUNTIME_ON") is not None:
            return
        codegen.SetRTFuncTransPair(
            "TVMBackendAllocWorkspace", "LiteBackendAllocWorkspace"
        )
        codegen.SetRTFuncTransPair(
            "TVMBackendFreeWorkspace", "LiteBackendFreeWorkspace"
        )
        codegen.SetRTFuncTransPair("TVMAPISetLastError", "LiteAPISetLastError")
        codegen.SetRTFuncTransPair(
            "TVMBackendParallelLaunch", "LiteBackendParallelLaunch"
        )
        codegen.SetRTFuncTransPair(
            "TVMBackendParallelBarrier", "LiteBackendParallelBarrier"
        )
        codegen.SetRTFuncTransPair(
            "TVMBackendRegisterSystemLibSymbol", "LiteBackendRegisterSystemLibSymbol"
        )
        codegen.SetRTFuncTransPair("TVMFuncCall", "LiteFuncCall")
        codegen.SetRTFuncTransPair(
            "TVMBackendGetFuncFromEnv", "LiteBackendGetFuncFromEnv"
        )

    def __exit__(self, ptype, value, trace):
        codegen.DelRTFuncTransPair("TVMBackendAllocWorkspace")
        codegen.DelRTFuncTransPair("TVMBackendFreeWorkspace")
        codegen.DelRTFuncTransPair("TVMAPISetLastError")
        codegen.DelRTFuncTransPair("TVMBackendParallelLaunch")
        codegen.DelRTFuncTransPair("TVMBackendParallelBarrier")
        codegen.DelRTFuncTransPair("TVMBackendRegisterSystemLibSymbol")
        codegen.DelRTFuncTransPair("TVMFuncCall")
        codegen.DelRTFuncTransPair("TVMBackendGetFuncFromEnv")
