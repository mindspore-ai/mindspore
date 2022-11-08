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

"""
Operators can be used in the construct function of Cell.

Examples:

    >>> import mindspore.ops as ops
"""
from __future__ import absolute_import

from mindspore.common import Tensor
from mindspore.ops.primitive import Primitive, PrimitiveWithInfer, PrimitiveWithCheck, prim_attr_register
from mindspore.ops.vm_impl_registry import get_vm_impl_fn, vm_impl_registry
from mindspore.ops.op_info_register import op_info_register, custom_info_register, AkgGpuRegOp, AkgAscendRegOp, \
    AiCPURegOp, TBERegOp, CpuRegOp, CustomRegOp, DataType
from mindspore.ops.primitive import constexpr
from mindspore.ops import composite, operations, functional, function
from mindspore.ops import signature
from mindspore.ops.composite import *
from mindspore.ops.operations import *
from mindspore.ops.function import *
from mindspore.ops.functional import *

__primitive__ = [
    "prim_attr_register", "Primitive", "PrimitiveWithInfer", "PrimitiveWithCheck", "signature"
]

__all__ = ["get_vm_impl_fn", "vm_impl_registry",
           "op_info_register", "custom_info_register", "AkgGpuRegOp", "AkgAscendRegOp", "AiCPURegOp", "TBERegOp",
           "CpuRegOp", "CustomRegOp", "DataType",
           "constexpr"]
__all__.extend(__primitive__)
__all__.extend(composite.__all__)
__all__.extend(operations.__all__)
__all__.extend(functional.__all__)
__all__.extend(function.__all__)
