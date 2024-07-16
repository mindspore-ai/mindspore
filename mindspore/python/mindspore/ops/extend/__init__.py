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

Operators with better performance

"""

from __future__ import absolute_import

from mindspore.common import Tensor
from mindspore.ops.primitive import Primitive, PrimitiveWithInfer, PrimitiveWithCheck, prim_attr_register
from mindspore.ops.vm_impl_registry import get_vm_impl_fn, vm_impl_registry
from mindspore.ops.op_info_register import op_info_register, custom_info_register, AkgGpuRegOp, AkgAscendRegOp, \
    AiCPURegOp, TBERegOp, CpuRegOp, CustomRegOp, DataType
from mindspore.ops.primitive import constexpr
from . import (
    array_func,
    math_func,
    nn_func,
)

from .array_func import gather, max, min, one_hot
from .math_func import (
    baddbmm,
    bmm,
    add,
    sub
)

from .nn_func import (
    conv2d,
    max_pool2d,
    leaky_relu_ext,
    l1_loss_ext
)

__all__ = []
__all__.extend(array_func.__all__)
__all__.extend(math_func.__all__)
__all__.extend(nn_func.__all__)
