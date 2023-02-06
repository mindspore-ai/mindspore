# Copyright 2023 Huawei Technologies Co., Ltd
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

"""Generate bprop for quantization aware ops"""

from mindspore.ops.operations import _scalar_ops
from mindspore.ops._grad.grad_base import bprop_getters
from mindspore.ops.composite.multitype_ops.zeros_like_impl import zeros_like


@bprop_getters.register(_scalar_ops.ScalarAdd)
def get_bprop_scalar_add(self):
    """Grad definition for `ScalarAdd` operation."""

    def bprop(x, y, out, dout):
        return dout, dout

    return bprop


@bprop_getters.register(_scalar_ops.ScalarSub)
def get_bprop_scalar_sub(self):
    """Grad definition for `ScalarSub` operation."""

    def bprop(x, y, out, dout):
        return dout, 0 - dout

    return bprop


@bprop_getters.register(_scalar_ops.ScalarMul)
def get_bprop_scalar_mul(self):
    """Grad definition for `ScalarMul` operation."""

    def bprop(x, y, out, dout):
        bc_dx = y * dout
        bc_dy = x * dout
        return bc_dx, bc_dy

    return bprop


@bprop_getters.register(_scalar_ops.ScalarDiv)
def get_bprop_scalar_div(self):
    """Grad definition for `ScalarDiv` operation."""

    def bprop(x, y, out, dout):
        bc_dx = dout / y
        bc_dy = 0 - bc_dx * out
        return bc_dx, bc_dy

    return bprop


@bprop_getters.register(_scalar_ops.ScalarFloordiv)
def get_bprop_scalar_floordiv(self):
    """Grad definition for `ScalarFloorDiv` operation."""

    def bprop(x, y, out, dout):
        return zeros_like(x), zeros_like(y)

    return bprop


@bprop_getters.register(_scalar_ops.ScalarMod)
def get_bprop_scalar_mod(self):
    """Grad definition for `ScalarMod` operation."""

    def bprop(x, y, out, dout):
        bc_dx = dout
        bc_dy = -dout * (x // y)
        return bc_dx, bc_dy

    return bprop


@bprop_getters.register(_scalar_ops.scalar_eq)
@bprop_getters.register(_scalar_ops.scalar_le)
@bprop_getters.register(_scalar_ops.scalar_lt)
@bprop_getters.register(_scalar_ops.scalar_ge)
@bprop_getters.register(_scalar_ops.scalar_gt)
@bprop_getters.register(_scalar_ops.bit_and)
@bprop_getters.register(_scalar_ops.bit_or)
def get_bprop_scalar_logic(self):
    """Grad definition for `ScalarLogicOps` operation."""

    def bprop(x, y, out, dout):
        return zeros_like(x), zeros_like(y)

    return bprop


@bprop_getters.register(_scalar_ops.ScalarBool)
def get_bprop_scalar_bool(self):
    """Grad definition for `ScalarBool` operation."""

    def bprop(x, out, dout):
        return zeros_like(x)

    return bprop
