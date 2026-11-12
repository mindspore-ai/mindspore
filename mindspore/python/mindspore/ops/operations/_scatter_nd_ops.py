# Copyright 2020-2023 Huawei Technologies Co., Ltd
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

"""Private primitives for the stateful scatter_nd functions and vmap."""
from mindspore import _checkparam as validator
from mindspore.ops import signature as sig
from mindspore.ops.primitive import Primitive, prim_attr_register
from .array_ops import _ScatterNdOp


class ScatterNdAdd(Primitive):
    """Internal stateful N-dimensional scatter primitive."""
    __mindspore_signature__ = (
        sig.make_sig('x', sig.sig_rw.RW_WRITE, dtype=sig.sig_dtype.T),
        sig.make_sig('indices', dtype=sig.sig_dtype.T1),
        sig.make_sig('updates', dtype=sig.sig_dtype.T)
    )

    @prim_attr_register
    def __init__(self, use_locking=False):
        """Initialize ScatterNdAdd."""
        validator.check_value_type('use_locking', use_locking, [bool], self.name)
        self.init_prim_io_names(inputs=['x', 'indices', 'updates'], outputs=['y'])
        self.add_prim_attr('side_effect_mem', True)


class ScatterNdSub(Primitive):
    """Internal stateful N-dimensional scatter primitive."""

    __mindspore_signature__ = (
        sig.make_sig('input_x', sig.sig_rw.RW_WRITE, dtype=sig.sig_dtype.T),
        sig.make_sig('indices', dtype=sig.sig_dtype.T1),
        sig.make_sig('updates', dtype=sig.sig_dtype.T)
    )

    @prim_attr_register
    def __init__(self, use_locking=False):
        """Initialize ScatterNdSub"""
        validator.check_value_type('use_locking', use_locking, [bool], self.name)
        self.init_prim_io_names(inputs=['input_x', 'indices', 'updates'], outputs=['y'])
        self.add_prim_attr('side_effect_mem', True)


class ScatterNdMul(_ScatterNdOp):
    """Internal stateful N-dimensional scatter primitive."""

    @prim_attr_register
    def __init__(self, use_locking=False):
        """Initialize ScatterNdMul."""
        super().__init__(use_locking)


class ScatterNdDiv(_ScatterNdOp):
    """Internal stateful N-dimensional scatter primitive."""

    @prim_attr_register
    def __init__(self, use_locking=False):
        """Initialize ScatterNdDiv."""
        super().__init__(use_locking)


class ScatterNdMax(_ScatterNdOp):
    """Internal stateful N-dimensional scatter primitive."""

    @prim_attr_register
    def __init__(self, use_locking=False):
        """Initialize ScatterNdMax"""
        super().__init__(use_locking)


class ScatterNdMin(_ScatterNdOp):
    """Internal stateful N-dimensional scatter primitive."""

    @prim_attr_register
    def __init__(self, use_locking=False):
        """Initialize ScatterNdMin"""
        super().__init__(use_locking)
