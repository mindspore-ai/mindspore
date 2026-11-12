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

"""Internal primitives used by the stateful scatter functions and vmap."""
from mindspore.ops.primitive import prim_attr_register
from .array_ops import _ScatterOpDynamic


class ScatterMax(_ScatterOpDynamic):
    """Internal stateful scatter primitive."""

    @prim_attr_register
    def __init__(self, use_locking=False):
        """Initialize ScatterMax."""
        super().__init__(use_locking)


class ScatterMin(_ScatterOpDynamic):
    """Internal stateful scatter primitive."""

    @prim_attr_register
    def __init__(self, use_locking=False):
        """Initialize ScatterMin."""
        super().__init__(use_locking)


class ScatterMul(_ScatterOpDynamic):
    """Internal stateful scatter primitive."""

    @prim_attr_register
    def __init__(self, use_locking=False):
        """Initialize ScatterMul."""
        super().__init__(use_locking)


class ScatterDiv(_ScatterOpDynamic):
    """Internal stateful scatter primitive."""

    @prim_attr_register
    def __init__(self, use_locking=False):
        """Initialize ScatterDiv."""
        super().__init__(use_locking)
