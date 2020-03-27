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

"""Registry the relation."""

from collections import UserDict
from .primitive import Primitive


class Registry(UserDict):
    """Registry class for registry functions for grad and vm_impl on Primitive."""

    def register(self, prim):
        def deco(fn):
            """Decorate the function."""
            if isinstance(prim, str):
                self[prim] = fn
            elif issubclass(prim, Primitive):
                self[id(prim)] = fn
            return fn
        return deco

    def get(self, prim_obj, default):
        """Get the value by primitive."""
        fn = default
        if isinstance(prim_obj, str) and prim_obj in self:
            fn = self[prim_obj]
        elif isinstance(prim_obj, Primitive):
            key = id(prim_obj.__class__)
            if key in self:
                fn = self[key]
            else:
                key = prim_obj.name
                if key in self:
                    fn = self[prim_obj.name]
        return fn
