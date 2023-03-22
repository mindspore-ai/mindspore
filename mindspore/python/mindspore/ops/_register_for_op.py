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
from __future__ import absolute_import
from __future__ import division

from collections import UserDict

from mindspore.ops.primitive import Primitive


class Registry(UserDict):
    """Registry class for registry functions for grad and vm_impl on Primitive."""

    def register(self, prim):
        """register the function."""
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


class PyFuncRegistry(UserDict):
    def register(self, key, value):
        self[key] = value

    def get(self, key):
        if key not in self:
            raise ValueError(f"Python function with key{key} not registered.")
        return self[key]


class OpaquePredicateRegistry(PyFuncRegistry):
    """Registry opaque predicate functions used for dynamic obfuscation"""
    def __init__(self):
        super(OpaquePredicateRegistry, self).__init__()
        self.func_names = []

    def register(self, key, value):
        self[key] = value
        self.func_names.append(key)
