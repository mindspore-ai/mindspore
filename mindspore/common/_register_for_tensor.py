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
from .. import context


class Registry(UserDict):
    """Registry class for registry functions for tensor call primitive ops function."""

    def register(self, obj_str, obj):
        if isinstance(obj_str, str):
            self[obj_str] = obj

    def get(self, obj_str):
        """Get the value by str."""
        if not isinstance(obj_str, str):
            raise TypeError("key for tensor registry must be string.")
        if context.get_context("enable_ge"):
            def wrap(*args):
                new_args = list(args)
                new_args.append(obj_str)
                return self["vm_compare"](*new_args)

            obj = wrap
        else:
            obj = self[obj_str]
        return obj


tensor_operator_registry = Registry()
