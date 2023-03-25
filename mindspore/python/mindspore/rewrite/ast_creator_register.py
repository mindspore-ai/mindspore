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

"""Registry the relation."""

from __future__ import absolute_import
from collections import UserDict


class Registry(UserDict):
    """Registry class for registry functions for creating ast node."""

    def register(self, obj_str, obj):
        """Register object by str."""
        if isinstance(obj_str, str):
            self[obj_str] = obj

    def get(self, obj_str):
        """Get the value by str."""
        if not isinstance(obj_str, str):
            raise TypeError("key for ast creator registry must be string.")
        return self[obj_str]


ast_creator_registry = Registry()
