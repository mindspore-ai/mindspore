# Copyright 2024 Huawei Technologies Co., Ltd
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
from types import FunctionType


class RecomputeRegistry:
    """Registry class for register recompute generator."""

    def __init__(self):
        self.recompute_generator = None

    def register(self, fn):
        """
        Register recompute generator function
        :param fn:
        :return:
        """
        if not isinstance(fn, FunctionType):
            raise TypeError("Fn should be function type, but got", type(fn))
        self.recompute_generator = fn

    def get(self):
        """
        Get recompute generator.
        :return:
        """
        if self.recompute_generator is None:
            raise TypeError("Recompute generator should be initialised before get()!")
        return self.recompute_generator


recompute_registry = RecomputeRegistry()
