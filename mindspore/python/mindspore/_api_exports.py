# Copyright 2026 Huawei Technologies Co., Ltd
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
"""Internal helpers for controlling package aliases."""


def _remove_exports(namespace, names):
    """Remove aliases and their wildcard exports without changing defining modules."""
    excluded = frozenset(names)
    namespace["__all__"] = [name for name in namespace["__all__"] if name not in excluded]
    for name in excluded:
        namespace.pop(name, None)
