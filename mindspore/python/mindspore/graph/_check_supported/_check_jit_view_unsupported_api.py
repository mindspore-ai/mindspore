# Copyright 2025 Huawei Technologies Co., Ltd
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
"""Check the view mint interface."""


def jit_view_unsupported(fn):
    """
    Use @jit_view_unsupported to decoratore the api which exhibits inconsistent view behavior between PYNATIVE
    mode and graph mode. The view mint interface which is not supported in graph mode, it might be implemented by
    fallback to other operators.
    """
    setattr(fn, '__jit_view_unsupported__', True)

    def mint_view(*args, **kwargs):
        return fn(*args, **kwargs)

    return mint_view
