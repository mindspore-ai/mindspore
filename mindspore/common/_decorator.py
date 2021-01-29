# Copyright 2021 Huawei Technologies Co., Ltd
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
"""Providing decorators."""


def deprecated(version, substitute):
    """deprecated warning

    Args:
        version (str): version that the operator or function will be deprecated.
        substitute (str): the substitute name for deprecated operator or function.
    """

    def decorate(func):
        def wrapper(*args, **kwargs):
            cls = getattr(args[0], "__class__", None) if args else None
            name = cls.__name__ if cls else func.__name__
            print(f"WARNING: '{name}' is deprecated from version {version} and will be removed in a future version, "
                  f"use '{substitute}' instead.")
            ret = func(*args, **kwargs)
            return ret

        return wrapper

    return decorate
