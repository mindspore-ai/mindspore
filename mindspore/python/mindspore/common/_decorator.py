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

from __future__ import absolute_import
from functools import wraps
from mindspore import log


DEPRECATE_SET = set()


def deprecated(version, substitute, use_substitute_name=False):
    """deprecated warning

    Args:
        version (str): version that the operator or function is deprecated.
        substitute (str): the substitute name for deprecated operator or function.
        use_substitute_name (bool): flag for whether to use substitute name for deprecated operator or function
    """

    def decorate(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cls = getattr(args[0], "__class__", None) if args else None
            name = cls.__name__ if cls else func.__name__
            if name + version not in DEPRECATE_SET:
                log.warning(f"'{name}' is deprecated from version {version} and "
                            f"will be removed in a future version, use '{substitute}' instead.")
                DEPRECATE_SET.add(name + version)
            if cls and use_substitute_name:
                cls.substitute_name = substitute
            ret = func(*args, **kwargs)
            return ret

        return wrapper

    return decorate
