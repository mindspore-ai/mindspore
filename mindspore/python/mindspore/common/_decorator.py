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


def deprecated(version, substitute, use_substitute_name=False,
               module_prefix=""):
    """deprecated warning

    Args:
        version (str): version that the operator or function is deprecated.
        substitute (str): the substitute name for deprecated operator or function.
            If ``None`` or empty string, the warning message will not include
            the "use XXX instead" suggestion.
        use_substitute_name (bool): flag for whether to use substitute name for
            deprecated operator or function.
        module_prefix (str): the module prefix of the deprecated api, such as
            'mindspore.'.
    """

    def decorate(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cls = getattr(args[0], "__class__", None) if args else None
            # NOTE:
            # - Many functional APIs take Tensor as the first arg, but they are
            #   NOT Tensor instance methods. We must not infer "Tensor.<method>"
            #   purely from args[0].
            # - For true Tensor/COOTensor/CSRTensor instance methods,
            #   func.__qualname__ is like "Tensor.ptp".
            qualname = getattr(func, "__qualname__", "") or ""
            if qualname.startswith(("Tensor.", "COOTensor.", "CSRTensor.")):
                name = qualname
            elif "." in qualname:
                # For instance methods, use class name by default (keeps
                # backward behavior for operators like Primitive.__init__).
                name = cls.__name__ if cls else func.__name__
            else:
                # For plain functions, use function name.
                name = func.__name__
            if name + version not in DEPRECATE_SET:
                base_msg = (
                    f"'{module_prefix}{name}' is deprecated from version {version}"
                    f" and "
                    f"will be removed in a future version"
                )
                if substitute:
                    log.warning(f"{base_msg}, use '{substitute}' instead.")
                else:
                    log.warning(f"{base_msg}.")
                DEPRECATE_SET.add(name + version)
            if cls and use_substitute_name and substitute:
                cls.substitute_name = substitute
            ret = func(*args, **kwargs)
            return ret

        return wrapper

    return decorate
