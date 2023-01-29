# Copyright 2023 Huawei Technologies Co., Ltd
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
"""Check jit forbidden api."""

import types

from mindspore import log as logger

# module: such as "mindspore.common.initializer"
_jit_forbidden_module = set()


def jit_forbidden_register(fn):
    setattr(fn, '__jit_forbidden__', True)
    def jit_forbidden(*args, **kwargs):
        return fn(*args, **kwargs)
    return jit_forbidden


def add_jit_forbidden_module(jit_forbidden_module):
    logger.debug(f'add jit_forbidden_module_set: {_jit_forbidden_module}')
    return _jit_forbidden_module.add(jit_forbidden_module)


def remove_jit_forbidden_module(jit_forbidden_module):
    logger.debug(f'remove jit_forbidden_module_set: {_jit_forbidden_module}')
    return _jit_forbidden_module.remove(jit_forbidden_module)


def get_jit_forbidden_module(jit_forbidden_module):
    logger.debug(f'get jit_forbidden_module_set: {_jit_forbidden_module}')
    return _jit_forbidden_module


def get_obj_module_and_name_info(obj):
    """Return the description of the object whose type is class, function or method."""
    if not hasattr(obj, "__module__"):
        return None
    if isinstance(obj, (types.FunctionType, types.MethodType)):
        return obj.__module__, obj.__qualname__, "method or function"
    return obj.__module__, obj.__name__, "class"


def is_jit_forbidden_module(obj_module):
    """Return the matching result of object module in jit forbidden module set."""
    if obj_module in _jit_forbidden_module:
        return True
    return False

add_jit_forbidden_module("mindspore.common.initializer")
