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
"""Define the namespace of MindSpore op definition."""
import os
import sys
import inspect
import types
from mindspore._extends.parse.namespace import ModuleNamespace
from mindspore.nn import CellList, SequentialCell


_ms_common_ns = ModuleNamespace('mindspore.common')
_ms_nn_ns = ModuleNamespace('mindspore.nn')
_ms_ops_ns = ModuleNamespace('mindspore.ops.operations')
_ms_functional_ns = ModuleNamespace('mindspore.ops.functional')

# Elements in _subtree_black_list will not be converted to symbol tree.
# Only str and types are stored in _subtree_black_list.
_subtree_black_list = [CellList, SequentialCell]
# Whether to convert mindspore built-in cells to symbol tree.
_ms_cells_to_subtree = False
# Paths of modules which will not be considered as third party module
_ignore_third_party_paths = []

def is_subtree(cls_inst):
    """Determine whether 'cls_inst' is a subtree."""
    cls_name = type(cls_inst).__name__
    if isinstance(cls_inst, tuple(_subtree_black_list)):
        return False
    if cls_name in _ms_common_ns and isinstance(cls_inst, _ms_common_ns[cls_name]):
        return False
    if cls_name in _ms_nn_ns and isinstance(cls_inst, _ms_nn_ns[cls_name]):
        return bool(_ms_cells_to_subtree)
    if cls_name in _ms_ops_ns and isinstance(cls_inst, _ms_ops_ns[cls_name]):
        return False
    return True


def is_ms_function(func_obj):
    """Determine whether 'func_obj' is a mindspore function."""
    if isinstance(func_obj, types.BuiltinFunctionType):
        return False
    try:
        # module, class, method, function, traceback, frame, or code object was expected
        func_file = inspect.getabsfile(func_obj)
    except TypeError:
        return False
    func_file = os.path.normcase(func_file)
    ms_module = sys.modules.get('mindspore')
    if ms_module is None:
        return False
    ms_path = ms_module.__file__
    ms_path = os.path.normcase(ms_path)
    ms_path = ms_path.rsplit(os.path.sep, 1)[0]
    return func_file.startswith(ms_path)


def is_functional(func_name):
    """Determine whether 'cls_name' is a functional."""
    return func_name in _ms_functional_ns


def get_functional(func_name):
    """Get the function corresponding to the func_name."""
    if func_name in _ms_functional_ns:
        return _ms_functional_ns[func_name]
    return None


def is_third_party(func_obj):
    """Check whether func_obj is from third party module"""
    module = inspect.getmodule(func_obj)
    # A module without __file__ attribute (normally to be a c++ lib) is considered to be third party module.
    if not hasattr(module, '__file__'):
        return True
    module_path = os.path.abspath(module.__file__)
    for path in _ignore_third_party_paths:
        if module_path.startswith(path):
            return False
    # Python builtin modules are treated as third-party libraries.
    python_builtin_dir = os.path.abspath(os.path.dirname(os.__file__))
    if module_path.startswith(python_builtin_dir):
        return True
    # Check if module is under user workspace directory.
    user_workspace_dir = get_top_level_module_path(os.getcwd())
    if module_path.startswith(user_workspace_dir):
        return False
    # Third-party modules are under site-packages.
    split_path = module_path.split(os.path.sep)
    if "site-packages" in split_path:
        return True
    return False


def get_top_level_module_path(module_path):
    """Get the path of the top level package of the current working directory."""
    module_abspath = os.path.abspath(module_path)
    upper_path = os.path.abspath(os.path.dirname(module_abspath))
    if module_abspath == upper_path:
        return module_abspath
    # Check whether __init__.py exists in the upper directory.
    init_path = os.path.join(upper_path, '__init__.py')
    # If the path does not exist or is accessed without permission, os.path.isfile returns false.
    if os.path.isfile(init_path):
        module_abspath = get_top_level_module_path(upper_path)
    return module_abspath
