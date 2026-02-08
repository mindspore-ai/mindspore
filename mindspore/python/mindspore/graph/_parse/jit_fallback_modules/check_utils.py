# This is the Python adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
#
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
""" Used to detect third party modules. """

import os
import sys
import types
import inspect
from mindspore import log as logger
from .third_party_modules import third_party_modules_whitelist
from ..resources import convert_object_map


class ThirdPartyLibraryChecker:
    """
    Check if a module or function is from third-party libraries.

    Rules for detecting third-party libraries:

    1. Python built-in modules and python standard libraries are third-party libraries.

    2. Modules in third_party_modules_whitelist are treated as third-party libraries.

    3. Third-party libraries usually have 'site-packages' in their installation path.

    4. Modules with module names provided by MS_JIT_IGNORE_MODULES are treated as third-party
       libraries, but those provided by MS_JIT_MODULES are not.

    """
    def __init__(self):
        self.python_builtin_dir = os.path.realpath(os.path.dirname(os.__file__))

    @staticmethod
    def get_jit_modules():
        """Modules in jit_modules require jit."""
        jit_modules = []
        # Get jit modules from environment variable.
        env_modules = os.getenv('MS_JIT_MODULES')
        if env_modules is not None:
            jit_modules = env_modules.split(',')
        return jit_modules

    @staticmethod
    def get_jit_ignore_modules():
        """Modules in jit_ignore_modules do not need jit."""
        jit_ignore_modules = []
        # Get jit ignore modules from environment variable.
        env_modules = os.getenv('MS_JIT_IGNORE_MODULES')
        if env_modules is not None:
            jit_ignore_modules = env_modules.split(',')
        # sys.builtin_module_names do not need jit.
        jit_ignore_modules.extend(sys.builtin_module_names)
        return jit_ignore_modules

    @staticmethod
    def in_convert_map(value):
        """Check if value in convert_object_map."""
        value_hashable = True
        try:
            hash(value)
        except TypeError:
            value_hashable = False
        return value_hashable and value in convert_object_map

    def is_third_party_module(self, module):
        """Check if module is a third-party library."""
        module_leftmost_name = module.__name__.split('.')[0]
        if module_leftmost_name == "mindspore":
            return False
        # Modules in jit_ignore_modules are treated as third-party libraries.
        jit_ignore_modules = self.get_jit_ignore_modules()
        if module_leftmost_name in jit_ignore_modules:
            logger.debug(f"Found third-party module '{module_leftmost_name}' in jit_ignore_modules.")
            return True
        # Modules in jit_modules require jit and they are considered to be in user workspace.
        jit_modules = self.get_jit_modules()
        if module_leftmost_name in jit_modules:
            logger.debug(f"Found user-defined module '{module_leftmost_name}' in jit_modules.")
            return False
        # A modules without __file__ attribute is considered to be in user workspace.
        if not hasattr(module, '__file__'):
            return False
        module_path = os.path.realpath(module.__file__)
        split_path = module_path.split(os.path.sep)
        under_site_packages = "site-packages" in split_path
        # Python builtin modules are treated as third-party libraries.
        if not under_site_packages and module_path.startswith(self.python_builtin_dir):
            logger.debug(f"Found python builtin module '{module.__name__}', which is a third-party module.")
            return True
        # Third-party modules are under site-packages.
        if under_site_packages and module_leftmost_name in third_party_modules_whitelist:
            logger.debug(f"Found third-party module '{module.__name__}' in path '{module_path}'")
            return True
        return False

    def is_from_third_party_module(self, value):
        """Check if value is from a third-party library."""
        if inspect.ismodule(value):
            return self.is_third_party_module(value)
        if (isinstance(value, types.FunctionType) and not hasattr(value, "__jit_function__")) or \
            (isinstance(value, types.MethodType) and not hasattr(value.__func__, "__jit_function__")):
            if self.in_convert_map(value):
                return False
            module = inspect.getmodule(value)
            return module is not None and self.is_third_party_module(module)
        return False


third_party_checker = ThirdPartyLibraryChecker()
