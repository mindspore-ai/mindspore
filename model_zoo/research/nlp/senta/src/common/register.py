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
"""
:py:class: Register modules for import
"""
import importlib

import logging
import os


class Register():
    """Register"""

    def __init__(self, registry_name):
        self._dict = {}
        self._name = registry_name

    def __setitem__(self, key, value):
        if not callable(value):
            raise Exception("Value of a Registry must be a callable.")
        if key is None:
            key = value.__name__
        self._dict[key] = value

    def register(self, param):
        """Decorator to register a function or class."""

        def decorator(key, value):
            """decorator"""
            self[key] = value
            return value

        if callable(param):
            # @reg.register
            return decorator(None, param)
        # @reg.register('alias')
        return lambda x: decorator(param, x)

    def __getitem__(self, key):
        try:
            return self._dict[key]
        except Exception as e:
            logging.error("module {key} not found: {e}")
            raise e

    def __contains__(self, key):
        return key in self._dict

    def keys(self):
        """key"""
        return self._dict.keys()


class RegisterSet():
    """RegisterSet"""
    field_reader = Register("field_reader")
    data_set_reader = Register("data_set_reader")
    models = Register("models")
    tokenizer = Register("tokenizer")
    trainer = Register("trainer")

    package_names = ['src.data.field_reader', 'src.data.data_set_reader', 'src.data.tokenizer',
                     'src.models', 'src.training']
    ALL_MODULES = []
    for package_name in package_names:
        module_dir = os.path.join(
            os.path.abspath(
                os.path.dirname(__file__)),
            "../../" +
            package_name.replace(
                ".",
                '/'))
        module_files = []
        for file in os.listdir(module_dir):
            if os.path.isfile(os.path.join(module_dir, file)
                              ) and file.endswith(".py"):
                module_files.append(file.replace(".py", ""))
        ALL_MODULES.append((package_name, module_files))


def import_modules():
    """import modules needed
    :return:
    """
    for base_dir, modules in RegisterSet.ALL_MODULES:
        for name in modules:
            try:
                if base_dir != "":
                    full_name = base_dir + "." + name
                else:
                    full_name = name
                importlib.import_module(full_name)

            except ImportError:
                logging.error("error in import modules")
