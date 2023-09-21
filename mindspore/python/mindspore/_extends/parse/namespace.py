# This is the Python adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
#
# Copyright 2020-2023 Huawei Technologies Co., Ltd
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
"""Define the namespace of parse."""

from __future__ import absolute_import
import builtins

from mindspore import log as logger


class Namespace:
    """
    Base class of namespace for resolve variables.

    Args:
        name (str): The namespace's name.
        dicts (dict): A list of dict containing the namespace's variable.
    """

    def __init__(self, name, *dicts):
        self.name = name
        self.dicts = dicts

    def __contains__(self, name):
        for d in self.dicts:
            if name in d:
                return True
        return False

    def __getitem__(self, name):
        for d in self.dicts:
            if name in d:
                return d[name]
        raise NameError(name)

    def __repr__(self):
        return f'Namespace:{self.name}'


class CellNamespace(Namespace):
    """
    Namespace for Cell object.

    Args:
        name (str): Valid module name, it can be imported.
    """

    def __init__(self, name):
        mod_dict = vars(__import__(name, fromlist=['_']))
        builtins_dict = vars(builtins)
        super().__init__(name, mod_dict, builtins_dict)

    def __getstate__(self):
        return (self.name,)

    def __setstate__(self, state):
        name, = state
        mod_dict = vars(__import__(name, fromlist=['_']))
        builtins_dict = vars(builtins)
        super().__init__(name, mod_dict, builtins_dict)


class ClosureNamespace(Namespace):
    """
    Namespace for function closure.

    Args:
        fn (Function): A python function.
    """

    def __init__(self, fn):
        name = f'{fn.__module__}..<{fn.__name__}>'
        names = fn.__code__.co_freevars
        cells = fn.__closure__
        ns = dict(zip(names, cells or ()))
        super().__init__(name, ns)

    def __getitem__(self, name):
        d, = self.dicts
        try:
            return d[name].cell_contents
        except ValueError:
            raise UnboundLocalError(name)


class ClassMemberNamespace(Namespace):
    """
    Namespace of a class's closure.

    Args:
        obj (Object): A python class object.
    """

    def __init__(self, obj):
        self.__class_member_namespace__ = True
        label = f'{obj.__module__}..<{obj.__class__.__name__}::{id(obj)}>'
        super().__init__(label, obj)

    def __getitem__(self, name):
        d, = self.dicts
        if name == "namespace":
            return self
        try:
            if hasattr(d, name):
                return getattr(d, name)
            return d.__dict__[name]
        except ValueError:
            raise UnboundLocalError(name)
        except KeyError:
            cls = d.__class__
            # Class private attribute.
            if name.startswith("__"):
                private_member = "_" + cls.__name__ + name
                if hasattr(d, private_member):
                    logger.warning(f"The private attribute or method '{name}' is used in '{cls.__name__}'. " + \
                                   f"In graph mode, '{name}' will be adjusted to '{private_member}' for parsing.")
                    return getattr(d, private_member)
            logger.info(f"'{cls.__name__}' object has no attribute or method: '{name}', so will return None.")
            raise AttributeError(name)

    def __getattr__(self, name):
        return self.__getitem__(name)
