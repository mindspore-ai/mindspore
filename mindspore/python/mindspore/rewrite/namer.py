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
"""Unique name producer for target, name of node, class name, etc."""

from typing import Union, Tuple

from .node.node import Node
from .api.node_type import NodeType


class Namer:
    """
    Used for unique identity in a class-scope. current used for target of construct-function.
    Namer records times of name been used, and add prefix to origin name for unique name. For example, when a Namer
    record "name1" has been used 10 times, when a new request require a unique name base on 'name1', namer will respond
    "name1_10" as unique name.
    """

    def __init__(self):
        """Constructor of Namer."""
        self._names: {str: int} = {}

    @staticmethod
    def _real_name(name: str) -> Tuple[str, int]:
        """
        Find real name. For example, "name1" is the real name of "name1_10", "name1" is the real name of "name1_10_3".
        If not find real name before find unique name, unique name may be not unique. For example:

            1. "name1" has been used 10 times, which means "name1", "name1_2", "name1_3" ... "name1_10" has been used;
            2. new request require a unique name base on 'name1_5'
            3. If namer not find real name of "name1_5", namer will find that "name1_5" is never used and respond
            "name1_5" as unique name which is used before, actually.

        Args:
            name (str): Origin name which may have digit prefix.

        Returns:
            A string represents real-name and a int represents suffix.
        """
        if name == '_':
            return name, None
        pos = name.rfind("_")
        if pos == -1 or pos == len(name) - 1:
            return name, None
        digit = True
        for i in range(pos + 1, len(name)):
            if not name[i].isdigit():
                digit = False
                break
        if digit:
            return name[:pos], int(name[pos + 1:])
        return name, None

    def get_name(self, origin_name: str) -> str:
        """
        Get unique name from 'origin_name'.

        Args:
            origin_name (str): Origin name which may be duplicated.

        Returns:
            A string represents unique-name.
        """
        if origin_name == '_':
            return origin_name
        real_name, suffix_idx = Namer._real_name(origin_name)
        name = origin_name
        number = self._names.get(name)
        if number is None:
            self._names[name] = 1
            if not suffix_idx:
                # When _names is {x:2} and origin_name is y,
                # origin_name is not in _names and can be returned.
                return name
            if suffix_idx and not self._names.get(real_name, -1) >= suffix_idx:
                # When _names is {x:2} and origin_name is x_3,
                # return x_3 and update _names to {x:2, x_3:1}
                return name
            # When _names is {x:2} and origin_name is x_1,
            # set new_name to x_1_1 by set number to 1, and continue to update name.
            number = 1
        while True:
            new_name = f"{name}_{number}"
            number += 1
            self._names[name] = number
            # When _names is {x:2, x_3:1}, origin_name is x and number is update to 3,
            # new_name x_3 is conflict with key x_3, so this new_name need to be skipped.
            if new_name in self._names.keys():
                continue
            return new_name

    def add_name(self, name: str):
        """
        Add a name to Namer which should be unique.

        Args:
            name (str): A name should be unique in current namer.

        Raises:
            RuntimeError: If name is not unique in current namer.
        """
        if self._names.get(name) is None:
            self._names[name] = 1


class TargetNamer(Namer):
    """
    Used for unique-ing targets of node.
    """
    def get_unique_name(self, origin_name: str) -> str:
        """
        Get unique name from 'origin_name'.

        Args:
            origin_name (str): Origin name which may be duplicated.

        Returns:
            A string represents unique-name.
        """
        return super(TargetNamer, self).get_name(origin_name)


class NodeNamer(Namer):
    """
    Used for unique-ing node-name which is also used as field of init-function and key of global_vars
    """

    def get_name(self, node_or_name: Union[Node, str]) -> str:
        """
        Override get_name in Namer class.
        Get unique node_name from 'origin_name' or an instance of node.

        Args:
            node_or_name (Union[Node, str]): A string represents candidate node_name or an instance of node who require
                                             A unique node_name.

        Returns:
            A string represents unique node_name.
        """
        if isinstance(node_or_name, Node):
            origin_name = node_or_name.get_name()
            if origin_name is None or not origin_name:
                if node_or_name.get_node_type() in (NodeType.CallCell, NodeType.CallPrimitive, NodeType.CallFunction,
                                                    NodeType.Tree):
                    origin_name = type(node_or_name.get_instance()).__name__
                elif node_or_name.get_node_type() == NodeType.Python:
                    if node_or_name.get_instance():
                        origin_name = type(node_or_name.get_instance()).__name__
                    else:
                        origin_name = "python_node"
                elif node_or_name.get_node_type() == NodeType.Input:
                    origin_name = "parameter"
                elif node_or_name.get_node_type() == NodeType.Output:
                    origin_name = "return"
                elif node_or_name.get_node_type() == NodeType.MathOps:
                    origin_name = "math_ops"
                else:
                    raise RuntimeError("Node type unsupported:", node_or_name.get_node_type())
        elif isinstance(node_or_name, str):
            if not node_or_name:
                raise RuntimeError("input node_name is empty.")
            origin_name = node_or_name
        else:
            raise RuntimeError("unexpected type of node_or_name: ", type(node_or_name))
        return super(NodeNamer, self).get_name(origin_name)


class ClassNamer(Namer):
    """
    Used for unique-ing class name in a network.

    Class name should be unique in a network, in other word, in a Rewrite process. So please do not invoke constructor
    of `ClassNamer` and call `instance()` of `ClassNamer` to obtain singleton of ClassNamer.
    """

    def __init__(self):
        super().__init__()
        self._prefix = "Opt"

    @classmethod
    def instance(cls):
        """
        Class method of `ClassNamer` for singleton of `ClassNamer`.

        Returns:
            An instance of `ClassNamer` as singleton of `ClassNamer`.
        """

        if not hasattr(ClassNamer, "_instance"):
            ClassNamer._instance = ClassNamer()
        return ClassNamer._instance

    def get_name(self, origin_class_name: str) -> str:
        """
        Unique input `origin_class_name`.

        Args:
            origin_class_name (str): A string represents original class name.

        Returns:
            A string represents a unique class name generated from `origin_class_name`.
        """

        return super(ClassNamer, self).get_name(origin_class_name + self._prefix)

    def add_name(self, class_name: str):
        """
        Declare a `class_name` so that other class can not apply this `class_name` anymore.

        Args:
            class_name (str): A string represents a class name.
        """

        super(ClassNamer, self).add_name(class_name + self._prefix)
