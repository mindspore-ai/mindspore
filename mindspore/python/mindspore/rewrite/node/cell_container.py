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
"""CellContainer Node."""
import ast
from mindspore import log as logger
from .node import Node
from .node_manager import NodeManager
from ..api.scoped_value import ScopedValue
from ..api.node_type import NodeType


class CellContainer(Node, NodeManager):
    """CellContainer is used for nn.SequencialCell."""

    def __init__(self, ast_node: ast.AST, targets: [ScopedValue], func_name: ScopedValue,
                 args: [ScopedValue], kwargs: {str: ScopedValue}, node_name: str, stree, instance):
        """Constructor of CellContainer.

        Args:
            ast_node (ast.AST): An instance of ast.AST represents corresponding node in ast.
            targets (list[ScopedValue]): A list of instance of ScopedValue. See detail in docstring of Node class.
            func_name ([ScopedValue, optional]): An instance of ScopedValue. See detail in docstring of Node class.
            args (list[ScopedValue]): A list of instance of ScopedValue. See detail in docstring of Node class.
            kwargs (dict{str: ScopedValue}): A list of instance of ScopedValue. See detail in docstring of Node class.
            node_name (str): A string represents name of node. Name of node will be unique when inserted into
                SymbolTree. Name of node also used as field name in network class.
            stree (SymbolTree): Symbol tree used to get node_namer.
            instance: Object in network corresponding to this node.
        """
        if isinstance(func_name, str):
            func_name = ScopedValue.create_naming_value(func_name)
        Node.__init__(self, NodeType.CellContainer, ast_node, targets, func_name, args, kwargs, node_name, instance)
        NodeManager.__init__(self, stree.get_node_namer())
        NodeManager.set_manager_name(self, func_name.value)

    def append(self, node, insert_to_ast: bool = True):
        """ Append new node to node list. """
        self.append_node(node, insert_to_ast)

    def append_node(self, node, insert_to_ast: bool = True):
        """ Append new node to node list. """
        self.insert_node(node, self.get_tail(), False, insert_to_ast)

    def erase(self, node):
        """Erase node from container."""
        self.erase_node(node)

    def erase_node(self, node):
        """Erase node from container."""
        # add code `del self.container_name[node_index]` into __init__ function
        _, init_ast_functiondef = self._get_stree_and_init_ast()
        if not init_ast_functiondef:
            logger.error(f"Erase node {node.get_name()} failed: get symboltree and __init__ ast failed.")
            return
        node_idx = self.nodes().index(node)
        erase_code = f"del {self.get_func_name()}[{node_idx}]"
        erase_ast = ast.parse(erase_code).body[0]
        init_ast_functiondef.body.append(erase_ast)
        # earse node in NodeManager
        NodeManager.erase_node(self, node)

    def insert(self, index, node, insert_to_ast: bool = True):
        """Insert node into container according index"""
        node_index = index + len(self._inputs)
        if node_index >= self.node_count:
            raise IndexError("In MindSpore Rewrite CellContainer, inserting a node raises index error! "
                             f"node_index: {node_index} >= node_num: {self.node_count}")
        self.insert_node(node, self.nodes()[node_index], False, insert_to_ast)

    def insert_node(self, new_node: Node, base_node: Node, before_node: bool, insert_to_ast: bool = True):
        """
        Insert a node before or after base_node.

        The instance is modified here. The scenario needs to be optimized.

        Args:
            new_node (Node): Node to be inserted.
            base_node (Node): New node will be inserted before or after base_node.
            before_node (bool): Indicate whether new node is inserted before base_node.
            insert_to_ast (bool): Indicate whether ast nodes need to be updated.
        """
        # Insert node to NodeManager firstly to update node_name, which is used during insert ast.
        # tail_node may be changed after insert node into node_manager, so we record tail node here.
        tail_node = self.get_tail()
        NodeManager.insert_node(self, new_node, base_node, before_node)
        new_node.set_func_name(ScopedValue.create_naming_value(new_node.get_name()))
        new_node.update_ast_node()
        # add insert/append code into __init__ function
        if insert_to_ast:
            stree, init_ast_functiondef = self._get_stree_and_init_ast()
            if not init_ast_functiondef:
                logger.error(f"Insert new_node {new_node.get_name()} failed: get symboltree and __init__ ast failed.")
                return
            setattr(stree.get_origin_network(), new_node.get_name(), new_node.get_instance())
            node_idx = self.nodes().index(base_node)
            if before_node:
                insert_code = f"{self.get_func_name()}._insert({node_idx}, self.{new_node.get_name()})"
            else:
                if base_node == tail_node:
                    insert_code = f"{self.get_func_name()}.append(self.{new_node.get_name()})"
                else:
                    insert_code = f"{self.get_func_name()}._insert({node_idx + 1}, self.{new_node.get_name()})"
            insert_ast = ast.parse(insert_code).body[0]
            init_ast_functiondef.body.append(insert_ast)

    def set_belong_symbol_tree(self, symbol_tree):
        """Set the symbol tree to which node belongs."""
        self._belong_tree = symbol_tree
        for node in self.nodes():
            node.set_belong_symbol_tree(symbol_tree)

    def _get_stree_and_init_ast(self):
        """Get symbol tree and ast of __init__ function from container."""
        # add codes `del self.container_name[node_index]`` into __init__ function
        stree = self.get_belong_symbol_tree()
        if stree is None:
            logger.error(f"Get symboltree of CellContainer {self.get_name()} failed.")
            return None, None
        init_ast_functiondef = stree.get_init_func_ast()
        if init_ast_functiondef is None:
            logger.error(f"Get ast of __init__ function from class {stree.get_opt_cls_name()} failed.")
            return None, None
        return stree, init_ast_functiondef
