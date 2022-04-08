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
"""SymbolTree class define of Rewrite according to forward function of a network."""
from typing import Optional, Union, Tuple
import os
import sys
import ast
import tempfile
import astunparse

from mindspore.nn import Cell
from mindspore import log as logger
from .node import Node, TreeNode
from .api.node_type import NodeType
from .ast_helpers import AstModifier, AstReplacer
from .api.scoped_value import ScopedValue, ValueType
from .symbol_tree_dumper import SymbolTreeDumper
from .topological_manager import TopoManager
from .namer import TargetNamer, NodeNamer, ClassNamer
from .common.observer import Observer


class Position:
    """
    Position indicates a source code position in one network.

    Rewrite recommend using class method `create()` of position rather than constructor of Position.

    Args:
        symbol_tree (SymbolTree): A handler of SymbolTree indicated position in which SymbolTree.
        node (Node): A handler of Node indicated position is around which Node.
        before_node (bool): A bool indicated position is before or after the 'node'.
    """

    def __init__(self, symbol_tree, node, before_node: bool):
        self.symbol_tree = symbol_tree
        self.node = node
        self.before_node = before_node

    @classmethod
    def create(cls, symbol_tree, node, before_node):
        """
        Class method of Position. Return None when symbol_tree or node is None.

        Args:
            symbol_tree: A handler of SymbolTree indicated position in which SymbolTree.
            node: A handler of Node indicated position is around which Node.
            before_node (bool): A bool indicated position is before or after the 'node'.

        Returns:
            A Position.
        """
        if symbol_tree is None or node is None:
            return None
        return Position(symbol_tree, node, before_node)


class SymbolTree(Observer):
    """
    A symbol-tree usually corresponding to forward method of a network.

    Rewrite recommend using SymbolTreeBuilder to instantiate an instance of SymbolTree rather than invoking constructor
    of SymbolTree directly.

    Args:
        origin_network (Cell): A handler to original network instance.
        module_ast (ast.Module): An instance of ast.AST represents ast node of original network.
    """

    def __init__(self, origin_network: Cell, module_ast: ast.Module):
        super().__init__()
        origin_network_key = "handler"
        # init unique-namers
        self._target_namer = TargetNamer()
        self._node_name_namer = NodeNamer()
        # name or node would use as name of field, so name of origin network handler field should be added into \
        # _node_name_namer.
        self._node_name_namer.add_name(origin_network_key)
        self._topo_mgr = TopoManager()
        self._topo_mgr.reg_observer(self)

        self._global_vars: {str, object} = {origin_network_key: origin_network}
        self._nodes: {str, Node} = {}
        # parameters of forward method
        self._inputs: [Node] = []
        self._ori_cls_name = type(origin_network).__name__
        self._opt_cls_name = ClassNamer.instance().get_name(self._ori_cls_name)
        self._origin_network = origin_network
        self._module_ast: ast.Module = module_ast
        self._class_ast: Optional[ast.ClassDef] = None
        self._root_ast: Optional[ast.FunctionDef] = None
        self._init_func_ast: Optional[ast.FunctionDef] = None

        # head node is always point to the first node(in source code order) of SymbolTree
        self._head = None
        # tail node is always point to the last node(in source code order) of SymbolTree
        self._tail = None
        self._return: Optional[Node] = None

        self._modified = False

    def _on_change(self):
        self._modified = True

    def get_ori_cls_name(self) -> str:
        """
        Get class name of original network.

        Returns:
            A str represents class name of original network.
        """
        return self._ori_cls_name

    def get_opt_cls_name(self) -> str:
        """
        Get class name of rewritten network.

        Returns:
            A str represents class name of rewritten network.
        """
        return self._opt_cls_name

    def get_module_ast(self):
        """
        Getter of `_module_ast`.

        Returns:
            An instance of ast.AST represents ast node of corresponding module.
        """
        return self._module_ast

    def set_module_ast(self, ast_node: ast.Module):
        """
        Setter of _module_ast.

        Args:
            ast_node (ast.Module): An instance of ast.Module represents ast node of module of corresponding network
                                   class.
        """
        self._module_ast = ast_node

    def get_ast_root(self):
        """
        Getter of `_root_ast`.

        Returns:
            An instance of ast.AST represents ast node of corresponding forward method.
        """
        return self._root_ast

    def set_ast_root(self, ast_node: ast.FunctionDef):
        """
        Setter of _root_ast.

        Args:
            ast_node (ast.FunctionDef): An instance of ast.FunctionDef represents ast node of forward method of
                                        corresponding network class.
        """
        self._root_ast = ast_node

    def get_class_ast(self):
        """
        Getter of `_class_ast`.

        Returns:
            An instance of ast.ClassDef represents ast node of corresponding network class.
        """
        return self._class_ast

    def set_class_ast(self, ast_node: ast.ClassDef):
        """
        Setter of `_init_func_ast`.

        Args:
            ast_node (ast.ClassDef): An instance of ast.ClassDef represents ast node of corresponding network class.
        """
        self._class_ast = ast_node

    def get_init_func_ast(self):
        """
        Getter of _init_func_ast.

        Returns:
            An instance of ast.FunctionDef represents ast node of init method of corresponding network class.
        """
        return self._init_func_ast

    def set_init_func_ast(self, ast_node: ast.FunctionDef):
        """
        Setter of _init_func_ast.

        Args:
            ast_node (ast.FunctionDef): An instance of ast.FunctionDef represents ast node of init method of
                                        corresponding network class.
        """
        self._init_func_ast = ast_node

    def get_inputs(self):
        """
        Getter of `_inputs` which represents parameters of current forward method.

        Returns:
            A list of instance of Node whose node_type is NodeType.Input as input nodes.
        """
        return self._inputs

    def get_head_node(self):
        """
        Getter of `_head` which represents the beginning node while iterating SymbolTree nodes.

        Returns:
            An instance of node.
        """
        return self._head

    def get_return_node(self):
        """
        Getter of `_return` which represents return statement of forward method of network.

        Returns:
            An instance of node.
        """
        return self._return

    def get_origin_network(self):
        """
        Getter of `_origin_network`.

        Returns:
            An instance of Cell which represents original network.
        """
        return self._origin_network

    def get_global_vars(self):
        return self._global_vars

    def add_global_vars(self, key: str, value):
        if self._global_vars.get(key) is not None:
            raise RuntimeError("Key of global_vars duplicated:", key)
        self._global_vars[key] = value

    def nodes(self, unfold_subtree=False):
        """
        Getter of nodes if current SymbolTree.

        Args:
            unfold_subtree (bool): Need to iterate into sub-symbol-tree recursively.

        Returns:
            A list of instance of Nodes.
        """
        if unfold_subtree:
            nodes = []
            for _, v in self._nodes.items():
                if isinstance(v, TreeNode):
                    nodes.extend(v.symbol_tree.nodes())
                else:
                    nodes.append(v)
            return nodes
        return self._nodes.values()

    def get_node(self, node_name: str) -> Optional[Node]:
        """
        Get node of current symbol_tree by `node_name`.

        Args:
            node_name (str): A str represents name of node as key of query.

        Returns:
            An instance of Node if found else None.
        """

        return self._nodes.get(node_name)

    def _get_real_node(self, node_or_name: Union[Node, str]) -> Optional[Node]:
        if isinstance(node_or_name, Node):
            result = self.get_node(node_or_name.get_name())
            return result if result is node_or_name else None
        if isinstance(node_or_name, str):
            return self.get_node(node_or_name)
        return None

    def get_node_inputs(self, node_or_name: Union[Node, str]) -> [Node]:
        """
        Getter of inputs in topological relation of current 'node_or_name'.

        Args:
            node_or_name (Union[Node, str]): An instance of node or a str represents name of node.

        Returns:
            A list of instances of Node as input nodes if 'node_or_name' belong to current SymbolTree. An empty list if
            'node_or_name' not belong to current SymbolTree.
        """

        real_node: Optional[Node] = self._get_real_node(node_or_name)
        if real_node is None:
            logger.info("Node(%s) is not belong to current SymbolTree", node_or_name)
            return []
        return node_or_name.get_inputs()

    def get_node_users(self, node_or_name: Union[Node, str]) -> [Tuple[Node, int]]:
        """
        Getter of outputs in topological relation of current 'node_or_name'.

        Args:
            node_or_name (Union[Node, str]): An instance of node or a str represents name of node.

        Returns:
            A list of instances of Node as output nodes if 'node_or_name' belong to current SymbolTree. An empty list if
            'node_or_name' not belong to current SymbolTree.
        """

        real_node: Optional[Node] = self._get_real_node(node_or_name)
        if real_node is None:
            logger.info("Node(%s) is not belong to current SymbolTree", node_or_name)
            return []
        return self._topo_mgr.get_node_users(node_or_name)

    def before(self, node_or_name: Union[Node, str]) -> Position:
        """
        Get insert position before 'node_or_name' in source code list.
        Consider using symbol_tree, node and before/after as position for sub-tree feature.

        Note:
            Topological order is not determined here which is determined by arguments of node and updated by
            TopologicalManager automatically.

        Args:
            node_or_name (Union[Node, str]): An instance of node or a str represents name of node.

        Returns:
            A Position represents an insert point.

        Raises:
            AssertError: If 'node_or_name' is not a Node or a str
            RuntimeError: If 'node_or_name' is not belong to this SymbolTree or any sub-SymbolTree of current
                SymbolTree.
        """

        node = self._get_real_node(node_or_name)
        if node is None:
            raise RuntimeError("Node is not belong to current SymbolTree: ", node_or_name)
        return Position.create(node.get_belong_symbol_tree(), node, True)

    def after(self, node_or_name: Union[Node, str]) -> Position:
        """
        Get insert position after 'node_or_name' in source code list.
        Consider using symbol_tree, node and before/after as position for sub-tree feature.

        Note:
            Topological order is not determined here which is determined by arguments of node and updated by
            TopologicalManager automatically.

        Args:
            node_or_name (Union[Node, str]): An instance of node or a str represents name of node.

        Returns:
            A Position represents an insert point.

        Raises:
            AssertError: If 'node_or_name' is not a Node or a str
            RuntimeError: If 'node_or_name' is not belong to this SymbolTree or any sub-SymbolTree of current
                SymbolTree.
        """

        node = self._get_real_node(node_or_name)
        if node is None:
            raise RuntimeError("Node is not belong to current SymbolTree: ", node_or_name)
        return Position.create(node.get_belong_symbol_tree(), node, False)

    def insert_node(self, position: Position, node: Node, insert_to_ast: bool = True) -> Node:
        """
        Insert a node into SymbolTree.
        Note:
            Name of node will be unique while inserting node into SymbolTree.

            ValueType.CustomObjValue type arguments will be converted to ValueType.NamingValue and custom object will
            be saved in global_vars dict while inserting node into SymbolTree.

            Targets of node will be unique while inserting node into SymbolTree.

            A field instantiation statement will be added into "init" function of network class using node name as field
            name when `insert_to_ast` is True while inserting node into SymbolTree.

            An assign statement represents invoking to this node will be added into forward function of network class
            corresponding to field-instantiation-statement when `insert_to_ast` is True while inserting node into
            SymbolTree.

            Topological relation is updated and inputs of corresponding node is updated.

        Args:
            position (Position): A Position indicates an insert position point.
            node (Node): An instance of node to be inserted in.
            insert_to_ast (bool): A bool indicates whether to update corresponding ast node at same time, default is
                True.

        Returns:
            An instance of node which has been inserted into SymbolTree.

        Raises:
            RuntimeError: If 'position' is not in current SymbolTree.
            RuntimeError: If corresponding ast node is not an ast.Assign when 'insert_to_ast' is True.
        """

        # if position in current SymbolTree
        if position is not None and position.symbol_tree is not self:
            raise RuntimeError("Position is not in current SymbolTree:", position)
        # unique targets, name while insert node into symbol_tree
        node_name = self._node_name_namer.get_name(node)
        node.set_name(node_name)
        self._handle_custom_obj_in_normalized_args(node)
        # _unique_targets must called after _update_args_for_unique and _update_kwargs_for_unique
        self._unique_targets(node)
        self._insert_node(position, node)
        # update init-function-ast and construct-function-ast
        if insert_to_ast:
            node.set_func(ScopedValue.create_naming_value(node_name, "self"))
            node_ast = node.get_ast()
            if not isinstance(node_ast, ast.Assign):
                raise RuntimeError("Only support insert cell op now")
            AstModifier.insert_assign_to_function(self._init_func_ast,
                                                  targets=[ScopedValue(ValueType.NamingValue, "self", node_name)],
                                                  expr=ScopedValue(ValueType.NamingValue, "global_vars", "get"),
                                                  args=[ScopedValue(ValueType.StringValue, "", node_name)])
            AstModifier.insert_assign_ast_to_function(self._root_ast, node_ast,
                                                      None if position is None else position.node.get_ast(),
                                                      position.before_node)
            self._global_vars[node_name] = node.get_instance()
        return node

    def append_node(self, node: Node, append_to_ast: bool = True) -> Node:
        """
        Append a node to SymbolTree.

        Args:
            node (Node): An instance of node to be appended.
            append_to_ast (bool): A bool indicates whether to update corresponding ast node at same time, default is
                True.

        Returns:
            An instance of node which has been appended to SymbolTree.
        """
        return self.insert_node(Position.create(self, self._tail, False), node, append_to_ast)

    def append_origin_field(self, node: Node) -> Node:
        """
        Append an original field node to SymbolTree. An original field node represents a node created from existing
        statement in forward method, from existing ast node in ast of forward method, so ast node do not need to update
        while these nodes appending to SymbolTree.
        This method is called while building SymbolTree usually.

        Args:
            node (Node): An instance of node to be appended.

        Returns:
            An instance of node which has been appended to SymbolTree.
        """
        self._update_args_kwargs_for_unique(node)
        if node.get_node_type() == NodeType.Output:
            self._return = node
        elif node.get_node_type() == NodeType.Input:
            self._inputs.append(node)
        return self.append_node(node, False)

    def append_input_node(self, param_name: str, default: Optional[ScopedValue] = None):
        """
        Append an input node to SymbolTree corresponding to parameter of forward method of network class.
        This method is called while building SymbolTree usually.

        Args:
            param_name (str): A str represents name of parameter of forward method of network class.
            default (ScopedValue, optional): A ScopedValue represents default value of parameter. Default is None which
                means parameter has no default value.

        Returns:
            An instance of input node which has been appended to SymbolTree.
        """
        if param_name == "self":
            return
        for input_node in self._inputs:
            targets = input_node.get_targets()
            if len(targets) != 1:
                raise RuntimeError("targets should have 1 elements")
            target: ScopedValue = targets[0]
            if target.type != ValueType.NamingValue:
                raise RuntimeError("target.type should equal to ValueType.NamingValue")
            if target.scope != "":
                raise RuntimeError("target.scope should be empty")
            exist_param = target.value
            if exist_param == param_name:
                raise RuntimeError("input duplicated:", param_name)
        input_node = Node.create_input_node(None, param_name, default, name=f"input_{param_name}")
        self.append_origin_field(input_node)

    def try_append_python_node(self, ast_scope: ast.AST, ast_node: ast.AST) -> Optional[Node]:
        """
        Try appending a python node to SymbolTree if 'ast_node' is not None and 'ast_node' is not Empty if 'ast_node' is
        a list or a dict.
        This method is called while building SymbolTree usually.

        Args:
            ast_scope (ast.AST): A ast node represents ast node of scope of node.
            ast_node (ast.AST): A ast node represents ast node.

        Returns:
            An instance of python node if a new node has been appended to SymbolTree else None.
        """
        if ast_node is None:
            return None
        if isinstance(ast_node, (list, dict)) and not ast_node:
            return None
        return self.append_python_node(ast_scope, ast_node)

    def append_python_node(self, ast_scope: ast.AST, ast_node: ast.AST) -> Node:
        """
        Append a python node to SymbolTree.
        This method is called while building SymbolTree usually.

        Args:
            ast_scope (ast.AST): A ast node represents ast node of scope of node.
            ast_node (ast.AST): A ast node represents ast node.

        Returns:
            An instance of python node which has been appended to SymbolTree.
        """
        logger.info("Ignoring unsupported node(%s) in %s.", type(ast_node).__name__, type(ast_scope).__name__)
        node_name = self._node_name_namer.get_name(type(ast_node).__name__)
        node = Node.create_python_node(ast_node, node_name)
        self._insert_node(Position.create(self, self._tail, True), node)
        return node

    def set_output(self, return_value: str, index: int) -> Node:
        """
        Update return value of return of forward method of network class.

        Args:
            return_value (str): A str represents new return value.
            index (int): A int indicates which return value to be updated.

        Returns:
            An instance of node represents return node after updated.
        """
        if self._return is None:
            raise RuntimeError("SymbolTree has no output")
        self.set_node_arg(self._return, index, return_value)
        return self._return

    def erase_node(self, node_or_name: Union[Node, str]) -> Node:
        """
        Erase a node from SymbolTree.
        Note:
            If node is depended on by other node, RuntimeError will raise.

            Topological relation is updated.

        Args:
            node_or_name (Union[Node, str]): An instance of node or a str represents name of node.

        Returns:
            An instance of node which has been erased from SymbolTree.

        Raises:
            RuntimeError: If 'node_or_name' is not in current SymbolTree.
            RuntimeError: If erase corresponding ast node failed.
        """

        node = self._get_real_node(node_or_name)
        if node is None:
            raise RuntimeError("Node is not belong to current SymbolTree: ", node_or_name)
        ret = AstModifier.erase_ast_from_function(self._root_ast, node.get_ast())
        if not ret:
            raise RuntimeError("node not in function ast tree.")
        for key, value in self._nodes.items():
            if id(value) == id(node):
                self._nodes.pop(key)
                value.isolate()
                break
        self._topo_mgr.on_erase_node(node)
        return node

    def _insert_tree(self, position: Position, root: Node, insert_to_ast: bool = True) -> Node:
        """
        Insert a node-tree into SymbolTree.
        Note:
            Inputs of intra sub-tree nodes need to be welly set.

            Inputs of inter sub-tree nodes will be updated by Rewrite automatically.

        Args:
            position (Position): A Position indicates an insert position point.
            root (Node): An instance of node as root of node-tree to be inserted in.
            insert_to_ast (bool): A bool indicates whether to update corresponding ast node at same time, default is
                True.

        Returns:
            An instance of node as root node of node-tree which has been inserted into SymbolTree.

        Raises:
            RuntimeError: If 'position' is not in current SymbolTree.
        """

        # if position not in current SymbolTree
        if position.symbol_tree is not self:
            raise RuntimeError("Position is not in current SymbolTree: ", position)

        queue: [Node] = [root]
        todos: [] = []
        inputs_list: [] = []
        while queue:
            cur_node = queue.pop(0)
            if cur_node in todos:
                continue
            todos.append(cur_node)
            node_inputs = cur_node.get_inputs()
            inputs_list.append(node_inputs)
            for node_input in node_inputs:
                if node_input is not None:
                    queue.append(node_input)
        todos.reverse()
        inputs_list.reverse()
        for index, todo in enumerate(todos):
            self.insert_node(position, todo, insert_to_ast)
            position = self.after(todo)
            # relink input of node
            original_inputs = inputs_list[index]
            for arg_idx, original_input in enumerate(original_inputs):
                if original_input is not None:
                    self.set_node_arg_by_node(todo, arg_idx, original_input)
        return root

    @staticmethod
    def _link_nodes_and_find_root(nodes: [Node]) -> Node:
        """
        Find inputs for all nodes created by Replacement according to their targets and arguments.

        Find root node of all nodes created by Replacement. One and Only one root should be found.

        Args:
            nodes (list[Node]): A list of instance of Node created by Replacement.

        Returns:
            An instance of Node represents root of input nodes.
        """
        consumers: [ScopedValue] = []
        target_dict: {ScopedValue: Node} = {}
        for node in nodes:
            consumers.extend(node.get_args())
            for _, arg in node.get_kwargs():
                consumers.append(arg)
            for target in node.get_targets():
                if target_dict.get(target) is not None:
                    raise RuntimeError("Target of node duplicated")
                target_dict[target] = node
        # find root node
        root = None
        for node in nodes:
            used = 0
            for target in node.get_targets():
                if target in consumers:
                    used += 1
            if used == 0:
                if root is not None:
                    raise RuntimeError("Replacement should only has one root")
                root = node
        if root is None:
            raise RuntimeError("No root node found in replacement nodes")
        # link node's input
        for node in nodes:
            inputs = []
            for _, arg in node.get_normalized_args().items():
                node_input: Node = target_dict.get(arg)
                if node_input is None:
                    inputs.append(None)
                else:
                    inputs.append(node_input)
            node.set_inputs(inputs)
        return root

    def replace(self, old_node: Node, new_nodes: [Node]) -> Node:
        """
        Replace an old_node with a node_tree. 'new_node' is the root node of the node_tree.
        Note:
            Rewrite will iterate all nodes linked to this root node and insert these nodes into symbol_tree.

            Inputs of intra sub-tree nodes need to be welly set.

            Inputs of inter sub-tree nodes will be updated by Rewrite automatically.

        Args:
            old_node (Node): Node to be replaced.
            new_nodes (list[Node]): Node tree to replace in.

        Returns:
            An instance of Node represents root of node_tree been replaced in.

        Raises:
            RuntimeError: If 'old_node' is isolated.
            RuntimeError: If 'old_node' is not belong to current SymbolTree.
        """

        real_old_node = self._get_real_node(old_node)
        if real_old_node is None:
            raise RuntimeError("Old node is not belong to current SymbolTree:", old_node)
        # get position
        next_node: Node = old_node.get_next()
        prev_node: Node = old_node.get_prev()
        if prev_node is None and next_node is None:
            raise RuntimeError("Try replacing a isolated node: ", old_node)
        if next_node is None:
            position = self.after(prev_node)
        else:
            position = self.before(next_node)
        # insert node first, because targets of new_node is determined after insert
        new_tree_root = SymbolTree._link_nodes_and_find_root(new_nodes)
        new_node = self._insert_tree(position, new_tree_root)
        # use targets of insert tree to redirect edge
        users = self.get_node_users(old_node)
        if len(new_node.get_targets()) != 1:
            raise RuntimeError("targets of new_node should have 1 elements")
        for user in users:
            self.set_node_arg_by_node(user[0], user[1], new_node)
        # erase old_node after edge is redirected because node can be erased only when node is isolated topologically
        self.erase_node(old_node)
        return new_node

    def set_node_arg(self, node: Union[Node, str], index: int, arg: Union[ScopedValue, str]):
        """
        Set argument of 'node'.

        Args:
            node (Union[Node, str]): Node to be modified. Can be a node or name of node.
            index (int): Indicate which input being modified.
            arg (Union[ScopedValue, str]): New argument to been set.

        Raises:
            RuntimeError: If 'node' is not belong to current SymbolTree.
        """

        real_node = self._get_real_node(node)
        if real_node is None:
            raise RuntimeError("Node is not belong to current SymbolTree: ", node)

        new_arg, old_arg = node.set_arg(arg, index)
        self._topo_mgr.on_update_arg(node, index, old_arg, new_arg)

    def set_node_arg_by_node(self, dst_node: Union[Node, str], arg_idx: int, src_node: Union[Node, str],
                             out_idx: Optional[int] = None):
        """
        Set argument of 'dst_node' by another Node.

        Args:
            dst_node (Node): Node to be modified. Can be a node or name of node.
            arg_idx (int): Indicate which input being modified.
            src_node (Node): Node as new input. Can be a node or name of node.
            out_idx ([int, optional]): Indicate which output of 'src_node' as new input of 'dst_node'. Default is None
                which means use first output of 'node_to_link' as new input.

        Raises:
            RuntimeError: If 'dst_node' is not belong to current SymbolTree.
            RuntimeError: If 'src_node' is not belong to current SymbolTree.
            RuntimeError: If 'out_idx' is out of range.
            RuntimeError: If 'src_node' has multi-outputs while 'out_idx' is None or 'out_idx' is not offered.
        """

        real_dst_node = self._get_real_node(dst_node)
        if real_dst_node is None:
            raise RuntimeError("dst_node is not belong to current SymbolTree: ", dst_node)
        real_src_node = self._get_real_node(src_node)
        if real_src_node is None:
            raise RuntimeError("src_node is not belong to current SymbolTree: ", src_node)

        targets = real_src_node.get_targets()
        if out_idx is None:
            if len(targets) != 1:
                raise RuntimeError("node should has one output when out_idx is not provided")
            out_idx = 0
        if out_idx >= len(targets):
            raise RuntimeError("out_idx out of range: ", out_idx)
        new_arg = targets[out_idx]
        self.set_node_arg(real_dst_node, arg_idx, new_arg)

    def dump(self):
        """Dump graph."""
        dump_st = SymbolTreeDumper(self)
        dump_st.dump()

    def get_code(self) -> str:
        """
        Get source code of modified network.

        Returns:
            A str represents source code of modified network.
        """
        ast.fix_missing_locations(self._module_ast)
        # Find all ast.ClassDef which can be export to code
        # Replace duplicated ast.ClassDef reference in main-ClassDef
        seen_class: {type, str} = {}
        allow_class_name = []
        replacer = AstReplacer(self._class_ast)
        for node in self.nodes():
            if not isinstance(node, TreeNode):
                continue
            sub_stree: SymbolTree = node.symbol_tree
            # all modified ast.ClassDef should export to code
            if sub_stree._modified:
                allow_class_name.append(sub_stree._class_ast.name)
                continue
            # all un-modified ast.ClassDef only keep one instance
            seen_cls_name = seen_class.get(type(sub_stree.get_origin_network()))
            if seen_cls_name is not None:
                replacer.replace_all(sub_stree._class_ast.name, seen_cls_name)
            else:
                seen_class[type(sub_stree.get_origin_network())] = sub_stree._class_ast.name
                allow_class_name.append(sub_stree._class_ast.name)
        allow_class_name.append(self._class_ast.name)
        # Add all non-ClassDef body to gencode_module
        # Add all ClassDef in allow_class_name to gencode_module
        # Use gencode_module to generate code
        bodies = []
        for body in self._module_ast.body:
            if not isinstance(body, ast.ClassDef):
                bodies.append(body)
                continue
            if body.name in allow_class_name:
                bodies.append(body)
        gencode_module = ast.Module(body=bodies)
        code = astunparse.unparse(gencode_module)
        # Restore main-ClassDef
        replacer.undo_all()
        return code

    def get_network(self):
        """
        Get modified network.

        Returns:
            A network object.
        """
        cls = self._get_cls_through_file()
        return cls(self._global_vars)

    def _unique_targets(self, node: Node):
        """
        Unique targets of node by _target_namer.

        Args:
            node (Node): A Node whose targets to be uniqued.
        """
        new_targets: [ScopedValue] = []
        if node.get_targets() is None:
            return
        for target in node.get_targets():
            if not isinstance(target, ScopedValue):
                raise TypeError("target should be ScopedValue, got: ", type(target))
            unique_target = self._target_namer.get_name(target.value)
            new_targets.append(ScopedValue.create_naming_value(unique_target, target.scope))
        node.set_targets(new_targets)

    def _update_args_kwargs_for_unique(self, node: Node):
        """
        Update arguments and keyword arguments of node because unique-ing of targets of other nodes.

        Args:
            node (Node): A Node whose arguments and keyword arguments to be updated.
        """
        result: {str: ScopedValue} = {}
        if node.get_normalized_args() is None:
            return
        for key, arg in node.get_normalized_args().items():
            if not isinstance(arg, ScopedValue):
                raise TypeError("arg should be ScopedValue, got: ", type(arg))
            if arg.type == ValueType.NamingValue:
                # unique name
                new_arg = ScopedValue(arg.type, arg.scope, self._target_namer.get_real_arg(arg.value))
                result[key] = new_arg
            else:
                result[key] = arg
        node.set_normalized_args(result)

    def _add_node2nodes(self, node: Node):
        """
        Add `node` to `_nodes` dict.

        Args:
            node (Node): A Node to be added into `_nodes`.

        Raises:
            RuntimeError: If name of the node is duplicated.
        """
        node_name = node.get_name()
        if self._nodes.get(node_name) is not None:
            raise RuntimeError("generated duplicated node name", node_name, self._nodes.get(node_name),
                               node)
        self._nodes[node_name] = node

    def _insert_node(self, position: Optional[Position], node: Node):
        """
        Insert a node into SymbolTree.
        1. Add `node` to `_nodes`.
        2. Insert `node` to node list(source code order).
        3. Update topological relation and update inputs of `node`.

        Args:
            position ([Position, optional]): Indicates node insert position. Position is None when inserting first node
                of SymbolTree.
            node (Node): A Node to be inserted into SymbolTree.

        Raises:
            RuntimeError: Position is None when _nodes of SymbolTree is not Empty. It means position can not be None
                unless inserting first node.
        """
        if position is None:
            if self._nodes:
                raise RuntimeError("self._nodes should be empty")
            self._head = node
        else:
            if position.before_node:
                position.node.insert_before(node)
            else:
                position.node.insert_after(node)
        self._tail = node
        self._add_node2nodes(node)
        self._topo_mgr.on_insert_node(node)
        node.set_belong_symbol_tree(self)

    def _handle_custom_obj_in_normalized_args(self, node: Node):
        """
        Convert CustomObjValue type argument to NamingValue type argument by storing custom object in global_vars dict.

        Args:
            node (Node): A Node whose arguments and keyword arguments to be handled.
        """
        result: {str, ScopedValue} = {}
        for arg, value in node.get_normalized_args().items():
            if not isinstance(value, ScopedValue):
                raise TypeError("value should be ScopedValue, got: ", type(value))
            if value.type == ValueType.CustomObjValue:
                field = self._node_name_namer.get_name(f"var_{type(value.value).__name__}")
                self._global_vars[field] = value.value
                init_targets = [ScopedValue.create_naming_value(field, "self")]
                AstModifier.append_global_vars_expr_to_init(self._init_func_ast, init_targets, field)
                result[arg] = init_targets[0]
            else:
                result[arg] = value
        node.set_normalized_args(result)

    def _get_cls_through_file(self):
        """
        Load rewritten network class of current SymbolTree.
        1. Get source code of current SymbolTree.
        2. Saving source code to a tempfile.
        3. Import rewritten network class using "__import__" function.

        Returns:
            A class handle.
        """
        source = self.get_code()
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.py')
        tmp_file.write(source.encode('utf8'))
        tmp_file.flush()
        tmp_file_name = tmp_file.name
        tmp_module_path, tmp_module_file = os.path.split(tmp_file_name)
        tmp_module_name = tmp_module_file[:-3]
        sys.path.append(tmp_module_path)
        tmp_module = __import__(tmp_module_name)
        network_cls = getattr(tmp_module, self._opt_cls_name)
        if network_cls is None:
            raise RuntimeError("Can not find network class:", self._opt_cls_name)
        return network_cls
