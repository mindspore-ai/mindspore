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
import stat
from typing import Optional, Union, Tuple, Any, Dict, List
import os
import sys
import ast
import importlib.util
import time

from mindspore.nn import Cell
from mindspore import log as logger
from .node.node import Node, TreeNode
from .api.node_type import NodeType
from .ast_helpers import AstModifier, AstReplacer, StrChecker, AstFinder, AstClassFinder, AstFunctionFinder
from .api.scoped_value import ScopedValue, ValueType
from .symbol_tree_dumper import SymbolTreeDumper
from .node.node_topological_manager import TopoManager
from .namer import TargetNamer, NodeNamer, ClassNamer
from .common.observer import Observer
from .common.observable import Observable
from .common.event import Event
from .node.node_manager import NodeManager

if sys.version_info >= (3, 9):
    import ast as astunparse # pylint: disable=reimported, ungrouped-imports
else:
    import astunparse

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


class FieldFinder(AstFinder):
    """
    Check whether field exist in specific scope.

    Args:
        scope (ast.AST): An instance of ast node as search scope.
    """

    def __init__(self, scope: ast.AST):
        super().__init__(scope)
        self._result = False
        self._field_name = ""

    def visit_Attribute(self, node: ast.Attribute) -> Any:
        """Visit a node of type ast.Attribute."""
        value = node.value
        if not isinstance(value, ast.Name):
            return super(FieldFinder, self).generic_visit(node)
        if value.id != "self":
            return super(FieldFinder, self).generic_visit(node)
        if node.attr == self._field_name:
            self._result = True
        return super(FieldFinder, self).generic_visit(node)

    def check(self, field) -> bool:
        """
        Check whether `field` exist in scope.

        Args:
            field (str): A string indicates target field name.

        Returns:
            A bool indicate whether `field` exist in scope.
        """
        self._result = False
        self._field_name = field
        self.visit(self._scope)
        return self._result


class IfFixer(ast.NodeTransformer):
    """
    Fix ast.If if body is empty while orelse is not empty.
    """

    def visit_If(self, node: ast.If) -> Any:
        """Visit a node of type ast.If."""
        if not node.body and node.orelse:
            node.body.append(ast.Pass())
        return super().generic_visit(node)

    def fix(self, node):
        """
        Fix ast.If node in `node` if whose body is empty while whose orelse is not empty.

        Args:
            node (ast.AST): An ast node to be fixed.
        """
        self.generic_visit(node)


class SymbolTree(Observer, Observable, NodeManager):
    """
    A symbol-tree usually corresponding to forward method of a network.

    Rewrite recommend using SymbolTreeBuilder to instantiate an instance of SymbolTree rather than invoking constructor
    of SymbolTree directly.

    Args:
        origin_network (Cell): A handler to original network instance.
        module_ast (ast.Module): An instance of ast.AST represents ast node of original network.
    """

    def __init__(self, origin_network: Cell, module_ast: ast.Module):
        Observer.__init__(self)
        Observable.__init__(self)
        self._node_namer = NodeNamer()
        self._node_namer.add_name('obj')
        NodeManager.__init__(self, self._node_namer)
        NodeManager.reg_observer(self, observer=self)
        # init unique-namers
        self._target_namer = TargetNamer()
        # input arguments of function
        self._ori_cls_name = type(origin_network).__name__
        self._opt_cls_name = ClassNamer.instance().get_name(self._ori_cls_name)
        NodeManager.set_manager_name(self, self._opt_cls_name)
        self._origin_network = origin_network
        self._module_ast: ast.Module = module_ast
        self._import_asts: Optional[ast.Ast] = []
        self._class_ast: Optional[ast.ClassDef] = None
        self._root_ast: Optional[ast.FunctionDef] = None
        self._init_func_ast: Optional[ast.FunctionDef] = None
        self._deleted_field = {}
        self._deleted_node = []
        self._external_ast = []
        self._father_class_ast = []
        self._modified = False
        self._tmp_file_limits = 20
        self._tmp_files = []
        self._saved_file_name = "./network_define.py"
        # used to insert "sys.path.append(xxx)"
        self._net_file_paths = []
        self._tmp_import_strs = []
        self._tmp_unmodified_strees: {type, str} = {}
        self._tmp_replacers = []
        # Record imported modules and names of each files
        # The meanings of `module` and `name` are like code: from `module` import `nameA`, `nameB`
        # Format: {file_path: {module: [name, ...], ...}, ...}
        self._imported_modules: Dict[str, Dict[str, List[str]]] = {}

    def __del__(self):
        for tmp_file in self._tmp_files:
            tmp_file.close()

    @staticmethod
    def _remove_unused_import(module_ast):
        """remove unused import in self._module_ast"""
        str_checker = StrChecker(module_ast)
        for i in range(len(module_ast.body) - 1, -1, -1):
            body = module_ast.body[i]
            if not isinstance(body, (ast.Import, ast.ImportFrom)):
                continue
            if isinstance(body, ast.Import):
                continue
            if isinstance(body, ast.ImportFrom) and body.module == "cell":
                module_ast.body.remove(body)
                continue
            for alias in body.names:
                name = alias.asname if alias.asname else alias.name
                if not str_checker.check(name):
                    if len(body.names) == 1:
                        module_ast.body.remove(body)
                        i += 1
                    else:
                        body.names.remove(alias)

    @staticmethod
    def _remove_duplicated_import(module_ast):
        """Remove duplicated import of 'net'."""
        imports = set()
        futures = set()
        classes = set()

        class TransImportNode(ast.NodeTransformer):
            """Find all import nodes from input ast node."""

            def visit_ClassDef(self, node: ast.ClassDef) -> Any:
                class_str = astunparse.unparse(node)
                if class_str not in classes:
                    classes.add(node.name)
                    return node
                return

            def visit_Try(self, node: ast.Try) -> Any:
                if isinstance(node.body[0], (ast.Import, ast.ImportFrom)):
                    import_str = astunparse.unparse(node)
                    if import_str not in imports:
                        imports.add(import_str)
                        return node
                return

            def visit_Import(self, node: ast.Import) -> Any:
                import_str = astunparse.unparse(node)
                if import_str not in imports:
                    imports.add(import_str)
                    return node
                return

            def visit_ImportFrom(self, node: ast.ImportFrom) -> Any:
                """
                Once the father class 'A' is defined in the current module, all the next imported class 'A' should
                be removed. e.g.
                    def class A():
                        ...
                    from xxx import A, B
                    =>
                    def class A():
                        ...
                    from xxx import B
                """
                import_str = astunparse.unparse(node)

                if import_str not in imports:
                    imports.add(import_str)
                    # remove "__future__" module
                    if node.module == '__future__':
                        futures.add(node.module)
                        return
                    # remove modules which have been defined in the code file
                    # it occurs when class A is a father class and other sub-classes import A
                    for alias in node.names[:]:
                        if alias.name in classes:
                            node.names.remove(alias)
                    # if the alias(es) in node.names are all removed, this import statement should be removed
                    if not node.names:
                        return
                    return node
                return

        get_node_handler = TransImportNode()
        get_node_handler.generic_visit(module_ast)

    def finish_build(self):
        """Add Event.TopologicalChangeEvent event when build is finished."""
        self.add_event(Event.TopologicalChangeEvent)

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
        NodeManager.set_ast_functiondef(self, ast_node)

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

    def get_origin_network(self):
        """
        Getter of `_origin_network`.

        Returns:
            An instance of Cell which represents original network.
        """
        return self._origin_network

    def get_nodes_dict(self):
        """Get dict of nodes"""
        return self._nodes

    def get_node_namer(self):
        """Get _node_namer"""
        return self._node_namer

    def is_modified(self):
        """
        Check whether symbol tree is modified.

        Symbol tree is considered as modified if operations like insert/replace/erase/set_arg is called after
        the symbol tree is created.
        """
        return self._modified

    def set_modified_true(self):
        """
        Set self._modified true.

        Self._modified is set true when 'if' exists in the original network.
        In this situation, different original network instance tends to be different.
        Hence, the class name should be updated.
        """
        self._modified = True

    def get_import_asts(self):
        """Get _import_asts"""
        return self._import_asts

    def get_external_ast(self):
        """Get _external_ast"""
        return self._external_ast

    def get_father_class_ast(self):
        """Get _father_class_ast"""
        return self._father_class_ast

    def get_imported_modules(self, file_path: str):
        """Get all modules and module_paths in file of `file_path` ."""
        return self._imported_modules.get(file_path, {})

    def save_imported_modules(self, file_path: str, module: str, names: List[str]):
        """Save module and names into _imported_modules."""
        imported_modules = self.get_imported_modules(file_path)
        if imported_modules.get(module):
            imported_modules[module].extend(names)
        else:
            imported_modules[module] = names
        self._imported_modules[file_path] = imported_modules

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
        if real_node.get_node_type() == NodeType.Output:
            return []
        return TopoManager.get_node_users(real_node)

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

    def insert_node(self, new_node: Node, base_node: Node, before_node: bool, node_manager: NodeManager = None,
                    insert_to_ast: bool = True):
        """
        Insert a node before or after base_node.

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
            new_node (Node): Node to be inserted.
            base_node (Node): New node will be inserted before or after base_node.
            before_node (bool): Indicate whether new node is inserted before base_node.
            node_manager (NodeManager): NodeManager those asts belong to. Default: None, means those asts belong to
                NodeManager of symboltree's construct function.
            insert_to_ast (bool): Indicate whether ast nodes need to be updated.

        Returns:
            An instance of node which has been inserted into SymbolTree.

        Raises:
            ValueError: Node in the SymbolTree is inserted into SymbolTree again.
            RuntimeError: If corresponding ast node is not an ast.Assign when 'insert_to_ast' is True.
        """
        if new_node.get_belong_symbol_tree():
            raise ValueError(f"Node in the SymbolTree cannot be inserted into SymbolTree again: {new_node.get_name()}")

        # Check if base_node in current SymbolTree
        if base_node is not None:
            stree = base_node.get_belong_symbol_tree()
            if stree is not None and stree is not self:
                raise RuntimeError(f"Position is not in current SymbolTree, node:{stree.get_ori_cls_name()}, "
                                   f"current: {self.get_ori_cls_name()}.")

        # Check if node is inserted between Input node
        if base_node is not None and base_node.get_node_type() == NodeType.Input:
            valid = True
            if before_node:
                valid = False
            if base_node.get_next() is not None and base_node.get_next().get_node_type() == NodeType.Input:
                valid = False
            if not valid:
                raise RuntimeError("Can not insert a node before or between parameters:", base_node.get_name())

        # save target name, which is used to provide unique target
        if new_node.get_targets():
            for target in new_node.get_targets():
                self._target_namer.add_name(str(target))

        self._handle_custom_obj_in_normalized_args(new_node)

        # Insert node into NodeManager
        if node_manager is None:
            if base_node is None:
                raise RuntimeError("node_manager and base_node cannot both be None when inserting a node.")
            node_manager = base_node.get_node_manager()

        # set node's _belong_symbol_tree
        new_node.set_belong_symbol_tree(self)

        if node_manager is self:
            NodeManager.insert_node(self, new_node, base_node, before_node)
            if insert_to_ast:
                # update init-function-ast and construct-function-ast
                self.insert_to_ast_while_insert_node(new_node, base_node, before_node, self)
        else:
            node_manager.insert_node(new_node, base_node, before_node, insert_to_ast)

        # register code changed event observer, which is used to update _modified flag.
        if new_node.get_node_type() == NodeType.Tree:
            new_node.symbol_tree.reg_observer(self)
        elif isinstance(new_node, NodeManager):
            new_node.reg_observer(self)

        return new_node

    def append_node(self, node: Node, node_manager: NodeManager = None, append_to_ast: bool = True) -> Node:
        """
        Append a node to SymbolTree.

        Args:
            node (Node): An instance of node to be appended.
            append_to_ast (bool): A bool indicates whether to update corresponding ast node at same time, default is
                True.
            node_manager (NodeManager): NodeManager those asts belong to. Default: None, means those asts belong to
                NodeManager of symboltree's construct function.

        Returns:
            An instance of node which has been appended to SymbolTree.
        """
        if node_manager is None:
            node_manager = self
        return self.insert_node(node, node_manager.get_tail(), False, node_manager, append_to_ast)

    def append_origin_field(self, node: Node, node_manager: NodeManager = None) -> Node:
        """
        Append an original field node to SymbolTree. An original field node represents a node created from existing
        statement in forward method, from existing ast node in ast of forward method, so ast node do not need to update
        while these nodes appending to SymbolTree.
        This method is called while building SymbolTree usually.

        Args:
            node (Node): An instance of node to be appended.
            node_manager (NodeManager): NodeManager those asts belong to. Default: None, means those asts belong to
                NodeManager of symboltree's construct function.

        Returns:
            An instance of node which has been appended to SymbolTree.
        """
        return self.append_node(node, node_manager, False)

    def append_input_node(self, ast_node, param_name: str, default: Optional[ScopedValue] = None,
                          node_manager: NodeManager = None):
        """
        Append an input node to SymbolTree corresponding to parameter of forward method of network class.
        This method is called while building SymbolTree usually.

        Args:
            ast_node (ast.AST): A ast Node corresponding to current parameter.
            param_name (str): A str represents name of parameter of forward method of network class.
            default (ScopedValue, optional): A ScopedValue represents default value of parameter. Default is None which
                means parameter has no default value.
            node_manager (NodeManager): NodeManager those asts belong to. Default: None, means those asts belong to
                NodeManager of symboltree's construct function.

        Returns:
            An instance of input node which has been appended to SymbolTree.
        """
        if param_name == "self":
            return
        # check param_name duplicated
        if node_manager is None:
            node_manager = self
        for input_node in node_manager._inputs:
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
        input_node = Node.create_input_node(ast_node, param_name, default, name=f"input_{param_name}")
        self.append_origin_field(input_node, node_manager)

    def try_append_python_node(self, ast_scope: ast.AST, ast_node: ast.AST,
                               node_manager: NodeManager = None) -> Optional[Node]:
        """
        Try appending a python node to SymbolTree if 'ast_node' is not None and 'ast_node' is not Empty if 'ast_node' is
        a list or a dict.
        This method is called while building SymbolTree usually.

        Args:
            ast_scope (ast.AST): A ast node represents ast node of scope of node.
            ast_node (ast.AST): A ast node represents ast node.
            node_manager (NodeManager): NodeManager those asts belong to. Default: None, means those asts belong to
                NodeManager of symboltree's construct function.

        Returns:
            An instance of python node if a new node has been appended to SymbolTree else None.
        """
        if ast_node is None:
            return None
        if isinstance(ast_node, (list, dict)) and not ast_node:
            return None
        return self.append_python_node(ast_scope, ast_node, node_manager)

    def append_python_node(self, ast_scope: ast.AST, ast_node: ast.AST, node_manager: NodeManager = None) -> Node:
        """
        Append a python node to SymbolTree.
        This method is called while building SymbolTree usually.

        Args:
            ast_scope (ast.AST): A ast node represents ast node of scope of node.
            ast_node (ast.AST): A ast node represents ast node.
            node_manager (NodeManager): NodeManager those asts belong to. Default: None, means those asts belong to
                NodeManager of symboltree's construct function.

        Returns:
            An instance of python node which has been appended to SymbolTree.
        """
        logger.info("Ignoring unsupported node (%s) (%s).", type(ast_node).__name__, type(ast_scope).__name__)
        node_name = type(ast_node).__name__
        node = Node.create_python_node(ast_node, node_name)
        if node_manager is None or node_manager is self:
            NodeManager.append_python_node(self, node)
        else:
            node_manager.append_python_node(node)
        return node

    def set_output(self, return_value: str, arg_index: int, return_idx: int = 0,
                   node_manager: NodeManager = None) -> Node:
        """
        Update return value of return of forward method of network class.

        Args:
            return_value (str): A str represents new return value.
            arg_index (int): A int indicates which value in return to be updated.
            return_idx (int): A int indicates which return node to be updated. Default: 0.
            node_manager (NodeManager): NodeManager those asts belong to. Default: None, means
                symboltree's construct function.

        Returns:
            An instance of node represents return node after updated.
        """
        node_returns = NodeManager.get_returns(self) if node_manager is None else node_manager.get_returns()
        if not node_returns:
            raise RuntimeError("Current node_manager has no output")
        if return_idx >= len(node_returns):
            raise RuntimeError(f"return_idx {return_idx} should be less than return num {len(node_returns)}.")
        node_return = node_returns[return_idx]
        self.set_node_arg(node_return, arg_index, return_value)
        return node_return

    def erase_node(self, node_or_name: Union[Node, str]) -> Node:
        """
        Erase a node from SymbolTree.

        Topological relation will be updated.

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
        # erase node in NodeManager
        node_manager = node.get_node_manager()

        logger.debug(f"[earse]stree: {self.get_opt_cls_name()}, "
                     f"node_manager: {node_manager.get_manager_name()}, "
                     f"code: {astunparse.unparse(node.get_ast()).strip()}, "
                     f"node_name:{node.get_name()}")

        if node_manager is self:
            NodeManager.erase_node(self, node)
            ret = AstModifier.erase_ast_from_function(self._root_ast, node.get_ast())
            if not ret:
                raise RuntimeError(f"erase node failed, node {node.get_name()} not in function ast tree.")
        else:
            node_manager.erase_node(node)
        self._deleted_node.append(node.get_name())
        return node

    def replace(self, old_node: Node, new_nodes: [Node]) -> Node:
        """
        Replace an old_node with a node list.

        Args:
            old_node (Node): Node to be replaced.
            new_nodes (list[Node]): Node list to replace in.

        Returns:
            Last node in new_nodes list.

        Raises:
            RuntimeError: If 'old_node' is isolated.
            RuntimeError: If 'old_node' is not belong to current SymbolTree.
        """
        real_old_node = self._get_real_node(old_node)
        if real_old_node is None:
            raise RuntimeError("Old node is not belong to current SymbolTree:", old_node)
        # insert new_nodes into node_manager
        node_manager = real_old_node.get_node_manager()
        # insert new_nodes into NodeManager
        base_node = old_node
        for node in new_nodes:
            self.insert_node(node, base_node, False, node_manager, True)
            base_node = node
        self.erase_node(old_node)
        return new_nodes[-1]

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
        real_dst_node.set_arg(new_arg, arg_idx)
        self._topo_mgr.on_update_arg_by_node(real_dst_node, arg_idx, real_src_node, out_idx)

    def unique_name(self, name: str):
        """Get a unique name in the symboltree"""
        return self._target_namer.get_name(name)

    def unique_func_name(self, name: str):
        """Get a unique function name in the symboltree"""
        if not hasattr(self._origin_network, name):
            return name
        suffix = 1
        while hasattr(self._origin_network, f"{name}_{suffix}"):
            suffix += 1
        return f"{name}_{suffix}"

    def set_node_target(self, node: Union[Node, str], index: int, target: Union[ScopedValue, str]):
        """
        Set target of `node` .

        Args:
            node (Union[Node, str]): Node to be modified. Can be a node or name of node.
            index (int): Indicate which target being modified.
            arg (Union[ScopedValue, str]): New target to been set.

        Raises:
            ValueError: If `node` is not belong to current SymbolTree.
            ValueError: If index of `node` 's target is greater than number of targets.
        """

        real_node = self._get_real_node(node)
        if real_node is None:
            raise ValueError("Node is not belong to current SymbolTree: ", node)
        if isinstance(target, str):
            target = ScopedValue.create_naming_value(target)
        targets = node.get_targets()
        if index >= len(targets):
            raise ValueError(f"Index of node's target should be less than {len(targets)}, but got {index}")
        old_target = targets[index]
        targets[index] = target
        node.set_targets(targets)
        self._topo_mgr.on_update_target(node, index, old_target, target)

    def all_nodes(self):
        """
        Get all nodes including nodes in CallFunction node, CellContainer node and sub symbol tree.

        Returns:
            A list of nodes.
        """
        nodes = []
        node_managers = [self]
        while node_managers:
            node_manager = node_managers.pop()
            nodes.extend(node_manager.nodes())
            for node in node_manager.nodes():
                if isinstance(node, NodeManager):
                    node_managers.append(node)
        for tree_node in self.get_tree_nodes():
            stree = tree_node.symbol_tree
            nodes.extend(stree.all_nodes())
        return nodes

    def get_node_from_name(self, node_name: str):
        """
        Get node from all NodeManagers in current symbol tree by `node_name`.

        Args:
            node_name (str): A str represents name of node as key of query.

        Returns:
            An instance of Node if found else None.
        """
        node_managers = [self]
        while node_managers:
            node_manager = node_managers.pop()
            node = node_manager.get_node(node_name)
            if node:
                return node
            for node in node_manager.nodes():
                if isinstance(node, NodeManager):
                    node_managers.append(node)
        return None

    def print_node_tabulate(self, all_nodes: bool = False):
        """
        Print nodes information and nodes' topological relations.

        Args:
            all_nodes (bool): Print nodes out of construct functions, such as nodes in CallFunction
                nodes, CellContainer nodes and sub symbol trees.
        """
        try:
            from tabulate import tabulate # pylint: disable=unused-import,reportMissingModuleSource
        except ImportError:
            logger.warning("print_node_tabulate relies on the library `tabulate`, "
                           "which could not be found on this machine. Run `pip "
                           "install tabulate` to install the library.")
            return ""
        print(NodeManager.dump(self, self.get_manager_name()))
        if all_nodes:
            node_managers = [self]
            while node_managers:
                node_manager = node_managers.pop()
                for node in node_manager.nodes():
                    if isinstance(node, NodeManager):
                        print(node.dump(node.get_manager_name()))
                        node_managers.append(node)
            for tree_node in self.get_tree_nodes():
                stree = tree_node.symbol_tree
                stree.print_node_tabulate(all_nodes)

    def dump(self):
        """Dump graph."""
        dump_st = SymbolTreeDumper(self)
        dump_st.dump()

    def check_body_exist(self, body, code_bodies):
        """Check whether body already exist in code_bodies"""
        # Check import ast node exist by saving import code string to self._tmp_import_strs
        if isinstance(body, (ast.Import, ast.ImportFrom, ast.Expr)):
            import_str = astunparse.unparse(body)
            if import_str in self._tmp_import_strs:
                return True
            self._tmp_import_strs.append(import_str)
            return False

        # Check ClassDef ast node exist by using AstClassFinder
        if isinstance(body, ast.ClassDef):
            if sys.version_info >= (3, 9):
                class_finder = AstClassFinder(ast.Module(body=code_bodies, type_ignores=[]))
            else:
                class_finder = AstClassFinder(ast.Module(body=code_bodies))
            results = class_finder.find_all(body.name)
            return bool(results)

        # Check FunctionDef ast node exist by using AstFunctionFinder
        if isinstance(body, ast.FunctionDef):
            if sys.version_info >= (3, 9):
                function_finder = AstFunctionFinder(ast.Module(body=code_bodies, type_ignores=[]))
            else:
                function_finder = AstFunctionFinder(ast.Module(body=code_bodies))
            results = function_finder.find_all(body.name)
            return bool(results)

        return False

    def update_class_name_of_unmodified_stree(self, stree, code_bodies) -> bool:
        """
        For the unmodified symbol tree, only one definition code remains in the generated code.
        Everywhere else calling this symbol tree will use the class in this definition code.
        """
        # all modified ast.ClassDef will be exported to code
        if stree.is_modified():
            return False
        # all un-modified ast.ClassDef only keep one instance
        first_cls_name = self._tmp_unmodified_strees.get(type(stree.get_origin_network()))
        if first_cls_name is None:
            class_ast = stree.get_class_ast()
            if class_ast:
                self._tmp_unmodified_strees[type(stree.get_origin_network())] = class_ast.name
            return False
        # Un-modified ast.ClassDef already exist in code_bodies,
        # replace class name to class name of first un-modified ast.ClassDef.
        if sys.version_info >= (3, 9):
            replacer = AstReplacer(ast.Module(body=code_bodies, type_ignores=[]))
        else:
            replacer = AstReplacer(ast.Module(body=code_bodies))
        replacer.replace_all(stree.get_class_ast().name, first_cls_name)
        self._tmp_replacers.append(replacer)
        return True

    def convert_stree_to_code_bodies(self, stree, code_bodies, insert_pos=0):
        """
        Convert nodes in stree to code_bodies

        1. Add import asts into code_bodies
        2. Add class, function and other type of asts into code_bodies
        3. Add father class asts into code_bodies
        4. Add external function asts into code_bodies
        5. Add subtrees to code_bodies
        5.1 Add subtrees in construct to code_bodies
        5.2 Add subtrees in CellContainers to code_bodies

        """
        # Add import asts into code_bodies
        for body in stree.get_import_asts():
            if not self.check_body_exist(body, code_bodies):
                code_bodies.insert(insert_pos, body)
                insert_pos += 1

        # Add class, function and other type of asts into code_bodies
        if stree.get_module_ast():
            for body in stree.get_module_ast().body:
                if self.check_body_exist(body, code_bodies):
                    continue
                if isinstance(body, (ast.ClassDef, ast.FunctionDef)):
                    code_bodies.insert(insert_pos, body)
                else:
                    code_bodies.append(body)

        # Add father class asts into code_bodies
        for body in reversed(stree.get_father_class_ast()):
            if self.check_body_exist(body, code_bodies):
                # remove exist ast in old position, then insert ast to upper position
                if sys.version_info >= (3, 9):
                    exist_ast = AstClassFinder(ast.Module(body=code_bodies, type_ignores=[])).find_all(body.name)[0]
                else:
                    exist_ast = AstClassFinder(ast.Module(body=code_bodies)).find_all(body.name)[0]
                code_bodies.remove(exist_ast)
            code_bodies.insert(insert_pos, body)

        # Add external asts into code_bodies
        for body in stree.get_external_ast():
            if not self.check_body_exist(body, code_bodies):
                code_bodies.insert(insert_pos, body)
                insert_pos += 1

        # Add subtrees to code_bodies
        for node in stree.get_tree_nodes():
            sub_stree = node.symbol_tree
            # Ignore TreeNode create by function in the class
            if isinstance(sub_stree.get_module_ast(), ast.FunctionDef):
                continue
            # For the unmodified class, update class name to name of first class
            if self.update_class_name_of_unmodified_stree(sub_stree, code_bodies):
                continue
            self.convert_stree_to_code_bodies(node.symbol_tree, code_bodies, insert_pos)

    def get_code(self) -> str:
        """
        Get source code of modified network.

        Returns:
            A str represents source code of modified network.
        """
        self._tmp_import_strs.clear()
        self._tmp_unmodified_strees.clear()
        self._tmp_replacers.clear()
        code_bodies = []
        self.convert_stree_to_code_bodies(self, code_bodies)
        if sys.version_info >= (3, 9):
            gencode_module = ast.Module(body=code_bodies, type_ignores=[])
        else:
            gencode_module = ast.Module(body=code_bodies)
        SymbolTree._remove_unused_import(gencode_module)
        SymbolTree._remove_duplicated_import(gencode_module)
        ast.fix_missing_locations(self._module_ast)
        IfFixer().fix(gencode_module)
        code = astunparse.unparse(gencode_module)
        # Revert the class name to its original state
        for replacer in self._tmp_replacers:
            replacer.undo_all()
        return code

    def get_network(self):
        """
        Get modified network.

        Returns:
            A network object.
        """
        cls = self._get_cls_through_file()
        new_net = cls(self._origin_network)
        self._merge_origin_property(new_net)
        return new_net

    def set_saved_file_name(self, file_name: str):
        if file_name.endswith(".py"):
            self._saved_file_name = file_name
        else:
            self._saved_file_name = file_name + ".py"

    def get_saved_file_name(self):
        return self._saved_file_name

    def save_network_to_file(self):
        abs_path = os.path.abspath(self._saved_file_name)
        if os.path.isfile(abs_path):
            os.remove(abs_path)
        with os.fdopen(os.open(self._saved_file_name, os.O_WRONLY | os.O_CREAT, stat.S_IRWXU), 'wb') as f:
            source = self.get_code()
            f.write(source.encode('utf-8'))
            f.flush()

    def insert_to_ast_while_insert_node(self, new_node: Node, base_node: Node, before_node: bool,
                                        node_manager: NodeManager):
        """ insert_to_ast_while_insert_node. """
        if new_node.get_node_type() == NodeType.Input:
            # insert a new input
            self._inputs.append(new_node)
            ast_construct = self.get_ast_root()
            arg: str = new_node.get_targets()[0].value
            ast_arg = ast.arg(arg=arg, annotation=None, type_comment=None)
            AstModifier.append_arg_to_function(ast_construct, ast_arg)
        else:
            # insert a new assign statement
            ast_assign = new_node.get_ast()
            if ast_assign is None:
                func_name = new_node.get_belong_symbol_tree().unique_func_name(new_node.get_name())
                new_node.set_func_name(ScopedValue.create_naming_value(func_name, "self"))
                ast_assign = new_node.update_ast_node()
            if not isinstance(ast_assign, ast.Assign):
                raise ValueError(f"Only support insert ast.Assign or Input now, but get {type(ast_assign)}")
            # Save instance into _origin_network.
            setattr(self._origin_network, new_node.get_name(), new_node.get_instance())
            # Insert ast to __init__ function
            if isinstance(new_node, TreeNode):
                init_code = f"self.{new_node.get_name()} = " \
                            f"{new_node.symbol_tree.get_opt_cls_name()}(obj.{new_node.get_name()})"
            else:
                init_code = f"self.{new_node.get_name()} = obj.{new_node.get_name()}"
            init_ast = ast.parse(init_code).body[0]
            AstModifier.insert_assign_ast_to_function(self._init_func_ast, init_ast)
            # Insert ast to construct_function/class_internal_function
            ast_base_node = base_node.get_ast() if base_node else None
            ast_functiondef = node_manager.get_ast_functiondef()
            if not ast_functiondef:
                raise RuntimeError(f"ast_functiondef is None in node_manager {node_manager.get_manager_name()} "
                                   "when inserting the ast.")
            AstModifier.insert_assign_ast_to_function(ast_functiondef, ast_assign, ast_base_node, before_node)

    def _get_real_node(self, node_or_name: Union[Node, str]) -> Optional[Node]:
        if isinstance(node_or_name, str):
            return self.get_node(node_or_name)
        return node_or_name

    def _handle_custom_obj_in_normalized_args(self, node: Node):
        """
        Convert CustomObjValue type argument to NamingValue type argument by storing custom object to obj.

        Args:
            node (Node): A Node whose arguments and keyword arguments to be handled.
        """
        normalized_args: {str, ScopedValue} = {}
        for key, value in node.get_normalized_args().items():
            if not isinstance(value, ScopedValue):
                raise TypeError("value should be ScopedValue, got: ", type(value))
            if value.type == ValueType.CustomObjValue:
                # Save CustomObjValue into _origin_network(i.e. obj): obj.arg_name = CustomObjValue
                arg_name = self.unique_name(f"arg_{type(value.value).__name__}")
                setattr(self._origin_network, arg_name, value.value)
                # Add new code to __init__(): self.arg_name = obj.arg_name
                new_ast = ast.parse(f"self.{arg_name} = obj.{arg_name}").body[0]
                self._init_func_ast.body.append(new_ast)
                # Modify node's normalized_args: CustomObjValue -> self.arg_name
                normalized_args[key] = ScopedValue.create_naming_value(arg_name, "self")
            else:
                normalized_args[key] = value
        node.set_normalized_args(normalized_args)

    def _get_cls_through_file(self):
        """
        Load rewritten network class of current SymbolTree.
        1. Get source code of current SymbolTree.
        2. Saving source code to a tempfile.
        3. Import rewritten network class using "__import__" function.

        Returns:
            A class handle.
        """
        file_path = os.getcwd()
        file_path = os.path.join(file_path, "rewritten_network")
        if not os.path.exists(file_path):
            try:
                os.mkdir(file_path, mode=0o700)
            except FileExistsError:
                pass
        file_name = f"{self._opt_cls_name}_{id(self)}.py"
        network_file = os.path.join(file_path, file_name)
        with os.fdopen(os.open(network_file, os.O_WRONLY | os.O_CREAT, stat.S_IRWXU), 'wb') as f:
            source = self.get_code()
            f.write(source.encode('utf-8'))
            f.flush()
            os.fsync(f)
        tmp_module_path, tmp_module_file = os.path.split(network_file)
        tmp_module_name = tmp_module_file[:-3]
        sys.path.append(tmp_module_path)
        tmp_module = None

        i = 0
        while not tmp_module:
            spec = importlib.util.spec_from_file_location(tmp_module_name, network_file)
            if spec:
                tmp_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(tmp_module)
            else:
                logger.warning(f"load module {tmp_module_name} failed, retrying.")
                if i > 10:
                    break
                time.sleep(0.5)
                i += 1
        if not tmp_module:
            logger.error(f"load module {tmp_module_name} failed.")
        # Save new module to sys.modules to support inspect.getsource().
        sys.modules[tmp_module_name] = tmp_module
        network_cls = getattr(tmp_module, self._opt_cls_name)
        if network_cls is None:
            raise RuntimeError("Can not find network class:", self._opt_cls_name)
        return network_cls

    def _on_change(self, event: Event):
        self._modified = True
        self.changed(event)

    def _cal_difference_set(self, input, other):
        """Calculate different set of two sets."""
        set1 = set(input)
        set2 = set(other)
        return set1 - set2

    def _merge_origin_property(self, new_net):
        """Merge property of two network."""
        tmp = self._cal_difference_set(dir(self._origin_network), dir(new_net))
        new_attr_names = self._cal_difference_set(tmp, self._deleted_field.keys())
        for name in new_attr_names:
            setattr(new_net, name, getattr(self._origin_network, name))
        # merger cells
        cells = self._cal_difference_set(self._origin_network.name_cells().keys(), new_net.name_cells().keys())
        cells = self._cal_difference_set(cells, self._deleted_node)
        for c in cells:
            new_net.insert_child_to_cell(c, self._origin_network.name_cells()[c])
        # merge primitives
        primitives = self._cal_difference_set(self._origin_network._primitives.keys(), new_net._primitives.keys())
        for p in primitives:
            new_net._primitives[p] = self._origin_network._primitives[p]
