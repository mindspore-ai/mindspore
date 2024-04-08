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
import types
import os
import sys
import ast
import importlib.util
import time
import inspect
from textwrap import dedent
from collections import OrderedDict

from mindspore.nn import Cell
from mindspore import log as logger
from .symbol_tree_dumper import SymbolTreeDumper
from ..node import Node, TreeNode, ControlFlow, CallFunction, NodeManager
from ..api.node_type import NodeType
from ..api.scoped_value import ScopedValue, ValueType
from ..ast_helpers import AstModifier, AstReplacer, StrChecker, AstFinder, AstClassFinder, AstFunctionFinder, \
    AstImportFinder
from ..common.namer import TargetNamer, NodeNamer, ClassNamer
from ..common.observer import Observer
from ..common.observable import Observable
from ..common.event import Event

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


class SymbolTree(Observer, Observable, NodeManager):
    """
    A symbol-tree usually corresponding to forward method of a network.

    Rewrite recommend using SymbolTreeBuilder to instantiate an instance of SymbolTree rather than invoking constructor
    of SymbolTree directly.

    Args:
        origin_network (Cell): A handler to original network instance.
        module_ast (ast.Module): An instance of ast.AST represents ast node of original network.
    """
    # whether parse CallFunction node inserted by user.
    _unparse_inserted_function = True

    def __init__(self, origin_network: Cell, module_ast: ast.Module):
        Observer.__init__(self)
        Observable.__init__(self)
        self._node_namer = NodeNamer()
        self._node_namer.add_name('obj')
        NodeManager.__init__(self)
        NodeManager.set_manager_node_namer(self, self._node_namer)
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
        # {ast_function: [import_asts]}
        self._external_ast: Dict[ast.FunctionDef, list] = OrderedDict()
        # {ast_class: [import_asts]}
        self._father_class_ast: Dict[ast.ClassDef, list] = OrderedDict()
        self._modified = False
        self._saved_file_name = "./network_define.py"
        # used to insert "sys.path.append(xxx)"
        self._net_file_paths = []
        self._tmp_import_strs = []
        self._tmp_unmodified_strees: {type, List[SymbolTree]} = {}
        self._tmp_replacers = []
        # user custom codes
        self._custom_codes: List[ast.AST] = []
        # local primitive instances initialized during forward method, e.g. abs_inst = P.Abs()
        self._local_prim_inits: List[Node] = []

    @staticmethod
    def _remove_unused_import(module_ast):
        """remove unused import in self._module_ast"""
        import_nodes: List[Union[ast.Import, ast.ImportFrom]] = []

        def is_divider(ast_node):
            """judge if ast node is divider of new class or function by checking ast.Expr of '#'."""
            return isinstance(ast_node, ast.Expr) and isinstance(ast_node.value, ast.Name) and ast_node.value.id == '#'

        for ast_node in module_ast.body[:]:
            if isinstance(ast_node, (ast.Import, ast.ImportFrom)):
                import_nodes.append(ast_node)
            if isinstance(ast_node, (ast.ClassDef, ast.FunctionDef)):
                str_checker = StrChecker(ast_node)
                for import_node in import_nodes:
                    for alias in import_node.names[:]:
                        name = alias.asname if alias.asname else alias.name
                        if name == '*':
                            continue
                        if not str_checker.check(name):
                            import_node.names.remove(alias)
                    if not import_node.names:
                        module_ast.body.remove(import_node)
            if is_divider(ast_node):
                import_nodes.clear()

    @staticmethod
    def _remove_duplicated_import(module_ast):
        """Remove duplicated import of 'net'."""
        imports = set()
        futures = set()
        names = set()

        class TransImportNode(ast.NodeTransformer):
            """Find all import nodes from input ast node."""

            def visit_ClassDef(self, node: ast.ClassDef) -> Any:
                if node.name not in names:
                    names.add(node.name)
                    return node
                return None

            def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
                if node.name not in names:
                    names.add(node.name)
                    return node
                return None

            def visit_Try(self, node: ast.Try) -> Any:
                if isinstance(node.body[0], (ast.Import, ast.ImportFrom)):
                    import_str = astunparse.unparse(node)
                    if import_str not in imports:
                        imports.add(import_str)
                        return node
                return None

            def visit_Import(self, node: ast.Import) -> Any:
                import_str = astunparse.unparse(node)
                if import_str not in imports:
                    imports.add(import_str)
                    return node
                return None

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
                        return None
                    # remove modules which have been defined in the code file
                    # it occurs when class A is a father class and other sub-classes import A
                    for alias in node.names[:]:
                        if alias.name in names:
                            node.names.remove(alias)
                    # if the alias(es) in node.names are all removed, this import statement should be removed
                    if not node.names:
                        return None
                    return node
                return None

        get_node_handler = TransImportNode()
        get_node_handler.generic_visit(module_ast)

    @staticmethod
    def _remove_arg_annotations(module_ast):
        """Remove annotations in ast.arg to avoid 'xxx is not defined'."""
        ast_args: List[ast.arg] = AstFinder(module_ast).find_all(ast.arg)
        for ast_arg in ast_args:
            ast_arg.annotation = None

    @staticmethod
    def _check_import(import_path: str, import_module: str):
        """
        Check whether import operation is valid when importing module from specific path.
        """
        if import_path not in sys.path:
            sys.path.append(import_path)
        try:
            importlib.import_module(name=import_module)
        except (ValueError, ImportError) as e:
            logger.info(f"Test import {import_module} from {import_path} failed: {e}.")
            return False
        except Exception as e: # pylint: disable=W0703
            logger.info(f"Test import {import_module} from {import_path} failed: {e}.")
            return False
        return True

    @staticmethod
    def _process_relative_import(import_node: Union[ast.Import, ast.ImportFrom], file_path: str):
        """Process relative imports"""
        file_path = os.path.normcase(file_path)
        file_path = os.path.normpath(file_path)
        if isinstance(import_node, ast.ImportFrom):
            # pad the ImportFrom with parent path
            # e.g. from ..C import xxx -> from A.B.C import xxx
            import_module = SymbolTree._get_valid_import_info(import_node, file_path)
            if import_module:
                import_node = ast.ImportFrom(module=import_module, names=import_node.names, level=0)
        return import_node

    @staticmethod
    def _get_valid_import_info(import_node: ast.ImportFrom, file_path: str):
        """Get valid import info while import_node.module is at form of relative path"""
        file_path = os.path.dirname(os.path.abspath(file_path))
        # get real path from import_node.level
        # from .(A) import xxx: current path
        # from ..(A) import xxx: last level path
        level = import_node.level
        # from A import xxx: it does not need to pad, directly return the module name
        if level == 0:
            return import_node.module
        if level > 1:
            for _ in range(level - 1):
                file_path = os.path.dirname(file_path)
        file_path_tmp = file_path[:]
        max_level_count = file_path.count(os.path.sep) - 1
        level_count = 0
        # suffix is the module_name, e.g. 'A' in 'from ..(A) import xxx'
        suffix = ''
        if import_node.module:
            suffix = '.' + import_node.module
        while level_count < max_level_count:
            file_path_tmp = os.path.dirname(file_path_tmp)
            if file_path_tmp not in sys.path:
                logger.debug(f"{file_path_tmp} not in sys.path, try upper level.")
                level_count += 1
                continue
            import_module = file_path[len(file_path_tmp) + 1:].replace(os.path.sep, '.') + suffix
            if SymbolTree._check_import(file_path_tmp, import_module):
                # try test code success
                return import_module
            # test import ast failed, try upper level
            level_count += 1
            logger.info(f"Try upper level.")
        # try codes with all level failed
        logger.info(f"Test import code: {astunparse.unparse(import_node).strip()} failed, ignore this import code.")
        return None

    @staticmethod
    def insert_to_ast_while_insert_input(new_node: Node, node_manager: NodeManager):
        """update ast when inserting NodeType.Input node"""
        if not isinstance(node_manager, (SymbolTree, CallFunction)):
            raise ValueError(f"Only support insert Input node into a SymbolTree or a node with type of "
                             f"CallFunction, but get {type(node_manager)}")
        # insert a new input
        node_manager.get_input_nodes().append(new_node)
        ast_function: ast.FunctionDef = node_manager.get_manager_ast()
        arg: str = new_node.get_targets()[0].value
        ast_arg = ast.arg(arg=arg, annotation=None, type_comment=None)
        AstModifier.append_arg_to_function(ast_function, ast_arg)

    @staticmethod
    def insert_to_ast_while_insert_cell_primitive(new_node: Node, base_node: Node, before_node: bool,
                                                  node_manager: NodeManager, stree):
        """update ast when inserting NodeType.CallCell or NodeType.CallPrimitive node"""
        # create a new assign statement
        ast_assign = new_node.get_ast()
        if ast_assign is None:
            func_name = stree.unique_func_name(new_node.get_name())
            new_node.set_func_name(ScopedValue.create_naming_value(func_name, "self"))
            ast_assign = new_node.update_ast_node()
        if not isinstance(ast_assign, ast.Assign):
            raise ValueError(f"Only support insert ast.Assign or Input now, but get {type(ast_assign)}")
        # Save instance into _origin_network.
        setattr(stree.get_origin_network(), new_node.get_name(), new_node.get_instance())
        # Insert ast to __init__ function
        if isinstance(new_node, TreeNode):
            init_code = f"{new_node.get_func_name()} = " \
                        f"{new_node.symbol_tree.get_opt_cls_name()}(obj.{new_node.get_name()})"
        else:
            init_code = f"{new_node.get_func_name()} = obj.{new_node.get_name()}"
        init_ast = ast.parse(init_code).body[0]
        AstModifier.insert_ast_to_function(stree.get_init_func_ast(), init_ast)
        # Insert ast to construct_function/class_internal_function
        ast_base_node = base_node.get_ast() if base_node else None
        ast_node_manager = node_manager.get_manager_ast()
        if not ast_node_manager:
            raise RuntimeError(f"ast_node_manager is None in node_manager {node_manager.get_manager_name()} "
                               "when inserting the ast.")
        AstModifier.insert_ast_to_ast(ast_node_manager, ast_assign, ast_base_node, before_node)

    @staticmethod
    def insert_to_ast_while_insert_function(new_node: CallFunction, base_node: Node, before_node: bool,
                                            node_manager: NodeManager, stree: 'SymbolTree'):
        """update ast when inserting NodeType.CallFunction node"""
        func_name = str(new_node.get_func_name())
        # create a new assign statement
        ast_assign = new_node.get_ast()
        if ast_assign is None:
            ast_assign = new_node.update_ast_node()
        # Insert ast to node_manager
        ast_base_node = base_node.get_ast() if base_node else None
        ast_node_manager = node_manager.get_manager_ast()
        if not ast_node_manager:
            raise RuntimeError(f"ast_node_manager is None in node_manager {node_manager.get_manager_name()} "
                               "when inserting the ast.")
        AstModifier.insert_ast_to_ast(ast_node_manager, ast_assign, ast_base_node, before_node)
        # Ignore Python builtin functions
        func_obj = new_node.get_instance()
        if isinstance(func_obj, types.BuiltinFunctionType):
            logger.warning(f"Ignore built in function: {func_name}")
            return
        # get ast.FunctionDef
        source_code = inspect.getsource(func_obj)
        ast_functiondef = ast.parse(dedent(source_code)).body[0]
        if SymbolTree._unparse_inserted_function or not isinstance(ast_functiondef, ast.FunctionDef):
            logger.debug(f"import '{func_name}' to access function object")
            # add import to make sure that the function object can be accessed.
            module = inspect.getmodule(func_obj)
            top_node_manager = node_manager.get_top_manager()
            belonging_ast = None if isinstance(top_node_manager, SymbolTree) else top_node_manager.get_manager_ast()
            stree.add_import(module, func_name, belonging_ast)
            return
        # parse nodes in inserted function.
        new_node.set_manager_ast(ast_functiondef)
        new_node.set_manager_node_namer(stree.get_node_namer())
        stree.get_external_ast()[ast_functiondef] = []
        # import module which function defined in
        func_file_path = inspect.getabsfile(func_obj)
        stree.save_imports_from_file(func_file_path, ast_functiondef)
        # expand ast codes in function
        from ..ast_helpers import AstFlattener
        ast_functiondef = AstFlattener().transform(ast_functiondef, [func_name], stree)
        # parse ast codes into CallFunction Node
        from ..parsers import ParserRegister
        parser = ParserRegister.instance().get_parser(ast.FunctionDef)
        parser.process(stree, ast_functiondef, node_manager=new_node)

    @staticmethod
    def insert_to_ast_while_insert_node(new_node: Node, base_node: Node, before_node: bool):
        """ insert_to_ast_while_insert_node. """
        stree = new_node.get_belong_symbol_tree()
        if not stree:
            raise ValueError(f"When inserting node to ast, the belonging symbol tree of new_node is None.")
        node_manager = new_node.get_node_manager()
        if not isinstance(node_manager, (SymbolTree, CallFunction, ControlFlow)):
            raise ValueError(f"When inserting node to ast, the node_manager of new_node {new_node.get_name()} can "
                             f"only be one of [SymbolTree, CallFunction, ControlFlow], but get {type(node_manager)}")
        if new_node.get_node_type() == NodeType.Input:
            SymbolTree.insert_to_ast_while_insert_input(new_node, node_manager)
        elif new_node.get_node_type() in (NodeType.CallCell, NodeType.CallPrimitive, NodeType.Tree):
            SymbolTree.insert_to_ast_while_insert_cell_primitive(new_node, base_node, before_node, node_manager,
                                                                 stree)
        elif new_node.get_node_type() == NodeType.CallFunction:
            SymbolTree.insert_to_ast_while_insert_function(new_node, base_node, before_node, node_manager, stree)
        else:
            raise ValueError(f"When insert node '{new_node.get_name()}' into ast, the type of node can only be "
                             f"one of [Input, CallCell, CallPrimitive, CallFunction, Tree], but got "
                             f"{new_node.get_node_type()}.")

    @staticmethod
    def get_node_full_name(node: Node) -> str:
        """Get full name of node"""
        name = node.get_manager_name() if isinstance(node, NodeManager) else node.get_name()
        # traverse node_manager with type of Node
        node_manager = node.get_node_manager()
        while isinstance(node_manager, Node):
            name = f"{node_manager.get_manager_name()}.{name}"
            node_manager = node_manager.get_node_manager()
        # type of node_manager is SymbolTree now
        name = f"{node_manager.get_manager_name()}.{name}"
        return name

    def local_prim_inits(self) -> List[Node]:
        """get local primitives constructed during forward method"""
        return self._local_prim_inits

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
        NodeManager.set_manager_ast(self, ast_node)

    def get_class_ast(self):
        """
        Getter of `_class_ast`.

        Returns:
            An instance of ast.ClassDef represents ast node of corresponding network class.
        """
        return self._class_ast

    def set_class_ast(self, ast_node: ast.ClassDef):
        """
        Setter of `_class_ast`.

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
        node_users = []
        for target_users in real_node.get_target_users().values():
            if not target_users:
                continue
            if target_users not in node_users:
                node_users.extend(target_users)
        return node_users

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
                raise ValueError(f"Position is not in current SymbolTree, node:{stree.get_ori_cls_name()}, "
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
                self.insert_to_ast_while_insert_node(new_node, base_node, before_node)
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
        for input_node in node_manager.get_input_nodes():
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
            if isinstance(node, ControlFlow):
                ret = AstModifier.earse_ast_of_control_flow(self._root_ast.body, node.get_ast(), node.is_orelse)
            else:
                ret = AstModifier.erase_ast_from_function(self._root_ast, node.get_ast())
            if not ret:
                raise RuntimeError(f"erase node failed, node {node.get_name()} not in function ast tree.")
        else:
            node_manager.erase_node(node)
        node.set_belong_symbol_tree(None)
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
        node.get_node_manager().on_update_arg(node, index, old_arg, new_arg)

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
        real_dst_node.get_node_manager().on_update_arg_by_node(real_dst_node, arg_idx, real_src_node, out_idx)

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

    def all_nodes(self, subtree_nodes: bool = True):
        """
        Get all nodes including nodes in CallFunction node, CellContainer node and sub symbol tree.

        Args:
            subtree_nodes (bool): Whether include nodes in subtree. Default: True.

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
        if subtree_nodes:
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

    def get_node_tabulate(self, all_nodes: bool = False) -> str:
        """
        Get nodes information and nodes' topological relations.

        Args:
            all_nodes (bool): Print nodes out of construct functions, such as nodes in CallFunction
                nodes, CellContainer nodes and sub symbol trees.

        Returns:
            String of nodes' information and topological relations.
        """
        try:
            from tabulate import tabulate # pylint: disable=unused-import,reportMissingModuleSource
        except ImportError:
            logger.warning("print_node_tabulate relies on the library `tabulate`, "
                           "which could not be found on this machine. Run `pip "
                           "install tabulate` to install the library.")
            return ""
        dump_str = NodeManager.dump(self, self.get_manager_name())
        if all_nodes:
            node_managers = [self]
            while node_managers:
                node_manager = node_managers.pop()
                for node in node_manager.nodes():
                    if isinstance(node, NodeManager):
                        dump_str += node.dump(SymbolTree.get_node_full_name(node))
                        node_managers.append(node)
            for tree_node in self.get_tree_nodes():
                stree = tree_node.symbol_tree
                dump_str += stree.get_node_tabulate(all_nodes)
        return dump_str

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

    def deduplicate_unmodified_stree(self, code_bodies):
        """
        Init function may be different even if stree is not modified manually, when subnets in stree is
        initialized by different arguments.
        In this case, we need to wait for code_bodies being fully generated, so that the name of subnets
        will be updated, then we can deduplicate again according to ast of init function.
        """
        # prepare AstClassFinder and AstReplacer
        if sys.version_info >= (3, 9):
            class_finder = AstClassFinder(ast.Module(body=code_bodies, type_ignores=[]))
            name_replacer = AstReplacer(ast.Module(body=code_bodies, type_ignores=[]))
        else:
            class_finder = AstClassFinder(ast.Module(body=code_bodies))
            name_replacer = AstReplacer(ast.Module(body=code_bodies))
        # deduplicate all unmodified strees in self._tmp_unmodified_strees
        deduplicated = False
        for _, unmodified_strees in self._tmp_unmodified_strees.items():
            if len(unmodified_strees) <= 1:
                continue
            init_func_codes = [astunparse.unparse(stree.get_init_func_ast()) for stree in unmodified_strees]
            # If the index of an element is not its own, it means that it is a duplicate element
            to_be_erase = []
            for idx, code in enumerate(init_func_codes):
                first_idx = init_func_codes.index(code)
                if first_idx != idx:
                    first_stree_cls_name = unmodified_strees[first_idx].get_opt_cls_name()
                    duplicated_stree_cls_name = unmodified_strees[idx].get_opt_cls_name()
                    logger.debug(f"replace stree:{duplicated_stree_cls_name} to {first_stree_cls_name}.")
                    # delete duplicated class from code_bodies
                    results = class_finder.find_all(duplicated_stree_cls_name)
                    for ast_cls in results:
                        code_bodies.remove(ast_cls)
                    # replace name of duplicated class in code_bodies to first_stree_cls_name
                    name_replacer.replace_all(duplicated_stree_cls_name, first_stree_cls_name)
                    # record deduplicated stree
                    to_be_erase.append(idx)
                    deduplicated = True
            # remove class in self._tmp_unmodified_strees
            for idx in reversed(to_be_erase):
                unmodified_strees.pop(idx)

        # the name of subnets is updated, so we need to deduplicate again.
        if deduplicated:
            self._tmp_replacers.append(name_replacer)
            self.deduplicate_unmodified_stree(code_bodies)

    def update_unmodified_stree(self, stree, code_bodies) -> bool:
        """
        For the unmodified symbol tree, only one definition code remains in the generated code.
        Everywhere else calling this symbol tree will use the class in this definition code.
        """
        # all modified ast.ClassDef will be exported to code
        if stree.is_modified():
            logger.debug(f"stree:{stree.get_opt_cls_name()} is modified.")
            return False
        # all un-modified ast.ClassDef only keep one instance
        unmodified_strees = self._tmp_unmodified_strees.get(type(stree.get_origin_network()))
        if not unmodified_strees:
            self._tmp_unmodified_strees[type(stree.get_origin_network())] = [stree]
            logger.debug(f"stree:{stree.get_opt_cls_name()} is the first stree.")
            return False
        # Init function may be different even if stree is not modified, when subnets in stree is
        # initialized by different arguments.
        first_stree = unmodified_strees[0]
        first_stree_cls_name = first_stree.get_opt_cls_name()
        if astunparse.unparse(stree.get_init_func_ast()) != astunparse.unparse(first_stree.get_init_func_ast()):
            # init ast may be updated after inserting subtrees of stree, so we need to save unmodified strees
            # and deduplicate later
            self._tmp_unmodified_strees[type(stree.get_origin_network())].append(stree)
            logger.debug(f"init func different, stree:{stree.get_opt_cls_name()}, first_stree:{first_stree_cls_name}.")
            return False
        # Un-modified ast.ClassDef already exist in code_bodies,
        # replace class name to class name of first un-modified ast.ClassDef.
        if sys.version_info >= (3, 9):
            replacer = AstReplacer(ast.Module(body=code_bodies, type_ignores=[]))
        else:
            replacer = AstReplacer(ast.Module(body=code_bodies))
        logger.debug(f"replace stree:{stree.get_opt_cls_name()} to {first_stree_cls_name}.")
        replacer.replace_all(stree.get_class_ast().name, first_stree_cls_name)
        self._tmp_replacers.append(replacer)
        return True

    def init_code_bodies(self, code_bodies: list) -> int:
        """Init code bodied"""
        # Add basic imports
        code_bodies.append(ast.Import([ast.alias(name='sys', asname=None)]))
        code_bodies.append(ast.Import([ast.alias(name='mindspore', asname=None)]))
        code_bodies.append(ast.ImportFrom(module='mindspore', names=[ast.alias(name='nn', asname=None)], level=0))
        code_bodies.append(ast.ImportFrom(module='mindspore.nn', names=[ast.alias(name='Cell', asname=None)], level=0))
        code_bodies.append(ast.ImportFrom(module='mindspore.ops',
                                          names=[ast.alias(name='functional', asname='F')], level=0))
        code_bodies.append(ast.Expr(ast.Name("#", ast.Load())))
        # Add user custom codes into code_bodies
        custom_codes = self.get_custom_codes()
        for code_ast in custom_codes:
            code_bodies.append(code_ast)
        code_bodies.append(ast.Expr(ast.Name("#", ast.Load())))
        return len(code_bodies)

    def convert_stree_to_code_bodies(self, stree: 'SymbolTree', code_bodies: list, dividing_pos=0) -> int:
        """
        Convert nodes in stree to code_bodies
        - Add external function asts into code_bodies
        - Add father class asts into code_bodies
        - Add import asts of symbol tree into code_bodies
        - Add user custom codes into code_bodies
        - Add class asts of symbol tree into code_bodies
        - Add subtrees to code_bodies
        """
        insert_pos = dividing_pos
        # Add external asts into code_bodies
        for ast_func, import_asts in reversed(stree.get_external_ast().items()):
            if self.check_body_exist(ast_func, code_bodies):
                continue
            # add imports of external_ast
            self._tmp_import_strs.clear()
            for ast_import in import_asts:
                if not self.check_body_exist(ast_import, code_bodies):
                    code_bodies.insert(insert_pos, ast_import)
                    insert_pos += 1
            # add external_ast
            code_bodies.insert(insert_pos, ast_func)
            insert_pos += 1
            # add divide
            code_bodies.insert(insert_pos, ast.Expr(ast.Name("#", ast.Load())))
            insert_pos += 1

        # Add father class asts into code_bodies
        for ast_class, import_asts in stree.get_father_class_ast().items():
            if self.check_body_exist(ast_class, code_bodies):
                continue
            # add imports of father class
            self._tmp_import_strs.clear()
            for ast_import in import_asts:
                if not self.check_body_exist(ast_import, code_bodies):
                    code_bodies.insert(insert_pos, ast_import)
                    insert_pos += 1
            # add ast of father class
            code_bodies.insert(insert_pos, ast_class)
            insert_pos += 1
            # add divide
            code_bodies.insert(insert_pos, ast.Expr(ast.Name("#", ast.Load())))
            insert_pos += 1

        # external functions and father class are above the dividing_pos to support deduplication.
        dividing_pos = insert_pos

        # Add import asts of symbol tree into code_bodies
        self._tmp_import_strs.clear()
        for body in stree.get_import_asts():
            if not self.check_body_exist(body, code_bodies):
                code_bodies.insert(insert_pos, body)
                insert_pos += 1

        # Add class asts of symbol tree into code_bodies
        if stree.get_module_ast():
            for body in stree.get_module_ast().body:
                if self.check_body_exist(body, code_bodies):
                    continue
                code_bodies.insert(insert_pos, body)
                insert_pos += 1

        # add divide
        code_bodies.insert(insert_pos, ast.Expr(ast.Name("#", ast.Load())))
        insert_pos += 1

        # Add subtrees to code_bodies
        for node in stree.get_tree_nodes():
            sub_stree = node.symbol_tree
            # For the unmodified class, update class name to name of first class
            if self.update_unmodified_stree(sub_stree, code_bodies):
                continue
            dividing_pos = self.convert_stree_to_code_bodies(node.symbol_tree, code_bodies, dividing_pos)

        # return new dividing position
        return dividing_pos

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
        begin_pos = self.init_code_bodies(code_bodies)
        self.convert_stree_to_code_bodies(self, code_bodies, begin_pos)
        self.deduplicate_unmodified_stree(code_bodies)
        if sys.version_info >= (3, 9):
            gencode_module = ast.Module(body=code_bodies, type_ignores=[])
        else:
            gencode_module = ast.Module(body=code_bodies)
        SymbolTree._remove_unused_import(gencode_module)
        self._process_duplicate_name_modules(gencode_module)
        SymbolTree._remove_duplicated_import(gencode_module)
        SymbolTree._remove_arg_annotations(gencode_module)
        ast.fix_missing_locations(self._module_ast)
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
        # update parameters' names to fix duplicated names bug
        # which occurs after inserting cell to celllist/sequentialcell
        new_net.update_parameters_name()
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


    def flatten_nodes(self, node, erase_another_branch: bool = False, erase_nodes_after_return: bool = False):
        """Flatten nodes in ControlFlow node."""
        if not isinstance(node, ControlFlow):
            raise ValueError(f"For flatten_nodes, the type of node can only be ControlFlow, but got {type(node)}.")
        upper_node_manager = node.get_node_manager()
        if isinstance(upper_node_manager, (SymbolTree, CallFunction)):
            ast_bodies = upper_node_manager.get_manager_ast().body
        elif isinstance(upper_node_manager, ControlFlow):
            ast_bodies = upper_node_manager.get_manager_ast()
        else:
            raise ValueError("For flatten_nodes, the node can only be contained in [SymbolTree, CallFunction, "
                             f"ControlFlow], but the node is in {type(upper_node_manager)}.")
        base_node = node.orelse_node if node.orelse_node else node.body_node
        for n in node.nodes()[:]:
            self.erase_node(n)
            self.insert_node(n, base_node, False, upper_node_manager, False)
            AstModifier.insert_ast_to_bodies(ast_bodies, n.get_ast(), base_node.get_ast(), False)
            base_node = n
        self.erase_node(node)
        # remove another branch
        if erase_another_branch:
            if node.is_orelse:
                self.erase_node(node.body_node)
            elif node.orelse_node is not None:
                self.erase_node(node.orelse_node)
        # remove nodes after return node
        if erase_nodes_after_return:
            has_return = False
            for n in upper_node_manager.nodes():
                if has_return:
                    logger.warning(f"Node {n.get_name()} which is behind the flatten return node is "
                                   f"automatically erased.")
                    self.erase_node(n)
                elif n.get_node_type() == NodeType.Output:
                    has_return = True

    def eval_ast_result(self, ast_node: ast.AST) -> (bool, bool):
        """
        Eval ast_node and get result, only used in control flow node.
        """
        # ast.Constant can be check without eval
        if isinstance(ast_node, ast.Constant):
            return True, bool(ast.value)
        # Get the module where the code of ast_node is located
        file_path = inspect.getfile(type(self.get_origin_network()))
        module = None
        for m in list(sys.modules.values()):
            if hasattr(m, "__file__") and m.__file__ and os.path.normcase(m.__file__) == os.path.normcase(file_path):
                module = m
                break
        if not module:
            logger.warning("Failed to get module of ast_node.")
            return False, False
        # eval ast_node and get result
        logger.debug(f"Eval ast node: {astunparse.unparse(ast_node)}")
        ast_expr = ast.Expression(ast_node)
        ast_expr = ast.fix_missing_locations(ast_expr)
        try:
            # eval with ast make this operation free of instruction injection
            # pylint: disable=eval-used
            result = eval(compile(ast_expr, "eval_ast_result", "eval"), {**globals(), **module.__dict__}, locals())
        except Exception as e: # pylint: disable=broad-except
            logger.debug(f"Cannot get result of ast_node by eval, err:{e}")
            return False, False
        logger.debug(f"Eval ast result success, result: {result}")
        return True, bool(result)

    def flatten_static_if_control_flow(self):
        """
        For static if control flow, flatten codes in branch which will be executed and erase another branch.
        """
        for node in self.all_nodes()[:]:
            if not node.get_belong_symbol_tree():
                # the node has been erased
                continue
            if isinstance(node, ControlFlow) and node.test_result is not None:
                stree = node.get_belong_symbol_tree()
                if node.test_result:
                    stree.flatten_nodes(node.body_node, True, True)
                else:
                    if node.orelse_node is not None:
                        stree.flatten_nodes(node.orelse_node, True, True)
                    else:
                        stree.erase_node(node.body_node)

    def add_custom_codes(self, code: str):
        """Add user custom codes"""
        code_ast = ast.parse(code)
        self._custom_codes.extend(code_ast.body)

    def get_custom_codes(self) -> List[ast.AST]:
        """Add user custom codes"""
        return self._custom_codes

    def save_file_path_to_sys(self, level_num, file_path, belonging_ast: ast.AST = None):
        """
        Save file path into stree._import_asts. `level_num` is used when level exist in ast.ImportFrom.

        When level_num = 0(e.g. from xxx import yyy), current path will be saved.
        When level_num = 1(e.g. from .xxx import yyy), current path will be saved.
        When level_num = 2(e.g. from ..xxx import yyy), the path one level above the current path will be saved.
        """
        file_path = os.path.dirname(os.path.abspath(file_path))
        file_path = os.path.normcase(file_path)
        file_path = os.path.normpath(file_path)
        if level_num > 1:
            for _ in range(level_num - 1):
                file_path = os.path.dirname(file_path)
        sys_path_append_ast = ast.parse(f"sys.path.insert(0, r'{file_path}')").body[0]
        # add imports to import_asts of belonging_ast
        import_asts = self._get_imports_list_of_ast(belonging_ast)
        import_asts.append(ast.Import([ast.alias(name='sys', asname=None)]))
        import_asts.append(sys_path_append_ast)

    def save_imports_from_file(self, file_path, belonging_ast: ast.AST = None):
        """Save imports from file"""
        self.save_file_path_to_sys(0, file_path, belonging_ast)
        if not os.path.exists(file_path):
            raise RuntimeError(f"For MindSpore Rewrite, in module parser, file {file_path} not exist.")
        with open(file_path, "r", encoding="utf-8") as f:
            source_code = f.read()
            import_nodes = AstImportFinder(ast.parse(dedent(source_code))).get_import_node()
        if not import_nodes:
            return
        # add imports to import_asts of belonging_ast
        import_asts = self._get_imports_list_of_ast(belonging_ast)
        for import_node in import_nodes:
            import_node = SymbolTree._process_relative_import(import_node, file_path)
            if import_node:
                import_asts.append(import_node)

    def add_import(self, module: types.ModuleType, name: str, belonging_ast: None):
        """add codes: from `module` import `name`"""
        if not isinstance(module, types.ModuleType):
            raise TypeError(f"For add_import, module should be ModuleType, but got {type(module)}")
        if not hasattr(module, name):
            logger.info(f"module {module.__name__} doesn't have attr '{name}', it may be a local variable.")
            return
        # add imports to import_asts of belonging_ast
        import_asts = self._get_imports_list_of_ast(belonging_ast)
        if module.__name__ == "__main__":
            # get attr from module instead of import to avoid duplicate execution of __main__ module
            code = f"{name} = getattr(sys.modules['__main__'], '{name}')"
            code_ast = ast.parse(code).body[0]
            import_asts.append(code_ast)
        elif module.__name__ == "builtins":
            # built-in functions are not need to be imported
            pass
        else:
            # add import of obj to ast
            func_file_path = inspect.getabsfile(module)
            func_file_path = os.path.normcase(func_file_path)
            prefix_paths = []
            for path in sys.path:
                path = os.path.normcase(path)
                if func_file_path.startswith(path):
                    prefix_paths.append(path)
            prefix_paths.sort(key=len, reverse=True)
            for path in prefix_paths:
                import_path = func_file_path[len(path):]
                import_str = import_path.replace(os.path.sep, '.')
                import_str = import_str[1:] # remove first '.'
                mod = import_str.rsplit('.', 1)[0]
                if SymbolTree._check_import(func_file_path[:len(path)], mod):
                    import_node = ast.ImportFrom(module=mod, names=[ast.alias(name=name, asname=None)], level=0)
                    import_asts.append(import_node)
                    break
            else:
                self.save_file_path_to_sys(0, func_file_path, belonging_ast)
                mod = os.path.basename(func_file_path).rsplit('.')[0]
                import_node = ast.ImportFrom(module=mod, names=[ast.alias(name=name, asname=None)], level=0)
                import_asts.append(import_node)

    def _get_imports_list_of_ast(self, belonging_ast: ast.AST):
        # get import_asts of belonging_ast
        import_asts = self._import_asts
        if belonging_ast is not None:
            if belonging_ast in self._father_class_ast:
                import_asts = self._father_class_ast.get(belonging_ast)
            elif belonging_ast in self._external_ast:
                import_asts = self._external_ast.get(belonging_ast)
        return import_asts

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
            raise ImportError(f"load module {tmp_module_name} failed.")
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
        # pylint: disable=protected-access
        primitives = self._cal_difference_set(self._origin_network._primitives.keys(), new_net._primitives.keys())
        for p in primitives:
            new_net._primitives[p] = self._origin_network._primitives[p] # pylint: disable=protected-access

    def _process_duplicate_name_modules(self, module_ast: ast.Module):
        """Adjust names of imported modules with same name and different import path."""
        # {name1: [path1, path2, ...], ...}
        name_path_dict: Dict[str, List[str]] = {}
        # names of modules need to be suffixed: {name1: suffixed_name1, ...}
        name_need_suffix: Dict[str, str] = {}
        # used to record replace actions in ast.ImportFrom
        import_replacer = AstReplacer(None)
        self._tmp_replacers.append(import_replacer)

        def suffix_alias(alias: ast.alias, suffix: int):
            """suffix the name of alias in ast.ImportFrom"""
            new_name = f"{alias.asname}_{suffix}" if alias.asname else f"{alias.name}_{suffix}"
            import_replacer._trace.append((alias, 'asname', alias.asname, new_name)) # pylint: disable=protected-access
            alias.asname = new_name
            return new_name

        def is_divider(ast_node):
            """judge if ast node is divider of new class or function by checking ast.Expr of '#'."""
            return isinstance(ast_node, ast.Expr) and isinstance(ast_node.value, ast.Name) and ast_node.value.id == '#'

        def record_imports(ast_node: ast.ImportFrom):
            """record name and path of imported modules to find the duplicate name modules."""
            for alias in ast_node.names[:]:
                name = alias.asname if alias.asname else alias.name
                if name == '*':
                    continue
                # current name is firstly imported, just record it
                if name not in name_path_dict:
                    name_path_dict[name] = [ast_node.module]
                    continue
                # current name is imported before, check whether it is a duplicated name
                for idx, path in enumerate(name_path_dict[name]):
                    if path.startswith(ast_node.module):
                        # e.g. origin code is 'from a.b.c import A' and new code is 'from a.b import A'
                        # then we update name_path_dict[name][idx] from 'a.b.c' to 'a.b' and update name to A_{idx}
                        name_path_dict[name][idx] = ast_node.module
                        if idx > 0:
                            name_need_suffix[name] = suffix_alias(alias, idx)
                        break
                    elif ast_node.module.startswith(path):
                        # e.g. origin code is 'from a.b import A' and new code is 'from a.b.c import A'
                        # then we just need to update name to A_{idx}
                        if idx > 0:
                            name_need_suffix[name] = suffix_alias(alias, idx)
                        break
                else:
                    # current name is imported from a new path, save the path and update the name
                    name_path_dict[name].append(ast_node.module)
                    name_need_suffix[name] = suffix_alias(alias, len(name_path_dict[name]) - 1)

        def suffix_names_in_ast(ast_node: Union[ast.ClassDef, ast.FunctionDef]):
            """suffix names in ast.ClassDef or ast.FunctionDef"""
            if not name_need_suffix:
                return
            name_replacer = AstReplacer(ast_node)
            self._tmp_replacers.append(name_replacer)
            for name, new_name in name_need_suffix.items():
                name_replacer.replace_all(name, new_name)

        for ast_node in module_ast.body:
            if isinstance(ast_node, ast.ImportFrom):
                record_imports(ast_node)
            if isinstance(ast_node, (ast.ClassDef, ast.FunctionDef)):
                suffix_names_in_ast(ast_node)
            if is_divider(ast_node):
                name_need_suffix.clear()
