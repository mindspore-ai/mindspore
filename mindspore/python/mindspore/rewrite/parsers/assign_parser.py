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
"""Parse ast.Assign in construct function to node of SymbolTree."""
from typing import Union, List, Dict
import types
import os
import ast
import sys
import inspect
import builtins
from textwrap import dedent

from mindspore import log as logger
from mindspore.nn import Cell, SequentialCell, CellList
from mindspore.ops.primitive import Primitive
import mindspore.ops.functional as F
from . import Parser, ParserRegister, reg_parser
from ..symbol_tree import SymbolTree
from ..node import Node, TreeNode, NodeManager, CallFunction, CellContainer, ControlFlow, LocalPrim
from ..api.scoped_value import ScopedValue
from ..ast_helpers import AstFlattener, AstConverter, AstFinder
from ..common.error_log import error_str
from ..common.namespace import is_subtree, is_ms_function, is_third_party
from ..common.namer import FunctionNamer


if sys.version_info >= (3, 9):
    import ast as astunparse # pylint: disable=reimported, ungrouped-imports
else:
    import astunparse


class AssignParser(Parser):
    """Parse ast.Assign in construct function to node of SymbolTree."""

    # Types for creating Cell Container node
    types_for_cell_container = [SequentialCell,]
    # If mindspore built-in function to be parsered or skipped
    _skip_ms_function = False
    # Functions in black list will not be parsed
    _function_parse_black_list = [F.arange]
    # Share one implementation for the same instances
    _share_one_implementation = False
    # Implementation caches of sub SymbolTrees, CallFunction nodes and CellContainer nodes
    # Keys are ids of the instance object
    _cached_trees: Dict[int, SymbolTree] = {}
    _cached_functions: Dict[int, Node] = {}
    _cached_cell_containers: Dict[int, Node] = {}

    def __init__(self):
        super().__init__()
        self._variables_cache = []
        self.stree: SymbolTree = None
        self.ast_assign: ast.Assign = None
        self.node_manager: NodeManager = None
        self.targets: List[ScopedValue] = None
        self.args: List[ScopedValue] = None
        self.kwargs: Dict[str, ScopedValue] = None

    @staticmethod
    def _get_func_name(ast_call: ast.Call) -> str:
        """
        Get the func name from ast.Call.

        Args:
            ast_call (ast.Call): Input ast.Call node.

        Returns:
            Func name.
        """
        func = ast_call.func
        if isinstance(func, ast.Name):
            return func.id
        if isinstance(func, ast.Attribute):
            return func.attr
        func_full_name = astunparse.unparse(func).strip()
        if func_full_name.count('.') > 0:
            return func_full_name.split('.')[-1]
        return func_full_name

    @staticmethod
    def _get_func_scope(ast_call: ast.Call) -> str:
        """
        Get the func scope from ast.Call.

        Args:
            ast_call (ast.Call): Input ast.Call node.

        Returns:
            Func scope.
        """
        func = ast_call.func
        if isinstance(func, ast.Name):
            return ""
        func_full_name = astunparse.unparse(func).strip()
        if func_full_name.count('.') > 0:
            return func_full_name.rsplit('.', 1)[0]
        return ""

    @staticmethod
    def _create_targets(ast_target: ast.AST) -> List[ScopedValue]:
        """Get targets from ast node."""
        ast_target_elems = AstConverter.get_ast_target_elems(ast_target)
        targets = [AstConverter.create_scopedvalue(ast_node) for ast_node in ast_target_elems]
        return targets

    @staticmethod
    def _create_kwargs(keywords: [ast.keyword]) -> Dict[str, ScopedValue]:
        """
        Transfer ast.Call keywords to a dict of ScopedValue when creating a symbol tree node.

        Args:
            keywords ([ast.keyword]): Keywords of ast.Call node.

        Returns:
            A dict of ScopedValue.
        """
        results = {}
        for keyword in keywords:
            results[keyword.arg] = AstConverter.create_scopedvalue(keyword.value)
        return results


    @staticmethod
    def _get_inst_and_name(ast_node: ast.Attribute, stree: SymbolTree):
        """
        Try to get instance object of ast_node from ast.Attribute.
        """
        if not isinstance(ast_node, ast.Attribute):
            return None, ""
        scope_name = astunparse.unparse(ast_node).strip()
        scope, name = scope_name.split('.', 1)
        if scope != 'self':
            return None, scope_name
        if not hasattr(stree.get_origin_network(), name):
            return None, scope_name
        return getattr(stree.get_origin_network(), name), scope_name

    @staticmethod
    def _list_of_cells(cell_list: list):
        """Check if elements in the list are all cells."""
        for item in cell_list:
            if not isinstance(item, Cell):
                return False
        return True

    @staticmethod
    def _get_path_of_node_manager(node_manager: NodeManager):
        """Get file path of type(instance) in NodeManager"""
        node_manager = node_manager.get_top_manager()
        if isinstance(node_manager, SymbolTree):
            return inspect.getfile(type(node_manager.get_origin_network()))
        return inspect.getfile(node_manager.get_instance())

    @staticmethod
    def _get_module_of_node_manager(node_manager: NodeManager):
        """Get module where the node manager is located"""
        # get module where function object is used
        func_path = AssignParser._get_path_of_node_manager(node_manager)
        func_path = os.path.normcase(os.path.normpath(func_path))
        modules = list(sys.modules.values())
        for m in modules:
            if hasattr(m, "__file__") and m.__file__ is not None and func_path == os.path.normcase(m.__file__):
                return m, func_path
        return None, func_path

    @staticmethod
    def _get_object_from_module(func_full_name: str, module: types.ModuleType):
        """Get object from module according to full name of function"""
        names = func_full_name.split('.')
        obj = module
        for attr in names:
            if not hasattr(obj, attr):
                logger.info(f"For '{func_full_name}', failed to get attr '{attr}' from '{obj}'")
                return None
            obj = getattr(obj, attr)
        return obj

    @staticmethod
    def _get_local_var_provider(node_manager: NodeManager, var: str) -> Node:
        """Get the node providing specific variable"""
        node = node_manager.get_tail()
        while node is not None:
            if var in [str(target) for target in node.get_targets()]:
                return node
            node = node.get_prev()
        # When node_manager is control flow, nodes in upper node_manager need to be traversed.
        if isinstance(node_manager, ControlFlow):
            return AssignParser._get_local_var_provider(node_manager.get_node_manager(), var)
        return None

    def target(self):
        """Parse target type."""
        return ast.Assign

    def store_env(self):
        """Store current environments"""
        self._variables_cache.append(
            [self.stree, self.ast_assign, self.node_manager, self.targets, self.args, self.kwargs])
        self.stree = None
        self.ast_assign = None
        self.node_manager = None
        self.targets = None
        self.args = None
        self.kwargs = None

    def restore_env(self):
        """Restore last environments"""
        self.stree, self.ast_assign, self.node_manager, self.targets, self.args, self.kwargs = \
            self._variables_cache.pop()

    def _get_cell_instance(self, func_scope, func_name):
        """
        Get object instance from ast.Call with type of Cell.

        Args:
            func_scope (str): Func scope.
            func_name (str): Func name.

        Returns:
            An instance represents operator instance.
        """
        if func_scope != "self":
            return None
        var_dict = self.stree.get_origin_network().__dict__
        # Instance is of type Cell
        for key, value in var_dict["_cells"].items():
            if key == func_name:
                return value
        # Instance is of other type.
        return None

    def _get_primitive_instance(self, func_scope, func_name):
        """
        Get object instance from ast.Call with type of Primitive.

        Args:
            func_scope (str): Func scope.
            func_name (str): Func name.

        Returns:
            An instance represents operator instance.
        """
        if func_scope != "self":
            return None
        var_dict = self.stree.get_origin_network().__dict__
        # Instance is of type Primitive
        for key, value in var_dict["_primitives"].items():
            if key == func_name:
                return value
        # Instance is of other type.
        return None

    def _get_method_object(self, func_scope, func_name):
        """Get method object from network instance."""
        stree = self.stree
        if func_scope in ('self', stree.get_opt_cls_name()) and hasattr(stree.get_origin_network(), func_name):
            return getattr(stree.get_origin_network(), func_name)
        return None

    def _get_local_variable(self, func_scope, func_name) -> (bool, object):
        """
        Get local variable

        Args:
            func_scope (str): Func scope.
            func_name (str): Func name.

        Returns:
            bool: Indicate whether local variable is found.
            object (Union[LocalPrim, type]): Instance of LocalPrim when calling the class, or class type
                object when initializing the class.
        """
        func_full_name = f"{func_scope}.{func_name}" if func_scope else func_name
        # try to find func_name in class variables initializing the primitive during forward method
        provider_node = None
        if func_scope == "self":
            for node in self.stree.local_prim_inits():
                if func_full_name in [str(target) for target in node.get_targets()]:
                    provider_node = node
        # try to find func_name in local variables
        if provider_node is None:
            provider_node = AssignParser._get_local_var_provider(self.node_manager, func_full_name)
        if provider_node:
            # when the node providering the local variable initialized a primitive during forward method,
            # we use LocalPrim to indicate the instance of this primitive. e.g. :
            # abs_inst = P.Abs()  -> 'abs_inst' is an instance of primitive initialized locally
            # y = abs_inst(x)     -> here we are parsing now
            cls_init = provider_node.get_init_cls()
            if cls_init and inspect.isclass(cls_init) and issubclass(cls_init, Primitive):
                return True, LocalPrim(cls_init)
            # when the node providering the local variable represent a primitive type object, we return
            # type-object to indicate that we are initializing this primitive. e.g. :
            # abs_ops = _get_cache_prim(P.Abs)  -> 'abs_ops' is an primitive type object
            # y = abs_ops(x)                    -> here we are parsing now
            cls_type = provider_node.get_type_cls()
            if cls_type and inspect.isclass(cls_type) and issubclass(cls_type, Primitive):
                return True, cls_type
            # local variable whose type is not primitive instance
            logger.info(f"Ignore local variable: {func_full_name}")
            return True, None
        # other local variable
        if AssignParser._get_local_var_provider(self.node_manager, func_full_name.split('.')[0]):
            logger.info(f"Ignore local variable: {func_full_name}")
            return True, None
        return False, None

    def _get_function_object(self, func_scope, func_name, ast_call) -> (object, bool):
        """
        Get function object from module.

        If the code represent a class type object, e.g. abs_ops = _get_cache_prim(P.Abs),
        return primitive type object with class type flag True.

        if the code represent an initializtion of a class, e.g. abs_inst = P.Abs(),
        return primitive type object with class type flag False.

        if the code represent the call of function or class instance, e.g. y = abs_inst(x)/func(x),
        return primitive instance or function object with class type flag False.

        Args:
            func_scope (str): Func scope.
            func_name (str): Func name.
            ast_call (ast.Call): ast.Call of ast.Assign.

        Returns:
            object: Class type object, class instance or function object
            bool: Flag indicate is node represent a class type object.
        """
        func_full_name = f"{func_scope}.{func_name}" if func_scope else func_name
        # get module where function object is used
        module, func_path = AssignParser._get_module_of_node_manager(self.node_manager)
        if module is None:
            logger.debug(f"When getting object of '{func_full_name}', failed to find module in '{func_path}'")
            return None, False
        # if name of function is _get_cache_prim, return primitive type object
        is_cls_type_obj = False
        if func_full_name == '_get_cache_prim':
            func_full_name = astunparse.unparse(ast_call.args[0]).strip()
            is_cls_type_obj = True
        # find object in module
        obj = AssignParser._get_object_from_module(func_full_name, module)
        return obj, is_cls_type_obj

    def _update_field_in_init(self, func_name: str, sub_tree: SymbolTree) -> bool:
        """
        When node is an invoking to sub-network, update value of ast.Assign of corresponding field in `__init__` method.
        Add the code like: `self.field = SubNetwork(self.field)`

        Args:
            func_name (str): A string represents scope and name of function symbol.
            sub_tree (SymbolTree): The SymbolTree corresponding to sub-network.
        """
        init_func_ast = self.stree.get_init_func_ast()
        sub_net_obj = sub_tree.get_origin_network()
        sub_net_opt_name = sub_tree.get_opt_cls_name()
        # Add .to_float(mindspore.float16) if origin subnet has this attribute
        new_code = f"{func_name} = {sub_net_opt_name}({func_name})"
        if hasattr(sub_net_obj, "fp16") and sub_net_obj.fp16:
            new_code = f"{new_code}.to_float(mindspore.float16)"
        elif hasattr(sub_net_obj, "bf16") and sub_net_obj.bf16:
            new_code = f"{new_code}.to_float(mindspore.bfloat16)"
        new_ast = ast.parse(new_code).body[0]
        init_func_ast.body.append(new_ast)

    def _update_cell_container_in_init(self, container_name, container_idx, subnet_opt_name):
        """
        When nn.SequentialCell include sub-symboltree, the new class definition will be used to create object.
        So the assign code will be got from origin code first, and then be modified to new class name.

        Codes like:

        `self.container = nn.SequentialCell([ReLU(), MyNet()])`

        will be updated by add codes:

        `self.container[1] = MyNetOpt(self.container[1])`

        """
        new_code = f"{container_name}[{container_idx}] = {subnet_opt_name}({container_name}[{container_idx}])"
        new_ast = ast.parse(new_code).body[0]
        self.stree.get_init_func_ast().body.append(new_ast)

    def _add_import(self, import_name: str):
        """ add import to current node manager."""
        module, _ = AssignParser._get_module_of_node_manager(self.node_manager)
        if module is None:
            logger.info(f"Cannot get module where '{import_name}' is located, ignore import info")
            return
        node_manager = self.node_manager.get_top_manager()
        belonging_ast = None if isinstance(node_manager, SymbolTree) else node_manager.get_manager_ast()
        self.stree.add_import(module, import_name, belonging_ast)

    def cell_container_process(self, func_name: str, node_name: str, container_obj: object):
        """ parse cell container object."""
        # create unparsable node if container is already parsed when sharing one implementation
        if AssignParser._share_one_implementation and id(container_obj) in AssignParser._cached_cell_containers:
            cell_container = Node.create_call_buildin_op(container_obj, self.ast_assign, self.targets,
                                                         func_name, self.args, self.kwargs, node_name)
            return cell_container
        cell_container = CellContainer(self.ast_assign, self.targets, func_name, self.args, self.kwargs,
                                       node_name, self.stree, container_obj)
        for i, cell in enumerate(container_obj):
            cell_name = type(cell).__name__
            # The type of cell is container of cells (e.g. SequentialCell)
            if isinstance(cell, tuple(AssignParser.types_for_cell_container)):
                sub_node = self.cell_container_process(f"{func_name}[{i}]", cell_name, cell)
            elif is_subtree(cell):
                # create unparsable node if tree node is already parsed when sharing one implementation
                if AssignParser._share_one_implementation and id(cell) in AssignParser._cached_trees:
                    first_stree = AssignParser._cached_trees.get(id(cell))
                    self._update_cell_container_in_init(func_name, i, first_stree.get_opt_cls_name())
                    sub_node = Node.create_call_buildin_op(cell, None, self.targets, cell_name, self.args,
                                                           self.kwargs, cell_name)
                else:
                    from ..symbol_tree import SymbolTreeBuilder
                    stb = SymbolTreeBuilder(cell)
                    new_stree = stb.build()
                    sub_node = TreeNode.create_tree_node(new_stree, None, self.targets, cell_name, self.args,
                                                         self.kwargs, cell_name, cell)
                    self._update_cell_container_in_init(func_name, i, new_stree.get_opt_cls_name())
                    # save symbol tree if it is firstly parsed when sharing one implementation
                    if AssignParser._share_one_implementation:
                        AssignParser._cached_trees[id(cell)] = new_stree
            else:
                sub_node = Node.create_call_buildin_op(cell, None, self.targets, cell_name, self.args,
                                                       self.kwargs, cell_name)
            # add sub node to cell_container
            cell_container.append(sub_node, False)
        # save the node if container is firstly parsed when sharing one implementation
        if AssignParser._share_one_implementation:
            AssignParser._cached_cell_containers[id(container_obj)] = cell_container
        return cell_container

    def process_cell(self, func_scope_name: ScopedValue, node_name: str, cell_inst: Cell):
        """Create CallCell node with instance of cell."""
        # The type of cell is container of cells (e.g. SequentialCell)
        if isinstance(cell_inst, tuple(AssignParser.types_for_cell_container)):
            node = self.cell_container_process(func_scope_name, node_name, cell_inst)
        # The type of cell is user custom network, then we create sub-symboltree
        elif is_subtree(cell_inst):
            # create unparsable node if tree node is already parsed when sharing one implementation
            if AssignParser._share_one_implementation and id(cell_inst) in AssignParser._cached_trees:
                first_stree = AssignParser._cached_trees.get(id(cell_inst))
                self._update_field_in_init(str(func_scope_name), first_stree)
                node = Node.create_call_buildin_op(cell_inst, self.ast_assign, self.targets, func_scope_name,
                                                   self.args, self.kwargs, node_name)
            else:
                from ..symbol_tree import SymbolTreeBuilder
                stb = SymbolTreeBuilder(cell_inst)
                new_stree = stb.build()
                self._update_field_in_init(str(func_scope_name), new_stree)
                node = TreeNode.create_tree_node(new_stree, self.ast_assign, self.targets, func_scope_name,
                                                 self.args, self.kwargs, node_name, new_stree.get_origin_network())
                # save symbol tree if it is firstly parsed when sharing one implementation
                if AssignParser._share_one_implementation:
                    AssignParser._cached_trees[id(cell_inst)] = new_stree
        else:
            # The type of cell is built-in cells
            node = Node.create_call_buildin_op(cell_inst, self.ast_assign, self.targets, func_scope_name, self.args,
                                               self.kwargs, node_name)
        self.stree.append_origin_field(node, self.node_manager)

    def process_primitive(self, func_scope_name: ScopedValue, node_name: str, primitive_inst: Primitive):
        """Create CallPrimitive node with instance of primitive."""
        node = Node.create_call_buildin_op(primitive_inst, self.ast_assign, self.targets, func_scope_name,
                                           self.args, self.kwargs, node_name)
        self.stree.append_origin_field(node, self.node_manager)

    def process_class_method(self, func_scope_name: ScopedValue, node_name: str, method_object: object):
        """Create CallFunction node for class method function."""
        func_name = func_scope_name.value
        # get ast.FunctionDef
        ast_functiondef = None
        for body in self.stree.get_class_ast().body:
            if isinstance(body, ast.FunctionDef) and func_name == body.name:
                ast_functiondef = body
        if ast_functiondef is None:
            # method of child class may be called and will be ignored now.
            logger.info(error_str(f"Find ast of function '{func_name}' in network '{self.stree.get_ori_cls_name()}' "
                                  f"failed", child_node=self.ast_assign))
            self.insert_callfunction_node(func_scope_name, node_name, None, None, False)
        else:
            # create CallFunction node
            self.insert_callfunction_node(func_scope_name, node_name, ast_functiondef, method_object, True)

    def process_function(self, func_scope_name: ScopedValue, node_name: str, function_object: object,
                         is_cls_type_obj: bool):
        """Create node for function."""
        # Ignore functions in _function_parse_black_list
        if function_object in AssignParser._function_parse_black_list:
            logger.debug(f"'{func_scope_name}' is in the _function_parse_black_list and will not be parsed")
            if not func_scope_name.scope:
                self._add_import(func_scope_name.value)
            self.insert_callfunction_node(func_scope_name, node_name, None, function_object, False)
            return
        # break loop function
        node_manager = self.node_manager
        while node_manager and isinstance(node_manager, Node):
            if isinstance(node_manager, CallFunction) and node_manager.get_instance() == function_object:
                logger.info(f"loop function detected in '{func_scope_name}', stop parsing function.")
                self.insert_callfunction_node(func_scope_name, node_name, None, function_object, False)
                return
            node_manager = node_manager.get_node_manager()
        # process primitive instances:
        # (global/local) _ops_func = P.FUNC()
        # (here) y = _ops_func(x) <- (process: _ops_func)
        if isinstance(function_object, Primitive):
            # when primitive instance is not a local variable, it will be a global object which need to be imported
            if not isinstance(function_object, LocalPrim):
                import_name = str(func_scope_name).split('.')[0]
                self._add_import(import_name)
            # create CallPrimitive node
            self.process_primitive(func_scope_name, func_scope_name.value, function_object)
            return
        # process primitive object:
        # (here) _ops_func = P.FUNC() <- (process: P.FUNC)
        # (later) y = _ops_func(x)
        if inspect.isclass(function_object):
            node = self.insert_callfunction_node(func_scope_name, node_name, None, None, False)
            if is_cls_type_obj:
                # represent a class type object, e.g. abs_ops = _get_cache_prim(P.Abs)
                node.set_type_cls(function_object)
                # add import
                if str(func_scope_name) == '_get_cache_prim':
                    import_name = astunparse.unparse(self.ast_assign.value.args[0]).strip()
                    if '.' not in import_name:
                        self._add_import(import_name)
            else:
                # represent the initialize of a class type, e.g. abs_inst = P.Abs()
                node.set_init_cls(function_object)
                # record local primitive objects
                if func_scope_name.scope == 'self' and issubclass(function_object, Primitive):
                    self.stree.local_prim_inits.append(node)
            return
        # process third party functions
        is_ms_func = is_ms_function(function_object)
        if not is_ms_func and is_third_party(function_object):
            logger.info(f"Ignore third party function '{func_scope_name}'.")
            self.insert_callfunction_node(func_scope_name, node_name, None, function_object, False)
            return
        # process mindspore functions
        if is_ms_func and AssignParser._skip_ms_function:
            logger.info(f"Ignore mindspore function '{func_scope_name}'.")
            self.insert_callfunction_node(func_scope_name, node_name, None, function_object, False)
            return
        # get ast.FunctionDef
        source_code = inspect.getsource(function_object)
        ast_functiondef = ast.parse(dedent(source_code)).body[0]
        if not isinstance(ast_functiondef, ast.FunctionDef):
            logger.info(error_str(f"Get ast.FunctionDef of function {str(func_scope_name)} failed, the type of "
                                  f"ast node is {type(ast_functiondef)}", child_node=self.ast_assign))
            self.insert_callfunction_node(func_scope_name, node_name, None, function_object, False)
            return
        if [n for n in ast_functiondef.body if isinstance(n, ast.FunctionDef)]:
            logger.info(error_str(f"closure syntax is not supported now, {str(func_scope_name)} will not be parsed.",
                                  child_node=ast_functiondef))
            if not func_scope_name.scope:
                self._add_import(func_scope_name.value)
            self.insert_callfunction_node(func_scope_name, node_name, None, function_object, False)
            return
        # update func_name, and remove scope
        new_name = ast_functiondef.name
        # when func_scope_name(e.g. 'C.uniform') is not the name in ast.FunctionDef(e.g. 'uniform'), this name may be
        # already used as variable(e.g. uniform = C.uniform(x)).
        # To avoid new function's name being duplicated with existed variable, an suffix '_opt' will be added.
        if new_name != str(func_scope_name):
            new_name = f"{new_name}_opt"
        new_name = FunctionNamer().instance().get_name(new_name)
        # create unparsable node if function is already parsed when sharing one implementation
        if AssignParser._share_one_implementation and id(function_object) in AssignParser._cached_functions:
            first_node = AssignParser._cached_functions.get(id(function_object))
            ast_call: ast.Call = self.ast_assign.value
            ast_call.func = ast.Name(id=str(first_node.get_func_name()), ctx=ast.Load())
            self.insert_callfunction_node(func_scope_name, new_name, None, function_object, False)
            return
        ast_functiondef.name = new_name
        ast_call: ast.Call = self.ast_assign.value
        ast_call.func = ast.Name(id=new_name, ctx=ast.Load())
        # save ast.FunctionDef into stree._external_ast
        self.stree.get_external_ast()[ast_functiondef] = []
        # import module which function defined in
        func_file_path = inspect.getabsfile(function_object)
        self.stree.save_imports_from_file(func_file_path, ast_functiondef)
        # create CallFunction node
        func_scope_name = ScopedValue.create_naming_value(new_name, "")
        node = self.insert_callfunction_node(func_scope_name, new_name, ast_functiondef, function_object, False)
        # save function node if it is firstly parsed when sharing one implementation
        if AssignParser._share_one_implementation:
            AssignParser._cached_functions[id(function_object)] = node

    def insert_callfunction_node(self, func_name: ScopedValue, node_name: str, ast_functiondef: ast.FunctionDef,
                                 func_obj: object, is_method: bool) -> Node:
        """Create CallFunction node for function."""
        if ast_functiondef is None:
            node = Node.inner_create_call_function(node_name, self.ast_assign, func_name, func_obj,
                                                   self.targets, self.args, self.kwargs)
            self.stree.append_origin_field(node, self.node_manager)
            return node
        # create CallFunction node
        node = CallFunction(self.targets, func_name, self.args, self.kwargs, node_name, self.ast_assign,
                            ast_functiondef, self.stree, func_obj, is_method)
        self.stree.append_origin_field(node, self.node_manager)
        # expand ast codes
        ast_functiondef = AstFlattener().transform(ast_functiondef, [func_name.value], self.stree)
        # parse ast codes into CallFunction Node
        parser = ParserRegister.instance().get_parser(ast.FunctionDef)
        parser.process(self.stree, ast_functiondef, node_manager=node)
        return node

    def process_ast_call(self, ast_call: ast.Call):
        """
        Convert ast.Call to a symbol tree node.

        Args:
            ast_call (ast.Call): An ast.Call of assign node in construct.
        """
        self.targets = AssignParser._create_targets(self.ast_assign.targets[0])
        self.args = [AstConverter.create_scopedvalue(arg) for arg in ast_call.args]
        self.kwargs = AssignParser._create_kwargs(ast_call.keywords)
        func_name = AssignParser._get_func_name(ast_call)
        func_scope = AssignParser._get_func_scope(ast_call)
        func_scope_name = ScopedValue.create_naming_value(func_name, func_scope)
        func_full_name = str(func_scope_name)
        # y = func(xxx)(xxx) / y = func1(xxx).func2(xxx) is not supported, and should be flattened before parsing.
        if AstFinder(ast_call.func).find_all(ast.Call):
            logger.info(error_str("ast.Call in func name of ast.Call is not supported.", ast_call, self.ast_assign))
            self.insert_callfunction_node(func_scope_name, func_name, None, None, False)
            return
        # Ignore built-in functions
        if func_full_name in dir(builtins):
            logger.info(f"Ignore built-in function: {func_scope_name}")
            self.insert_callfunction_node(func_scope_name, func_name, None, None, False)
            return
        # Ignore function name is target of for loop
        if isinstance(self.node_manager, ControlFlow) and func_full_name in self.node_manager.loop_vars:
            logger.info(f"Ignore function of loop variable: {func_scope_name}")
            self.insert_callfunction_node(func_scope_name, func_name, None, None, False)
            return
        # Instance with type of Cell
        cell_inst = self._get_cell_instance(func_scope, func_name)
        if cell_inst is not None:
            self.process_cell(func_scope_name, func_name, cell_inst)
            return
        # Instance with type of Primitive
        primitive_inst = self._get_primitive_instance(func_scope, func_name)
        if primitive_inst is not None:
            self.process_primitive(func_scope_name, func_name, primitive_inst)
            return
        # Class method object
        method_object = self._get_method_object(func_scope, func_name)
        if method_object is not None:
            if inspect.ismethod(method_object):
                self.process_class_method(func_scope_name, func_name, method_object)
            elif isinstance(inspect.getattr_static(self.stree.get_origin_network(), func_name), staticmethod):
                self.insert_callfunction_node(func_scope_name, func_name, None, None, False)
            else:
                self.process_function(func_scope_name, func_name, method_object, False)
            return
        # Local variable
        is_local_var, primitive_obj = self._get_local_variable(func_scope, func_name)
        if primitive_obj is not None:
            self.process_function(func_scope_name, func_name, primitive_obj, False)
            return
        if is_local_var:
            # for a variable whose type is not primitive instance, create normal node for it
            self.insert_callfunction_node(func_scope_name, func_name, None, None, False)
            return
        # Function object
        function_object, is_cls_type_obj = self._get_function_object(func_scope, func_name, ast_call)
        if function_object is not None:
            self.process_function(func_scope_name, func_name, function_object, is_cls_type_obj)
            return
        logger.info(error_str("Failed to get instance or object of ast.Call.", ast_call, self.ast_assign))
        self.insert_callfunction_node(func_scope_name, func_name, None, None, False)

    def process_ast_mathops(self, ast_op: Union[ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare]):
        """
        Convert ast node of math operations(ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare) to
        a symbol tree node.

        Args:
            ast_op (Union[ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare]): An assign node with mathematival
                operation in construct function.

        Raises:
            TypeError: The type of parameter 'ast_op' is not in (ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare).

        """
        if not isinstance(ast_op, (ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare)):
            raise TypeError("The type of parameter 'ast_op' must be one of (ast.BinOp, ast.UnaryOp, "
                            "ast.BoolOp, ast.Compare), but got ", type(ast_op))

        targets = AssignParser._create_targets(self.ast_assign.targets[0])
        args = []
        op_type_str = type(ast_op).__name__
        op_type = ScopedValue.create_naming_value(op_type_str)
        name = op_type_str
        if isinstance(ast_op, ast.BinOp):
            op = type(ast_op.op).__name__
            name = f'{name}_{op}'
            args.append(AstConverter.create_scopedvalue(ast_op.left))
            args.append(AstConverter.create_scopedvalue(ast_op.right))
        elif isinstance(ast_op, ast.UnaryOp):
            op = type(ast_op.op).__name__
            name = f'{name}_{op}'
            args.append(AstConverter.create_scopedvalue(ast_op.operand))
        elif isinstance(ast_op, ast.BoolOp):
            op = type(ast_op.op).__name__
            name = f'{name}_{op}'
            for value in ast_op.values:
                args.append(AstConverter.create_scopedvalue(value))
        elif isinstance(ast_op, ast.Compare):
            args.append(AstConverter.create_scopedvalue(ast_op.left))
            for idx, ast_cmp_op in enumerate(ast_op.ops):
                op = type(ast_cmp_op).__name__
                name = f'{name}_{op}'
                args.append(AstConverter.create_scopedvalue(ast_op.comparators[idx]))
        name = name.lower()
        node = Node.create_mathops_node(self.ast_assign, targets, op_type, args, name)
        self.stree.append_origin_field(node, self.node_manager)

    def process_ast_constant(self, ast_constant: Union[ast.Constant, ast.NameConstant, ast.Num, ast.Bytes, ast.Str]):
        """
        Convert ast node of constant types (ast.Constant, ast.NameConstant, ast.Num, ast.Bytes, ast.Str) to
        a symbol tree node.
        """
        node_name = f"{type(ast_constant).__name__.lower()}_assign"
        targets = AssignParser._create_targets(self.ast_assign.targets[0])
        args = [AstConverter.create_scopedvalue(ast_constant)]
        node = Node.create_call_method(self.ast_assign, targets, "pass_through", args, {}, node_name)
        self.stree.append_origin_field(node, self.node_manager)

    def process_ast_name(self, ast_node: Union[ast.Name, ast.Attribute]):
        """
        Convert ast node of ast.Name and ast.Attribute to a symbol tree node.
        """
        self.targets = AssignParser._create_targets(self.ast_assign.targets[0])
        inst, scope_name = AssignParser._get_inst_and_name(ast_node, self.stree)
        if inst is not None and (isinstance(inst, CellList) or
                                 isinstance(inst, list) and AssignParser._list_of_cells(inst)):
            node = self.cell_container_process(scope_name, scope_name, inst)
        else:
            node_name = f"{type(ast_node).__name__.lower()}_assign"
            args = [AstConverter.create_scopedvalue(ast_node)]
            node = Node.create_call_method(self.ast_assign, self.targets, "pass_through", args, {}, node_name)
        self.stree.append_origin_field(node, self.node_manager)

    def process_ast_tuple(self, ast_node: Union[ast.Tuple, ast.List]):
        """
        Convert ast node of ast.Tuple or ast.List to a symbol tree node.
        """
        # ensure that each element's type in tuple is supported by scopled value
        if AstConverter.ast_tuple_elts_support_scopledvalue(ast_node):
            targets = AssignParser._create_targets(self.ast_assign.targets[0])
            args = []
            for elt in ast_node.elts:
                args.append(AstConverter.create_scopedvalue(elt))
            func_name = "tuple" if isinstance(ast_node, ast.Tuple) else "list"
            node = Node.create_call_method(self.ast_assign, targets, func_name, args, {}, func_name)
            self.stree.append_origin_field(node, self.node_manager)
        else:
            logger.info(f"some elements in assign({astunparse.unparse(self.ast_assign)}) are not supported "
                        "in rewrite, fallback to python")
            self.stree.try_append_python_node(self.ast_assign, self.ast_assign, self.node_manager)

    def process_ast_dict(self, ast_dict: ast.Dict):
        """
        Convert ast node of ast.Dict to a symbol tree node.
        """
        # ensure that each element's type in dict is supported by scopled value
        if AstConverter.ast_dict_support_scopledvalue(ast_dict):
            targets = AssignParser._create_targets(self.ast_assign.targets[0])
            kwargs = {}
            for idx, key in enumerate(ast_dict.keys):
                kwargs[key.value] = AstConverter.create_scopedvalue(ast_dict.values[idx])
            func_name = ScopedValue.create_naming_value("dict")
            node = Node.create_call_method(self.ast_assign, targets, func_name, [], kwargs, "dict")
            self.stree.append_origin_field(node, self.node_manager)
        else:
            logger.info(f"some elements in assign({astunparse.unparse(self.ast_assign)}) are not supported "
                        "in rewrite, fallback to python")
            self.stree.try_append_python_node(self.ast_assign, self.ast_assign, self.node_manager)

    def process_ast_subscript(self, ast_subscript: ast.Subscript):
        """
        Convert ast node of ast.Subscript to a symbol tree node.
        """
        targets = AssignParser._create_targets(self.ast_assign.targets[0])
        args = [AstConverter.create_scopedvalue(ast_subscript)]
        node = Node.create_call_method(self.ast_assign, targets, "pass_through", args, {}, "subscript_var")
        self.stree.append_origin_field(node, self.node_manager)

    def process(self, stree: SymbolTree, node: ast.Assign, node_manager: NodeManager):
        """
        Parse ast.Assign and create a node in symbol tree.

        - Create node when value of ast.Assign is in [ast.Call, ast.Name, ast.Constant, ast.Attribute].
        - Create python node when value of ast.Assign is in [ast.BinOp, ast.BoolOp, ast.Subscript, ast.List, ast.Tuple,
          ast.Dict].
        - Other value types are not supported.

        Args:
            stree ([SymbolTree]): Symbol Tree under parsing.
            node ([ast.Assign]): An ast.Assign node.
            node_manager (NodeManager): NodeManager those asts belong to.
        """
        if len(node.targets) != 1:
            logger.info(error_str(f"Continuous assignment statement(e.g. 'a = b = 1') should be flatten before.",
                                  child_node=node))
            stree.try_append_python_node(node, node, node_manager)
            return

        self.store_env()
        self.stree = stree
        self.ast_assign = node
        self.node_manager = node_manager
        value = node.value
        if isinstance(value, ast.Call):
            self.process_ast_call(value)
        elif isinstance(value, (ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare)):
            self.process_ast_mathops(value)
        elif isinstance(value, ast.Subscript):
            self.process_ast_subscript(value)
        elif isinstance(value, (ast.Constant, ast.NameConstant, ast.Num, ast.Bytes, ast.Str)):
            self.process_ast_constant(value)
        elif isinstance(value, (ast.Name, ast.Attribute)):
            self.process_ast_name(value)
        elif isinstance(value, (ast.Tuple, ast.List)):
            self.process_ast_tuple(value)
        elif isinstance(value, ast.Dict):
            self.process_ast_dict(value)
        else:
            logger.info(f"ops-call({astunparse.unparse(node).strip()}) in assign will be supported in near feature, "
                        f"ignored as a python node now")
            stree.try_append_python_node(node, node, node_manager)
        self.restore_env()


g_assign_parser = reg_parser(AssignParser())
