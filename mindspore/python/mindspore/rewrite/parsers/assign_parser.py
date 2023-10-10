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
from typing import Union
import os
import ast
import sys
import inspect

from mindspore import log as logger
from mindspore.nn import Cell, SequentialCell
from mindspore.ops import Primitive
from mindspore.rewrite.parsers.parser_register import ParserRegister, reg_parser
from mindspore.rewrite.namespace import is_subtree, is_functional, get_functional
from mindspore.rewrite.symbol_tree import SymbolTree
from mindspore.rewrite.node.node import Node, TreeNode
from mindspore.rewrite.node.node_manager import NodeManager
from mindspore.rewrite.node.call_function import CallFunction
from mindspore.rewrite.node.cell_container import CellContainer
from mindspore.rewrite.parsers.parser import Parser
from mindspore.rewrite.api.scoped_value import ScopedValue, ValueType
from mindspore.rewrite.symbol_tree_builder import SymbolTreeBuilder
from mindspore.rewrite.ast_transformers.flatten_recursive_stmt import FlattenRecursiveStmt
from mindspore.rewrite.ast_helpers import AstReplacer
from ..common import error_str

if sys.version_info >= (3, 9):
    import ast as astunparse # pylint: disable=reimported, ungrouped-imports
else:
    import astunparse


class AssignParser(Parser):
    """Parse ast.Assign in construct function to node of SymbolTree."""

    # Types for creating Cell Container node
    types_for_cell_container = [SequentialCell,]

    def target(self):
        """Parse target type."""
        return ast.Assign

    @staticmethod
    def _create_scopedvalue_from_tuple_ast(node: ast.Tuple) -> ScopedValue:
        """
        Create ScopedValue from a tuple ast node.

        Args:
            node (ast.Tuple): A tuple node.

        Returns:
            An instance of ScopedValue.

        Raises:
            RuntimeError: Only support ast.Constant as elts of ast.Tuple.
        """
        tuple_elts = node.elts
        tuple_values = []
        for tuple_elt in tuple_elts:
            if not isinstance(tuple_elt, (ast.Constant, ast.Name, ast.Attribute)):
                raise RuntimeError(error_str(f"Only support ast.Constant or ast.Name as elts of ast.Tuple, "
                                             f"but got ast type {type(tuple_elt).__name__}",
                                             child_node=tuple_elt, father_node=node))
            if isinstance(tuple_elt, ast.Constant):
                tuple_values.append(tuple_elt.value)
            elif isinstance(tuple_elt, ast.Name):
                tuple_values.append(tuple_elt.id)
            elif isinstance(tuple_elt, ast.Attribute):
                tuple_values.append("".join([tuple_elt.value.id, '.', tuple_elt.attr]))
        return ScopedValue.create_variable_value(tuple(tuple_values))

    @staticmethod
    def _create_scopedvalue(node: ast.expr) -> ScopedValue:
        """
        Create ScopedValue from an ast node.

        Args:
            node (ast.expr): An ast node.

        Returns:
            An instance of ScopedValue.

        Raises:
            RuntimeError: Value of target of ast.Assign should be an ast.Name when target is an ast.Attribute.
            RuntimeError: Type of input node is unsupported.
        """
        if isinstance(node, ast.Name):
            return ScopedValue.create_naming_value(node.id)
        if isinstance(node, ast.Attribute):
            scope = node.value
            if not isinstance(scope, ast.Name):
                raise RuntimeError(error_str(f"value of target of ast.Assign should be a ast.Name when target is a "
                                             f"ast.Attribute, but got ast type '{type(scope).__name__}'",
                                             child_node=scope, father_node=node))
            return ScopedValue.create_naming_value(node.attr, scope.id)
        if isinstance(node, ast.Tuple):
            return AssignParser._create_scopedvalue_from_tuple_ast(node)
        if isinstance(node, (ast.Constant, ast.NameConstant)):
            return ScopedValue.create_variable_value(node.value)
        if isinstance(node, ast.Num):
            return ScopedValue.create_variable_value(node.n)
        if isinstance(node, (ast.Str, ast.Bytes)):
            return ScopedValue.create_variable_value(node.s)
        raise RuntimeError(error_str(f"only support (ast.Name, ast.Attribute, ast.Tuple, ast.Constant, ast.Num"
                                     f"ast.Str, ast.Bytes to argument), but got ast type '{type(node).__name__}'",
                                     father_node=node))

    @staticmethod
    def _get_func_name(ast_call: ast.Call) -> str:
        """
        Get the func name from ast.Call.

        Args:
            ast_call (ast.Call): Input ast.Call node.

        Returns:
            Func name.

        Raises:
            RuntimeError: Func of input ast node is not ast.Name or ast.Attribute.
        """
        func = ast_call.func
        if isinstance(func, ast.Name):
            return func.id
        if isinstance(func, ast.Attribute):
            return func.attr
        if isinstance(func, ast.Call):
            return AssignParser._get_func_name(func)
        raise RuntimeError(error_str(f"funcValue should be Name or a Attribute or a Call, but got ast type "
                                     f"'{type(func).__name__}'", child_node=func, father_node=ast_call))

    @staticmethod
    def _get_func_scope(ast_call: ast.Call, node_manager: NodeManager = None) -> str:
        """
        Get the func scope from ast.Call.

        Args:
            ast_call (ast.Call): Input ast.Call node.
            node_manager (NodeManager): NodeManager those asts belong to.

        Returns:
            Func scope.

        Raises:
            RuntimeError: FuncValue is not an ast.Name when func is an ast.Attribute.
            RuntimeError: Func of input ast node is not ast.Name or ast.Attribute.
        """
        func = ast_call.func
        if isinstance(func, ast.Name):
            return ""
        if isinstance(func, ast.Attribute):
            parser = ParserRegister.instance().get_parser(type(func))
            value = parser.process(None, func, node_manager)
            return value.rsplit(".", 1)[0]
        if isinstance(func, ast.Call):
            return AssignParser._get_func_scope(func, node_manager)
        raise RuntimeError(error_str(f"funcValue should be Name or a Attribute or a Call, but got ast type "
                                     f"'{type(func).__name__}'", child_node=func, father_node=ast_call))

    @staticmethod
    def _get_symbol_object(symbol_name, origin_net):
        """
        Get the func scope from ast.Call.

        Args:
            symbol_name (str): Func name.
            origin_net ([nn.Cell]): Network instance.

        Returns:
            Symbol Object.
        """
        var_dict = origin_net.__dict__
        for key, value in var_dict["_cells"].items():
            if key == symbol_name:
                return value

        for key, value in var_dict["_primitives"].items():
            if key == symbol_name:
                return value
        return None

    @staticmethod
    def _create_kwargs(keywords: [ast.keyword]) -> {str, ScopedValue}:
        """
        Transfer ast.Call keywords to a dict of ScopedValue when creating a symbol tree node.

        Args:
            keywords ([ast.keyword]): Keywords of ast.Call node.

        Returns:
            A dict of ScopedValue.
        """
        results = {}
        for keyword in keywords:
            results[keyword.arg] = AssignParser._create_scopedvalue(keyword.value)
        return results

    @staticmethod
    def _get_call_instance(func_scope, func_name, stree: SymbolTree):
        """
        Get object instance from ast.Call with type of Cell or Primitive.

        Args:
            func_scope (str): Func scope.
            func_name (str): Func name.
            stree (SymbolTree): Belong SymbolTree.

        Returns:
            An instance represents operator instance.
        """
        if func_scope != "self":
            return None
        var_dict = stree.get_origin_network().__dict__
        # Instance is of type Cell
        for key, value in var_dict["_cells"].items():
            if key == func_name:
                return value
        # Instance is of type Primitive
        for key, value in var_dict["_primitives"].items():
            if key == func_name:
                return value
        # Instance is of other type.
        return None

    @staticmethod
    def _get_targets(all_targets: ScopedValue) -> [Union[ScopedValue, str]]:
        """Get targets from tuple or single value."""
        targets: [Union[ScopedValue, str]] = []
        if all_targets.type == ValueType.TupleValue:
            for single_target in all_targets.value:
                if not isinstance(single_target, ScopedValue) and not isinstance(single_target.value, str):
                    raise RuntimeError(f"For MindSpore Rewrite, only support str target in tuple, but got type "
                                       f"{type(single_target).__name__}")
                if single_target.type == ValueType.ConstantValue and isinstance(single_target.value, str):
                    single_target.type = ValueType.NamingValue
                targets.append(single_target)
        else:
            targets.append(all_targets)
        return targets

    @staticmethod
    def _update_field_in_init(func_scope, func_name, stree: SymbolTree, sub_tree: SymbolTree) -> bool:
        """
        When node is an invoking to sub-network, update value of ast.Assign of corresponding field in `__init__` method.
        Add the code like: `self.field = SubNetwork(self.field)`

        Args:
            func_scope (str): A string represents scope of function symbol.
            func_name (str): A string represents function symbol.
            stree (SymbolTree): The SymbolTree corresponding to main-network.
            sub_tree (SymbolTree): The SymbolTree corresponding to sub-network.

        Raises:
            NotImplementedError: If `func_scope` is not "self", it means corresponding op is inited in forward method.
            NotImplementedError: If targets of ast.Assign of corresponding field in `__init__` method.
        """
        if func_scope != "self":
            logger.warning("Not support parse operator which is instantiated at runtime now: %s; name: %s", func_scope,
                           func_name)
        init_func_ast = stree.get_init_func_ast()
        sub_net_obj = sub_tree.get_origin_network()
        sub_net_opt_name = sub_tree.get_opt_cls_name()
        # Add .to_float(mindspore.float16) if origin subnet has this attribute
        new_code = f"{func_scope}.{func_name} = {sub_net_opt_name}({func_scope}.{func_name})"
        if hasattr(sub_net_obj, "fp16") and sub_net_obj.fp16:
            new_code = f"{new_code}.to_float(mindspore.float16)"
        elif hasattr(sub_net_obj, "bf16") and sub_net_obj.bf16:
            new_code = f"{new_code}.to_float(mindspore.bfloat16)"
        new_ast = ast.parse(new_code).body[0]
        init_func_ast.body.append(new_ast)

    @staticmethod
    def _create_inputs_for_cell_container(ast_assign) -> ['Node']:
        """Create inputs for cell container first node."""
        call_ast_node = ast_assign.value
        if not isinstance(call_ast_node, ast.Call):
            raise RuntimeError(error_str(f"when creating input node for cellcontainer, value of input father ast node"
                                         "is not ast.Call!'", child_node=call_ast_node, father_node=ast_assign))
        first_node_inputs: ['Node'] = []
        exist_param_name = []
        for arg in call_ast_node.args:
            if isinstance(arg, ast.Name):
                param_name = arg.id
            elif isinstance(arg, ast.arg):
                param_name = arg.arg
            else:
                raise RuntimeError(error_str(f"only support ast.arg, ast.arg in arguments arg, but got "
                                             f"'{type(arg).__name__}'", child_node=arg, father_node=call_ast_node))
            if param_name in exist_param_name:
                raise RuntimeError(error_str(f"Cellcontianer has duplicate input names", child_node=arg,
                                             father_node=call_ast_node))
            exist_param_name.append(param_name)
            node = Node.create_input_node(arg, param_name, name=f"input_{param_name}")
            first_node_inputs.append(node)

        if call_ast_node.keywords:
            raise RuntimeError(error_str(f"Not support keyword input for cellcontainer now.",
                                         child_node=call_ast_node, father_node=ast_assign))

        return first_node_inputs

    @staticmethod
    def _update_cell_container_in_init(stree, container_name, container_idx, subnet_opt_name):
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
        stree.get_init_func_ast().body.append(new_ast)

    @staticmethod
    def cell_container_process(ast_assign, stree, targets, func_scope_name, call_args, call_kwargs,
                               op_name, container_obj):
        """ parse cell container object."""
        cell_container = CellContainer(ast_assign, targets, func_scope_name, call_args, call_kwargs,
                                       op_name, stree, container_obj)
        first_node_inputs = AssignParser._create_inputs_for_cell_container(ast_assign)
        for i, cell in enumerate(container_obj):
            cell_name = type(cell).__name__
            is_sub_tree = is_subtree(cell)
            if is_sub_tree:
                stb = SymbolTreeBuilder(cell)
                new_stree = stb.build()
                sub_node = TreeNode.create_tree_node(new_stree, None, targets, cell_name, call_args,
                                                     call_kwargs, cell_name, cell)
                AssignParser._update_cell_container_in_init(stree, func_scope_name, i, new_stree.get_opt_cls_name())
            else:
                sub_node = Node.create_call_buildin_op(cell, None, targets, cell_name, call_args,
                                                       call_kwargs, cell_name)
            # add sub node to cell_container
            cell_container.append(sub_node, False)
            # set node inputs, those input nodes are NOT inserted in container, only
            # topological relationship is updated.
            if i == 0:
                for idx, arg_provider in enumerate(first_node_inputs):
                    sub_node.set_arg_providers(idx, (arg_provider, 0))
            else:
                sub_node.set_arg_providers(0, (cell_container.node_list[i-1], 0))
        return cell_container

    @staticmethod
    def process_external_function(stree, func_name, file_path):
        """
        Process external function.
        Ast of external function defined in specifical file_path will be saved to generate codes.
        """
        for k, m in sys.modules.items():
            if k in ("_ast", "ast"):
                continue
            if hasattr(m, func_name):
                func = getattr(m, func_name)
                if not inspect.isfunction(func):
                    continue
                func_source_code_file = inspect.getfile(func)
                if func_source_code_file != file_path:
                    continue
                source_code = inspect.getsource(func)
                ast_root: ast.Module = ast.parse(source_code)
                stree.get_external_ast().append(ast_root.body[0])
                return func, ast_root.body[0]
        logger.info(f"Cannot get ast of function {func_name} from {file_path}.")
        return None, None

    def _process_internal_function(self, stree: SymbolTree, func_name):
        """Process internal function."""
        func_inst = getattr(stree.get_origin_network(), func_name)
        ast_functiondef = None
        for body in stree.get_class_ast().body:
            if isinstance(body, ast.FunctionDef) and func_name == body.name:
                ast_functiondef = body
        return func_inst, ast_functiondef

    def _create_callfunction_node(self, targets: [ScopedValue], func_scope_name: ScopedValue, args: [ScopedValue],
                                  kwargs: {str: ScopedValue}, node_name: str, ast_assign: ast.Assign,
                                  ast_functiondef: ast.FunctionDef, stree: SymbolTree, instance):
        """Create CallFunction node for class internal function."""
        node = CallFunction(targets, func_scope_name, args, kwargs, node_name, ast_assign, ast_functiondef,
                            stree, instance)
        # expand ast codes
        ast_functiondef = FlattenRecursiveStmt().transform(ast_functiondef, [func_scope_name.value], stree)
        # parse ast codes into CallFunction Node
        parser = ParserRegister.instance().get_parser(ast.FunctionDef)
        parser.process(stree, ast_functiondef, node_manager=node)
        return node

    def _convert_ast_call_to_node(self, ast_call: ast.Call, ast_assign: ast.Assign, stree: SymbolTree,
                                  node_manager: NodeManager) -> Node:
        """
        Convert ast.Call to a symbol tree node.

        Args:
            ast_call (ast.Call): An ast.Call of assign node in construct.
            ast_assign (ast.Assign): Assign node in construct.
            stree (SymbolTree): Symbol Tree under parsing.
            node_manager (NodeManager): NodeManager those asts belong to.

        Returns:
            An instance of Node in Symbol Tree.

        Raises:
            RuntimeError: If operator instance invoked by assign is undefined.
        """
        targets = AssignParser._get_targets(AssignParser._create_scopedvalue(ast_assign.targets[0]))
        func_name = AssignParser._get_func_name(ast_call)
        if func_name is None or func_name == "":
            raise RuntimeError("function name not exist")
        func_scope = AssignParser._get_func_scope(ast_call, node_manager)
        func_scope_name = ScopedValue.create_naming_value(func_name, func_scope)
        call_args = [AssignParser._create_scopedvalue(arg) for arg in ast_call.args]
        call_kwargs = AssignParser._create_kwargs(ast_call.keywords)

        func_inst = AssignParser._get_call_instance(func_scope, func_name, stree)
        if func_inst is None:
            # Function is not Cell and Primitive
            if func_scope in ('self', stree.get_opt_cls_name()) and hasattr(stree.get_origin_network(), func_name):
                # Function defined in current class
                func_inst, ast_functiondef = self._process_internal_function(stree, func_name)
                if ast_functiondef is None:
                    raise RuntimeError(f"Find ast of function {func_scope}.{func_name} in symbol tree class failed.")
                node = self._create_callfunction_node(targets, func_scope_name, call_args, call_kwargs, func_name,
                                                      ast_assign, ast_functiondef, stree, func_inst)
            elif is_functional(func_name):
                # Function defined in mindspore.ops.functional
                parser = ParserRegister.instance().get_parser(type(ast_call.func)) # ast.Name or ast.Attribute
                func_name = parser.process(stree, ast_call.func, node_manager).split(".")[-1]
                func_inst = get_functional(func_name)
                node = Node.inner_create_call_function(func_name, ast_assign, func_name, func_inst, targets,
                                                       call_args, call_kwargs)
            else:
                origin_net_file = inspect.getfile(type(stree.get_origin_network()))
                if not os.path.exists(origin_net_file):
                    raise RuntimeError(f"For MindSpore Rewrite, in assign parser, origin_net_file "
                                       f"{origin_net_file} not exist")
                func_inst, ast_functiondef = AssignParser.process_external_function(stree, func_name, origin_net_file)
                node = Node.inner_create_call_function(func_name, ast_assign, func_name, func_inst, targets,
                                                       call_args, call_kwargs)
            return node
        if isinstance(func_inst, tuple(AssignParser.types_for_cell_container)):
            node = AssignParser.cell_container_process(ast_assign, stree, targets, func_scope_name, call_args,
                                                       call_kwargs, func_name, func_inst)
            return node
        if isinstance(func_inst, Primitive):
            return Node.create_call_buildin_op(func_inst, ast_assign, targets, func_scope_name, call_args, call_kwargs,
                                               func_name)
        if isinstance(func_inst, Cell):
            if is_subtree(func_inst):
                # Instance of function is user custom network, create sub-symboltree
                stb = SymbolTreeBuilder(func_inst)
                new_stree = stb.build()
                AssignParser._update_field_in_init(func_scope, func_name, stree, new_stree)
                replacer = AstReplacer(new_stree.get_class_ast())
                replacer.replace_all(new_stree.get_ori_cls_name(), new_stree.get_opt_cls_name())
                return TreeNode.create_tree_node(new_stree, ast_assign, targets, func_scope_name, call_args,
                                                 call_kwargs, func_name, new_stree.get_origin_network())
            # Instance of function is buildin cells
            return Node.create_call_buildin_op(func_inst, ast_assign, targets, func_scope_name, call_args, call_kwargs,
                                               func_name)
        raise RuntimeError("For MindSpore Rewrite, unsupported operation in ast.Call found: ",
                           type(func_inst).__name__)

    @staticmethod
    def _tuple_elts_support_scopledvalue(value: ast.Tuple) -> bool:
        """ check whether each element's type in tuple is supported by scopled value. """
        if not isinstance(value, ast.Tuple):
            raise RuntimeError("For AssignParser._tuple_elts_support_scopledvalue(), the type of value should be "
                               f"Tuple, but got {type(value).__name__}")

        for elt in value.elts:
            if not isinstance(elt, (ast.Name, ast.Attribute, ast.Tuple, ast.Constant, ast.Num, ast.Str, ast.Bytes)):
                return False
        return True

    @staticmethod
    def _convert_ast_mathops_to_node(ast_op: Union[ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare],
                                     ast_assign: ast.Assign) -> Node:
        """
        Convert ast node of math operations(ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare) to
        a symbol tree node.

        Args:
            ast_op (Union[ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare]): An assign node with mathematival
                operation in construct function.
            ast_assign (ast.Assign): Assign node in construct.

        Returns:
            An instance of Node in Symbol Tree.

        Raises:
            TypeError: The type of parameter 'ast_op' is not in (ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare).

        """
        if not isinstance(ast_op, (ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare)):
            raise TypeError("The type of parameter 'ast_op' must be one of (ast.BinOp, ast.UnaryOp, "
                            "ast.BoolOp, ast.Compare), but got ", type(ast_op))

        targets = AssignParser._get_targets(AssignParser._create_scopedvalue(ast_assign.targets[0]))
        args = []
        op_type_str = type(ast_op).__name__
        op_type = ScopedValue.create_naming_value(op_type_str)
        ops = {}
        name = op_type_str
        if isinstance(ast_op, ast.BinOp):
            op = type(ast_op.op).__name__
            name = f'{name}_{op}'
            ops['0'] = ScopedValue.create_naming_value(op)
            args.append(AssignParser._create_scopedvalue(ast_op.left))
            args.append(AssignParser._create_scopedvalue(ast_op.right))
        elif isinstance(ast_op, ast.UnaryOp):
            op = type(ast_op.op).__name__
            name = f'{name}_{op}'
            ops['0'] = ScopedValue.create_naming_value(op)
            args.append(AssignParser._create_scopedvalue(ast_op.operand))
        elif isinstance(ast_op, ast.BoolOp):
            op = type(ast_op.op).__name__
            name = f'{name}_{op}'
            ops['0'] = ScopedValue.create_naming_value(op)
            for value in ast_op.values:
                args.append(AssignParser._create_scopedvalue(value))
        elif isinstance(ast_op, ast.Compare):
            args.append(AssignParser._create_scopedvalue(ast_op.left))
            for idx, ast_cmp_op in enumerate(ast_op.ops):
                op = type(ast_cmp_op).__name__
                name = f'{name}_{op}'
                ops[str(idx)] = ScopedValue.create_naming_value(op)
                args.append(AssignParser._create_scopedvalue(ast_op.comparators[idx]))
        name = name.lower()
        return Node.create_mathops_node(ast_assign, targets, op_type, args, ops, name)

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

        Raises:
            RuntimeError: Only support one target in assign now.
            RuntimeError: Unsupported node type in construct function.
        """

        targets = node.targets
        try:
            if len(targets) != 1:
                raise RuntimeError(
                    error_str(f"only support one target in assign now.", targets, node))
            value = node.value
            if isinstance(value, ast.Call):
                node_ = self._convert_ast_call_to_node(value, node, stree, node_manager)
                stree.append_origin_field(node_, node_manager)
            elif isinstance(value, (ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare)):
                node_ = AssignParser._convert_ast_mathops_to_node(value, node)
                stree.append_origin_field(node_, node_manager)
            elif isinstance(value, ast.Subscript):
                logger.info(f"ops-call({astunparse.unparse(node)}) in assign will be supported in near feature, "
                            f"ignored as a python node now")
                stree.try_append_python_node(node, node, node_manager)
            elif isinstance(value, (ast.Name, ast.Constant, ast.Attribute, ast.Num, ast.NameConstant,
                                    ast.Bytes, ast.Str)):
                if isinstance(value, ast.Name):
                    node_name = "name_assign"
                elif isinstance(value, ast.Constant):
                    node_name = "constant_assign"
                elif isinstance(value, ast.Attribute):
                    node_name = "attribute_assign"
                else:
                    node_name = "other_assign"
                targets = AssignParser._get_targets(AssignParser._create_scopedvalue(node.targets[0]))
                call_args = [AssignParser._create_scopedvalue(value)]
                node_ = Node.create_call_pass_through_method(node, targets, call_args, {}, node_name)
                stree.append_origin_field(node_, node_manager)
            elif isinstance(value, ast.Tuple):
                if AssignParser._tuple_elts_support_scopledvalue(value):
                    # ensure that each element's type in tuple is supported by scopled value
                    targets = AssignParser._get_targets(AssignParser._create_scopedvalue(node.targets[0]))
                    args = []
                    for elt in value.elts:
                        args.append(AssignParser._create_scopedvalue(elt))
                    node_ = Node.create_call_method(node, targets, ScopedValue.create_naming_value("tuple"),
                                                    args, {}, "tuple")
                    stree.append_origin_field(node_, node_manager)
                else:
                    logger.info(f"some elements in Tuple of assign({astunparse.unparse(node)}) are not supported "
                                "in rewrite, fallback to python")
                    stree.try_append_python_node(node, node, node_manager)
            elif isinstance(value, (ast.List, ast.Dict)):
                # add these as callmethod node if necessary
                stree.try_append_python_node(node, node, node_manager)
            else:
                raise RuntimeError(
                    error_str(f"only support (ast.Call, ast.BinOp, ast.BoolOp, ast.Subscript, ast.Name, ast.Constant, "
                              f"ast.Attribute, ast.Num, ast.NameConstant, ast.Bytes, ast.Str, ast.Tuple, ast.List, "
                              f"ast.Dict) as value of ast.assign, but got ast type '{type(value).__name__}'",
                              child_node=value, father_node=node))
        except RuntimeError:
            logger.info(f"ops-call({astunparse.unparse(node).strip()}) not supported in rewrite, fallback to python")
            stree.try_append_python_node(node, node, node_manager)


g_assign_parser = reg_parser(AssignParser())
