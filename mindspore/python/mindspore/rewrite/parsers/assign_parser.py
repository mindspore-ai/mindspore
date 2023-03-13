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
import ast
import os
import astunparse

from mindspore import log as logger
from mindspore._extends.parse.namespace import CellNamespace
from mindspore.nn import Cell, SequentialCell
from mindspore.ops import operations as P
from mindspore.ops import Primitive
from mindspore.rewrite.parser_register import ParserRegister
from mindspore.rewrite.namespace import is_subtree, is_functional, get_functional
from mindspore.rewrite.symbol_tree import SymbolTree
from mindspore.rewrite.node import Node, TreeNode, CellContainer
from mindspore.rewrite.parser import Parser
from mindspore.rewrite.parser_register import reg_parser
from mindspore.rewrite.api.scoped_value import ScopedValue, ValueType
from mindspore.rewrite.symbol_tree_builder import SymbolTreeBuilder
from mindspore.rewrite.ast_helpers import AstReplacer, AstModifier
from mindspore.rewrite.common.event import Event
from ..common import error_str


class AssignParser(Parser):
    """Parse ast.Assign in construct function to node of SymbolTree."""

    def __init__(self):
        """Constructor"""
        super(AssignParser, self).__init__()
        self._cell_namespce = CellNamespace('mindspore.nn')
        self._primitive_namespce = CellNamespace('mindspore.ops.operations')

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
            if not isinstance(tuple_elt, (ast.Constant, ast.Name)):
                raise RuntimeError(f"Only support ast.Constant or ast.Name as elts of ast.Tuple, "
                                   f"but got ast type {type(tuple_elt).__name__}",
                                   child_node=tuple_elt, father_node=node)
            if isinstance(tuple_elt, ast.Constant):
                tuple_values.append(tuple_elt.value)
            elif isinstance(tuple_elt, ast.Name):
                tuple_values.append(tuple_elt.id)
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
    def _get_func_name(ast_node: ast.Call) -> str:
        """
        Get the func name from ast.Call.

        Args:
            ast_node (ast.Call): Input ast.Call node.

        Returns:
            Func name.

        Raises:
            RuntimeError: Func of input ast node is not ast.Name or ast.Attribute.
        """
        func = ast_node.func
        if isinstance(func, ast.Name):
            return func.id
        if isinstance(func, ast.Attribute):
            return func.attr
        if isinstance(func, ast.Call):
            return AssignParser._get_func_name(func)
        raise RuntimeError(error_str(f"funcValue should be Name or a Attribute or a Call, but got ast type "
                                     f"'{type(func).__name__}'", child_node=func, father_node=ast_node))

    @staticmethod
    def _get_func_scope(ast_node: ast.Call) -> str:
        """
        Get the func scope from ast.Call.

        Args:
            ast_node (ast.Call): Input ast.Call node.

        Returns:
            Func scope.

        Raises:
            RuntimeError: FuncValue is not an ast.Name when func is an ast.Attribute.
            RuntimeError: Func of input ast node is not ast.Name or ast.Attribute.
        """
        func = ast_node.func
        if isinstance(func, ast.Name):
            return ""
        if isinstance(func, ast.Attribute):
            parser = ParserRegister.instance().get_parser(type(func))
            value = parser.process(None, func)
            return value.rsplit(".", 1)[0]
        if isinstance(func, ast.Call):
            return AssignParser._get_func_scope(func)
        raise RuntimeError(error_str(f"funcValue should be Name or a Attribute or a Call, but got ast type "
                                     f"'{type(func).__name__}'", child_node=func, father_node=ast_node))

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
    def _find_op_and_type(func_scope, func_name, stree: SymbolTree):
        """
        Get the func scope from ast.Call.

        Args:
            func_scope (str): Func scope.
            func_name (str): Func name.
            stree (SymbolTree): Belong SymbolTree.

        Returns:
            A type represents type of op and an instance represents operator instance.
        """

        if func_scope != "self":
            logger.warning("Not support parse operator which is instantiated at runtime now: %s; name: %s", func_scope,
                           func_name)
        var_dict = stree.get_origin_network().__dict__
        for key, value in var_dict["_cells"].items():
            if key == func_name:
                return type(value), value

        for key, value in var_dict["_primitives"].items():
            if key == func_name:
                return type(value), value
        return type(None), None

    @staticmethod
    def _get_targets(all_targets: ScopedValue) -> [Union[ScopedValue, str]]:
        """Get targets from tuple or single value."""
        targets: [Union[ScopedValue, str]] = []
        if all_targets.type == ValueType.TupleValue:
            for single_target in all_targets.value:
                if not isinstance(single_target, ScopedValue) and not isinstance(single_target.value, str):
                    raise RuntimeError(f"For MindSpore Rewrite, only support str target in tuple, but got type "
                                       f"{type(single_target).__name__}")
                targets.append(single_target)
        else:
            targets.append(all_targets)
        return targets

    @staticmethod
    def _update_field_in_init(func_scope, func_name, stree: SymbolTree, sub_tree: SymbolTree) -> bool:
        """
        When node is an invoking to sub-network, update value of ast.Assign of corresponding field in `__init__` method.

        Update from:

        .. code-block::

        self.field = getattr(self._handler, "field")

        to:

        .. code-block::

        self.field = SubNetwork(global_vars.get("field_args"))

        Args:
            func_scope (str): A string represents scope of function symbol.
            func_name (str): A string represents function symbol.
            stree (SymbolTree): The SymbolTree corresponding to main-network.
            sub_tree (SymbolTree): The SymbolTree corresponding to sub-network.

        Raises:
            NotImplementedError: If `func_scope` is not "self", it means corresponding op is inited in forward method.
            NotImplementedError: If targets of ast.Assign of corresponding field in `__init__` method.
        """

        changed = False
        if func_scope != "self":
            logger.warning("Not support parse operator which is instantiated at runtime now: %s; name: %s", func_scope,
                           func_name)
        init_func_ast = stree.get_init_func_ast()
        class_name = sub_tree.get_opt_cls_name()
        for body in init_func_ast.body:
            if not isinstance(body, ast.Assign):
                continue
            if len(body.targets) > 1:
                raise NotImplementedError(error_str("not support multi-targets in assign now!", father_node=body))
            target = body.targets[0]
            if not isinstance(target, ast.Attribute) or not isinstance(target.value, ast.Name):
                continue
            if target.value.id != "self" or target.attr != func_name:
                continue
            changed = True
            setattr(stree.get_origin_network(), func_name, sub_tree.get_origin_network())
            args_call = AstModifier.create_call(ScopedValue(ValueType.NamingValue, "", "getattr"),
                                                [ScopedValue(ValueType.NamingValue, "", "obj"),
                                                 ScopedValue(ValueType.StringValue, "", func_name)])
            body.value = ast.Call(func=ast.Name(class_name, ast.Store()), args=[args_call], keywords=[])
            break
        return changed

    @staticmethod
    def _convert_ast_binop_to_node(ast_node: ast.BinOp, father_ast_node: ast.Assign) -> Node:
        """convert ast.BinOp to Node"""

        # only support ast.Add now
        op = P.Add()
        func_ast = ast.Attribute(value=ast.Name(id='F', ctx=ast.Load()), attr='add', ctx=ast.Load())
        func = ScopedValue.create_naming_value('add', 'F')

        father_ast_node.value = ast.Call(func=func_ast, args=[ast_node.left, ast_node.right], keywords=[])
        targets = AssignParser._get_targets(AssignParser._create_scopedvalue(father_ast_node.targets[0]))
        call_args = [AssignParser._create_scopedvalue(arg) for arg in father_ast_node.value.args]
        return Node.create_call_buildin_op(op, father_ast_node, targets, func, call_args, {})

    @staticmethod
    def _create_inputs_for_cell_container(father_ast_node) -> ['Node']:
        """Create inputs for cell container first node."""
        call_ast_node = father_ast_node.value
        if not isinstance(call_ast_node, ast.Call):
            raise RuntimeError(error_str(f"when creating input node for cellcontainer, value of input father ast node"
                                         "is not ast.Call!'", child_node=call_ast_node, father_node=father_ast_node))
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
                                         child_node=call_ast_node, father_node=father_ast_node))

        return first_node_inputs

    def _cell_container_process(self, ast_node, stree, targets, func, call_args, call_kwargs, op_name, container_obj):
        """ parse cell container object."""
        cell_container = CellContainer(ast_node, targets, func, call_args, call_kwargs, op_name, container_obj)
        cell_container.set_belong_symbol_tree(stree)
        first_node_inputs = AssignParser._create_inputs_for_cell_container(ast_node)
        for i, cell in enumerate(container_obj):
            is_sub_tree = is_subtree(type(cell).__name__)
            if is_sub_tree:
                stb = SymbolTreeBuilder(cell)
                new_stree = stb.build()
                replacer = AstReplacer(new_stree.get_class_ast())
                replacer.replace_all(new_stree.get_ori_cls_name(), new_stree.get_opt_cls_name())
                sub_node = TreeNode.create_tree_node(new_stree, ast_node, targets, func, call_args, call_kwargs,
                                                     type(cell).__name__, cell)
            else:
                sub_node = Node.create_call_buildin_op(cell, ast_node, targets, func, call_args, call_kwargs,
                                                       type(cell).__name__)
            # add sub node to cell_container
            cell_container.append(sub_node)
            # set node inputs
            if i == 0:
                sub_node.set_inputs(first_node_inputs)
            else:
                sub_node.set_inputs([cell_container.node_list[i-1]])
        return cell_container

    def _convert_ast_call_to_node(self, ast_node: ast.Call, father_ast_node: ast.Assign, stree: SymbolTree) -> Node:
        """
        Convert ast.Call to a symbol tree node.

        Args:
            ast_node (ast.Call): An ast.Call of assign node in construct.
            father_ast_node (ast.Assign): Assign node in construct.
            stree (SymbolTree): Symbol Tree under parsing.

        Returns:
            An instance of Node in Symbol Tree.

        Raises:
            RuntimeError: If operator instance invoked by assign is undefined.
        """
        targets = AssignParser._get_targets(AssignParser._create_scopedvalue(father_ast_node.targets[0]))
        func_name = AssignParser._get_func_name(ast_node)
        if func_name is None or func_name == "":
            raise RuntimeError("function name not exist")
        func_scope = AssignParser._get_func_scope(ast_node)
        func = ScopedValue.create_naming_value(func_name, func_scope)
        call_args = [AssignParser._create_scopedvalue(arg) for arg in ast_node.args]
        call_kwargs = AssignParser._create_kwargs(ast_node.keywords)

        _, op = AssignParser._find_op_and_type(func_scope, func_name, stree)
        if op is None:
            if is_functional(func_name):
                parser = ParserRegister.instance().get_parser(type(ast_node.func))
                func_name = parser.process(stree, ast_node.func)
                func = get_functional(func_name.split(".")[-1])
                node = stree.inner_create_call_function(func_name, father_ast_node, func_name, func, targets,
                                                        call_args, call_kwargs)
                return node
            raise RuntimeError(error_str(f"operator instance undefined.",
                                         child_node=ast_node.func, father_node=ast_node))
        if isinstance(op, SequentialCell):
            node = self._cell_container_process(father_ast_node, stree, targets, func, call_args, call_kwargs,
                                                func_name, op)
            return node
        if isinstance(op, Primitive):
            return Node.create_call_buildin_op(op, father_ast_node, targets, func, call_args, call_kwargs, func_name)
        if isinstance(op, Cell):
            is_sub_tree = is_subtree(type(op).__name__)
            if is_sub_tree:
                stb = SymbolTreeBuilder(op)
                new_stree = stb.build()
                changed = AssignParser._update_field_in_init(func_scope, func_name, stree, new_stree)
                if changed:
                    # class SubSubNet:
                    #     def __init__(self, global_vars):
                    #         self._handler = global_vars.get("handler")
                    #
                    # class SubNet:
                    #     def __init__(self, global_vars):
                    #         self._handler = global_vars.get("handler")
                    #         self._subsubnet = None
                    #         if xxx:
                    #             self._subsubnet = SubSubNet(xxx)
                    #
                    # Assuming there are two instance of SubNet A and B. "if xxx" in A is True, and in B is False.
                    # So self._subsubnet in A is an instance of SubSubNet, and in B is None.
                    # So After rewrite, A's code:
                    # class SubNetA:
                    #     def __init__(self, global_vars):
                    #         self._handler = global_vars.get("handler")
                    #         self._subsubnet = SubSubNet(global_vars.get("subsubnet_args"))
                    # while B's code:
                    # class SubNetB:
                    #     def __init__(self, global_vars):
                    #         self._handler = global_vars.get("handler")
                    #         self._subsubnet = getattr(self._handler, "_subsubnet")
                    # So SubNet should use SubNetA as its code when _update_field_in_init return True.
                    # So SubNet should use SubNetB as its code when _update_field_in_init return False or undefined
                    # error will occur to "global_vars.get("subsubnet_args")".
                    stree.on_change(Event.CodeChangeEvent)
                # Sub-network in main-network is expressed as:
                # self._subnet = SubNet(global_vars.get("subnet_args"))
                # when subnet is changed, its class will change, take SubNet1 as new class-name, so code main-network
                # also need to change:
                # self._subnet = SubNet1(global_vars.get("subnet_args"))
                # so a change in sub-network should also be identified as a change in main-network.
                # so main-network should observe sub-network
                replacer = AstReplacer(new_stree.get_class_ast())
                replacer.replace_all(new_stree.get_ori_cls_name(), new_stree.get_opt_cls_name())
                return TreeNode.create_tree_node(new_stree, father_ast_node, targets, func, call_args, call_kwargs,
                                                 func_name, new_stree.get_origin_network())
            return Node.create_call_buildin_op(op, father_ast_node, targets, func, call_args, call_kwargs, func_name)
        raise RuntimeError("For MindSpore Rewrite, only support Primitive or Cell operator or Primitive operator, got ",
                           type(op).__name__)

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
    def _convert_ast_mathops_to_node(ast_node: Union[ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare],
                                     father_ast_node: ast.Assign) -> Node:
        """
        Convert ast node of math operations(ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare) to
        a symbol tree node.

        Args:
            ast_node (Union[ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare]): An assign node with mathematival
                operation in construct function.
            father_ast_node (ast.Assign): Assign node in construct.

        Returns:
            An instance of Node in Symbol Tree.

        Raises:
            TypeError: The type of parameter 'ast_node' is not in (ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare).

        """
        if not isinstance(ast_node, (ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare)):
            raise TypeError("The type of parameter 'ast_node' must be one of (ast.BinOp, ast.UnaryOp, "
                            "ast.BoolOp, ast.Compare), but got ", type(ast_node))

        targets = AssignParser._get_targets(AssignParser._create_scopedvalue(father_ast_node.targets[0]))
        args = []
        op_type_str = type(ast_node).__name__
        op_type = ScopedValue.create_naming_value(op_type_str)
        ops = {}
        name = op_type_str
        if isinstance(ast_node, ast.BinOp):
            op = type(ast_node.op).__name__
            name = name + '_' + op
            ops['0'] = ScopedValue.create_naming_value(op)
            args.append(AssignParser._create_scopedvalue(ast_node.left))
            args.append(AssignParser._create_scopedvalue(ast_node.right))
        elif isinstance(ast_node, ast.UnaryOp):
            op = type(ast_node.op).__name__
            name = name + '_' + op
            ops['0'] = ScopedValue.create_naming_value(op)
            args.append(AssignParser._create_scopedvalue(ast_node.operand))
        elif isinstance(ast_node, ast.BoolOp):
            op = type(ast_node.op).__name__
            name = name + '_' + op
            ops['0'] = ScopedValue.create_naming_value(op)
            for value in ast_node.values:
                args.append(AssignParser._create_scopedvalue(value))
        elif isinstance(ast_node, ast.Compare):
            args.append(AssignParser._create_scopedvalue(ast_node.left))
            for idx, ast_op in enumerate(ast_node.ops):
                op = type(ast_op).__name__
                name = name + '_' + op
                ops[str(idx)] = ScopedValue.create_naming_value(op)
                args.append(AssignParser._create_scopedvalue(ast_node.comparators[idx]))
        name = name.lower()
        return Node.create_mathops_node(father_ast_node, targets, op_type, args, ops, name)

    def process(self, stree: SymbolTree, node: ast.Assign):
        """
        Parse ast.Assign and create a node in symbol tree.

        - Create node when value of ast.Assign is in [ast.Call, ast.Name, ast.Constant, ast.Attribute].
        - Create python node when value of ast.Assign is in [ast.BinOp, ast.BoolOp, ast.Subscript, ast.List, ast.Tuple,
          ast.Dict].
        - Other value types are not supported.

        Args:
            stree ([SymbolTree]): Symbol Tree under parsing.
            node ([ast.Assign]): An ast.Assign node.

        Raises:
            RuntimeError: Only support one target in assign now.
            RuntimeError: Unsupported node type in construct function.
        """

        targets = node.targets
        try:
            if len(targets) != 1:
                raise RuntimeError(
                    error_str(f"only support one target in assign now.", child_node=targets, father_node=node))
            value = node.value
            if isinstance(value, ast.Call):
                node_ = self._convert_ast_call_to_node(value, node, stree)
                stree.append_origin_field(node_)
            elif isinstance(value, (ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare)):
                node_ = AssignParser._convert_ast_mathops_to_node(value, node)
                stree.append_origin_field(node_)
            elif isinstance(value, ast.Subscript):
                logger.info(f"ops-call({astunparse.unparse(node)}) in assign will be supported in near feature, "
                            f"ignored as a python node now")
                stree.try_append_python_node(node, node)
            elif isinstance(value, (ast.Name, ast.Constant, ast.Attribute, ast.Num, ast.NameConstant,
                                    ast.Bytes, ast.Str)):
                if isinstance(value, ast.Name):
                    node_name = "name_assign"
                elif isinstance(value, ast.Constant):
                    node_name = "constant_assign"
                else:
                    node_name = "attribute_assign"
                targets = AssignParser._get_targets(AssignParser._create_scopedvalue(node.targets[0]))
                call_args = [AssignParser._create_scopedvalue(value)]
                node_ = Node.create_call_pass_through_method(node, targets, call_args, {}, node_name)
                stree.append_origin_field(node_)
            elif isinstance(value, ast.Tuple):
                if AssignParser._tuple_elts_support_scopledvalue(value):
                    # ensure that each element's type in tuple is supported by scopled value
                    targets = AssignParser._get_targets(AssignParser._create_scopedvalue(node.targets[0]))
                    args = []
                    for elt in value.elts:
                        args.append(AssignParser._create_scopedvalue(elt))
                    node_ = Node.create_call_method(node, targets, ScopedValue.create_naming_value("tuple"),
                                                    args, {}, "tuple")
                    stree.append_origin_field(node_)
                else:
                    logger.warning(f"some elements in Tuple of assign({astunparse.unparse(node)}) are not supported "
                                   "in rewrite, fallback to python")
                    stree.try_append_python_node(node, node)
            elif isinstance(value, (ast.List, ast.Dict)):
                # add these as callmethod node if necessary
                stree.try_append_python_node(node, node)
            else:
                raise RuntimeError(
                    error_str(f"only support (ast.Call, ast.BinOp, ast.BoolOp, ast.Subscript, ast.Name, ast.Constant, "
                              f"ast.Attribute, ast.Num, ast.NameConstant, ast.Bytes, ast.Str, ast.Tuple, ast.List, "
                              f"ast.Dict) as value of ast.assign, but got ast type '{type(value).__name__}'",
                              child_node=value, father_node=node))
        except RuntimeError as e:
            if os.getenv("STREE_PYTHON_FALLBACK"):
                logger.info(f"ops-call({astunparse.unparse(node)}) not supported in rewrite, fallback to python")
                stree.try_append_python_node(node, node)
            else:
                raise e


g_assign_parser = reg_parser(AssignParser())
