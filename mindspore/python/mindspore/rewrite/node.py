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
"""Node class define of Rewrite. See detail in Node class docstring."""
from typing import Optional, Union
import ast
import inspect

from mindspore.nn import Cell
from mindspore.ops import Primitive
from mindspore import log as logger
from .._checkparam import Validator, Rel
from .ast_helpers import AstModifier
from .api.scoped_value import ScopedValue, ValueType
from .api.node_type import NodeType
from .namespace import is_subtree
from .ast_helpers.ast_replacer import AstReplacer

PASS_THROUGH_METHOD = ScopedValue.create_naming_value("PassThrough")


class Node:
    """
    Node is a data structure represents a source code line in network. For the most part, Node represents an operator
    invoking in forward which could be an instance of Cell, an instance of Primitive or a callable method. Fields of
    Node has different meaning in different type of node:

    - CallCell: a call-cell node represents an assign statement whose value is a calling to cell in mindspore. `targets`
      is corresponding to targets of ast.Assign which means return values of this cell-op. `args` and `kwargs` are
      corresponding to args and keywords of ast.Call which mean arguments to invoke cell-op's forward method. `func` is
      corresponding to func of call expression which means symbol of the cell-op.
    - CallPrimitive: a call-primitive node represents an ast.Assign whose value is a calling to operator in mindspore.
      `targets`, `args`, `kwargs` and `func` are as previous.
    - CallMethod: a call-method node represents an ast.Assign whose value is a calling to python-method such as `len`.
      `targets` is corresponding to targets of ast.Assign which means return values of this method. `func` represents
      the string name of method. `args` and `kwargs` are corresponding to args and keywords to invoke the method. When
      value of ast.Assign is an ast.Name or ast.Attribute, it means a simplest assign which would also be mapped to
      CallMethod node whose `func` is "PassThrough".
    - GetAttr: retrieves a parameter from the SymbolTree hierarchy. `func` represents which parameter in SymbolTree
      hierarchy. `targets` is corresponding to targets of ast.Assign which means what symbol to accept the result of
      get-attr. `args` and `kwargs` are don't-care.
    - Python: a python node holds an ast-node which is not parsed. a python node means some python statement is not
      supported by Rewrite or ignored by Rewrite. `targets`, `args`, `kwargs` and `func` are don't-care.
    - Input: an input node represents an input of current network which also a parameter of forward method of Cell.
      `targets` is corresponding to arg-name of parameter of forward function. `args` means default-value of parameter
      of forward function. `kwargs` and `func` are don't-care.
    - Output: an output node represents the output of current network which is corresponding to return statement of
      forward method of Cell. `args` represents return values. `func` are always be "return". `targets` and `kwargs` are
      don't-care.
    - Tree: a tree node represents a sub-network call in current network. A sub-network is also a Cell in mindspore, so
      `targets`, `args`, `kwargs` and `func` are same as a call-cell node. `symbol_tree` is a handler of a SymbolTree
      instance.
    """

    def __init__(self, node_type: NodeType, ast_node: Optional[ast.AST], targets: [ScopedValue],
                 func: Optional[ScopedValue], args: [ScopedValue], kwargs: {str: ScopedValue}, name: str, instance):
        """
        Constructor of Node. Rewrite recommend invoking class method of Node to instantiate an instance of Node such
        as `create_call_op`, `create_call_method`, `create_python_node`, `create_input_node` and
        `create_output_node`, etc. rather than invoking constructor of Node directly.

        Args:
            node_type (NodeType): A NodeType as type of Node.
            ast_node (ast.AST, optional): An instance of ast.AST represents corresponding node in ast. `ast_node` should
                not be None except when node type is Unknown.
            targets (list[ScopedValue]): A list of instance of ScopedValue. See detail in docstring of Node class.
            func (ScopedValue, optional): An instance of ScopedValue. See detail in docstring of Node class.
            args (list[ScopedValue]): A list of instance of ScopedValue. See detail in docstring of Node class.
            kwargs (dict{str: ScopedValue}): A list of instance of ScopedValue. See detail in docstring of Node class.
            name (str): A string represents name of node. Name of node will be unique when inserted into SymbolTree.
                Name of node also used as field name in network class.
            instance: Object in network corresponding to this node.
        """
        self._node_type: NodeType = node_type
        self._ast_node: Optional[ast.AST] = ast_node
        self._attribute: {str, object} = {}
        if node_type in (NodeType.CallModule, NodeType.CallCell, NodeType.CallPrimitive):
            self._attribute = Node._get_cell_or_prim_op_attribute(instance)
        self._instance = instance
        self._name = name
        self._func: Optional[ScopedValue] = func
        self._targets: [ScopedValue] = targets
        self._args_num = len(args) if args is not None else 0
        self._kwargs_num = len(kwargs) if kwargs is not None else 0
        self._normalized_args_keys = []  # for saving args' order
        self._normalized_args = self._get_normalized_args(args, kwargs)
        # edge of node
        self._inputs: [Node] = []
        # position in graph nodes list
        # it will affect code-order of python code
        self._prev: Optional[Node] = None
        self._next: Optional[Node] = None
        # A handler of SymbolTree current node belonging to
        self._belong_tree = None

    @classmethod
    def create_call_buildin_op(cls, op: Union[Cell, Primitive], ast_node: Optional[ast.AST], targets: [ScopedValue],
                               func: ScopedValue, args: [ScopedValue] = None, kwargs: {str: ScopedValue}=None,
                               name: str = ""):
        """
        Class method of Node. Instantiate an instance of node whose type is `CallCell` or `CallPrimitive`.
        A `CallCell` node represents an invoking to cell-op.
        A `CallPrimitive` node represents an invoking to primitive-op.

        Args:
            op (Union[Cell, Primitive]): An instance of `Cell` or `Primitive` corresponding to this node.
            ast_node ([ast.AST, optional]): An instance of ast.AST represents corresponding node in ast.
            targets (list[ScopedValue]): A list of instance of `ScopedValue`. See detail in docstring of Node class.
            func ([ScopedValue, optional]): An instance of `ScopedValue`. See detail in docstring of Node class.
            args (list[ScopedValue]): A list of instance of `ScopedValue`. See detail in docstring of Node class.
            kwargs (dict{str: ScopedValue}): A list of instance of `ScopedValue`. See detail in docstring of `Node`
                class.
            name (str): A string represents name of node. Name of node will be unique when inserted into `SymbolTree`.
                Name of node also used as field name in network class.
        """

        if not isinstance(op, (Cell, Primitive)):
            raise ValueError("Input op is not a buildin op(Cell or Primitive): ", type(op))
        non_custom_args = Node._handle_custom_obj_in_args(args)
        non_custom_kwargs = Node._handle_custom_obj_in_kwargs(kwargs)
        if ast_node is None:
            ast_node = AstModifier.create_call_assign(targets, func, non_custom_args, non_custom_kwargs)
        if isinstance(op, Cell):
            node_type = NodeType.CallCell
        else:
            node_type = NodeType.CallPrimitive
        return cls(node_type, ast_node, targets, func, args, kwargs, name, op)

    @classmethod
    def create_call_method(cls, ast_node: Optional[ast.AST], targets: [Union[ScopedValue, str]],
                           func: Union[ScopedValue, str], args: [ScopedValue] = None, kwargs: {str: ScopedValue}=None,
                           name: str = ""):
        """
        Class method of Node. Instantiate an instance of node whose type is CallCell. A CallCell node represents an
        invoking to cell-op.

        Args:
            ast_node ([ast.AST, optional]): An instance of ast.AST represents corresponding node in ast. `ast_node`
                should not be None currently.
            targets (list[ScopedValue]): A list of instance of ScopedValue. See detail in docstring of Node class.
            func ([ScopedValue, optional]): An instance of ScopedValue. See detail in docstring of Node class.
            args (list[ScopedValue]): A list of instance of ScopedValue. See detail in docstring of Node class.
            kwargs (dict{str: ScopedValue}): A list of instance of ScopedValue. See detail in docstring of Node class.
            name (str): A string represents name of node. Name of node will be unique when inserted into SymbolTree.
                Name of node also used as field name in network class.
        """
        if args is None:
            args = []
        if kwargs is None:
            kwargs = {}
        if isinstance(func, str):
            func = ScopedValue.create_naming_value(func)
        new_targets = Node._handle_targets(targets)
        if ast_node is None:
            raise RuntimeError("Input ast_node is None")
        return cls(NodeType.CallMethod, ast_node, new_targets, func, args, kwargs, name, None)

    @classmethod
    def create_call_pass_through_method(cls, ast_node: Optional[ast.AST], targets: [Union[ScopedValue, str]],
                                        args: [ScopedValue] = None, kwargs: {str: ScopedValue}=None, name: str = ""):
        """Create pass through node."""
        return Node.create_call_method(ast_node, targets, PASS_THROUGH_METHOD, args, kwargs, name)

    @classmethod
    def create_python_node(cls, ast_node: ast.AST, name: str = "", instance=None):
        """
        Class method of Node. Instantiate an instance of node whose type is Python. A Python node represents some python
        statement is not supported by Rewrite or ignored by Rewrite.

        Args:
            ast_node (ast.AST): An instance of ast.AST represents corresponding node in ast.
            name (str): A string represents name of node. Name of node will be unique when inserted into SymbolTree.
                Name of node also used as field name in network class.
            instance: An object corresponding to this node in network.
        """
        return cls(NodeType.Python, ast_node, None, None, [], {}, name, instance)

    @classmethod
    def create_input_node(cls, ast_node: ast.AST, arg_name: str, default: Optional[ScopedValue] = None, name: str = ""):
        """
        Class method of Node. Instantiate an instance of node whose type is Input. An Input node represents input of
        SymbolTree which is corresponding to parameters of forward function.

        Args:
            ast_node (ast.AST): An instance of ast.AST represents corresponding node in ast.
            arg_name (str): A string represents name of parameter.
            default ([ScopedValue, optional]): An instance of ScopedValue represents default value of parameter.
            name (str): A string represents name of node. Name of node will be unique when inserted into SymbolTree.
                Name of node also used as field name in network class.
        """
        target = ScopedValue.create_naming_value(arg_name)
        if default is None:
            args = []
        else:
            args = [default]
        return cls(NodeType.Input, ast_node, [target], None, args, {}, name, None)

    @classmethod
    def create_output_node(cls, ast_node: ast.AST, return_values: [str], name: str = "return"):
        """
        Class method of Node. Instantiate an instance of node whose type is Output. An Output node represents output of
        SymbolTree which is corresponding to return statement of forward function.

        Args:
            ast_node (ast.AST): An instance of ast.AST represents corresponding node in ast.
            return_values (list[str]): A list of string represents name of return values.
            name (str): A string represents name of node. Name of node will be unique when inserted into SymbolTree.
                Name of node also used as field name in network class.
        """
        real_return_values = ScopedValue.create_name_values(return_values)
        return cls(NodeType.Output, ast_node, None, ScopedValue.create_naming_value("return"), real_return_values, {},
                   name, None)

    @classmethod
    def create_mathops_node(cls, ast_node: ast.AST, targets: [ScopedValue],
                            op_type: ScopedValue, args: [ScopedValue],
                            ops: {str: list}, name: str = ""):
        """
        Class method of Node. Instantiate an instance of node whose type is `MathOps` .
        A mathops node is used to represent a node with mathematical operations, such as
        `y = a + b` , `y = not a` , `y = 0 < a < 1`, `y = a or b` , etc.

        Args:
            ast_node ([ast.AST, optional]): An instance of ast.AST represents corresponding node in ast. The type of
                node is ast.Assign, and the type of ast_node.value is one of ast.BinOp, ast.UnaryOp, ast.BoolOp and
                ast.Compare.
            targets (list[ScopedValue]): Targets of mathematical operations. A list of instance of `ScopedValue`.
                See detail in docstring of Node class.
            op_type (ScopedValue): The type of ast_node.value saved by string. A ScopedValue with NamingValue type.
            args (list[ScopedValue]): Values participating in the mathematical operations. All values are saved
                sequentially in the list.
            ops (dict[str:ScopedValue]): Operators participating in the mathematical operations. All operators are
            saved sequentially in the dict, and keys are numbers in string format, such as {'0':'add', '1':'sub'}.
            name (str): A string represents name of node. Name of node will be unique when inserted into `SymbolTree`.
                Name of node also used as field name in network class. The format of mathops node name
                is 'AstNodeName_AstOpName_n'.
        """
        return cls(NodeType.MathOps, ast_node, targets, op_type, args, ops, name, None)

    @staticmethod
    def create_call_op(op: Union[Cell, Primitive], ast_node: Optional[ast.AST], targets: [Union[ScopedValue, str]],
                       func: Union[ScopedValue, str], args: [ScopedValue] = None, kwargs: {str: ScopedValue}=None,
                       name: str = "", is_sub_net: bool = False):
        """
        Static method of Node. Instantiate an instance of node whose type is `CallCell` or `CallPrimitive`.
        If op is custom defined, it is treated by TreeNode.
        A `CallCell` node represents an invoking to cell-op.
        A `CallPrimitive` node represents an invoking to primitive-op.

        Args:
            op (Union[Cell, Primitive]): An instance of `Cell` or `Primitive` corresponding to this node.
            ast_node ([ast.AST, optional]): An instance of ast.AST represents corresponding node in ast.
            targets (list[ScopedValue]): A list of instance of `ScopedValue`. See detail in docstring of Node class.
            func ([ScopedValue, optional]): An instance of `ScopedValue`. See detail in docstring of Node class.
            args (list[ScopedValue]): A list of instance of `ScopedValue`. See detail in docstring of Node class.
            kwargs (dict{str: ScopedValue}): A list of instance of `ScopedValue`. See detail in docstring of `Node`
                class.
            name (str): A string represents name of node. Name of node will be unique when inserted into `SymbolTree`.
                Name of node also used as field name in network class.
            is_sub_net (bool): Indicate that is `cell` a network. If `is_sub_net` is true, Rewrite will try to parse the
                `cell` to a TreeNode, else a CallCell Node. Default is a False.
        """
        Validator.check_value_type("op", op, [Cell, Primitive], "Node")
        if ast_node is not None:
            Validator.check_value_type("ast_node", ast_node, [ast.AST], "Node")
        Validator.check_element_type_of_iterable("targets", targets, [ScopedValue, str], "Node")
        Validator.check_value_type("func", func, [ScopedValue, str], "Node")
        if args is not None:
            Validator.check_element_type_of_iterable("args", args, [ScopedValue], "Node")
        if kwargs is not None:
            Validator.check_element_type_of_dict("kwargs", kwargs, [str], [ScopedValue], "Node")
        cls_name = type(op).__name__

        if args is None:
            args = []
        if kwargs is None:
            kwargs = {}
        if isinstance(func, str):
            func = ScopedValue.create_naming_value(func)
        new_targets = Node._handle_targets(targets)
        if is_sub_net and is_subtree(cls_name):
            from .symbol_tree_builder import SymbolTreeBuilder
            stb = SymbolTreeBuilder(op)
            stree = stb.build()
            replacer = AstReplacer(stree.get_class_ast())
            replacer.replace_all(stree.get_ori_cls_name(), stree.get_opt_cls_name())
            return TreeNode.create_tree_node(stree, ast_node, new_targets, func, args, kwargs, name, op)

        return Node.create_call_buildin_op(op, ast_node, new_targets, func, args, kwargs, name)

    @staticmethod
    def _get_construct_arg_names(parameters):
        """
        Static method of `Node`. Get parameters' names of the construct function.

        Args:
            parameters (MappingProxyType): An ordered mapping of parameters' names to the corresponding Parameter
                objects.

        Raises:
            RuntimeError: Invalid parameter kind.

        Returns:
            - arg_names, Parameters' names, contain parameters of types in [POSITIONAL_ONLY, POSITIONAL_OR_KEYWORD].
            - var_positional_name, Name of VAR_POSITIONAL parameters.
            - var_keyword_name, Name of VAR_KEYWORD parameters.
        """
        position_only_names: [str] = []
        positional_or_keyword_names: [str] = []
        var_positional_name = None
        keyword_only_names: [str] = []
        var_keyword_name = None
        for name, para in parameters.items():
            if para.kind == inspect.Parameter.POSITIONAL_ONLY:  # parameters which appear before a '/'
                position_only_names.append(name)
            elif para.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD:  # parameters which appear before '*' or '*args'
                positional_or_keyword_names.append(name)
            elif para.kind == inspect.Parameter.VAR_POSITIONAL:  # corresponds to a '*args'
                var_positional_name = name
            elif para.kind == inspect.Parameter.KEYWORD_ONLY:  # parameters which appear after '*' and before '**'
                keyword_only_names.append(name)
            elif para.kind == inspect.Parameter.VAR_KEYWORD:  # corresponds to a '**kwargs'
                var_keyword_name = name
            else:
                raise RuntimeError("invalid para kind", para.kind)
        if "self" in position_only_names:
            position_only_names.remove("self")
        if "self" in positional_or_keyword_names:
            positional_or_keyword_names.remove("self")
        names = (position_only_names, positional_or_keyword_names, var_positional_name, keyword_only_names,
                 var_keyword_name)
        return names

    @staticmethod
    def _map_args_names(names: tuple, args: [ScopedValue], kwargs: {str: ScopedValue},
                        normalized_args_keys: [str], normalized_args: {str: ScopedValue}):
        """
        Fill in normalized_args according to the order of parameters of construct func.

        Args:
            names (tuple): Parameters' name got from construct func.
            args (list[ScopedValue]): A list of instance of ScopedValue. See detail in docstring of Node class.
            kwargs (dict{str: ScopedValue}): A list of instance of ScopedValue. See detail in docstring of Node class.
            normalized_args (dict{str: ScopedValue}): The normalized args to be filled.

        Raises:
            RuntimeError: Input args are invalid.
            RuntimeError: Arg name already exist in kwargs.
            RuntimeError: Input kwargs invalid.
        """
        position_only_names, positional_or_keyword_names, var_positional_name, keyword_only_names, var_keyword_name = \
            names
        for arg_index, arg in enumerate(args):
            if arg_index < len(position_only_names):
                arg_key = position_only_names[arg_index]
            elif arg_index < len(position_only_names) + len(positional_or_keyword_names):
                arg_key = positional_or_keyword_names[arg_index - len(position_only_names)]
            elif var_positional_name:
                arg_key = "{}_{}".format(var_positional_name, arg_index)
            else:
                raise RuntimeError("Input args are invalid.")

            if arg_key in kwargs.keys():
                raise RuntimeError("Arg name already exist in kwargs.")
            normalized_args[arg_key] = arg
            normalized_args_keys.append(arg_key)

        # add kwargs according to parameters' order
        parameters_order: [str] = []
        parameters_order.extend(position_only_names)
        parameters_order.extend(positional_or_keyword_names)
        parameters_order.append(var_keyword_name)
        parameters_order.extend(keyword_only_names)
        parameters_order.append(var_keyword_name)

        sorted_kwargs = []
        var_keyword_count = len(parameters_order)
        for arg_key, value in kwargs.items():
            if arg_key not in parameters_order and not var_keyword_name:
                raise RuntimeError("Input kwargs invalid.")
            if arg_key in parameters_order:
                sorted_kwargs.append([arg_key, value, parameters_order.index(arg_key)])
            else:
                sorted_kwargs.append([arg_key, value, var_keyword_count])
                var_keyword_count += 1

        sorted_kwargs.sort(key=lambda x: x[2])
        for sorted_kwarg in sorted_kwargs:
            normalized_args[sorted_kwarg[0]] = sorted_kwarg[1]
            normalized_args_keys.append(sorted_kwarg[0])

    @staticmethod
    def _handle_custom_obj_in_args(args: [ScopedValue]) -> [ScopedValue]:
        """
        Convert CustomObjValue type argument to NamingValue type argument.

        Args:
            args (list[ScopedValue]): A list of instance of ScopedValue to be converted.

        Returns:
            A list of instance of ScopedValue which have been converted.
        """
        result = []
        for arg in args:
            if not isinstance(arg, ScopedValue):
                raise TypeError("arg should be ScopedValue, got: ", type(arg))
            if arg.type == ValueType.CustomObjValue:
                logger.info("custom-object exist in args, should be replace before compile")
                result.append(ScopedValue.create_naming_value("custom-object", "self"))
            else:
                result.append(arg)
        return result

    @staticmethod
    def _handle_custom_obj_in_kwargs(kwargs: {str: ScopedValue}) -> {str: ScopedValue}:
        """
        Convert CustomObjValue type argument to NamingValue type argument.

        Args:
            kwargs (dict{str: ScopedValue}): A str to instance of ScopedValue dict whose value to be converted.

        Returns:
            A str to instance of ScopedValue dict whose value has be converted.
        """
        result: {str, ScopedValue} = {}
        for arg, value in kwargs.items():
            if not isinstance(value, ScopedValue):
                raise TypeError("value should be ScopedValue, got: ", type(value))
            if value.type == ValueType.CustomObjValue:
                result[arg] = ScopedValue.create_naming_value("custom-object", "self")
            else:
                result[arg] = value
        return result

    @staticmethod
    def _handle_targets(targets: [Union[ScopedValue, str]]) -> [ScopedValue]:
        """
        Normalize targets to be a list of ScopedValue. If target is a str, it will be converted to NamingValue type
        ScopedValue.

        Args:
            targets (Union[ScopedValue, str]]): A list whose element could be a ScopedValue or a str to be normalized.

        Returns:
            A list of instance of ScopedValue which have been converted.
        """
        if not isinstance(targets, list):
            raise TypeError("targets should be list, got: ", type(targets))
        results = []
        for target in targets:
            if isinstance(target, str):
                results.append(ScopedValue.create_naming_value(target))
            elif isinstance(target, ScopedValue):
                results.append(target)
            else:
                raise RuntimeError("Invalid symbol type: ", target)
        return results

    @staticmethod
    def _get_cell_or_prim_op_attribute(obj) -> dict:
        """
        Find attributes of cell-op or primitive-op.

        Args:
            obj: A cell-op or a primitive-op.

        Returns:
            A dict represents attributes of input 'obj'.
        """
        attributes = {}
        if obj is None:
            return attributes
        for k, v in obj.__dict__.items():
            if k.startswith("_"):
                continue
            attributes[k] = v
        attributes["cls"] = obj.__class__
        return attributes

    def get_prev(self) -> 'Node':
        """
        Get previous node of current node in source code order.

        Returns:
            An instance of Node as previous node.
        """
        return self._prev

    def get_next(self) -> 'Node':
        """
        Get next node of current node in source code order.

        Returns:
            An instance of Node as next node.
        """
        return self._next

    def has_same_ast(self, node: Union['Node', ast.AST]) -> bool:
        """
        Check if other node holds same ast node with self.

        Args:
            node (Union[Node, ast.AST]): An instance of ast.AST or an instance of node to be compared.

        Returns:
            A bool.
        """
        if isinstance(node, Node):
            return self.has_same_ast(node._ast_node)
        if isinstance(node, ast.AST):
            return id(self._ast_node) == id(node)
        return False

    def get_ast(self) -> Optional[ast.AST]:
        """
        Getter of _ast_node.

        Returns:
            An instance of ast.AST if self._ast_node if not None else None.
        """
        return self._ast_node

    def set_ast(self, ast_node: ast.AST):
        """
        Setter of _ast_node.

        Args:
            ast_node (ast.AST): An instance of ast.AST as new value for _ast_node.
        """
        if not isinstance(ast_node, ast.AST):
            raise TypeError("ast_node should be ast.AST, got: ", type(ast_node))
        self._ast_node = ast_node

    def get_belong_symbol_tree(self):
        """Get the symbol tree to which node belongs."""
        return self._belong_tree

    def set_belong_symbol_tree(self, symbol_tree):
        """Set the symbol tree to which node belongs."""
        self._belong_tree = symbol_tree

    def isolate(self):
        """Link prev node to next node and isolate node from source code order list."""
        origin_prev: Optional[Node] = self._prev
        origin_next: Optional[Node] = self._next
        if origin_prev is not None:
            origin_prev._next = origin_next
        if origin_next is not None:
            origin_next._prev = origin_prev
        self._prev = None
        self._next = None

    def insert_before(self, node: 'Node'):
        """
        Insert a node before current node in source code list. Note that topological order is not determined here.

        Args:
            node (Node): An instance of node to be inserted in.
        """
        node.isolate()
        origin_prev: Optional[Node] = self._prev
        if origin_prev is not None:
            origin_prev._next = node
        node._prev = origin_prev
        node._next = self
        self._prev = node

    def insert_after(self, node: 'Node'):
        """
        Insert a node after current node in source code list. Note that topological order is not determined here.

        Args:
            node (Node): An instance of node to be inserted in.
        """
        node.isolate()
        origin_next: Optional[Node] = self._next
        self._next = node
        node._prev = self
        node._next = origin_next
        if origin_next is not None:
            origin_next._prev = node

    def get_inputs(self) -> ['Node']:
        """
        Getter of _inputs which represents input nodes of current node in topological order.

        Returns:
            A list of instances of Node as input nodes.
        """
        return self._inputs

    def set_inputs(self, inputs: ['Node']):
        """
        Setter of _inputs which represents input nodes of current node in topological order.


        Args:
            inputs (list[Node]): A list of instances of Node as new input nodes.
        """
        self._inputs = inputs

    def get_targets(self) -> [ScopedValue]:
        """
        Getter of _targets.

        - When node_type of current node is CallCell or CallPrimitive or CallMethod or Tree, `targets` are strings
          represents invoke result of the cell-op or primitive-op or function-call which are corresponding to targets of
          ast.Assign.
        - When node_type of current node is Input, `targets` should have only one element which is a string represents
          name of parameter of function.
        - When node_type of current node is Python or Output, `targets` are don't-care.

        Returns:
            A list of instances of ScopedValue as targets of node.
        """
        return self._targets

    def set_targets(self, targets: [ScopedValue]):
        """
        Setter of _targets.

        Note:
            This interface can only be called before node been inserted into symbol-tree because target will be unique
            while insert into symbol-tree, in other word, set_targets is not a user-interface.

            When `_targets` is updated, corresponding ast node would be updated also.

            When node_type of current node is CallCell or CallPrimitive or CallMethod or Tree, `targets` are strings
            represents invoke result of the cell-op or primitive-op or function-call which are corresponding to targets
            of ast.Assign.

            When node_type of current node is Input, `targets` should have only one element which is a string represents
            name of parameter of function.

            When node_type of current node is Python or Output, `targets` are don't-care.

        Args:
            targets ([ScopedValue]): A list of instances of ScopedValue as new targets.
        """
        self._targets = targets
        if self._node_type in (NodeType.CallCell, NodeType.CallMethod, NodeType.CallPrimitive,
                               NodeType.Tree, NodeType.CallFunction, NodeType.CellContainer,
                               NodeType.MathOps):
            self._sync_assign_targets_to_ast()

    def get_func(self) -> ScopedValue:
        """
        Getter of `_func`. See detail in docstring of Node class for meaning of func.

        Returns:
            An instance of ScopedValue.
        """
        return self._func

    def set_func(self, func: ScopedValue):
        """
        Setter of `_func`. See detail in docstring of Node class for meaning of func.

        Note:
            When `_func` is updated, corresponding ast node would be updated also.

        Args:
            func (ScopedValue): An instance of ScopedValue as new func.
        """
        self._func = func
        if self._node_type in (NodeType.CallCell, NodeType.CallPrimitive):
            self._sync_assign_func_to_ast()

    def get_name(self) -> str:
        """
        Getter of `_name`.

        Returns:
            A str represents name of node.
        """
        return self._name

    def set_name(self, name: str):
        """
        Setter of `_name`.

        Args:
            name (str): A str as new name of node.
        """
        self._name = name

    def get_node_type(self) -> NodeType:
        """
        Get the node_type of current node.

        Returns:
            A NodeType as node_type of node.
        """
        return self._node_type

    def get_instance_type(self) -> type:
        """
        Get the instance_type of current node.

        - When node_type of current node is CallCell, instance_type is type of cell-op.
        - When node_type of current node is CallPrimitive, instance_type is type of primitive-op.
        - When node_type of current node is Tree, instance_type is type of network-cell.
        - When node_type of current node is Python, Input, Output or CallMethod, instance_type should be NoneType

        Returns:
            A type.
        """
        return type(self._instance)

    def get_instance(self):
        """
        Get the instance of current node.

        - When node_type of current node is CallCell, instance is an instance of Cell.
        - When node_type of current node is CallPrimitive, instance is an instance of primitive.
        - When node_type of current node is Tree, instance is an instance of network-cell.
        - When node_type of current node is Python, Input, Output or CallMethod, instance should be None

        Returns:
            A object.
        """
        return self._instance

    def set_arg_by_node(self, arg_idx: int, node: 'Node', out_idx: Optional[int] = None):
        """
        Set argument by another Node.
        Note that when _normalized_args is updated, corresponding ast node would be updated also.

        Args:
            arg_idx (int): Indicate which input being modified.
            node (Node): Node as new input. Can be a node or name of node.
            out_idx ([int, optional]): Indicate which output of `node` as new argument. Default is None which means use
                first output of `node_to_link` as new input.

        Raises:
            ValueError: If `arg_idx` is out of range.
            ValueError: If `node` has multi-outputs while `out_idx` is None or `out_idx` is not offered.
        """
        Validator.check_value_type("node", node, [Node], "Node")
        Validator.check_int_range(arg_idx, 0, self._args_num, Rel.INC_LEFT, "arg_idx")
        if out_idx is None:
            if len(node._targets) != 1:
                raise RuntimeError("node should has one output when out_idx is not provided")
            out_idx = 0
        Validator.check_int_range(out_idx, 0, len(node._targets), Rel.INC_LEFT, "arg_idx")
        new_arg = node._targets[out_idx]
        self._normalized_args[self._normalized_args_keys[arg_idx]] = new_arg
        self._sync_arg()

    def set_arg(self, arg: Union[ScopedValue, str], index: int) -> (ScopedValue, ScopedValue):
        """
        Set argument of `node`.
        Note that when _normalized_args is updated, corresponding ast node would be updated also.

        Args:
            index (int): Indicate which input being modified.
            arg (Union[ScopedValue, str]): New argument to been set.

        Raises:
            ValueError: If `index` is out of range.
        """
        Validator.check_int_range(index, 0, self._args_num, Rel.INC_LEFT, "index")
        Validator.check_value_type("arg", arg, [ScopedValue, str], "Node")
        if isinstance(arg, str):
            arg = ScopedValue.create_naming_value(arg)
        old_arg = self._normalized_args.get(self._normalized_args_keys[index])
        self._normalized_args[self._normalized_args_keys[index]] = arg
        self._sync_arg()
        return arg, old_arg

    def set_args(self, args: [ScopedValue]):
        """
        Set arguments of `node`.
        Note that when _normalized_args is updated, corresponding ast node would be updated also.

        Args:
            args (list[ScopedValue]): New arguments to been set.

        Raises:
            TypeError: Element of new argument is not an instance of ScopedValue.
        """
        Validator.check_int_range(len(args), 0, self._args_num, Rel.INC_LEFT, "Length of args")
        Validator.check_element_type_of_iterable("args", args, [ScopedValue], "Node")
        for arg_index, arg in enumerate(args):
            if not isinstance(arg, ScopedValue):
                raise TypeError("arg should be ScopedValue, got: ", type(arg))
            self._normalized_args[self._normalized_args_keys[arg_index]] = arg
        self._sync_arg()

    def set_kwargs(self, kwargs: {str: ScopedValue}):
        """
        Set keywords arguments of 'node'.
        Note that when _normalized_args is updated, corresponding ast node would be updated also.

        Args:
            kwargs (dict{str: ScopedValue}): New arguments to been set.

        Raises:
            TypeError: Value of new argument is not an instance of ScopedValue.
            RuntimeError: Length of new arguments is not equal to length of old arguments.
        """
        Validator.check_int_range(len(kwargs), 0, self._kwargs_num, Rel.INC_LEFT, "Length of kwargs")
        Validator.check_element_type_of_dict("kwargs", kwargs, [str], [ScopedValue], "Node")
        for key, arg in kwargs.items():
            if key not in self._normalized_args.keys() or key not in self._normalized_args_keys:
                raise RuntimeError("Input key is not exist, ", key)
            if not isinstance(arg, ScopedValue):
                raise TypeError("arg should be ScopedValue, got: ", type(arg))
            self._normalized_args[key] = arg
        self._sync_arg()

    def set_kwarg(self, key: str, arg: ScopedValue):
        """
        Set keyword argument of 'node'.
        Note that when _normalized_args is updated, corresponding ast node would be updated also.

        Args:
            key (str): A str represents key of new argument.
            arg (ScopedValue): An instance of ScopedValue represents argument.

        Raises:
            RuntimeError: If 'key' is not in original kwargs' keys.
        """
        if key not in self._normalized_args_keys[self._args_num:] or key not in self._normalized_args.keys():
            raise RuntimeError("Input key is not exist, ", key)
        self._normalized_args[key] = arg
        self._sync_arg()

    def get_args(self):
        """
        Get the arguments of current node.

        - When node_type of current node is CallCell, CallPrimitive or Tree, arguments are corresponding to args of
          ast.Call which represents arguments to invoke cell-op's forward method or primitive-op's `call()` method.
        - When node_type of current node is Input, arguments represents default-value of argument of function.
        - When node_type of current node is Output, arguments represents return values.
        - When node_type of current node is Python, arguments are don't-care.

        Returns:
            A list of instances of ScopedValue.
        """
        args = []
        for arg_index in range(self._args_num):
            args.append(self._normalized_args.get(self._normalized_args_keys[arg_index]))
        return args

    def get_kwargs(self):
        """
        Get the keyword arguments of current node.

        - When node_type of current node is CallCell, CallPrimitive or Tree, keyword arguments are corresponding to
          kwargs of ast.Call which represents arguments to invoke cell-op's forward method or primitive-op's `call()`
          method.
        - When node_type of current node is Python, Input or Output, keyword arguments are don't-care.

        Returns:
            A dict of str to instance of ScopedValue.
        """
        kwargs: {str, ScopedValue} = {}
        for arg_index in range(self._args_num, self._args_num + self._kwargs_num):
            key = self._normalized_args_keys[arg_index]
            kwargs[key] = self._normalized_args.get(key)
        return kwargs

    def get_normalized_args(self) -> {str: ScopedValue}:
        """
        Get the normalized keyword arguments of current node.
        Normalized arguments combine arguments and keyword arguments into keyword arguments by using parameter name as
        key of arguments.

        Returns:
            A dict of str to instance of ScopedValue.
        """
        output = {}
        for key in self._normalized_args_keys:
            output[key] = self._normalized_args.get(key)
        return output

    def set_normalized_args(self, args: {str, ScopedValue}):
        """
        Set the normalized keyword arguments of current node.
        Normalized arguments combine arguments and keyword arguments into keyword arguments by using parameter name as
        key of arguments.

        Args:
            args ({str, ScopedValue}): A dict of str to instance of ScopedValue represents new normalized_args.
        """
        if len(args.values()) != len(self._normalized_args_keys):
            raise RuntimeError("Length of args.values() should be equal to length of _normalized_args_keys, ",
                               len(args.values()), " vs ", len(self._normalized_args_keys))
        for key, arg in args.items():
            self._normalized_args[key] = arg
        self._sync_arg()

    def set_attribute(self, key: str, value):
        """
        Set attribute of current node.

        Args:
            key (str): Key of new attribute.
            value (object): Value of new attribute.
        """
        self._attribute[key] = value

    def set_attributes(self, attributes):
        """
        Set attributes of current node.

        Args:
            attributes (dict): A dict represents new attributes.
        """
        self._attribute = attributes

    def get_attributes(self):
        """
        Get all attributes of current node.

        Returns:
            A dict of str to instance of object as attributes.
        """
        return self._attribute

    def get_attribute(self, key: str):
        """
        Get attribute of current node by key.

        Args:
            key (str): A str represents key of attribute you want to get.

        Returns:
            A object as attribute.
        """
        return self._attribute.get(key)

    def _get_normalized_args(self, args: [ScopedValue], kwargs: {str: ScopedValue}) -> dict:
        """
        Merge args and kwargs to normalized args.
        The keys of args are obtained from the construct function of type(self._instance).

        Args:
            args (list[ScopedValue]): A list of instance of ScopedValue. See detail in docstring of Node class.
            kwargs (dict{str: ScopedValue}): A list of instance of ScopedValue. See detail in docstring of Node class.

        Raises:
            RuntimeError: Input args are invalid.
            RuntimeError: Arg name already exist in kwargs.

        Returns:
            The normalized args.
        """
        if not args:
            args = []
        if not kwargs:
            kwargs = {}
        normalized_args: dict = dict()
        if self._instance and hasattr(type(self._instance), "construct"):
            parameters = inspect.signature(type(self._instance).construct).parameters
            names = Node._get_construct_arg_names(parameters)
            Node._map_args_names(names, args, kwargs, self._normalized_args_keys, normalized_args)
        else:
            logger.debug("fail to get arg name from op, using arg_xx for args' name")
            arg_temp_name, suffix = "arg", 0
            for arg in args:
                arg_key = "{}_{}".format(arg_temp_name, suffix)
                while arg_key in kwargs.keys() or arg_key in normalized_args.keys():
                    suffix += 1
                    arg_key = "{}_{}".format(arg_temp_name, suffix)
                normalized_args[arg_key] = arg
                self._normalized_args_keys.append(arg_key)
            for arg_key, value in kwargs.items():
                normalized_args[arg_key] = value
                self._normalized_args_keys.append(arg_key)
        return normalized_args

    def _sync_assign_func_to_ast(self):
        """Sync func of ast.Call of ast.Assign from self._name when NodeType is CallCell or CallPrimitive."""
        if self._ast_node is None:
            return
        assign_ast = self._ast_node
        if not isinstance(assign_ast, ast.Assign):
            raise TypeError("assign_ast should be ast.Assign, got: ", type(assign_ast))
        call_ast = assign_ast.value
        if not isinstance(call_ast, ast.Call):
            raise TypeError("call_ast should be ast.Call, got: ", type(call_ast))
        func_ast = call_ast.func
        if not self._func.value:
            if isinstance(func_ast, ast.Name):
                func_ast.id = self._func.value
            else:
                call_ast.func = ast.Name(self._func.value, ast.Store())
        else:
            if isinstance(func_ast, ast.Attribute):
                func_value = func_ast.value
                if not isinstance(func_value, ast.Name):
                    raise RuntimeError("Only support ast.Name as value of attribute ", type(func_ast.value))
                func_value.id = self._func.scope
                func_ast.attr = self._func.value
            else:
                call_ast.func = ast.Attribute(ast.Name(self._func.scope, ast.Load()), self._func.value, ast.Store())
        ast.fix_missing_locations(assign_ast)

    def _sync_assign_targets_to_ast(self):
        """Sync targets of ast.Assign from self._targets when NodeType is CallCell, CallPrimitive or CallMethod."""
        if self._ast_node is None:
            return
        assign_ast = self._ast_node
        if not isinstance(assign_ast, ast.Assign):
            raise TypeError("assign_ast should be ast.Assign, got: ", type(assign_ast))
        # update targets
        targets_ast = assign_ast.targets
        if isinstance(targets_ast[0], ast.Tuple) and len(self._targets) != len(targets_ast[0].elts):
            raise RuntimeError("self._targets should have the same length as targets_ast's elts")
        if not isinstance(targets_ast[0], ast.Tuple) and len(self._targets) != len(targets_ast):
            raise RuntimeError("self._targets should have targets_ast same length")
        for i in range(0, len(self._targets)):
            target = self._targets[i]
            target_ast = targets_ast[0]
            if isinstance(target_ast, ast.Name):
                target_ast.id = target.value
            elif isinstance(target_ast, ast.Tuple):
                if not isinstance(target_ast.elts[i], ast.Name):
                    raise TypeError("target should be ast.Name, got:", type(target_ast.elts[i]))
                target_ast.elts[i].id = target.value
            else:
                raise TypeError("target_ast should be ast.Name or ast.Tuple, got: ", type(target_ast))
            target_ast.id = target.value
        ast.fix_missing_locations(assign_ast)

    def _sync_call_cell_args_to_ast(self):
        """Sync args of ast.Cell of ast.Assign from self._normalized_args when NodeType is CallCell or CallPrimitive."""
        if self._ast_node is None:
            return
        assign_ast = self._ast_node
        if not isinstance(assign_ast, ast.Assign):
            raise TypeError("assign_ast should be ast.Assign, got: ", type(assign_ast))
        assign_value = assign_ast.value
        if not isinstance(assign_value, ast.Call):
            return
        keywords_ast = assign_value.keywords
        args_ast = assign_value.args
        if len(self._normalized_args_keys) != (len(keywords_ast) + len(args_ast)):
            raise RuntimeError("ast keywords plus args len is not equal to self._normalized_args value")

        for arg_index in range(self._args_num):
            arg_ast = args_ast[arg_index]
            AstModifier.update_arg_value(self._normalized_args.get(self._normalized_args_keys[arg_index]), arg_ast)

        # the order of kwargs may not the same as that in keywords_ast
        keyword_map_index = {}
        for index, keyword_ast in enumerate(keywords_ast):
            keyword_map_index[keyword_ast.arg] = index
        for keyword_index in range(self._kwargs_num):
            key = self._normalized_args_keys[keyword_index + self._args_num]
            AstModifier.update_arg_value(self._normalized_args.get(key),
                                         keywords_ast[keyword_map_index.get(key)].value)

    def _sync_call_pass_through_method_args_to_ast(self, assign_value):
        """
        Sync args of PASS_THROUGH_METHOD type ast.Cell of ast.Assign from self._normalized_args when NodeType is
        CallMethod.
        """
        if isinstance(assign_value, ast.Name):
            if len(self._normalized_args_keys) != 1:
                raise RuntimeError("self._normalized_args_keys should have 1 elements")
            arg = self._normalized_args.get(self._normalized_args_keys[0])
            if arg.type != ValueType.NamingValue:
                raise RuntimeError("arg.type should equal to ValueType.NamingValue")
            if arg.scope != "":
                raise RuntimeError("arg.scope should be empty")
            assign_value.id = arg.value
        elif isinstance(assign_value, ast.Attribute):
            if len(self._normalized_args_keys) != 1:
                raise RuntimeError("self._normalized_args_keys should have 1 elements")
            arg = self._normalized_args.get(self._normalized_args_keys[0])
            if arg.type != ValueType.NamingValue:
                raise RuntimeError("arg.type should equal to ValueType.NamingValue")
            assign_value.attr = arg.value
            assign_value_value = assign_value.value
            if not isinstance(assign_value_value, ast.Name):
                raise RuntimeError("Only support ast.Name as value of attribute ", type(assign_value_value))
            assign_value_value.id = arg.scope
        else:
            if len(self._normalized_args_keys) != 1:
                raise RuntimeError("self._normalized_args_keys should have 1 elements")
            arg = self._normalized_args.get(self._normalized_args_keys[0])
            if arg.type not in (ValueType.IntValue, ValueType.FloatValue, ValueType.StringValue):
                raise RuntimeError("arg should be an IntValue, FloatValue or StringValue")
            if arg.scope != "":
                raise RuntimeError("arg.scope should be empty")
            assign_value.value = arg.value

    def _sync_call_method_args_to_ast(self):
        """Sync args of ast.Cell of ast.Assign from self._normalized_args when NodeType is CallMethod."""
        if self._ast_node is None:
            return
        assign_ast = self._ast_node
        if not isinstance(assign_ast, ast.Assign):
            raise TypeError("assign_ast should be ast.Assign, got: ", type(assign_ast))
        assign_value = assign_ast.value
        if self._func == PASS_THROUGH_METHOD:
            self._sync_call_pass_through_method_args_to_ast(assign_value)
        elif self._func.value == "tuple":
            tuple_ast: ast.Tuple = assign_value
            if len(self._normalized_args_keys) != len(tuple_ast.elts):
                raise RuntimeError("size of self._normalized_args_keys should be equal to size of elements of tuple")
            for index, elt in enumerate(tuple_ast.elts):
                scoped_value: ScopedValue = self._normalized_args.get(self._normalized_args_keys[index])
                if isinstance(elt, ast.Constant):
                    elt.value = scoped_value.value
                elif isinstance(elt, (ast.Str, ast.Bytes)):
                    elt.s = scoped_value.value
                elif isinstance(elt, ast.Num):
                    elt.n = scoped_value.value
                elif isinstance(elt, ast.Name):
                    elt.id = scoped_value.value
                elif isinstance(elt, ast.Attribute) and isinstance(elt.value, ast.Name):
                    elt.value.id = scoped_value.scope
                    elt.value = scoped_value.value
                else:
                    raise RuntimeError("Only support constant or symbol in tuple now")
        else:
            raise RuntimeError("Only support pass_through or tuple method as call_method now, ", self._func.value)

    def _sync_return_node_to_ast(self):
        """Sync return value of ast.Return from self._normalized_args when NodeType is Output."""
        if self._ast_node is None:
            return
        return_ast = self._ast_node
        if not isinstance(return_ast, ast.Return):
            raise TypeError("return_ast should be ast.Return, got: ", type(return_ast))
        # update args
        return_value_ast = return_ast.value
        if isinstance(return_value_ast, ast.Name):
            if len(self._normalized_args_keys) != 1:
                raise RuntimeError("self._normalized_args_keys should have 1 elements")
            return_value_ast.id = self._normalized_args.get(self._normalized_args_keys[0]).value
        elif isinstance(return_value_ast, ast.Tuple):
            elements = return_value_ast.elts
            if len(self._normalized_args.values()) != len(elements):
                raise RuntimeError("self._normalized_args.values() should have elements same length")
            for elt_index, elt in enumerate(elements):
                if not isinstance(elt, ast.Name):
                    raise RuntimeError("Only support ast.Name as return value: ", elt)
                arg = self._normalized_args.get(self._normalized_args_keys[elt_index])
                if not isinstance(arg, ScopedValue):
                    raise TypeError("arg should be ScopedValue, got: ", type(arg))
                elt.id = arg.value
        else:
            raise RuntimeError("Unsupported return value type: ", return_value_ast)
        ast.fix_missing_locations(return_ast)

    def _sync_mathops_node_args_to_ast(self):
        """
        Sync values from self._normalized_args to the ast node for mathematical operations.
        """
        if self._ast_node is None:
            return
        if not isinstance(self._ast_node, ast.Assign):
            raise TypeError(f"type of node should be ast.Assign, but got {type(self._ast_node)}")
        mathops_node = self._ast_node.value
        if isinstance(mathops_node, ast.BinOp):
            left = mathops_node.left
            right = mathops_node.right
            AstModifier.update_arg_value(self._normalized_args.get(self._normalized_args_keys[0]), left)
            AstModifier.update_arg_value(self._normalized_args.get(self._normalized_args_keys[1]), right)
        elif isinstance(mathops_node, ast.UnaryOp):
            operand = mathops_node.operand
            AstModifier.update_arg_value(self._normalized_args.get(self._normalized_args_keys[0]), operand)
        elif isinstance(mathops_node, ast.BoolOp):
            values = mathops_node.values
            for arg_index in range(self._args_num):
                arg_value = self._normalized_args.get(self._normalized_args_keys[arg_index])
                AstModifier.update_arg_value(arg_value, values[arg_index])
        elif isinstance(mathops_node, ast.Compare):
            left = mathops_node.left
            AstModifier.update_arg_value(self._normalized_args.get(self._normalized_args_keys[0]), left)
            comparators = mathops_node.comparators
            for arg_index in range(1, self._args_num):
                arg_value = self._normalized_args.get(self._normalized_args_keys[arg_index])
                AstModifier.update_arg_value(arg_value, comparators[arg_index - 1])
        else:
            raise TypeError("The type of 'mathops_node' must be one of (ast.BinOp, ast.UnaryOp, "
                            "ast.BoolOp, ast.Compare), but got ", type(mathops_node))

    def _sync_arg(self):
        """Sync _normalized_args to corresponding ast node when updated."""
        if self._node_type in (NodeType.CallCell, NodeType.CallPrimitive, NodeType.Tree,\
                               NodeType.CellContainer, NodeType.CallFunction):
            self._sync_call_cell_args_to_ast()
        elif self._node_type == NodeType.Output:
            self._sync_return_node_to_ast()
        elif self._node_type == NodeType.CallMethod:
            self._sync_call_method_args_to_ast()
        elif self._node_type == NodeType.MathOps:
            self._sync_mathops_node_args_to_ast()


class TreeNode(Node):
    """Tree type Node who holds a handler of SymbolTree."""

    def __init__(self, tree, ast_node: ast.AST, targets: [ScopedValue], func: ScopedValue,
                 args: [ScopedValue], kwargs: {str: ScopedValue}, name: str, instance):
        """
        Constructor of Node. Rewrite recommend to invoking class method of Node to instantiate an instance of Node such
        as `create_call_buildin_op`, `create_call_method`, `create_python_node`, `create_input_node` and
        `create_output_node`, etc. rather than invoking constructor of Node directly.

        Args:
            tree: An instance of SymbolTree represents a handler of sub-symbol-tree.
            ast_node (ast.AST): An instance of ast.AST represents corresponding node in ast.
            targets (list[ScopedValue]): A list of instance of ScopedValue. See detail in docstring of Node class.
            func ([ScopedValue, optional]): An instance of ScopedValue. See detail in docstring of Node class.
            args (list[ScopedValue]): A list of instance of ScopedValue. See detail in docstring of Node class.
            kwargs (dict{str: ScopedValue}): A list of instance of ScopedValue. See detail in docstring of Node class.
            name (str): A string represents name of node. Name of node will be unique when inserted into SymbolTree.
                Name of node also used as field name in network class.
            instance: Object in network corresponding to this node.
        """
        if isinstance(func, str):
            func = ScopedValue.create_naming_value(func)
        super().__init__(NodeType.Tree, ast_node, targets, func, args, kwargs, name, instance)
        self.symbol_tree = tree

    @classmethod
    def create_tree_node(cls, tree, ast_node: ast.AST, targets: Union[ScopedValue, str], func: Union[ScopedValue, str],
                         args: [ScopedValue], kwargs: {str: ScopedValue}, name: str = "", instance=None):
        """
        Class method of TreeNode. Instantiate an instance of node whose type is Tree. A Tree node represents an invoking
        to sub-network.

        Args:
            tree: An instance of SymbolTree represents a handler of sub-symbol-tree.
            ast_node (ast.AST): An instance of ast.AST represents corresponding node in ast.
            targets (list[ScopedValue]): A list of instance of ScopedValue. See detail in docstring of Node class.
            func ([ScopedValue, optional]): An instance of ScopedValue. See detail in docstring of Node class.
            args (list[ScopedValue]): A list of instance of ScopedValue. See detail in docstring of Node class.
            kwargs (dict{str: ScopedValue}): A list of instance of ScopedValue. See detail in docstring of Node class.
            name (str): A string represents name of node. Name of node will be unique when inserted into SymbolTree.
                Name of node also used as field name in network class.
            instance: Object in network corresponding to this node.
        """

        if not isinstance(instance, Cell):
            raise ValueError("Argument instance should be a Cell: ", type(instance))
        non_custom_args = Node._handle_custom_obj_in_args(args)
        non_custom_kwargs = Node._handle_custom_obj_in_kwargs(kwargs)
        new_targets = Node._handle_targets(targets)
        if isinstance(func, str):
            func = ScopedValue.create_naming_value(func)
        if ast_node is None:
            ast_node = AstModifier.create_call_assign(new_targets, func, non_custom_args, non_custom_kwargs)
        return cls(tree, ast_node, new_targets, func, args, kwargs, name, instance)


class CellContainer(Node):
    """ Container for saving cell-objects node. """
    class _Visitor():
        """ A iterator of CellContainer nodes. """
        def __init__(self, cellcontainer):
            self._cellcontainer = cellcontainer

        def __len__(self):
            """ Get the number of nodes. """
            return self._cellcontainer.node_count

        def __iter__(self):
            """Create an iterator over the CellContainer."""
            count = len(self._cellcontainer.node_list)
            i = 0
            while i < count:
                curr = self._cellcontainer.node_list[i]
                if curr.valid:
                    yield curr
                i += 1

    def __init__(self, ast_node: ast.AST, targets: [ScopedValue], func: ScopedValue,
                 args: [ScopedValue], kwargs: {str: ScopedValue}, name: str, instance):
        """Constructor of CellContainer.

        Args:
            ast_node (ast.AST): An instance of ast.AST represents corresponding node in ast.
            targets (list[ScopedValue]): A list of instance of ScopedValue. See detail in docstring of Node class.
            func ([ScopedValue, optional]): An instance of ScopedValue. See detail in docstring of Node class.
            args (list[ScopedValue]): A list of instance of ScopedValue. See detail in docstring of Node class.
            kwargs (dict{str: ScopedValue}): A list of instance of ScopedValue. See detail in docstring of Node class.
            name (str): A string represents name of node. Name of node will be unique when inserted into SymbolTree.
                Name of node also used as field name in network class.
            instance: Object in network corresponding to this node.
        """
        if isinstance(func, str):
            func = ScopedValue.create_naming_value(func)
        super().__init__(NodeType.CellContainer, ast_node, targets, func, args, kwargs, name, instance)
        self._node_list = list()
        self._node_count = 0

    @property
    def node_count(self):
        """Number of nodes."""
        return len(self._node_list)

    @property
    def node_list(self):
        """ Get node list. """
        return self._node_list

    def append(self, node):
        """ Append new node to node list. """
        setattr(node, "container", self)
        setattr(node, "valid", True)
        node.set_belong_symbol_tree(self.get_belong_symbol_tree())
        self._node_list.append(node)
        # when creating a cell_container, node instance is already in SequentialCell cell_list
        # so here we need to write a if judgement
        if node.get_instance() not in self.get_instance().cell_list:
            self.get_instance().append(node.get_instance())

    def erase(self, node):
        """Erase node form container."""
        index_node = self.node_list.index(node)
        index_instance = self.get_instance().cell_list.index(node.get_instance())
        if index_node != index_instance:
            raise RuntimeError("In MindSpore Rewrite CellContainer, erasing a node raises index error!!!")
        setattr(node, "valid", False)
        del self.get_instance()[index_node]
        del self._node_list[index_node]

    def insert(self, index, node):
        """Insert node into container"""
        self.node_list.insert(index, node)
        setattr(node, "container", self)
        setattr(node, "valid", True)
        node.set_belong_symbol_tree(self.get_belong_symbol_tree())
        self.get_instance()._insert(index, node.get_instance())

    def nodes(self):
        """ Return a iterator of node."""
        return self._Visitor(self)
