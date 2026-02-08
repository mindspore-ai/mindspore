# This is the Python adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
#
# Copyright 2020-2025 Huawei Technologies Co., Ltd
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
"""The module of parser python object, called by c++."""

from __future__ import absolute_import
import os
import ast
import re
import hashlib
import inspect
import types
from collections import namedtuple
from typing import NamedTuple
from textwrap import dedent
import builtins
import numpy

import asttokens
import astunparse

from mindspore import Tensor, CSRTensor, COOTensor, RowTensor
from mindspore import log as logger
from mindspore import nn
from mindspore import ops
from mindspore import context
from mindspore import tensor
from mindspore.graph.api import _JitExecutor
from mindspore.common import dtype as mstype
from mindspore.common.parameter import Parameter
from mindspore.common import mutable
from mindspore._extends.ast_checker import AstChecker
from mindspore.runtime.event import Event
from mindspore.runtime import StreamCtx, StreamLimitCtx
from .namespace import Namespace, ModuleNamespace, ClosureNamespace, ClassMemberNamespace
from .resources import (parse_object_map, parse_augassign_object_map, ops_symbol_map, convert_object_map,
                        convert_class_to_function_map, trope_ns)
from .resources import SYMBOL_UNDEFINE, constant_fold_functions
from .jit_fallback_modules.check_utils import third_party_checker
from mindspore.graph.api import _convert_python_data

# Define resolve type
RESOLVE_TYPE_NONE = 0                   # Resolve None.
RESOLVE_TYPE_FUNCTION = 1               # Resolve function.
RESOLVE_TYPE_METHOD = 2                 # Resolve class method.
RESOLVE_TYPE_CLASS_TYPE = 3             # Resolve class type.
RESOLVE_TYPE_CLASS_INSTANCE = 4         # Resolve the class instance of common class.
RESOLVE_TYPE_NAMESPACE_INSTANCE = 5     # Resolve the namespace instance.
RESOLVE_TYPE_NUMPY_INT_NUMBER = 6       # Resolve numpy int number.
RESOLVE_TYPE_NUMPY_FLOAT_NUMBER = 7     # Resolve numpy float number.
RESOLVE_TYPE_NUMPY_BOOL_NUMBER = 8      # Resolve numpy bool number.
RESOLVE_TYPE_TUPLE = 9                  # Resolve builtin tuple type.
RESOLVE_TYPE_LIST = 10                  # Resolve builtin list type.
RESOLVE_TYPE_BUILTIN_METHOD = 11        # Resolve builtin type.
RESOLVE_TYPE_EVENT = 12
RESOLVE_TYPE_INVALID = 0xFF             # Resolve invalid.

# Define the class instance detail type
# When the type is RESOLVE_TYPE_CLASS_INSTANCE
CLASS_INSTANCE_TYPE_CELL = 0            # Class instance type is Cell
CLASS_INSTANCE_TYPE_PRIMITIVE = 1       # Class instance type is Primitive
CLASS_INSTANCE_TYPE_NUMPY_ARRAY = 2     # Class instance type is Numpy Array
CLASS_INSTANCE_TYPE_TENSOR = 3          # Class instance type is Tensor
CLASS_INSTANCE_TYPE_INVALID = 0xFF

# Ast main type
AST_MAIN_TYPE_STMT = 0                  # ast.Stmt
AST_MAIN_TYPE_EXPR = 1                  # ast.Expr
AST_MAIN_TYPE_SLICE = 2                 # ast.Slice
AST_MAIN_TYPE_UNKNOWN = 0xFF            # unknown

# Ast sub type
AST_SUB_TYPE_AND = 3                   # ast.And
AST_SUB_TYPE_OR = 4                    # ast.Or
AST_SUB_TYPE_NAME = 5                  # ast.Name
AST_SUB_TYPE_TUPLE = 6                 # ast.Tuple
AST_SUB_TYPE_LIST = 7                  # ast.List
AST_SUB_TYPE_SUBSCRIPT = 8             # ast.Subscript
AST_SUB_TYPE_STARRED = 9               # ast.Starred
AST_SUB_TYPE_ATTRIBUTE = 10            # ast.Attribute
AST_SUB_TYPE_DICT = 11                 # ast.Dict
AST_SUB_TYPE_UNKNOWN = 0xFF            # unknown

# Syntax support
SYNTAX_SUPPORTED = 0                   # Supported syntax
SYNTAX_UNSUPPORTED_INTERNAL_TYPE = 1   # Unsupported internal type
SYNTAX_UNSUPPORTED_EXTERNAL_TYPE = 2   # Unsupported external type
SYNTAX_HYBRID_TYPE = 3                 # Hybrid type
SYNTAX_UNSUPPORTED_NAMESPACE = 4       # Unsupported namespace

# Process expr statement white list
# Add as needed, eg: "clear", "extend", "insert", "remove", "reverse"
parse_expr_statement_white_list = (
    "append", "insert", "clear", "reverse", "extend", "update", "register_hook",
)

_need_reorder_methods = (
    "register_hook",
)

_builtin_function_or_method_type = type(abs)

# Unsupported python builtin type in graph mode.
_unsupported_python_builtin_type = (
    set, dict, slice, complex, reversed, type,
)


_global_params = {}


def create_slice_obj(start, end, step):
    """Create slice object"""
    return slice(start, end, step)


def parse_cb(func, parse_method=None):
    """Implements the function of parse."""
    return Parser(func, parse_method)


def get_attr_from_object(obj, attr_name=None):
    """
    Get attr from object.

    Args:
        obj(Object): Instance of class or module.
        attr_name(str): Attribute name to check.

    Returns:
        Object, obj's attr.
    """

    if obj is not None and attr_name is not None and hasattr(obj, attr_name):
        return getattr(obj, attr_name)
    return None


def check_attr_is_property(obj, attr_name):
    """
    Check if the attribute is decorated by @property.

    Args:
        obj(Object): Instance of a class.
        attr_name(str): Attribute name to check.

    Returns:
        obj(bool): If the attribute is decorated by @property.
    """
    logger.debug(f"attr_name:{attr_name}")
    logger.debug(f"obj.__class__.__dict__.keys():{obj.__class__.__dict__.keys()}")
    if attr_name in obj.__class__.__dict__.keys() and isinstance(obj.__class__.__dict__[attr_name], property):
        attr_obj = obj.__class__.__dict__[attr_name]
        if (hasattr(attr_obj, 'fget')) and hasattr(attr_obj.fget, '__code__'):
            logger.debug(f'The attribute {attr_name} is decorated by @property.')
            return True
    return False


def get_parse_method_of_class(obj, parse_method=None):
    """
    Get parse method of class.

    Args:
        obj(Object): Instance of class.
        parse_method(str): Save the method name. Cell object has default method named 'construct'.

    Returns:
        Function, obj's method.
    """

    method_name = None
    if parse_method is not None:
        method_name = parse_method
    elif isinstance(obj, nn.Cell):
        method_name = "construct"

    return get_attr_from_object(obj, method_name)


def get_bprop_method_of_class(obj, parse_method=None):
    """
    Get bprop method of class.

    Args:
        obj (Object): Instance of class.
        parse_method(str): Save the method name. Cell object has default method named 'bprop'.

    Returns:
        Function, obj's method.
    """

    if isinstance(obj, nn.Cell):
        method_name = "bprop"
        return get_attr_from_object(obj, method_name)
    return None


def resolve_symbol(namespace, symbol):
    """
    Resolve a symbol.

    Note:
        Can't get function when use closure function. So save the fn on namespace.

    Args:
        namespace (Object): Symbol's namespace.
        symbol (str): Need resolve symbol.

    Returns:
        Object, resolve result of symbol.
    """
    # All exceptions need to be caught in this function
    try:
        resolve_ = namespace[symbol]

        # The list and dict is not hashable, it can not be key for the map, just return the result
        if isinstance(resolve_, (tuple, list, dict)):
            return resolve_
        if hasattr(resolve_, "__self__") and isinstance(resolve_.__self__, (tuple, list, dict)):
            return resolve_
        if getattr(resolve_, "__hash__") is None:
            return resolve_

        # If need trope the obj
        if resolve_ in convert_object_map:
            resolve_ = convert_object_map.get(resolve_)
            logger.debug("Convert resolve: %r", resolve_)
    except Exception as e:
        if isinstance(e, NotImplementedError):
            raise e
        resolve_ = mstype._null
        logger.debug("Resolve exception occurred, value: %r", e)
        logger.debug("Resolve type is invalid, namespace: %s, symbol: %s",
                     namespace.__str__(), symbol)

    if isinstance(resolve_, _JitExecutor):
        logger.debug("Resolve class _JitExecutor, resolve fn instead.")
        resolve_ = resolve_.fn
    logger.debug("Found '%s' in %s, resolved: %s / %s", symbol, namespace, resolve_, type(resolve_))
    return resolve_


def generate_scope(obj):
    """Generate the scope for every cell object in the network."""
    if isinstance(obj, nn.Cell):
        obj.generate_scope()


def get_scope_name(obj):
    """Returns the scope of a cell object in one network."""
    if isinstance(obj, nn.Cell):
        return obj.get_scope()
    return None


def get_type(obj):
    """Returns the type string of input object"""
    return type(obj)


def get_object_key(obj):
    """Return the function key: module + name."""
    obj_key = ""
    if hasattr(obj, "__name__"):
        if hasattr(obj, "cell_init_args"):
            obj_key = "%s_ID" % (str(obj.__class__.__name__) + str(obj.__name__) + obj.cell_init_args)
        obj_id = "%s_ID%d" % (str(obj.__class__.__name__) + str(obj.__name__), id(obj))
    else:
        # `<class 'xxxxxxx'>`
        # -> `xxxxxxx`
        tag = str(obj.__class__)[8:-2]
        if hasattr(obj, "cell_init_args"):
            obj_key = "%s_ID" % (tag + obj.cell_init_args)
        obj_id = "%s_ID%d" % (tag, id(obj))
    logger.debug("obj_key: %s, obj_id: %s", obj_key, obj_id)

    # Method has same id of different instance
    if isinstance(obj, types.MethodType) or \
        (isinstance(obj, types.BuiltinMethodType) and obj.__qualname__.split('.')[0] == Tensor.__name__):
        method_instance = obj.__self__
        instance_id = "%s_ID%d" % (str(method_instance.__class__.__name__), id(method_instance))
        if isinstance(method_instance, (tuple, list, dict)):
            obj_id = instance_id + obj_id
        else:
            obj_id = instance_id + obj_id + str(obj.__hash__())
    return obj_id, obj_key


def is_class_member_of_self(node):
    """Check the attr is class member variable."""
    type_ = node.__class__.__name__
    if type_ == "Attribute":
        if not hasattr(node.value, "id"):
            return False
        id_ = node.value.id
        if id_ == "self":
            return True
    return False


def is_class_member_recursive(node):
    """Check the attr is class member variable resurcively."""
    type_ = node.__class__.__name__
    if type_ == "Attribute":
        if hasattr(node.value, "value"):
            return is_class_member_recursive(node.value)
        if not hasattr(node.value, "id"):
            return False
        id_ = node.value.id
        if id_ == "self":
            return True
    return False


def get_obj_id(obj):
    """Get the obj id."""
    return str(id(obj))


def is_lambda_function(obj):
    """Determine whether is a lambda function."""
    if isinstance(obj, types.FunctionType):
        source_code = inspect.getsource(obj)
        return "lambda" in source_code and "<function" in str(obj) and "<lambda>" in str(obj)
    return False


def get_obj_type(obj):
    """Get the obj type."""
    logger.debug("Get object type: %r", obj)
    obj_type = RESOLVE_TYPE_INVALID
    if obj is None:
        obj_type = RESOLVE_TYPE_NONE
    elif isinstance(obj, types.FunctionType) or type(obj).__name__ == 'cython_function_or_method':
        obj_type = RESOLVE_TYPE_FUNCTION
    elif isinstance(obj, types.MethodType):
        obj_type = RESOLVE_TYPE_METHOD
    elif isinstance(obj, type):
        obj_type = RESOLVE_TYPE_CLASS_TYPE
    elif isinstance(obj, Namespace):
        obj_type = RESOLVE_TYPE_NAMESPACE_INSTANCE
    elif isinstance(obj, tuple):
        obj_type = RESOLVE_TYPE_TUPLE
    elif isinstance(obj, list):
        obj_type = RESOLVE_TYPE_LIST
    elif _is_class_instance(obj):
        obj_type = RESOLVE_TYPE_CLASS_INSTANCE
    elif _is_numpy_int_number(obj):
        obj_type = RESOLVE_TYPE_NUMPY_INT_NUMBER
    elif _is_numpy_float_number(obj):
        obj_type = RESOLVE_TYPE_NUMPY_FLOAT_NUMBER
    elif _is_numpy_bool_number(obj):
        obj_type = RESOLVE_TYPE_NUMPY_BOOL_NUMBER
    elif isinstance(obj, types.BuiltinMethodType) and obj.__qualname__.split('.')[0] == Tensor.__name__:
        obj_type = RESOLVE_TYPE_BUILTIN_METHOD
    elif isinstance(obj, Event):
        obj_type = RESOLVE_TYPE_EVENT
    else:
        obj_type = RESOLVE_TYPE_INVALID
    return obj_type


def check_obj_bool(obj):
    """Check if the type of the current object is bool."""
    logger.debug("Check if the type of the current object(%r) is bool: %r", obj, bool(obj))
    return bool(obj)


def get_class_instance_type(obj):
    """Get the class instance detail type."""
    # Check the obj type
    logger.debug("Get the class type(%r)", obj)
    if isinstance(obj, nn.Cell):
        return CLASS_INSTANCE_TYPE_CELL
    if isinstance(obj, ops.Primitive):
        return CLASS_INSTANCE_TYPE_PRIMITIVE
    if isinstance(obj, numpy.ndarray):
        return CLASS_INSTANCE_TYPE_NUMPY_ARRAY
    return CLASS_INSTANCE_TYPE_INVALID


def _is_ms_class(obj):
    """Check if obj is ms_class object."""
    return hasattr(obj, '__ms_class__')


def _is_class_instance(obj):
    """Confirm the obj is class instance."""
    return isinstance(obj, (nn.Cell, ops.Primitive)) or _is_ms_class(obj) or hasattr(obj, '__parse_method__')


def _is_numpy_int_number(obj):
    """Confirm the obj is numpy int number."""
    return isinstance(obj, (numpy.int8, numpy.int16, numpy.int64, numpy.uint8, numpy.uint16, numpy.uint64))


def _is_numpy_float_number(obj):
    """Confirm the obj is numpy float number."""
    return isinstance(obj, (numpy.float16, numpy.float32, numpy.float64))


def _is_numpy_bool_number(obj):
    """Confirm the obj is numpy bool number."""
    return isinstance(obj, numpy.bool_)


def _convert_tuple_to_args_kwargs(params):
    """Convert tuple to args and kwargs."""
    args = tuple()
    kwargs = {}
    for param in params:
        if isinstance(param, dict):
            kwargs.update(param)
        else:
            args += (param,)
    return (args, kwargs)


def is_supported_create_instance_type(cls_type):
    """Check if cls_type is a supported instance type."""
    return issubclass(cls_type, (nn.Cell, ops.Primitive, ops.GradOperation)) or _is_ms_class(cls_type)


def create_instance(cls_type, params=None):
    """Create python instance."""
    if not isinstance(cls_type, type):
        logger.warning(f"create_instance(), cls_type is not a type, cls_type: {cls_type}")
        return None

    # Check the type, now only support nn.Cell and Primitive.
    obj = None
    if is_supported_create_instance_type(cls_type):
        # Check arguments, only support *args or **kwargs.
        if params is None:
            obj = cls_type()
        elif isinstance(params, tuple):
            args, kwargs = _convert_tuple_to_args_kwargs(params)
            logger.debug(f"create_instance(), args: {args}, kwargs: {kwargs}")
            if args and kwargs:
                obj = cls_type(*args, **kwargs)
            elif args:
                obj = cls_type(*args)
            elif kwargs:
                obj = cls_type(**kwargs)
        # If invalid parameters.
        if obj is None:
            raise ValueError(f"When call 'create_instance', the parameter should be *args or **kwargs, "
                             f"but got {params.__class__.__name__}, params: {params}")
    return obj


def convert_class_to_function(cls_str, cls_obj):
    """Convert class to function."""
    if issubclass(cls_obj, (Parameter, ops.MultitypeFuncGraph)):
        raise ValueError(f"Failed to compile in GRAPH_MODE because creating {cls_str} instances is not "
                         f"supported in 'construct' or @jit decorated function. Try to create {cls_str} "
                         f"instances external such as initialized in the method '__init__' before assigning. "
                         f"For more details, please refer to "
                         f"https://www.mindspore.cn/tutorials/zh-CN/master/compile/static_graph.html \n")
    return convert_class_to_function_map.get(cls_str)


def python_isinstance(x, cmp_type):
    """Python isinstance function."""
    # Convert _c_expression tensor to python tensor.
    x = _convert_python_data(x)
    return isinstance(x, cmp_type)


def ms_isinstance(x, cmp_type):
    """Isinstance for ms type."""
    pytype_to_mstype = {
        bool: mstype.Bool,
        int: mstype.Int,
        float: mstype.Float,
        str: mstype.String,
        list: mstype.List,
        tuple: mstype.Tuple,
        dict: mstype.Dict,
        Tensor: mstype.TensorType,
        Parameter: mstype.RefType,
        slice: mstype.Slice,
    }
    if cmp_type not in pytype_to_mstype:
        return False
    if isinstance(x, mstype.Bool) and cmp_type == int:
        return True
    return isinstance(x, pytype_to_mstype.get(cmp_type))


def is_cell_list(obj):
    """Check if obj is nn.CellList"""
    return isinstance(obj, nn.CellList)


def convert_cell_list_to_sequence(obj):
    """Convert nn.CellList to sequence."""
    if not hasattr(obj, "__cell_as_list__"):
        raise TypeError(f"Obj should be nn.CellList, but got {obj}")
    if not hasattr(obj, "_cells"):
        raise AttributeError("nn.CellList is missing _cells property.")
    cells = getattr(obj, "_cells")
    return list(cells.values())


def get_obj_from_sequence(obj, index):
    """Implement `tuple_getitem`."""
    if not isinstance(obj, (tuple, list)):
        raise TypeError(f"Should not get item from a object that not sequence type, obj: {obj}")
    # Not check index out of range by self.
    return obj[index]


def get_module_namespace(obj):
    """Get the module's namespace."""
    logger.debug("get module namespace, module: %r", obj)
    mod_namespace = None
    if isinstance(obj, types.ModuleType):
        # When the obj is mindspore.ops, do not add built-in functions to avoid incorrect matches.
        if obj.__name__ == "mindspore.ops":
            mod_namespace = ModuleNamespace(obj.__name__, False)
        else:
            mod_namespace = ModuleNamespace(obj.__name__)
    else:
        logger.warning("Module(%r) is invalid, get namespace failure!", obj)
    return mod_namespace


def get_class_member_namespace_symbol(obj):
    """Get obj class member type."""
    logger.debug("get class instance namespace, object: %r", obj)
    class_namespace = ClassMemberNamespace(obj)
    logger.debug("class namespace: %r", class_namespace)
    return class_namespace


def get_obj_defined_from_obj_type(obj_type):
    """Get the class defined from object type which is in BuiltInMap."""
    logger.debug("get the object type: %r", obj_type)

    def func():
        pass

    obj_type_defined_map = {
        "Tensor": Tensor,
        "RowTensor": RowTensor,
        "COOTensor": COOTensor,
        "CSRTensor": CSRTensor,
        "Parameter": Parameter,
        "String": "",
        "Function": func,
        "Int": int,
        "Float": float,
        "UInt": int,
        "Bool": bool,
        "List": list,
        "Tuple": tuple,
        "Dictionary": dict,
        "NamedTuple": NamedTuple,
    }

    return obj_type_defined_map.get(obj_type)


def is_class_type(cls):
    """Check if cls is a class type."""
    return isinstance(cls, type)


def get_ms_class_name(cls):
    """Get the name of the class instance decorated with jit_class."""
    if isinstance(cls, type):
        return cls.__name__
    return cls.__class__.__name__


def convert_to_ms_tensor(data):
    """Convert C++ tensor to mindspore tensor."""
    return Tensor(data)


def convert_to_ms_csrtensor(data):
    """Convert C++ csrtensor to mindspore csrtensor."""
    return CSRTensor(csr_tensor=data)


def convert_to_ms_cootensor(data):
    """Convert C++ cootensor to mindspore cootensor."""
    return COOTensor(coo_tensor=data)


def convert_to_namedtuple(type_name, key_sequeue, value_sequeue):
    """Convert C++ namedtuple to python object namedtuple."""
    logger.debug(f"type_name: {type_name}, key_sequeue: {key_sequeue}, value_sequeue: {value_sequeue}")
    return namedtuple(type_name, [*key_sequeue])(*value_sequeue)


def get_object_description(obj, fname, fline):
    """Return method or funcition description for error report, include location, class name, etc."""
    if isinstance(obj, types.MethodType):
        obj_cls = obj.__self__.__class__
        class_name = f"{obj_cls.__module__}.{obj_cls.__qualname__}"
        cls_fname = inspect.getfile(obj_cls)
        _, cls_fline = inspect.getsourcelines(obj_cls)
        class_loc = f"{cls_fname}:{cls_fline}"
        return f"bound method '{obj.__name__}' at {fname}:{fline} of <{class_name} at {class_loc} object>"
    if isinstance(obj, types.FunctionType):
        return f"function '{obj.__name__}' at {fname}:{fline}"
    if isinstance(obj, ast.FunctionDef):
        return f"function '{obj.name}' at {fname}:{fline}"
    if isinstance(obj, ast.Attribute):
        return "attribute "
    return str(obj)


def expand_expr_statement(node):
    """
    Process the expr statement and expand it.

    Returns:
        (False,)/(True, expr.value, target, bool)/(True, expr.value).
    """
    if isinstance(node, ast.Expr):
        expr_value = node.value
        if isinstance(expr_value, ast.Call):
            func = expr_value.func
            if isinstance(func, ast.Attribute) and \
                    hasattr(func, "attr") and \
                    hasattr(func, "value"):
                method = func.attr
                target = func.value
                if method in parse_expr_statement_white_list:
                    logger.debug("Expand expr, target:%s, method:%s", target, method)
                    return True, expr_value, target, method in _need_reorder_methods
        if not AstChecker.check_type(expr_value, "ast.Str"):
            return True, expr_value
    return (False,)


def check_event_record_wait(node):
    """
    Check is record or wait.

    Returns:
        record or wait target or None.
    """
    if isinstance(node, ast.Expr):
        expr_value = node.value
        if isinstance(expr_value, ast.Call):
            func = expr_value.func
            if isinstance(func, ast.Attribute) and \
                    hasattr(func, "attr") and \
                    hasattr(func, "value"):
                method = func.attr
                target = func.value
                logger.debug("Expand expr, target:%s, method:%s", target, method)
                if method in ("record", "wait"):
                    return target
    return None


def get_ast_namespace_symbol(obj):
    """Get obj type and namespace and symbol."""
    # Get symbol from object map.
    ops_info = parse_object_map.get(type(obj), SYMBOL_UNDEFINE)
    logger.debug("ops info: %r", ops_info)
    return ops_info


def get_ast_augassign_namespace_symbol(obj):
    """Get obj type and namespace and symbol."""
    # Get symbol from object map.
    ops_info = parse_augassign_object_map.get(type(obj), SYMBOL_UNDEFINE)
    logger.debug("ops info: %r", ops_info)
    return ops_info


def get_operation_symbol(obj):
    """Get obj operation symbol."""
    ops_symbol = ops_symbol_map.get(type(obj), SYMBOL_UNDEFINE)
    logger.debug("ops symbol: %s", ops_symbol)
    return ops_symbol


def get_operation_namespace_symbol(var: str):
    """Get operation namespace and symbol."""
    ops_info = (trope_ns, var)
    logger.debug("get operation ops info: %r", ops_info)
    return ops_info


def get_ast_type(node):
    """Get the ast type."""
    ast_type = AST_SUB_TYPE_UNKNOWN
    if isinstance(node, ast.And):
        ast_type = AST_SUB_TYPE_AND
    elif isinstance(node, ast.Or):
        ast_type = AST_SUB_TYPE_OR
    elif isinstance(node, ast.Name):
        ast_type = AST_SUB_TYPE_NAME
    elif isinstance(node, ast.Tuple):
        ast_type = AST_SUB_TYPE_TUPLE
    elif isinstance(node, ast.List):
        ast_type = AST_SUB_TYPE_LIST
    elif isinstance(node, ast.Subscript):
        ast_type = AST_SUB_TYPE_SUBSCRIPT
    elif isinstance(node, ast.Starred):
        ast_type = AST_SUB_TYPE_STARRED
    elif isinstance(node, ast.Attribute):
        ast_type = AST_SUB_TYPE_ATTRIBUTE
    elif isinstance(node, ast.Dict):
        ast_type = AST_SUB_TYPE_DICT
    else:
        ast_type = AST_SUB_TYPE_UNKNOWN
    return ast_type


def get_node_type(node):
    """Process an ast node."""
    method_name = f"{node.__class__.__name__}"
    node_type = [method_name]
    # Judge the ast main type.
    if isinstance(node, ast.stmt):
        node_type.append(AST_MAIN_TYPE_STMT)
    elif isinstance(node, (ast.expr, ast.slice)) or node is None:
        # ast.slice and ast.expr should be expr.
        node_type.append(AST_MAIN_TYPE_EXPR)
    else:
        node_type.append(AST_MAIN_TYPE_UNKNOWN)
    return node_type


def get_args_default_values(node):
    """
    Get the args'default values of parse object.

    Examples:
        - Function:
        func(a, b, *c, d=0, **e)
        - The ast is as below:
        args=arguments(
            args=[arg(a), arg(b)], vararg=arg(c), kwonlyargs=[arg(d)], kw_defaults=[Num(0)], kwarg=arg(e)
        )

        - Function:
        func(a, b, c=1)
        - The ast is as below:
        args=arguments(
            args=[arg(a), arg(b), arg(c)], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[Num(1)]
        )
    """
    defaults = [None] * (len(node.args.args) - len(node.args.defaults))
    defaults = defaults + node.args.defaults
    if node.args.vararg:
        defaults.append(None)
    defaults = defaults + node.args.kw_defaults
    if node.args.kwarg:
        defaults.append(None)
    return defaults


def get_args(node):
    """Get the arg of parse object. The order is [args, vararg, kwonlyargs, kwarg]"""
    args = []
    # Process position args.
    for arg in node.args.args:
        args.append(arg)
    # Process vararg: vararg is append after position.
    if node.args.vararg:
        args.append(node.args.vararg)
    # Process kwonlyargs: kwonlyargs is append after vararg.
    if node.args.kwonlyargs:
        for kwonlyarg in node.args.kwonlyargs:
            args.append(kwonlyarg)
    # Process kwarg: kwarg is append after vararg.
    if node.args.kwarg:
        args.append(node.args.kwarg)
    return args


def get_arg_spec_and_default_values(func):
    """Get the full arg specification and the default arg values of a function"""
    arg_spec = inspect.getfullargspec(func)
    defaults = {}
    args_len = len(arg_spec.args)
    if arg_spec.defaults:
        defaults_len = len(arg_spec.defaults)
        for i in range(defaults_len):
            defaults[arg_spec.args[args_len - i - 1]] = arg_spec.defaults[defaults_len - i - 1]
    if arg_spec.kwonlydefaults:
        for k, v in arg_spec.kwonlydefaults.items():
            defaults[k] = v
    return arg_spec, defaults


def eval_script(exp_str, params):
    """Evaluate a python expression."""
    if not isinstance(params, tuple):
        raise ValueError(f"eval_script(), params is not a tuple, params: {params}")
    if len(params) != 2:
        raise ValueError(f"eval_script(), params tuple length is wrong, params: {params}")

    # Eval function parses the expression argument and evaluates it as a python expression.
    global_params = params[0]
    local_params = params[1]
    try:
        local_params = _convert_python_data(local_params)
        # There are two sources of scripts:
        # 1. The user's original Python script code, which is directly passed back to Python for execution,
        #    and its behavior is guaranteed by the user.
        # 2. Internally provided Python expression code, similar to
        #    `__iternal_sequence_input__[__internal_sequence_index__]`.
        # In addition, MindIR load and export do not involve the use of the `eval_script` function.
        res = eval(exp_str, global_params, local_params)
    except Exception as e:
        error_info = f"When eval '{exp_str}' by using JIT Fallback feature, an error occurred: " + str(e)
        logger.debug(error_info)
        raise e

    return res


def get_script_id_attrs(script):
    """Get the ids for the ast of script"""
    ast_tokens = asttokens.ASTTokens(script, parse=True)
    ast_tree = ast_tokens.tree
    ast_str = astunparse.dump(ast_tree)
    ids = re.findall(r"id='(.+?)'", ast_str)
    id_sets = set(ids)
    pattern = r"Attribute\(\s*value.*?id='(.*?)'.*?attr='(.*?)'.*?\)"
    matches = re.findall(pattern, ast_str, re.DOTALL)
    id_attrs = ["{}.{}".format(match[0], match[1]) for match in matches]
    logger.debug(f'id_attrs: {id_attrs}')
    id_attrs_set = set(id_attrs)
    logger.debug(f'id_attrs_set: {id_attrs_set}')
    res = id_sets.union(id_attrs_set)
    logger.debug(f'res: {res}')
    return res


def generate_lambda_object(script):
    """Generate lambda expression object using script"""
    return eval_script(script, ({}, {}))


def get_global_params():
    """Get the global parameter."""
    logger.debug(f"get global_dict: {_global_params}")
    return _global_params


def get_dtype(name: str):
    """get mstype from name"""
    return get_attr_from_object(mstype, name)


def check_attrs(target_object, func_name: str):
    """Check if attr is overridden."""
    if isinstance(target_object, Tensor):
        return False
    if hasattr(target_object, func_name):
        if not hasattr(target_object.__class__.__base__, func_name):
            if target_object.__class__.__base__ is object:
                return False
            return True
        if getattr(target_object.__class__, func_name) is not getattr(target_object.__class__.__base__, func_name):
            return True
    return False


def check_is_subclass(target_object, parent):
    """Check if target_object is a subclass."""
    if issubclass(target_object.__class__, parent):
        if target_object.__class__ is not parent:
            return True
    return False


def is_from_third_party_library(value):
    """Check if value is from a third-party library."""
    return third_party_checker.is_from_third_party_module(value)


def convert_to_mutable(object):
    return mutable(object)


def get_const_abs(obj):
    """Get absolute value of const object."""
    return abs(obj)


def get_const_round(obj):
    """Get round value of const object."""
    if isinstance(obj, tuple):
        val = obj[0]
        point_num = obj[1]
        return round(val, point_num)
    return round(obj)


def get_const_len(obj):
    """Get the length of const object."""
    return len(obj)


def get_method_info(obj):
    """Get the class name of the object from its method."""
    if not (inspect.ismethod(obj) or 'built-in method' in repr(obj)):
        return None, None
    class_name_and_method_name = obj.__qualname__.split('.')
    return class_name_and_method_name[0], class_name_and_method_name[1]


def can_constant_fold(obj):
    """Check if the obj is the function can be constantly folded."""
    return obj in constant_fold_functions


def hook_wrapper(hook_fn):
    """
    Decorator wrapper for gradient hook functions.
    Handles custom logic when the hook returns None to ensure execution dependencies.

    Args:
        hook_fn (function): The original hook function to be wrapped.

    Returns:
        function: Wrapped inner hook function with dependency handling logic.
    """
    def inner(dout):
        fdout = hook_fn(dout)
        if fdout is None:
            dout = ops.Depend()(dout, fdout)
            return dout
        return fdout
    return inner


def get_original_cell_construct(obj):
    """Returns the original (unwrapped) 'construct' function of a Cell subclass.

    If `obj` is an instance of a subclass of Cell and its class defines a 'construct' method,
    return the unwrapped (via inspect.unwrap) unbound 'construct' function.
    Otherwise, return None.

    Args:
        obj: An instance of a subclass of ``mindspore.nn.Cell``.

    Returns:
        The original unbound 'construct' function if `obj` is a Cell instance and its class
        has a callable 'construct' method; otherwise None.
    """
    if not isinstance(obj, nn.Cell):
        return None
    construct_func = getattr(type(obj), 'construct', None)
    if not callable(construct_func):
        return None
    return inspect.unwrap(construct_func)


class Parser:
    """
    Parser python code to ast tree.

    Args:
        fn(FunctionType/MethodType): Need parse object instance.
        parse_method(ExtendInfoOfParseObj): Extend information for parse the function.
        ast_cache: Dictionary for caching ast tree.
    """
    ast_cache = {}

    def __init__(self, fn: (types.FunctionType, types.MethodType), parse_method=None) -> None:
        self.fn = inspect.unwrap(fn.__func__ if isinstance(fn, types.MethodType) else fn)
        self.parse_method = parse_method
        self.line_offset = 0
        self.filename: str = self.fn.__code__.co_filename

        # Used to resolve the function's globals namespace.
        self.global_namespace = ModuleNamespace(self.fn.__module__)
        self.global_namespace.dicts[0]["__ms_tensor_func__"] = tensor

        self.function_module = self.fn.__module__
        # Used to resolve the function's nonlocals.
        self.closure_namespace = ClosureNamespace(self.fn)
        self.function_name = self.fn.__qualname__
        self.lines = []
        self.col_offset = 0

    @staticmethod
    def is_unsupported_namespace(value):
        """To check if not supported for namespace"""
        unsupported = isinstance(value, _builtin_function_or_method_type) and value not in convert_object_map
        logger.debug(f"'{value}' unsupported: {unsupported}.")
        return unsupported

    @staticmethod
    def is_unsupported_python_builtin_type(value):
        """To check if not supported for builtin type"""
        unsupported = value in _unsupported_python_builtin_type
        logger.debug(f"value: '{value}', unsupported builtin type: {unsupported}.")
        return unsupported

    @staticmethod
    def get_tensor_class_type(value):
        """To check if is class Tensor type"""
        if value == Tensor:
            return CLASS_INSTANCE_TYPE_TENSOR
        return CLASS_INSTANCE_TYPE_INVALID

    @staticmethod
    def is_unsupported_internal_type(value):
        """To check if not supported internal type, such as Tensor"""
        if not inspect.isclass(value):
            return False
        if value == Tensor:
            logger.debug(f"Found unsupported internal type: '{value}'.")
            return True
        return False

    @staticmethod
    def get_convert_object_for_mutable(value):
        """Get the convert object for value which don't support to be converted in C++."""
        # The value may not be supported to do ConvertData such as api 'mutable',
        # and we get its converted object from python.
        if inspect.isfunction(value) and value in (mutable,):
            return convert_object_map.get(value)
        return value

    def get_syntax_support_type(self, value):
        """Get syntax support type."""
        if is_from_third_party_library(value):
            logger.debug(f"value: '{value}' is from third party library.")
            return SYNTAX_UNSUPPORTED_NAMESPACE
        if inspect.isclass(value) or isinstance(value, _builtin_function_or_method_type):
            if self.is_unsupported_internal_type(value):
                return SYNTAX_UNSUPPORTED_INTERNAL_TYPE
            if self.is_unsupported_namespace(value):
                return SYNTAX_UNSUPPORTED_NAMESPACE
            if self.is_unsupported_python_builtin_type(value):
                return SYNTAX_UNSUPPORTED_EXTERNAL_TYPE
        return SYNTAX_SUPPORTED

    def check_lambda(self, src):
        """Check if the lamda expressions is correct."""
        obj_type = get_obj_type(self.fn)
        if (obj_type != RESOLVE_TYPE_FUNCTION or src[:4] == "def ") and is_lambda_function(self.fn):
            logger.debug("fn is lambda: %r", self.fn)
            raise ValueError("An error occurred while parsing the positional information of the lambda expression. "
                             "Please write the lambda expression on a separate line.\nFor example, "
                             "the code 'def __init__(self, combine_fn=lambda x: x + 1):' rewritten as\n"
                             "'def __init__(self, combine_fn=\nlambda x: x + 1\n):' will solve the problem.")

    def save_source_code(self, attr_name, source_lines):
        """Save cell and func source code to support run graph mode with pyc or so."""
        # pylint: disable=W1514
        if '/mindspore/' in self.filename or '\\mindspore\\' in self.filename:
            return
        if getattr(self.fn, attr_name, None) == source_lines:
            return
        if not os.access(self.filename, os.W_OK):
            raise PermissionError(f"Don't have the write permission on the file {self.filename}.")
        with open(self.filename, 'a', encoding='utf-8') as f:
            logger.debug(f"setattr for {self.fn}, attr: {attr_name}, value: {source_lines}")
            f.write(f"\n# Set source attribute for function {self.function_name} "
                    f"to support run so or pyc file in Graph Mode."
                    f"\nsetattr({self.function_name}, '{attr_name}', {source_lines})\n")
            setattr(self.fn, attr_name, source_lines)

    def parse(self):
        """Parse the function or method."""
        # pylint: disable=W0707
        # pylint: disable=W1309
        # pylint: disable=C0207
        logger.debug("fn: %r", self.fn)
        if isinstance(self.fn, (types.FunctionType, types.MethodType)) or \
           type(self.fn).__name__ == 'cython_function_or_method':
            attr_name = 'source'
            try:
                source_lines = inspect.getsourcelines(self.fn)
                if context.get_context('support_binary') or os.getenv('MS_SUPPORT_BINARY', None) == '1':
                    self.save_source_code(attr_name, source_lines)
            except (OSError, TypeError) as e:
                if hasattr(self.fn, attr_name):
                    source_lines = getattr(self.fn, attr_name)
                elif e.__str__() == "could not get source code":
                    raise OSError("Mindspore can not compile temporary source code in terminal. "
                                  "Please write source code to a python file and run the file.") from e
                else:
                    raise e
            self.lines, self.line_offset = source_lines
            original_src = ''.join(self.lines)
            hexstr = hashlib.sha256(original_src.encode()).hexdigest()
            ast_tokens_cache = Parser.ast_cache.get(hexstr)
            if not ast_tokens_cache:
                src = dedent(original_src)
                self.col_offset = \
                    len(original_src.split('\n', maxsplit=1)[0]) - len(src.split('\n')[0])
                logger.debug("Get source: %s", src)
                if not hasattr(self.fn, attr_name):
                    self.check_lambda(src)
                try:
                    ast_tokens = asttokens.ASTTokens(src, parse=True)
                except IndentationError as idt_err:
                    idt_err.filename = self.filename
                    idt_err.lineno = self.line_offset
                    idt_err.msg = f"There are incorrect indentations in definition or comment of function: " \
                                  f"'{self.function_name}'."
                    raise idt_err
                ast_tokens_cache = (ast_tokens, self.col_offset)
                Parser.ast_cache[hexstr] = ast_tokens_cache
            else:
                self.col_offset = ast_tokens_cache[1]
            return ast_tokens_cache[0], ast_tokens_cache[0].tree

        logger.error("Fn type is invalid")
        return None, None

    def get_name_from_namespace(self, value):
        """Get the name of value from namespace"""
        try:
            value_str = value.__name__
            logger.debug(
                f"value: {type(value)}, '{value_str}', hasattr(__name__): {hasattr(value, '__name__')}.")
        except:
            value_str = str(value)
            logger.debug(f"value: {type(value)}, '{value_str}'.")
        return value_str


    def is_builtin_function_name(self, var):
        """Check if the var is builtin_function name."""
        logger.debug(f"Check if the var'{var}' is builtin function.")
        builtin_function_names = vars(builtins).keys()
        if var in builtin_function_names:
            return True
        return False


    def get_namespace_symbol(self, var: str):
        """Get mindspore builtin namespace and symbol."""
        if var in self.closure_namespace:
            logger.debug(f"Found '{var}' in closure_namespace {self.closure_namespace.__str__()}.")
            try:
                value = self.closure_namespace[var]
                return self.closure_namespace, var, value
            except UnboundLocalError:
                return self.closure_namespace, var, None
        if var in self.global_namespace:
            logger.debug(f"Found '{var}' in global_namespace {self.global_namespace.__str__()}.")
            value = self.global_namespace[var]
            self.get_name_from_namespace(value)
            # To check if allowed to support.
            value = self.get_convert_object_for_mutable(value)
            support_type = self.get_syntax_support_type(value)
            support_info = self.global_namespace, var, value, support_type
            return support_info

        logger.debug(f"The name '{var}' is an undefined symbol.")
        return None, None, None

    def get_stream_obj_id(self, stream_name: str):
        """Get the object of stream."""
        if stream_name in self.global_namespace:
            logger.debug(f"Found '{stream_name}' in global_namespace {self.global_namespace.__str__()}.")
            stream_obj = self.global_namespace[stream_name]
            logger.debug(f"stream_obj.stream_id: '{stream_obj.stream_id()}'.")
            return stream_obj.stream_id()
        return None

    def check_is_base_ctx(self, var: str):
        """Check if is CtxBase, which currently supports StreamCtx or StreamLimitCtx."""
        logger.debug(f"global_namespace {self.global_namespace.__str__()}.")
        logger.debug(f"self.global_namespace.__dict__:{self.global_namespace.__dict__}")

        if var in self.global_namespace:
            logger.debug(f"Found '{var}' in global_namespace {self.global_namespace.__str__()}.")
            value = self.global_namespace[var]
            logger.debug(f"value: '{value}'.")
            if issubclass(value, StreamCtx):
                logger.debug(f"Found '{value}' is StreamCtx.")
                return 1
            if issubclass(value, StreamLimitCtx):
                logger.debug(f"Found '{value}' is StreamLimitCtx.")
                return 2
        if var == "StreamCtx":
            return 1
        if var == "StreamLimitCtx":
            return 2
        return 0

    def check_third_party_library_side_effect(self, var, attr):
        """Check if value is from a third-party library."""
        logger.debug(f"var '{var}'.")
        logger.debug(f"attr '{attr}'.")
        side_effect_attrs = {
            "numpy": {"load", "save", "savez", "savez_compressed", "loadtxt", "savetxt", "genfromtxt", "fromregex",
                      "fromstring", "tofile", "memmap", "open_memmap", "open", "exists", "abspath", "DataSource",
                      "format"},
            "pandas": {"read_csv", "to_csv", "read_excel", "to_excel", "read_json", "to_json", "read_html", "to_html",
                       "read_sql", "to_sql", "read_feather", "to_feather", "read_parquet", "to_parquet", "read_pickle",
                       "to_pickle"},
            "scipy": {"loadmat", "savemat"},
            "csv": {"reader", "writer"},
            "json": {"load", "loads", "dump", "dumps"},
            "pickle": {"load", "loads", "dump", "dumps"},
            "h5py": {"File", "Group", "Dataset"},
            "os": {"listdir", "isfile", "exists", "isdir", "mkdir", "remove", "rmdir", "symlink", "rename"},
            "shutil": {"copy", "copy2", "copytree", "move", "rmtree"},
            "pathlib": {"Path", "mkdir", "rmdir", "unlink", "rename", "symlink_to"},
            "glob": {"glob", "iglob"},
            "zipfile": {"zipfile", "ZipFile", "write", "extractall"},
            "troubleshooter": {"save", "load"}}
        if var in self.global_namespace:
            logger.debug(f"Found '{var}' in global_namespace {self.global_namespace.__str__()}.")
            value = self.global_namespace[var]
            value_str = self.get_name_from_namespace(value)
            value = self.get_convert_object_for_mutable(value)
            if is_from_third_party_library(value):
                logger.debug(f"value: '{value}' is from third party library.")
                # pylint: disable=get-dict-value-exception
                if value_str in side_effect_attrs and attr in side_effect_attrs[value_str]:
                    return True
        return False

    def analyze_super(self, class_type_node, subclass_instance):
        """Analyze super and return a class instance."""
        sub_class = type(subclass_instance)
        if class_type_node is None:
            return super(sub_class, subclass_instance)
        if isinstance(class_type_node, ast.Name):
            class_name = getattr(class_type_node, 'id')
        elif isinstance(class_type_node, ast.Attribute):
            class_name = getattr(class_type_node, 'attr')
        else:
            raise ValueError(f"The first argument of 'super()' must be a class type, "
                             f"but got {class_type_node.__class__.__name__}.")

        target_father_class = None
        for class_element in sub_class.mro():
            if class_element.__name__ == class_name:
                target_father_class = class_element
                break
        if target_father_class is None:
            raise ValueError(f"The second argument of 'super()' must be 'self', "
                             f"but got {subclass_instance}.")
        return super(target_father_class, subclass_instance)

    def get_jit_comments(self, start_lineno, end_lineno):
        """
        Get the comments at the location, starting with '# @jit'.

        Args:
            start_lineno: The start line no.
            end_lineno: The end line no.

        Returns:
            list[str], the comment strings.
        """
        comments = []
        # Ignore if to fetch the whole lines's comments.
        if start_lineno == 1 and end_lineno == len(self.lines):
            return comments

        # Add previous line comment.
        if start_lineno > 1:
            previous_lineno = start_lineno - 1
            previous_line = self.lines[previous_lineno - 1]
            striped_previous_line = previous_line.strip(' \t')
            result = re.search(r'^#\s*@jit[^\'\"]*?(?=\n|$)', striped_previous_line)
            if result:
                comments.append(result.group())

        # Add line ending comments.
        if start_lineno >= 1:
            while start_lineno <= end_lineno:
                line = self.lines[start_lineno - 1]
                result = re.search(r'#\s*@jit[^\'\"]*?(?=\n|$)', line)
                if result:
                    comments.append(result.group())
                start_lineno += 1
        return comments

    def get_source_code(self, start_lineno, start_colno, end_lineno, end_colno):
        """
        Get the script source at the location.

        Args:
            start_lineno: The start line no.
            start_colno: The start column no.
            end_lineno: The end line no.
            end_colno: The end column no.

        Returns:
            str, the source string.
        """

        if start_lineno == 0:
            logger.critical('start_lineno should not be 0')

        first_line = self.lines[start_lineno - 1]
        if start_lineno == end_lineno:
            src = first_line[self.col_offset + start_colno:self.col_offset + end_colno]
            return src

        src = first_line[self.col_offset + start_colno:]
        while start_lineno < end_lineno - 1:
            src += self.lines[start_lineno]
            start_lineno += 1
        last_line = self.lines[end_lineno - 1]
        src += last_line[:self.col_offset + end_colno]
        return src

    def get_location(self, node):
        """
        Get location of node start and end line no.

        Args:
            node: AST op node or tuple or List. This is a node in the ANF diagram,
                  here is the code location to get this node.

        Returns:
            List, [fileName, linestart, colstart, lineend, colend].
        """
        res = [self.filename]
        err_exit = 0
        start_node = None
        end_node = None
        if isinstance(node, (list, tuple)):
            node_size = len(node)
            if node_size == 0:
                err_exit = 1
            else:
                start_node = node[0]
                end_node = node[-1]
        else:
            start_node = node
            end_node = node

        if err_exit == 0:
            if hasattr(start_node, "first_token") and \
                    hasattr(end_node, "last_token"):
                start_lineno, start_colno = start_node.first_token.start
                end_lineno, end_colno = end_node.last_token.end
                expr_src = self.get_source_code(start_lineno, start_colno, end_lineno, end_colno)
                comments = self.get_jit_comments(start_lineno, end_lineno)
                start_lineno += self.line_offset - 1
                start_colno += self.col_offset
                end_lineno += self.line_offset - 1
                end_colno += self.col_offset
                res = res + [start_lineno, start_colno, end_lineno, end_colno, expr_src, comments]
            else:
                res = res + [0, 0, 0, 0, '', []]
        return res
