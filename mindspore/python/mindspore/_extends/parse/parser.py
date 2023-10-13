# This is the Python adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
#
# Copyright 2020-2023 Huawei Technologies Co., Ltd
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
import sys
import ast
import re
import hashlib
import inspect
import types
from collections import namedtuple
from typing import NamedTuple
from textwrap import dedent
import numpy

import asttokens
import astunparse

from mindspore import Tensor, CSRTensor, COOTensor, RowTensor
from mindspore import log as logger
from mindspore import nn
from mindspore import ops
from mindspore import context
from mindspore.common.api import _MindsporeFunctionExecutor
from mindspore.common import dtype as mstype
from mindspore.common.parameter import Parameter
from mindspore.common import mutable
from mindspore.common._register_for_adapter import ms_adapter_registry
from mindspore._checkparam import is_stub_tensor
from mindspore.ops._tracefunc import _PackSourceBuilder
from .namespace import Namespace, CellNamespace, ClosureNamespace, ClassMemberNamespace
from .resources import parse_object_map, ops_symbol_map, convert_object_map, convert_class_to_function_map, trope_ns
from .resources import SYMBOL_UNDEFINE
from ...common.api import _convert_python_data

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
RESOLVE_TYPE_INVALID = 0xFF             # Resolve invalid.

# Define the class instance detail type
# When the type is RESOLVE_TYPE_CLASS_INSTANCE
CLASS_INSTANCE_TYPE_CELL = 0            # Class instance type is Cell
CLASS_INSTANCE_TYPE_PRIMITIVE = 1       # Class instance type is Primitive
CLASS_INSTANCE_TYPE_NUMPY_ARRAY = 2     # Class instance type is Numpy Array
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
AST_SUB_TYPE_UNKNOWN = 0xFF            # unknown

# Syntax support
SYNTAX_SUPPORTED = 0                   # Supported syntax
SYNTAX_UNSUPPORTED_INTERNAL_TYPE = 1   # Unsupported internal type
SYNTAX_UNSUPPORTED_EXTERNAL_TYPE = 2   # Unsupported external type
SYNTAX_HYBRID_TYPE = 3                 # Hybrid type
SYNTAX_UNSUPPORTED_NAMESPACE = 4       # Unsupported namespace

# Module source location
MODULE_FROM_MINDSPORE = 0
MODULE_FROM_THIRDPARTY = 1
MODULE_FROM_USER_WORKSPACE = 2


# Process expr statement white list
# Add as needed, eg: "clear", "extend", "insert", "remove", "reverse"
parse_expr_statement_white_list = (
    "append", "insert", "clear", "reverse", "extend", "update",
)

_builtin_function_or_method_type = type(abs)

# Unsupported python builtin type in graph mode.
_unsupported_python_builtin_type = (
    set, dict, slice, complex, reversed, type,
)

_unsupported_internal_type = (
    Tensor,
)

_hybrid_type = (
    print, enumerate, zip, map, filter, abs, round, max, min, sum, getattr, hasattr, list, tuple
)

# Unsupported python builtin type in JIT Fallback.
_fallback_unsupported_python_builtin_type = (
    compile, eval, exec
)

_modules_from_mindspore = (
    "mindspore", "msadapter", "mindocr", "mindyolo", "mindnlp", "mindcv", "mindspore_rec", "mindaudio", "mindone",
    "mindspore_rl", "mindformers", "mindpet", "mindpose", "mindface", "mindsearch", "mindinsight", "mindelec",
    "mindflow", "mindsponge", "mindearth", "sciai", "mindquantum", "mindarmour", "mindpandas", "mindvision",
    "mindspore_gl", "mindspore_federated", "mindspore_gs", "mindspore_serving", "mindspore_xai", "mindspore_hub",
    "ringmo_framework", "troubleshooter", "mindtorch",
)

_global_params = {}


def _convert_map():
    """Get convert object map"""
    adapter_convert_map = ms_adapter_registry.convert_map
    return adapter_convert_map if adapter_convert_map else convert_object_map


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
        if obj._enable_backward_hook:
            method_name = "_backward_hook_construct"
        else:
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
        if getattr(resolve_, "__hash__") is None:
            return resolve_

        # If need trope the obj
        convert_map = _convert_map()
        if resolve_ in convert_map:
            resolve_ = convert_map.get(resolve_)
            logger.debug("Convert resolve: %r", resolve_)
    except Exception as e:
        if isinstance(e, NotImplementedError):
            raise e
        resolve_ = mstype._null
        logger.debug("Resolve exception occurred, value: %r", e)
        logger.debug("Resolve type is invalid, namespace: %s, symbol: %s",
                     namespace.__str__(), symbol)

    if isinstance(resolve_, _MindsporeFunctionExecutor):
        logger.debug("Resolve class _MindsporeFunctionExecutor, resolve fn instead.")
        resolve_ = resolve_.fn
    logger.debug(f"Found '{symbol}' in {namespace.__str__()}, resolved: {resolve_} / {type(resolve_)}")
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
    if isinstance(obj, types.MethodType):
        method_instance = obj.__self__
        instance_id = "%s_ID%d" % (str(method_instance.__class__.__name__), id(method_instance))
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
    kwargs = dict()
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
                         f"https://www.mindspore.cn/docs/zh-CN/master/design/dynamic_graph_and_static_graph.html \n")
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
    if not isinstance(obj, nn.CellList):
        raise TypeError(f"Obj should be nn.CellList, but got {obj}")
    if not hasattr(obj, "_cells"):
        raise AttributeError(f"nn.CellList is missing _cells property.")
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
        mod_namespace = CellNamespace(obj.__name__)
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


def get_adapter_tensor_attr(name):
    """Get the method or @property modified function of the class, excluding those inherited from parent class."""
    cls = ms_adapter_registry.tensor
    properties = [key for key, value in vars(cls).items() if isinstance(value, property)]
    if name in properties:
        return getattr(cls, name).fget, True
    methods = [key for key, value in vars(cls).items() if inspect.isfunction(value)]
    if name in methods:
        return getattr(cls, name), False
    return None, False


def is_adapter_tensor_class(cls):
    """Check if cls is adapter tensor type."""
    return cls in (Tensor, ms_adapter_registry.tensor)


def is_adapter_parameter_class(cls):
    """Check if cls is adapter parameter type."""
    return cls in (Parameter, ms_adapter_registry.parameter)


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
        return f"attribute "
    return str(obj)


def expand_expr_statement(node):
    """
    Process the expr statement and expand it.

    Returns:
        tuple, (True, expr.value, x)/(False, None, None).
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
                    return True, expr_value, target
        if not isinstance(expr_value, ast.Str):
            return True, expr_value
    return (False,)


def get_ast_namespace_symbol(obj):
    """Get obj type and namespace and symbol."""
    # Get symbol from object map.
    ops_info = parse_object_map.get(type(obj), SYMBOL_UNDEFINE)
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


def _convert_stub_tensor(data):
    """Convert stub tensor output to tensor"""
    if is_stub_tensor(data):
        return data.stub_sync()
    if isinstance(data, tuple):
        # Handle namedtuple since its type is tuple.
        if hasattr(data, "_fields"):
            type_name = data.__class__.__name__
            data_dict = data._asdict()
            fields = data_dict.keys()
            return namedtuple(type_name, fields)(**_convert_stub_tensor(data_dict))
        return tuple(_convert_stub_tensor(x) for x in data)
    if isinstance(data, list):
        # Keep the list object not change.
        for i in range(len(data)):
            data[i] = _convert_stub_tensor(data[i])
        return data
    if isinstance(data, dict):
        # Keep the dict object not change.
        keys = tuple(data.keys())
        for key in keys:
            data[_convert_stub_tensor(key)] = _convert_stub_tensor(data.pop(key))
        return data
    return data


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
        res = eval(exp_str, global_params, local_params)
        res = _convert_stub_tensor(res)
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


def get_global_params():
    """Get the global parameter."""
    logger.debug(f"get global_dict: {_global_params}")
    return _global_params


def get_dtype(name: str):
    """get mstype from name"""
    return get_attr_from_object(mstype, name)


class ThirdPartyLibraryChecker:
    """
    Check if a module or function is from third-party libraries.

    Rules for detecting third-party libraries:

    1. The mindspore module and its suite are not third-party libraries.

    2. Python built-in modules and python standard libraries are third-party libraries.

    3. Modules with module names provided by MS_JIT_IGNORE_MODULES are treated as third-party
       libraries, but those provided by MS_JIT_MODULES are not.

    4. Third-party libraries have 'site-packages' in their installation path.
    """
    def __init__(self):
        self.user_workspace_dir = self.get_top_level_module_path(os.getcwd())
        self.python_builtin_dir = os.path.abspath(os.path.dirname(os.__file__))

    @staticmethod
    def get_jit_modules():
        """Modules in jit_modules require jit."""
        jit_modules = []
        # Get jit modules from environment variable.
        env_modules = os.getenv('MS_JIT_MODULES')
        if env_modules is not None:
            jit_modules = env_modules.split(',')
        return jit_modules

    @staticmethod
    def get_jit_ignore_modules():
        """Modules in jit_ignore_modules do not need jit."""
        jit_ignore_modules = []
        # Get jit ignore modules from environment variable.
        env_modules = os.getenv('MS_JIT_IGNORE_MODULES')
        if env_modules is not None:
            jit_ignore_modules = env_modules.split(',')
        # sys.builtin_module_names do not need jit.
        jit_ignore_modules.extend(sys.builtin_module_names)
        return jit_ignore_modules

    @staticmethod
    def is_mindspore_related_module(module):
        """Check if module is mindspore module or its suite."""
        module_leftmost_name = module.__name__.split('.')[0]
        return module_leftmost_name in _modules_from_mindspore

    def get_top_level_module_path(self, module_path):
        """Get the path of the top level package of the current working directory."""
        module_abspath = os.path.abspath(module_path)
        upper_path = os.path.abspath(os.path.dirname(module_abspath))
        if module_abspath == upper_path:
            return module_abspath
        # Check whether __init__.py exists in the upper directory.
        init_path = os.path.join(upper_path, '__init__.py')
        # If the path does not exist or is accessed without permission, os.path.isfile returns false.
        if os.path.isfile(init_path):
            module_abspath = self.get_top_level_module_path(upper_path)
        return module_abspath

    def is_third_party_module(self, module):
        """Check if module is a third-party library."""
        module_leftmost_name = module.__name__.split('.')[0]
        # Modules in jit_ignore_modules are treated as third-party libraries, such as sys.builtin_module_names.
        jit_ignore_modules = self.get_jit_ignore_modules()
        if module_leftmost_name in jit_ignore_modules:
            logger.debug(f"Found third-party module '{module_leftmost_name}' in jit_ignore_modules.")
            return True
        # Modules in jit_modules require jit and they are considered to be in user workspace.
        jit_modules = self.get_jit_modules()
        if module_leftmost_name in jit_modules:
            logger.debug(f"Found user-defined module '{module_leftmost_name}' in jit_modules.")
            return False
        # A modules without __file__ attribute is considered to be in user workspace.
        if not hasattr(module, '__file__'):
            return False
        module_path = os.path.abspath(module.__file__)
        # Python builtin modules are treated as third-party libraries.
        if module_path.startswith(self.python_builtin_dir):
            logger.debug(f"Found python builtin module '{module.__name__}', which is a third-party module.")
            return True
        # Check if module is under user workspace directory.
        if module_path.startswith(self.user_workspace_dir):
            logger.debug(f"Found module '{module.__name__}' in user_workspace_dir: {self.user_workspace_dir}")
            return False
        # Third-party modules are under site-packages.
        split_path = module_path.split(os.path.sep)
        result = "site-packages" in split_path
        if result:
            logger.debug(f"Found third-party module '{module.__name__}' in path '{module_path}'")
        return result

    def get_module_source_location(self, module):
        """Get the source location of the module."""
        if self.is_mindspore_related_module(module):
            return MODULE_FROM_MINDSPORE
        if self.is_third_party_module(module):
            return MODULE_FROM_THIRDPARTY
        return MODULE_FROM_USER_WORKSPACE

    def is_third_party_module_or_function(self, value):
        """Check if value is from a third-party library."""
        if inspect.ismodule(value):
            module = value
        elif (isinstance(value, types.FunctionType) and not hasattr(value, "__jit_function__")) or \
            (isinstance(value, types.MethodType) and not hasattr(value.__func__, "__jit_function__")):
            if value in _convert_map():
                return False
            module = inspect.getmodule(value)
            if module is None:
                return False
        else:
            return False
        return self.get_module_source_location(module) == MODULE_FROM_THIRDPARTY


third_party_checker = ThirdPartyLibraryChecker()


def is_from_third_party_library(value):
    """Check if value is from a third-party library."""
    return third_party_checker.is_third_party_module_or_function(value)


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
        self.pack_builder = _PackSourceBuilder(fn) if hasattr(fn, "pack_fn") else None
        self.parse_method = parse_method
        self.line_offset = 0
        self.filename: str = self.fn.__code__.co_filename

        # Used to resolve the function's globals namespace.
        self.global_namespace = CellNamespace(self.fn.__module__)
        self.function_module = self.fn.__module__
        # Used to resolve the function's nonlocals.
        self.closure_namespace = ClosureNamespace(self.fn)
        self.function_name = self.fn.__qualname__
        self.lines = []
        self.col_offset = 0

    @staticmethod
    def is_unsupported_namespace(value):
        """To check if not supported for namespace"""
        unsupported = isinstance(value, _builtin_function_or_method_type) and value not in _convert_map()
        logger.debug(f"'{value}' unsupported: {unsupported}.")
        if unsupported and value in _fallback_unsupported_python_builtin_type:
            raise TypeError(f"'{value}' is not supported both in JIT Fallback and graph mode.")
        return unsupported

    @staticmethod
    def is_unsupported_python_builtin_type(value):
        """To check if not supported for builtin type"""
        unsupported = value in _unsupported_python_builtin_type
        logger.debug(f"value: '{value}', unsupported builtin type: {unsupported}.")
        return unsupported

    @staticmethod
    def is_unsupported_internal_type(value):
        """To check if not supported internal type, such as Tensor"""
        for item in _unsupported_internal_type:
            if value == item:
                logger.debug(f"Found unsupported internal type: '{value}'.")
                return True
        if ms_adapter_registry.is_registered and value == ms_adapter_registry.tensor:
            return True
        return False

    @staticmethod
    def is_hybrid_type(value):
        """To check if hybrid type, such as print"""
        for item in _hybrid_type:
            if value == item:
                logger.debug(f"Found hybrid type: '{value}'.")
                return True
        return False

    @staticmethod
    def get_convert_object_for_mutable(value):
        """Get the convert object for value which don't support to be converted in C++."""
        # The value may not be supported to do ConvertData such as api 'mutable',
        # and we get its converted object from python.
        if inspect.isfunction(value) and value in (mutable,):
            return _convert_map().get(value)
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
            if self.is_hybrid_type(value):
                return SYNTAX_HYBRID_TYPE
        return SYNTAX_SUPPORTED

    def parse(self):
        """Parse the function or method."""
        logger.debug("fn: %r", self.fn)
        if isinstance(self.fn, (types.FunctionType, types.MethodType)) or \
           type(self.fn).__name__ == 'cython_function_or_method':
            attr = 'source'
            try:
                source_lines = inspect.getsourcelines(self.fn)
                if context.get_context('support_binary') and \
                   '/mindspore/' not in self.filename and '\\mindspore\\' not in self.filename and \
                   (not hasattr(self.fn, attr) or getattr(self.fn, attr) != source_lines):
                    if not os.access(self.filename, os.W_OK):
                        raise PermissionError(f"Don't have the write permission on the file {self.filename}.")
                    with open(self.filename, 'a') as f:
                        f.write(f"\n# Set source attribute for function {self.function_name} "
                                f"to support run so or pyc file in Graph Mode."
                                f"\nsetattr({self.function_name}, '{attr}', {source_lines})\n")
                        setattr(self.fn, attr, source_lines)
            except (OSError, TypeError) as e:
                if hasattr(self.fn, attr):
                    source_lines = getattr(self.fn, attr)
                else:
                    if e.__str__() == "could not get source code":
                        raise OSError(f"Mindspore can not compile temporary source code in terminal. "
                                      f"Please write source code to a python file and run the file.")
                    raise e
            self.lines, self.line_offset = source_lines
            original_src = ''.join(self.lines)
            hexstr = hashlib.sha256(original_src.encode()).hexdigest()
            ast_tokens_cache = Parser.ast_cache.get(hexstr)
            if not ast_tokens_cache:
                src = dedent(original_src)
                self.col_offset = \
                    len(original_src.split('\n')[0]) - len(src.split('\n')[0])
                logger.debug("Get source: %s", src)
                try:
                    if self.pack_builder:
                        src = self.pack_builder.get_code_source()
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

    def is_jit_supported_attribute(self, var, attr):
        """Check whether the value is a constant."""
        if var in self.global_namespace:
            module = self.global_namespace[var]
            if hasattr(module, attr):
                value = getattr(module, attr)
                # Check if value is constant.
                if isinstance(value, (int, float, bool)):
                    return True
                # Check if value in convert_map.
                if isinstance(value, (tuple, list, dict)) or getattr(value, "__hash__") is None:
                    return False
                if inspect.ismodule(module) and value in _convert_map():
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
            value_str = value.__name__ if hasattr(value, '__name__') else str(value)
            logger.debug(f"value: {type(value)}, '{value_str}', hasattr(__name__): {hasattr(value, '__name__')}.")
            # To check if allowed to support.
            value = self.get_convert_object_for_mutable(value)
            support_type = self.get_syntax_support_type(value)
            support_info = self.global_namespace, var, value, support_type
            return support_info

        logger.debug(f"The name '{var}' is an undefined symbol.")
        return None, None, None

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
            value_str = value.__name__ if hasattr(value, '__name__') else str(value)
            logger.debug(f"value: {type(value)}, '{value_str}', hasattr(__name__): {hasattr(value, '__name__')}.")
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
