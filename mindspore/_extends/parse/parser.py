# This is the Python adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
#
# Copyright 2020 Huawei Technologies Co., Ltd
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

import ast
import hashlib
import inspect
import types
from dataclasses import is_dataclass
from textwrap import dedent

import asttokens

from mindspore import Tensor as MsTensor
from mindspore import context
from mindspore import log as logger
from mindspore import nn
from mindspore import ops
from mindspore.common.api import _MindSporeFunction
from mindspore.common.dtype import pytype_to_dtype
from .namespace import CellNamespace, ClosureNamespace, ClassMemberNamespace
from .resources import parse_object_map, convert_object_map, trope_ns, SYMBOL_UNDEFINE, NO_IMPLEMENT

# define return value
RET_SUCCESS = 0
RET_FAILURE = 0xFF

# define resolve type
RESOLVE_TYPE_NONE = 0                   # resolve None
RESOLVE_TYPE_FUNCTION = 1               # resolve function
RESOLVE_TYPE_METHOD = 2                 # resolve class method
RESOLVE_TYPE_CLASS_TYPE = 3             # resolve class type
RESOLVE_TYPE_CLASS_INSTANCE = 4         # resolve the class instance of common class
RESOLVE_TYPE_INVALID = 0xFF

# define the class instance detail type
# When the type is RESOLVE_TYPE_CLASS_INSTANCE
CLASS_INSTANCE_TYPE_CELL = 0            # class instance type is Cell
CLASS_INSTANCE_TYPE_PRIMITIVE = 1       # class instance type is Primitive
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
AST_SUB_TYPE_SUBSCRIPT = 7             # ast.Subscript
AST_SUB_TYPE_STARRED = 8               # ast.Starred
AST_SUB_TYPE_ATTRIBUTE = 9             # ast.Attribute
AST_SUB_TYPE_UNKNOWN = 0xFF            # unknown

# Process expr statement white list
# add as needed, eg: "clear", "extend", "insert", "remove", "reverse"
parse_expr_statement_white_list = (
    "append",
)


def create_slice_obj(start, end, step):
    """Create slice object"""
    return slice(start, end, step)


def parse_cb(func, parse_method=None):
    """Implements the function of parse."""
    return Parser(func, parse_method)


def get_parse_method_of_class(obj, parse_method=None):
    """
    Het parse method of class.

    Args:
        obj(Object): Instance of class.
        parse_method(str): Save the method name. Cell object has default method named 'construct'.

    Returns:
        Function, obj's method.
    """
    method = None
    method_name = None
    if parse_method is not None:
        method_name = parse_method
    else:
        if isinstance(obj, nn.Cell):
            if obj.enable_hook:
                if context.get_context("mode") == context.GRAPH_MODE:
                    raise ValueError("The graph mode does not support hook function.")
                method_name = "_hook_construct"
            else:
                method_name = "construct"
    if method_name is not None:
        if hasattr(obj, method_name):
            method = getattr(obj, method_name)
    return method


def get_bprop_method_of_class(obj, parse_method=None):
    """
    Get bprop method of class.

    Args:
        obj (Object): Instance of class.
        parse_method(str): Save the method name. Cell object has default method named 'bprop'.

    Returns:
        Function, obj's method.
    """
    method = None
    if isinstance(obj, nn.Cell):
        method_name = "bprop"
        if hasattr(obj, method_name):
            method = getattr(obj, method_name)
    return method


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

        # list and dict is not hashable ,it can not be key for the map, just return the result
        if isinstance(resolve_, (tuple, list, dict)):
            return resolve_

        # dataclass may not be hashable
        if getattr(resolve_, "__hash__") is None:
            return resolve_

        # If need trope the obj
        if resolve_ in convert_object_map:
            resolve_ = convert_object_map.get(resolve_)
            logger.debug("convert resolve = %r", resolve_)
            if resolve_ == NO_IMPLEMENT:
                raise NotImplementedError("not implemented for ", str(symbol))
    except Exception as e:
        if isinstance(e, NotImplementedError):
            raise e
        resolve_ = None
        logger.debug("resolve exception occurred, value = %r", e)
        logger.debug("resolve type is invalid, namespace = %s, symbol = %s",
                     namespace.__str__(), symbol)
    if isinstance(resolve_, _MindSporeFunction):
        logger.debug("resolve class _MindSporeFunction, resolve fn instead.")
        resolve_ = resolve_.fn
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
    logger.debug("obj_key %s obj_id = %s", obj_key, obj_id)

    # method has same id of different instance
    if isinstance(obj, types.MethodType):
        method_instance = obj.__self__
        instance_id = "%s_ID%d" % (str(method_instance.__class__.__name__), id(method_instance))
        obj_id = instance_id + obj_id + str(obj.__hash__())
    return obj_id, obj_key


def is_class_member(node):
    """Check the attr is class member variable."""
    type_ = node.__class__.__name__
    if type_ == "Attribute":
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
    obj_type = RESOLVE_TYPE_INVALID
    if obj is None:
        obj_type = RESOLVE_TYPE_NONE
    elif isinstance(obj, types.FunctionType):
        obj_type = RESOLVE_TYPE_FUNCTION
    elif isinstance(obj, types.MethodType):
        obj_type = RESOLVE_TYPE_METHOD
    elif isinstance(obj, type):
        obj_type = RESOLVE_TYPE_CLASS_TYPE
    elif _is_class_instance(obj):
        obj_type = RESOLVE_TYPE_CLASS_INSTANCE
    else:
        # here for ndarray, just print its shape (in case of the array to large and print many data in screen)
        is_ndarray = type(obj).__name__ == 'ndarray' and hasattr(obj, 'shape')
        raise TypeError(f'Invalid object with type `{type(obj)}` and {"shape" if is_ndarray else "value"} '
                        f'`{obj.shape if is_ndarray else obj}`.')
    return obj_type


def get_class_instance_type(obj):
    """Get the class instance detail type."""
    # check the obj type
    logger.debug("Get the class type(%r)", obj)
    class_type = CLASS_INSTANCE_TYPE_INVALID
    if _is_class_instance(obj):
        if isinstance(obj, nn.Cell):
            class_type = CLASS_INSTANCE_TYPE_CELL
        elif isinstance(obj, ops.Primitive):
            class_type = CLASS_INSTANCE_TYPE_PRIMITIVE
        # Add the other type base requirement
    return class_type


def _is_class_instance(obj):
    """Confirm the obj is class instance."""
    return isinstance(obj, (nn.Cell, ops.Primitive)) or _is_dataclass_instance(obj)


def _is_dataclass_instance(obj):
    """check whether a class is an instance of a dataclass (and not a dataclass itself)"""
    return is_dataclass(obj) and not isinstance(obj, type)


def create_obj_instance(cls_type, args_tuple=None):
    """Create python instance."""
    obj = None
    if isinstance(cls_type, type):
        # check the type, now only support nn.Cell and Primitive
        if issubclass(cls_type, (nn.Cell, ops.Primitive)):
            if args_tuple is not None:
                obj = cls_type(*args_tuple)
            else:
                obj = cls_type()
    return obj


def get_module_namespace(obj):
    """Get the module's namespace."""
    logger.debug("get module namespace, module = %r", obj)
    mod_namespace = None
    if isinstance(obj, types.ModuleType):
        mod_namespace = CellNamespace(obj.__name__)
    else:
        logger.warning("Module(%r) is invalid, get namespace failure!", obj)
    return mod_namespace


def get_class_member_namespace_symbol(obj):
    """Get obj class member type."""
    logger.debug("get class instance namespace, object = %r", obj)
    class_namespace = ClassMemberNamespace(obj)
    logger.debug("class namesapce = %r", class_namespace)
    return class_namespace


def get_dataclass_attributes(cls):
    """Get attributes of dataclass."""
    fields = cls.__dataclass_fields__
    attributes = {name: pytype_to_dtype(field.type)
                  for name, field in fields.items()}
    return attributes


def get_dataclass_methods(cls):
    """Get functions of dataclass."""
    methods = {name: getattr(cls, name)
               for name in dir(cls)
               if isinstance(getattr(cls, name), (types.FunctionType,))}
    return methods


def convert_to_ms_tensor(data):
    """Convert C++ tensor to mindspore tensor."""
    return MsTensor(data)


def get_object_description(obj, fname, fline):
    """return method or funcition description for error report, include location, class name, etc."""
    if isinstance(obj, types.MethodType):
        obj_cls = obj.__self__.__class__
        class_name = f'{obj_cls.__module__}.{obj_cls.__qualname__}'
        cls_fname = inspect.getfile(obj_cls)
        _, cls_fline = inspect.getsourcelines(obj_cls)
        class_loc = f'{cls_fname}:{cls_fline}'
        return f"bound method '{obj.__name__}' at {fname}:{fline} of <{class_name} at {class_loc} object>"
    if isinstance(obj, types.FunctionType):
        return f"function '{obj.__name__}' at {fname}:{fline}"
    if isinstance(obj, ast.FunctionDef):
        return f"function '{obj.name}' at {fname}:{fline}"
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
    # step 1:get symbol from object map
    ops_info = parse_object_map.get(type(obj), SYMBOL_UNDEFINE)
    logger.debug("ops info = %r", ops_info)
    return ops_info


def get_operation_namespace_symbol(var: str):
    """Get operation namespace and symbol."""
    ops_info = (trope_ns, var)
    logger.debug("get operation ops info = %r", ops_info)
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
    method_name = f'{node.__class__.__name__}'
    node_type = [method_name]
    # judge the ast main type
    if isinstance(node, ast.stmt):
        node_type.append(AST_MAIN_TYPE_STMT)
    elif isinstance(node, (ast.expr, ast.slice)) or node is None:
        # ast.slice and ast.expr should be expr
        node_type.append(AST_MAIN_TYPE_EXPR)
    else:
        node_type.append(AST_MAIN_TYPE_UNKNOWN)
    return node_type


def get_args_default_values(node):
    """get the args'default values of parse object."""
    nondefaults = [None] * (len(node.args.args) - len(node.args.defaults))
    defaults = nondefaults + node.args.defaults + node.args.kw_defaults
    if node.args.vararg:
        defaults.append(None)
    if node.args.kwarg:
        defaults.append(None)
    return defaults


def get_args(node):
    """Get the arg of parse object."""
    args = []
    # process position args
    for arg in node.args.args:
        args.append(arg)

    # process kwonlyargs: kwonlyargs is append after position args
    if node.args.kwonlyargs:
        for kwarg in node.args.kwonlyargs:
            args.append(kwarg)
    # process vararg: vararg is append after kwonlyargs
    if node.args.vararg:
        args.append(node.args.vararg)
    # process kwarg: kwarg is append after vararg
    if node.args.kwarg:
        args.append(node.args.kwarg)
    return args


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
        self.fn = fn
        self.parse_method = parse_method
        self.line_offset = 0
        self.filename: str = inspect.getfile(self.fn)

        # Used to resolve the function's globals Namespace.
        self.global_namespace = CellNamespace(fn.__module__)
        self.function_module = fn.__module__
        # Used to resolve the function's nonlocals.
        self.closure_namespace = ClosureNamespace(fn)
        self.function_name = fn.__name__
        self.col_offset = 0

    def parse(self):
        """Parse the function or method."""
        logger.debug("fn = %r", self.fn)
        tree = None
        if isinstance(self.fn, (types.FunctionType, types.MethodType)):
            lines, self.line_offset = inspect.getsourcelines(self.fn)
            original_src = ''.join(lines)
            hexstr = hashlib.sha256(original_src.encode()).hexdigest()
            tree = Parser.ast_cache.get(hexstr)
            if not tree:
                src = dedent(original_src)
                self.col_offset = \
                    len(original_src.split('\n')[0]) - len(src.split('\n')[0])
                logger.debug("get source = %s", src)
                try:
                    tree = asttokens.ASTTokens(src, parse=True).tree
                except IndentationError as idt_err:
                    idt_err.filename = self.filename
                    idt_err.lineno = self.line_offset
                    idt_err.msg = f"There are incorrect indentations in definition or comment of function: " \
                                  f"'{self.fn.__qualname__}'."
                    raise idt_err
                Parser.ast_cache[hexstr] = tree
        else:
            logger.error("Fn type is invalid")
        return tree

    def get_namespace_symbol(self, var: str):

        """Get symbol type and namespace and symbol."""
        if var in self.closure_namespace:
            logger.debug("in closure_namespace")
            return self.closure_namespace, var
        if var in self.global_namespace:
            logger.debug("in global_namespace")
            value = self.global_namespace[var]
            if isinstance(value, type(abs)) and self.global_namespace[var] not in convert_object_map:
                error_info = f"The builtin function '{var}' is not supported in graph mode."
                return None, var, error_info
            return self.global_namespace, var
        error_info = f"The name '{var}' is not defined."
        return None, var, error_info

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
            raise ValueError(f"When call 'super', the first arg should be a class type, "
                             f"but got {class_type_node.__class__.__name__}.")

        target_father_class = None
        for class_element in sub_class.mro():
            if class_element.__name__ == class_name:
                target_father_class = class_element
                break
        if target_father_class is None:
            raise ValueError("When call 'super', the second arg should be an instance of first arg.")
        return super(target_father_class, subclass_instance)

    def get_location(self, node):
        """
        Get location of node start and end line no.

        Args:
            node: AST op node or tuple or List. This is a node in the ANF diagram,
                  here is the code location to get this node.

        Returns:
            List, [fileName, linestart, colstart, lineend, colend].
        """
        ret = [self.filename]
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
            if hasattr(start_node, "lineno") and \
                    hasattr(end_node, "col_offset"):
                start_lineno, start_colno = start_node.first_token.start
                end_lineno, end_colno = end_node.last_token.end
                start_lineno += self.line_offset - 1
                start_colno += self.col_offset
                end_lineno += self.line_offset - 1
                end_colno += self.col_offset
                ret = ret + [start_lineno, start_colno, end_lineno, end_colno]
            else:
                ret = ret + [0, 0, 0, 0]
        return ret
