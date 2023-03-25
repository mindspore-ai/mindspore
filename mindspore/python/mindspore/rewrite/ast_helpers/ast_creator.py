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
"""Create ast node in specific scope."""
import ast
from mindspore.rewrite.ast_creator_register import ast_creator_registry
import mindspore as ms


def ast_args_creator(args: list):
    """Create an arg list."""
    ast_args = list()
    for arg in args:
        ast_arg = ast_create_arg_value(arg)
        ast_args.append(ast_arg)
    return ast_args


def ast_assign_creator(targets: ast.AST, value: ast.AST):
    """Create an ast.Assign node."""
    return ast.Assign(targets=targets, value=value, lineno=0, col_offset=0)


def ast_attributer_creator(attribute: str):
    """Create an ast.Attribute node"""
    value, attr = attribute.rsplit(".", 1)
    if "." in value:
        ast_value = ast_attributer_creator(value)
    else:
        ast_value = ast_name_creator(value)
    return ast.Attribute(value=ast_value, attr=attr, lineno=0, col_offset=0, ctx="Load()")


def ast_call_creator(func: ast.AST, args: list, keywords: list):
    """Create an ast.Call node"""
    return ast.Call(func=func, args=args, keywords=keywords, lineno=0, col_offset=0)


def ast_create_arg_value(value):
    """Create arg node by type."""
    if isinstance(value, (int, float)):
        ast_value = ast_num_creator(value)
    elif isinstance(value, str):
        ast_value = ast_str_creator(value)
    elif value in (ms.float16, ms.float32, ms.float64):
        ast_value = ast_attributer_creator(".".join(["mindspore", str(value).lower()]))
    elif isinstance(value, ms.rewrite.node.Node):
        ast_value = ast_str_creator(value.get_targets()[0])
    else:
        raise TypeError("Unsupported arg type: ", type(value))
    return ast_value


def ast_index_creator(index: ast.Num):
    """Create ast.Index node."""
    index = ast_num_creator(index)
    return ast.Index(value=index, lineno=0, col_offset=0)


def ast_keyword_creator(arg: str, value):
    """Create ast.keyword node."""
    ast_value = ast_create_arg_value(value)
    return ast.keyword(arg=arg, value=ast_value)


def ast_kwargs_creator(kwargs: dict):
    """Create ast.keyword node list."""
    keywords = list()
    for k, v in kwargs.items():
        kw = ast_keyword_creator(k, v)
        keywords.append(kw)
    return keywords


def ast_name_creator(id_: str):
    """Create an ast.Name node."""
    return ast.Name(id=id_, ctx=ast.Store, lineno=0, col_offset=0)


def ast_num_creator(num: int):
    """Create an ast.Num node."""
    return ast.Num(n=num, lineno=0, col_offset=0)


def ast_str_creator(s: str):
    """Create an ast.str node."""
    return ast.Str(s=s, lineno=0, col_offset=0)


def ast_subscript_creator(value: ast.AST, index: int):
    """Create an ast.Subscript node."""
    s = ast_index_creator(index)
    return ast.Subscript(value=value, slice=s, lineno=0, col_offset=0, ctx=ast.Load)


ast_creator_registry.register("Args", ast_args_creator)
ast_creator_registry.register("Assign", ast_assign_creator)
ast_creator_registry.register("Attribute", ast_attributer_creator)
ast_creator_registry.register("Call", ast_call_creator)
ast_creator_registry.register("Index", ast_index_creator)
ast_creator_registry.register("KwArgs", ast_kwargs_creator)
ast_creator_registry.register("Name", ast_name_creator)
ast_creator_registry.register("Num", ast_num_creator)
ast_creator_registry.register("Subscript", ast_subscript_creator)
