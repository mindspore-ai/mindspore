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
"""
MindSpore Rewrite package.
This is an experimental python package that is subject to change or deletion.
"""
from mindspore.rewrite.parsers.attribute_parser import g_attribute_parser
from mindspore.rewrite.parsers.constant_parser import g_name_parser, g_num_parser, g_str_parser
from mindspore.rewrite.parsers.container_parser import g_list_parser, g_tuple_parser
from .parsers.module_parser import g_module_parser
from .parsers.class_def_parser import g_classdef_parser
from .parsers.function_def_parser import g_functiondef_parser
from .parsers.arguments_parser import g_arguments_parser
from .parsers.assign_parser import g_assign_parser
from .parsers.if_parser import g_if_parser
from .parsers.return_parser import g_return_parser
from .parsers.for_parser import g_for_parser
from .api.scoped_value import ScopedValue, ValueType
from .api.symbol_tree import SymbolTree
from .api.node import Node
from .api.node_type import NodeType
from .api.pattern_engine import PatternEngine, PatternNode, VarNode, Replacement
from .api.tree_node_helper import TreeNodeHelper
from .sparsify.sparsify import sparsify
from .sparsify.utils import ArgType, SparseFunc

__all__ = ["SymbolTree", "Node", "NodeType", "ScopedValue", "ValueType", "PatternEngine", "PatternNode", "VarNode",
           "Replacement", "TreeNodeHelper", "sparsify", "ArgType", "SparseFunc"]
