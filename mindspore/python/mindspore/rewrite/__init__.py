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
from .api.scoped_value import ScopedValue, ValueType
from .api.symbol_tree import SymbolTree
from .api.node import Node
from .api.node_type import NodeType
from .api.pattern_engine import PatternEngine, PatternNode, VarNode, Replacement
from .sparsify.sparsify import sparsify
from .sparsify.utils import ArgType, SparseFunc


__all__ = ["SymbolTree", "Node", "NodeType", "ScopedValue", "ValueType", "PatternEngine", "PatternNode", "VarNode",
           "Replacement"]
