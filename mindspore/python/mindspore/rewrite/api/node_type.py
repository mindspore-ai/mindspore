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
"""Rewrite module api: NodeType."""
from enum import Enum


class NodeType(Enum):
    """
    `NodeType` represents type of `Node`.

    - Unknown: Not inited NodeType.
    - CallCell: `CallCell` node represents invoking cell-op in forward method.
    - CallPrimitive: `CallPrimitive` node represents invoking primitive-op in forward method.
    - CallMethod: `CallMethod` node represents invoking of method in forward method which can not be mapped to
      cell-op or primitive-op in MindSpore.
    - Python: `Python` node holds unsupported-ast-node or unnecessary-to-parse-ast-node.
    - Input: `Input` node represents input of `SymbolTree` corresponding to arguments of forward method.
    - Output: `Output` node represents output of SymbolTree corresponding to return statement of forward method.
    - Tree: `Tree` node represents sub-network invoking in forward method.
    - MathOps: `MathOps` node represents a mathematical operation, such as adding or comparing in forward method.

    """
    Unknown = 0
    # Compute node type
    CallCell = 1
    CallPrimitive = 2
    CallModule = 3
    CallFunction = 4
    CallMethod = 5
    # Other node type
    Python = 6
    Input = 7
    Output = 8
    Tree = 9
    CellContainer = 10
    MathOps = 11
