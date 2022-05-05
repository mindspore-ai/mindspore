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
"""Visit nods of SymbolTree."""


class NodeVisitor:
    """Iterator class to access SymbolTree nodes"""
    def __init__(self, stree):
        self._stree = stree
        self._nodes = []
        self._index = 0

    def __iter__(self):
        self._nodes = list(self._stree.get_nodes_dict().values())
        self._index = 0
        return self

    def __next__(self):
        if self._index < len(self._nodes):
            node = self._nodes[self._index]
            self._index += 1
            return node

        raise StopIteration

    def append_node(self, node):
        """append new node to iterator"""
        self._nodes.append(node)

    def remove_node(self, node):
        """remove node of iterator"""
        self._nodes.remove(node)
