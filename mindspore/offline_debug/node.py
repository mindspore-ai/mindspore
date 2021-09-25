# Copyright 2021 Huawei Technologies Co., Ltd
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
# ==============================================================================
"""Node in the computational graph."""
from abc import ABC


class Node(ABC):
    """Node in the computational graph."""
    @property
    def name(self):
        """
        Get the full name of this node.

        Returns:
            str, the full name of the node.
        """
        return ""

    @property
    def stack(self):
        """Get stack info."""
        return None

    def get_input_tensors(
            self,
            iterations=None,
            ranks=None,
            slots=None):
        """
        Get the input tensors of the node.

        Returns:
            Iterable[DebuggerTensor], the input tensors of the node.
        """

    def get_output_tensors(
            self,
            iterations=None,
            ranks=None,
            slots=None):
        """
        Get the output tensors of this node.

        Returns:
            Iterable[DebuggerTensor], the output tensors of the node.
        """

    def get_input_nodes(self):
        """
        Get the input nodes of this node.

        Returns:
            Iterable[Node], the input nodes of this node.

        """

    def get_output_nodes(self):
        """
        Get the nodes that use the output tensors of this node.

        Returns:
            Iterable[Node], the output nodes of this node.
        """
