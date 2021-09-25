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
"""DebuggerTensor."""
from abc import ABC


class DebuggerTensor(ABC):
    """
    The tensor with specific rank, iteration and debugging info.

    Note:
        - Users should not instantiate this class manually.
        - The instances of this class is immutable.
        - A DebuggerTensor is always the output tensor of a node.
    """
    @property
    def node(self):
        """
        Get the node that outputs this tensor.

        Returns:
            Node, the node that outputs this tensor.
        """
        return None

    @property
    def name(self):
        """
        Get the name of this tensor.

        The name is composed of full name of a node and the slot number.

        Returns:
            str, the name of this tensor.
        """
        return ""

    @property
    def slot(self):
        """
        Get slot.

        Returns:
            int, the slot of the tensor on the node.
        """
        return -1

    @property
    def iteration(self):
        """
        Get the iteration for this tensor.

        Returns:
            int, the iteration for this tensor.
        """
        return -1

    @property
    def rank(self):
        """
        Get the rank for this tensor.

        Returns:
            int, the rank for this tensor.

        """
        return -1

    def get_value(self):
        """
        Get the value of the tensor.

        Returns:
            numpy.ndarray, the value of the debugger tensor.
        """

    def get_affected_nodes(self):
        """
        Get the nodes that use current tensor as input.
        """
