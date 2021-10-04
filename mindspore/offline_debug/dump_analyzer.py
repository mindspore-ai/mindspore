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
"""Debugger python API."""

from typing import Iterable

from mindspore.offline_debug.debugger_tensor import DebuggerTensor
from mindspore.offline_debug.node import Node
from mindspore.offline_debug.watchpoints import WatchpointBase, WatchpointHit


class DumpAnalyzer:
    """
    Analyzer to inspect the dump data.

    Args:
        summary_dir (str): The path of the summary directory which contains
            dump folder.
        mem_limit (int, optional): The memory limit for this debugger session in
            MB. Default: None, which means no limit.
    """

    def __init__(self, summary_dir, mem_limit=None):
        self._summary_dir = summary_dir
        self._mem_limit = mem_limit

    def export_graphs(self, output_dir=None):
        """
        Export the computational graph(s) in xlsx file(s) to the output_dir.

        The file(s) will contain the stack info of graph nodes.

        Args:
            output_dir (str, optional): Output directory to save the file.
                Default: None, which means to use the current working directory.

        Returns:
            str. The path of the generated file.
        """

    def select_nodes(
            self,
            query_string,
            use_regex=False,
            match_target="name",
            case_sensitive=True) -> Iterable[Node]:
        """
        Select nodes.

        Args:
            query_string (str): Query string. For a node to be selected, the
                match target field must contains or matches the query string.
            use_regex (bool): Indicates whether query is a regex. Default: False.
            match_target (str, optional): The field to search when selecting
                nodes. Available values are "name", "stack".
                "name" means to search the name of the nodes in the
                graph. "stack" means the stack info of
                the node. Default: "name".
            case_sensitive (bool, optional): Whether case-sensitive when
                selecting tensors. Default: True.

        Returns:
            Iterable[Node], the matched nodes.
        """

    def select_tensors(
            self,
            query_string,
            use_regex=False,
            match_target="name",
            iterations=None,
            ranks=None,
            slots=None,
            case_sensitive=True) -> Iterable[DebuggerTensor]:
        """
        Select tensors.

        Args:
            query_string (str): Query string. For a tensor to be selected, the
                match target field must contains or matches the query string.
            use_regex (bool): Indicates whether query is a regex. Default: False.
            match_target (str, optional): The field to search when selecting
                tensors. Available values are "name", "stack".
                "name" means to search the name of the tensors in the
                graph. "name" is composed of graph node's full_name
                and the tensor's slot number. "stack" means the stack info of
                the node that outputs this tensor. Default: "name".
            iterations (list[int], optional): The iterations to select. Default:
                None, which means all iterations will be selected.
            ranks (list(int], optional): The ranks to select. Default: None,
                which means all ranks will be selected.
            slots (list[int], optional): The slot of the selected tensor.
                Default: None, which means all slots will be selected.
            case_sensitive (bool, optional): Whether case-sensitive when
                selecting tensors. Default: True.

        Returns:
          Iterable[DebuggerTensor], the matched tensors.
        """

    def get_iterations(self) -> Iterable[int]:
        """Get the available iterations this run."""

    def get_ranks(self) -> Iterable[int]:
        """Get the available ranks in this run."""

    def check_watchpoints(
            self,
            watchpoints: Iterable[WatchpointBase]) -> Iterable[WatchpointHit]:
        """
        Check the given watch points on specified nodes(if available) on the
        given iterations(if available) in a batch.

        Note:
            For speed, all watchpoints for the iteration should be given at
            the same time to avoid reading tensors len(watchpoints) times.

        Args:
            watchpoints (Iterable[WatchpointBase]): The list of watchpoints.

        Returns:
            Iterable[WatchpointHit], the watchpoint hist list is carefully
                sorted so that the user can see the most import hit on the
                top of the list. When there are many many watchpoint hits,
                we will display the list in a designed clear way.
        """
