# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
"""The container of metadata used in profiler parser."""
GIGABYTES = 1024 * 1024 * 1024


class HWTSContainer:
    """
    HWTS output container.

    Args:
        split_list (list): The split list of metadata in HWTS output file.
    """
    def __init__(self, split_list):
        self._op_name = ''
        self._duration = None
        self._status = split_list[0]
        self._task_id = split_list[6]
        self._cycle_counter = float(split_list[7])
        self._stream_id = split_list[8]

    @property
    def status(self):
        """Get the status of the operator, i.e. Start or End."""
        return self._status

    @property
    def task_id(self):
        """Get the task id of the operator."""
        return self._task_id

    @property
    def cycle_counter(self):
        """Get the cycle counter."""
        return self._cycle_counter

    @property
    def stream_id(self):
        """Get the stream id of the operator."""
        return self._stream_id

    @property
    def op_name(self):
        """Get the name of the operator."""
        return self._op_name

    @op_name.setter
    def op_name(self, name):
        """Set the name of the operator."""
        self._op_name = name

    @property
    def duration(self):
        """Get the duration of the operator execution."""
        return self._duration

    @duration.setter
    def duration(self, value):
        """Set the duration of the operator execution."""
        self._duration = value


class TimelineContainer:
    """
    A container of operator computation metadata.

    Args:
        split_list (list): The split list of metadata in op_compute output file.
    """
    def __init__(self, split_list):
        self._op_name = split_list[0]
        self._stream_id = str(split_list[1])
        self._start_time = float(split_list[2])
        self._duration = float(split_list[3])
        self._pid = None
        if len(split_list) == 5:
            self._pid = int(split_list[4])

    @property
    def op_name(self):
        """Get the name of the operator."""
        return self._op_name

    @property
    def stream_id(self):
        """Get the stream id of the operator."""
        return self._stream_id

    @property
    def start_time(self):
        """Get the execution start time of the operator."""
        return self._start_time

    @property
    def duration(self):
        """Get the duration of the operator execution."""
        return self._duration

    @property
    def pid(self):
        """Get the pid of the operator execution."""
        return self._pid


class MemoryGraph:
    """
    A container for graph.

    Args:
        graph_proto (proto): Graph proto, defined in profiler module.
    """
    def __init__(self, graph_proto):
        self._graph_proto = graph_proto
        self.graph_id = graph_proto.graph_id
        self.static_mem = graph_proto.static_mem / GIGABYTES
        self.fp_start = None
        self.bp_end = None
        self.lines = []
        self.nodes = {}
        self.breakdowns = []

    def to_dict(self):
        """Convert Graph to dict."""
        graph = {
            'graph_id': self.graph_id,
            'static_mem': self.static_mem,
            'nodes': self.nodes,
            'fp_start': self.fp_start,
            'bp_end': self.bp_end,
            'lines': self.lines,
            'breakdowns': self.breakdowns
        }

        return graph


class MemoryNode:
    """
    A container for node.

    Args:
        node_proto (proto): Node proto.
    """
    def __init__(self, node_proto):
        self._node_proto = node_proto
        self.node_id = node_proto.node_id
        self.name = node_proto.node_name
        self.fullname = ""
        self.input_ids = list(node_proto.input_tensor_id)
        self.output_ids = list(node_proto.output_tensor_id)
        self.workspace_ids = list(node_proto.workspace_tensor_id)
        self.inputs = []
        self.outputs = []
        self.workspaces = []
        self.allocations = 0
        self.deallocations = 0
        self.size = 0
        self.mem_change = 0

    def to_dict(self):
        """Convert Node to dict."""
        node = {
            'name': self.name,
            'fullname': self.fullname,
            'node_id': self.node_id,
            'allocations': self.allocations,
            'size': self.size,
            'allocated': self.mem_change,
            'inputs': self.inputs,
            'outputs': self.outputs
        }

        return node


class MemoryTensor:
    """
    A container for tensor.

    Args:
        tensor_proto (proto): Tensor proto.
    """
    def __init__(self, tensor_proto):
        self._tensor_proto = tensor_proto
        self.tensor_id = tensor_proto.tensor_id
        self.life_long = tensor_proto.life_long
        self.life_start = tensor_proto.life_start
        self.life_end = tensor_proto.life_end
        self.size = tensor_proto.size / GIGABYTES
        self.type = tensor_proto.type
        self.shape = ""
        self.format = ""
        self.dtype = ""
        self.source_node = ""
        self.name = ""

    def to_dict(self):
        """Convert Tensor to a dict."""
        tensor = {
            'tensor_name': self.name,
            'tensor_id': self.tensor_id,
            'size': self.size,
            'type': self.type,
            'shape': self.shape,
            'format': self.format,
            'data_type': self.dtype,
            'life_long': self.life_long,
            'life_start': self.life_start,
            'life_end': self.life_end
        }

        return tensor
