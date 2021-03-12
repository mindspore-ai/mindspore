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
# ============================================================================
"""Memory Usage Parser."""
from collections import OrderedDict
import json
import os
import stat

from google.protobuf.text_format import ParseError

from mindspore import log as logger
from mindspore.profiler.common.exceptions.exceptions import ProfilerIOException, \
    ProfilerFileNotFoundException, ProfilerRawFileException
from mindspore.profiler.common.validator.validate_path import validate_and_normalize_path
from mindspore.profiler.parser.container import MemoryGraph as Graph
from mindspore.profiler.parser.container import MemoryNode as Node
from mindspore.profiler.parser.container import MemoryTensor as Tensor
from mindspore.train.memory_profiling_pb2 import MemoryProto

GIGABYTES = 1024 * 1024 * 1024


class MemoryUsageParser:
    """MemoryUsageParser to parse memory raw data."""
    def __init__(self, profiling_dir, device_id):
        self._profiling_dir = profiling_dir
        self._device_id = device_id
        self._proto_file_path = 'memory_usage_{}.pb'
        self._summary_filename = 'memory_usage_summary_{}.json'
        self._details_filename = 'memory_usage_details_{}.json'
        self._graphs_dict = {}
        self._peak_mem = 0
        self._mem_summary = {
            'capacity': 0,
            'allocations': 0,
            'deallocations': 0,
            'peak_mem': 0,
            'static_mem': 0
        }
        self._framework = {}
        self._points = {}

    def _get_file_path(self):
        """Get the proto file path."""
        file_path = os.path.join(
            self._profiling_dir,
            self._proto_file_path.format(self._device_id)
        )
        file_path = validate_and_normalize_path(file_path)

        if not os.path.exists(file_path):
            logger.warning('The memory file does not exist! Please ignore the warning '
                           'if you are running heterogeneous training.')
            raise ProfilerFileNotFoundException(msg=file_path)

        return file_path

    def init_memory_usage_info(self, aicore_detail_data, points):
        """Init memory usage information."""
        logger.info("Start to load memory usage data from pb file")
        file_path = self._get_file_path()
        self._framework = self._process_framework_info(aicore_detail_data)
        self._points = points

        # Open memory protobuf file.
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
        except (IOError, OSError) as err:
            logger.error('Failed to read memory file: %s', err)
            raise ProfilerIOException

        # Parse memory raw data from file.
        memory_proto = MemoryProto()
        try:
            memory_proto.ParseFromString(content)
        except ParseError as err:
            msg = "Fail to parse memory proto file."
            logger.error("Cannot parse the memory file. Please check the file schema.\n%s", err)
            raise ProfilerRawFileException(msg)

        # Parse memory details based on graphs in the network.
        graphs = memory_proto.graph_mem
        self._parse_graph_memory(graphs)
        # Update memory summary information.
        self._mem_summary['capacity'] = memory_proto.total_mem / GIGABYTES
        self._mem_summary['peak_mem'] = self._peak_mem

        logger.info('Finished processing memory usage data.')

    def _parse_graph_memory(self, graphs):
        """Parse memory usage based on subgraphs."""
        for graph_proto in graphs:
            graph_id = graph_proto.graph_id
            if graph_id is None:
                logger.info('Graph id is missing, skipped the graph.')
                continue

            graph_parser = GraphMemoryParser(graph_proto, self._points, self._framework)
            graph = graph_parser.parse_graph()
            if graph:
                self._graphs_dict[graph_id] = graph

            # update global memory usage data
            self._peak_mem = max(self._peak_mem, graph_parser.peak_mem)
            self._mem_summary['static_mem'] += graph_parser.static_mem
            self._mem_summary['allocations'] += graph_parser.allocations
            self._mem_summary['deallocations'] += graph_parser.deallocations

    def _write_memory_files(self, filename, content):
        """Write the summary and top breakdowns of memory usage."""
        file_path = os.path.join(self._profiling_dir, filename)
        file_path = validate_and_normalize_path(file_path)

        try:
            with open(file_path, 'w') as json_file:
                json.dump(content, json_file)
                os.chmod(file_path, stat.S_IREAD | stat.S_IWRITE)
        except (IOError, OSError) as err:
            logger.error('Fail to write memory file.\n%s', err)
            raise ProfilerIOException

    def write_memory_files(self):
        """Write memory files."""
        logger.info('Start recording memory data into files...')
        # write memory summary to json file
        summary_filename = self._summary_filename.format(self._device_id)
        self._write_memory_files(summary_filename, self._mem_summary)

        # write memory details to json file
        details_filename = self._details_filename.format(self._device_id)
        self._write_memory_files(details_filename, self._graphs_dict)
        logger.info('Successfully write memory data into files.')

    @staticmethod
    def _process_framework_info(aicore_detail_data):
        """Process framework info."""
        framework_info_dict = {}
        for framework_obj in aicore_detail_data:
            op_name = framework_obj[0]
            op_full_name = framework_obj[4]
            op_info = framework_obj[5]
            framework_info_dict[op_name] = {
                'fullname': op_full_name,
                'name': op_name,
                'args': op_info
            }

        return framework_info_dict


class GraphMemoryParser:
    """Parse memory usage data for each graph."""
    def __init__(self, graph_proto, points, framework):
        self.graph = None
        self.nodes = OrderedDict()
        self.tensors = OrderedDict()
        self._framework = framework
        self._points = points
        self._graph_proto = graph_proto
        self.peak_mem = 0
        self.static_mem = 0
        self.allocations = 0
        self.deallocations = 0
        self._mem_change = []
        self.breakdowns = []
        self._lifetime = []

    def parse_graph(self):
        """Parse memory usage data for subgraphs."""
        graph_dict = {}
        self.graph = Graph(self._graph_proto)
        # process tensors in the graph
        tensors_proto = self._graph_proto.tensor_mems
        if not tensors_proto:
            logger.info('No tensor in graph %s, skipped.', self.graph.graph_id)
            return graph_dict
        self._parse_tensors(tensors_proto)

        # calculate memory usage of the graph by number of nodes and details of tensors
        nodes_proto = self._graph_proto.node_mems
        # init memory usage list with static memory
        self._mem_change = [self.graph.static_mem for _ in range(len(nodes_proto))]
        self._lifetime = [[] for _ in range(len(nodes_proto))]
        self._calc_mem_change()  # update self._mem_change and self._lifetime
        self.graph.lines = self._mem_change

        # process nodes in graph
        self.graph.nodes = self._parse_nodes(nodes_proto)

        self._process_memory_breakdowns()
        self.graph.breakdowns = self.breakdowns

        # update fp_start and bp_end
        point_id = self._locate_fp_bp_id()
        self.graph.fp_start = point_id.get('fp_start')
        self.graph.bp_end = point_id.get('bp_end')

        graph_dict = self.graph.to_dict()

        self.static_mem = self.graph.static_mem
        self.allocations = len(self.tensors)
        self.deallocations = len(self.tensors)
        self.peak_mem = max(max(self._mem_change), self.peak_mem)

        return graph_dict

    def _parse_tensors(self, tensors_proto):
        """Parse tensors."""
        for tensor_proto in tensors_proto:
            tensor = Tensor(tensor_proto)
            self.tensors.update({tensor.tensor_id: tensor})

    def _parse_nodes(self, nodes_proto):
        """Parse nodes."""
        nodes_list = []
        for index, node_proto in enumerate(nodes_proto):
            node = Node(node_proto)
            # Calculate memory size allocated for this node
            tensor_ids = set(node.output_ids + node.workspace_ids)
            node.size = self._calc_node_memory(tensor_ids)
            node.allocations = len(tensor_ids)
            node.deallocations = len(tensor_ids)

            # calculate the allocated/deallocated memory size on the node
            if index == 0:
                node.mem_change = self._mem_change[index] - self.graph.static_mem
            else:
                node.mem_change = self._mem_change[index] - self._mem_change[index-1]

            self._update_nodes(node)
            self._update_tensor_source(node)
            self.nodes[node.name] = node
            nodes_list.append(node.to_dict())

        return nodes_list

    def _update_nodes(self, node):
        """Update nodes."""
        # Remove duplicate tensors
        self._remove_duplicate_tensors(node)
        name = node.name
        if self._framework and name in self._framework:
            node_frame = self._framework[name]
            node.fullname = node_frame.get('fullname')
            info = node_frame.get('args')
            for key, value in info.items():
                if 'input' in key:
                    node.inputs.append(value)
                else:
                    node.outputs.append(value)

    def _update_tensor_source(self, node):
        """Update source node for tensors."""
        for t_id in node.output_ids:
            tensor = self.tensors.get(t_id)
            tensor.source_node = node.name

    @staticmethod
    def _remove_duplicate_tensors(node):
        """Find conflict tensors in node."""
        if node.workspace_ids:
            i = 0
            while i < len(node.workspace_ids):
                t_id = node.workspace_ids[i]
                if t_id in node.output_ids:
                    del node.workspace_ids[i]  # remove duplicate tensor
                    continue
                i += 1

    def _calc_node_memory(self, tensor_ids):
        """Calculate the allocated memory for the node."""
        node_mem = 0
        for t_id in tensor_ids:
            tensor = self.tensors[t_id]
            size = tensor.size
            node_mem += size

        return node_mem

    def _calc_mem_change(self):
        """Calculate the memory change for the subgraph."""
        node_num = len(self._mem_change)
        for tensor_id, tensor in self.tensors.items():
            life_long = tensor.life_long
            life_start = tensor.life_start
            life_end = tensor.life_end
            size = tensor.size

            # Update memory change for the entire graph.
            # If a tensor's lifetime cannot be fully located, it will be ignored as 0 change.
            if life_long == 'LifeLongGraphAll':  # lifetime is from graph start to graph end
                tensor.life_start = 0
                tensor.life_end = node_num
                self._update_mem_change(size, 0, node_num, tensor_id)
            elif life_long == 'LifeLongGraphStart':  # lifetime is from graph start to tensor end
                if life_end is not None and life_end >= 0:
                    tensor.life_start = 0
                    self._update_mem_change(size, 0, life_end+1, tensor_id)
                else:
                    logger.info('Cannot locate lifetime end for tensor: %s', tensor_id)
            elif life_long == 'LifeLongGraphEnd':  # lifetime is from tensor start to graph end
                if life_start is not None and life_start <= node_num:
                    tensor.life_end = node_num
                    self._update_mem_change(size, life_start, node_num, tensor_id)
                else:
                    logger.info('Cannot locate lifetime start for tensor: %s', tensor_id)
            elif life_long == 'LifeLongNone':  # lifetime is from tensor start to tensor end
                if life_start is not None and life_end is not None and life_start <= life_end:
                    self._update_mem_change(size, life_start, life_end+1, tensor_id)
                else:
                    logger.info('Cannot locate lifetime start or end for tensor: %s', tensor_id)

    def _update_mem_change(self, size, start, end, tensor_id):
        """Update memory change for the subgraph."""
        for i in range(start, end):
            self._mem_change[i] += size
            # Update tensor lifetime list.
            self._lifetime[i].append(tensor_id)

    def _locate_fp_bp_id(self):
        """Locate the node id of fp_start and bp_end in graph."""
        point_id = {
            'fp_start': None,
            'bp_end': None
        }
        fp_start = self._points.get('fp_start')
        bp_end = self._points.get('bp_end')
        fp_name = fp_start.split('/')[-1] if fp_start else ""
        bp_name = bp_end.split('/')[-1] if bp_end else ""
        if fp_name in self.nodes:
            point_id['fp_start'] = self.nodes[fp_name].node_id
        if bp_name in self.nodes:
            point_id['bp_end'] = self.nodes[bp_name].node_id

        return point_id

    def _process_memory_breakdowns(self):
        """Process memory breakdowns for each node."""
        self.breakdowns = [[] for _ in range(len(self.nodes))]
        for index, breakdown in enumerate(self._lifetime):
            for t_id in breakdown:
                tensor = self.tensors.get(t_id)
                source_node = tensor.source_node
                if not source_node:
                    continue
                node = self.nodes.get(source_node)
                for i, output_id in enumerate(node.output_ids):
                    if t_id == output_id:
                        output = node.outputs[i] if i < len(node.outputs) else {}
                        tensor.name = node.name + ':' + str(i)
                        tensor.shape = output.get('shape')
                        tensor.dtype = output.get('data_type')
                        tensor.format = output.get('format')
                        tensor.type = 'output'

                tensor_dict = tensor.to_dict()
                self.breakdowns[index].append(tensor_dict)
