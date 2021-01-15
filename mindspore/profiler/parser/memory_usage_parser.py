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
import json
import os
import stat

from google.protobuf.text_format import ParseError

from mindspore import log as logger
from mindspore.profiler.common.exceptions.exceptions import ProfilerIOException, \
    ProfilerFileNotFoundException, ProfilerRawFileException
from mindspore.profiler.common.proto_files.memory_usage_pb2 import MemoryProto
from mindspore.profiler.common.validator.validate_path import validate_and_normalize_path
from mindspore.profiler.parser.container import MemoryGraph as Graph
from mindspore.profiler.parser.container import MemoryNode as Node
from mindspore.profiler.parser.container import MemoryQueue
from mindspore.profiler.parser.container import MemoryTensor as Tensor

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
            'static_mem': 0,
            'breakdowns': []
        }
        self._active_nodes = MemoryQueue(size=10)
        self._framework = {}

    def _get_file_path(self):
        """Get the proto file path."""
        file_path = os.path.join(
            self._profiling_dir,
            self._proto_file_path.format(self._device_id)
        )
        file_path = validate_and_normalize_path(file_path)

        if not os.path.exists(file_path):
            msg = 'The memory file does not exist!'
            logger.error(msg)
            raise ProfilerFileNotFoundException(msg=msg)

        return file_path

    def init_memory_usage_info(self, aicore_detail_data, points):
        """Init memory usage information."""
        logger.info("Start to load memory usage data from pb file")
        file_path = self._get_file_path()
        self._framework = self._process_framework_info(aicore_detail_data)

        try:
            with open(file_path, 'rb') as f:
                content = f.read()
        except (IOError, OSError) as err:
            logger.error('Failed to read memory file: %s', err)
            raise ProfilerIOException

        model_proto = MemoryProto()
        try:
            model_proto.ParseFromString(content)
        except ParseError as err:
            msg = "Fail to parse memory proto file."
            logger.error("Cannot parse the memory file. Please check the file schema.\n%s", err)
            raise ProfilerRawFileException(msg)

        graphs = model_proto.graph_mem
        self._graphs_dict = self._parse_graphs(graphs, points)
        self._mem_summary['capacity'] = model_proto.total_mem / GIGABYTES
        self._mem_summary['peak_mem'] = self._peak_mem
        self._process_memory_breakdowns()

        logger.info('Finished processing memory usage data.')

    def _parse_graphs(self, graphs, points):
        """Parse subgraphs."""
        graphs_dict = {}
        for graph_proto in graphs:
            graph_id = graph_proto.graph_id
            if graph_id is None:
                logger.info('Graph id is missing, skipped the graph.')
                continue

            graph = Graph(graph_proto)

            # process tensors in the graph
            tensors_proto = graph_proto.tensor_mems
            if not tensors_proto:
                logger.info('No tensor in graph %s, skipped.', graph_id)
                continue
            tensors_dict = self._parse_tensors(tensors_proto, graph_id)

            # calculate memory usage of the graph by number of nodes and details of tensors
            nodes_proto = graph_proto.node_mems
            # init memory usage list with static memory
            mem_change = [graph.static_mem for _ in range(len(nodes_proto))]
            self._calc_mem_change(mem_change, tensors_dict)
            graph.lines = mem_change

            # process nodes in graph
            graph.nodes = self._parse_nodes(
                nodes_proto, mem_change, tensors_dict, graph
            )

            # update fp_start and bp_end
            point_id = self._locate_fp_bp_id(points, graph.nodes)
            graph.fp_start = point_id.get('fp_start')
            graph.bp_end = point_id.get('bp_end')

            graphs_dict.update({graph_id: graph.to_dict()})

            self._mem_summary['static_mem'] += graph.static_mem
            self._mem_summary['allocations'] += len(tensors_dict) + 1
            self._mem_summary['deallocations'] += len(tensors_dict) + 1
            self._peak_mem = max(max(mem_change), self._peak_mem)

        return graphs_dict

    @staticmethod
    def _parse_tensors(tensors_proto, graph_id):
        """Parse tensors."""
        tensors_dict = {}
        for tensor_proto in tensors_proto:
            tensor = Tensor(tensor_proto, graph_id)
            tensors_dict.update({tensor.tensor_id: tensor})

        return tensors_dict

    def _parse_nodes(self, nodes_proto, mem_change, tensors_dict, graph):
        """Parse nodes."""
        nodes_dict = {}
        for index, node_proto in enumerate(nodes_proto):
            node = Node(node_proto, graph.graph_id)
            tensors = set(node.output_ids + node.workspace_ids)
            node.size = self._calc_node_memory(tensors, tensors_dict)
            node.allocations = len(tensors)
            node.deallocations = len(tensors)

            # calculate the allocated/deallocated memory size on the node
            if index == 0:
                node.mem_change = mem_change[index] - graph.static_mem
            else:
                node.mem_change = mem_change[index] - mem_change[index-1]

            self._update_nodes(node, tensors_dict)
            nodes_dict[node.name] = node.to_dict()

            # update active nodes
            self._active_nodes.push(
                item=(node.name, node.node_id, node.size, graph.graph_id),
                priority=-node.size  # priority is the negative value of node size
            )

        return nodes_dict

    def _update_nodes(self, node, tensors_dict):
        """Update nodes."""
        skipped = self._find_conflict_tensors(node)
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

        node.inputs = self._fill_tensor_dict(
            node.inputs, node.input_ids, tensors_dict, 'input'
        )
        node.outputs = self._fill_tensor_dict(
            node.outputs, node.output_ids, tensors_dict, 'output'
        )
        node.workspaces = self._fill_tensor_dict(
            node.workspaces, node.workspace_ids, tensors_dict, 'workspace', skipped
        )

    @staticmethod
    def _find_conflict_tensors(node):
        """Find conflict tensors in node."""
        output_list = []
        if node.output_ids:
            output_list = node.output_ids
        skipped = []
        if node.workspace_ids:
            for t_id in node.workspace_ids:
                if t_id in output_list:
                    skipped.append(t_id)

        return skipped

    @staticmethod
    def _fill_tensor_dict(node_ios, tensor_ids, tensors_dict, tensor_type, skipped=None):
        """Fill tensor dict."""
        full_list = []
        for t_id, io_dict in zip(tensor_ids, node_ios):
            if tensor_type == 'workspace' and t_id in skipped:
                continue
            tensor = tensors_dict.get(t_id)
            tensor.type = tensor_type
            io_dict.update(tensor.to_dict())
            full_list.append(io_dict)

        return full_list

    @staticmethod
    def _calc_node_memory(tensors, tensors_dict):
        """Calculate the allocated memory for the node."""
        node_mem = 0
        for t_id in tensors:
            tensor = tensors_dict[t_id]
            size = tensor.size
            node_mem += size

        return node_mem

    def _calc_mem_change(self, mem_change, tensors_dict):
        """Calculate the memory change for the subgraph."""
        node_num = len(mem_change)
        for tensor_id, tensor in tensors_dict.items():
            life_long = tensor.life_long
            life_start = tensor.life_start
            life_end = tensor.life_end
            size = tensor.size

            # Update memory change for the entire graph.
            # If a tensor's lifetime cannot be fully located, it will be ignored as 0 change.
            if life_long == 'LifeLongGraphAll':  # lifetime is from graph start to graph end
                tensor.life_start = 0
                tensor.life_end = node_num
                self._update_mem_change(mem_change, size, 0, node_num)
            elif life_long == 'LifeLongGraphStart':  # lifetime is from graph start to tensor end
                if life_end is not None and life_end >= 0:
                    tensor.life_start = 0
                    self._update_mem_change(mem_change, size, 0, life_end+1)
                else:
                    logger.info('Cannot locate lifetime end for tensor: %s', tensor_id)
            elif life_long == 'LifeLongGraphEnd':  # lifetime is from tensor start to graph end
                if life_start is not None and life_start <= node_num:
                    tensor.life_end = node_num
                    self._update_mem_change(mem_change, size, life_start, node_num)
                else:
                    logger.info('Cannot locate lifetime start for tensor: %s', tensor_id)
            elif life_long == 'LifeLongNone':  # lifetime is from tensor start to tensor end
                if life_start is not None and life_end is not None and life_start <= life_end:
                    self._update_mem_change(mem_change, size, life_start, life_end+1)
                else:
                    logger.info('Cannot locate lifetime start or end for tensor: %s', tensor_id)

    @staticmethod
    def _update_mem_change(mem_change, size, start, end):
        """Update memory change for the subgraph."""
        for i in range(start, end):
            mem_change[i] += size

    @staticmethod
    def _locate_fp_bp_id(points, nodes):
        """Locate the node id of fp_start and bp_end in graph."""
        point_id = {
            'fp_start': None,
            'bp_end': None
        }
        fp_start = points.get('fp_start')
        bp_end = points.get('bp_end')
        fp_name = fp_start.split('/')[-1] if fp_start else ""
        bp_name = bp_end.split('/')[-1] if bp_end else ""
        if fp_name in nodes:
            point_id['fp_start'] = nodes[fp_name].get('node_id')
        if bp_name in nodes:
            point_id['bp_end'] = nodes[bp_name].get('node_id')

        return point_id

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
        summary = self._summary_filename.format(self._device_id)
        self._write_memory_files(summary, self._mem_summary)

        # write memory details to json file
        details = self._details_filename.format(self._device_id)
        self._write_memory_files(details, self._graphs_dict)
        logger.info('Successfully write memory data into files.')

    def _process_memory_breakdowns(self):
        """Process memory breakdowns."""
        breakdowns = []
        active_nodes = self._active_nodes.get_items()
        for _, node_meta in active_nodes:
            node_name, _, _, graph_id = node_meta
            graph = self._graphs_dict[graph_id]
            nodes_dict = graph.get('nodes')
            node = nodes_dict.get(node_name)
            if 'inputs' in node:
                node.pop('inputs')
            breakdowns.append(node)

        self._mem_summary['breakdowns'] = breakdowns

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
