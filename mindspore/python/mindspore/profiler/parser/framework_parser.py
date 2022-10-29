# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
"""The parser for parsing framework files."""
import csv
import os
import re
import struct
import json
from pathlib import Path
from typing import List
from collections import defaultdict
from collections import namedtuple
import glob
import numpy as np

from mindspore import log as logger
from mindspore.profiler.parser.framework_struct import TASK_DESC_STRUCT, TENSOR_DATA_STRUCT, STEP_INFO_STRUCT
from mindspore.profiler.parser.framework_enum import VmDataType, VmFormat, FileDataType, MSPROF_DIFFERENCE
from mindspore.profiler.parser.framework_enum import MSPROF_MIX_DATA_STRING
from mindspore.profiler.common.struct_type import StructType
from mindspore.profiler.common.util import combine_stream_task_id
from mindspore.profiler.common.exceptions.exceptions import ProfilerDirNotFoundException
from mindspore.profiler.common.exceptions.exceptions import ProfilerFileNotFoundException
from mindspore.profiler.common.exceptions.exceptions import ProfilerParamValueErrorException
from mindspore.profiler.common.validator.validate_path import validate_and_normalize_path
from mindspore.profiler.parser.profiler_info import ProfilerInfo

FILE_DATA_STRUCT_DICT = {
    FileDataType.STEP_INFO.value: STEP_INFO_STRUCT,
    FileDataType.TENSOR_DATA_INFO.value: TENSOR_DATA_STRUCT,
    FileDataType.TASK_DESC_INFO.value: TASK_DESC_STRUCT
}

COL_NAMES = ['task_id', 'stream_id', 'block_dim', 'full_op_name', 'op_name', 'op_type', 'subgraph', 'op_info',
             'graph_id']
OpData = namedtuple('OpData', field_names=COL_NAMES)


class FrameworkParser:
    """
    The parser for parsing framework files.

    Args:
        profiling_path (str): The profiling path which should contain CANN profiling data.
        rank_id (str): The rank ID.
        output_path (str): The directory of the parsed file. Default: `./`.
    """
    _regex_framework = r'Framework\.(?P<data_type>.+)\.(?P<device_id>\d).+'
    _graph_attr_name = [
        'input_format', 'input_data_type', 'input_shape', 'output_format',
        'output_data_type', 'output_shape'
    ]
    output_file_format = 'framework_raw_{rank_id}.csv'

    def __init__(self, profiling_path, rank_id, output_path='./'):
        self._profiling_path = profiling_path
        self._output_path = output_path
        self._rank_id = rank_id

        self._hash_dict = {}
        self._task_id_full_op_name_dict = {}
        self._point_info = {}

    @property
    def save_path(self):
        """
        The property of save path.

        Returns:
            str, the save path.
        """
        return os.path.realpath(os.path.join(self._output_path, self.output_file_format.format(rank_id=self._rank_id)))

    @property
    def point_info(self):
        """
        The property of the framework point information.

        Returns:
            dict, the framework point information, key is tag, value is op name.
        """
        # Note: In the multi-subgraph or multi-tag scenario, op name is overwritten.
        return self._point_info

    @staticmethod
    def _check_output_path(path):
        if not os.path.exists(path) or not os.path.isdir(path):
            raise ProfilerDirNotFoundException(path)

    @staticmethod
    def _parse_hash_dic(framework_path_dict):
        """Parse the hash dic files, and return a hash value map op name dict."""
        hash_op_dict = {}
        for path in framework_path_dict[FileDataType.HASH_DIC.value]:
            with open(path, 'r') as file:
                for hash_str in file:
                    hash_value, op_name = hash_str.strip().split(':')
                    hash_op_dict[hash_value] = op_name
        return hash_op_dict

    @staticmethod
    def _special_process_tensor_data(item_binary_data, data_type, tensor_num):
        """The tensor data depends tensor num, so need to special process."""
        start = 0
        op_attr_struct = data_type[0]
        op_attr_size = StructType.sizeof(op_attr_struct)
        unpack_data = []

        for _ in range(tensor_num):
            buffer = item_binary_data[start:start + op_attr_size]
            values = struct.unpack(StructType.format(op_attr_struct), buffer)
            one_data = dict(
                tensorType=values[0],
                format=values[1],
                dataType=values[2],
                shape=list(filter(lambda x: x != 0, values[3:]))
            )
            unpack_data.append(one_data)
            start += op_attr_size

        return unpack_data

    @staticmethod
    def _special_process_tensor_num(item_binary_data, data_type):
        """The memory of tensorNum is aligned, so here need to special process"""
        cursor = 0
        tensor_num_struct = data_type[0]
        size = StructType.sizeof(tensor_num_struct)
        unpack_data = struct.unpack(tensor_num_struct.value, item_binary_data[cursor:cursor + size])[0]
        return unpack_data

    @staticmethod
    def _construct_task_id_full_op_name_dict(task_desc_info):
        """The task desc info is a list[task_desc], task_desc is a dict, key is same as TASK_DESC_STRUCT."""
        task_id_full_op_name = {}
        for task_desc in task_desc_info:
            task_id = combine_stream_task_id(task_desc['streamId'], task_desc['taskId'])
            task_id_full_op_name[task_id] = task_desc['opName']
        return task_id_full_op_name

    @staticmethod
    def _construct_point_info(task_id_full_op_name_dict, step_point_data):
        """step_point_data is a list[step_data], step data is a dict, key is same as STEP_INFO_STRUCT."""
        point_info = {}
        for step_point in step_point_data:
            task_id = combine_stream_task_id(step_point['streamId'], step_point['taskId'])
            tag = step_point['tag']
            full_op_name = task_id_full_op_name_dict[task_id]
            point_info[tag] = full_op_name
        return point_info

    @staticmethod
    def _get_vm_data_type(msprof_data_type):
        """Get the mapped vm data type of msprof."""
        if msprof_data_type >= MSPROF_DIFFERENCE:
            return msprof_data_type - MSPROF_DIFFERENCE
        return msprof_data_type

    @staticmethod
    def _get_vm_op_format(msprof_op_format):
        """Get the mapped op format type of msprof."""
        if msprof_op_format >= MSPROF_DIFFERENCE:
            return msprof_op_format - MSPROF_DIFFERENCE
        return msprof_op_format

    @staticmethod
    def _construct_task_id_op_attr_dict(prof_tensor_data):
        """prof_tensor_data is a list[tensor_data], tensor_data is a dict, key is same as TENSOR_DATA_STRUCT."""
        task_id_op_attr_dict = defaultdict(list)
        for tensor_data in prof_tensor_data:
            task_id = combine_stream_task_id(tensor_data['streamId'], tensor_data['taskId'])
            for tensor_attr in tensor_data['tensorData']:
                tensor_type = 'input' if tensor_attr['tensorType'] == 0 else 'output'
                tensor_format = VmFormat.get_format_name(FrameworkParser._get_vm_data_type(tensor_attr['format']))
                op_attr = dict(
                    tensor_type=tensor_type,
                    format=tensor_format,
                    data_type=VmDataType.get_data_type_name(FrameworkParser._get_vm_op_format(tensor_attr['dataType'])),
                    shape=tensor_attr['shape']
                )
                task_id_op_attr_dict[task_id].append(op_attr)

        for task_id, op_attrs in task_id_op_attr_dict.items():
            input_count = 0
            output_count = 0
            new_op_attr = {}
            for op_attr in op_attrs:
                if op_attr['tensor_type'] == 'input':
                    op_attr.pop('tensor_type')
                    new_op_attr[f'input_{input_count}'] = op_attr
                    input_count += 1
                else:
                    op_attr.pop('tensor_type')
                    new_op_attr[f'output_{output_count}'] = op_attr
                    output_count += 1
            task_id_op_attr_dict[task_id] = new_op_attr

        return task_id_op_attr_dict

    @staticmethod
    def _write_framework_to_file(all_op_data: List[OpData], output_file):
        with open(output_file, 'w') as file_handler:
            csv_writer = csv.writer(file_handler)
            csv_writer.writerow(COL_NAMES)
            csv_writer.writerows(all_op_data)

    @staticmethod
    def _get_subgraph_name(full_op_name):
        """
        Get subgraph name.

        Args:
            full_op_name (str): The full operator name.

        Returns:
            str, the subgraph name.
        """
        subgraph_name = full_op_name.split('/', 1)[0]
        if subgraph_name in ['Default', 'Gradients']:
            return subgraph_name
        return None

    def check_op_name(self, op_name, is_prefix=True):
        """
        Check whether the operator name exists.

        Args:
            op_name (str): The operator name or operator name prefix.
            is_prefix (bool): `True` if the op_name is prefix, else `False`.
                Default: True.

        Returns:
            bool, `True` if the operator name does exist in framework file, else
            `False`.
        """
        if not op_name:
            raise ProfilerParamValueErrorException('The op_name should exist.')
        for full_op_name in self._task_id_full_op_name_dict.values():
            if full_op_name:
                if is_prefix and full_op_name.startswith(op_name):
                    return True
                if not is_prefix and op_name == full_op_name:
                    return True
        return False

    def to_task_id_full_op_name_dict(self):
        """
        Get the task id and full operator name dict.

        Returns:
            dict, the task id and full operator name dict.
        """
        return self._task_id_full_op_name_dict

    def parse(self):
        """Parse the framework files."""
        framework_path_dict = self._search_file(self._profiling_path)
        self._hash_dict = self._parse_hash_dic(framework_path_dict)

        all_file_data = self._parse_binary_data(framework_path_dict)
        task_id_full_op_name_dict = self._construct_task_id_full_op_name_dict(
            all_file_data[FileDataType.TASK_DESC_INFO.value])
        point_info = self._construct_point_info(task_id_full_op_name_dict, all_file_data[FileDataType.STEP_INFO.value])
        task_id_op_attr_dict = self._construct_task_id_op_attr_dict(all_file_data[FileDataType.TENSOR_DATA_INFO.value])

        self._point_info = point_info
        self._task_id_full_op_name_dict = task_id_full_op_name_dict

        all_op_data = self._construct_op_data_to_file(all_file_data[FileDataType.TASK_DESC_INFO.value],
                                                      task_id_op_attr_dict)

        self._write_framework_to_file(all_op_data, output_file=self.save_path)

    def _search_file(self, profiling_path):
        """
        Search all framework files in raw profiling path.

        Args:
            profiling_path (str): This profiling path should contain data dir.

        Return:
            dict, return a dict container all framework file paths. Format is {FileDataType: [file paths]}.

        Raises:
            ProfilerFileNotFoundException: If the framework files are not found.
        """
        data_dir = os.path.join(profiling_path, 'data')
        if not os.path.isdir(data_dir):
            raise ProfilerDirNotFoundException(data_dir)

        framework_path_dict = defaultdict(list)
        for file in Path(data_dir).glob(r'Framework*[0-9]'):
            file_name = file.name
            match = re.search(self._regex_framework, file_name)
            if match is None:
                logger.warning("Profiler does not support to analyse file(%s), this file name format is not %s, "
                               "skip this file.", file.resolve(), self._regex_framework)
                continue

            if match['data_type'] not in FileDataType.members():
                logger.warning("Profiler does not support to analyse file(%s), this file data type is %s, "
                               "skip this file.", file.resolve(), match['data_type'])
                if match['data_type'].startswith('vm'):
                    raise RuntimeError("The current profiler file is generated by MindSpore 1.5 or earlier. Use "
                                       "MindSpore 1.5 or the matching MindSpore version to parse the profiler file.")
                continue

            framework_path_dict[match['data_type']].append(file.resolve())

        empty_files = [data_type for data_type, files in framework_path_dict.items() if not files]
        if not framework_path_dict or empty_files:
            if empty_files:
                logger.error("Can not find %s files when parse profiler framework file.", ','.join(empty_files))
            raise ProfilerFileNotFoundException('Framework')

        for data_type in FileDataType.members():
            if data_type not in framework_path_dict:
                logger.warning("Can not find %s file when parse profiler framework file.", data_type)
                continue
            framework_path_dict[data_type].sort()

        return framework_path_dict

    def _parse_binary_data(self, framework_path_dict):
        """Parse binary data in the FILE_DATA_STRUCT_DICT from given files, such as task data, step point data"""
        all_file_data = defaultdict(list)
        for file_data_type, data_struct in FILE_DATA_STRUCT_DICT.items():
            line_size = StructType.sizeof(data_struct.values())
            for path in framework_path_dict[file_data_type]:
                with open(path, 'rb') as file_handler:
                    while True:
                        binary_data = file_handler.read(line_size)
                        if len(binary_data) < line_size:
                            break
                        line_data = StructType.unpack_binary_data(data_struct, binary_data,
                                                                  self._special_process_binary_data)
                        all_file_data[file_data_type].append(line_data)
        return all_file_data

    def _special_process_binary_data(self, item_binary_data, data_name, data_type, unpacked_data):
        """Specially processes binary data."""
        unpack_data = None
        success = False
        if isinstance(data_type, list):
            if data_name in ('opName', 'opType'):
                unpack_data = self._special_process_mixed_data(item_binary_data)
            elif data_name == 'tensorData':
                tensor_num = unpacked_data['tensorNum']
                unpack_data = self._special_process_tensor_data(item_binary_data, data_type, tensor_num)
            elif data_name == 'tensorNum':
                unpack_data = self._special_process_tensor_num(item_binary_data, data_type)
            else:
                # skip reserve data
                unpack_data = None
            success = True
        return unpack_data, success

    def _special_process_mixed_data(self, item_binary_data):
        """Specially processes mixed data, for example, opName and opType"""
        # The first byte is type flag, 0 means data is string, 1 means data is hash value
        cursor = 0
        data_size = len(item_binary_data)
        flag = struct.unpack(StructType.UINT8.value, item_binary_data[cursor:cursor + 1])[0]

        # skip rsv data, rsv has 7 bytes
        skip_size = 8
        remain_size = data_size - skip_size
        if flag == MSPROF_MIX_DATA_STRING:
            unpack_data = struct.unpack(StructType.CHAR.value * remain_size,
                                        item_binary_data[cursor + skip_size:cursor + data_size])
            unpack_data = ''.join(list(map(lambda c: c.decode(), filter(lambda c: c != b'\x00', unpack_data))))
        else:
            size = StructType.sizeof(StructType.UINT64) + skip_size
            hash_value = struct.unpack(StructType.UINT64.value,
                                       item_binary_data[cursor + skip_size:cursor + size])[0]
            unpack_data = self._hash_dict[str(hash_value)]
        return unpack_data

    def _construct_op_data_to_file(self, task_desc_info, task_id_op_attr_dict):
        """Build data written to a file."""
        all_op_data = []
        graph_ids = set()
        for task_desc in task_desc_info:
            task_id = task_desc['taskId']
            full_op_name = task_desc['opName']
            subgraph = self._get_subgraph_name(full_op_name)
            combined_task_id = combine_stream_task_id(task_desc['streamId'], task_id)
            op_data = OpData(task_id=task_id,
                             stream_id=task_desc['streamId'],
                             block_dim=task_desc['blockDims'],
                             full_op_name=full_op_name,
                             op_name=full_op_name.split('/')[-1],
                             op_type=task_desc['opType'],
                             subgraph=subgraph,
                             op_info=json.dumps(task_id_op_attr_dict.get(combined_task_id, {})),
                             graph_id=task_desc['modelId'])
            graph_ids.add(task_desc['modelId'])
            all_op_data.append(op_data)
        ProfilerInfo.set_graph_ids(list(graph_ids))
        return all_op_data


class GpuFrameWorkParser:
    """
    The parser for parsing framework files.

    Args:
        output_path (str): The profiling path which should contain GPU profiling data.
        dev_id (str): The device ID.
    """

    _STEPS_TID = 100000
    _GPU_OP_TID = 100002

    def __init__(self, output_path, dev_id, op_names=None):
        """The parser for parsing framework files."""
        self._dev_id = dev_id
        self._output_path = output_path
        self.op_names = op_names
        self.op_name = ''
        self.framework_list = []
        self.op_detail = {}
        self.operation_info = {}
        self.activity_info_dir = []
        self.framework_info_dir = []
        self.cpu_detail_info_dir = []
        self.gpu_detail_info_dir = []
        self.op_execute_times = {}
        self.op_step_shape_info = defaultdict(list)
        self.one_step_op_time = dict()
        self.one_step_kernel_time = dict()

    def parse(self):
        """Parse op performance data."""
        self.get_device_target_filename()
        self.get_framework_summary()
        self.get_cpu_op_detail_info()
        self.get_activity_op_info()
        if isinstance(self.op_names, str):
            self.combine_performance_data(self.op_names)
        elif isinstance(self.op_names, list):
            for op_name in self.op_names:
                self.combine_performance_data(op_name)
        self.operation_info["device_id"] = self._dev_id
        return json.dumps(self.operation_info)

    def get_framework_summary(self):
        """Get framework data."""
        for filename in self.framework_info_dir:
            op_side = filename.split('_')[0]
            framework_file_path = os.path.join(self._output_path, filename)
            framework_file_path = validate_and_normalize_path(framework_file_path)
            with open(framework_file_path, 'r') as f_obj:
                framework_info = f_obj.readlines()
            for line_info in framework_info:
                line_info = line_info.strip(' ').strip('\n').split(';')
                # line_info[0]: op_type, line_info[1]: op_name, line_info[2]: graph_id, line_info[3]: input_shape;
                input_shape = line_info[3:]
                item = [line_info[0], line_info[1], input_shape, op_side]
                if not self.op_step_shape_info.get(line_info[1]):
                    self.op_step_shape_info[line_info[1]].append(op_side)
                self.op_step_shape_info[line_info[1]].append(input_shape)
                if item not in self.framework_list:
                    self.framework_list.append(item)

    def get_cpu_op_detail_info(self):
        """Get cpu operators detail data."""
        for filename in self.cpu_detail_info_dir:
            op_side = filename.split('_')[0]
            op_detail_file_path = os.path.join(self._output_path, filename)
            op_detail_file_path = validate_and_normalize_path(op_detail_file_path)
            with open(op_detail_file_path, 'r') as f_obj:
                op_detail_info = f_obj.readlines()
            for line_info in op_detail_info[1:]:
                line_info = line_info.strip(' ').strip('\n').split(',')
                if not self.op_detail.get(line_info[2]):
                    # line_info[4]: op_occurrences, line_info[5]: op_detail_time(us), line_info[6]: op_avg_time(us);
                    self.op_detail[line_info[2]] = [line_info[4], line_info[5], line_info[6], op_side]

    def get_execute_times(self):
        """Get gpu operators execute times."""
        if self.gpu_detail_info_dir:
            gpu_op_detail_file_path = os.path.join(self._output_path, self.gpu_detail_info_dir[0])
            gpu_op_detail_file_path = validate_and_normalize_path(gpu_op_detail_file_path)
            with open(gpu_op_detail_file_path, 'r') as fp:
                op_detail_info = fp.readlines()
                for line_info in op_detail_info[1:]:
                    line_info = line_info.strip(' ').strip('\n').split(',')
                    self.op_execute_times[line_info[2]] = line_info[4]

    def get_activity_op_info(self):
        """Get op detail data."""
        all_file = os.listdir(self._output_path)
        for file_name in all_file:
            if file_name.startswith('gpu_op_detail') and file_name.endswith(f'{self._dev_id}.csv'):
                self.gpu_detail_info_dir.append(file_name)
        if not self.gpu_detail_info_dir and self.activity_info_dir:
            raise RuntimeError(f'The output file <%s> is not found.' % self.gpu_detail_info_dir)
        self.get_execute_times()
        for filename in self.activity_info_dir:
            op_side = filename.split('_')[0]
            activity_file_path = os.path.join(self._output_path, filename)
            activity_file_path = validate_and_normalize_path(activity_file_path)
            with open(activity_file_path, 'r') as file:
                activity_info = file.readlines()
            for line_info in activity_info[1:]:
                line_info = line_info.strip(' ').strip('\n').replace(', ', ';').split(',')
                op_name = line_info[2].split('/')[-1]
                op_occurrences = int(self.op_execute_times.get(op_name))
                op_total_time = float(line_info[-4])
                if not self.op_detail.get(op_name):
                    # line_info[4]: op_occurrences, line_info[5]: op_detail_time(us), line_info[6]: op_avg_time(us);
                    self.op_detail[op_name] = [op_occurrences, op_total_time,
                                               round(op_total_time / op_occurrences, 4), op_side]
                else:
                    self.op_detail.get(op_name)[1] += op_total_time
                    self.op_detail.get(op_name)[2] = self.op_detail.get(op_name)[1] / self.op_detail.get(op_name)[0]
                    self.op_detail[op_name] = [self.op_detail.get(op_name)[0],
                                               round(self.op_detail.get(op_name)[1], 4),
                                               round(self.op_detail.get(op_name)[2], 4), op_side]

    def combine_performance_data(self, op_name):
        """Combine operator detail info with framework info."""
        unique_op_info = []
        op_shape_dict = {}
        operation_info = {}
        factor = 1000  # convert time unit from ms to us.
        for line_info in self.framework_list:
            op_detail = self.op_detail.get(line_info[1])
            if not op_detail:
                continue
            if op_name in line_info and line_info[3] == op_detail[3]:
                op_side = line_info[3]
                op_shape = '[{}]{}'.format(op_side, ','.join(line_info[2]))
                op_occurrences = int(op_detail[0])
                op_total_time = float(op_detail[1])
                op_avg_time = float(op_detail[2])
                if op_shape in op_shape_dict.keys():
                    # Classify according to the operator information of the same shape.
                    op_shape_dict.get(op_shape)[0] += op_occurrences
                    op_shape_dict.get(op_shape)[1] += op_total_time
                    op_shape_dict.get(op_shape)[2] = op_shape_dict.get(op_shape)[1] / op_shape_dict.get(op_shape)[0]
                    op_shape_dict[op_shape] = [op_shape_dict.get(op_shape)[0], round(op_shape_dict.get(op_shape)[1], 4),
                                               round(op_shape_dict.get(op_shape)[2], 4), op_side]
                else:
                    op_shape_dict[op_shape] = [op_occurrences, op_total_time, op_avg_time, op_side]

        for input_shape in op_shape_dict:
            # 0: op_occurrences, 1: op_total_time, 2: op_avg_time, 3: op_side
            operation_info['op_side'] = op_shape_dict.get(input_shape)[3]
            operation_info['input_shape'] = input_shape.strip('[').split(']')[-1]
            operation_info['op_occurrences'] = op_shape_dict.get(input_shape)[0]
            if operation_info.get('op_side') == 'cpu':
                operation_info['op_total_time(us)'] = round(op_shape_dict.get(input_shape)[1] * factor, 4)
                operation_info['op_avg_time(us)'] = round(op_shape_dict.get(input_shape)[2] * factor, 4)
            else:
                operation_info['op_total_time(us)'] = op_shape_dict.get(input_shape)[1]
                operation_info['op_avg_time(us)'] = op_shape_dict.get(input_shape)[2]
            unique_op_info.append(operation_info)
            operation_info = dict()

        if unique_op_info:
            self.operation_info[op_name] = unique_op_info
        else:
            raise RuntimeError(f'The information of <{op_name}> is not found. Please verify that the operator name is'
                               f' correct or the operator is used in the network.')

    def get_device_target_filename(self):
        """Get device target filename."""
        gpu_framework_file = f'gpu_framework_{self._dev_id}.txt'
        cpu_framework_file = f'cpu_framework_{self._dev_id}.txt'
        gpu_activity_file = f'gpu_activity_data_{self._dev_id}.csv'
        cpu_op_detail_file = f'cpu_op_detail_info_{self._dev_id}.csv'
        all_file = os.listdir(self._output_path)
        if not all_file:
            raise RuntimeError(f'No profiler file is found in the path <%s>. '
                               f'Check whether the profiler path is correct.' % self._output_path)
        if gpu_activity_file in all_file and gpu_framework_file not in all_file:
            raise RuntimeError(f'The output file <%s> is not found.' % gpu_framework_file)
        if cpu_op_detail_file in all_file and cpu_framework_file not in all_file:
            raise RuntimeError(f'The output file <%s> is not found.' % cpu_framework_file)
        if gpu_framework_file in all_file and gpu_activity_file not in all_file:
            raise RuntimeError(f'The output file <%s> is not found.' % gpu_activity_file)
        if cpu_framework_file in all_file and cpu_op_detail_file not in all_file:
            raise RuntimeError(f'The output file <%s> is not found.' % cpu_op_detail_file)
        if gpu_activity_file not in all_file and cpu_op_detail_file not in all_file:
            raise RuntimeError(f'The profiling data of this card which device_id is equal to {self._dev_id} does not'
                               f' exist. Check whether device_id is correct.')
        for file_name in all_file:
            if file_name.endswith(f'activity_data_{self._dev_id}.csv'):
                self.activity_info_dir.append(file_name)
            if file_name.endswith(f'framework_{self._dev_id}.txt'):
                self.framework_info_dir.append(file_name)
            if file_name.startswith('cpu_op_detail') and file_name.endswith(f'{self._dev_id}.csv'):
                self.cpu_detail_info_dir.append(file_name)

    def analyse_dynamic_shape_data(self, timeline_meta):
        """Analyse gpu operators's information and cudakernel's information."""
        kernel_info = defaultdict(list)
        operator_info = defaultdict(list)
        kernel_type_step_time = dict()
        op_type_step_time = dict()
        step, first_update = 1, 0
        self.get_device_target_filename()
        self.get_framework_summary()
        for op_info in timeline_meta:
            args = op_info.get("args", {})
            if op_info.get("tid") == self._STEPS_TID and op_info.get('dur'):
                step = int(op_info.get("name"))
                if first_update:
                    self.one_step_op_time = dict()
                    self.one_step_kernel_time = dict()
                first_update = 1
            elif args and args.get("type") == "cuLaunchKernel":
                item = self._organize_result(step, op_info, args)
                kernel_info[step].append(item)
                self._get_one_step_info(item, "kernel")
            elif (op_info.get("tid") == self._GPU_OP_TID and not op_info.get("cat")) or \
                    str(op_info.get("tid")).startswith('HostCpu'):
                item = self._organize_result(step, op_info, args)
                operator_info[step].append(item)
                self._get_one_step_info(item, "operator")
            op_type_step_time[step] = self.one_step_op_time
            kernel_type_step_time[step] = self.one_step_kernel_time
        self.write_dynamic_shape_data(operator_info, kernel_info, op_type_step_time, kernel_type_step_time)

    def write_dynamic_shape_data(self, operator_info, kernel_info, op_type_step_time, kernel_type_step_time):
        """Organize the result."""
        output_dynamic_shape_file_name = f"dynamic_shape_info_{self._dev_id}.json"
        result = {
            "operator": operator_info,
            "kernel": kernel_info,
            "operator_type": op_type_step_time,
            "kernel_type": kernel_type_step_time,
        }
        dynamic_shape_file_path = os.path.join(self._output_path, output_dynamic_shape_file_name)
        with os.fdopen(os.open(dynamic_shape_file_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o660), 'w') as fp:
            json.dump(result, fp)

    def get_graph_ids(self):
        """Get gpu graph ids."""
        gpu_framework_file = list(glob.glob(os.path.join(self._output_path,
                                                         'gpu_framework_{}.txt'.format(self._dev_id))))
        if not gpu_framework_file:
            return []
        graph_ids = set()
        with open(gpu_framework_file[0], 'r') as f_obj:
            framework_info = f_obj.readlines()
        for line_info in framework_info:
            if line_info.startswith("InitDataSetQueue"):
                continue
            line_info = line_info.strip(' ').strip('\n').split(';')
            if len(line_info) > 2 and line_info[2].isdigit():
                graph_ids.add(int(line_info[2]))
        return list(graph_ids)

    def _organize_result(self, step, op_info, args):
        """Organize the results."""
        if args.get("type", "") == "cuLaunchKernel":
            item = {
                "step": step,
                "op_type": args.get("type"),
                "op_name": op_info.get('name'),
                "op_full_name": args.get('op_full_name'),
                "dur": op_info.get('dur'),
                "block_dim": args.get('block_dim'),
                "grid_dim": args.get('grid_dim')
            }
        else:
            item = {
                "step": step,
                "op_side": self.op_step_shape_info.get(op_info.get('name'))[0],
                "op_type": op_info.get('name').split('-')[0],
                "op_name": op_info.get('name'),
                "dur": op_info.get('dur'),
                "shape_info": self.op_step_shape_info.get(op_info.get('name'))[step],
            }
        return item

    def _get_one_step_info(self, item, op_type):
        """Get operator type information in step."""
        duration = item.get("dur")
        if op_type == "operator":
            sort_type = item.get("op_type")
            if not self.one_step_op_time.get(sort_type):
                # duration, times, avg_time
                self.one_step_op_time[sort_type] = [duration, 1, duration]
            else:
                self.one_step_op_time[sort_type][0] += duration
                self.one_step_op_time[sort_type][1] += 1
                self.one_step_op_time[sort_type] = [self.one_step_op_time[sort_type][0],
                                                    self.one_step_op_time[sort_type][1],
                                                    round(self.one_step_op_time[sort_type][0] /
                                                          self.one_step_op_time[sort_type][1], 4)]
        else:
            sort_type = item.get("op_name")
            op_full_name = item.get("op_full_name")
            if not self.one_step_kernel_time.get(sort_type):
                # duration, times, avg_time
                self.one_step_kernel_time[sort_type] = [duration, 1, duration, op_full_name]
            else:
                self.one_step_kernel_time[sort_type][0] += duration
                self.one_step_kernel_time[sort_type][1] += 1
                self.one_step_kernel_time[sort_type] = [self.one_step_kernel_time[sort_type][0],
                                                        self.one_step_kernel_time[sort_type][1],
                                                        round(self.one_step_kernel_time[sort_type][0] /
                                                              self.one_step_kernel_time[sort_type][1], 4),
                                                        op_full_name]


class DynamicFrameWorkParser:
    """
    Thr parser for parsing dynamic shape framework files.

    Args:
        output_path (str): The profiling path which should contain Ascend profiling data.
        rank_id (int): The rank ID.
    """

    def __init__(self, output_path, rank_id):
        """Initialization of parsing framework data."""
        self._output_path = output_path
        self._all_op_exe_time = defaultdict(list)
        self._op_shape_info = defaultdict(list)
        self._op_info = dict()
        self._rank_id = rank_id
        self._op_type_exe_time = defaultdict(list)
        self._exe_time_and_shape_detail = defaultdict(dict)
        self._dynamic_shape_info = defaultdict(list)
        self._step = 0

    def write_dynamic_shape_data(self):
        """Analyze dynamic shape data and write to dynamic shape file."""
        self._get_total_step_num()
        output_dynamic_shape_file_name = f'dynamic_shape_info_{self._rank_id}.json'
        for op_name in self._exe_time_and_shape_detail:
            if self._exe_time_and_shape_detail[op_name]['op_exe_occurrences'] == self._step:
                self._op_info[op_name] = self._exe_time_and_shape_detail.get(op_name)
        for op_name, op_detail in self._op_info.items():
            op_type = op_name.split('-', maxsplit=1)[0]
            item = {op_name: op_detail}
            self._dynamic_shape_info[op_type].append(item)
        self._op_info["op_type"] = dict()
        for op_name in self._op_info:
            if op_name != 'op_type':
                op_type = op_name.split('-')[0]
                self._op_type_exe_time[op_type].append(self._all_op_exe_time[op_name])
        for op_type in self._op_type_exe_time:
            if self._op_type_exe_time[op_type]:
                self._op_info.get("op_type", {})[op_type] = (
                    np.around(np.sum(self._op_type_exe_time[op_type], axis=0, dtype='float') /
                              len(self._op_type_exe_time[op_type]), 4)).tolist()
        self._dynamic_shape_info['op_type'] = self._op_info.get("op_type")
        dynamic_shape_file_path = os.path.join(self._output_path, output_dynamic_shape_file_name)
        with os.fdopen(os.open(dynamic_shape_file_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o660), 'w') as fp:
            json.dump(self._dynamic_shape_info, fp)

    def _analyse_op_execute_time(self):
        """Obtain the execution time of aicpu operator and aicore operator."""
        timeline_origin_file_name = f'output_timeline_data_{self._rank_id}.txt'
        aicpu_file_name = f'aicpu_intermediate_{self._rank_id}.csv'
        timeline_origin_file_path = os.path.join(self._output_path, timeline_origin_file_name)
        timeline_origin_file_path = validate_and_normalize_path(timeline_origin_file_path)
        aicpu_file_path = os.path.join(self._output_path, aicpu_file_name)

        def read_file(file_path):
            """Read file data."""
            with open(file_path, 'r') as fp:
                file_info = fp.readlines()[1:]
                return file_info

        timeline_info = read_file(timeline_origin_file_path)
        for line_info in timeline_info:
            line_info = line_info.strip('\n').split(',')
            op_name = line_info[0].split('/')[-1]
            op_exe_time = float(line_info[3])
            self._all_op_exe_time[op_name].append(op_exe_time)
        if os.path.exists(aicpu_file_path):
            aicpu_info = read_file(aicpu_file_path)
            for line_info in aicpu_info:
                line_info = line_info.strip('\n').split(',')
                op_name = line_info[1]
                op_exe_time = float(line_info[3])
                self._all_op_exe_time[op_name].append(op_exe_time)

    def _get_dynamic_shape_info(self):
        """Get the shape information of AICPU and aicore."""
        framework_file_name = f'framework_raw_{self._rank_id}.csv'
        self._analyse_op_execute_time()
        framework_file_path = os.path.join(self._output_path, framework_file_name)
        framework_file_path = validate_and_normalize_path(framework_file_path)
        with open(framework_file_path, 'r') as f_obj:
            framework_info = f_obj.readlines()[1:]
            for line_info in framework_info:
                line_info = line_info.strip('\n').split(',')
                op_name = line_info[3].split('/')[-1]
                shape_info = ','.join(line_info[7:]).replace('"', '')
                self._op_shape_info[op_name].append(shape_info)

    def _get_total_step_num(self):
        """Get the number of steps."""
        self._get_dynamic_shape_info()
        all_exe_occurrences = list()
        for op_name in self._all_op_exe_time:
            op_shape = self._op_shape_info.get(op_name)
            op_exe_time_list = self._all_op_exe_time.get(op_name)
            if not op_shape:
                continue
            if len(op_shape) == len(op_exe_time_list):
                self._exe_time_and_shape_detail[op_name]['op_exe_time'] = op_exe_time_list
                self._exe_time_and_shape_detail[op_name]['op_shape'] = op_shape
                self._exe_time_and_shape_detail[op_name]['op_exe_occurrences'] = len(op_exe_time_list)
                all_exe_occurrences.append(len(op_exe_time_list))
        self._step = max(set(all_exe_occurrences), key=all_exe_occurrences.count)
