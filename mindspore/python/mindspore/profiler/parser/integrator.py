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
"""The integrator for integrating parsed profiling files."""
import csv
import json
import os
import stat
from decimal import Decimal
from enum import Enum
import sys

from mindspore import log as logger
from mindspore import context
from mindspore.context import get_auto_parallel_context
from mindspore.profiler.common.exceptions.exceptions import ProfilerIOException, \
    ProfilerFileNotFoundException, ProfilerRawFileException, ProfilerParamValueErrorException
from mindspore.profiler.common.util import query_latest_trace_time_file, to_int, to_millisecond
from mindspore.profiler.common.validator.validate_path import validate_and_normalize_path
from mindspore.profiler.parser.container import TimelineContainer
from mindspore.profiler.parser.op_intermediate_parser import OPIntermediateParser

SIZE_LIMIT_DEFAULT = 20 * 1024 * 1024  # 20MB


class Integrator:
    """
    The integrator for integrating parsed profiling files.

    Args:
        profiling_dir (str): The directory where the parsed profiling files are
            located.
        device_id (str): The device ID.
    """
    _file_name_aicore_detail_time = 'output_op_compute_time_{}.txt'
    _file_name_aicpu_time = 'output_data_preprocess_aicpu_{}.txt'
    _file_name_framework = 'framework_raw_{}.csv'
    _header_aicore_type = ['op_type', 'execution_time', 'execution_frequency',
                           'percent']
    _header_aicore_detail = ['full_op_name', 'execution_time']
    _header_aicpu = ['serial_number', 'op_type', 'total_time', 'dispatch_time',
                     'execution_time', 'run_start', 'run_end']

    _file_name_aicore_type_time = 'aicore_intermediate_{}_type.csv'
    _file_name_aicore_detail_info = 'aicore_intermediate_{}_detail.csv'
    _col_names_detail = ['op_name', 'op_type', 'avg_execution_time', 'subgraph', 'full_op_name', 'op_info']
    _none_filter_condition_key = ['is_display_detail', 'is_display_full_op_name']
    _none_sort_col_names = ['op_info']
    _aicore_data = []
    _aicore_detail_data = []
    _aicore_trace_data = []

    def __init__(self, profiling_dir, device_id):
        self._profiling_dir = profiling_dir
        self._device_id = device_id
        self._op_time_cache = {}
        self._total_time = Decimal('0.0')
        self._column = ""
        self._result = []
        csv.field_size_limit(sys.maxsize)

    @staticmethod
    def _is_match_condition(exp_key, exp_value, actual_value):
        """
        Check whether the actual value meets the expect condition.

        Args:
            exp_key (str): Expect key of the condition.
            exp_value (str): Expect value.
            actual_value (str): Actual value.

        Returns:
            bool, `True` if the actual meets the expect condition, else `False`.
        """
        if exp_key == 'in':
            if actual_value not in exp_value:
                return False
        elif exp_key == 'not_in':
            if actual_value in exp_value:
                return False
        elif exp_key == 'partial_match_str_in':
            for partial_match_str in exp_value:
                if partial_match_str in actual_value:
                    return True
            return False
        else:
            return False

        return True

    def integrate(self):
        """Integrate the parsed profiling files."""
        self._parse_aicore_detail_time()
        self._parse_aicore_type_time()
        self._parse_aicpu_time()

    def get_aicore_data(self):
        """Get ai core data."""
        self._aicore_data_load()
        return self._aicore_data

    def get_aicore_detail_data(self):
        """Get ai core detail data."""
        self._aicore_detail_data_load()
        return self._aicore_detail_data

    def get_aicore_trace_data(self):
        """Get ai core trace data."""
        self._aicore_trace_data_load()
        return self._aicore_trace_data

    def query_for_all_reduce(self):
        """Query all reduce data."""
        return self._query_for_all_reduce()

    def query_and_sort_by_op_type(self, filter_condition, op_type_order):
        """Query and sort by op type."""
        return self._query_and_sort_by_op_type(filter_condition, op_type_order)

    def _parse_aicore_type_time(self):
        """Parse the parsed AICORE operator type file."""
        if self._total_time == 0:
            logger.info("AICORE data does not exist.")
            return

        framework_file = os.path.join(
            self._profiling_dir,
            self._file_name_framework.format(self._device_id)
        )
        framework_file = validate_and_normalize_path(framework_file)
        if not os.path.isfile(framework_file):
            return

        op_name_type_cache = {}
        with open(framework_file, 'r') as src_file:
            csv_reader = csv.reader(src_file)
            _ = next(csv_reader)
            for row in csv_reader:
                op_name_type_cache[row[3]] = row[5]

        op_type_time_cache = {}
        for full_op_name, op_time in self._op_time_cache.items():
            op_type = op_name_type_cache.get(full_op_name)
            op_type_time = op_type_time_cache.get(op_type)
            if not op_type_time:
                op_type_time = [op_time, 1]
                op_type_time_cache[op_type] = op_type_time
            else:
                op_type_time[0] += op_time
                op_type_time[1] += 1

        op_type_file_name = 'aicore_intermediate_' + self._device_id + '_type.csv'
        op_type_file_path = os.path.join(self._profiling_dir, op_type_file_name)
        with open(op_type_file_path, 'w') as type_file:
            csv_writer = csv.writer(type_file)
            csv_writer.writerow(self._header_aicore_type)

            for op_type, op_type_time_info in op_type_time_cache.items():
                type_info = [
                    op_type, op_type_time_info[0], op_type_time_info[1],
                    round((op_type_time_info[0] / self._total_time) * 100, 2)
                ]
                csv_writer.writerow(type_info)

    def _parse_aicore_detail_time(self):
        """Parse the parsed AICORE operator time file."""
        aicore_detail_file = os.path.join(
            self._profiling_dir,
            self._file_name_aicore_detail_time.format(self._device_id)
        )
        aicore_detail_file = validate_and_normalize_path(aicore_detail_file)
        if not os.path.isfile(aicore_detail_file):
            return

        op_detail_file_name = 'aicore_intermediate_' + self._device_id + '_detail.csv'
        op_detail_file_path = os.path.join(
            self._profiling_dir, op_detail_file_name
        )
        with open(aicore_detail_file, 'r') as src_file:
            row = src_file.readline()
            if row.startswith('op_name'):
                _ = src_file.readline()
            elif row.startswith('====='):
                _ = src_file.readline()
                _ = src_file.readline()
            else:
                return

            with open(op_detail_file_path, 'w') as detail_file:
                csv_writer = csv.writer(detail_file)
                csv_writer.writerow(self._header_aicore_detail)

                while True:
                    row = src_file.readline()
                    if not row:
                        break

                    op_infos = row.split()
                    if op_infos[0] == 'total':
                        self._total_time = Decimal(op_infos[2])
                        continue
                    self._op_time_cache[op_infos[0]] = Decimal(op_infos[1])
                    csv_writer.writerow([op_infos[0], op_infos[1]])

    def _parse_aicpu_time(self):
        """Parse the parsed AICPU operator time file."""
        aicpu_file = os.path.join(
            self._profiling_dir,
            self._file_name_aicpu_time.format(self._device_id)
        )
        aicpu_file = validate_and_normalize_path(aicpu_file)
        if not os.path.isfile(aicpu_file):
            return

        save_file_name = 'aicpu_intermediate_' + self._device_id + '.csv'
        save_file_path = os.path.join(self._profiling_dir, save_file_name)
        with open(aicpu_file, 'r') as src_file:
            row = src_file.readline()
            if not row.startswith('serial_number'):
                return
            with open(save_file_path, 'w') as save_file:
                csv_writer = csv.writer(save_file)
                csv_writer.writerow(self._header_aicpu)

                while True:
                    row = src_file.readline()
                    if not row:
                        break
                    infos = row.split()
                    if infos[0] == 'AI':
                        continue
                    csv_writer.writerow(infos)

    def _aicore_data_load(self):
        """Load data according to the parsed AICORE operator types file."""
        op_type_file_path = os.path.join(
            self._profiling_dir,
            self._file_name_aicore_type_time.format(self._device_id)
        )
        op_type_file_path = validate_and_normalize_path(op_type_file_path)
        if not os.path.isfile(op_type_file_path):
            logger.warning('The file <%s> does not exist.', op_type_file_path)
            return

        with open(op_type_file_path, 'r') as file:
            csv_reader = csv.reader(file)
            _ = next(csv_reader)
            for info in csv_reader:
                self._aicore_data.append([info[0], float(info[1]), int(info[2]), float(info[3])])

    def _aicore_detail_data_load(self):
        """Load data according to the parsed AICORE operator file."""
        op_detail_file_path = os.path.join(
            self._profiling_dir,
            self._file_name_aicore_detail_info.format(self._device_id)
        )
        framework_file_path = os.path.join(
            self._profiling_dir,
            self._file_name_framework.format(self._device_id)
        )
        op_detail_file_path = validate_and_normalize_path(op_detail_file_path)
        framework_file_path = validate_and_normalize_path(framework_file_path)
        if not os.path.isfile(op_detail_file_path):
            logger.warning('The file <%s> does not exist.', op_detail_file_path)
            return
        if not os.path.isfile(framework_file_path):
            logger.warning('The file <%s> does not exist.', framework_file_path)
            return

        framework_infos = dict()
        with open(framework_file_path, 'r') as file:
            csv_reader = csv.reader(file)
            _ = next(csv_reader)
            for info in csv_reader:
                framework_infos[info[3]] = [
                    info[3], info[4], info[5], info[6], json.loads(info[7]) if info[7] else None]

        with open(op_detail_file_path, 'r') as file:
            csv_reader = csv.reader(file)
            _ = next(csv_reader)
            for info in csv_reader:
                framework_info = framework_infos.get(info[0])
                self._aicore_detail_data.append(
                    [
                        framework_info[1], framework_info[2], float(info[1]),
                        framework_info[3], framework_info[0], framework_info[4]
                    ]
                )
        del framework_infos

    def _aicore_trace_data_load(self):
        """Load data according to the parsed AICORE operator types file."""
        file_path = query_latest_trace_time_file(self._profiling_dir, int(self._device_id))
        if not file_path:
            logger.warning("Failed to find parsed trace time file. Dynamic Shape networks can ignore this warning.")
            return
        file_path = validate_and_normalize_path(file_path)
        with open(file_path, 'r') as handle:
            csv_reader = csv.reader(handle)
            self._column = next(csv_reader)
            self._aicore_trace_data = list(csv_reader)
        self._size = len(self._aicore_trace_data) - 1
        self._load_point_info()

    def _load_point_info(self):
        """Load point info."""
        file_path = os.path.join(self._profiling_dir, 'step_trace_point_info.json')
        file_path = validate_and_normalize_path(file_path)
        if os.path.isfile(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                try:
                    self._point_info = json.load(file)
                except (json.JSONDecodeError, TypeError) as err:
                    logger.warning(err)
                    raise ProfilerRawFileException('Fail to parse point info file.') from err

    def _query_for_all_reduce(self):
        """
        Query for all reduce info.

        Returns:
            list[dict], reduce information. Each item is the reduce info for one step.
            The reduce info is format like:
            {stream_id: List[Tuple(start_point, end_point, duration, field_name)]}.
        """
        self._aicore_trace_data_load()
        reduce_infos = []
        for row_info in self._aicore_trace_data[:-1]:
            row_info_dict = self._get_info_dict_from_row_data(row_info, 'systime')
            reduce_info = self._sort_reduce_by_time(row_info_dict)
            if reduce_info:
                reduce_infos.extend(reduce_info)
        reduce_infos.sort(key=lambda x: float(x[2]))
        return reduce_infos

    def _get_info_dict_from_row_data(self, row_info, time_type):
        """
        Get step info in dict format.

        Args:
            row_info (list[str]): Step info, the value is corresponding to `__column__`.
            time_type (str): The value type. `systime` keeps the original value.
                `realtime` transforms the value in millisecond. Default: `realtime`.

        Returns:
            dict, step trace information. The key is in `__column__`.
        """
        row_info_dict = {}
        for key, value in zip(self._column, row_info):
            if key == 'step_num':
                continue
            value = to_int(value, key)
            row_info_dict[key] = to_millisecond(value) if time_type == 'realtime' else value
        return row_info_dict

    def _sort_reduce_by_time(self, row_info_dict):
        """
        Sort reduce info by time.

        Args:
            row_info_dict (dict): Step trace information.

        Returns:
            list, including the all reduce info sorted by start time only.
            [
                [reduce_field, stream_id, reduce_start, reduce_duration],
                [...],
                [...]
            ]
        """
        factor = 1e5  # convert time unit from 10ns to 1ms
        reduce_pid = 10000
        reduce_info = []
        reduce_fields = [field_name for field_name in self._column
                         if field_name.startswith('stream_') and not field_name.endswith('point')]
        for reduce_field in reduce_fields:
            reduce_start = row_info_dict.get(reduce_field + '_start_point')
            reduce_start = reduce_start / factor \
                if reduce_start else 0
            reduce_duration = row_info_dict.get(reduce_field)
            reduce_duration = reduce_duration / factor if reduce_duration else 0
            if not (reduce_start and reduce_duration):
                logger.info("Reduce event missing value.")
                continue
            cur_stream_id = reduce_field.split('_', 3)[1]
            if reduce_field.split('_', 2)[1] == 'ops':
                cur_stream_id = reduce_field.split('_', 3)[2]
            reduce_meta = [reduce_field, int(cur_stream_id), reduce_start,
                           reduce_duration, reduce_pid]
            reduce_info.append(reduce_meta)

        return reduce_info

    def _query_and_sort_by_op_type(self, filter_condition, op_type_order: list):
        """
        Query the AICORE operator detail information by `filter_condition`,
        and sort by `op_type_order` and execution time.

        Args:
            filter_condition (dict): The filter condition.
            op_type_order (list[str]): The name of the operator type in order.

        Returns:
            dict, The results are filtered and sorted.
        """
        self._aicore_detail_data_load()
        if filter_condition is None:
            filter_condition = {}
        self._filter(filter_condition)

        type_detail_cache = {}
        for detail_info in self._result:
            op_type = detail_info[1]
            if op_type not in op_type_order:
                continue
            infos = type_detail_cache.get(op_type)
            if infos:
                infos.append(detail_info)
            else:
                type_detail_cache[op_type] = [detail_info]

        result = []
        for op_type in op_type_order:
            detail_infos = type_detail_cache.get(op_type)
            if detail_infos is None:
                continue
            detail_infos.sort(key=lambda item: item[2], reverse=True)
            result.extend(detail_infos)

        return {
            'col_name_detail': self._display_col_names_detail,
            'object': result
        }

    def _filter(self, filter_condition):
        """
        Filter the profiling data according to the filter condition.

        Args:
            filter_condition (dict): The filter condition.
        """

        def _inner_filter(item: list):
            return self._default_filter(item, filter_condition)

        def _inner_map(item: list):
            inner_item = item[0:4]
            if is_display_full_op_name:
                inner_item.append(item[4])
            if is_display_detail:
                inner_item.append(item[5])
            return inner_item

        is_display_detail = filter_condition.get('is_display_detail', True)
        is_display_full_op_name = filter_condition.get(
            'is_display_full_op_name', True
        )
        self._set_display_col_name(is_display_detail, is_display_full_op_name)
        if is_display_detail and is_display_full_op_name:
            self._result = list(filter(_inner_filter, self._aicore_detail_data))
        else:
            self._result = list(
                map(_inner_map, filter(_inner_filter, self._aicore_detail_data))
            )

    def _default_filter(self, item, condition):
        """
        The default filter method.

        Args:
            item (list[Union[str, float, int]]): A piece of data to be filtered.
            condition (dict): The filter condition.

        Returns:
            bool, `True` if the item is satisfied.
        """
        for condition_key, condition_value in condition.items():
            if condition_key in self._none_filter_condition_key:
                continue
            if condition_key in self._col_names_detail:
                index = self._col_names_detail.index(condition_key)
                actual_value = item[index]
                for exp_key, exp_value in condition_value.items():
                    if not self._is_match_condition(
                            exp_key, exp_value, actual_value):
                        return False
        return True

    def _set_display_col_name(self, is_display_detail, is_display_full_op_name):
        """
        Set the display column name according to the filter condition.

        Args:
            is_display_detail (bool): Whether to display the detailed operator
                information.
            is_display_full_op_name (bool): Whether to display the operator full
                name.
        """
        self._display_col_names_detail = self._col_names_detail[0:4]
        if is_display_full_op_name:
            self._display_col_names_detail.append(self._col_names_detail[4])
        if is_display_detail:
            self._display_col_names_detail.append(self._col_names_detail[5])


class DeviceTarget(Enum):
    """The device target enum."""
    CPU = 'cpu'
    GPU = 'gpu'
    ASCEND = 'ascend'


class BaseTimelineGenerator:
    """
    Analyse timeline data from file.
    """
    # AI Core Op pid is device_id
    _AI_CPU_PID = 9000
    _COMMUNICATION_OP_PID = 10000
    _HOST_CPU_PID = 11000
    _OP_OVERLAP_PID = 12000

    _OP_GPU_ACTIVITY_PID = 13000

    _RECEIVE_ALONE = 7997
    _ALLREDUCE_ALONE = 7998
    _MERGED_COMPUTATION_TID = 7999
    _PURE_COMMUNICATION_TID = 8000
    _MERGED_COMMUNICATION_TID = 8001
    _FREE_TIME_TID = 8002
    _STEPS_TID = 100000
    _SCOPE_NAME_TID = 100001
    _GPU_OP_TID = 100002
    _HOST_CPU_OP_TID = 100003
    _SINGLE_TID = 0

    _STEPS_SORT_INDEX = -4

    _output_timeline_data_file_path = 'output_timeline_data_{}.txt'
    _timeline_meta = []
    _format_meta_data_list = []
    _thread_processed_list = []

    _map_tid_name_to_int = {
        "Steps": (-4, _STEPS_TID),
        "Scope Name": (-3, _SCOPE_NAME_TID),
        "GpuOps": (-2, _GPU_OP_TID),
        "HostCpuOps": (-1, _HOST_CPU_OP_TID)
    }
    _timeline_summary = {
        'total_time': 0,
        'num_of_streams': 0,
        'num_of_ops': 0,
        'op_exe_times': 0,
        'max_scope_name_num': 0,
    }
    _op_name_idx, _tid_idx, _start_time_idx, _duration_idx = 0, 1, 2, 3
    _max_scope_name_num = 0
    _host_cpu_op_label = 'Host CPU OP'
    _gpu_op_label = "GPU Op"
    _ascend_op_label = "Ascend Op"
    _aicore_op_label = "AICORE OP"
    _aicpu_op_label = "AICPU OP"

    _device_id = 0
    _profiling_dir = ""
    _timeline_summary_filename = ""
    _display_filename = ""
    _op_name_list = []
    _device_target = DeviceTarget.ASCEND.value
    _model = context.GRAPH_MODE

    __col_names = ['op_name', 'stream_id', 'start_time', 'duration']

    def __init__(self, device_target, model):
        self._tid_dict = {
            "computation_op": (self._MERGED_COMPUTATION_TID, self._OP_OVERLAP_PID),
            "communication_not_overlapped": (self._PURE_COMMUNICATION_TID, self._OP_OVERLAP_PID),
            "communication": (self._MERGED_COMMUNICATION_TID, self._OP_OVERLAP_PID),
            "free_time": (self._FREE_TIME_TID, self._OP_OVERLAP_PID)
        }
        self._device_target = str(device_target).lower()
        self._model = model
        self._step_start_op_name = ""
        self._step_end_op_name = ""

    @staticmethod
    def get_parallel_context():
        """Get parallel context."""
        try:
            parallel_mode = get_auto_parallel_context("parallel_mode")
            stage_num = get_auto_parallel_context("pipeline_stages")
        except RuntimeError:
            logger.warning("[profiler] the feature of cluster bottleneck analyse "
                           "is not supported in offline parse mode.")
            parallel_mode = "data_parallel"
            stage_num = 1
        if stage_num > 1:
            parallel_mode = "pipeline-parallel"
        elif parallel_mode != "data_parallel":
            parallel_mode = "model-parallel"
        else:
            parallel_mode = "data-parallel"
        return parallel_mode, stage_num

    @staticmethod
    def _update_num_of_streams(timeline, stream_count_dict):
        """Update number of streams."""
        stream_id = timeline[1]
        if stream_id in ["Steps", "Scope Name"]:
            return
        if stream_id not in stream_count_dict.keys():
            stream_count_dict[stream_id] = 1
        else:
            stream_count_dict[stream_id] += 1

    def get_thread_label_name(self):
        """Get process and thread config."""
        device_process_label = self._get_device_process_label()
        return [
            {"name": "process_labels", "ph": "M", "pid": self._device_id, "args": {"labels": device_process_label}},
            {"name": "process_labels", "ph": "M", "pid": self._AI_CPU_PID, "args": {"labels": self._aicpu_op_label}},
            {"name": "process_labels", "ph": "M", "pid": self._COMMUNICATION_OP_PID,
             "args": {"labels": "Communication Op"}},
            {"name": "process_labels", "ph": "M", "pid": self._HOST_CPU_PID,
             "args": {"labels": self._host_cpu_op_label}},
            {"name": "process_labels", "ph": "M", "pid": self._OP_OVERLAP_PID,
             "args": {"labels": "Op Overlap Analyse"}},
            {"name": "process_labels", "ph": "M", "pid": self._OP_GPU_ACTIVITY_PID,
             "args": {"labels": "Activity Op"}},

            {"name": "process_sort_index", "ph": "M", "pid": self._device_id, "args": {"sort_index": 0}},
            {"name": "process_sort_index", "ph": "M", "pid": self._AI_CPU_PID, "args": {"sort_index": 10}},
            {"name": "process_sort_index", "ph": "M", "pid": self._COMMUNICATION_OP_PID, "args": {"sort_index": 20}},
            {"name": "process_sort_index", "ph": "M", "pid": self._HOST_CPU_PID, "args": {"sort_index": 30}},
            {"name": "process_sort_index", "ph": "M", "pid": self._OP_OVERLAP_PID, "args": {"sort_index": 40}},

            {"name": "thread_name", "ph": "M", "pid": self._HOST_CPU_PID, "tid": self._HOST_CPU_OP_TID,
             "args": {"name": "Host CPU Op"}},
            {"name": "thread_name", "ph": "M", "pid": self._OP_OVERLAP_PID, "tid": self._MERGED_COMPUTATION_TID,
             "args": {"name": "Merged Computation Op"}},
            {"name": "thread_name", "ph": "M", "pid": self._OP_OVERLAP_PID, "tid": self._PURE_COMMUNICATION_TID,
             "args": {"name": "Pure Communication Op"}},
            {"name": "thread_name", "ph": "M", "pid": self._OP_OVERLAP_PID, "tid": self._MERGED_COMMUNICATION_TID,
             "args": {"name": "Merged Communication Op"}},
            {"name": "thread_name", "ph": "M", "pid": self._OP_OVERLAP_PID, "tid": self._FREE_TIME_TID,
             "args": {"name": "Free Time"}},
            {"name": "thread_name", "ph": "M", "pid": self._device_id, "tid": self._STEPS_TID,
             "args": {"name": "Steps"}},
            {"name": "thread_name", "ph": "M", "pid": self._device_id, "tid": self._SINGLE_TID,
             "args": {"name": "Ops"}},

            {"name": "thread_sort_index", "ph": "M", "pid": self._OP_OVERLAP_PID, "tid": self._MERGED_COMPUTATION_TID,
             "args": {"sort_index": self._MERGED_COMPUTATION_TID}},
            {"name": "thread_sort_index", "ph": "M", "pid": self._OP_OVERLAP_PID, "tid": self._PURE_COMMUNICATION_TID,
             "args": {"sort_index": self._PURE_COMMUNICATION_TID}},
            {"name": "thread_sort_index", "ph": "M", "pid": self._OP_OVERLAP_PID, "tid": self._MERGED_COMMUNICATION_TID,
             "args": {"sort_index": self._MERGED_COMMUNICATION_TID}},
            {"name": "thread_sort_index", "ph": "M", "pid": self._OP_OVERLAP_PID, "tid": self._FREE_TIME_TID,
             "args": {"sort_index": self._FREE_TIME_TID}},
            {"name": "thread_sort_index", "ph": "M", "pid": self._device_id, "tid": self._STEPS_TID,
             "args": {"sort_index": self._STEPS_SORT_INDEX}},
        ]

    def write_timeline(self, size_limit=SIZE_LIMIT_DEFAULT):
        """Load data according to the parsed profiling files."""
        # Write timeline to file.
        logger.info('Writing timeline file...')
        timeline_meta = self.write_timeline_to_json_by_limitation(size_limit)
        logger.info('Finished file writing!')
        return timeline_meta

    def write_timeline_to_json_by_limitation(self, size_limit):
        """Write timeline to json by limitation."""
        display_file_path = os.path.join(
            self._profiling_dir,
            self._display_filename
        )
        display_file_path = validate_and_normalize_path(display_file_path)

        try:
            with open(display_file_path, 'w') as json_file:
                json_file.write('[')
                for _, item in enumerate(self._timeline_meta):
                    json.dump(item, json_file)
                    if "scope_level" in item.keys():
                        self._max_scope_name_num = max(
                            self._max_scope_name_num, item["scope_level"] + 1)
                    file_size = os.path.getsize(display_file_path)
                    json_file.write(',')
                    if file_size > size_limit:
                        break
                label_name_json = json.dumps(self.get_thread_label_name())
                label_name_json = label_name_json.lstrip('[')
                json_file.write(label_name_json)
                os.chmod(display_file_path, stat.S_IREAD | stat.S_IWRITE)
            return self._timeline_meta
        except (IOError, OSError) as err:
            logger.critical('Error occurred when write timeline display file: %s', err)
            raise ProfilerIOException() from err

    def write_timeline_summary(self):
        """Write timeline summary to json."""
        timeline_summary_file_path = os.path.join(
            self._profiling_dir,
            self._timeline_summary_filename
        )

        timeline_summary_file_path = validate_and_normalize_path(timeline_summary_file_path)

        try:
            with open(timeline_summary_file_path, 'w') as json_file:
                json.dump(self._timeline_summary, json_file)
            os.chmod(timeline_summary_file_path, stat.S_IREAD | stat.S_IWRITE)
        except (IOError, OSError) as err:
            logger.critical('Error occurred when write timeline summary file: %s', err)
            raise ProfilerIOException() from err

    def _get_device_process_label(self):
        """Get device process label."""
        device_process_label = self._aicore_op_label
        if self._device_target == DeviceTarget.ASCEND.value:
            if self._model == context.GRAPH_MODE:
                device_process_label = self._aicore_op_label
            elif self._model == context.PYNATIVE_MODE:
                device_process_label = self._ascend_op_label
        elif self._device_target == DeviceTarget.GPU.value:
            device_process_label = self._gpu_op_label
        elif self._device_target == DeviceTarget.CPU.value:
            device_process_label = self._host_cpu_op_label
        return device_process_label

    def _get_merged_time_list(self, time_list, get_interval_time=False, display_name="computation_op", factor=1):
        """
        Get merged time segment list.

        The process of merge is, for example, there is a list [[1,5], [2,6], [7,8]],
        each items in this list contains a start_time and end_time,
        the merged result is [[1,6], [7,8]].
        """
        time_merged_segment_list = []
        tid = self._tid_dict.get(display_name, (0, 0))[0]
        pid = self._tid_dict.get(display_name, (0, 0))[1]
        for time_item in time_list:
            time_segment = list(map(float, time_item[self._start_time_idx:self._duration_idx + 1]))
            time_segment[1] = time_segment[0] + time_segment[1] / factor
            if not time_merged_segment_list or \
                    time_segment[0] > time_merged_segment_list[-1]:
                time_merged_segment_list.extend(time_segment)
            else:
                time_merged_segment_list[-1] = max(
                    time_merged_segment_list[-1],
                    time_segment[1]
                )

        # merged_display_list data used for ui page.
        merged_display_list = []
        for i in range(len(time_merged_segment_list) // 2):
            merged_display_list.append([display_name, tid, time_merged_segment_list[i * 2],
                                        (time_merged_segment_list[i * 2 + 1] - time_merged_segment_list[
                                            i * 2]) * factor, pid])

        if get_interval_time:
            time_merged_segment_list = time_merged_segment_list[1:-1]

        # merged_res_list data used to compute overlap with other time_list.
        merged_res_list = []
        for i in range(len(time_merged_segment_list) // 2):
            merged_res_list.append([display_name, tid, time_merged_segment_list[i * 2],
                                    time_merged_segment_list[i * 2 + 1], pid])

        # interval_display_list is interval time used for ui page.
        interval_display_list = [
            [display_name, tid, time_merged_segment_list[i * 2],
             (time_merged_segment_list[i * 2 + 1] - time_merged_segment_list[i * 2]) * factor, pid]
            for i in range(len(time_merged_segment_list) // 2)
        ]

        return merged_res_list, interval_display_list, merged_display_list

    def _update_format_meta_data(self, timeline_dict):
        """Update format meta data which control the display arrange and map the thread name."""
        thread_name_meta_data = {
            "name": "thread_name",
            "pid": int(self._device_id),
            "tid": 100000,
            "ts": 0,
            "ph": "M",
            "cat": "__metadata",
            "args": {
                "name": "Steps"
            }
        }
        tid_name = timeline_dict['tid']
        sort_index = 0

        if tid_name in self._map_tid_name_to_int.keys():
            sort_index, tid = self._map_tid_name_to_int[tid_name]
        elif tid_name.startswith("Stream"):
            tid = int(tid_name.split("#")[-1])
            sort_index = tid
        else:
            return

        if self._host_cpu_op_label == tid_name[:len(self._host_cpu_op_label)]:
            thread_name_meta_data['pid'] = self._HOST_CPU_PID

        thread_name_meta_data["tid"] = tid
        thread_name_meta_data["args"]["name"] = tid_name
        thread_sort_meta_data = thread_name_meta_data.copy()
        thread_sort_meta_data['name'] = "thread_sort_index"
        thread_sort_meta_data["args"] = {"sort_index": sort_index}
        timeline_dict["tid"] = tid

        if tid_name in self._thread_processed_list:
            return

        self._thread_processed_list.append(tid_name)
        self._format_meta_data_list.append(thread_name_meta_data)
        self._format_meta_data_list.append(thread_sort_meta_data)

    def _get_max_scope_name_num(self, timeline_list):
        """Get the max number of scope level from all operator."""
        max_scope_name_num = 0
        for time_item in timeline_list:
            cur_scope_name_num = len(time_item[self._op_name_idx].split('/')) - 1
            max_scope_name_num = max(cur_scope_name_num, max_scope_name_num)

        return max_scope_name_num

    def _get_scope_name_time_list(self, timeline_list, subgraph, factor_start_time_to_duration=1):
        """Produce the timeline of hierarchical scope name."""
        # the key of scope_name_start_duration_dict is scope name, the value is a dict which store the
        # start and end index of time_item in timeline_list.
        scope_name_start_duration_dict = {}
        scope_name_time_list = []
        op_full_name_idx, scope_name_idx, invalid_idx = 0, 0, -1
        tid = "Scope Name"
        for idx, time_item in enumerate(timeline_list):
            scope_name_list = time_item[op_full_name_idx].split('/')[:-1]
            # skip Default/InitDataSetQueue operator.
            if time_item[op_full_name_idx].startswith("Default/InitDataSetQueue"):
                scope_name_list = []
            # process scope name of subgraph(Default/Gradients/recompute_Default) only.
            if scope_name_list and scope_name_list[0] != subgraph:
                scope_name_list = []
            # add the level of scope name, used to distinguish the same name at different scope level.
            scope_name_list = [f"{scope_level}-{scope_name}"
                               for scope_level, scope_name in enumerate(scope_name_list)]

            # update the start and end index of time_item according to current scope_name
            for scope_name in scope_name_list:
                init_start_end_idx_dict = {'start_item_idx': idx, 'end_item_idx': idx}
                if scope_name not in scope_name_start_duration_dict:
                    scope_name_start_duration_dict[scope_name] = init_start_end_idx_dict
                if scope_name_start_duration_dict[scope_name]['start_item_idx'] == invalid_idx:
                    scope_name_start_duration_dict[scope_name] = init_start_end_idx_dict
                else:
                    scope_name_start_duration_dict[scope_name]['end_item_idx'] = idx
            # if the key(scope name) in scope_name_start_duration_dict does not appear in scope_name_list,
            # it means this key(scope name) is end and it is append to scope_name_time_list.
            for key, val in scope_name_start_duration_dict.items():
                if val['start_item_idx'] == invalid_idx:
                    continue
                if (key not in scope_name_list) \
                        or idx == (len(timeline_list) - 1) \
                        or time_item[op_full_name_idx] == self._step_end_op_name:
                    start_time = timeline_list[val['start_item_idx']][self._start_time_idx]
                    duration = (float(timeline_list[val['end_item_idx']][self._start_time_idx]) - float(start_time)) * \
                               factor_start_time_to_duration + \
                               float(timeline_list[val['end_item_idx']][self._duration_idx])
                    scope_name_time_item = [key, tid, start_time, duration]
                    scope_name_time_list.append(scope_name_time_item)
                    scope_name_start_duration_dict[key]['start_item_idx'] = invalid_idx

        # x[scope_name_idx] is a scope name like "0-Default".
        # if two element in scope_name_time_list have the same start time,
        # the previous element in list will displayed at the higher line in UI page.
        scope_name_time_list.sort(
            key=lambda x: (float(x[self._start_time_idx]), int(x[scope_name_idx].split('-')[0]))
        )

        return scope_name_time_list

    def _set_step_start_and_end_op_name(self, timeline_list):
        """Set the start and end operator full name of each step."""
        if not timeline_list:
            return
        start_op_idx = 0
        if timeline_list[0][self._op_name_idx].startswith("Default/InitDataSetQueue"):
            start_op_idx = 1
        self._step_start_op_name = timeline_list[start_op_idx][self._op_name_idx]
        self._step_end_op_name = self._step_start_op_name
        if len(timeline_list) > (start_op_idx + 1):
            for time_item in timeline_list[start_op_idx + 1:]:
                if time_item[self._op_name_idx] != self._step_start_op_name:
                    self._step_end_op_name = time_item[self._op_name_idx]
                else:
                    break

    def _get_step_time_list(self, timeline_list, factor_start_time_to_duration=1):
        """Produce the time of each step."""
        # Record the time of each step.
        step_time_list = []
        step_num = 1
        tid = "Steps"
        cur_step_start_time, cur_step_duration_time = 0, 0
        for time_item in timeline_list:
            if time_item[self._op_name_idx] == self._step_start_op_name:
                cur_step_start_time = time_item[self._start_time_idx]
            if time_item[self._op_name_idx] == self._step_end_op_name:
                cur_step_duration_time = (float(time_item[self._start_time_idx]) - float(cur_step_start_time)) * \
                                         float(factor_start_time_to_duration) + float(time_item[self._duration_idx])
                step_time_item = [str(step_num), tid, float(cur_step_start_time), cur_step_duration_time]
                step_time_list.append(step_time_item)
                step_num += 1

        return step_time_list

    def _write_cluster_metrices(self, metrices, is_pipeline_parallel, device_target, dev_id):
        """Write cluster metric."""
        # Note that the feature of cluster bottleneck analyse is not supported in offline parse mode,
        # due to that parallel context is not set.
        parallel_mode, stage_num = BaseTimelineGenerator.get_parallel_context()

        unit = 1 if device_target == "Ascend" else 1e3
        time_decimal_digits = 4
        cluster_analyse_file_path = os.path.join(
            self._profiling_dir,
            self._cluster_analyse_filename.format(parallel_mode, stage_num, self._rank_size, dev_id)
        )
        cluster_analyse_file_path = validate_and_normalize_path(cluster_analyse_file_path)

        try:
            with open(cluster_analyse_file_path, 'w') as file_handle:
                csv_writer = csv.writer(file_handle)
                if is_pipeline_parallel:
                    header = ['computation_time', 'communication_alone_time', 'stage_time',
                              'receive_alone_time', 'collective_communication_alone_time']
                    zip_metrices = zip(metrices[0], metrices[1], metrices[2], metrices[3], metrices[4])
                else:
                    header = ['computation_time', 'communication_alone_time']
                    zip_metrices = zip(metrices[0], metrices[1])
                csv_writer.writerow(header)
                for row_data in zip_metrices:
                    row_data = [round(val / unit, time_decimal_digits) for val in row_data]
                    csv_writer.writerow(row_data)
            os.chmod(cluster_analyse_file_path, stat.S_IREAD | stat.S_IWRITE)
        except (IOError, OSError) as err:
            logger.warning(f'Failed to save {cluster_analyse_file_path}. {err}')
            raise ProfilerIOException from err

    def _register_op_name(self, timeline_list):
        """Register op name to op name list."""
        for timeline in timeline_list:
            if timeline and timeline[self._op_name_idx] not in self._op_name_list:
                self._op_name_list.append(timeline[self._op_name_idx])


class GpuTimelineGenerator(BaseTimelineGenerator):
    """Generate gpu Timeline data from file."""
    _display_filename = 'gpu_timeline_display_{}.json'
    _timeline_summary_filename = 'gpu_timeline_summary_{}.json'
    _output_op_execute_time_file_path = "gpu_op_execute_timestamp_{}.txt"
    _output_activity_execute_time_file_path = "activity_execute_timestamp_{}.txt"
    _output_gpu_activity_info_file_path = "gpu_activity_data_{}.csv"
    _step_trace_original_filename = 'step_trace_profiling_{}.txt'
    _cluster_analyse_filename = 'gpu_cluster_analyse_{}_{}_{}_{}.csv'
    _activity_keys_list = []

    def __init__(self, profiling_dir, device_id, rank_size, model):
        super().__init__(DeviceTarget.GPU.value, model)
        self._device_id = device_id
        self._rank_size = rank_size
        self._profiling_dir = profiling_dir
        self._device_id = device_id
        self._timeline_meta = []
        self._display_filename = self._display_filename.format(device_id)
        self._timeline_summary_filename = self._timeline_summary_filename.format(device_id)
        self._tid_dict = {
            "receive_op_not_overlapped": (self._RECEIVE_ALONE, self._OP_OVERLAP_PID),
            "exclude_receive_op": (self._ALLREDUCE_ALONE, self._OP_OVERLAP_PID),
            "computation_op": (self._MERGED_COMPUTATION_TID, self._OP_OVERLAP_PID),
            "communication_not_overlapped": (self._PURE_COMMUNICATION_TID, self._OP_OVERLAP_PID),
            "communication": (self._MERGED_COMMUNICATION_TID, self._OP_OVERLAP_PID),
            "free_time": (self._FREE_TIME_TID, self._OP_OVERLAP_PID)
        }

    def init_timeline(self, reduce_op_type):
        """Init timeline metadata, adding all collected info."""
        timeline_list = self._load_timeline_data(reduce_op_type)

        # Init a dict for counting the num of streams.
        stream_count_dict = {}
        for timeline in timeline_list:
            self._parse_timeline_data(timeline, 0)
            # Updating the collection of streams.
            if len(timeline) == 4:
                self._update_num_of_streams(timeline, stream_count_dict)

        # Add format thread meta data.
        self._format_meta_data_list.extend(self._timeline_meta)
        self._timeline_meta = self._format_meta_data_list

        # Update timeline summary info
        self._timeline_summary['num_of_streams'] += len(stream_count_dict)

    def check_op_name(self, op_name):
        """
        Check whether the operator name exists.

        Args:
            op_name (str): The operator name or operator name prefix.

        Returns:
            bool, `True` if the operator name does exist, else `False`.
        """
        if not op_name:
            raise ProfilerParamValueErrorException('The op_name should exist.')
        for op_time_info in self._timeline_meta:
            full_op_name = op_time_info['name']
            if full_op_name and full_op_name.startswith(op_name):
                return True
        return False

    def is_gpu_kernel_async_launch(self):
        """Recognize the solution that launch the gpu kernel async."""
        step_trace_profiling_path = self._get_and_validate_path(
            self._step_trace_original_filename
        )
        try:
            with open(step_trace_profiling_path, 'r') as f_obj:
                line = next(f_obj)
                first_string = line.strip().split()[0]
                # the data format of launch the gpu kernel async is "Default/op1,160123 op-name"
                # otherwise, the data format is "Default/op1 160123,12 "
                return bool(len(first_string.split(',')) == 2)
        except (IOError, OSError) as err:
            logger.critical(f'Error occurred when read {step_trace_profiling_path}: {err}')
            raise ProfilerIOException() from err
        except StopIteration:
            logger.warning('No step trace data exists.')
            return False

    def _get_and_validate_path(self, file_name):
        """Generate op or activity file path from file name, and validate this path."""
        file_path = os.path.join(
            self._profiling_dir,
            file_name.format(self._device_id)
        )
        file_path = validate_and_normalize_path(file_path)
        if not os.path.exists(file_path):
            logger.critical(f"Failed to find parsed timeline file {file_path}.")
            raise ProfilerFileNotFoundException('parsed timeline file')

        return file_path

    def _parse_timeline_data(self, timeline, min_cycle_counter):
        """Parse timeline data."""
        # factor to convert the time unit of start_time(ts) from 1ns to 1us for timeline display
        factor = 1000
        op_meta = TimelineContainer(timeline)
        timeline_dict = {}
        timeline_dict['name'] = op_meta.op_name.split('/')[-1]
        timeline_dict['ph'] = 'X'
        timeline_dict['tid'] = op_meta.stream_id
        timeline_dict['ts'] = (op_meta.start_time - min_cycle_counter) / factor
        dur = op_meta.duration
        timeline_dict['dur'] = dur
        if op_meta.pid is None:
            timeline_dict['pid'] = int(self._device_id)
        else:
            timeline_dict['pid'] = op_meta.pid
        if op_meta.stream_id == "Scope Name":
            # remove the level of scope name which has a format like "0-conv2-Conv2d".
            timeline_dict['name'] = "-".join(op_meta.op_name.split('-')[1:])
            timeline_dict['scope_level'] = int(op_meta.op_name.split('-')[0])
        elif op_meta.stream_id[:len(self._host_cpu_op_label)] == self._host_cpu_op_label:
            timeline_dict['pid'] = self._HOST_CPU_PID

        if len(timeline) > 4:
            # len(timeline) > 4 refers to activity data, else op data.
            # Add args for activity data
            args_dict = {}
            for ix, value in enumerate(timeline[4:]):
                args_dict[self._activity_keys_list[ix]] = value
            timeline_dict['args'] = args_dict
            timeline_dict['tid'] = f"Stream #{timeline_dict['tid']}"
        elif op_meta.stream_id not in ["Scope Name", "Steps"]:
            # Update total time of operator execution.
            self._timeline_summary['total_time'] += dur / factor
            self._timeline_summary['op_exe_times'] += 1

        self._update_format_meta_data(timeline_dict)
        self._timeline_meta.append(timeline_dict)

    def _load_timeline_data(self, reduce_op_type):
        """Load timeline data from file."""
        op_file_path = self._get_and_validate_path(
            self._output_op_execute_time_file_path)
        activity_file_path = self._get_and_validate_path(
            self._output_activity_execute_time_file_path)
        activity_args_file_path = self._get_and_validate_path(
            self._output_gpu_activity_info_file_path)

        timeline_list, communication_info = self._load_op_data(op_file_path, reduce_op_type)
        communication_info.sort(key=lambda x: float(x[2]))
        # Add host cpu op timeline.
        cpu_timeline_generator = CpuTimelineGenerator(self._profiling_dir, self._model)
        cpu_timeline_list = cpu_timeline_generator.load_cpu_op_data()
        if cpu_timeline_list:
            self._clock_synchronize_to_gpu(cpu_timeline_list)
            timeline_list.extend(cpu_timeline_list)
        timeline_list.sort(key=lambda x: float(x[2]))
        self._max_scope_name_num = self._get_max_scope_name_num(timeline_list)
        self._timeline_summary['max_scope_name_num'] = self._max_scope_name_num

        # Generate step time.
        factor_start_time_uint_to_duration = 1e-3
        self._set_step_start_and_end_op_name(timeline_list)
        # Fit gpu kernel async launch solution.
        if self.is_gpu_kernel_async_launch():
            step_time_list = self._get_step_time_list_from_step_trace()
        else:
            step_time_list = self._get_step_time_list(timeline_list, factor_start_time_uint_to_duration)

        # Add Scope Name.
        default_scope_name_time_list = self._get_scope_name_time_list(timeline_list, "Default",
                                                                      factor_start_time_uint_to_duration)
        gradient_scope_name_time_list = self._get_scope_name_time_list(timeline_list, "Gradients",
                                                                       factor_start_time_uint_to_duration)
        recompute_scope_name_time_list = self._get_scope_name_time_list(timeline_list, "recompute_Default",
                                                                        factor_start_time_uint_to_duration)

        activity_timeline_list, cuda_compute_ops_timeline_list = self._load_activity_data( \
            activity_file_path, activity_args_file_path)

        # Add AllReduce info to timeline temp list and sort by start time.
        if communication_info:
            logger.debug('Allreduce info found, Start adding info to timeline...')
            cluster_related_timeline = self._get_cluster_timeline(
                timeline_list, cuda_compute_ops_timeline_list, communication_info, step_time_list)
            timeline_list.extend(cluster_related_timeline)
            timeline_list.extend(communication_info)
            timeline_list.sort(key=lambda x: float(x[self._start_time_idx]))

        timeline_list.extend(default_scope_name_time_list)
        timeline_list.extend(gradient_scope_name_time_list)
        timeline_list.extend(recompute_scope_name_time_list)
        timeline_list.extend(step_time_list)

        timeline_list.sort(key=lambda x: (float(x[self._start_time_idx])))

        # Add cuda activity timeline.
        timeline_list.extend(activity_timeline_list)
        timeline_list.sort(key=lambda x: float(x[2]))

        return timeline_list

    def _clock_synchronize_to_gpu(self, timeline_list):
        """Synchronize the timestamp from device to host."""
        start_time_file_path = os.path.join(self._profiling_dir, f"start_time_{self._device_id}.txt")

        try:
            with open(start_time_file_path) as f_obj:
                lines = f_obj.readlines()
                # lines[0] stores the host monotonic time of start training.
                host_monotonic_start_time = int(lines[0].strip().split(':')[-1])
                # lines[1] stores the gpu time of start training.
                gpu_start_time = int(lines[1].strip().split(':')[-1])
        except (IOError, OSError) as err:
            logger.critical(f'Error occurred when read {start_time_file_path}: {err}')
            raise ProfilerIOException() from err

        time_diff = gpu_start_time - host_monotonic_start_time
        for idx, time_item in enumerate(timeline_list):
            timeline_list[idx][self._start_time_idx] = int(time_item[self._start_time_idx]) + time_diff

    def _load_op_data(self, op_file_path, reduce_op_type):
        """Load operator data from file"""
        op_timeline_list = []
        communication_info = []
        try:
            with open(op_file_path, 'r') as f_obj:
                for line in f_obj:
                    self._timeline_summary['num_of_ops'] += 1
                    op_list = line.strip('\n').strip().split(';')
                    time_arr = op_list[-1]
                    time_arr = time_arr.split(" ")
                    for time in time_arr:
                        time = time.split(",")
                        line_list = op_list[:2] + time
                        communication_op_name = line_list[0].strip().split('/')[-1]
                        if communication_op_name not in reduce_op_type:
                            op_timeline_list.append(line_list)
                        else:
                            communication_info.append(line_list)
        except (IOError, OSError) as err:
            logger.critical('Error occurred when load operator timeline data intermediate file: %s', err)
            raise ProfilerIOException() from err

        return op_timeline_list, communication_info

    def _load_activity_data(self, activity_file_path, activity_args_file_path):
        """Load activity data from file"""
        activity_timeline_list = []
        cuda_compute_ops_timeline_list = []
        args_dict = {}
        try:
            with open(activity_args_file_path, 'r') as args_file:
                csv_reader = csv.reader(args_file)
                keys_list = next(csv_reader)
                # keys_list [name, type, op_full_name, stream_id, block_dim, grid_dim, ...]
                self._activity_keys_list = keys_list[1:3] + keys_list[4:6]
                for info in csv_reader:
                    args_dict[info[0]] = info[1:3] + info[4:6]
            with open(activity_file_path, 'r') as f_obj:
                for line in f_obj:
                    line_list = line.strip('\n').split(';')
                    # concat activity args info.
                    line_list += args_dict[line_list[0]]
                    if not line_list[0].startswith('nccl'):
                        cuda_compute_ops_timeline_list.append(line_list)
                    activity_timeline_list.append(line_list)
        except (IOError, OSError) as err:
            logger.critical('Error occurred when load activity timeline data intermediate file: %s', err)
            raise ProfilerIOException() from err

        return activity_timeline_list, cuda_compute_ops_timeline_list

    def _get_step_time_list_from_step_trace(self):
        """Produce the time of each step based on step_trace_profiling file."""
        # Record the time of each step.
        step_time_list = []
        step_start_op_name = []
        step_end_op_name = []
        step_num = 1
        tid = "Steps"
        step_trace_profiling_path = self._get_and_validate_path(
            self._step_trace_original_filename
        )

        try:
            with open(step_trace_profiling_path, 'r') as f_obj:
                for line in f_obj:
                    line = line.strip().split()
                    step_start_op_name.append(line[0].split(',')[0])
                    step_end_op_name.append(line[3].split(',')[0])
                    cur_step_start_time = float(line[0].split(',')[1])
                    cur_step_end_time = float(line[3].split(',')[1])
                    # convert duration time unit from ns to us.
                    cur_step_duration_time = (cur_step_end_time - cur_step_start_time) / 1e3
                    step_time_item = [str(step_num), tid, cur_step_start_time, cur_step_duration_time]
                    step_time_list.append(step_time_item)
                    step_num += 1
        except (IOError, OSError) as err:
            logger.critical(f'Error occurred when read {step_trace_profiling_path}: {err}')
            raise ProfilerIOException() from err

        return step_time_list

    def _get_cluster_timeline(self, timeline, activity_info, comm_info, step_info):
        """
        Analyse the cluster communication and computation data, and write result to file.

        To analyse the cluster performance bottleneck based on timeline, define the time of a training
        step as "t_total", propose five metrics as follows:
            1) The time that "receive" operators not overlapped by others(t1)
            2) The time that is consumed inside the stage(t_total - t1)
            3) The time that "communication" operators not overlapped by others(t2)
            4) The time that consumed by computation(t_total - t2)
            5) The time that "collective communication" operators not overlapped by others(t3)
        In pipeline parallel mode, we can locate slow stage based on t_total - t1. Inside each stage,
        we can locate slow card based on t_total - t2. The value of t1 indicates the degree that
        communication time between stages slow down the training. The value of t3 indicates the degree
        that communication inside each stage slow down the training.
        """
        step_num = len(step_info)
        is_pipeline_parallel = False
        comm_merged_timeline, _, comm_display_timeline = self._get_merged_time_list(
            comm_info,
            display_name="communication",
            factor=1e-3
        )
        compute_op_timeline = timeline + activity_info
        compute_op_timeline.sort(key=lambda x: float(x[self._start_time_idx]))
        compute_op_timeline_interval, _, compute_op_display_timeline = self._get_merged_time_list(
            compute_op_timeline,
            get_interval_time=True,
            factor=1e-3
        )
        # Consider if the overlap will be 0 or not.
        comm_not_overlapped_timeline = self._get_intersection_time(
            compute_op_timeline_interval,
            comm_merged_timeline
        )

        # Process receive part.
        all_timeline = timeline + comm_info
        all_timeline.sort(key=lambda x: float(x[self._start_time_idx]))
        receive_op_timeline = self._produce_two_separated_timeline(
            all_timeline,
            "Receive-op"
        )[0]
        if receive_op_timeline:
            is_pipeline_parallel = True
        receive_op_merged_timeline = self._get_merged_time_list(receive_op_timeline,
                                                                factor=1e-3)[0]

        receive_op_not_overlapped_timeline = self._get_intersection_time(
            compute_op_timeline_interval,
            receive_op_merged_timeline,
            display_name="receive_op_not_overlapped"
        )

        # Process collective communication part.
        collective_comm_timeline = self._produce_two_separated_timeline(
            comm_info,
            "Receive-op"
        )[-1]
        collective_comm_merged_timeline = self._get_merged_time_list(collective_comm_timeline,
                                                                     factor=1e-3)[0]
        collective_comm_not_overlapped_timeline = self._get_intersection_time(
            compute_op_timeline_interval,
            collective_comm_merged_timeline,
            display_name="exclude_receive_op"
        )

        # Generate free time that exclude computation and communication time.
        all_timeline = compute_op_timeline + comm_info
        all_timeline.sort(key=lambda x: float(x[self._start_time_idx]))
        free_timeline = self._get_merged_time_list(
            all_timeline,
            get_interval_time=True,
            display_name="free_time",
            factor=1e-3
        )[1]

        # Compute these five metrics mentioned above per step.
        recieve_alone_time = self._compute_time_inside_step(receive_op_not_overlapped_timeline, step_info)
        stage_time, computation_time = [], []
        comm_alone_time = self._compute_time_inside_step(comm_not_overlapped_timeline, step_info)
        collective_comm_alone_time = self._compute_time_inside_step(
            collective_comm_not_overlapped_timeline,
            step_info
        )
        for step in range(step_num):
            try:
                if is_pipeline_parallel:
                    stage_time.append(step_info[step][self._duration_idx] - recieve_alone_time[step])
                computation_time.append(step_info[step][self._duration_idx] - comm_alone_time[step])
            except IndexError as e:
                logger.error(e)

        metrices_per_step_list = [computation_time, comm_alone_time, stage_time,
                                  recieve_alone_time, collective_comm_alone_time]
        if step_num > 1:
            for metric in metrices_per_step_list:
                metric.append(sum(metric[1:]) / (step_num - 1))
        self._write_cluster_metrices(metrices_per_step_list, is_pipeline_parallel, "Gpu", self._device_id)

        res_timeline = []
        res_timeline.extend(comm_not_overlapped_timeline)
        res_timeline.extend(compute_op_display_timeline)
        res_timeline.extend(comm_display_timeline)
        res_timeline.extend(free_timeline)
        return res_timeline

    def _compute_time_inside_step(self, metric_timeline, step_time_list):
        """Compute per step time of metric_timeline."""
        per_step_time_list = []
        step = 0
        cur_step_metric_time = 0
        factor_us_to_ns = 1e3
        step_end_time = step_time_list[step][self._start_time_idx] + \
                        step_time_list[step][self._duration_idx] * factor_us_to_ns
        for time_item in metric_timeline:
            start_time = time_item[self._start_time_idx]
            if start_time > step_end_time:
                per_step_time_list.append(cur_step_metric_time)
                step += 1
                if step >= len(step_time_list):
                    logger.warning("Compute profiler compute_time_inside_step time, "
                                   "find the data length is more than step count, "
                                   "maybe current graph has multi sub graph, skip the last data.")
                    break
                step_end_time = step_time_list[step][self._start_time_idx] + \
                                step_time_list[step][self._duration_idx] * factor_us_to_ns
                cur_step_metric_time = 0
            cur_step_metric_time += time_item[self._duration_idx]
        per_step_time_list.append(cur_step_metric_time)

        return per_step_time_list

    def _get_intersection_time(self, first_time_list, second_time_list,
                               display_name="communication_not_overlapped"):
        """Get intersection time of two time list."""
        first_list_idx, second_list_idx = 0, 0
        first_list_len = len(first_time_list)
        second_list_len = len(second_time_list)
        intersection_segment_display_list = []
        factor_ns_to_us = 1e-3
        while first_list_idx < first_list_len and second_list_idx < second_list_len:
            intersection_start = max(
                first_time_list[first_list_idx][self._start_time_idx],
                second_time_list[second_list_idx][self._start_time_idx]
            )
            intersection_end = min(
                first_time_list[first_list_idx][self._duration_idx],
                second_time_list[second_list_idx][self._duration_idx]
            )
            if intersection_start < intersection_end:
                intersection_segment_display_list.append(
                    [display_name, self._tid_dict[display_name][0],
                     intersection_start, (intersection_end - intersection_start) * factor_ns_to_us,
                     self._tid_dict[display_name][1]]
                )
            if first_time_list[first_list_idx][self._duration_idx] >= \
                    second_time_list[second_list_idx][self._duration_idx]:
                second_list_idx += 1
            else:
                first_list_idx += 1

        return intersection_segment_display_list

    def _produce_two_separated_timeline(self, timeline, op_name):
        """Produce two separated timeline based on op_name."""
        timeline_include_op_name = []
        timeline_exclude_op_name = []
        for time_item in timeline:
            if op_name in time_item[self._op_name_idx]:
                timeline_include_op_name.append(time_item)
            else:
                timeline_exclude_op_name.append(time_item)
        return timeline_include_op_name, timeline_exclude_op_name


class AscendTimelineGenerator(BaseTimelineGenerator):
    """Generate ascend Timeline data from file."""
    _display_filename = 'ascend_timeline_display_{}.json'
    _timeline_summary_filename = 'ascend_timeline_summary_{}.json'
    _cluster_analyse_filename = 'ascend_cluster_analyse_{}_{}_{}_{}.csv'

    def __init__(self, profiling_dir, device_id, rank_id, rank_size, model):
        super().__init__(DeviceTarget.ASCEND.value, model)
        self._profiling_dir = profiling_dir
        self._device_id = device_id
        self._rank_id = rank_id
        self._rank_size = rank_size
        self._display_filename = self._display_filename.format(rank_id)
        self._timeline_summary_filename = self._timeline_summary_filename.format(rank_id)

    @staticmethod
    def _get_all_reduce_names(communication_info):
        names = []
        for info in communication_info:
            # all_reduce_name format: stream_stream_id_stream_op_index_opname
            all_reduce_name = info[0][info[0].rindex('_') + 1:]
            if all_reduce_name not in names:
                names.append(all_reduce_name)
        return names

    def init_timeline(self, communication_info, framework_info, aicpu_info, min_cycle_counter, source_path):
        """
        Init timeline metadata, adding all collected info.

        Args:
            communication_info (list[list]): The metadata of communication operator.
            framework_info (dict): The framework metadata.
            aicpu_info (dict): The metadata of AI CPU operator.
            min_cycle_counter (float): The minimum cycle counter of the timeline.
            source_path (str): The source of file.
        """
        if min_cycle_counter == float('inf'):
            min_cycle_counter = 0

        logger.info('Initiating timeline...')
        timeline_list = []
        op_timeline_list = self._get_op_timeline(communication_info, source_path)
        timeline_list.extend(op_timeline_list)

        # Generate step time.
        self._set_step_start_and_end_op_name(timeline_list)
        step_time_list = self._get_step_time_list(timeline_list)

        # Add Scope Name.
        default_scope_name_time_list = self._get_scope_name_time_list(timeline_list, "Default")
        gradient_scope_name_time_list = self._get_scope_name_time_list(timeline_list, "Gradients")
        recompute_scope_name_time_list = self._get_scope_name_time_list(timeline_list, "recompute_Default")

        # Add AI CPU data into timeline temp list and sort by start time.
        aicpu_data = aicpu_info.get('info')
        if aicpu_data:
            timeline_list.extend(aicpu_data)
            self._timeline_summary['op_exe_times'] += aicpu_info.get('op_exe_times', 0)
            self._timeline_summary['num_of_streams'] += aicpu_info.get('num_of_streams', 0)
            self._timeline_summary['num_of_ops'] += aicpu_info.get('num_of_ops', 0)
            self._timeline_summary['total_time'] += aicpu_info.get('total_time', 0)

        timeline_list.sort(key=lambda x: float(x[self._start_time_idx]))

        # Add AllReduce info to timeline temp list and sort by start time.
        if communication_info:
            logger.debug('AllReduce info found. Start adding info into timeline...')
            cluster_related_timeline = self._get_cluster_timeline(
                timeline_list, communication_info, step_time_list)
            timeline_list.extend(cluster_related_timeline)
            timeline_list.extend(communication_info)
            timeline_list.sort(key=lambda x: float(x[self._start_time_idx]))

        # Add step time and scope name info.
        timeline_list.extend(step_time_list)
        timeline_list.extend(default_scope_name_time_list)
        timeline_list.extend(recompute_scope_name_time_list)
        timeline_list.extend(gradient_scope_name_time_list)
        timeline_list.sort(key=lambda x: float(x[self._start_time_idx]))

        # Init a dict for counting the num of streams.
        stream_count_dict = {}
        for timeline in timeline_list:
            self._parse_timeline_data(timeline, min_cycle_counter)
            # Updating the collection of streams.
            if len(timeline) == 4:
                self._update_num_of_streams(timeline, stream_count_dict)

        # Add format thread meta data.
        self._format_meta_data_list.extend(self._timeline_meta)
        self._timeline_meta = self._format_meta_data_list
        # Get framework metadata.
        framework_obj_list = framework_info.get('object')
        # The length of list is the number of operators.
        self._timeline_summary['num_of_ops'] += len(framework_obj_list)
        self._add_framework_info(framework_obj_list)
        logger.info('Finished adding info into timeline...')

        # Update timeline summary info
        self._timeline_summary['num_of_streams'] += len(stream_count_dict.keys())

    def init_pynative_timeline(self):
        """Init timeline for pynative model."""
        timeline_list = OPIntermediateParser(self._profiling_dir, self._rank_id).get_timeline_data()
        cpu_timeline_generator = CpuTimelineGenerator(self._profiling_dir, self._model)
        cpu_timeline_list = cpu_timeline_generator.load_cpu_op_data()
        if cpu_timeline_list:
            self._pynative_clock_synchronize(cpu_timeline_list)
            timeline_list.extend(cpu_timeline_list)

        self._register_op_name(timeline_list)
        self._timeline_summary['op_exe_times'] = len(timeline_list)
        self._max_scope_name_num = self._get_max_scope_name_num(timeline_list)
        self._timeline_summary['max_scope_name_num'] = self._max_scope_name_num
        self._timeline_summary['num_of_ops'] = len(self._op_name_list)

        timeline_list.sort(key=lambda x: float(x[self._start_time_idx]))
        min_cycle_counter = float(timeline_list[0][self._start_time_idx])

        step_timeline = self._pynative_get_step_timeline_list(timeline_list)
        timeline_list.extend(step_timeline)

        stream_count_dict = {}
        max_scope_name_num = 0
        for timeline in timeline_list:
            self._parse_timeline_data(timeline, min_cycle_counter)
            self._update_num_of_streams(timeline, stream_count_dict)
            cur_scope_name_num = len(timeline[self._op_name_idx].split('/')) - 1
            max_scope_name_num = max(cur_scope_name_num, max_scope_name_num)

        self._timeline_summary['max_scope_name_num'] = max_scope_name_num
        self._timeline_summary['num_of_streams'] = len(stream_count_dict)

    def _parse_timeline_data(self, timeline, min_cycle_counter):
        """Parse timeline data."""
        # factor to convert the time unit from 1ms to 1us for timeline display
        factor = 1000
        op_meta = TimelineContainer(timeline)
        timeline_dict = {}
        timeline_dict['name'] = op_meta.op_name.split('/')[-1]
        timeline_dict['ph'] = 'X'
        timeline_dict['tid'] = op_meta.stream_id
        timeline_dict['ts'] = (op_meta.start_time - min_cycle_counter) * factor
        dur = op_meta.duration * factor
        timeline_dict['dur'] = dur
        if op_meta.pid is None:
            timeline_dict['pid'] = int(self._device_id)
            # Update total time of operator execution.
            if op_meta.stream_id not in ["Steps", "Scope Name"]:
                self._timeline_summary['total_time'] += op_meta.duration
        else:  # AllReduce and AI CPU pid
            timeline_dict['pid'] = op_meta.pid

        if op_meta.stream_id == "Scope Name":
            # remove the level of scope name which has a format like "0-conv2-Conv2d".
            timeline_dict['name'] = "-".join(op_meta.op_name.split('-')[1:])
            timeline_dict['scope_level'] = int(op_meta.op_name.split('-')[0])
        elif op_meta.stream_id[:len(self._host_cpu_op_label)] == self._host_cpu_op_label:
            timeline_dict['pid'] = self._HOST_CPU_PID

        self._update_format_meta_data(timeline_dict)
        self._timeline_meta.append(timeline_dict)

    def _get_op_timeline(self, communication_info, source_path):
        """get ai_core and cpu timeline."""
        all_reduce_names = AscendTimelineGenerator._get_all_reduce_names(communication_info)
        timeline_list = OPIntermediateParser(self._profiling_dir, self._rank_id).get_timeline_data(all_reduce_names)
        for timeline in timeline_list:
            timeline[self._tid_idx] = f"Stream #{timeline[self._tid_idx]}"

        cpu_timeline_generator = CpuTimelineGenerator(self._profiling_dir, self._model)
        cpu_timeline_list = cpu_timeline_generator.get_timeline_data()
        if cpu_timeline_list:
            self._clock_synchronize_to_device(cpu_timeline_list, source_path)
            timeline_list.extend(cpu_timeline_list)
        timeline_list.sort(key=lambda x: float(x[self._start_time_idx]))
        self._max_scope_name_num = self._get_max_scope_name_num(timeline_list)
        self._timeline_summary['op_exe_times'] = len(timeline_list)
        self._timeline_summary['max_scope_name_num'] = self._max_scope_name_num
        return timeline_list

    def _clock_synchronize_to_device(self, timeline_list, source_path):
        """Synchronize the timestamp from host to device."""
        host_start_file_path = os.path.join(source_path, f"host_start.log.{self._device_id}")
        dev_start_file_path = os.path.join(source_path, f"dev_start.log.{self._device_id}")

        try:
            with open(host_start_file_path) as f_obj:
                lines = f_obj.readlines()
                # lines[2] stores host monotonic_raw time of start training.
                host_monotonic = int(lines[2].strip().split(':')[1])
        except (IOError, OSError) as err:
            logger.critical('Error occurred when read host_start.log: %s', err)
            raise ProfilerIOException() from err
        try:
            with open(dev_start_file_path) as f_obj:
                lines = f_obj.readlines()
                # lines[2] stores device cycle counter of start training.
                dev_cntvct = int(lines[2].strip().split(':')[1])
        except (IOError, OSError) as err:
            logger.critical('Error occurred when read dev_start.log: %s', err)
            raise ProfilerIOException() from err

        factor_ns_to_ms = 1e-6
        factor_ten_ns_to_ns = 10
        factor_ms_to_ns = 1e6
        for idx, time_item in enumerate(timeline_list):
            host_time = int(float(time_item[self._start_time_idx]) * factor_ms_to_ns)
            device_time = dev_cntvct * factor_ten_ns_to_ns + (host_time - host_monotonic)
            timeline_list[idx][self._start_time_idx] = device_time * factor_ns_to_ms

    def _add_framework_info(self, framework_obj_list):
        """
        Add framework info into timeline metadata.

        Args:
            framework_obj_list (list): The framework metadata.
        """
        logger.debug('Start adding framework info into timeline...')
        # Get the framework info that will be written into timeline.
        framework_info_dict = {}
        for framework_obj in framework_obj_list:
            op_name = framework_obj[0]
            op_type = framework_obj[1]
            op_full_name = framework_obj[4]
            op_info = framework_obj[5]
            framework_info = {
                'name': op_name,
                'args': {
                    'type': op_type,
                    'fullname': op_full_name
                }
            }
            framework_info.get('args').update(op_info)
            framework_info_dict[op_full_name] = framework_info

        # Insert framework info into timeline.
        for timeline_item in self._timeline_meta:
            op_full_name = timeline_item.get('name')
            framework_item = framework_info_dict.get(op_full_name)
            if framework_item:
                timeline_item['name'] = framework_item.get('name')
                timeline_item['args'] = framework_item.get('args')
        logger.debug('Finished adding framework info into timeline...')

    def _produce_two_separated_timeline(self, timeline, op_name):
        """Produce two separated timeline based on op_name."""
        timeline_include_op_name = []
        timeline_exclude_op_name = []
        for time_item in timeline:
            if op_name in time_item[self._op_name_idx]:
                timeline_include_op_name.append(time_item)
            else:
                timeline_exclude_op_name.append(time_item)
        return timeline_include_op_name, timeline_exclude_op_name

    def _get_cluster_timeline(self, aicore_info, comm_info, step_info):
        """
        Analyse the cluster communication and computation data, and write result to file.

        To analyse the cluster performance bottleneck based on timeline, define the time of a training
        step as "t_total", propose five metrics as follows:
            1) The time that "receive" operators not overlapped by others(t1)
            2) The time that is consumed inside the stage(t_total - t1)
            3) The time that "communication" operators not overlapped by others(t2)
            4) The time that consumed by computation(t_total - t2)
            5) The time that "collective communication" operators not overlapped by others(t3)
        In pipeline parallel mode, we can locate slow stage based on t_total - t1. Inside each stage,
        we can locate slow card based on t_total - t2. The value of t1 indicates the degree that
        communication time between stages slow down the training. The value of t3 indicates the degree
        that communication inside each stage slow down the training.
        """
        is_pipeline_parallel = False
        comm_merged_timeline, _, comm_display_timeline = self._get_merged_time_list(
            comm_info, display_name="communication"
        )
        aicore_timeline_interval, _, aicore_display_timeline = self._get_merged_time_list(
            aicore_info, get_interval_time=True
        )
        # Consider if the overlap will be 0 or not.
        comm_not_overlapped_timeline = self._get_intersection_time(
            aicore_timeline_interval, comm_merged_timeline
        )

        # Process receive part.
        all_timeline = aicore_info + comm_info
        all_timeline.sort(key=lambda x: float(x[self._start_time_idx]))
        receive_op_timeline, timeline_exclude_receive_op = self._produce_two_separated_timeline(
            all_timeline, "Receive-op"
        )
        if receive_op_timeline:
            is_pipeline_parallel = True
        receive_op_merged_timeline = self._get_merged_time_list(receive_op_timeline)[0]
        timeline_exclude_receive_op_interval = self._get_merged_time_list(
            timeline_exclude_receive_op, get_interval_time=True
        )[0]
        receive_op_not_overlapped_timeline = self._get_intersection_time(
            timeline_exclude_receive_op_interval, receive_op_merged_timeline
        )

        # Process collective communication part.
        collective_comm_timeline = self._produce_two_separated_timeline(
            comm_info, "Receive-op"
        )[-1]
        collective_comm_merged_timeline = self._get_merged_time_list(collective_comm_timeline)[0]
        collective_comm_not_overlapped_timeline = self._get_intersection_time(
            aicore_timeline_interval, collective_comm_merged_timeline
        )

        # Generate free time that exclude computation and communication time.
        free_timeline = self._get_merged_time_list(
            all_timeline, get_interval_time=True, display_name="free_time"
        )[1]

        self._parse_cluster_metrices(step_info, receive_op_not_overlapped_timeline, comm_not_overlapped_timeline
                                     , collective_comm_not_overlapped_timeline, is_pipeline_parallel)

        res_timeline = []
        res_timeline.extend(comm_not_overlapped_timeline)
        res_timeline.extend(aicore_display_timeline)
        res_timeline.extend(comm_display_timeline)
        res_timeline.extend(free_timeline)

        return res_timeline

    def _parse_cluster_metrices(self, step_info, receive_op_not_overlapped_timeline, comm_not_overlapped_timeline
                                , collective_comm_not_overlapped_timeline, is_pipeline_parallel):
        """Write the cluster metrices"""
        step_num = len(step_info)
        # Compute these five metrics mentioned above per step.
        recieve_alone_time = self._compute_time_inside_step(receive_op_not_overlapped_timeline, step_info)
        stage_time, computation_time = [], []
        comm_alone_time = self._compute_time_inside_step(comm_not_overlapped_timeline, step_info)
        collective_comm_alone_time = self._compute_time_inside_step(
            collective_comm_not_overlapped_timeline, step_info
        )
        for step in range(step_num):
            try:
                if is_pipeline_parallel:
                    stage_time.append(step_info[step][self._duration_idx] - recieve_alone_time[step])
                computation_time.append(step_info[step][self._duration_idx] - comm_alone_time[step])
            except IndexError as err:
                logger.error(err)
        metrices_per_step_list = [computation_time, comm_alone_time, stage_time,
                                  recieve_alone_time, collective_comm_alone_time]
        if step_num > 1:
            for metric in metrices_per_step_list:
                metric.append(sum(metric[1:]) / (step_num - 1))
        self._write_cluster_metrices(metrices_per_step_list, is_pipeline_parallel, "Ascend", self._rank_id)

    def _compute_time_inside_step(self, metric_timeline, step_time_list):
        """Compute per step time of metric_timeline."""
        per_step_time_list = [0 for i in range(len(step_time_list))]
        step = 0
        step_end_time = step_time_list[step][self._start_time_idx] + \
                        step_time_list[step][self._duration_idx]
        for time_item in metric_timeline:
            start_time = time_item[self._start_time_idx]
            if start_time > step_end_time:
                step += 1
                if step >= len(step_time_list):
                    logger.warning("Compute profiler compute_time_inside_step time, "
                                   "find the data length is more than step count, "
                                   "maybe current graph has multi sub graph, skip the last data.")
                    break
                step_end_time = step_time_list[step][self._start_time_idx] + \
                                step_time_list[step][self._duration_idx]
            per_step_time_list[step] += time_item[self._duration_idx]

        return per_step_time_list

    def _get_intersection_time(self, first_time_list, second_time_list,
                               display_name="communication_not_overlapped"):
        """Get intersection time of two time list."""
        first_list_idx, second_list_idx = 0, 0
        first_list_len = len(first_time_list)
        second_list_len = len(second_time_list)
        intersection_segment_display_list = []

        while first_list_idx < first_list_len and \
                second_list_idx < second_list_len:
            intersection_start = max(
                first_time_list[first_list_idx][self._start_time_idx],
                second_time_list[second_list_idx][self._start_time_idx]
            )
            intersection_end = min(
                first_time_list[first_list_idx][self._duration_idx],
                second_time_list[second_list_idx][self._duration_idx]
            )
            if intersection_start < intersection_end:
                tid = self._tid_dict.get(display_name, [0, 0])
                intersection_segment_display_list.append(
                    [display_name, tid[0],
                     intersection_start, intersection_end - intersection_start, tid[1]]
                )
            if first_time_list[first_list_idx][self._duration_idx] >= \
                    second_time_list[second_list_idx][self._duration_idx]:
                second_list_idx += 1
            else:
                first_list_idx += 1

        return intersection_segment_display_list

    def _pynative_get_step_timeline_list(self, timeline_list):
        """Get step timeline list for pynative model."""
        step_list = []
        # The timeline starts with the GetNext op
        if len(timeline_list) < 2 or 'GetNext' not in timeline_list[0][self._op_name_idx] and \
                'GetNext' not in timeline_list[1][self._op_name_idx]:
            return step_list
        step = [-1, -1]
        step_num = 0
        tid = "Steps"
        for timeline in timeline_list:
            if 'GetNext' not in timeline[self._op_name_idx]:
                continue
            start_time = float(timeline[self._start_time_idx])
            if step[0] == -1:
                step[0] = start_time
            else:
                step[1] = start_time - step[0]
                step_num = step_num + 1
                step_list.append([str(step_num), tid, float(step[0]), step[1]])
                step = [start_time, -1]
        if step[0] != -1 and step[1] == -1:
            step_num = step_num + 1
            step_list.append([str(step_num), tid, float(step[0]),
                              float(timeline_list[-1][self._start_time_idx]) - step[0]])
        return step_list

    def _pynative_clock_synchronize(self, timeline_list):
        """Synchronize the timestamp from device to host."""
        start_time_file_path = os.path.join(self._profiling_dir, f"start_time_{self._rank_id}.txt")
        try:
            with open(start_time_file_path) as f_obj:
                lines = f_obj.readlines()
                # lines[0] stores the host monotonic time of start training.
                host_monotonic_start_time = int(lines[0].strip().split(':')[-1])
                # lines[1] stores the gpu time of start training.
                gpu_start_time = int(lines[1].strip().split(':')[-1])
        except (IOError, OSError) as err:
            logger.critical(f'Error occurred when read {start_time_file_path}: {err}')
            raise ProfilerIOException() from err
        time_diff = gpu_start_time * 1000 - host_monotonic_start_time
        for idx, time_item in enumerate(timeline_list):
            timeline_list[idx][self._start_time_idx] = int(time_item[self._start_time_idx]) + time_diff
            timeline_list[idx][self._start_time_idx] = timeline_list[idx][self._start_time_idx] / 1000000
            timeline_list[idx][self._duration_idx] = timeline_list[idx][self._duration_idx] / 1000

    def _set_step_start_and_end_op_name(self, timeline_list):
        """Set the start and end operator full name of each step."""
        if not timeline_list or len(timeline_list) < 2:
            return

        start_op_idx = 0
        self._step_end_op_name = timeline_list[-1][self._op_name_idx]
        for i, timeline in enumerate(timeline_list):
            if timeline[self._op_name_idx] == self._step_end_op_name:
                start_op_idx = i + 1
                break

        if start_op_idx >= len(timeline_list):
            start_op_idx = 0
        self._step_start_op_name = timeline_list[start_op_idx][self._op_name_idx]


class CpuTimelineGenerator(GpuTimelineGenerator):
    """Generate cpu Timeline data from file."""
    _output_op_execute_time_file_path = "cpu_op_execute_timestamp_{}.txt"
    _display_filename = 'cpu_timeline_display_{}.json'
    _timeline_summary_filename = 'cpu_timeline_summary_{}.json'

    def __init__(self, profiling_dir, model):
        super().__init__(profiling_dir, 0, 0, model)
        self._device_target = DeviceTarget.CPU.value

    def get_timeline_data(self):
        """Get timeline data from file."""
        timeline_list = self.load_cpu_op_data()
        factor_ns_to_ms = 1e6
        factor_us_to_ms = 1e3
        for time_item in timeline_list:
            time_item[self._start_time_idx] = float(time_item[self._start_time_idx]) / factor_ns_to_ms
            time_item[self._duration_idx] = float(time_item[self._duration_idx]) / factor_us_to_ms

        return timeline_list

    def init_timeline(self):
        """Init timeline metadata, adding all collected info."""
        timeline_list = self._load_timeline_data()

        # Init a dict for counting the num of streams.
        stream_count_dict = {}
        for timeline in timeline_list:
            self._parse_timeline_data(timeline, 0)
            # Updating the collection of streams.
            if len(timeline) == 4:
                self._update_num_of_streams(timeline, stream_count_dict)

        # Add format thread meta data.
        self._format_meta_data_list.extend(self._timeline_meta)
        self._timeline_meta = self._format_meta_data_list

        # Update timeline summary info
        self._timeline_summary['num_of_streams'] += len(stream_count_dict.keys())

    def load_cpu_op_data(self):
        """Load cpu operator data from file"""
        op_file_path = self._get_and_validate_path(
            self._output_op_execute_time_file_path)
        timeline_list = []
        if not os.path.exists(op_file_path):
            logger.info("No cpu operator info.")
            return timeline_list
        timeline_list = self._load_op_data(op_file_path)
        factor_ms_to_us = 1e-3
        for time_item in timeline_list:
            time_item[self._duration_idx] = float(time_item[self._duration_idx]) / factor_ms_to_us

        return timeline_list

    def _get_and_validate_path(self, file_name):
        """Generate op or activity file path from file name, and validate this path."""
        file_path = os.path.join(
            self._profiling_dir,
            file_name.format(self._device_id)
        )
        file_path = validate_and_normalize_path(file_path)

        return file_path

    def _load_op_data(self, op_file_path):
        """Load operator data from file"""
        op_timeline_list = []
        try:
            with open(op_file_path, 'r') as f_obj:
                for line in f_obj:
                    self._timeline_summary['num_of_ops'] += 1
                    op_list = line.strip('\n').strip().split(';')
                    time_arr = op_list[-1]
                    time_arr = time_arr.split(" ")
                    for time in time_arr:
                        time = time.split(",")
                        if len(time) == 3:
                            # for time value is [start_timestamp, duration, tid]
                            # line_list[1] would be like "HostCpuOps" + str(tid)
                            line_list = op_list[:1] + [op_list[1] + str(time[-1])] + time[:-1]
                        else:
                            # for time value is [start_timestamp, duration]
                            line_list = op_list[:2] + time
                        op_timeline_list.append(line_list)
        except (IOError, OSError) as err:
            logger.critical('Error occurred when load operator timeline data intermediate file: %s', err)
            raise ProfilerIOException() from err

        return op_timeline_list

    def _load_timeline_data(self):
        """Load timeline data from file."""
        timeline_list = self.load_cpu_op_data()

        timeline_list.sort(key=lambda x: float(x[self._start_time_idx]))
        self._max_scope_name_num = self._get_max_scope_name_num(timeline_list)
        self._timeline_summary['max_scope_name_num'] = self._max_scope_name_num

        # Generate step time.
        factor_start_time_uint_to_duration = 1e-3
        self._set_step_start_and_end_op_name(timeline_list)

        step_time_list = self._get_step_time_list(timeline_list, factor_start_time_uint_to_duration)

        # Add merge compute time and free time
        merge_compute_timeline = self._get_merged_time_list(
            timeline_list, False, "computation_op", factor_start_time_uint_to_duration)[2]
        free_time_timeline = self._get_merged_time_list(
            timeline_list, True, "free_time", factor_start_time_uint_to_duration)[1]

        # Add Scope Name.
        default_scope_name_time_list = self._get_scope_name_time_list(timeline_list, "Default",
                                                                      factor_start_time_uint_to_duration)
        gradient_scope_name_time_list = self._get_scope_name_time_list(timeline_list, "Gradients",
                                                                       factor_start_time_uint_to_duration)
        recompute_scope_name_time_list = self._get_scope_name_time_list(timeline_list, "recompute_Default",
                                                                        factor_start_time_uint_to_duration)
        timeline_list.extend(default_scope_name_time_list)
        timeline_list.extend(gradient_scope_name_time_list)
        timeline_list.extend(recompute_scope_name_time_list)
        timeline_list.extend(step_time_list)

        timeline_list.sort(key=lambda x: (float(x[self._start_time_idx]), x[self._tid_idx]))
        timeline_list.sort(key=lambda x: float(x[2]))
        timeline_list.extend(merge_compute_timeline)
        timeline_list.extend(free_time_timeline)

        return timeline_list

    def _parse_timeline_data(self, timeline, min_cycle_counter):
        """Parse timeline data."""
        # factor to convert the time unit of start_time(ts) from 1ns to 1us for timeline display
        factor = 1000
        op_meta = TimelineContainer(timeline)
        timeline_dict = {}
        timeline_dict['name'] = op_meta.op_name.split('/')[-1]
        timeline_dict['ph'] = 'X'
        timeline_dict['tid'] = op_meta.stream_id
        timeline_dict['ts'] = (op_meta.start_time - min_cycle_counter) / factor
        dur = op_meta.duration
        timeline_dict['dur'] = dur
        timeline_dict['pid'] = int(self._device_id)
        if op_meta.stream_id == "Scope Name":
            # remove the level of scope name which has a format like "0-conv2-Conv2d".
            timeline_dict['name'] = "-".join(op_meta.op_name.split('-')[1:])
            timeline_dict['scope_level'] = int(op_meta.op_name.split('-')[0])
        elif self._host_cpu_op_label == op_meta.stream_id[:len(self._host_cpu_op_label)]:
            timeline_dict['pid'] = self._HOST_CPU_PID

        if len(timeline) == 5:
            # len(timeline) == 5 refers to analyse data.
            timeline_dict["pid"] = op_meta.pid
        elif op_meta.stream_id not in ["Scope Name", "Steps"]:
            # Update total time of operator execution.
            self._timeline_summary['total_time'] += dur / factor
            self._timeline_summary['op_exe_times'] += 1

        self._update_format_meta_data(timeline_dict)
        self._timeline_meta.append(timeline_dict)
